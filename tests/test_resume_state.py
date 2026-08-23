"""Round-trip check: does a saved MDPO checkpoint fully restore training state?

MDPO trains two independent actor-critics that regularize each other through a
mutual-distillation KL. Persisting only the first silently restarts half the agent
on every resume, which is invisible in the loss curves -- hence this test.

    python sru-navigation-learning/tests/test_resume_state.py
"""

import copy
import os
import sys
import tempfile
import types

import torch

# The runner imports SummaryWriter at module scope. Stub it only where tensorboard
# is absent (login nodes) so the test runs outside the training container too.
try:
    import torch.utils.tensorboard  # noqa: F401
except ImportError:
    _tb = types.ModuleType("torch.utils.tensorboard")
    _tb.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = _tb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402


class DummyEnv:
    num_envs = 4
    num_actions = 3
    max_episode_length = 100

    def __init__(self):
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.cfg = {}
        self.unwrapped = self

    def get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 8), "critic": torch.zeros(self.num_envs, 8)}


CFG = {
    "num_steps_per_env": 4,
    "save_interval": 50,
    "empirical_normalization": False,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "MDPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "learning_rate": 1e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}


def build():
    # deepcopy: the runner pops "class_name" out of the nested cfg dicts.
    return OnPolicyRunner(DummyEnv(), copy.deepcopy(CFG), log_dir=None, device="cpu")


def perturb(runner):
    """Make both networks and both optimizers differ from their init state."""
    for net, opt in (
        (runner.alg.actor_critic_1, runner.alg.optimizer_1),
        (runner.alg.actor_critic_2, runner.alg.optimizer_2),
    ):
        for _ in range(3):
            opt.zero_grad()
            loss = sum(p.square().sum() for p in net.parameters())
            loss.backward()
            opt.step()


ckpt = os.path.join(tempfile.mkdtemp(), "model_1234.pt")

fails = []

torch.manual_seed(0)
src = build()
if src.current_learning_iteration != 0:
    fails.append(f"fresh completed iterations = {src.current_learning_iteration}, want 0")
if src.remaining_iterations(1500) != 1500:
    fails.append(f"fresh remaining iterations = {src.remaining_iterations(1500)}, want 1500")
perturb(src)
# Diverge the two halves so a clone-net-1 bug is detectable.
with torch.no_grad():
    for p in src.alg.actor_critic_2.parameters():
        p.add_(0.5)
# Exactly 1234 updates have completed; zero-based update 1234 executes next.
src.current_learning_iteration = 1234
src.env.common_step_counter = 98765
torch.manual_seed(9)
expected = float(torch.rand(1))
torch.manual_seed(9)
src.save(ckpt)

torch.manual_seed(999)
dst = build()
dst.load(ckpt)

for name, a, b in (
    ("actor_critic_1", src.alg.actor_critic_1, dst.alg.actor_critic_1),
    ("actor_critic_2", src.alg.actor_critic_2, dst.alg.actor_critic_2),
):
    for (k, pa), pb in zip(a.state_dict().items(), b.state_dict().values()):
        if not torch.equal(pa, pb):
            fails.append(f"{name}.{k} mismatch")

if src.alg.actor_critic_1.state_dict()["actor.0.weight"].equal(
    dst.alg.actor_critic_2.state_dict()["actor.0.weight"]
):
    fails.append("net 2 was cloned from net 1 (the original bug)")

for name, a, b in (
    ("optimizer_1", src.alg.optimizer_1, dst.alg.optimizer_1),
    ("optimizer_2", src.alg.optimizer_2, dst.alg.optimizer_2),
):
    sa, sb = a.state_dict()["state"], b.state_dict()["state"]
    if set(sa) != set(sb):
        fails.append(f"{name} state keys differ")
    if not sa:
        fails.append(f"{name} has no optimizer state to compare")
    compared = 0
    # Muon and AdamW use different field names, so compare whatever is present.
    for k in sa:
        for field, va in sa[k].items():
            vb = sb[k][field]
            same = torch.equal(va, vb) if torch.is_tensor(va) else va == vb
            if not same:
                fails.append(f"{name}[{k}].{field} mismatch")
            compared += 1
    print(f"[info] {name}: compared {compared} state fields across {len(sa)} params")

if dst.current_learning_iteration != 1234:
    fails.append(f"completed iterations = {dst.current_learning_iteration}, want 1234")
if dst.remaining_iterations(1500) != 266:
    fails.append(f"remaining iterations = {dst.remaining_iterations(1500)}, want 266")
if dst.env.common_step_counter != 98765:
    fails.append(f"common_step_counter = {dst.env.common_step_counter}, want 98765")

got = float(torch.rand(1))
if got != expected:
    fails.append(f"torch RNG stream diverged: got {got}, want {expected}")

# numpy/python state is deliberately not persisted: rsl_rl never draws from them.
if any(k.startswith(("numpy", "python")) for k in torch.load(ckpt, weights_only=True)["rng_state"]):
    fails.append("checkpoint still carries numpy/python RNG state")

# Weights-only load must work: it is what runner.load uses.
try:
    checkpoint_dict = torch.load(ckpt, weights_only=True)
except Exception as exc:  # noqa: BLE001
    fails.append(f"weights_only=True load failed: {exc}")
else:
    if checkpoint_dict["iter"] != 1234:
        fails.append(f"checkpoint iter = {checkpoint_dict['iter']}, want 1234")
    if "next_iter" in checkpoint_dict:
        fails.append("checkpoint redundantly stores next_iter")

# The first optimizer step after restoring must match a continuous step exactly,
# including both networks and both sets of optimizer moments.
perturb(src)
perturb(dst)
for name, a, b in (
    ("actor_critic_1 after next update", src.alg.actor_critic_1, dst.alg.actor_critic_1),
    ("actor_critic_2 after next update", src.alg.actor_critic_2, dst.alg.actor_critic_2),
):
    for (key, pa), pb in zip(a.state_dict().items(), b.state_dict().values()):
        if not torch.equal(pa, pb):
            fails.append(f"{name}.{key} mismatch")
for name, a, b in (
    ("optimizer_1 after next update", src.alg.optimizer_1, dst.alg.optimizer_1),
    ("optimizer_2 after next update", src.alg.optimizer_2, dst.alg.optimizer_2),
):
    state_a = a.state_dict()
    state_b = b.state_dict()
    if state_a["param_groups"] != state_b["param_groups"]:
        fails.append(f"{name} param_groups mismatch")
    if set(state_a["state"]) != set(state_b["state"]):
        fails.append(f"{name} state keys differ")
        continue
    for param_id in state_a["state"]:
        for field, value_a in state_a["state"][param_id].items():
            value_b = state_b["state"][param_id][field]
            same = (
                torch.equal(value_a, value_b)
                if torch.is_tensor(value_a)
                else value_a == value_b
            )
            if not same:
                fails.append(f"{name}[{param_id}].{field} mismatch")

# Eval path must not touch optimizers or RNG. Build first: network init draws from
# the torch stream, so constructing inside the measured window would mask the result.
eval_runner = build()
torch.manual_seed(4242)
before = float(torch.rand(1))
torch.manual_seed(4242)
eval_runner.load(ckpt, load_optimizer=False)
if float(torch.rand(1)) != before:
    fails.append("load_optimizer=False perturbed the RNG stream")

print("\n".join(f"FAIL: {f}" for f in fails) if fails else "PASS: all resume state round-trips")
sys.exit(1 if fails else 0)
