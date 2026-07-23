"""Integration test: ActorCriticSRU in geometric depth+LiDAR fusion mode (CPU-only).

Validates observation slicing and a full forward through the actor and critic with the
two heterogeneous latents (depth (64,5,8) + lidar (64,4,12)).

    python sru-navigation-learning/tests/test_actor_critic_fusion.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rsl_rl.modules.actor_critic_sru import ActorCriticSRU  # noqa: E402

PROPRIO = 32
DEPTH = (64, 5, 8)
LIDAR = (64, 4, 12)
HEIGHT = (64, 7, 7)
NUM_ACTIONS = 3


def _dims():
    n_depth = DEPTH[0] * DEPTH[1] * DEPTH[2]
    n_lidar = LIDAR[0] * LIDAR[1] * LIDAR[2]
    n_img = n_depth + n_lidar
    n_height = HEIGHT[0] * HEIGHT[1] * HEIGHT[2]
    num_actor_obs = PROPRIO + n_img
    num_critic_obs = PROPRIO + 1 + n_height + n_img  # [proprio | time | height | image]
    return num_actor_obs, num_critic_obs


def test_build_and_forward():
    num_actor_obs, num_critic_obs = _dims()
    model = ActorCriticSRU(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=NUM_ACTIONS,
        image_input_dims=DEPTH,          # shared channel dim comes from here (64)
        height_input_dims=HEIGHT,
        fusion_view_dims=[DEPTH, LIDAR],  # view 0 = depth, view 1 = lidar
        rnn_hidden_size=64,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
    )
    model.eval()

    assert model.fusion is True
    assert model.total_image_features == num_actor_obs - PROPRIO
    assert model.actor_proprioceptive_input_dim == PROPRIO

    B = 4
    actor_obs = torch.randn(B, num_actor_obs)
    critic_obs = torch.randn(B, num_critic_obs)

    with torch.no_grad():
        actions = model.act_inference(actor_obs)
        assert actions.shape == (B, NUM_ACTIONS)
        assert torch.isfinite(actions).all()

        value = model.evaluate(critic_obs)
        assert value.shape == (B, 1)
        assert torch.isfinite(value).all()

    # Attention weights are available for visualization straight off the image net.
    other = actor_obs[..., :PROPRIO]
    views = model._extract_image_observations(actor_obs)
    _, attn = model.attn_image_net(views, other, return_attn=True)
    n_depth = DEPTH[0] * DEPTH[1] * DEPTH[2] // DEPTH[0]  # H*W tokens
    n_lidar = LIDAR[1] * LIDAR[2]
    assert attn["view_sizes"] == [n_depth, n_lidar]
    assert attn["cross_attn"].shape == (B, n_depth + n_lidar)


if __name__ == "__main__":
    test_build_and_forward()
    print("ActorCriticSRU fusion integration test passed.")
