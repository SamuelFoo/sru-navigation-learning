"""Unit tests for the depth+LiDAR cross-modal (learned-PE) fusion module.

    python sru-navigation-learning/tests/test_cross_modal_fusion.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rsl_rl.networks.sru_memory.cross_modal_fusion import CrossModalFuseModule  # noqa: E402

# Depth latent (64, 5, 8), LiDAR latent (64, 4, 12): the real policy contracts.
DEPTH_C, DEPTH_H, DEPTH_W = 64, 5, 8
LIDAR_C, LIDAR_H, LIDAR_W = 64, 4, 12
N_DEPTH = DEPTH_H * DEPTH_W
N_LIDAR = LIDAR_H * LIDAR_W
INFO_DIM = 48
NUM_HEADS = 4


def _build_module():
    return CrossModalFuseModule(
        image_dim=DEPTH_C,
        info_dim=INFO_DIM,
        num_heads=NUM_HEADS,
        view_sizes=[N_DEPTH, N_LIDAR],
    )


def _inputs(batch=2, requires_grad=False):
    depth = torch.randn(batch, DEPTH_C, DEPTH_H, DEPTH_W, requires_grad=requires_grad)
    lidar = torch.randn(batch, LIDAR_C, LIDAR_H, LIDAR_W, requires_grad=requires_grad)
    info = torch.randn(batch, INFO_DIM)
    return depth, lidar, info


def test_forward_shape_and_finiteness():
    module = _build_module()
    module.eval()
    depth, lidar, info = _inputs(batch=3)
    out = module([depth, lidar], info)
    assert out.shape == (3, DEPTH_C)
    assert torch.isfinite(out).all()


def test_learned_pos_embed():
    module = _build_module()
    assert tuple(module.pos_embed.shape) == (1, N_DEPTH + N_LIDAR, DEPTH_C)
    assert module.view_sizes == [N_DEPTH, N_LIDAR]


def test_return_attn_weights():
    module = _build_module()
    module.eval()
    depth, lidar, info = _inputs(batch=2)
    out, attn = module([depth, lidar], info, return_attn=True)
    n_total = N_DEPTH + N_LIDAR
    assert out.shape == (2, DEPTH_C)
    assert attn["view_sizes"] == [N_DEPTH, N_LIDAR]
    # Self-attention rows are probability distributions over all tokens.
    sa = attn["self_attn"]
    assert sa.shape == (2, n_total, n_total)
    assert torch.allclose(sa.sum(dim=-1), torch.ones(2, n_total), atol=1e-4)
    # Cross-attention (proprio query) is a distribution over all tokens.
    ca = attn["cross_attn"]
    assert ca.shape == (2, n_total)
    assert torch.allclose(ca.sum(dim=-1), torch.ones(2), atol=1e-4)
    # The depth-vs-lidar mass split the viz relies on.
    depth_mass = ca[:, :N_DEPTH].sum(dim=-1)
    lidar_mass = ca[:, N_DEPTH:].sum(dim=-1)
    assert torch.allclose(depth_mass + lidar_mass, torch.ones(2), atol=1e-4)


def test_backward_grad():
    module = _build_module()
    module.train()
    depth, lidar, info = _inputs(batch=2, requires_grad=True)
    module([depth, lidar], info).sum().backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert lidar.grad is not None and torch.isfinite(lidar.grad).all()
    assert module.pos_embed.grad is not None and torch.isfinite(module.pos_embed.grad).all()


if __name__ == "__main__":
    test_forward_shape_and_finiteness()
    test_learned_pos_embed()
    test_return_attn_weights()
    test_backward_grad()
    print("All cross-modal fusion tests passed.")
