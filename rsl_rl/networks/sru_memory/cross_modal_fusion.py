#  Copyright 2025 ETH Zurich
#  SPDX-License-Identifier: BSD-3-Clause

"""Cross-modal attention fusion of heterogeneous perception latents.

Fuses pre-encoded depth-camera and LiDAR latents into one proprio-conditioned summary:
self-attention over the union of tokens (cross-modal fusion) -> FFN -> a proprio-query
cross-attention pooling to (B, C). A learned per-token positional embedding tags each token.
(An earlier geometric RoPE-on-ray-directions variant underperformed this and was dropped.)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CrossModalFuseModule(nn.Module):
    """Cross-attention fusion of heterogeneous image latents + proprio, with a learned PE.

    ``view_sizes``: per-view token counts (e.g. ``[depth_tokens, lidar_tokens]``); their sum
    is the sequence length and the split used to report per-modality attention. Forward
    returns ``(B, image_dim)``; ``return_attn=True`` also returns attention weights for the viz.
    """

    def __init__(
        self,
        image_dim: int,
        info_dim: int,
        num_heads: int,
        view_sizes: List[int],
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"
        self.image_dim = image_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.view_sizes: list[int] = [int(n) for n in view_sizes]
        self.num_tokens = sum(self.view_sizes)

        # Proprio (info) projection: 2-layer MLP with ELU (matches CrossAttentionFuseModule).
        expand_dim = image_dim * 2
        self.info_proj = nn.Sequential(
            nn.Linear(info_dim, expand_dim),
            nn.ELU(inplace=True),
            nn.Linear(expand_dim, image_dim),
            nn.ELU(inplace=True),
        )

        # Learned per-token positional embedding.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, image_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Self-attention sub-layer.
        self.norm1 = nn.LayerNorm(image_dim)
        self.qkv = nn.Linear(image_dim, image_dim * 3, bias=False)
        self.self_proj = nn.Linear(image_dim, image_dim)

        # Feed-forward sub-layer.
        self.norm2 = nn.LayerNorm(image_dim)
        hidden = int(image_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, image_dim),
            nn.ELU(inplace=True),
        )

        # Cross-attention sub-layer (proprio query over the fused tokens).
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim, num_heads=num_heads, batch_first=True
        )

    def _stack_views(self, img) -> torch.Tensor:
        """Flatten a list of ``[B, C, H_i, W_i]`` views into a ``[B, N, C]`` sequence."""
        if not isinstance(img, (list, tuple)):
            img = [img]
        tokens = [v.reshape(v.shape[0], v.shape[1], -1).permute(0, 2, 1) for v in img]
        return torch.cat(tokens, dim=1)  # [B, N, C]

    def forward(self, img, info: torch.Tensor, return_attn: bool = False):
        x = self._stack_views(img)                           # [B, N, C]
        x = x + self.pos_embed.to(x.dtype)
        B, N, C = x.shape

        # --- Self-attention (cross-modal fusion) ---
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale        # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        sa = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.self_proj(sa)

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        # --- Cross-attention with proprio query ---
        query = self.info_proj(info).unsqueeze(1)            # [B, 1, C]
        ca, cross_w = self.cross_attn(query, x, x, need_weights=return_attn)
        out = ca.squeeze(1)                                  # [B, C]

        if not return_attn:
            return out
        attn_info = {
            "self_attn": attn.mean(dim=1).detach(),          # [B, N, N]
            "cross_attn": cross_w.squeeze(1).detach() if cross_w is not None else None,
            "view_sizes": list(self.view_sizes),
        }
        return out, attn_info
