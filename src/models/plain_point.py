r"""Plain (non-equivariant) point-cloud baseline — the 3D moat's other side (manifest T0.2).

PointNet-style: per-point shared MLP -> symmetric pool (max) -> global feature. NO equivariance
(the latent does NOT co-rotate with the cloud). Matched to the VN stack's interface: same flat
latent dim and the same train_jepa wrapper contract (.encoder / .predictor). Param count is
tuned (via `width`) to match VNDGCNNEncoder(c_vec=32,k=16,width=64) at call sites.

Honest framing: this is the fair non-equivariant control for the 3D reliability/θ comparisons —
permutation-invariant (a point cloud has no order) but NOT rotation-equivariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    r"""Permutation-invariant, NON-equivariant cloud encoder. ``(B, N, 3) -> (B, latent_dim)``."""

    def __init__(self, latent_dim: int = 192, width: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width),
        )
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)          # center (translation quotient, like VN)
        f = self.point_mlp(x)                         # (B, N, width)
        g = f.max(dim=1).values                       # symmetric pool (B, width)
        return self.head(g)


class PlainPredictor(nn.Module):
    r"""Action-conditioned residual MLP on the flat latent. ``z (B,D), a (B,A) -> (B,D)``."""

    def __init__(self, latent_dim: int = 192, action_dim: int = 7, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))
