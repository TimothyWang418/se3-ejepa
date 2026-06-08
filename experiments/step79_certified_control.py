r"""Step 79 — certified-control co-demonstration on controlled Lorenz-96.

Anchor experiment for the two-axis co-demonstration (design: docs/specs/2026-06-07-step5-certified-control-codemo.md;
plan: docs/plans/2026-06-07-step79-certified-control.md). Phase 0: the controlled dynamics + Z_N-equivariance.
"""
import torch

DTYPE = torch.float64
F_FORCE = 8.0
DT = 0.01


def l96_controlled_rhs(x: torch.Tensor, u: torch.Tensor, F: float = F_FORCE) -> torch.Tensor:
    r"""Controlled Lorenz-96 field $\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F+u_i$. Jointly $\mathbb{Z}_N$-equivariant in
    $(x,u)$: a cyclic shift of both inputs shifts the output (roll commutes with the local coupling and with $u$)."""
    return (torch.roll(x, -1, -1) - torch.roll(x, 2, -1)) * torch.roll(x, 1, -1) - x + F + u


def rk4_controlled(x: torch.Tensor, u: torch.Tensor, dt: float = DT, F: float = F_FORCE) -> torch.Tensor:
    r"""One RK4 step of the controlled $\Delta t$-map with a zero-order-hold control $u$ (held constant over the step)."""
    k1 = l96_controlled_rhs(x, u, F); k2 = l96_controlled_rhs(x + 0.5 * dt * k1, u, F)
    k3 = l96_controlled_rhs(x + 0.5 * dt * k2, u, F); k4 = l96_controlled_rhs(x + dt * k3, u, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
