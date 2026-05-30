r"""SO(2) utilities for the coordinate / structured-state path.

The pixel-grid encoder (Step 3) is only float-exact for $90^\circ$ rotations
because rotating a square raster by an off-grid angle needs interpolation. Acting
directly on **coordinates** removes that floor: a 2D vector $v$ transforms as
$v \mapsto R(\alpha)\,v$ for *any* angle $\alpha$, exactly.

$$ R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix} \in \mathrm{SO}(2). $$

Convention: a "vector field" tensor has shape ``(..., 2)`` (the last axis is the
2D vector). Rotation is applied on the right as ``x @ R(alpha).T`` so that each
row vector $v$ maps to $R(\alpha) v$.
"""

from __future__ import annotations

import math

import torch


def rot_matrix(alpha: float | torch.Tensor, *, device=None, dtype=torch.float32) -> torch.Tensor:
    r"""Return the $2\times2$ rotation matrix $R(\alpha)\in\mathrm{SO}(2)$.

    ``alpha`` may be a python float or a tensor of angles; for a tensor of shape
    ``(...,)`` the result has shape ``(..., 2, 2)``.
    """
    a = torch.as_tensor(alpha, device=device, dtype=dtype)
    c, s = torch.cos(a), torch.sin(a)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)  # (..., 2, 2)


def rotate_vectors(x: torch.Tensor, alpha: float | torch.Tensor) -> torch.Tensor:
    r"""Rotate a vector-field tensor by angle ``alpha``.

    ``x: (..., 2) -> (..., 2)`` with each row vector $v \mapsto R(\alpha)v$.
    Implemented as ``x @ R(alpha).T`` (a single scalar angle for the whole batch).
    """
    R = rot_matrix(alpha, device=x.device, dtype=x.dtype)  # (2, 2)
    return x @ R.transpose(-1, -2)


def angle_to_unit_vector(theta: torch.Tensor) -> torch.Tensor:
    r"""Map an orientation scalar $\theta$ to the unit vector $(\cos\theta, \sin\theta)$.

    Under a global rotation by $\alpha$, an orientation transforms as
    $\theta \mapsto \theta + \alpha$, so its unit vector transforms as a genuine
    type-1 vector: $(\cos\theta,\sin\theta) \mapsto R(\alpha)(\cos\theta,\sin\theta)$.
    This lets us treat angles on the same footing as positions and velocities.

    ``theta: (...) -> (..., 2)``
    """
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def random_angle(generator: torch.Generator | None = None) -> float:
    r"""Sample a uniform angle in $[0, 2\pi)$ (for off-grid equivariance tests)."""
    u = torch.rand((), generator=generator).item()
    return 2.0 * math.pi * u
