r"""Structured-state (coordinate) path: an *exactly* SO(2)-equivariant encoder.

Motivation. The pixel encoder (Step 3/4) is equivariant by construction but can
only be *measured* exactly at $90^\circ$ rotations — off-grid angles hit a
bilinear-interpolation floor (~$10^{-1}$). PushT, however, exposes its full SE(2)
state (agent position, block pose $(x,y,\theta)$, goal pose), so we can encode
**coordinates** instead of pixels and obtain equivariance that is exact for
*every* angle $\alpha$, down to the float floor (~$10^{-6}$).

Representation. We turn the state into a set of 2D **type-1 vectors** — objects
$v$ that transform as $v \mapsto R(\alpha)\,v$ under a global rotation:

* positions (centred at the canvas centre): agent, block, goal;
* orientations as unit vectors $(\cos\theta,\sin\theta)$ (since $\theta\mapsto\theta+\alpha$);
* the agent velocity.

Encoder. A 2D analogue of **Vector Neurons** (Deng et al., *Vector Neurons: A
General Framework for SO(3)-Equivariant Networks*, ICCV 2021). A VN-linear layer
mixes vector channels with a plain matrix $W$ acting on the channel axis,
$V'_{o} = \sum_i W_{oi} V_i$, which commutes with rotation because $R$ acts on the
*spatial* axis. A VN-nonlinearity rectifies each vector against a learned,
equivariant direction. Stacking these yields a latent that is a set of vectors;
flattened, the latent $z$ satisfies $z(R\cdot x) = \rho(R)\,z(x)$ with $\rho(R)$
block-diagonal rotations — **orthogonal**, hence the JEPA cost is rotation-invariant.

This is the 2D rehearsal for the e3nn SE(3) encoder in Step 6.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

__all__ = [
    "VNLinear",
    "VNReLU",
    "StructuredStateEncoder",
    "VNPredictor",
    "extract_pusht_vectors",
]


class VNLinear(nn.Module):
    r"""Vector-Neuron linear map. ``(B, C_in, 2) -> (B, C_out, 2)``.

    Mixes channels by $V'_{o} = \sum_i W_{oi} V_i$. No bias: a constant offset
    vector is not rotation-equivariant (it would not transform with the input).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, d) -> (B, C_out, d); rotation acts on d, W on the channel axis
        return torch.einsum("oi,bid->bod", self.weight, x)


class VNReLU(nn.Module):
    r"""Vector-Neuron nonlinearity (equivariant). ``(B, C, 2) -> (B, C, 2)``.

    For each channel, predicts an equivariant direction $q = K V$ and removes the
    component of $V$ opposing $q$:
    $$ V' = \begin{cases} V & \langle V, \hat q\rangle \ge 0 \\ V - \langle V,\hat q\rangle\,\hat q & \text{otherwise.}\end{cases} $$
    Every operation is built from equivariant vectors and invariant inner
    products, so $V'(R\cdot V) = R\cdot V'(V)$.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.dir = VNLinear(channels, channels)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.dir(x)  # (B, C, 2) equivariant direction
        q_hat = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        dot = (x * q_hat).sum(dim=-1, keepdim=True)  # (B, C, 1) invariant scalar
        return torch.where(dot >= 0, x, x - dot * q_hat)


class StructuredStateEncoder(nn.Module):
    r"""SO(2)-equivariant coordinate encoder: a set of 2D vectors -> latent.

    ``forward: (B, n_vec, 2) -> (B, latent_dim)`` where ``latent_dim`` must be
    even (the latent is ``latent_dim/2`` 2D vectors, flattened). The map is exact
    for arbitrary continuous rotations — there is no spatial grid to interpolate.
    """

    def __init__(self, n_vec: int = 6, latent_dim: int = 128, hidden: int = 64):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be even (vectors are 2D).")
        self.latent_dim = latent_dim
        self.l1 = VNLinear(n_vec, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, hidden)
        self.a2 = VNReLU(hidden)
        self.l3 = VNLinear(hidden, latent_dim // 2)

    def encode_vectors(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return the equivariant vector latent ``(B, latent_dim/2, 2)``."""
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return self.l3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_vec, 2) -> (B, latent_dim)
        return self.encode_vectors(x).flatten(1)


class VNPredictor(nn.Module):
    r"""Jointly-equivariant latent dynamics $f_\phi(z, a)$ (2D / SO(2)).

    The latent $z$ is a set of type-1 vectors and the action $a$ is a type-1
    vector; the action is concatenated as an extra vector channel and a VN-MLP
    predicts a residual update. Because every operation is VN-equivariant and a
    residual sum of equivariant tensors is equivariant, the predictor satisfies

    $$ f_\phi\big(\rho(g)\,z,\; R(g)\,a\big) = \rho(g)\, f_\phi(z, a), $$

    i.e. rotating the scene *and* the action rotates the predicted next latent the
    same way. Combined with the equivariant encoder this makes the **entire**
    latent world model SO(2)-equivariant, so the planning cost is invariant to a
    joint rotation of (state, goal, action sequence) — not just the encoder.

    Interface matches :class:`~src.models.eqjepa.LatentPredictor` so it drops into
    :meth:`EqJEPA.rollout`:  ``forward((B, latent_dim), (B, action_dim)) -> (B, latent_dim)``.

    ``dim`` selects the vector dimension: ``2`` for the SO(2) / PushT path,
    ``3`` for the SO(3) / point-cloud path (VN layers are dimension-agnostic and
    equivariant for both, so the same predictor serves the e3nn SE(3) encoder).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        hidden: int = 64,
        dim: int = 2,
    ):
        super().__init__()
        if latent_dim % dim != 0 or action_dim % dim != 0:
            raise ValueError(f"latent_dim and action_dim must be divisible by dim={dim}.")
        self.dim = dim
        self.n_lat = latent_dim // dim
        self.n_act = action_dim // dim
        self.l1 = VNLinear(self.n_lat + self.n_act, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, hidden)
        self.a2 = VNReLU(hidden)
        self.l3 = VNLinear(hidden, self.n_lat)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), a: (B, action_dim) -> (B, latent_dim)
        b = z.shape[0]
        z_vec = z.reshape(b, self.n_lat, self.dim)
        a_vec = a.reshape(b, self.n_act, self.dim)
        x = torch.cat([z_vec, a_vec], dim=1)  # (B, n_lat + n_act, dim)
        delta = self.l3(self.a2(self.l2(self.a1(self.l1(x)))))
        return (z_vec + delta).reshape(b, -1)  # residual, equivariant


def extract_pusht_vectors(
    infos: dict,
    *,
    center: float = 256.0,
    pos_scale: float = 256.0,
    vel_scale: float = 512.0,
) -> torch.Tensor:
    r"""Turn a PushT ``infos`` dict into type-1 vectors. ``-> (B, 6, 2)``.

    Channels (all transform as $v\mapsto R(\alpha)v$ under a global scene
    rotation about the canvas centre):
    ``[agent_pos, block_pos, goal_pos, block_dir, goal_dir, agent_vel]``.
    Positions are centred and scaled to ~$[-1,1]$; orientations become unit
    vectors $(\cos\theta,\sin\theta)$; velocity is scaled.
    """

    def last(key: str) -> np.ndarray:
        v = np.asarray(infos[key])
        return v[:, -1] if v.ndim >= 3 else v  # drop leading time dim

    pos_agent = last("pos_agent")  # (B, 2)
    block_pose = last("block_pose")  # (B, 3) = (x, y, theta)
    goal_pose = last("goal_pose")  # (B, 3)
    vel_agent = last("vel_agent")  # (B, 2)

    agent_xy = (pos_agent - center) / pos_scale
    block_xy = (block_pose[:, :2] - center) / pos_scale
    goal_xy = (goal_pose[:, :2] - center) / pos_scale
    block_dir = np.stack([np.cos(block_pose[:, 2]), np.sin(block_pose[:, 2])], axis=-1)
    goal_dir = np.stack([np.cos(goal_pose[:, 2]), np.sin(goal_pose[:, 2])], axis=-1)
    vel = vel_agent / vel_scale

    vecs = np.stack([agent_xy, block_xy, goal_xy, block_dir, goal_dir, vel], axis=1)
    return torch.from_numpy(vecs).float()  # (B, 6, 2)
