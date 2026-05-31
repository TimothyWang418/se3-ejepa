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
    "VNTensorProduct",
    "StructuredStateEncoder",
    "VNPredictor",
    "VNTPPredictor",
    "VNTPLadderPredictor",
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


class VNTensorProduct(nn.Module):
    r"""Equivariant **bilinear** (degree-2) layer via the SO(3) cross product. ``(B, C_in, 3) -> (B, C_out, 3)``.

    The vanilla VN layers (:class:`VNLinear`, :class:`VNReLU`) are **degree-1 homogeneous**:
    a composition only ever takes *linear combinations* of the input vectors, so it can never
    form a **product** of two distinct input vectors. The torque coupling in an interacting
    SO(3) world (Step 24), $\omega=\hat r\times a$, is exactly such a product, which is why a
    degree-1 VN predictor hits a hard cross-product ceiling and a plain MLP fits the interaction
    better in-distribution. This layer supplies the missing primitive.

    Group theory. For two type-1 (vector, $\ell=1$) features the tensor product decomposes as
    $$ \mathbf 1 \otimes \mathbf 1 \;=\; \mathbf 0 \,\oplus\, \mathbf 1 \,\oplus\, \mathbf 2 , $$
    a scalar (dot product), a vector (the **antisymmetric** part $=$ cross product), and a
    symmetric-traceless $\ell{=}2$ part. We realise the $\mathbf 1$ channel: project the input to
    two learned channel-mixes $U=W_uX$, $V=W_vX$ (each :class:`VNLinear`, so still equivariant) and
    return their channel-wise cross product
    $$ \mathrm{out}_o \;=\; U_o \times V_o , \qquad U_o,V_o\in\mathbb R^3 . $$

    Equivariance is exact for **proper** rotations $R\in\mathrm{SO}(3)$:
    $(RU)\times(RV)=R\,(U\times V)$. Under an *improper* $R$ ($\det R=-1$) the cross product is a
    **pseudovector**, $(RU)\times(RV)=\det(R)\,R(U\times V)=-R(U\times V)$ — so this layer is
    $\mathrm{SO}(3)$-equivariant but **not** $\mathrm{O}(3)$-equivariant. That is the correct and
    intended scope: the Step-24 teacher's torque is itself built from cross products and is likewise
    only $\mathrm{SO}(3)$- (not $\mathrm{O}(3)$-) equivariant, and the project tests generalisation
    over proper rotations throughout. There is no bias and no nonlinearity here — the bilinear map is
    its own (degree-2) nonlinearity.

    Only defined for ``dim == 3`` (in 2D the cross product $u\times v$ is a pseudo-*scalar*, the
    $\mathfrak{so}(2)$ generator coefficient — a different representation that would need a separate
    type-0 channel). Cf. Deng et al., *Vector Neurons* (ICCV 2021), which omits this $\ell{=}1$
    bilinear; the construction here is the SO(3) cross-product instance of an e3nn-style
    tensor-product layer (Geiger & Smidt, *e3nn*, 2022).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.wu = VNLinear(in_channels, out_channels)
        self.wv = VNLinear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, 3) -> (B, C_out, 3); both factors are equivariant channel-mixes, and the
        # cross product of two type-1 vectors is the equivariant (antisymmetric) 1⊗1->1 part.
        if x.shape[-1] != 3:
            raise ValueError(f"VNTensorProduct is SO(3)-only: expected last dim 3, got {x.shape[-1]}.")
        u = self.wu(x)
        v = self.wv(x)
        return torch.cross(u, v, dim=-1)


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


class VNTPPredictor(nn.Module):
    r"""SO(3)-equivariant latent dynamics with **tensor-product (bilinear) messages**.

    Identical interface and residual structure to :class:`VNPredictor`
    (``forward: (B, latent_dim), (B, action_dim) -> (B, latent_dim)``), but each hidden
    block carries a :class:`VNTensorProduct` branch alongside the usual :class:`VNLinear`
    branch. This lifts the predictor from **degree-1** (linear combinations only) to a
    polynomial of the input vectors, which is what the interacting SO(3) teacher requires.

    Why two tensor-product compositions. The Step-24 object-interaction teacher applies a
    **torque-on-points** update
    $$ \Delta \tilde x_k \;=\; \kappa\,\big(\omega_i \times \tilde x_k\big), \qquad
       \omega_i \;=\; \hat r_{ij}\times a_i , $$
    i.e. the target is the **trilinear** quantity $(\hat r_{ij}\times a_i)\times\tilde x_k$ —
    degree-3 in the input vectors. A degree-1 VN predictor cannot even form the inner product
    $\hat r_{ij}\times a_i$ (a degree-2 term), which is the ceiling diagnosed in Step 24. Here:

    * **block 1** mixes a linear pass-through ``lin1`` (keeps degree-1 copies of positions /
      action available downstream) with a tensor-product branch ``tp1`` that forms the degree-2
      cross products $\hat r_{ij}\times a_i$;
    * **block 2** crosses block-1's degree-2 channels against the surviving degree-1 channels
      (``tp2``), yielding the degree-3 torque $(\hat r_{ij}\times a_i)\times\tilde x_k$.

    Concatenating ``[lin, tp]`` at each block is what keeps **both** degrees alive, so the second
    cross product has a degree-1 factor to multiply. :class:`VNReLU` is degree-1 positively
    homogeneous, so it preserves these degrees while adding capacity.

    Equivariance. Every branch — :class:`VNLinear`, :class:`VNReLU`, :class:`VNTensorProduct` —
    is $\mathrm{SO}(3)$-equivariant, and a residual sum of equivariant tensors is equivariant, so
    $$ f_\phi\big(\rho(g)\,z,\; R(g)\,a\big) \;=\; \rho(g)\, f_\phi(z, a), \qquad g\in\mathrm{SO}(3). $$
    Because the cross product is a **pseudovector**, the predictor is $\mathrm{SO}(3)$- but not
    $\mathrm{O}(3)$-equivariant — matching the teacher, which is built from the same cross products.
    ``dim`` must be ``3`` (the cross product is the $\ell{=}1$ piece of $\mathbf 1\otimes\mathbf 1$
    in 3D; see :class:`VNTensorProduct`). Cf. Deng et al., *Vector Neurons* (ICCV 2021) for the
    linear/nonlinear primitives and Geiger & Smidt, *e3nn* (2022) for the tensor-product view.
    """

    def __init__(
        self,
        latent_dim: int = 48,
        action_dim: int = 6,
        hidden: int = 64,
        dim: int = 3,
        tp_channels: int | None = None,
    ):
        super().__init__()
        if dim != 3:
            raise ValueError(f"VNTPPredictor uses the SO(3) cross product; dim must be 3, got {dim}.")
        if latent_dim % dim != 0 or action_dim % dim != 0:
            raise ValueError(f"latent_dim and action_dim must be divisible by dim={dim}.")
        self.dim = dim
        self.n_lat = latent_dim // dim
        self.n_act = action_dim // dim
        n_in = self.n_lat + self.n_act
        tp = tp_channels if tp_channels is not None else hidden
        # block 1: degree-1 pass-through ++ degree-2 cross products of the raw [z, a] channels
        self.lin1 = VNLinear(n_in, hidden)
        self.tp1 = VNTensorProduct(n_in, tp)
        self.act1 = VNReLU(hidden + tp)
        # block 2: a second tensor product -> reaches the degree-3 (trilinear) torque
        self.lin2 = VNLinear(hidden + tp, hidden)
        self.tp2 = VNTensorProduct(hidden + tp, tp)
        self.act2 = VNReLU(hidden + tp)
        # project the degree-{1,2,3} feature stack down to the latent residual
        self.out = VNLinear(hidden + tp, self.n_lat)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), a: (B, action_dim) -> (B, latent_dim)
        b = z.shape[0]
        z_vec = z.reshape(b, self.n_lat, self.dim)
        a_vec = a.reshape(b, self.n_act, self.dim)
        x = torch.cat([z_vec, a_vec], dim=1)  # (B, n_lat + n_act, 3)
        h = torch.cat([self.lin1(x), self.tp1(x)], dim=1)  # (B, hidden + tp, 3): degree-{1, 2}
        h = self.act1(h)
        h = torch.cat([self.lin2(h), self.tp2(h)], dim=1)  # second TP -> includes degree-3
        h = self.act2(h)
        delta = self.out(h)
        return (z_vec + delta).reshape(b, -1)  # residual, equivariant


class VNTPLadderPredictor(nn.Module):
    r"""SO(3)-equivariant latent dynamics with a **tunable tensor-product degree** (Step 32).

    A *degree ladder* generalising :class:`VNPredictor` (degree-1) and :class:`VNTPPredictor`
    (a fixed two-tensor-product stack). The predictor is a fixed stack of ``n_blocks_total``
    equivariant blocks; the **first** ``n_tp_blocks`` of them carry a :class:`VNTensorProduct`
    (cross-product) branch and the rest are pure :class:`VNLinear`. Stacking $L$ tensor-product
    blocks *at the front* lets the cross products **compound**, so the maximum representable
    polynomial degree in the input vectors is

    $$ d_{\max}(L) \;=\; 2^{L}, \qquad L=\texttt{n\_tp\_blocks}, $$

    because each cross product multiplies degrees: block $0$ produces degree-$\{1,2\}$ features,
    block $1$ crosses those to reach degree-$\{2,3,4\}$, block $2$ reaches up to degree-$8$, and so
    on. The Step-24 interacting teacher's torque target $(\hat r_{ij}\times a_i)\times\tilde x_k$ is
    **degree-3**, so the *first* rung that can represent it is $L=2$ ($d_{\max}=4\ge 3$); $L=0,1$ are
    structurally capped and $L\ge 2$ should saturate — a degree signature, not a capacity one.

    **Constant-width, constant-depth control.** Every block — TP or linear — outputs exactly
    ``hidden`` channels (a TP block splits its width as ``hidden - tp_channels`` linear $+$
    ``tp_channels`` cross-product, then concatenates), and the number of blocks (hence the number of
    :class:`VNReLU` nonlinearities) is fixed at ``n_blocks_total`` for every rung. So sweeping
    ``n_tp_blocks`` varies **only** the representable degree at near-constant depth, width, and
    parameter count — isolating *degree* from raw capacity (a TP block carries only
    ``c * tp_channels`` extra weights over a linear block, reported per rung by the caller).

    Equivariance. Every branch (:class:`VNLinear`, :class:`VNReLU`, :class:`VNTensorProduct`) is
    $\mathrm{SO}(3)$-equivariant and a residual sum of equivariant tensors is equivariant, so for
    **every** rung $L$
    $$ f_\phi\big(\rho(g)\,z,\; R(g)\,a\big) \;=\; \rho(g)\, f_\phi(z, a), \qquad g\in\mathrm{SO}(3). $$
    Because the cross product is a pseudovector the predictor is $\mathrm{SO}(3)$- but not
    $\mathrm{O}(3)$-equivariant — matching the teacher. ``dim`` must be ``3``.

    Interface matches :class:`VNPredictor` / :class:`VNTPPredictor`:
    ``forward: (B, latent_dim), (B, action_dim) -> (B, latent_dim)``.

    Args:
        latent_dim: total latent width; the latent is ``latent_dim // dim`` type-1 vectors.
        action_dim: total action width; ``action_dim // dim`` type-1 vectors are concatenated.
        hidden: per-block output channel count (constant across the ladder).
        dim: vector dimension; must be ``3`` (SO(3) cross product).
        n_tp_blocks: number $L$ of front-loaded tensor-product blocks (the ladder rung,
            $0\le L\le$ ``n_blocks_total``). $L=0$ recovers a degree-1 VN-MLP.
        n_blocks_total: fixed total number of blocks (depth), held constant across rungs.
        tp_channels: cross-product channels inside each TP block (``hidden // 2`` if ``None``);
            the linear branch of a TP block then has ``hidden - tp_channels`` channels.
    """

    def __init__(
        self,
        latent_dim: int = 48,
        action_dim: int = 6,
        hidden: int = 64,
        dim: int = 3,
        n_tp_blocks: int = 2,
        n_blocks_total: int = 3,
        tp_channels: int | None = None,
    ):
        super().__init__()
        if dim != 3:
            raise ValueError(f"VNTPLadderPredictor uses the SO(3) cross product; dim must be 3, got {dim}.")
        if latent_dim % dim != 0 or action_dim % dim != 0:
            raise ValueError(f"latent_dim and action_dim must be divisible by dim={dim}.")
        if not (0 <= n_tp_blocks <= n_blocks_total):
            raise ValueError(f"need 0 <= n_tp_blocks ({n_tp_blocks}) <= n_blocks_total ({n_blocks_total}).")
        self.dim = dim
        self.n_lat = latent_dim // dim
        self.n_act = action_dim // dim
        self.n_tp_blocks = n_tp_blocks
        self.n_blocks_total = n_blocks_total
        tp = tp_channels if tp_channels is not None else hidden // 2
        if n_tp_blocks > 0 and not (1 <= tp <= hidden - 1):
            raise ValueError(f"tp_channels ({tp}) must satisfy 1 <= tp <= hidden-1 ({hidden - 1}).")

        c = self.n_lat + self.n_act  # input channel count: [z vectors, (a, message) vectors]
        self.blocks = nn.ModuleList()
        for i in range(n_blocks_total):
            use_tp = i < n_tp_blocks  # front-load TP blocks so the cross products compound
            block = nn.ModuleDict()
            if use_tp:
                # split the block width: a degree-1 linear branch ++ a degree-2 cross-product branch,
                # concatenated to exactly `hidden` channels (constant width across the ladder).
                block["lin"] = VNLinear(c, hidden - tp)
                block["tp"] = VNTensorProduct(c, tp)
            else:
                block["lin"] = VNLinear(c, hidden)
            block["act"] = VNReLU(hidden)
            self.blocks.append(block)
            c = hidden
        self._use_tp = [i < n_tp_blocks for i in range(n_blocks_total)]
        self.out = VNLinear(hidden, self.n_lat)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), a: (B, action_dim) -> (B, latent_dim)
        b = z.shape[0]
        z_vec = z.reshape(b, self.n_lat, self.dim)
        a_vec = a.reshape(b, self.n_act, self.dim)
        h = torch.cat([z_vec, a_vec], dim=1)  # (B, n_lat + n_act, 3)
        for block, use_tp in zip(self.blocks, self._use_tp):
            if use_tp:
                h = torch.cat([block["lin"](h), block["tp"](h)], dim=1)  # (B, hidden, 3): degree compounds
            else:
                h = block["lin"](h)  # (B, hidden, 3): degree-1 mix
            h = block["act"](h)
        delta = self.out(h)  # (B, n_lat, 3)
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
