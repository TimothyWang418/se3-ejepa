r"""Vector Neuron (VN) SO(3)-equivariant point-cloud JEPA components — paper3 3D lane.

Math (Deng et al. 2021, *Vector Neurons: A General Framework for SO(3)-Equivariant Networks*,
ICCV 2021; DGCNN graph features from Wang et al. 2019). A VN feature is a channel list of
3-vectors $V \in \mathbb{R}^{C \times 3}$. For a rotation $R \in SO(3)$ acting on points
$x \mapsto R x$ (so a cloud $X \in \mathbb{R}^{N \times 3}$ maps to $X R^\top$), features
co-rotate $V \mapsto V R^\top$. Three facts carry the whole design:

1. **VNLinear** $V \mapsto W V$ (mix channels, never touch the 3-axis) commutes with
   $\cdot R^\top$ by associativity: $(W V) R^\top = W (V R^\top)$. Equivariance is
   architectural, exact in exact arithmetic (float: matmul-order epsilon only).
2. **VNLeakyReLU** gates each channel $v$ by a *learned co-rotating direction* $k$: the test
   $\langle v, k \rangle$ is invariant, and the surgery
   $v \mapsto v - \min(0, \langle v, \hat k \rangle)\, \hat k$ is equivariant.
3. **Invariants** enter only through inner products of co-rotating quantities
   $\langle V_c, \hat k_j \rangle$; translations are removed up front by centering the cloud
   ($SE(3)$ = centroid $\oplus$ $SO(3)$ on the centered shape).

Latent layout (flat, for JEPA/audit plumbing): $z = [\mathrm{vec}(V)\,;\, I] \in
\mathbb{R}^{3C + C_I}$ with $V \in \mathbb{R}^{C\times 3}$ equivariant and $I$ invariant.
The group action on $z$ is exposed by :func:`rotate_latent` — this is the transport the
paper2 Thm B machinery consumes ($m\,\hat\epsilon_{\max}$ terms need an explicit $g$-action).

Repo rule: every layer here has an equivariance unit test in ``tests/test_p4_3d_g0.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-8


class VNLinear(nn.Module):
    r"""Channel-mixing linear map on vector features: $V' = W V$.

    Shapes: ``(B, C_in, 3, *spatial) -> (B, C_out, 3, *spatial)`` — the 3-axis and any
    trailing point/neighbor axes are untouched, so equivariance is exact by construction.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        # Kaiming-style scale on the channel dimension (the only mixing dimension).
        self.weight = nn.Parameter(torch.randn(c_out, c_in) / (c_in ** 0.5))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # einsum over the channel axis only: (oc,ic) x (B,ic,3,...) -> (B,oc,3,...)
        return torch.einsum("oi,bi...->bo...", self.weight, v)


class VNLeakyReLU(nn.Module):
    r"""Direction-gated LeakyReLU (Deng et al. 2021, Eq. 6).

    For each channel $v$ and learned direction $k = (UV)$ (itself a VN feature, hence
    co-rotating): with $\hat k = k / \|k\|$ and $d = \langle v, \hat k \rangle$,

    $$ v' = \alpha v + (1-\alpha)\,\bigl(v - \min(d, 0)\, \hat k\bigr). $$

    $d$ is rotation-invariant and $\hat k$ co-rotates, so $v'$ co-rotates. When $d \ge 0$
    the nonlinearity is the identity on that channel (the "positive half-space").
    """

    def __init__(self, c: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.dir_map = VNLinear(c, c)
        self.alpha = negative_slope

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        k = self.dir_map(v)                                    # (B, C, 3, ...)
        kn = k / (k.norm(dim=2, keepdim=True) + EPS)           # \hat k
        d = (v * kn).sum(dim=2, keepdim=True)                  # invariant gate
        relu_part = v - torch.clamp(d, max=0.0) * kn           # subtract negative component
        return self.alpha * v + (1.0 - self.alpha) * relu_part


class VNInvariant(nn.Module):
    r"""Invariant readout: coordinates of $V$ in a learned co-rotating (non-orthonormal) frame.

    $I_{c,j} = \langle V_c, \hat k_j \rangle$ for $j = 1..3$ learned directions — each entry is
    an inner product of two co-rotating vectors, hence rotation-invariant. (Deng et al.'s
    VN-STN uses an orthonormalized frame; for the JEPA latent the raw normalized-frame
    coordinates are sufficient and cheaper.)

    Shapes: ``(B, C, 3) -> (B, C*3)``.
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        self.frame_map = VNLinear(c, 3)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        k = self.frame_map(v)                                  # (B, 3, 3): 3 direction channels
        kn = k / (k.norm(dim=2, keepdim=True) + EPS)
        # (B, C, 3) x (B, 3dirs, 3) -> (B, C, 3dirs), all pairwise inner products
        return torch.einsum("bcx,bjx->bcj", v, kn).flatten(1)


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    r"""Indices of the $k$ nearest neighbors per point. ``(B, N, 3) -> (B, N, k)`` (long).

    Pairwise distances are rotation- and translation-invariant, so the graph is too
    (float caveat: ties could flip under rotation; generic clouds have none a.s.).
    """
    d = torch.cdist(x, x)                                      # (B, N, N)
    return d.topk(k + 1, largest=False).indices[..., 1:]       # drop self


class VNDGCNNEncoder(nn.Module):
    r"""Minimal VN-DGCNN encoder: centered cloud -> flat latent $[\mathrm{vec}(V); I]$.

    Pipeline (all-equivariant): center cloud -> knn edge features $(x_i,\, x_j - x_i)$ as two
    vector channels -> VN-MLP -> mean over neighbors -> VN-MLP -> mean over points -> heads.
    Mean pooling everywhere (exactly equivariant; Deng et al.'s VN-max is a later refinement).

    forward: ``(B, N, 3) float -> (B, 3*c_vec + 3*c_vec)``  (invariant head emits 3 coords
    per vector channel, see :class:`VNInvariant`).
    """

    def __init__(self, c_vec: int = 32, k: int = 16, width: int = 64) -> None:
        super().__init__()
        self.k = k
        self.c_vec = c_vec
        # OPTIMIZED (2026-06-13): early-pool DGCNN. Only ONE VN op pair runs at the expensive
        # (B, W, 3, N, k) edge resolution; the deep MLP runs at (B, W, 3, N) — 16× smaller. The
        # original "4 VNLinears before pool" peaked 11.1GB (> 3080's 10GB → swap → 20s/batch).
        # This is a standard DGCNN edge-conv design; equivariance preserved (all VN ops; V-III
        # re-tested). Function class differs from the pre-06-13 encoder → 3D runs re-baselined.
        self.edge_conv = nn.Sequential(VNLinear(2, width), VNLeakyReLU(width))
        self.point1 = nn.Sequential(VNLinear(width, width), VNLeakyReLU(width),
                                    VNLinear(width, width), VNLeakyReLU(width))
        self.head_vec = VNLinear(width, c_vec)
        self.head_inv = VNInvariant(c_vec)

    @property
    def latent_dim(self) -> int:
        return 3 * self.c_vec + 3 * self.c_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x - x.mean(dim=1, keepdim=True)                    # translation quotient
        idx = knn_graph(x, self.k)                             # (B, N, k)
        nbr = torch.gather(x.unsqueeze(1).expand(b, n, n, 3), 2,
                           idx.unsqueeze(-1).expand(b, n, self.k, 3))   # (B, N, k, 3)
        # two vector channels per edge: anchor x_i and relative x_j - x_i
        anchor = x.unsqueeze(2).expand(b, n, self.k, 3)
        e = torch.stack([anchor, nbr - anchor], dim=1)          # (B, 2, N, k, 3)
        e = e.permute(0, 1, 4, 2, 3)                            # (B, 2, 3, N, k)
        h = self.edge_conv(e).mean(dim=-1)                      # edge-conv then POOL: (B, W, 3, N)
        h = self.point1(h).mean(dim=-1)                         # deep MLP at point res: (B, W, 3)
        v = self.head_vec(h)                                    # (B, c_vec, 3)
        return torch.cat([v.flatten(1), self.head_inv(v)], dim=1)


class VNPredictor(nn.Module):
    r"""Action-conditioned equivariant predictor on the structured latent (residual).

    Action contract (3D ee-delta control): ``a = [Δp(3), ω(3), inv(rest)]`` — $\Delta p$ and
    the axis-angle $\omega$ transform as vectors ($\omega \mapsto R\omega$ is the adjoint
    action of $SO(3)$ on its Lie algebra $\mathfrak{so}(3) \cong \mathbb{R}^3$), gripper etc.
    are invariant scalars.

    Vector path: $[V; \Delta p; \omega]$ -> VN-MLP -> $\Delta V$, channel-wise rescaled by an
    invariant gain $g_c = \mathrm{MLP}(I, \|\Delta p\|, \|\omega\|, a_{\mathrm{inv}})$
    (scalar × vector is equivariant — this is how invariants steer the vector flow).
    Invariant path: plain MLP residual on $I$.

    forward: ``z (B, 3C+I), a (B, 6+n_inv) -> (B, 3C+I)``.
    """

    def __init__(self, c_vec: int, n_inv: int, a_inv_dim: int = 1, width: int = 64) -> None:
        super().__init__()
        self.c_vec, self.n_inv = c_vec, n_inv
        self.vec_mlp = nn.Sequential(VNLinear(c_vec + 2, width), VNLeakyReLU(width),
                                     VNLinear(width, c_vec))
        gate_in = n_inv + 2 + a_inv_dim
        self.gate = nn.Sequential(nn.Linear(gate_in, width), nn.SiLU(),
                                  nn.Linear(width, c_vec), nn.Tanh())
        self.inv_mlp = nn.Sequential(nn.Linear(gate_in, width), nn.SiLU(),
                                     nn.Linear(width, n_inv))

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = z[:, : 3 * self.c_vec].reshape(-1, self.c_vec, 3)
        return v, z[:, 3 * self.c_vec:]

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        v, i = self.split(z)
        dp, om, a_inv = a[:, 0:3], a[:, 3:6], a[:, 6:]
        v_aug = torch.cat([v, dp.unsqueeze(1), om.unsqueeze(1)], dim=1)   # (B, C+2, 3)
        scal = torch.cat([i, dp.norm(dim=1, keepdim=True),
                          om.norm(dim=1, keepdim=True), a_inv], dim=1)
        dv = self.vec_mlp(v_aug) * self.gate(scal).unsqueeze(-1)   # invariant gain × vector
        di = self.inv_mlp(scal)
        return torch.cat([(v + dv).flatten(1), i + di], dim=1)


def rotate_latent(z: torch.Tensor, r: torch.Tensor, c_vec: int) -> torch.Tensor:
    r"""Group action of $R$ on the flat latent: rotate the vector block, fix the invariants.

    $z = [\mathrm{vec}(V); I] \mapsto [\mathrm{vec}(V R^\top); I]$ — the transport used by
    equivariance tests and (later) by Thm-B-style orbit-transported audits.
    """
    v = z[:, : 3 * c_vec].reshape(-1, c_vec, 3)
    return torch.cat([(v @ r.T).flatten(1), z[:, 3 * c_vec:]], dim=1)


def rotate_action(a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""$[\Delta p; \omega; a_{\mathrm{inv}}] \mapsto [R\Delta p; R\omega; a_{\mathrm{inv}}]$."""
    return torch.cat([a[:, 0:3] @ r.T, a[:, 3:6] @ r.T, a[:, 6:]], dim=1)


def random_rotation(dtype: torch.dtype = torch.float64,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    r"""Haar-ish random $R \in SO(3)$ via QR of a Gaussian matrix (det corrected to $+1$)."""
    m = torch.randn(3, 3, dtype=dtype, generator=generator)
    q, rr = torch.linalg.qr(m)
    q = q * torch.sign(torch.diagonal(rr)).unsqueeze(0)        # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q
