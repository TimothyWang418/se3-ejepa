r"""SE(3)-equivariant point-cloud encoder (Step 6 — the 3D payoff).

The coordinate path (Step 5) gave us *exact* continuous SO(2) on PushT's 2D
state. Step 6 lifts that to **3D**: a point cloud $\{x_i\}\subset\mathbb{R}^3$ is
encoded to a latent that is

* **translation-invariant** — we centre the cloud, so a global shift
  $x_i\mapsto x_i+t$ leaves the encoder output unchanged;
* **SO(3)-equivariant** — a global rotation $x_i\mapsto R\,x_i$ rotates the
  latent by the *same* representation, exactly (down to the float floor), for
  *every* $R\in\mathrm{SO}(3)$ — no grid, no interpolation.

Architecture. A minimal **Tensor Field Network** / NequIP-style aggregation
(Thomas et al. 2018; Batzner et al. 2022) built from e3nn primitives:

1. Centre the cloud (translation invariance), giving relative vectors $r_i$.
2. Embed each $r_i$ on the sphere via real spherical harmonics
   $Y^{(\ell)}(\hat r_i)$ up to $\ell=\ell_{\max}$ (an SO(3)-equivariant feature).
3. A radial MLP on the *invariant* length $\lVert r_i\rVert$ supplies the weights
   of a tensor product $\,\text{node}\otimes Y\,$ — equivariant message per point.
4. **Sum** over points (a sum of equivariant features is equivariant) to a single
   global descriptor, then a couple of equivariant ``o3.Linear`` +
   ``NormActivation`` layers.
5. Read out $N$ vectors of type $\ell=1$ (``Nx1o``).

Convention bridge. e3nn orders the $\ell=1$ basis as $(y,z,x)$ and acts on it
with the Wigner matrix $D^{(1)}(R)=P\,R\,P^{-1}$ (a permutation $P$ of the raw
axes). We undo $P$ at the readout so the returned vectors live in the **standard
$(x,y,z)$ basis and transform by the plain rotation** $v\mapsto R\,v$. That makes
the latent a set of ordinary type-1 vectors — exactly what the dimension-agnostic
:class:`~src.models.structured.VNPredictor` (with ``dim=3``) consumes, so the same
Vector-Neuron dynamics that served SO(2) now closes the loop for SO(3): encoder
*and* predictor are jointly equivariant, hence the **whole** 3D world model is.
"""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.nn import NormActivation
from torch import nn

__all__ = ["SE3PointEncoder"]


class SE3PointEncoder(nn.Module):
    r"""SE(3)-equivariant point-cloud encoder. ``(B, N, 3) -> (B, 3*n_out_vec)``.

    The flattened latent is ``n_out_vec`` stacked 3D vectors (type $\ell=1$) in
    the standard $(x,y,z)$ basis, so ``latent_dim = 3 * n_out_vec`` and the
    representation $\rho(R)$ is block-diagonal copies of $R$ — orthogonal, hence
    the JEPA L2 cost is rotation-invariant. Pair with
    :class:`~src.models.structured.VNPredictor` (``dim=3``) for an equivariant
    latent world model.

    Args:
        n_out_vec: number of $\ell=1$ output vectors (``latent_dim = 3*n_out_vec``).
        lmax: maximum spherical-harmonic degree for the geometric embedding.
        mul: channel multiplicity of each irrep in the hidden layer.
        n_radial: number of Gaussian radial-basis functions on $\lVert r\rVert$.
        r_max: spatial scale of the radial basis (RBF centres span $[0, r_{\max}]$).
    """

    def __init__(
        self,
        n_out_vec: int = 8,
        lmax: int = 2,
        mul: int = 8,
        n_radial: int = 8,
        r_max: float = 2.0,
    ):
        super().__init__()
        self.n_out_vec = n_out_vec
        self.latent_dim = 3 * n_out_vec

        # --- irreps ---------------------------------------------------------
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)  # 1x0e+1x1o+...
        irreps_node = o3.Irreps("1x0e")  # scalar seed feature (all ones)
        # hidden carries the irreps producible from node (x) sh: 0e, 1o, 2e, ...
        self.irreps_hidden = o3.Irreps(
            [(mul, (ir.l, ir.p)) for _, ir in self.irreps_sh]
        )
        irreps_out = o3.Irreps(f"{n_out_vec}x1o")

        # --- per-point tensor product (weights from the radial MLP) ---------
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_node,
            self.irreps_sh,
            self.irreps_hidden,
            shared_weights=False,
            internal_weights=False,
        )

        # --- radial basis + MLP on the invariant length ||r|| --------------
        centers = torch.linspace(0.0, r_max, n_radial)
        self.register_buffer("centers", centers)
        # Gaussian RBF width ~ centre spacing (smooth, invariant; value is free
        # for equivariance since ||r|| is rotation/translation-invariant).
        self.gamma = (n_radial / max(r_max, 1e-6)) ** 2
        self.radial_mlp = nn.Sequential(
            nn.Linear(n_radial, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )

        # --- equivariant head on the pooled global descriptor --------------
        self.act1 = NormActivation(self.irreps_hidden, torch.sigmoid)
        self.lin1 = o3.Linear(self.irreps_hidden, self.irreps_hidden)
        self.act2 = NormActivation(self.irreps_hidden, torch.sigmoid)
        self.lin_out = o3.Linear(self.irreps_hidden, irreps_out)

        # Change-of-basis from e3nn's 1o convention to the standard (x, y, z)
        # vector, so the readout transforms by the *plain* rotation v -> R v
        # (block-diagonal rho) rather than the Wigner matrix D^{(1)}(R). We derive
        # it from e3nn itself instead of hard-coding: M = Y^{(1)}(e_i) maps a raw
        # vector into the 1o basis, so Q = (M^T)^{-1} satisfies Q D^{(1)}(R) Q^{-1}
        # = R for every R. (For e3nn 0.6.0, M = sqrt(3) I, so Q is just a scale —
        # but computing it keeps the encoder correct under any convention change.)
        M = o3.spherical_harmonics(
            o3.Irreps("1x1o"), torch.eye(3), normalize=False, normalization="component"
        )
        self.register_buffer("basis_xyz", torch.linalg.inv(M).T)  # (3, 3)

    def encode_vectors(self, pos: torch.Tensor) -> torch.Tensor:
        r"""Return the equivariant vector latent ``(B, n_out_vec, 3)``.

        Each output row $v$ satisfies $v(R\,x) = R\,v(x)$ (standard $(x,y,z)$
        basis) and $v(x+t)=v(x)$ (translation invariance via centering).
        """
        # (B, N, 3) point cloud
        r = pos - pos.mean(dim=1, keepdim=True)  # centre: translation-invariant
        length = r.norm(dim=-1)  # (B, N) rotation/translation-invariant scalar

        # geometric embedding: spherical harmonics of the direction (equivariant)
        sh = o3.spherical_harmonics(
            self.irreps_sh, r, normalize=True, normalization="component"
        )  # (B, N, sh_dim)

        # radial weights from the invariant length
        rbf = torch.exp(-self.gamma * (length[..., None] - self.centers) ** 2)  # (B,N,n_radial)
        weights = self.radial_mlp(rbf)  # (B, N, tp.weight_numel)

        node = r.new_ones(*r.shape[:-1], 1)  # (B, N, 1) scalar seed
        msg = self.tp(node, sh, weights)  # (B, N, hidden_dim) equivariant
        h = msg.sum(dim=1)  # (B, hidden_dim) permutation-invariant aggregate

        h = self.act1(h)
        h = self.lin1(h)
        h = self.act2(h)
        out = self.lin_out(h)  # (B, n_out_vec*3) in e3nn (y,z,x) basis

        vecs = out.reshape(out.shape[0], self.n_out_vec, 3)
        return vecs @ self.basis_xyz.T  # e3nn 1o basis -> standard (x, y, z)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # (B, N, 3) -> (B, 3*n_out_vec)
        return self.encode_vectors(pos).flatten(1)
