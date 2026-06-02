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

from ..geometry.irreps import IrrepBlock

__all__ = ["SE3PointEncoder"]


class SE3PointEncoder(nn.Module):
    r"""SE(3)-equivariant point-cloud encoder. ``(B, N, 3) -> (B, latent_dim)``.

    By default the flattened latent is ``n_out_vec`` stacked 3D vectors (type
    $\ell=1$) in the standard $(x,y,z)$ basis, so ``latent_dim = 3 * n_out_vec``
    and the representation $\rho(R)$ is block-diagonal copies of $R$ — orthogonal,
    hence the JEPA L2 cost is rotation-invariant. Pair with
    :class:`~src.models.structured.VNPredictor` (``dim=3``) for an equivariant
    latent world model.

    **Mixed-type readout (``n_out_scalar > 0``).** Optionally also emit
    ``n_out_scalar`` SO(3)-**invariant** scalars (type $\ell=0$, ``0e``), giving a
    latent with *two* isotypic blocks — scalars (a $G$-invariant block) and vectors
    (a type-1 block). The flattened layout is **scalars first, then vectors**:

    $$ z = [\,\underbrace{s_1,\dots,s_{n_0}}_{\ell=0},\ \underbrace{v_1,\dots,v_{n_1}}_{\ell=1}\,],
       \qquad \rho(R) = \mathbf I_{n_0}\oplus(\mathbf I_{n_1}\otimes R), $$

    so ``latent_dim = n_out_scalar + 3 * n_out_vec``. This is the layout the
    block-SIGReg experiment (task #83) needs: with two *inequivalent* irreps,
    vanilla isotropic SIGReg and block-SIGReg genuinely differ (Prop. 1). The
    isotypic block layout is exposed by :meth:`irrep_blocks`.

    Args:
        n_out_vec: number of $\ell=1$ output vectors.
        lmax: maximum spherical-harmonic degree for the geometric embedding.
        mul: channel multiplicity of each irrep in the hidden layer.
        n_radial: number of Gaussian radial-basis functions on $\lVert r\rVert$.
        r_max: spatial scale of the radial basis (RBF centres span $[0, r_{\max}]$).
        n_out_scalar: number of $\ell=0$ invariant output scalars (default ``0``,
            which reproduces the pure-vector latent byte-for-byte).
    """

    def __init__(
        self,
        n_out_vec: int = 8,
        lmax: int = 2,
        mul: int = 8,
        n_radial: int = 8,
        r_max: float = 2.0,
        n_out_scalar: int = 0,
        pool: str = "sum",
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_out_vec = n_out_vec
        self.n_out_scalar = n_out_scalar
        self.latent_dim = n_out_scalar + 3 * n_out_vec
        self.pool = pool

        # --- irreps ---------------------------------------------------------
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)  # 1x0e+1x1o+...
        irreps_node = o3.Irreps("1x0e")  # scalar seed feature (all ones)
        # hidden carries the irreps producible from node (x) sh: 0e, 1o, 2e, ...
        self.irreps_hidden = o3.Irreps(
            [(mul, (ir.l, ir.p)) for _, ir in self.irreps_sh]
        )
        # scalars first, then vectors (e3nn preserves declared order); for
        # n_out_scalar == 0 this is exactly o3.Irreps(f"{n_out_vec}x1o") as before.
        irreps_out = (
            o3.Irreps(f"{n_out_vec}x1o")
            if n_out_scalar == 0
            else o3.Irreps(f"{n_out_scalar}x0e + {n_out_vec}x1o")
        )

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

        # --- pooling over points -------------------------------------------
        # "sum"  : the original permutation-invariant mean-field aggregate
        #          ``msg.sum(dim=1)`` (byte-for-byte the published encoder).
        # "attn" : multi-head *equivariant* attention pooling — the design-rule
        #          "enrich the aggregator, keep the prior" cure for the lossy
        #          sum-pool (Steps 42-44 localized the residual interaction cap
        #          to this single aggregate). Per-point attention scores are read
        #          off the *invariant* (l=0) channels of ``msg`` (so the weights
        #          are SO(3)-invariant), softmax-normalized over the N points
        #          (so the head is permutation-invariant), and applied as K
        #          distinct weighted sums of the equivariant per-point features
        #          ``msg`` (so each head stays equivariant). An ``o3.Linear``
        #          then recombines the K heads; because the heads enter the
        #          downstream ``NormActivation`` *separately* (not pre-summed),
        #          the pool is strictly richer than a single weighted sum —
        #          which would collapse back to one aggregate. No irrep widening:
        #          the latent is the same fixed-size abstract ``{n_out_vec}x1o``
        #          (it does NOT regress toward the raw-cloud oracle).
        if pool not in ("sum", "attn"):
            raise ValueError(f"unknown pool={pool!r} (expected 'sum' or 'attn')")
        if pool == "attn":
            self.n_heads = n_heads
            self._mul0 = mul  # # of l=0 (invariant) channels in irreps_hidden (first `mul` dims)
            self.attn_score = nn.Sequential(
                nn.Linear(mul, 64), nn.SiLU(), nn.Linear(64, n_heads)
            )
            # K stacked copies of irreps_hidden (matches torch.cat over heads,
            # each head a full irreps_hidden block in its native order) -> hidden.
            irreps_cat = o3.Irreps("+".join([str(self.irreps_hidden)] * n_heads))
            self.attn_combine = o3.Linear(irreps_cat, self.irreps_hidden)

    def _pool(self, msg: torch.Tensor) -> torch.Tensor:
        r"""Aggregate per-point equivariant features ``(B, N, hidden) -> (B, hidden)``.

        Permutation-invariant and SE(3)-equivariant for both modes. ``"sum"`` is the
        mean-field aggregate; ``"attn"`` uses $K$ heads of invariant-scored attention
        (weights from the $\ell=0$ block, $\mathrm{softmax}$ over the $N$ points) and an
        ``o3.Linear`` recombination — strictly richer than a single weighted sum.
        """
        if self.pool == "sum":
            return msg.sum(dim=1)  # (B, hidden) permutation-invariant aggregate
        # "attn": K-head equivariant attention pooling
        inv = msg[..., : self._mul0]  # (B, N, mul) invariant l=0 part -> invariant scores
        w = torch.softmax(self.attn_score(inv), dim=1)  # (B, N, K) softmax over points
        heads = torch.einsum("bnk,bnd->bkd", w, msg)  # (B, K, hidden): K weighted sums
        return self.attn_combine(heads.reshape(heads.shape[0], -1))  # (B, hidden)

    def _encode(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Shared encoder body. ``(B, N, 3) -> (scalars, vectors)``.

        ``scalars`` is ``(B, n_out_scalar)`` (SO(3)-invariant, type ``0e``; an empty
        width-0 tensor when ``n_out_scalar == 0``) and ``vectors`` is
        ``(B, n_out_vec, 3)`` in the standard $(x,y,z)$ basis (type ``1o``). The
        change-of-basis is applied to the vector block only — the scalars are already
        invariant, so they need no rotation.
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
        h = self._pool(msg)  # (B, hidden_dim) permutation-invariant aggregate ("sum" | "attn")

        h = self.act1(h)
        h = self.lin1(h)
        h = self.act2(h)
        out = self.lin_out(h)  # (B, n_out_scalar + 3*n_out_vec); scalars first (0e), then 1o

        n0 = self.n_out_scalar
        scalars = out[:, :n0]  # (B, n_out_scalar) SO(3)-invariant (0e); empty when n0 == 0
        vecs = out[:, n0:].reshape(out.shape[0], self.n_out_vec, 3)  # e3nn (y,z,x) basis
        vecs = vecs @ self.basis_xyz.T  # e3nn 1o basis -> standard (x, y, z)
        return scalars, vecs

    def encode_vectors(self, pos: torch.Tensor) -> torch.Tensor:
        r"""Return the equivariant **vector** latent ``(B, n_out_vec, 3)``.

        Each output row $v$ satisfies $v(R\,x) = R\,v(x)$ (standard $(x,y,z)$
        basis) and $v(x+t)=v(x)$ (translation invariance via centering). Any
        invariant scalar block is dropped here; use :meth:`forward` for the full
        flattened latent ``[scalars, vectors]``. For ``n_out_scalar == 0`` this is
        byte-for-byte the original pure-vector encoder.
        """
        return self._encode(pos)[1]

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # (B, N, 3) -> (B, n_out_scalar + 3*n_out_vec): [scalars (0e) | vectors (1o)].
        # When n_out_scalar == 0, scalars is width-0 and this equals encode_vectors(pos).flatten(1).
        scalars, vecs = self._encode(pos)
        return torch.cat([scalars, vecs.flatten(1)], dim=-1)

    def irrep_blocks(self) -> list[IrrepBlock]:
        r"""Isotypic-block layout of the flattened latent (scalars first, then vectors).

        Returns the contiguous blocks of :meth:`forward`'s output: the invariant
        scalar block (``"0e"``, $d=1$, multiplicity ``n_out_scalar``) when present,
        followed by the vector block (``"1o"``, $d=3$, multiplicity ``n_out_vec``).
        These are the real isotypic components over which **Proposition 1** makes the
        latent covariance block-isotropic, $\Sigma=\bigoplus_i\mathbf I_{d_i}\otimes
        B_i$, and which :func:`src.training.sigreg.sigreg_block` Gaussianises at
        independent (free) per-block scales.
        """
        blocks: list[IrrepBlock] = []
        start = 0
        if self.n_out_scalar > 0:
            blocks.append(IrrepBlock("0e", start, 1, self.n_out_scalar))
            start += self.n_out_scalar
        blocks.append(IrrepBlock("1o", start, 3, self.n_out_vec))
        return blocks
