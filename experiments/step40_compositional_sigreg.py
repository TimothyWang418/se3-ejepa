r"""Step 40: **bi-block-SIGReg** — the *compositional* refinement of block-SIGReg.

Where this sits (Direction 3, task #84), and the one claim it tests
-------------------------------------------------------------------
Step 39 sharpened LeJEPA's isotropic target for a *single* SO(3) object: a
$G$-invariant latent is **block-isotropic** (Prop. 1), so its per-irrep scales are
free and, when distinct, the residual gauge drops from $O(n)$ to the orthogonal
commutant. But a scene has *many* objects, and the relevant symmetry is the
**product group** $S_O\times SO(3)$ — relabel objects *and* rotate the world.
Step 40 is the identifiability/gauge face of that product group (Steps 19/35 are
its dynamics face).

**Our contribution (C2, Prop. 1′ in ``papers/equivariant_lejepa.md`` §10).** Take a
scene of $O$ distinguishable objects, each encoded to $n_0$ invariant scalars
(``0e``) and $n_1$ vectors (``1o``); the scene latent is $\mathbb R^{O}\otimes
\mathbb R^{D_{\mathrm{obj}}}$ and carries the **outer tensor** representation
$P\boxtimes\rho_{SE3}$ of $S_O\times SO(3)$. The permutation module splits
$\mathbb R^{O}=\mathbb 1\oplus\mathbf{std}$ (trivial $\oplus$ standard, $\dim=O-1$),
so by **product-group Schur** the latent decomposes into **four bi-isotypic blocks**

$$\Sigma=\bigoplus_{\lambda\in\{\mathbb 1,\mathbf{std}\}}\ \bigoplus_{\ell\in\{0e,1o\}}
        \mathbf I_{d_\lambda d_\ell}\otimes B_{(\lambda,\ell)},$$

with residual commutant $\prod O(m_{(\lambda,\ell)})$. The two $S_O$-pieces have a
*physical reading*: $(\mathbb 1,\cdot)$ is the **aggregate** (permutation-invariant)
content, $(\mathbf{std},\cdot)$ the **relational** (how objects differ) content.

For the config below ($O=4$, $n_0=n_1=2$, scene $n=32$) the **gauge ladder** is

$$O(32)\,[496]\ \xrightarrow{\text{SE(3) blocks}}\ O(8)\times O(24)\,[304]
   \ \xrightarrow{+\,S_O}\ [184]\ \xrightarrow{\text{known }\rho}\ O(2)^4\,[4].$$

The **$304\!\to\!184$** rung is the *compositional* payoff: SE(3)-block-SIGReg
(Step 39, two SO(3)-isotypic blocks) forces the same scale on the aggregate and
relational parts *within* a type and stops at 304; **bi-block-SIGReg** (this step,
four blocks) frees the $S_O$ split and reaches 184. That rung does **not** exist for
a single object — it is what "many objects" buys.

Two halves, each guarded (mirrors Step 39's discipline):

* **[A] objective-level (deterministic).** On *synthetic* block-isotropic Gaussians,
  bi-block-SIGReg is **flat** across a four-way scale split while SE(3)-block-SIGReg
  **grows** on the within-type $S_O$ split (and vanilla isotropic grows on any split).
  The spectral gauge reads $304$ on a type-only law and $184$ on a bi-type law,
  deterministically. Guarded by a **positive control** (bi-block-SIGReg must *spike*
  on a law outside Prop. 1′ — a spatially-anisotropic vector block) and a
  **gap_factor robustness sweep** (the $304/184$ ladder must hold across thresholds).
* **[B–E] learned (the JEPA payoff).** Train a shared-weight per-object SE(3)
  encoder under faithful **LeJEPA** on a scene law that is invariant under *both*
  global rotation **and** object relabeling, against a non-equivariant slot-MLP:
    [B/A'] equivariance under global SO(3) **and** $S_O$ permutation, init+post-train;
    [C]    Prop. 1′ bi-block isotropy of the *learned* latent (all four cross-block
           couplings $\to 0$; each vector block's $3\times3$ covariance isotropic),
           with a **negative control** — the *same* encoder on a non-$S_O$-invariant
           (fixed-slot) law must *fail* the relational blocks;
    [D]    diagnostic — per-bi-block scales & spectrum (pure SSL leaves the four
           scales underdetermined, exactly as Step 39 §7);
    [E]    **the payoff that closes Step 39's loop** — the compositional task signal.
           [E1] a bi-block-respecting probe for a *relational* target transfers across
           $SO(3)\times S_O$ (flat over rotations AND permutations) while an MLP probe
           degrades; [E2] adding a small relational task head to LeJEPA drives the
           targeted $(\mathbf{std},1o)$ block's scale apart, *realizing* the gauge
           refinement ($304\!\to$ toward $184$) that pure SSL leaves underdetermined.

Honest scope. Synthetic clouds and synthetic probe target (the price of a *provable*
product symmetry at laptop scale); objects do not interact (a direct-sum scene law,
as in Step 19). [A] is the rigorous core; [B–E] show a *learned* latent moves the way
Prop. 1′ predicts. No guard is loosened to force a pass — a run that fails to separate
is reported ``INCONCLUSIVE``.

Run:
    python experiments/step40_compositional_sigreg.py
    # fast smoke: STEP40_SMOKE=1 python experiments/step40_compositional_sigreg.py
"""

import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # for `src.*`
sys.path.insert(0, str(HERE))  # for the Step 39 helpers we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.training.sigreg import (  # noqa: E402
    IrrepBlock,
    block_scales,
    sigreg_block,
    sigreg_isotropic,
    variance_budget_penalty,
)

# Reuse Step 39's *validated* SO(3) helpers and the spectral-gauge reader verbatim —
# nothing geometric is re-invented here (the product-group structure is the only new math).
from step39_block_sigreg import (  # noqa: E402
    _dim_on,
    rand_so3,
    rot_z,
    rotate_points,
    spectral_gauge,
)

torch.set_default_dtype(torch.float32)
SMOKE = bool(os.environ.get("STEP40_SMOKE"))

# --------------------------------------------------------------------------- #
# scene layout: O distinguishable objects, each with n_0 scalars (0e) + n_1
# vectors (1o). Scene latent = O stacked per-object blocks of width D_OBJ.
#   O=4, n_0=2, n_1=2 -> D_OBJ = 2 + 3*2 = 8, scene n = O*D_OBJ = 32.
# The four bi-isotypic blocks (irrep dim d, multiplicity m, eigen-mult d*m):
#   (1,0e)   d=1 m=2  dm=2     (std,0e) d=3 m=2  dm=6
#   (1,1o)   d=3 m=2  dm=6     (std,1o) d=9 m=2  dm=18      (sum dm = 32)
# Gauge ladder: O(32)=496 --SE(3)--> O(8)xO(24)=304 --+S_O--> 184 --known rho--> O(2)^4=4.
# --------------------------------------------------------------------------- #
N_OBJ = 4
N_SCALAR = 2  # n_0 per object (0e)
N_VEC = 2  # n_1 per object (1o)
D_OBJ = N_SCALAR + 3 * N_VEC  # 8
N_SCENE = N_OBJ * D_OBJ  # 32
N_POINTS = 24

# eigen-multiplicities d*m of the four bi-blocks, at distinct scales (for gauge asserts)
_BI_DM = (1 * N_SCALAR, 3 * N_VEC, (N_OBJ - 1) * N_SCALAR, 3 * (N_OBJ - 1) * N_VEC)  # (2,6,6,18)
_SE3_DM = (N_OBJ * N_SCALAR, 3 * N_OBJ * N_VEC)  # (8, 24): the two SO(3)-isotypic blocks
GAUGE_ISO = _dim_on(N_SCENE)  # 496
GAUGE_SE3 = sum(_dim_on(k) for k in _SE3_DM)  # 304
GAUGE_BI = sum(_dim_on(k) for k in _BI_DM)  # 184
GAUGE_COMMUTANT = 2 * _dim_on(N_SCALAR) + 2 * _dim_on(N_VEC)  # prod O(m_i) = O(2)^4 = 4


# --------------------------------------------------------------------------- #
# Helmert change-of-basis: expose the bi-isotypic blocks as CONTIGUOUS columns.
#
# The raw scene latent is object-stacked, [obj0 | obj1 | ... ], so the S_O-trivial
# ("mean over objects") and S_O-standard ("object differences") parts are *mixed*
# across columns. Apply the Helmert orthogonal U on the object axis (row 0 = mean =
# trivial; rows 1..O-1 = an orthonormal basis of 1^perp = standard), then gather by
# (S_O-type, SO(3)-type). The whole map U(x)I_{D_obj} followed by a permutation gather
# is ORTHOGONAL, so spectral_gauge (eigenvalues) is unchanged on the raw latent, while
# sigreg_block / the Prop-1' metrics get contiguous blocks to read.
# --------------------------------------------------------------------------- #
def helmert_matrix(o: int) -> torch.Tensor:
    r"""Orthogonal $U\in\mathbb R^{O\times O}$: row 0 $=\tfrac1{\sqrt O}\mathbf 1$ (trivial),
    rows $1..O-1$ an orthonormal basis of $\mathbf 1^\perp$ (the standard rep)."""
    U = torch.zeros(o, o)
    U[0] = 1.0 / math.sqrt(o)
    for k in range(1, o):
        U[k, :k] = 1.0 / math.sqrt(k * (k + 1))
        U[k, k] = -k / math.sqrt(k * (k + 1))
    return U


def scene_bibasis(
    o: int = N_OBJ, n0: int = N_SCALAR, n1: int = N_VEC
) -> tuple[torch.Tensor, torch.Tensor, list[IrrepBlock], list[IrrepBlock]]:
    r"""Return ``(U, perm, bi_blocks, se3_blocks)`` for the bi-isotypic layout.

    ``U`` is the $O\times O$ Helmert matrix (apply on the object axis). ``perm`` gathers
    the flattened $(m,d)$ transformed latent into the contiguous order
    $[(\mathbb 1,0e),(\mathbb 1,1o),(\mathbf{std},0e),(\mathbf{std},1o)]$. ``bi_blocks``
    are the four :class:`IrrepBlock`s in that layout; ``se3_blocks`` coarsen them to the
    two SO(3)-isotypic blocks (all scalars, all vectors) — the Step-39 baseline, which is
    *also* contiguous here (scalars precede vectors within each $S_O$-type, and we order
    $S_O$-trivial scalars+vectors then standard, so [A,B] = scalars... no: see below).
    """
    dobj = n0 + 3 * n1
    U = helmert_matrix(o)
    scal_d = list(range(n0))  # scalar columns within D_obj
    vec_d = list(range(n0, dobj))  # vector columns within D_obj
    # gather order: A=(1,0e), B=(1,1o), C=(std,0e), D=(std,1o); flat index = m*dobj + d
    idx_A = [0 * dobj + d for d in scal_d]
    idx_B = [0 * dobj + d for d in vec_d]
    idx_C = [m * dobj + d for m in range(1, o) for d in scal_d]
    idx_D = [m * dobj + d for m in range(1, o) for d in vec_d]
    perm = torch.tensor(idx_A + idx_B + idx_C + idx_D, dtype=torch.long)
    wA, wB, wC, wD = len(idx_A), len(idx_B), len(idx_C), len(idx_D)  # 2,6,6,18
    bi_blocks = [
        IrrepBlock("1.0e", 0, 1, n0),  # d=1,        m=n0
        IrrepBlock("1.1o", wA, 3, n1),  # d=3,        m=n1
        IrrepBlock("std.0e", wA + wB, o - 1, n0),  # d=(O-1),    m=n0
        IrrepBlock("std.1o", wA + wB + wC, 3 * (o - 1), n1),  # d=3(O-1), m=n1
    ]
    # se3 coarsening: the contiguous layout above is [A=scal | B=vec | C=scal | D=vec],
    # so scalars are NOT contiguous as one run. Provide a SEPARATE gather for the se3
    # baseline that puts both scalar pieces first: order [A, C, B, D].
    se3_perm = torch.tensor(idx_A + idx_C + idx_B + idx_D, dtype=torch.long)
    se3_blocks = [
        IrrepBlock("scal", 0, 1, o * n0),  # all O*n0 scalar dims
        IrrepBlock("vec", o * n0, 3, o * n1),  # all O*n1 vectors
    ]
    return U, perm, bi_blocks, (se3_perm, se3_blocks)


def to_isobasis(z: torch.Tensor, U: torch.Tensor, perm: torch.Tensor, o: int = N_OBJ) -> torch.Tensor:
    r"""Map the raw object-stacked scene latent ``(N, O*D_obj)`` to a contiguous-block
    layout via $U\otimes I$ (Helmert on the object axis) then a ``perm`` gather. Orthogonal,
    so :func:`spectral_gauge` is unchanged; pass ``bi_perm`` for the four bi-isotypic blocks
    or ``se3_perm`` for the two coarse SO(3)-isotypic blocks (the Step-39 baseline)."""
    n = z.shape[0]
    dobj = z.shape[1] // o
    zt = torch.einsum("nod,mo->nmd", z.reshape(n, o, dobj), U)  # Helmert on object axis
    return zt.reshape(n, o * dobj)[:, perm]


# --------------------------------------------------------------------------- #
# data: a scene law invariant under BOTH global SO(3) (diagonal rotation of every
# object) AND object relabeling S_O (random template->slot assignment). Both are
# required for Prop. 1' to apply to the marginal latent law. Distinct anisotropic
# templates make the objects distinguishable, so the S_O-standard ("relational")
# blocks carry real content.
# --------------------------------------------------------------------------- #
_TEMPLATES = [
    (
        np.random.default_rng(100 + i).standard_normal((N_POINTS, 3)).astype(np.float32)
        * np.array([1.0 + 0.25 * i, 0.7, 0.45 + 0.1 * i], np.float32)  # distinct shapes
    )
    for i in range(N_OBJ)
]


def make_scenes(
    n: int, *, seed: int, orient: str = "haar", permute: bool = True, aniso: float = 0.08
) -> torch.Tensor:
    r"""``n`` scenes ``(n, O, P, 3)``. ``orient='haar'`` rotates every object by ONE
    Haar $R$ (diagonal SO(3) -> rotation-invariant law); ``'wedge'`` uses a z-rotation
    in $[0,90°)$. ``permute=True`` assigns templates to slots by a random permutation
    ($S_O$-invariant law); ``False`` is the fixed-slot **negative control** (breaks
    $S_O$-invariance, so Prop. 1''s relational premise is removed).
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, N_OBJ, N_POINTS, 3), np.float32)
    for i in range(n):
        slots = rng.permutation(N_OBJ) if permute else np.arange(N_OBJ)
        if orient == "haar":
            R = rand_so3(rng_to_torch(rng)).numpy()
        elif orient == "wedge":
            R = rot_z(rng.uniform(0.0, 90.0)).numpy()
        else:
            raise ValueError(orient)
        for o in range(N_OBJ):
            tmpl = _TEMPLATES[slots[o]]
            jitter = rng.standard_normal((N_POINTS, 3)).astype(np.float32) * 0.02
            size = rng.uniform(0.6, 1.6)
            axis_scale = (1.0 + rng.standard_normal(3) * aniso).astype(np.float32)
            cloud = (tmpl + jitter) * size * axis_scale
            out[i, o] = cloud @ R.T  # SAME R for every object (diagonal SO(3))
    return torch.from_numpy(out)


def rng_to_torch(rng: np.random.Generator) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(rng.integers(0, 2**31 - 1)))
    return g


# --------------------------------------------------------------------------- #
# scene encoders, differing ONLY in the SE(3) prior (both are slotted => perm-equiv).
# --------------------------------------------------------------------------- #
class SceneSE3Encoder(nn.Module):
    r"""Shared per-object :class:`SE3PointEncoder` (mixed 0e+1o readout) applied per slot:
    ``(B, O, P, 3) -> (B, O*D_obj)``. Shared weights => $S_O$-permutation-equivariant and
    leakage-free; per object translation-invariant + SO(3)-equivariant. So the scene latent
    carries $P\boxtimes\rho_{SE3}$ of $S_O\times SO(3)$ exactly."""

    def __init__(self, n_obj: int = N_OBJ, n0: int = N_SCALAR, n1: int = N_VEC, lmax: int = 2, mul: int = 8):
        super().__init__()
        self.n_obj = n_obj
        self.obj_enc = SE3PointEncoder(n_out_vec=n1, lmax=lmax, mul=mul, n_out_scalar=n0)
        self.latent_dim = n_obj * self.obj_enc.latent_dim

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        b, o, p, _ = S.shape
        z = self.obj_enc(S.reshape(b * o, p, 3))  # (B*O, D_obj)
        return z.reshape(b, o * self.obj_enc.latent_dim)

    def obj_blocks(self) -> list[IrrepBlock]:
        return self.obj_enc.irrep_blocks()


class SlotMLPEncoder(nn.Module):
    r"""Factorized but **non-rotation-equivariant** control: center each object then a shared
    per-object MLP. ``(B,O,P,3)->(B,O*D_obj)``. Perm-equivariant + translation-invariant (so
    the $S_O$ and aggregate/relational *bookkeeping* is identical), but a rotation has no
    predictable latent action — the isolated control for the SE(3) prior (cf. Step 19)."""

    def __init__(self, n_obj: int = N_OBJ, n_points: int = N_POINTS, d_obj: int = D_OBJ, hidden: int = 128):
        super().__init__()
        self.n_obj, self.d_obj = n_obj, d_obj
        self.latent_dim = n_obj * d_obj
        self.net = nn.Sequential(
            nn.Linear(n_points * 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_obj),
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        b, o, p, _ = S.shape
        Sc = S - S.mean(dim=2, keepdim=True)  # center each object -> translation-invariant
        z = self.net(Sc.reshape(b * o, p * 3))
        return z.reshape(b, o * self.d_obj)

    def obj_blocks(self) -> list[IrrepBlock]:  # nominal layout (no real equivariance)
        return [IrrepBlock("0e", 0, 1, N_SCALAR), IrrepBlock("1o", N_SCALAR, 3, N_VEC)]


def build_scene_eq() -> SceneSE3Encoder:
    return SceneSE3Encoder()


def build_scene_mlp() -> SlotMLPEncoder:
    return SlotMLPEncoder()


# --------------------------------------------------------------------------- #
# equivariant scene targets (for [E]). Per object: y_o = mean_i ||r_i|| r_i (Step 39's
# equivariant target, per slot) -> type-1, equivariant under global R and translation-
# invariant. Aggregate (S_O-invariant, lives in (1,1o)): y_agg = mean_o y_o. Relational
# (S_O-standard, lives in (std,1o)): y_rel_o = y_o - y_agg (sums to 0 over o).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def scene_obj_vec(S: torch.Tensor) -> torch.Tensor:
    r"""Per-object equivariant vector. ``(B, O, P, 3) -> (B, O, 3)``."""
    r = S - S.mean(dim=2, keepdim=True)  # (B,O,P,3) per-object centred
    w = r.norm(dim=-1, keepdim=True)  # (B,O,P,1)
    return (w * r).mean(dim=2)  # (B,O,3)


@torch.no_grad()
def rel_target_std(S: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    r"""Relational target projected onto the Helmert standard modes. ``(B,O,3) -> (B,O-1,3)``.

    $y_{\rm rel}=y-\bar y$ lives in $\mathbf 1^\perp$; its coordinates in the std basis are
    $\tilde y_m=\sum_o U_{m,o}\,y_o$ for $m=1..O-1$. This is exactly what the $(\mathbf{std},1o)$
    latent block should linearly predict — a *relational* (compositional) quantity."""
    y = scene_obj_vec(S)  # (B,O,3)
    return torch.einsum("boc,mo->bmc", y, U[1:])  # (B,O-1,3)


def std_vec_block(z_bi: torch.Tensor, bi_blocks: list[IrrepBlock], o: int = N_OBJ, n1: int = N_VEC) -> torch.Tensor:
    r"""Extract the $(\mathbf{std},1o)$ block from a bibasis latent, shaped as
    ``(B, O-1, n1, 3)`` — $(O-1)$ std modes $\times$ $n_1$ vector copies $\times$ xyz."""
    blk = bi_blocks[3]  # (std,1o)
    return blk.slice(z_bi).reshape(z_bi.shape[0], o - 1, n1, 3)


def rel_task_loss(z_bi: torch.Tensor, y_rel: torch.Tensor, bi_blocks: list[IrrepBlock]) -> torch.Tensor:
    r"""Differentiable relMSE of a **parameter-free, scale-sensitive** equivariant readout from the
    $(\mathbf{std},1o)$ block to the relational target — the [E2] task signal.

    The readout is the channel mean $\hat y_m=\frac1{n_1}\sum_k V_{m,k}$ (equivariant, no free
    weight), matched against the target *normalised to unit RMS* (a constant w.r.t. the network).
    Crucially there is **no free multiplicative weight**: a collapsed block ($V\to0$) cannot be
    rescaled to fit, so driving the loss down forces the $(\mathbf{std},1o)$ block to carry the
    relational signal at $O(1)$ scale, pulling it apart from the (budget-shared) others.

    .. note::
       An earlier design fitted a free ``lstsq`` weight $\hat y_m=\sum_k w_k V_{m,k}$ and reported
       relMSE. That objective is *scale-invariant in the latent* — $w$ absorbs any block scale —
       so it provides **zero** pressure on the block's scale and cannot realise a *scale*-based
       gauge refinement. Removing the free weight (mean readout) is exactly what makes the task
       scale-sensitive; see Step 40 [E2] / §10.
    """
    V = std_vec_block(z_bi, bi_blocks)  # (B,O-1,n1,3)
    pred = V.mean(dim=2)  # (B,O-1,3): parameter-free equivariant readout (NO free rescale)
    rms = y_rel.pow(2).mean().sqrt().clamp_min(1e-6)  # detached data constant -> unit-RMS target
    y = y_rel / rms
    num = ((pred - y) ** 2).sum()
    den = (y**2).sum().clamp_min(1e-12)
    return num / den


# --------------------------------------------------------------------------- #
# faithful LeJEPA training for scenes (jitter views -> mean, no EMA/stop-grad) + a
# SIGReg variant. variant in {none, vanilla, se3, bi}; bi/se3 act on the Helmert-
# transformed latent so their blocks are contiguous. task_gamma>0 adds the [E2]
# relational task signal on the (std,1o) block.
# --------------------------------------------------------------------------- #
def train_lejepa(
    encoder: nn.Module,
    scenes: torch.Tensor,
    *,
    variant: str,
    basis: dict,
    epochs: int,
    batch: int = 64,
    lam: float = 0.08,
    beta: float = 1.0,
    n_views: int = 2,
    jitter: float = 0.02,
    n_proj: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
    task_gamma: float = 0.0,
) -> dict[str, float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    proj_gen = torch.Generator().manual_seed(seed + 100)
    aug_gen = torch.Generator().manual_seed(seed + 200)
    U, bi_perm, bi_blocks = basis["U"], basis["bi_perm"], basis["bi_blocks"]
    se3_perm, se3_blocks = basis["se3_perm"], basis["se3_blocks"]
    m = scenes.shape[0]
    last = {"pred": float("nan"), "reg": float("nan"), "std": float("nan"), "task": float("nan")}
    for ep in range(epochs):
        perm = torch.randperm(m, generator=torch.Generator().manual_seed(seed + ep))
        for s in range(0, m, batch):
            idx = perm[s : s + batch]
            X = scenes[idx]  # (b,O,P,3)
            views = [X + jitter * torch.randn(X.shape, generator=aug_gen) for _ in range(n_views)]
            Z = torch.stack([encoder(v) for v in views], dim=0)  # (V,b,n)
            mu = Z.mean(dim=0)  # (b,n) grad through mu (no stop-grad)
            pred = ((Z - mu[None]) ** 2).sum(-1).mean()
            zpool = Z.reshape(-1, Z.shape[-1])  # (V*b, n)
            if variant == "none":
                reg = zpool.new_zeros(())
            elif variant == "vanilla":
                reg = sigreg_isotropic(zpool, n_proj=n_proj, generator=proj_gen)
            elif variant == "se3":
                reg = sigreg_block(to_isobasis(zpool, U, se3_perm), se3_blocks, n_proj=n_proj, generator=proj_gen) \
                    + beta * variance_budget_penalty(zpool)
            elif variant == "bi":
                reg = sigreg_block(to_isobasis(zpool, U, bi_perm), bi_blocks, n_proj=n_proj, generator=proj_gen) \
                    + beta * variance_budget_penalty(zpool)
            else:
                raise ValueError(variant)
            task = zpool.new_zeros(())
            if task_gamma > 0.0:
                y_rel = rel_target_std(X, U)  # (b,O-1,3)
                task = rel_task_loss(to_isobasis(mu, U, bi_perm), y_rel, bi_blocks)
            loss = (1.0 - lam) * pred + lam * reg + task_gamma * task
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {
                "pred": float(pred.detach()), "reg": float(reg.detach()),
                "std": float(zpool.detach().std()), "task": float(task.detach()),
            }
    return last


# --------------------------------------------------------------------------- #
# equivariance probes: global SO(3) (scalar-inv + vector-equiv), S_O permutation,
# translation. The scene latent is O stacked [scalars | vectors] per object.
# --------------------------------------------------------------------------- #
def _split_scene(z: torch.Tensor, o: int = N_OBJ, n0: int = N_SCALAR, n1: int = N_VEC):
    r"""``(B, O*D_obj) -> (scalars (B,O,n0), vectors (B,O,n1,3))``."""
    zb = z.reshape(z.shape[0], o, n0 + 3 * n1)
    scal = zb[:, :, :n0]
    vecs = zb[:, :, n0:].reshape(z.shape[0], o, n1, 3)
    return scal, vecs


@torch.no_grad()
def equiv_errs_scene(enc: nn.Module, S: torch.Tensor, R: torch.Tensor) -> tuple[float, float]:
    r"""(scalar-invariance, vector-equivariance) under ONE global rotation $R$ of every object."""
    s0, v0 = _split_scene(enc(S))
    s1, v1 = _split_scene(enc(rotate_points(S, R)))
    inv = (s1 - s0).abs().max().item() if s0.numel() else 0.0
    equ = (v1 - v0 @ R.T).abs().max().item()
    return inv, equ


@torch.no_grad()
def perm_err_scene(enc: nn.Module, S: torch.Tensor, sigma: torch.Tensor, o: int = N_OBJ, d_obj: int = D_OBJ) -> float:
    r"""$\max\lVert\sigma\!\cdot\!E(S)-E(\sigma\!\cdot\!S)\rVert_\infty$ — $S_O$-permutation equivariance.

    Permuting object slots in the input must permute the latent blocks identically (exact for a
    shared-weight slotted encoder)."""
    zperm = enc(S)[:, :].reshape(S.shape[0], o, d_obj)[:, sigma].reshape(S.shape[0], -1)
    return (zperm - enc(S[:, sigma])).abs().max().item()


@torch.no_grad()
def trans_inv_err_scene(enc: nn.Module, S: torch.Tensor, gen: torch.Generator) -> float:
    t = torch.randn(1, 1, 1, 3, generator=gen)
    return (enc(S + t) - enc(S)).abs().max().item()


# --------------------------------------------------------------------------- #
# Proposition 1' witnesses on a learned latent: all six cross-block couplings -> 0,
# and each 1o block's per-copy 3x3 covariance isotropic.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prop1_bi_metrics(z_bi: torch.Tensor, bi_blocks: list[IrrepBlock]) -> dict[str, float]:
    r"""``cross``: max over the 6 block-pairs of $\lVert\Sigma_{ij}\rVert_F/\sqrt{\lVert\Sigma_{ii}\rVert
    \lVert\Sigma_{jj}\rVert}$ (Prop. 1' -> 0). ``iso_agg``/``iso_rel``: mean $\lambda_{\max}/\lambda_{\min}$
    of the per-copy $3\times3$ covariances in the $(\mathbb 1,1o)$ and $(\mathbf{std},1o)$ blocks
    (Prop. 1' -> 1)."""
    zc = z_bi - z_bi.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (zc.shape[0] - 1)
    diag = [cov[b.start:b.stop, b.start:b.stop].norm().item() for b in bi_blocks]
    cross = 0.0
    for i in range(len(bi_blocks)):
        for j in range(i + 1, len(bi_blocks)):
            bi_, bj = bi_blocks[i], bi_blocks[j]
            c = cov[bi_.start:bi_.stop, bj.start:bj.stop].norm().item()
            cross = max(cross, c / (math.sqrt(diag[i] * diag[j]) + 1e-12))

    def _iso(blk: IrrepBlock) -> float:
        V = blk.slice(zc).reshape(zc.shape[0], -1, 3)  # (N, n_copies, 3)
        rs = []
        for a in range(V.shape[1]):
            va = V[:, a, :]
            ev = torch.linalg.eigvalsh((va.T @ va) / (va.shape[0] - 1)).clamp_min(1e-12)
            rs.append((ev[-1] / ev[0]).item())
        return float(np.mean(rs))

    return {"cross": cross, "iso_agg": _iso(bi_blocks[1]), "iso_rel": _iso(bi_blocks[3])}


# --------------------------------------------------------------------------- #
# [A] synthetic block-isotropic laws in the bibasis (contiguous blocks A|B|C|D),
# with controlled per-block scales. Total budget fixed: mean per-dim variance = 1.
# --------------------------------------------------------------------------- #
def _scaled_blocks(n: int, scales: tuple[float, ...], gen: torch.Generator) -> torch.Tensor:
    r"""iid Gaussian per bi-block at the given per-dim std, renormalised to total mean var 1."""
    dm = _BI_DM  # (2,6,6,18)
    var = torch.tensor(scales) ** 2
    tot = float((var * torch.tensor(dm, dtype=torch.float32)).sum() / sum(dm))
    s = torch.tensor(scales) / math.sqrt(max(tot, 1e-12))  # renormalise -> mean per-dim var 1
    cols = [torch.randn(n, dm[i], generator=gen) * s[i] for i in range(4)]
    return torch.cat(cols, dim=1)


def synth_biblock_iso(n: int, scales: tuple[float, float, float, float], gen: torch.Generator) -> torch.Tensor:
    r"""Valid Prop.-1' law with the four bi-block scales (relative). ``scales=(a,b,c,d)`` for
    blocks $(\mathbb 1,0e),(\mathbb 1,1o),(\mathbf{std},0e),(\mathbf{std},1o)$."""
    return _scaled_blocks(n, scales, gen)


def synth_spatial_aniso(n: int, spatial_ratio: float, gen: torch.Generator) -> torch.Tensor:
    r"""Positive control: equal block scales, but the $(\mathbf{std},1o)$ block's per-copy
    $3\times3$ is spatially **anisotropic** ($\propto\mathrm{diag}(r,1,1/r)$, mean 1) — a law
    *outside* Prop. 1' (breaks within-block spatial isotropy at the same total budget). Bi-block-
    SIGReg must spike on it (anti-vacuity)."""
    z = _scaled_blocks(n, (1.0, 1.0, 1.0, 1.0), gen)
    start = _BI_DM[0] + _BI_DM[1] + _BI_DM[2]  # start of (std,1o)
    g = torch.tensor([spatial_ratio, 1.0, 1.0 / spatial_ratio])
    g = (g / g.mean()).sqrt()
    d = z[:, start:].reshape(n, -1, 3) * g[None, None, :]
    z = z.clone()
    z[:, start:] = d.reshape(n, -1)
    return z


# --------------------------------------------------------------------------- #
# [E1] compositional 举一反三: a bi-block-respecting probe for the RELATIONAL target,
# fitted on a seen slice and tested across SO(3) (rotations) AND S_O (permutations).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rel_probe_rel_mse(
    enc: nn.Module, basis: dict, X_fit: torch.Tensor, X_eval: torch.Tensor, *, equivariant: bool
) -> float:
    r"""Fit a relational probe on ``X_fit`` -> relMSE on ``X_eval`` (target ``rel_target_std``).

    ``equivariant=True``: a type-respecting probe on the $(\mathbf{std},1o)$ latent block
    ($\hat y_m=\sum_k w_k V_{m,k}$, one weight per vector copy) — itself $SO(3)\times S_O$-
    equivariant, so its relMSE is identical under any rotation or relabeling. ``False``: a plain
    affine probe $Wz+b$ on the whole raw latent (the MLP control), with no such guarantee."""
    U, bi_perm, bi_blocks = basis["U"], basis["bi_perm"], basis["bi_blocks"]
    Yf = rel_target_std(X_fit, U)  # (Nf,O-1,3)
    if equivariant:
        Vf = std_vec_block(to_isobasis(enc(X_fit), U, bi_perm), bi_blocks)  # (Nf,O-1,n1,3)
        A = Vf.permute(0, 1, 3, 2).reshape(-1, Vf.shape[2])  # (Nf*(O-1)*3, n1)
        w = torch.linalg.lstsq(A, Yf.reshape(-1, 1)).solution  # (n1,1)

        def predict(Xe):
            Ve = std_vec_block(to_isobasis(enc(Xe), U, bi_perm), bi_blocks)  # (Ne,O-1,n1,3)
            return torch.einsum("nmkc,k->nmc", Ve, w.squeeze(1))  # (Ne,O-1,3)
    else:
        Zf = enc(X_fit)
        Aug = torch.cat([Zf, torch.ones(Zf.shape[0], 1)], dim=1)  # affine
        W = torch.linalg.lstsq(Aug, Yf.reshape(Yf.shape[0], -1)).solution  # (n+1, (O-1)*3)

        def predict(Xe):
            Ze = enc(Xe)
            out = torch.cat([Ze, torch.ones(Ze.shape[0], 1)], dim=1) @ W
            return out.reshape(Xe.shape[0], N_OBJ - 1, 3)

    Ye = rel_target_std(X_eval, U)
    num = ((predict(X_eval) - Ye) ** 2).sum()
    den = (Ye**2).sum().clamp_min(1e-12)
    return float(num / den)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _make_basis() -> dict:
    U, bi_perm, bi_blocks, (se3_perm, se3_blocks) = scene_bibasis()
    # synth laws are built in bibasis order [A|B|C|D]; gather to [A|C|B|D] for the se3 baseline.
    wA, wB, wC, _ = (b.width for b in bi_blocks)
    bi_to_se3 = torch.cat([
        torch.arange(0, wA), torch.arange(wA + wB, wA + wB + wC),
        torch.arange(wA, wA + wB), torch.arange(wA + wB + wC, N_SCENE),
    ])
    return {"U": U, "bi_perm": bi_perm, "bi_blocks": bi_blocks, "se3_perm": se3_perm,
            "se3_blocks": se3_blocks, "bi_to_se3": bi_to_se3}


def main() -> None:
    torch.manual_seed(0)
    line = "=" * 80
    basis = _make_basis()
    U, bi_perm, bi_blocks = basis["U"], basis["bi_perm"], basis["bi_blocks"]
    se3_blocks, bi_to_se3 = basis["se3_blocks"], basis["bi_to_se3"]

    N_TRAIN = 128 if SMOKE else 1500
    N_EVAL = 128 if SMOKE else 400
    EPOCHS = 4 if SMOKE else 40
    N_SYNTH = 4000 if SMOKE else 20000
    N_COV = 1024 if SMOKE else 6144

    print(line)
    print(f"STEP 40  compositional bi-block-SIGReg  (S_O x SO(3))  ({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    scene: O={N_OBJ} distinguishable objects, n_0={N_SCALAR} scalars + n_1={N_VEC} vectors each;"
          f" latent n={N_SCENE}")
    print(f"    4 bi-isotypic blocks (dm): (1,0e)={_BI_DM[0]} (1,1o)={_BI_DM[1]} "
          f"(std,0e)={_BI_DM[2]} (std,1o)={_BI_DM[3]}")
    print(f"    gauge ladder: O(32)={GAUGE_ISO} --SE(3)--> O(8)xO(24)={GAUGE_SE3} "
          f"--+S_O--> {GAUGE_BI} --known rho--> O(2)^4={GAUGE_COMMUTANT}")

    # ----------------------------------------------------------------- [A]
    print()
    print(line)
    print("[A] OBJECTIVE-LEVEL (deterministic): bi-block-SIGReg flat on ALL block-isotropic")
    print("    laws; se3-block-SIGReg (Step 39) GROWS on the within-type S_O split it can't see.")
    print(line)
    laws = {
        "iso (1,1,1,1)": (1.0, 1.0, 1.0, 1.0),
        "se3-type (1,4,1,4)": (1.0, 4.0, 1.0, 4.0),  # scalar!=vector, trivial==std within type
        "bi-type (.5,1,2,4)": (0.5, 1.0, 2.0, 4.0),  # all four distinct (compositional split)
    }
    print(f"    {'law':22s} | {'vanilla':>9s} | {'se3-block':>9s} | {'bi-block':>9s}")
    print("    " + "-" * 60)
    A = {}
    for nm, sc in laws.items():
        Z = synth_biblock_iso(N_SYNTH, sc, torch.Generator().manual_seed(7))
        van = float(sigreg_isotropic(Z, n_proj=128, generator=torch.Generator().manual_seed(11)))
        se3 = float(sigreg_block(Z[:, bi_to_se3], se3_blocks, n_proj=128, generator=torch.Generator().manual_seed(11)))
        bi = float(sigreg_block(Z, bi_blocks, n_proj=128, generator=torch.Generator().manual_seed(11)))
        A[nm] = {"van": van, "se3": se3, "bi": bi}
        print(f"    {nm:22s} | {van:9.5f} | {se3:9.5f} | {bi:9.5f}")
    van_growth = A["bi-type (.5,1,2,4)"]["van"] / max(A["iso (1,1,1,1)"]["van"], 1e-9)
    se3_split = A["bi-type (.5,1,2,4)"]["se3"] / max(A["se3-type (1,4,1,4)"]["se3"], 1e-9)
    bi_growth = A["bi-type (.5,1,2,4)"]["bi"] / max(A["iso (1,1,1,1)"]["bi"], 1e-9)
    print(f"    => vanilla grows x{van_growth:.0f} on the bi-type split; se3-block grows x{se3_split:.0f}")
    print(f"       on the WITHIN-TYPE S_O split (it cannot represent it); bi-block flat (x{bi_growth:.2f}).")

    # positive control (anti-vacuity): bi-block must SPIKE on a law OUTSIDE Prop. 1'.
    Z_an = synth_spatial_aniso(N_SYNTH, 4.0, torch.Generator().manual_seed(7))
    bi_an = float(sigreg_block(Z_an, bi_blocks, n_proj=128, generator=torch.Generator().manual_seed(11)))
    an_spike = bi_an / max(A["bi-type (.5,1,2,4)"]["bi"], 1e-12)
    print(f"    positive control — spatially-anisotropic (std,1o) block (cov!∝I_3, OUTSIDE Prop.1'): "
          f"bi-block={bi_an:.5f} (x{an_spike:.0f} the valid-law floor)")
    ok_block_discriminates = bi_an > 1e-3 and bi_an > 5.0 * A["bi-type (.5,1,2,4)"]["bi"]

    # gauge ladder, deterministically: se3-type law -> 304 (two SO(3) blocks); bi-type -> 184.
    # Use a WELL-SEPARATED geometric law (per-dim var ratios = 9 between adjacent bi-blocks) so
    # the clustering is unambiguous for any gap_factor inside the separating window (1, 9): the
    # robustness sweep {1.5,2,3,4} then sits comfortably interior, not on a knife-edge boundary.
    Z_se3law = synth_biblock_iso(N_SYNTH, (1.0, 4.0, 1.0, 4.0), torch.Generator().manual_seed(21))
    Z_bilaw = synth_biblock_iso(N_SYNTH, (1.0, 3.0, 9.0, 27.0), torch.Generator().manual_seed(21))
    g_se3 = spectral_gauge(Z_se3law)
    g_bi = spectral_gauge(Z_bilaw)
    print(f"    spectral gauge: se3-type law -> {g_se3['gauge_dim']} (clusters {g_se3['cluster_sizes']}); "
          f"bi-type law -> {g_bi['gauge_dim']} (clusters {g_bi['cluster_sizes']}).")
    print(f"    => the compositional rung: SE(3)-block stops at {GAUGE_SE3}; the S_O split reaches {GAUGE_BI}.")
    gap_grid = (1.5, 2.0, 3.0, 4.0)
    sweep = [(gf, spectral_gauge(Z_se3law, gap_factor=gf)["gauge_dim"],
              spectral_gauge(Z_bilaw, gap_factor=gf)["gauge_dim"]) for gf in gap_grid]
    print("    gauge-ladder robustness vs gap_factor (window (1,9)): "
          + "  ".join(f"{gf:g}->{e}/{d}" for gf, e, d in sweep) + f"  (target {GAUGE_SE3}/{GAUGE_BI})")
    ok_gauge_synth = g_se3["gauge_dim"] == GAUGE_SE3 and g_bi["gauge_dim"] == GAUGE_BI
    ok_gauge_robust = all(e == GAUGE_SE3 and d == GAUGE_BI for _, e, d in sweep)

    # ----------------------------------------------------------------- data + models
    train_haar = make_scenes(N_TRAIN, seed=0, orient="haar", permute=True)
    eval_haar = make_scenes(N_EVAL, seed=999, orient="haar", permute=True)
    cov_haar = make_scenes(N_COV, seed=777, orient="haar", permute=True)
    g = torch.Generator().manual_seed(1)
    R_test = rand_so3(torch.Generator().manual_seed(3))
    sigma = torch.randperm(N_OBJ, generator=torch.Generator().manual_seed(5))

    eq = {v: build_scene_eq() for v in ("none", "vanilla", "se3", "bi")}
    mlp = build_scene_mlp()

    # ----------------------------------------------------------------- [B] init equiv
    print()
    print(line)
    print("[B] EQUIVARIANCE OF THE SCENE ENCODER — AT INIT (global SO(3) + S_O permutation)")
    print(line)
    inv0, equ0 = equiv_errs_scene(eq["bi"], eval_haar, R_test)
    pe0 = perm_err_scene(eq["bi"], eval_haar, sigma)
    minv0, mequ0 = equiv_errs_scene(mlp, eval_haar, R_test)
    mpe0 = perm_err_scene(mlp, eval_haar, sigma)
    print(f"    eq  : scalar-inv={inv0:.3e}  vector-equiv={equ0:.3e}  perm={pe0:.3e}  "
          f"trans-inv={trans_inv_err_scene(eq['bi'], eval_haar, g):.3e}")
    print(f"    mlp : scalar-inv={minv0:.3e}  vector-equiv={mequ0:.3e}  perm={mpe0:.3e}  (no rotation prior)")

    # ----------------------------------------------------------------- train
    print()
    print(line)
    print(f"[train] faithful LeJEPA (jitter views -> mean, no EMA/stop-grad) — {EPOCHS} epochs")
    print(line)
    for v in ("none", "vanilla", "se3", "bi"):
        h = train_lejepa(eq[v], train_haar, variant=v, basis=basis, epochs=EPOCHS, seed=0)
        print(f"    eq/{v:8s}: pred={h['pred']:.4f}  reg={h['reg']:.4f}  latent_std={h['std']:.3f}")
    hm = train_lejepa(mlp, train_haar, variant="bi", basis=basis, epochs=EPOCHS, seed=0)
    print(f"    mlp/bi    : pred={hm['pred']:.4f}  reg={hm['reg']:.4f}  latent_std={hm['std']:.3f}")

    # ----------------------------------------------------------------- [A'] post equiv
    print()
    print(line)
    print("[A'] EQUIVARIANCE AFTER TRAINING (must survive optimisation)")
    print(line)
    inv1, equ1 = equiv_errs_scene(eq["bi"], eval_haar, R_test)
    pe1 = perm_err_scene(eq["bi"], eval_haar, sigma)
    minv1, mequ1 = equiv_errs_scene(mlp, eval_haar, R_test)
    print(f"    eq/bi : scalar-inv={inv1:.3e}  vector-equiv={equ1:.3e}  perm={pe1:.3e}  (still exact)")
    print(f"    mlp   : scalar-inv={minv1:.3e}  vector-equiv={mequ1:.3e}")

    # ----------------------------------------------------------------- [C] Prop 1'
    print()
    print(line)
    print(f"[C] PROPOSITION 1' on the LEARNED latent (N={N_COV}): 4 bi-blocks decouple (cross->0),")
    print("    each 1o block 3x3-isotropic (->1). Neg control: same encoder on a non-S_O-invariant law.")
    print(line)
    with torch.no_grad():
        p_eq = prop1_bi_metrics(to_isobasis(eq["bi"](cov_haar), U, bi_perm), bi_blocks)
        p_mlp = prop1_bi_metrics(to_isobasis(mlp(cov_haar), U, bi_perm), bi_blocks)
        # negative control: fixed-slot law (permute=False) is NOT S_O-invariant -> Prop. 1''s
        # relational premise removed -> the (std,*) blocks must fail to decouple / isotropise.
        fixed = make_scenes(N_COV, seed=777, orient="haar", permute=False)
        p_fix = prop1_bi_metrics(to_isobasis(eq["bi"](fixed), U, bi_perm), bi_blocks)
    print(f"    {'model / law':22s} | {'cross':>8s} | {'iso_agg':>8s} | {'iso_rel':>8s}")
    print("    " + "-" * 56)
    print(f"    {'eq/bi   HAAR+perm':22s} | {p_eq['cross']:8.4f} | {p_eq['iso_agg']:8.3f} | {p_eq['iso_rel']:8.3f}")
    print(f"    {'mlp/bi  HAAR+perm':22s} | {p_mlp['cross']:8.4f} | {p_mlp['iso_agg']:8.3f} | {p_mlp['iso_rel']:8.3f}")
    print(f"    {'eq/bi   FIXED-SLOT':22s} | {p_fix['cross']:8.4f} | {p_fix['iso_agg']:8.3f} | {p_fix['iso_rel']:8.3f}"
          f"  <- neg-control: S_O premise removed")
    ok_prop1 = p_eq["cross"] < 0.18 and p_eq["iso_agg"] < 1.5 and p_eq["iso_rel"] < 1.5 and (
        p_mlp["cross"] > 0.2 or p_mlp["iso_agg"] > 2.0 or p_mlp["iso_rel"] > 2.0
    )
    ok_prop1_negctrl = p_fix["cross"] > 0.3 or p_fix["iso_rel"] > 3.0

    # ----------------------------------------------------------------- [D] diagnostic
    print()
    print(line)
    print("[D] DIAGNOSTIC (not gated) — per-bi-block scales & spectrum of the LEARNED latent")
    print(line)
    print(f"    {'variant':8s} | {'(1,0e)':>8s} | {'(1,1o)':>8s} | {'(std,0e)':>8s} | {'(std,1o)':>8s} | "
          f"{'clusters':>16s} | {'gauge':>5s}")
    print("    " + "-" * 78)
    diag = {}
    for v in ("none", "vanilla", "se3", "bi"):
        with torch.no_grad():
            zt = to_isobasis(eq[v](cov_haar), U, bi_perm)
            zr = eq[v](cov_haar)
        sc = block_scales(zt, bi_blocks)
        sg = spectral_gauge(zr)
        diag[v] = {"scales": sc, "gauge": sg["gauge_dim"], "sizes": sg["cluster_sizes"]}
        print(f"    eq/{v:5s} | {sc['1.0e']:8.4f} | {sc['1.1o']:8.4f} | {sc['std.0e']:8.4f} | "
              f"{sc['std.1o']:8.4f} | {str(sg['cluster_sizes']):>16s} | {sg['gauge_dim']:5d}")
    print("    NOTE: in *pure* SSL the four per-block scales are UNDERDETERMINED (Step 39 §7): bi-block-")
    print("    SIGReg is scale-detached, the budget pins only the total, the pull-to-mean is block-")
    print("    symmetric. The gauge claim is gated DETERMINISTICALLY in [A]; [E2] shows a TASK signal")
    print("    (which a compositional scene supplies naturally) realises the refinement on a learned net.")

    # ----------------------------------------------------------------- [E1] transfer
    print()
    print(line)
    print("[E1] 举一反三 — relational probe fit on a seen slice, tested across SO(3) AND S_O")
    print(line)
    X_fit = make_scenes(N_EVAL, seed=2026, orient="wedge", permute=False)  # seen: z-wedge, fixed slots
    X_seen = make_scenes(N_EVAL, seed=4040, orient="wedge", permute=False)
    X_rot = make_scenes(N_EVAL, seed=5050, orient="haar", permute=False)  # OOD axis 1: full SO(3)
    X_perm = make_scenes(N_EVAL, seed=6060, orient="wedge", permute=True)  # OOD axis 2: relabeling
    eq_seen = rel_probe_rel_mse(eq["bi"], basis, X_fit, X_seen, equivariant=True)
    eq_rot = rel_probe_rel_mse(eq["bi"], basis, X_fit, X_rot, equivariant=True)
    eq_perm = rel_probe_rel_mse(eq["bi"], basis, X_fit, X_perm, equivariant=True)
    mlp_seen = rel_probe_rel_mse(mlp, basis, X_fit, X_seen, equivariant=False)
    mlp_rot = rel_probe_rel_mse(mlp, basis, X_fit, X_rot, equivariant=False)
    mlp_perm = rel_probe_rel_mse(mlp, basis, X_fit, X_perm, equivariant=False)
    print(f"    {'model':8s} | {'seen':>10s} | {'rot-OOD':>10s} | {'perm-OOD':>10s} | {'rot/seen':>9s} | {'perm/seen':>9s}")
    print("    " + "-" * 70)
    print(f"    {'eq/bi':8s} | {eq_seen:10.3e} | {eq_rot:10.3e} | {eq_perm:10.3e} | "
          f"x{eq_rot / max(eq_seen, 1e-12):7.2f} | x{eq_perm / max(eq_seen, 1e-12):7.2f}")
    print(f"    {'mlp':8s} | {mlp_seen:10.3e} | {mlp_rot:10.3e} | {mlp_perm:10.3e} | "
          f"x{mlp_rot / max(mlp_seen, 1e-12):7.2f} | x{mlp_perm / max(mlp_seen, 1e-12):7.2f}")
    eq_rot_r, eq_perm_r = eq_rot / max(eq_seen, 1e-12), eq_perm / max(eq_seen, 1e-12)
    mlp_rot_r = mlp_rot / max(mlp_seen, 1e-12)
    ok_jyfs = eq_rot_r < 1.3 and eq_perm_r < 1.3 and mlp_rot_r > 1.5

    # ----------------------------------------------------------------- [E2] task realises gauge
    print()
    print(line)
    print("[E2] DIAGNOSTIC (not gated) — can a SCALE-SENSITIVE relational task (natural to a compositional")
    print("     scene) pin the (std,1o) block apart, realising on a learned net the rung [A] proves reachable?")
    print(line)
    eq_ssl = build_scene_eq()
    eq_task = build_scene_eq()
    train_lejepa(eq_ssl, train_haar, variant="bi", basis=basis, epochs=EPOCHS, seed=0, task_gamma=0.0)
    train_lejepa(eq_task, train_haar, variant="bi", basis=basis, epochs=EPOCHS, seed=0, task_gamma=0.5)
    with torch.no_grad():
        sc_ssl = block_scales(to_isobasis(eq_ssl(cov_haar), U, bi_perm), bi_blocks)
        sc_task = block_scales(to_isobasis(eq_task(cov_haar), U, bi_perm), bi_blocks)
        gss = spectral_gauge(eq_ssl(cov_haar))
        gtt = spectral_gauge(eq_task(cov_haar))
    rel_ratio_ssl = sc_ssl["std.1o"] / max(np.mean([sc_ssl["1.0e"], sc_ssl["1.1o"], sc_ssl["std.0e"]]), 1e-12)
    rel_ratio_task = sc_task["std.1o"] / max(np.mean([sc_task["1.0e"], sc_task["1.1o"], sc_task["std.0e"]]), 1e-12)
    print(f"    {'encoder':10s} | {'(std,1o) scale':>14s} | {'others mean':>12s} | {'rel/others':>10s} | {'gauge':>6s} | {'clusters'}")
    print("    " + "-" * 78)
    print(f"    {'pure-SSL':10s} | {sc_ssl['std.1o']:14.4f} | "
          f"{np.mean([sc_ssl['1.0e'], sc_ssl['1.1o'], sc_ssl['std.0e']]):12.4f} | "
          f"x{rel_ratio_ssl:9.2f} | {gss['gauge_dim']:6d} | {gss['cluster_sizes']}")
    print(f"    {'task-aug':10s} | {sc_task['std.1o']:14.4f} | "
          f"{np.mean([sc_task['1.0e'], sc_task['1.1o'], sc_task['std.0e']]):12.4f} | "
          f"x{rel_ratio_task:9.2f} | {gtt['gauge_dim']:6d} | {gtt['cluster_sizes']}")
    e2_realises = rel_ratio_task > 1.5 * rel_ratio_ssl and gtt["gauge_dim"] < gss["gauge_dim"]
    arrow = "drops" if gtt["gauge_dim"] < gss["gauge_dim"] else "does NOT drop"
    print(f"    => task vs pure-SSL: (std,1o) rel-scale {rel_ratio_ssl:.2f}->{rel_ratio_task:.2f}; "
          f"residual gauge {arrow} ({gss['gauge_dim']}->{gtt['gauge_dim']}; target rung {GAUGE_BI}).")
    print("    NOTE: [E2] is a DIAGNOSTIC, not gated. The gauge refinement is a property of the")
    print("    TARGET CLASS, proven DETERMINISTICALLY in [A]; on a learned net the per-block split")
    print("    is underdetermined ([D]) and only a SCALE-SENSITIVE task can pin it. (A scale-INVARIANT")
    print("    relMSE fit — free lstsq weight — gives zero scale pressure and CANNOT realise it; the")
    print("    parameter-free readout here is the principled fix. Whether it reaches the rung at this")
    print(f"    1-GPU scale is reported honestly above, not asserted.) realised this run: {e2_realises}")

    # ----------------------------------------------------------------- summary + guards
    print()
    print(line)
    print("STEP 40 SUMMARY")
    print(line)
    ok_obj = van_growth > 3.0 and bi_growth < 1.5
    ok_compositional = se3_split > 5.0 and A["bi-type (.5,1,2,4)"]["se3"] > 1e-3 and bi_growth < 1.5
    ok_equiv = max(inv1, equ1, pe1) < 1e-4 and mequ1 > 1e-2
    # NOTE: [E2] is a DIAGNOSTIC (like [D]), NOT a gate. The gauge refinement is gated
    # DETERMINISTICALLY in [A] (the target class); on a learned net the split is underdetermined
    # (Step 39 §7, reproduced in [D]), so realising it via a task is reported honestly, not gated.
    passed = (
        ok_obj and ok_compositional and ok_gauge_synth and ok_gauge_robust
        and ok_block_discriminates and ok_equiv and ok_prop1 and ok_prop1_negctrl and ok_jyfs
    )
    print(f"    [A]  vanilla penalises distinct-scale laws (x{van_growth:.0f}), bi-block flat (x{bi_growth:.2f}): {ok_obj}")
    print(f"    [A]  COMPOSITIONAL: se3-block grows x{se3_split:.0f} on the within-type S_O split, bi-block flat: {ok_compositional}")
    print(f"    [A]  bi-block NOT vacuous: spikes x{an_spike:.0f} on a non-Prop.1' (anisotropic) law: {ok_block_discriminates}")
    print(f"    [A]  deterministic gauge ladder se3-type->{g_se3['gauge_dim']} / bi-type->{g_bi['gauge_dim']} ({GAUGE_SE3} vs {GAUGE_BI}): {ok_gauge_synth}")
    print(f"    [A]  gauge ladder stable across gap_factor in {{1.5,2,3,4}}: {ok_gauge_robust}")
    print(f"    [A'] eq stays equivariant (SO(3)+S_O, {max(inv1, equ1, pe1):.1e}), MLP not: {ok_equiv}")
    print(f"    [C]  learned latent bi-block-isotropic (eq cross={p_eq['cross']:.3f} iso_rel={p_eq['iso_rel']:.2f}): {ok_prop1}")
    print(f"    [C]  neg-control: same eq encoder FAILS on non-S_O-invariant law (cross={p_fix['cross']:.2f} iso_rel={p_fix['iso_rel']:.1f}): {ok_prop1_negctrl}")
    print(f"    [E1] eq probe flat across SO(3) (x{eq_rot_r:.2f}) AND S_O (x{eq_perm_r:.2f}); MLP degrades (x{mlp_rot_r:.2f}): {ok_jyfs}")
    print(f"    [E2] (DIAGNOSTIC, not gated) scale-sensitive task realises the rung this run: {e2_realises} (gauge {gss['gauge_dim']}->{gtt['gauge_dim']})")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: a scene's product symmetry S_O x SO(3) makes the latent")
    print("    decompose into FOUR bi-isotypic blocks (Prop. 1'); bi-block-SIGReg targets that full class,")
    print(f"    reaching the compositional gauge rung ({GAUGE_SE3}->{GAUGE_BI}) that single-object block-SIGReg cannot")
    print("    (deterministic [A]). The learned encoder stays exactly S_O x SO(3)-equivariant [A'], is")
    print("    bi-block-isotropic on the invariant law [C], and its relational content transfers across BOTH")
    print("    groups [E1]. Realising the per-block split on the learned net needs a scale-sensitive task")
    print("    ([E2], diagnostic) — the pure-SSL split stays underdetermined ([D]), exactly as Step 39 §7.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
