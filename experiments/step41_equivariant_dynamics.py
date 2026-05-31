r"""Step 41: **equivariant latent dynamics** — making C2 (orbit transport of the world
condition) a theorem + a falsifiable experiment.

Where this sits, and the one claim it finishes
-----------------------------------------------
"When Does LeJEPA Learn a World Model?" (Klindt, LeCun, Balestriero, arXiv:2605.26379;
"KLB") shows LeJEPA's isotropic-Gaussian target identifies the latent only up to a
*global* orthogonal nuisance $Q\in O(n)$, and that a world (a stationary, additive-noise
latent transition) forces the **Ornstein–Uhlenbeck** form $z_{t+1}=r\,z_t+\sqrt{1-r^2}\,\eta$.
Their forward theorem (5.1) is a Hermite-spectral statement; their planning theorem (5.4)
assumes an $O(n)$-invariant cost. Steps 39/40 sharpened the *static* picture: a known
$\rho:G\to O(n)$ makes a $G$-invariant second moment block-isotropic (Prop. 1), dropping the
gauge from $O(n)$ to the commutant $\prod_i O(m_i)$ — **but only when the per-irrep scales
differ.** In *pure* SSL those scales are underdetermined (Step 39 §7), so the gauge stays
stuck at $O(n)$ and the refinement is a statement about the target class, not something SSL
reaches unaided.

**C2 / Proposition 2 (this step).** Put a *world* on top: a $G$-equivariant latent
transition (its kernel commutes with $\rho$, $T(\rho(g)z'\mid\rho(g)z)=T(z'\mid z)$). Then

  (a) **Schur block structure of the dynamics.** The OU drift, noise and stationary
      covariance are simultaneously block-diagonal, $A=\bigoplus_i\mathbf I_{d_i}\otimes A_i$,
      $Q=\bigoplus_i\mathbf I_{d_i}\otimes Q_i$, $\Sigma_\infty=\bigoplus_i\mathbf I_{d_i}\otimes S_i$
      (the dynamical analogue of Prop. 1).

  (b) **The world model resolves the gauge SSL leaves free.** Choose per-irrep dynamics
      $r_i$ that *differ* while the stationary scales *coincide* ($S_i=\sigma^2\mathbf I$).
      Then the **static** covariance spectrum is degenerate — $\Sigma_\infty=\sigma^2\mathbf I_n$,
      gauge $O(n)$, exactly the regime Step 39 calls underdetermined — yet the **dynamical**
      (drift-operator) spectrum *separates the irreps*, $\operatorname{spec}A=\{r_i\ (\times d_im_i)\}$,
      so its eigenspaces are the isotypic blocks and the gauge drops to $\prod_i O(d_im_i)$.
      The predictor *is* the "scale-sensitive task" Step 40 [E2] had to install by hand —
      here supplied for free by the dynamics.

  (c) **Orbit transport of the world condition (C2 proper).** Because the kernel commutes
      with $\rho$, stationarity + additive-noise need only be checked on a fundamental domain
      $\mathcal F$; equivariance transports them to every orbit. Operationally: the one-step
      **Bayes error is orbit-constant**, $\;\mathbb E\lVert z'-A\,\rho(g)z\rVert^2$ independent
      of $g$ — a flatness theorem certifying the transport. The optimal predictor is then the
      linear-equivariant $z\mapsto Az$, with refined forward bound (cf. KLB Thm 5.1)
      $$\mathcal L^\star=\operatorname{tr}Q=\sum_i d_i\,\operatorname{tr}Q_i=\sum_i d_i m_i\,\sigma^2(1-r_i^2).$$

Two halves, each guarded (mirrors Step 39):

* **[A] objective-level (deterministic).** No learning. [A1] the commutant-projected drift
  commutes with $\rho(R)$ to the float floor (a generic drift does not). [A2] **HEADLINE** —
  the dynamical gauge ladder: the *static* spectrum of $\Sigma_\infty=\mathbf I$ gives one
  $O(22)=231$ cluster (the stuck regime), while the *dynamical* spectrum of $A$ splits into
  the isotypic eigenspaces $O(18)\times O(4)=159$; robust across the clustering gap_factor.
  [A3] the one-step Bayes error is orbit-constant for the commuting drift (~float floor) and
  *varies* for a non-commuting drift (positive control) — C2's flatness, made discriminating.
* **[B–E] learned (the JEPA payoff).** Train a mixed-type **equivariant** predictor (and a
  non-equivariant MLP control) by one-step MSE on the equivariant OU. Then: [B/A'] predictor
  equivariance at init and after training (MLP fails); [C] the learned latent transition is
  Schur block-diagonal on the $G$-invariant law (cross-time cross-block $\to 0$) and recovers
  the true per-irrep $r_i$ — with a **negative control** (same predictor, a non-$G$-invariant
  *anisotropic* law: block structure must break); [D] **the payoff** — the learned drift's
  *dynamical* gauge lands on the $159$ rung while the *static* covariance gauge stays $231$,
  realising on a learned net the refinement pure SSL leaves free (gated cautiously; a run that
  fails to separate is reported INCONCLUSIVE, never forced); [E] 举一反三 — a predictor fit on
  a thin orientation wedge transfers across all of SO(3) for the equivariant model (orbit-
  constant error, by [A3]) while the MLP degrades off the wedge.

Honest scope. The latent OU is synthetic (the price of a *provable* 3D symmetry at laptop
scale). [A] is the rigorous core; [B–E] show the structure survives on a learned net. We do
**not** loosen any guard to force a pass — a run that fails to separate is INCONCLUSIVE.

Run:
    python experiments/step41_equivariant_dynamics.py
    # fast smoke: STEP41_SMOKE=1 python experiments/step41_equivariant_dynamics.py
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.geometry.irreps import IrrepBlock  # noqa: E402
from src.models.structured import VNLinear  # noqa: E402

torch.set_default_dtype(torch.float32)
SMOKE = bool(os.environ.get("STEP41_SMOKE"))

# --------------------------------------------------------------------------- #
# latent layout: two inequivalent SO(3) irreps (same as Step 39), so the drift's
# block structure is non-trivial and the dynamical gauge ladder has somewhere to go.
#   n_0 = 4 invariant scalars (0e, d=1, multiplicity 4)
#   n_1 = 6 vectors           (1o, d=3, multiplicity 6)  -> latent_dim = 22
# rho(R) = I_4 (+) (I_6 ⊗ R), vectors stored CHANNEL-MAJOR ([ch0_xyz, ch1_xyz, ...]).
# Dynamical gauge ladder (Prop. 2): static spectrum of Sigma_inf=I -> O(22) [231];
#   dynamical spectrum of A (distinct r_i) -> O(18)xO(4) [159] --(known rho ⊗ I_3)--> O(6)xO(4) [21].
# --------------------------------------------------------------------------- #
N_SCALAR = 4
N_VEC = 6
LATENT_DIM = N_SCALAR + 3 * N_VEC  # 22
BLOCKS = [IrrepBlock("0e", 0, 1, N_SCALAR), IrrepBlock("1o", N_SCALAR, 3, N_VEC)]

# headline dynamics: DISTINCT per-irrep AR coefficients, EQUAL stationary scales.
R0 = 0.2  # scalar-block AR coefficient r_0
R1 = 0.9  # vector-block AR coefficient r_1  (r_1/r_0 = 4.5 -> robust spectral clustering)
SIGMA2 = 1.0  # common stationary variance -> Sigma_inf = I_22 (the degenerate STATIC spectrum)


def _dim_on(k: int) -> int:
    r"""$\dim O(k)=\binom{k}{2}$ — the gauge carried by a $k$-fold eigenvalue."""
    return k * (k - 1) // 2


# --------------------------------------------------------------------------- #
# SO(3) helper (Haar-random rotation, via a uniform unit quaternion on S^3). Same as
# Step 39: the orbit-transport tests need genuinely uniform group elements.
# --------------------------------------------------------------------------- #
def rand_so3(gen: torch.Generator) -> torch.Tensor:
    r"""A **Haar-random** rotation, via a uniform unit quaternion on $S^3$ (Shoemake 1992)."""
    q = torch.randn(4, generator=gen)
    q = q / q.norm().clamp_min(1e-12)
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def apply_rho(z: torch.Tensor, R: torch.Tensor, blocks=BLOCKS) -> torch.Tensor:
    r"""Act with $\rho(R)=\mathbf I_{n_0}\oplus(\mathbf I_{n_1}\otimes R)$ on a flat latent.

    Scalars (0e) are invariant; each vector channel (1o, channel-major) rotates by $R$.
    ``(N, n) -> (N, n)``.
    """
    b0, b1 = blocks
    out = z.clone()
    v = z[:, b1.start : b1.stop].reshape(z.shape[0], b1.m, 3)  # (N, m, 3)
    out[:, b1.start : b1.stop] = (v @ R.T).reshape(z.shape[0], -1)
    return out


# --------------------------------------------------------------------------- #
# the equivariant OU world: z_{t+1} = A z_t + eps,  eps ~ N(0, Q).
# A and Q commute with rho (Schur block-diagonal). The headline picks distinct r_i but a
# common stationary scale, so Sigma_inf = sigma^2 I (static spectrum degenerate) while
# spec(A) = {r_0 (x4), r_1 (x18)} (dynamical spectrum separates the irreps).
# --------------------------------------------------------------------------- #
def equivariant_drift(r0: float = R0, r1: float = R1) -> torch.Tensor:
    r"""Block drift $A=r_0\mathbf I_4\;\oplus\;r_1\mathbf I_{18}$ (a Schur-diagonal, hence
    $\rho$-commuting, OU drift with the headline distinct-$r$/equal-scale choice)."""
    A = torch.zeros(LATENT_DIM, LATENT_DIM)
    b0, b1 = BLOCKS
    A[b0.start : b0.stop, b0.start : b0.stop] = r0 * torch.eye(b0.width)
    A[b1.start : b1.stop, b1.start : b1.stop] = r1 * torch.eye(b1.width)
    return A


def stationary_noise(r0: float = R0, r1: float = R1, sigma2: float = SIGMA2) -> torch.Tensor:
    r"""Per-coordinate noise variance $q_i=\sigma^2(1-r_i^2)$, returned as a length-$n$ vector,
    so the discrete OU is stationary at $\Sigma_\infty=\sigma^2\mathbf I_n$."""
    q = torch.empty(LATENT_DIM)
    b0, b1 = BLOCKS
    q[b0.start : b0.stop] = sigma2 * (1.0 - r0**2)
    q[b1.start : b1.stop] = sigma2 * (1.0 - r1**2)
    return q


def generic_drift(gen: torch.Generator, scale: float = 0.5) -> torch.Tensor:
    r"""A dense random drift that does **not** commute with $\rho$ (the [A1] positive control:
    a generic operator is not in the commutant)."""
    return scale * torch.randn(LATENT_DIM, LATENT_DIM, generator=gen)


def anisotropic_drift(r0: float = R0, r1: float = R1, spatial=(2.0, 1.0, 0.5)) -> torch.Tensor:
    r"""A drift whose vector block applies a fixed **spatial** scaling $\operatorname{diag}(s)$ to
    every channel — $A_{1o}=r_1\,(\mathbf I_{m}\otimes\operatorname{diag}(s))$. Since
    $\operatorname{diag}(s)$ does not commute with $R\in SO(3)$, this drift is **not** in the
    commutant: a decisive, interpretable [A3] positive control (the dynamical analogue of the
    spatially-anisotropic *law* used in Step 39/40 [A] — same budget, broken equivariance)."""
    A = torch.zeros(LATENT_DIM, LATENT_DIM)
    b0, b1 = BLOCKS
    A[b0.start : b0.stop, b0.start : b0.stop] = r0 * torch.eye(b0.width)
    S = torch.diag(torch.tensor(spatial, dtype=torch.float32))
    A[b1.start : b1.stop, b1.start : b1.stop] = r1 * torch.kron(torch.eye(b1.m), S)
    return A


# stationary latent covariance factors for the two input laws used as z_t marginals.
#   "invariant": z_t ~ N(0, sigma^2 I)  -> G-invariant (isotropic), Sigma_inf exactly.
#   "aniso":     vector channels carry a fixed anisotropic spatial covariance -> NOT
#                rho-invariant (a global rotation changes the law). The [C]/[E] negative
#                control: Prop. 2's premise (G-invariance) removed, block structure must break.
_ANISO_SPATIAL = torch.diag(torch.tensor([2.2, 1.0, 0.45]))  # cov !∝ I_3 -> orientation observable


def _sample_marginal(n: int, law: str, gen: torch.Generator, sigma2: float = SIGMA2) -> torch.Tensor:
    r"""Draw $z_t$ from the stationary marginal of the chosen ``law`` (``'invariant'``/``'aniso'``)."""
    b0, b1 = BLOCKS
    z = torch.zeros(n, LATENT_DIM)
    z[:, b0.start : b0.stop] = math.sqrt(sigma2) * torch.randn(n, b0.width, generator=gen)
    if law == "invariant":
        z[:, b1.start : b1.stop] = math.sqrt(sigma2) * torch.randn(n, b1.width, generator=gen)
    elif law == "aniso":
        # each vector channel ~ N(0, sigma^2 * S) with S a FIXED anisotropic 3x3 (mean-1 trace
        # so the per-channel budget matches 'invariant'); a non-rho-invariant law.
        S = _ANISO_SPATIAL * (3.0 / _ANISO_SPATIAL.trace())  # tr = 3 -> mean eigval 1
        L = torch.linalg.cholesky(sigma2 * S)
        v = torch.randn(n, b1.m, 3, generator=gen) @ L.T
        z[:, b1.start : b1.stop] = v.reshape(n, -1)
    else:
        raise ValueError(law)
    return z


def sample_ou_pairs(
    n: int, A: torch.Tensor, q: torch.Tensor, gen: torch.Generator, *, law: str = "invariant"
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""One-step OU pairs $(z_t, z_{t+1})$, $z_{t+1}=A z_t+\varepsilon$, $\varepsilon\sim\mathcal N(0,\operatorname{diag}q)$."""
    zt = _sample_marginal(n, law, gen)
    eps = torch.randn(n, LATENT_DIM, generator=gen) * q.sqrt()[None, :]
    ztp1 = zt @ A.T + eps
    return zt, ztp1


def wedge_rotate(z: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    r"""Apply an independent **z-axis** rotation in $[0,90°)$ to each sample's vector block —
    the thin orientation slice for the [E] 举一反三 test (a non-uniform sub-orbit of SO(3))."""
    b1 = BLOCKS[1]
    n = z.shape[0]
    ang = torch.rand(n, generator=gen) * (math.pi / 2)
    c, s = torch.cos(ang), torch.sin(ang)
    out = z.clone()
    v = z[:, b1.start : b1.stop].reshape(n, b1.m, 3)
    Rz = torch.zeros(n, 3, 3)
    Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1.0
    out[:, b1.start : b1.stop] = torch.einsum("nmd,nde->nme", v, Rz.transpose(1, 2)).reshape(n, -1)
    return out


# --------------------------------------------------------------------------- #
# predictors. The equivariant one is a mixed-type (scalar+vector) map that commutes with
# rho BY CONSTRUCTION; it is NON-residual (the true drift A=diag(0.2,0.9) is far from I, so
# a residual-around-identity parametrisation would fight the target). It has cross-block and
# nonlinear *capacity* (gains/scalars read invariant features of BOTH types) yet, fit on the
# linear equivariant OU, it learns a block-diagonal linear drift — that is the content of [C].
# --------------------------------------------------------------------------- #
class MixedTypePredictor(nn.Module):
    r"""SO(3)-equivariant one-step predictor $\hat z_{t+1}=f(z_t)$ on the [4 scalar | 6 vector] latent.

    Invariant features $u=[\,s,\ \lVert v_a\rVert\,]\in\mathbb R^{10}$ feed two MLPs: one emits the
    next scalars $\hat s$, one emits per-channel gains $g\in\mathbb R^{6}$; the next vectors are
    $\hat v = \operatorname{diag}(g)\,(W v)$ with $W$ a :class:`VNLinear` channel-mix. Scalars
    are invariant and $\hat v(R\cdot v)=R\,\hat v(v)$ (the gains and $W$ act on invariant/channel
    axes), so $f(\rho(R)z)=\rho(R)f(z)$ exactly. Cross-type *capacity* is present (gains depend
    on $s$; $\hat s$ depends on $\lVert v\rVert$), so block-diagonality of the learned map is a
    *fitted* outcome, not an architectural tautology.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.n_scalar, self.n_vec = N_SCALAR, N_VEC
        n_inv = N_SCALAR + N_VEC  # [s, ||v||]
        self.scal_mlp = nn.Sequential(
            nn.Linear(n_inv, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, N_SCALAR)
        )
        self.gain_mlp = nn.Sequential(
            nn.Linear(n_inv, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, N_VEC)
        )
        self.vlin = VNLinear(N_VEC, N_VEC)

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b0, b1 = BLOCKS
        return b0.slice(z), b1.slice(z).reshape(z.shape[0], b1.m, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, 22) -> (B, 22)
        s, v = self._split(z)  # (B,4), (B,6,3)
        vnorm = v.norm(dim=-1)  # (B,6) invariant
        u = torch.cat([s, vnorm], dim=-1)  # (B,10) invariant features
        s_new = self.scal_mlp(u)  # (B,4) invariant -> scalars
        gains = self.gain_mlp(u)  # (B,6) invariant gates
        v_mix = self.vlin(v)  # (B,6,3) equivariant channel-mix
        v_new = v_mix * gains[..., None]  # (B,6,3) equivariant (gains are invariant)
        return torch.cat([s_new, v_new.reshape(z.shape[0], -1)], dim=-1)

    def irrep_blocks(self) -> list[IrrepBlock]:
        return list(BLOCKS)


class MLPPredictor(nn.Module):
    r"""Non-equivariant control: a plain MLP $\mathbb R^{22}\to\mathbb R^{22}$ (same nominal blocks)."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, LATENT_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def irrep_blocks(self) -> list[IrrepBlock]:
        return list(BLOCKS)


def train_predictor(
    model: nn.Module,
    zt: torch.Tensor,
    ztp1: torch.Tensor,
    *,
    epochs: int,
    batch: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
) -> float:
    r"""One-step MSE fit $\min\lVert f(z_t)-z_{t+1}\rVert^2$ (Adam, seeded). Returns final loss."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    m = zt.shape[0]
    last = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(m, generator=torch.Generator().manual_seed(seed + ep))
        for s in range(0, m, batch):
            idx = perm[s : s + batch]
            pred = model(zt[idx])
            loss = ((pred - ztp1[idx]) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
    return last


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def rho_matrix(R: torch.Tensor) -> torch.Tensor:
    r"""The orthogonal representation $\rho(R)=\mathbf I_{n_0}\oplus(\mathbf I_{n_1}\otimes R)$ as a matrix."""
    P = torch.zeros(LATENT_DIM, LATENT_DIM)
    b0, b1 = BLOCKS
    P[b0.start : b0.stop, b0.start : b0.stop] = torch.eye(b0.width)
    P[b1.start : b1.stop, b1.start : b1.stop] = torch.kron(torch.eye(b1.m), R)
    return P


def commutant_project(M: torch.Tensor) -> torch.Tensor:
    r"""Orthogonal projection of a $22\times22$ matrix onto the commutant of $\rho(G)$.

    By Schur the commutant is $\{A_0\oplus(A_1\otimes\mathbf I_3)\}$: the scalar block is free,
    the vector block must be a channel-mix $\otimes\,\mathbf I_3$, and cross-type blocks vanish.
    The projection keeps the scalar block, zeros the cross blocks, and replaces each $3\times3$
    sub-block $M_{ab}$ of the vector block by $\tfrac13\operatorname{tr}(M_{ab})\,\mathbf I_3$.
    """
    P = torch.zeros_like(M)
    b0, b1 = BLOCKS
    P[b0.start : b0.stop, b0.start : b0.stop] = M[b0.start : b0.stop, b0.start : b0.stop]
    V = M[b1.start : b1.stop, b1.start : b1.stop].reshape(b1.m, 3, b1.m, 3)
    A1 = torch.einsum("aibi->ab", V) / 3.0  # trace over the spatial axis -> (m, m) channel-mix
    P[b1.start : b1.stop, b1.start : b1.stop] = torch.kron(A1, torch.eye(3))
    return P


def commute_err(M: torch.Tensor, R: torch.Tensor) -> float:
    r"""$\lVert M\,\rho(R)-\rho(R)\,M\rVert_\infty$ — zero iff $M$ lies in the commutant of $\rho(R)$."""
    P = rho_matrix(R)
    return (M @ P - P @ M).abs().max().item()


def _cluster_sizes(ev_desc: list[float], gap_factor: float) -> list[int]:
    r"""Group a DESCENDING positive spectrum into clusters: a new cluster opens when the
    multiplicative drop exceeds ``gap_factor``. Shared by the static and dynamical gauges."""
    clusters = [[ev_desc[0]]]
    for x in ev_desc[1:]:
        if clusters[-1][-1] / max(x, 1e-12) > gap_factor:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return [len(c) for c in clusters]


@torch.no_grad()
def static_gauge(z: torch.Tensor, *, gap_factor: float = 2.0) -> dict:
    r"""Covariance spectrum -> static spectral gauge $\dim\operatorname{Stab}_{O(n)}(\hat\Sigma)$.

    A $k$-fold eigenvalue contributes $\dim O(k)=\binom k2$. The headline law has
    $\Sigma_\infty=\sigma^2\mathbf I$ -> one cluster of $n$ -> full $O(n)$ (the stuck regime).
    """
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (zc.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-12).flip(0).tolist()
    sizes = _cluster_sizes(ev, gap_factor)
    return {"eigs": ev, "cluster_sizes": sizes, "gauge_dim": sum(_dim_on(k) for k in sizes)}


@torch.no_grad()
def dynamical_gauge(zt: torch.Tensor, ztp1: torch.Tensor, *, gap_factor: float = 2.0) -> dict:
    r"""Drift-operator spectrum -> **dynamical** gauge.

    Whiten by the static covariance and symmetrise the cross-time covariance:
    $M=\Sigma^{-1/2}\,\operatorname{sym}(C_1)\,\Sigma^{-1/2}$, $C_1=\mathbb E[z_{t+1}z_t^\top]$.
    For the equivariant OU $M$'s eigenvalues are the AR coefficients $r_i$ (mult. $d_im_i$),
    so distinct $r_i$ split the spectrum into the isotypic eigenspaces even when $\Sigma=\mathbf I$.
    Cluster $|$eigenvalues$|$ descending (sign of $r_i$ is immaterial to the eigenspace gauge).
    """
    zc = zt - zt.mean(0, keepdim=True)
    pc = ztp1 - ztp1.mean(0, keepdim=True)
    n = zc.shape[0]
    Sigma = (zc.T @ zc) / (n - 1)
    C1 = (pc.T @ zc) / (n - 1)  # E[z' z^T]
    evS, U = torch.linalg.eigh(Sigma)
    inv_sqrt = U @ torch.diag(evS.clamp_min(1e-8).rsqrt()) @ U.T
    M = inv_sqrt @ (0.5 * (C1 + C1.T)) @ inv_sqrt
    ev = torch.linalg.eigvalsh(M).abs().clamp_min(1e-12)
    ev = ev.sort(descending=True).values.tolist()
    sizes = _cluster_sizes(ev, gap_factor)
    return {"eigs": ev, "cluster_sizes": sizes, "gauge_dim": sum(_dim_on(k) for k in sizes)}


@torch.no_grad()
def cross_block_coupling(M: torch.Tensor) -> float:
    r"""Normalised scalar$\leftrightarrow$vector coupling of a $22\times22$ matrix:
    $\lVert M_{0e,1o}\rVert_F\!/\!\sqrt{\lVert M_{0e}\rVert_F\lVert M_{1o}\rVert_F}$ (Prop. 2 forces $0$)."""
    b0, b1 = BLOCKS
    s0, s1 = slice(b0.start, b0.stop), slice(b1.start, b1.stop)
    off = torch.cat([M[s0, s1].flatten(), M[s1, s0].flatten()])
    denom = math.sqrt(M[s0, s0].norm().item() * M[s1, s1].norm().item() + 1e-12)
    return off.norm().item() / (denom + 1e-12)


@torch.no_grad()
def transition_second_moment(model: nn.Module, zt: torch.Tensor) -> torch.Tensor:
    r"""Cross-time second moment of the LEARNED transition, $C_1=\tfrac1N\sum f(z_t)\,z_t^\top$.

    Under a $G$-invariant input law and an equivariant $f$, $C_1$ lies in the commutant
    (block-diagonal); a non-invariant law or a non-equivariant $f$ breaks that — the content of [C].
    """
    pred = model(zt)
    pc = pred - pred.mean(0, keepdim=True)
    zc = zt - zt.mean(0, keepdim=True)
    return (pc.T @ zc) / (zc.shape[0] - 1)


@torch.no_grad()
def per_block_r(model: nn.Module, zt: torch.Tensor) -> dict[str, float]:
    r"""Recovered per-irrep AR coefficient $\hat r_i=\langle f(z)_i, z_i\rangle/\langle z_i,z_i\rangle$
    per block (the least-squares scalar drift restricted to each isotypic component)."""
    pred = model(zt)
    out = {}
    for b in BLOCKS:
        zb, pb = b.slice(zt), b.slice(pred)
        out[b.name] = float((pb * zb).sum() / (zb * zb).sum().clamp_min(1e-12))
    return out


@torch.no_grad()
def predictor_equiv_err(model: nn.Module, z: torch.Tensor, R: torch.Tensor) -> float:
    r"""$\max\lVert\rho(R)f(z)-f(\rho(R)z)\rVert_\infty$ — the predictor's equivariance residual."""
    lhs = apply_rho(model(z), R)
    rhs = model(apply_rho(z, R))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def one_step_relmse(model: nn.Module, zt: torch.Tensor, ztp1: torch.Tensor) -> float:
    r"""Relative one-step MSE $\sum\lVert f(z_t)-z_{t+1}\rVert^2/\sum\lVert z_{t+1}\rVert^2$.

    The denominator $\mathbb E\lVert z'\rVert^2$ is orbit-invariant (the law commutes with $\rho$),
    so seen-vs-OOD differences come entirely from the numerator — the [E] orbit-transport test.
    """
    num = ((model(zt) - ztp1) ** 2).sum()
    den = (ztp1**2).sum().clamp_min(1e-12)
    return float(num / den)


@torch.no_grad()
def bayes_orbit_errors(A: torch.Tensor, zt: torch.Tensor, ztp1: torch.Tensor, Rs) -> list[float]:
    r"""One-step Bayes error $\mathbb E\lVert\rho(R)z'-A\,\rho(R)z\rVert^2$ on each orbit-transport
    $R$ of the same base batch, using drift ``A`` as the predictor. Orbit-constant iff $A$
    commutes with $\rho$ (C2's flatness); varies otherwise (the [A3] positive control)."""
    errs = []
    for R in Rs:
        ztr, ztpr = apply_rho(zt, R), apply_rho(ztp1, R)
        resid = ztpr - ztr @ A.T
        errs.append(float((resid**2).sum(-1).mean()))
    return errs


@torch.no_grad()
def block_vec_iso(M: torch.Tensor) -> float:
    r"""Mean over the $m$ vector channels of $\lambda_{\max}/\lambda_{\min}$ of each channel's
    diagonal $3\times3$ sub-block of $M$ — Prop. 2 forces every such block $\propto\mathbf I_3$
    (ratio $1$) under a $G$-invariant law; a non-invariant (anisotropic) law inflates it."""
    b1 = BLOCKS[1]
    V = M[b1.start : b1.stop, b1.start : b1.stop].reshape(b1.m, 3, b1.m, 3)
    ratios = []
    for a in range(b1.m):
        sym = 0.5 * (V[a, :, a, :] + V[a, :, a, :].T)
        ev = torch.linalg.eigvalsh(sym).abs().clamp_min(1e-12)
        ratios.append((ev[-1] / ev[0]).item())
    return float(np.mean(ratios))


# --------------------------------------------------------------------------- #
# [E] data: a FIXED anisotropic latent template, presented at a thin z-wedge of
# orientations (seen) or across all of SO(3) (OOD). Rotating an anisotropic template is
# what makes "seen" a strict sub-orbit and OOD the full orbit (cf. Step 39 make_clouds).
# --------------------------------------------------------------------------- #
_TPL_S = torch.from_numpy(np.random.default_rng(1).standard_normal(N_SCALAR).astype(np.float32))
_TPL_V = torch.from_numpy(
    np.random.default_rng(2).standard_normal((N_VEC, 3)).astype(np.float32)
    * np.array([1.3, 0.7, 0.4], np.float32)  # anisotropic -> orientation observable
)


def make_dynamics_slice(
    n: int, A: torch.Tensor, q: torch.Tensor, *, orient: str, gen: torch.Generator, jit: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""OU pairs whose $z_t$ is the fixed template at a wedge (``'wedge'``: z-rotations $[0,90°)$)
    or Haar (``'haar'``) orientation. The [E] orbit-transport train/test split."""
    s = _TPL_S[None] + jit * torch.randn(n, N_SCALAR, generator=gen)
    v = _TPL_V[None] + jit * torch.randn(n, N_VEC, 3, generator=gen)
    Rs = torch.empty(n, 3, 3)
    if orient == "wedge":
        ang = torch.rand(n, generator=gen) * (math.pi / 2)
        c, s_ = torch.cos(ang), torch.sin(ang)
        Rs[:] = torch.eye(3)
        Rs[:, 0, 0] = c; Rs[:, 0, 1] = -s_; Rs[:, 1, 0] = s_; Rs[:, 1, 1] = c
    elif orient == "haar":
        for i in range(n):
            Rs[i] = rand_so3(gen)
    else:
        raise ValueError(orient)
    v = torch.einsum("nmd,nde->nme", v, Rs.transpose(1, 2))  # rotate each channel
    zt = torch.cat([s, v.reshape(n, -1)], dim=-1)
    eps = torch.randn(n, LATENT_DIM, generator=gen) * q.sqrt()[None, :]
    return zt, zt @ A.T + eps


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 76

    N_TRAIN = 512 if SMOKE else 8000
    N_EVAL = 256 if SMOKE else 1024
    EPOCHS = 4 if SMOKE else 30
    N_COV = 1024 if SMOKE else 8192
    K_ORBIT = 6 if SMOKE else 16

    A = equivariant_drift()
    q = stationary_noise()

    print(line)
    print(f"STEP 41  equivariant latent dynamics  (C2 / Prop. 2){'  (SMOKE)' if SMOKE else ''}")
    print(line)
    print(f"    latent n={LATENT_DIM}: {N_SCALAR} scalars (0e) + {N_VEC} vectors (1o); "
          f"rho(R)=I_4 (+) (I_6 ⊗ R)")
    print(f"    equivariant OU: A = {R0}*I_4 (+) {R1}*I_18,  Q = diag(sigma^2(1-r_i^2)),  "
          f"Sigma_inf = {SIGMA2:g}*I_22")
    print(f"    DISTINCT dynamics r_0={R0}, r_1={R1} but EQUAL stationary scale -> the static")
    print(f"    spectrum is degenerate (O(22)) yet the dynamical spectrum separates (O(18)xO(4)).")

    # ----------------------------------------------------------------- [A1]
    print()
    print(line)
    print("[A1] OBJECTIVE-LEVEL (deterministic): the OU drift is Schur block-diagonal —")
    print("     its commutant projection commutes with rho(R); a generic drift does not.")
    print(line)
    g_a = torch.Generator().manual_seed(11)
    Rs_chk = [rand_so3(g_a) for _ in range(4)]
    G = generic_drift(torch.Generator().manual_seed(3))
    cpG = commutant_project(G)
    err_raw = max(commute_err(G, R) for R in Rs_chk)
    err_proj = max(commute_err(cpG, R) for R in Rs_chk)
    err_A = max(commute_err(A, R) for R in Rs_chk)
    faithful = (commutant_project(A) - A).abs().max().item()
    print(f"    generic drift G            : max||G rho - rho G||      = {err_raw:.3e}  (NOT in commutant)")
    print(f"    commutant-projected P_C(G) : max||P_C(G) rho - rho ..|| = {err_proj:.3e}  (commutes)")
    print(f"    headline drift A           : max||A rho - rho A||       = {err_A:.3e}  (already Schur)")
    print(f"    projection faithful on A: ||P_C(A)-A||_inf = {faithful:.3e}")
    ok_schur = err_proj < 1e-5 and err_raw > 1e-2 and err_A < 1e-5 and faithful < 1e-6

    # ----------------------------------------------------------------- [A2]
    print()
    print(line)
    print("[A2] HEADLINE (deterministic): the DYNAMICAL gauge ladder. Static spectrum of")
    print("     Sigma_inf=I is degenerate (one O(22) cluster); the drift spectrum of A splits")
    print("     into the isotypic eigenspaces (O(18)xO(4)) — the world model resolves what SSL leaves free.")
    print(line)
    Sigma_inf = SIGMA2 * torch.eye(LATENT_DIM)
    ev_stat = torch.linalg.eigvalsh(Sigma_inf).clamp_min(1e-12).flip(0).tolist()
    ev_dyn = torch.linalg.eigvalsh(0.5 * (A + A.T)).abs().clamp_min(1e-12).sort(descending=True).values.tolist()
    sweep = []
    for gf in (1.5, 2.0, 3.0, 4.0):
        ss, ds = _cluster_sizes(ev_stat, gf), _cluster_sizes(ev_dyn, gf)
        sweep.append((gf, sum(_dim_on(k) for k in ss), sum(_dim_on(k) for k in ds), ss, ds))
    _, static_dim, dyn_dim, ss2, ds2 = sweep[1]  # gap_factor = 2
    print(f"    static  spectrum (Sigma_inf): clusters {ss2} -> gauge dim {static_dim}  (= O(22) = 231)")
    print(f"    dynamic spectrum (drift A)  : clusters {ds2} -> gauge dim {dyn_dim}  (= O(18)xO(4) = 159)")
    print("    gauge-ladder robustness vs gap_factor: "
          + "  ".join(f"{gf:g}->{s}/{d}" for gf, s, d, _, _ in sweep)
          + "  (target 231/159 throughout)")
    print("    analytic ladder: O(22)=231 --(distinct-r DYNAMICS)--> O(18)xO(4)=159 "
          "--(known rho ⊗ I_3)--> O(6)xO(4)=21")
    ok_gauge_ladder = static_dim == _dim_on(22) and dyn_dim == _dim_on(18) + _dim_on(4)
    ok_gauge_robust = all(
        s == _dim_on(22) and d == _dim_on(18) + _dim_on(4) for _, s, d, _, _ in sweep
    )

    # ----------------------------------------------------------------- [A3]
    print()
    print(line)
    print("[A3] C2 FLATNESS (deterministic): the one-step Bayes error is ORBIT-CONSTANT for the")
    print("     commuting drift (stationarity transports along orbits); a non-commuting drift varies.")
    print(line)
    # Test on the ANISOTROPIC z_t law: there the orbit-transport is discriminating. (On the
    # isotropic law E_z||.||^2 collapses to a Frobenius norm that is rotation-invariant, so even
    # a WRONG drift looks orbit-flat in expectation — the equivariant predictor is flat for *any*
    # law, but the positive control only reveals variation where the law itself is anisotropic.)
    g_d = torch.Generator().manual_seed(5)
    zt0, ztp0 = sample_ou_pairs(N_EVAL, A, q, g_d, law="aniso")
    Rs = [torch.eye(3)] + [rand_so3(g_d) for _ in range(K_ORBIT)]
    errs_c = bayes_orbit_errors(A, zt0, ztp0, Rs)  # commuting drift: orbit-constant (per-sample exact)
    Aprime = anisotropic_drift()  # spatially-anisotropic vector drift -> NOT rho-commuting
    errs_g = bayes_orbit_errors(Aprime, zt0, ztp0, Rs)
    mean_c = sum(errs_c) / len(errs_c)
    spread_c = (max(errs_c) - min(errs_c)) / max(mean_c, 1e-12)
    spread_g = (max(errs_g) - min(errs_g)) / max(sum(errs_g) / len(errs_g), 1e-12)
    tr_Q = float(q.sum())
    print(f"    commuting drift A : Bayes error over {len(Rs)} orbit transports = {mean_c:.4f} "
          f"(predicted tr Q = {tr_Q:.4f}), spread (max-min)/mean = {spread_c:.3e}")
    print(f"    non-commuting A'  : spread (max-min)/mean = {spread_g:.3e}  (positive control: NOT orbit-constant)")
    print(f"    => C2's premise (kernel commutes with rho) is exactly what makes stationarity transport.")
    ok_flat = spread_c < 1e-4 and spread_g > 0.05

    # ----------------------------------------------------------------- data + models
    g_tr = torch.Generator().manual_seed(0)
    zt_tr, ztp1_tr = sample_ou_pairs(N_TRAIN, A, q, g_tr, law="invariant")
    zt_ev, ztp1_ev = sample_ou_pairs(N_EVAL, A, q, torch.Generator().manual_seed(99), law="invariant")
    zt_cov, _ = sample_ou_pairs(N_COV, A, q, torch.Generator().manual_seed(777), law="invariant")
    zt_aniso, _ = sample_ou_pairs(N_COV, A, q, torch.Generator().manual_seed(778), law="aniso")
    R_test = rand_so3(torch.Generator().manual_seed(3))
    eq, mlp = MixedTypePredictor(), MLPPredictor()

    # ----------------------------------------------------------------- [B] init equiv
    print()
    print(line)
    print("[B] EQUIVARIANCE OF THE PREDICTOR — AT INIT")
    print(line)
    eq_e0 = predictor_equiv_err(eq, zt_ev, R_test)
    mlp_e0 = predictor_equiv_err(mlp, zt_ev, R_test)
    print(f"    eq  predictor: max||rho f(z) - f(rho z)|| = {eq_e0:.3e}")
    print(f"    MLP predictor: max||rho f(z) - f(rho z)|| = {mlp_e0:.3e}  (no prior)")

    # ----------------------------------------------------------------- train
    print()
    print(line)
    print(f"[train] one-step MSE on the equivariant OU — {EPOCHS} epochs")
    print(line)
    eq_loss = train_predictor(eq, zt_tr, ztp1_tr, epochs=EPOCHS, seed=0)
    mlp_loss = train_predictor(mlp, zt_tr, ztp1_tr, epochs=EPOCHS, seed=0)
    eq_te = one_step_relmse(eq, zt_ev, ztp1_ev)
    mlp_te = one_step_relmse(mlp, zt_ev, ztp1_ev)
    print(f"    eq  : train MSE = {eq_loss:.4f}   eval relMSE = {eq_te:.4f}")
    print(f"    mlp : train MSE = {mlp_loss:.4f}   eval relMSE = {mlp_te:.4f}")

    # ----------------------------------------------------------------- [A'] post equiv
    print()
    print(line)
    print("[A'] EQUIVARIANCE AFTER TRAINING (must survive optimisation)")
    print(line)
    eq_e1 = predictor_equiv_err(eq, zt_ev, R_test)
    mlp_e1 = predictor_equiv_err(mlp, zt_ev, R_test)
    print(f"    eq  : max||rho f(z) - f(rho z)|| = {eq_e1:.3e}  (still equivariant)")
    print(f"    mlp : max||rho f(z) - f(rho z)|| = {mlp_e1:.3e}")
    ok_equiv = eq_e1 < 1e-4 and mlp_e1 > 1e-2

    # ----------------------------------------------------------------- [C] Prop 2 on learned transition
    print()
    print(line)
    print(f"[C] PROPOSITION 2 on the LEARNED transition (N={N_COV}): cross-time second moment")
    print("    C_1=E[f(z)z^T] is Schur block-diagonal (cross->0, each 1o block 3x3-isotropic),")
    print("    and the learned per-irrep r_i match. Neg control: same eq predictor, non-G-invariant law.")
    print(line)
    C1_eq = transition_second_moment(eq, zt_cov)
    C1_mlp = transition_second_moment(mlp, zt_cov)
    C1_eq_an = transition_second_moment(eq, zt_aniso)
    cr_eq, iso_eq = cross_block_coupling(C1_eq), block_vec_iso(C1_eq)
    cr_mlp, iso_mlp = cross_block_coupling(C1_mlp), block_vec_iso(C1_mlp)
    cr_an, iso_an = cross_block_coupling(C1_eq_an), block_vec_iso(C1_eq_an)
    r_eq, r_mlp = per_block_r(eq, zt_cov), per_block_r(mlp, zt_cov)
    print(f"    {'model / law':22s} | {'cross':>8s} | {'1o 3x3 iso':>10s} | {'r_hat(0e)':>9s} | {'r_hat(1o)':>9s}")
    print("    " + "-" * 70)
    print(f"    {'eq    INVARIANT':22s} | {cr_eq:8.4f} | {iso_eq:10.3f} | {r_eq['0e']:9.3f} | {r_eq['1o']:9.3f}")
    print(f"    {'mlp   INVARIANT':22s} | {cr_mlp:8.4f} | {iso_mlp:10.3f} | {r_mlp['0e']:9.3f} | {r_mlp['1o']:9.3f}")
    print(f"    {'eq    ANISO (neg-ctrl)':22s} | {cr_an:8.4f} | {iso_an:10.3f} |     ----- |     -----  <- premise removed")
    print(f"    => true (r_0,r_1)=({R0},{R1}); on the G-invariant law BOTH nets fit a near-Schur C_1 and")
    print("       recover r_i (the linear OU drift is an easy target) — Prop. 2's content is the EXACT")
    print("       Schur structure, gated on eq below; discrimination from the MLP shows up off-orbit ([A'],[E]).")
    print("       Neg-ctrl: the SAME eq map on a non-G-invariant (anisotropic) law breaks 3x3-isotropy — [C] can fail.")
    # Prop. 2 [C] gate is on the EQUIVARIANT predictor's recovery of the exact Schur transition on the
    # G-invariant law. We deliberately do NOT gate on the MLP here: the linear OU drift is an easy target,
    # so the MLP also fits a near-block-diagonal C_1 and recovers r_i (full run: cross=0.17, iso=1.07,
    # r_hat=0.20/0.89). The MLP's failure is NOT block-isotropy on the seen law — it is equivariance ([A'])
    # and off-orbit transport ([E]). The falsifiable content of Prop. 2 is (i) eq recovers the EXACT Schur
    # transition on the invariant law, and (ii) the negative control: the same eq map breaks isotropy once
    # the law is no longer G-invariant (premise removed).
    ok_prop2 = (
        cr_eq < 0.12 and iso_eq < 1.35
        and abs(r_eq["0e"] - R0) < 0.1 and abs(r_eq["1o"] - R1) < 0.1
    )
    ok_prop2_negctrl = iso_an > 3.0

    # ----------------------------------------------------------------- [D] learned dynamical gauge (payoff)
    print()
    print(line)
    print("[D] THE PAYOFF — does the LEARNED equivariant transition realise the dynamical gauge")
    print("    rung? Static covariance of z_t is degenerate (231); the learned drift's spectrum")
    print("    should land on 159 — the refinement pure SSL ([D] in Steps 39/40) leaves free.")
    print(line)
    sg_static = static_gauge(zt_cov)
    dg_learn = dynamical_gauge(zt_cov, eq(zt_cov))
    dg_mlp = dynamical_gauge(zt_cov, mlp(zt_cov))
    print(f"    {'spectrum':28s} | {'clusters':>14s} | {'gauge dim':>9s}")
    print("    " + "-" * 58)
    print(f"    {'static cov of z_t':28s} | {str(sg_static['cluster_sizes']):>14s} | {sg_static['gauge_dim']:9d}")
    print(f"    {'learned eq drift (dynamical)':28s} | {str(dg_learn['cluster_sizes']):>14s} | {dg_learn['gauge_dim']:9d}")
    print(f"    {'learned mlp drift (dynamical)':28s} | {str(dg_mlp['cluster_sizes']):>14s} | {dg_mlp['gauge_dim']:9d}")
    print("    => the learned equivariant DYNAMICS separates the irreps where the STATIC law cannot.")
    ok_payoff = sg_static["gauge_dim"] == _dim_on(22) and dg_learn["gauge_dim"] == _dim_on(18) + _dim_on(4)

    # ----------------------------------------------------------------- [E] 举一反三
    print()
    print(line)
    print("[E] 举一反三 — predictor fit on a thin z-wedge of orientations, tested across all SO(3).")
    print("    Equivariance transports the learned drift to every orbit (flat); MLP degrades OOD.")
    print(line)
    g_e = torch.Generator().manual_seed(2026)
    zt_fit, ztp1_fit = make_dynamics_slice(N_TRAIN, A, q, orient="wedge", gen=g_e)
    zt_seen, ztp1_seen = make_dynamics_slice(N_EVAL, A, q, orient="wedge", gen=torch.Generator().manual_seed(4040))
    zt_ood, ztp1_ood = make_dynamics_slice(N_EVAL, A, q, orient="haar", gen=torch.Generator().manual_seed(5050))
    eqE, mlpE = MixedTypePredictor(), MLPPredictor()
    train_predictor(eqE, zt_fit, ztp1_fit, epochs=EPOCHS, seed=1)
    train_predictor(mlpE, zt_fit, ztp1_fit, epochs=EPOCHS, seed=1)
    eq_seen = one_step_relmse(eqE, zt_seen, ztp1_seen)
    eq_ood = one_step_relmse(eqE, zt_ood, ztp1_ood)
    mlp_seen = one_step_relmse(mlpE, zt_seen, ztp1_seen)
    mlp_ood = one_step_relmse(mlpE, zt_ood, ztp1_ood)
    eq_ratio = eq_ood / max(eq_seen, 1e-12)
    mlp_ratio = mlp_ood / max(mlp_seen, 1e-12)
    print(f"    {'model':10s} | {'seen relMSE':>12s} | {'rot-OOD relMSE':>14s} | {'OOD/seen':>9s}")
    print("    " + "-" * 54)
    print(f"    {'eq':10s} | {eq_seen:12.4e} | {eq_ood:14.4e} | x{eq_ratio:7.2f}")
    print(f"    {'mlp':10s} | {mlp_seen:12.4e} | {mlp_ood:14.4e} | x{mlp_ratio:7.2f}")
    print("    => equivariant predictor transfers across the group (orbit-constant by [A3]); MLP does not.")
    ok_jyfs = eq_ratio < 1.3 and mlp_ratio > 1.5

    # ----------------------------------------------------------------- summary + guards
    print()
    print(line)
    print("STEP 41 SUMMARY")
    print(line)
    passed = (
        ok_schur and ok_gauge_ladder and ok_gauge_robust and ok_flat
        and ok_equiv and ok_prop2 and ok_prop2_negctrl and ok_payoff and ok_jyfs
    )
    print(f"    [A1] commutant-projected drift commutes ({err_proj:.1e}), generic does not ({err_raw:.1e}): {ok_schur}")
    print(f"    [A2] dynamical gauge ladder static->{static_dim} / dynamical->{dyn_dim} (231 vs 159): {ok_gauge_ladder}")
    print(f"    [A2] gauge ladder stable across gap_factor in {{1.5,2,3,4}}: {ok_gauge_robust}")
    print(f"    [A3] one-step Bayes error orbit-constant (spread {spread_c:.1e}), non-commuting varies ({spread_g:.1e}): {ok_flat}")
    print(f"    [A'] eq predictor stays equivariant ({eq_e1:.1e}), MLP not ({mlp_e1:.1e}): {ok_equiv}")
    print(f"    [C]  learned transition Schur (eq cross={cr_eq:.3f} iso={iso_eq:.2f}, r_hat={r_eq['0e']:.2f}/{r_eq['1o']:.2f}): {ok_prop2}")
    print(f"    [C]  neg-control: same eq map breaks on non-invariant law (1o iso={iso_an:.2f}): {ok_prop2_negctrl}")
    print(f"    [D]  PAYOFF: learned eq dynamics realises the 159 rung (static {sg_static['gauge_dim']} / dynamical {dg_learn['gauge_dim']}): {ok_payoff}")
    print(f"    [E]  eq predictor flat across SO(3) (x{eq_ratio:.2f}), MLP degrades (x{mlp_ratio:.2f}): {ok_jyfs}")
    verdict = "PASS" if passed else "INCONCLUSIVE"
    print(f"    {verdict}: a G-equivariant latent WORLD (OU transition commuting with rho) has Schur-")
    print("    block dynamics [A1]; choosing distinct per-irrep r_i with equal stationary scale makes the")
    print("    DYNAMICAL gauge separate the irreps (159) where the STATIC covariance cannot (231) [A2] —")
    print("    the world model resolves the gauge pure SSL leaves free (Steps 39/40 §7). Stationarity")
    print("    transports along orbits (one-step Bayes error orbit-constant [A3], C2 proper); the learned")
    print("    equivariant predictor stays exactly equivariant [A'], recovers the Schur transition [C],")
    print("    realises the dynamical rung on a learned net [D], and generalises across SO(3) from a thin")
    print("    orientation wedge [E]. This finishes C2: 'recover up to rho(G)' + stationarity, transported.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
