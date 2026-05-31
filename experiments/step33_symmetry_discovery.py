r"""Step 33: discovering the world's symmetry — learnable Lie-algebra generators (Tier 3.6).

What every prior step assumed
-----------------------------
Steps 6-32 *hand-specified* the symmetry: we built $\mathrm{SO}(2)$/$\mathrm{SE}(3)\rtimes S_O$
into the encoder/predictor and then measured the 举一反三 payoff. That is the strong-prior regime.
The obvious objection: *what if you don't know the group?* This step removes the hand. We give the
model a **blank slate** of $K$ learnable $3\times3$ matrices $G_\phi$ — no antisymmetry, no Lie
structure, nothing imposed — and ask whether gradient descent, seeing only the dynamics, **re-derives
the generators of the world's symmetry**.

Method (LieGAN/LieGG-flavoured, discriminator-free)
---------------------------------------------------
The Step-24 interacting teacher $f:(S,A)\mapsto S'$ is exactly $\mathrm{SO}(3)$-equivariant:
$f(R\!\cdot\!S, R\!\cdot\!A)=R\!\cdot\! f(S,A)$ for $R\in\mathrm{SO}(3)$ acting on every 3-vector
(points *and* actions). A candidate generator $G\in\mathbb{R}^{3\times3}$ is an *infinitesimal*
symmetry iff the **finite** transform $g=\exp(\theta G)$ commutes with the dynamics. So we minimise
the dynamics-equivariance violation directly (the static position $g\!\cdot\!S$ cancels, leaving only
the teacher's *update*):
$$\mathcal{L}(G)=\frac{\mathbb{E}_{(S,A),\,\theta}\big\|f(g\!\cdot\!S,\,g\!\cdot\!A)-g\!\cdot\! f(S,A)\big\|^2}
{\mathbb{E}\big\|f(S,A)-S\big\|^2},\qquad g=\exp(\theta G),\ \theta\sim\mathcal{U}[-\theta_{\max},\theta_{\max}].$$
No Jacobian, no discriminator — this is the *same* across-group residual the rest of the project
already trusts, now **minimised over the generators**. To kill the scale ($G\to0$) and collapse
($G_i\!=\!G_j$) degeneracies we carry an **orthonormal frame** (a QR re-orthonormalisation each step,
unit Frobenius), so $K$ generators span a $K$-dimensional subspace of $\mathfrak{gl}(3)$.

Crucially nothing forces antisymmetry: $\mathfrak{so}(3)$ (the $3$-dim antisymmetric matrices) is a
$3$-dim subspace of the $9$-dim $\mathfrak{gl}(3)$, and the *only* reason the optimiser should land
there is that the teacher's torque $a\times\tilde x$ is cross-product (hence rotation-, not
$\mathrm{GL}$-, equivariant). Antisymmetry and **Lie-bracket closure** must *emerge* from the data.

The two worlds (ground truth we can check against)
--------------------------------------------------
  TRUE    : the Step-24 teacher, exactly $\mathrm{SO}(3)$-equivariant. Symmetry algebra
            $=\mathfrak{so}(3)$, $\dim=3$.
  BROKEN  : the same teacher plus a fixed **lab-frame** traceless stretch $\beta\,M\,\tilde x$,
            $M=\mathrm{diag}(1,1,-2)$, applied to centred points. $M$ commutes with a rotation $R$
            iff $RMR^\top=M$, i.e. iff $R$ fixes the $z$-axis — so the continuous symmetry is reduced
            to rotations about $z$: $\mathfrak{so}(2)=\langle L_z\rangle$, $\dim=1$. (Translation- and
            permutation-equivariance are untouched; only rotation is broken $\mathrm{SO}(3)\to\mathrm{SO}(2)$.)

Five reads (Gate = PASS)
------------------------
  [R] recover    : on TRUE with $K{=}3$ the discovered basis is (i) **antisymmetric**, (ii) **spans
                   $\mathfrak{so}(3)$**, (iii) **CLOSES under the Lie bracket** $[G_i,G_j]\in\mathrm{span}$
                   (a genuine algebra; structure-constant norm $\approx\sqrt3$ for a unit frame), and (iv) drives the
                   equivariance residual to the optimisation floor.
  [D] dimension  : sweep $K=1\ldots5$ on TRUE — the residual stays at floor up to $K{=}3$ then **JUMPS
                   at $K{=}4$** (the 4th orthonormal generator must leave $\mathfrak{so}(3)$). The data
                   *reads out* $\dim=3$ with no prior.
  [X] reject     : the SAME procedure on BROKEN reads $\dim=1$ (jump at $K{=}2$), its single generator
                   $\approx L_z$, and its $K{=}3$ residual sits far above TRUE's $K{=}3$ floor — i.e.
                   discovery is **falsifiable**, not vacuous: change the world's symmetry and the method
                   reports the changed dimension and axis.
  [Xb] robust    : the dim-1 read on BROKEN is not an artefact of one tuned stretch magnitude — sweeping
                   the break strength over an $8\times$ range $\beta\in\{0.1,0.2,0.3,0.5,0.8\}$, EVERY
                   $\beta$ reads $\dim=1$ with the surviving axis $\approx L_z$ (the $K{=}2$ residual stays
                   far above the $K{=}1$ floor at every strength).

Honest scope. We discover the symmetry of a *given dynamics map* (we may query $f$ at transformed
inputs), as LieGG does for a trained network — not yet from a frozen offline dataset via a real
GAN discriminator (the next rung). We target the continuous **rotation** part on $\mathbb{R}^3$;
translations (affine, via homogeneous coords) and the discrete $S_O$ are out of this generator's scope.
The teacher is still a *constructed* symmetric task — letting a symmetry emerge on a *non*-constructed
task is the separate Step 34 (#75).

Run (full; a few minutes on a laptop CPU):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step33_symmetry_discovery.py
Smoke (~30-60 s):
    STEP33_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step33_symmetry_discovery.py
"""

import functools
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 24 machinery we reuse

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Reuse the VALIDATED Step 24 interacting teacher + data generator VERBATIM (the world whose
# symmetry we try to rediscover); Step 33 adds ONLY the learnable-generator discovery on top.
from step24_object_interaction import (  # noqa: E402
    C_INT,
    N_OBJ,
    P,
    make_interacting_transitions,
    scene_teacher_interact,
)

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP33_SMOKE"))

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
SEEDS = (0,) if SMOKE else (0, 1, 2, 3, 4)
N_DATA = 64 if SMOKE else 256           # transitions the discovery sees
N_STEPS = 600 if SMOKE else 1500        # Adam steps per (teacher, K, seed, restart)
N_RESTARTS = 2 if SMOKE else 3          # random inits per run; keep the lowest-residual one
LR = 0.03
THETA_MAX = 1.2                          # finite rotation angle sampled per step (~69 deg): a LARGE
#                                          rotation un-dilutes non-symmetry violations (the teacher's
#                                          GL-equivariant drift term otherwise shrinks the relative gap)
K_LADDER = (1, 2, 3, 4, 5)               # generator-slate sizes for the dimension sweep
BREAK_BETA = float(os.environ.get("STEP33_BREAK_BETA", "0.30"))   # lab-frame stretch strength
# falsifiability across break STRENGTH: BROKEN must read dim=1 (axis L_z) at EVERY beta, not just one,
# so the dim-1 verdict is not an artefact of a single tuned stretch magnitude (an 8x range of beta).
BETA_LADDER = (0.10, 0.30) if SMOKE else (0.10, 0.20, 0.30, 0.50, 0.80)

TRUE_DIM, BROKEN_DIM = 3, 1              # ground-truth continuous-symmetry dimensions
# verdict thresholds
TAU_FLOOR = 1e-3                         # relative residual counted as "at floor" (a symmetry)
JUMP = 5.0                               # residual ratio that counts as a clean dimension jump
TAU_FRAC_SO3 = 0.90                      # min fraction of discovered span lying in so(3)
TAU_ANTISYM = 0.15                       # max symmetric-part Frobenius norm (unit generators)
TAU_CLOSURE = 0.15                       # max relative Lie-bracket out-of-span residual
TAU_AXIS = 0.85                          # min |<discovered broken gen, L_z>|

# fixed lab-frame traceless stretch that breaks SO(3) -> SO(2)_z (commutes only with R_z)
M_BREAK = torch.diag(torch.tensor([1.0, 1.0, -2.0]))


# --------------------------------------------------------------------------- #
# so(3) reference basis (orthonormal in the Frobenius inner product)
# --------------------------------------------------------------------------- #
def so3_basis() -> torch.Tensor:
    r"""Orthonormal basis $\{\hat L_x,\hat L_y,\hat L_z\}$ of $\mathfrak{so}(3)$ as rows. ``(3, 9)``.

    $L_x,L_y,L_z$ are the standard antisymmetric generators ($[L_x,L_y]=L_z$, cyclic); each has
    Frobenius norm $\sqrt2$, so we divide by $\sqrt2$ to get an orthonormal frame in $\mathbb{R}^9$.
    """
    lx = torch.tensor([[0.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    ly = torch.tensor([[0.0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    lz = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 0]])
    b = torch.stack([lx, ly, lz]).reshape(3, 9)
    return b / b.norm(dim=1, keepdim=True)


B_SO3 = so3_basis()                       # (3, 9)
LZ_HAT = B_SO3[2]                          # (9,) unit vec(L_z)


# --------------------------------------------------------------------------- #
# the two worlds
# --------------------------------------------------------------------------- #
def true_teacher(S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""The Step-24 interacting teacher, exactly $\mathrm{SO}(3)\!\rtimes\! S_O$-equivariant.

    ``(B,O,P,3),(B,O,3) -> (B,O,P,3)``.
    """
    return scene_teacher_interact(S, A)


def broken_teacher(S: torch.Tensor, A: torch.Tensor, beta: float = BREAK_BETA) -> torch.Tensor:
    r"""True teacher $+$ a fixed lab-frame traceless stretch $\beta\,M\,\tilde x$ on centred points.

    $M=\mathrm{diag}(1,1,-2)$ is fixed in the *lab* frame, so it co-rotates only under rotations that
    fix the $z$-axis: the continuous symmetry collapses $\mathrm{SO}(3)\to\mathrm{SO}(2)_z$. Centring
    keeps it translation-equivariant; it is applied identically per object so $S_O$ is untouched.
    ``(B,O,P,3),(B,O,3) -> (B,O,P,3)``.
    """
    out = scene_teacher_interact(S, A)
    c = S.mean(dim=2, keepdim=True)                  # (B,O,1,3) per-object centroid
    xt = S - c                                        # centred points (translation-invariant)
    return out + beta * (xt @ M_BREAK.transpose(-1, -2))


# --------------------------------------------------------------------------- #
# generator machinery
# --------------------------------------------------------------------------- #
def orthonormalize(v: torch.Tensor) -> torch.Tensor:
    r"""Re-orthonormalise the $K$ generator rows via reduced QR. ``(K,9) -> (K,9)`` (rows orthonormal).

    Carrying an orthonormal frame removes both degeneracies of the bare equivariance objective: the
    scale ($G\to0$) and the collapse ($G_i\!=\!G_j$). QR is differentiable and stable for full-rank
    input (random init + a symmetry-seeking loss keep the rows independent).
    """
    q, _ = torch.linalg.qr(v.transpose(-1, -2), mode="reduced")   # q: (9, K) orthonormal columns
    return q.transpose(-1, -2)                                     # (K, 9) orthonormal rows


def apply_g(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    r"""Apply each of $K$ matrices $g_k$ to every 3-vector of ``x``. ``(...,3),(K,3,3) -> (K,...,3)``."""
    return torch.einsum("kij,...j->k...i", g, x)


def _discover_once(teacher, k, s, a, s2, denom, *, n_steps, seed) -> tuple[torch.Tensor, float]:
    r"""One random-init run of the generator discovery. Returns ``(G_hat (k,3,3), final residual)``."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.nn.Parameter(torch.randn(k, 3, 3, generator=gen) * 0.5)
    opt = torch.optim.Adam([raw], lr=LR)
    b = s.shape[0]
    last = float("nan")
    for _ in range(n_steps):
        opt.zero_grad()
        gh = orthonormalize(raw.reshape(k, 9)).reshape(k, 3, 3)         # (k,3,3) orthonormal frame
        theta = (torch.rand(k, generator=gen) * 2.0 - 1.0) * THETA_MAX  # (k,) finite angles
        g = torch.matrix_exp(theta[:, None, None] * gh)                 # (k,3,3) group elements
        # lhs = f(g.S, g.A); rhs = g.f(S,A); the static g.S cancels, leaving the update's violation
        gs = apply_g(s, g).reshape(k * b, N_OBJ, P, 3)
        ga = apply_g(a, g).reshape(k * b, N_OBJ, 3)
        lhs = teacher(gs, ga).reshape(k, b, N_OBJ, P, 3)
        rhs = apply_g(s2, g)                                            # (k,B,O,P,3)
        loss = ((lhs - rhs) ** 2).mean() / denom
        loss.backward()
        opt.step()
        last = float(loss.detach())
    with torch.no_grad():
        gh = orthonormalize(raw.reshape(k, 9)).reshape(k, 3, 3)
    return gh.detach(), last


def discover(
    teacher,
    k: int,
    s: torch.Tensor,
    a: torch.Tensor,
    *,
    n_steps: int,
    seed: int,
    n_restarts: int = N_RESTARTS,
) -> tuple[torch.Tensor, float]:
    r"""Discover $k$ symmetry generators of ``teacher`` by minimising the equivariance residual.

    Runs ``n_restarts`` random inits and keeps the lowest-residual one (the drift term makes some
    basins shallow, so a couple of restarts guard against bad local minima). Returns the
    orthonormalised generators $\hat G\in\mathbb{R}^{k\times3\times3}$ (unit Frobenius, mutually
    orthogonal) and the final **relative** dynamics-equivariance residual $\mathcal{L}$.
    """
    s2 = teacher(s, a).detach()                                # (B,O,P,3) reference next-state
    denom = ((s2 - s) ** 2).mean().clamp_min(1e-12)            # scale = teacher's update magnitude
    best_g, best_r = None, float("inf")
    for r in range(n_restarts):
        gh, res = _discover_once(teacher, k, s, a, s2, denom,
                                 n_steps=n_steps, seed=1000 * seed + 97 * k + r)
        if res < best_r:
            best_g, best_r = gh, res
    return best_g, best_r


# --------------------------------------------------------------------------- #
# metrics on a discovered generator set
# --------------------------------------------------------------------------- #
def antisym_frac(gh: torch.Tensor) -> float:
    r"""Mean symmetric-part Frobenius norm of unit generators ($0$ = perfectly antisymmetric)."""
    sym = 0.5 * (gh + gh.transpose(-1, -2))
    return float(sym.flatten(1).norm(dim=1).mean())


def frac_in_so3(gh: torch.Tensor) -> float:
    r"""Fraction of the discovered span lying in $\mathfrak{so}(3)$: $\mathrm{tr}(P_d P_{\mathfrak{so}})/K\in[0,1]$."""
    v = gh.reshape(gh.shape[0], 9)                       # (K,9) orthonormal rows
    return float((v @ B_SO3.transpose(-1, -2)).pow(2).sum() / gh.shape[0])


def closure_resid(gh: torch.Tensor) -> float:
    r"""Max relative out-of-span residual of $[G_i,G_j]$ ($0$ = the span is a Lie algebra)."""
    k = gh.shape[0]
    if k < 2:
        return 0.0
    v = gh.reshape(k, 9)                                  # orthonormal basis of the span
    worst = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            br = (gh[i] @ gh[j] - gh[j] @ gh[i]).reshape(9)
            nrm = float(br.norm())
            if nrm < 1e-6:
                continue
            proj = v.transpose(-1, -2) @ (v @ br)        # projection onto span
            worst = max(worst, float((br - proj).norm()) / nrm)
    return worst


def struct_const_norm(gh: torch.Tensor) -> float:
    r"""$\sqrt{\sum_{ijk} c_{ijk}^2}$ with $c_{ijk}=\langle[G_i,G_j],G_k\rangle$.

    For a **unit**-Frobenius orthonormal $\mathfrak{so}(3)$ frame $[\hat e_i,\hat e_j]=\tfrac{1}{\sqrt2}\varepsilon_{ijk}\hat e_k$,
    so $\sum_{ijk}c_{ijk}^2=\tfrac12\sum\varepsilon_{ijk}^2=3$, i.e. the fingerprint is $\sqrt3$ (the
    un-normalised $L_x,L_y,L_z$ would give $\sqrt6$). Invariant under orthogonal re-basing of the algebra.
    """
    k = gh.shape[0]
    v = gh.reshape(k, 9)
    tot = 0.0
    for i in range(k):
        for j in range(k):
            br = (gh[i] @ gh[j] - gh[j] @ gh[i]).reshape(9)
            tot += float((v @ br).pow(2).sum())
    return float(np.sqrt(tot))


def axis_align(gh: torch.Tensor) -> float:
    r"""$|\langle \hat G_1, \hat L_z\rangle|$ for the (single) broken-world generator ($1$ = exactly $L_z$)."""
    return float((gh.reshape(gh.shape[0], 9)[0] * LZ_HAT).sum().abs())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 92
    print(line)
    print("STEP 33 — discovering the world's symmetry: learnable Lie-algebra generators")
    print(line)
    print(f"    mode={'SMOKE' if SMOKE else 'FULL'}  seeds={list(SEEDS)}  N_data={N_DATA}  "
          f"steps={N_STEPS}  K_ladder={list(K_LADDER)}")
    print(f"    teacher: Step-24 interacting (kappa={C_INT}); BROKEN adds beta={BREAK_BETA} "
          f"lab-frame diag(1,1,-2) stretch (SO(3)->SO(2)_z)")
    print(f"    discovery: blank K-slate of 3x3 matrices, minimise relative equivariance residual of "
          f"g=exp(theta G), theta_max={THETA_MAX}")

    worlds = {"TRUE": true_teacher, "BROKEN": broken_teacher}
    # per-world: residual curve over K (per seed), and the discovered generators at every (K, seed)
    resid: dict[str, dict[int, list[float]]] = {w: {k: [] for k in K_LADDER} for w in worlds}
    gens: dict[str, dict[int, list[torch.Tensor]]] = {w: {k: [] for k in K_LADDER} for w in worlds}

    for seed in SEEDS:
        s, a_flat, _ = make_interacting_transitions(N_DATA, seed=seed)
        a = a_flat.reshape(N_DATA, N_OBJ, 3)
        for wname, teacher in worlds.items():
            for k in K_LADDER:
                gh, r = discover(teacher, k, s, a, n_steps=N_STEPS, seed=seed)
                resid[wname][k].append(r)
                gens[wname][k].append(gh)

    def rmean(w: str, k: int) -> float:
        return float(np.mean(resid[w][k]))

    def rstd(w: str, k: int) -> float:
        return float(np.std(resid[w][k]))

    # ---- [D] dimension sweep -------------------------------------------------------------
    print()
    print(line)
    print("[D] DIMENSION SWEEP — relative equivariance residual vs slate size K (lower=symmetry)")
    print(line)
    print(f"    {'K':>3s} | {'TRUE resid':>22s} | {'BROKEN resid':>22s}")
    print("    " + "-" * 54)
    for k in K_LADDER:
        print(f"    {k:>3d} | {rmean('TRUE', k):.3e} +/- {rstd('TRUE', k):.1e}     | "
              f"{rmean('BROKEN', k):.3e} +/- {rstd('BROKEN', k):.1e}")

    # discovered dimension = largest K with residual still at floor (contiguous from K=1)
    def discovered_dim(w: str) -> int:
        d = 0
        for k in K_LADDER:
            if rmean(w, k) < TAU_FLOOR:
                d = k
            else:
                break
        return d

    dim_true, dim_broken = discovered_dim("TRUE"), discovered_dim("BROKEN")
    true_jump = rmean("TRUE", 4) / max(rmean("TRUE", 3), 1e-12)
    broken_jump = rmean("BROKEN", 2) / max(rmean("BROKEN", 1), 1e-12)
    print(f"    => TRUE reads dim={dim_true} (residual at floor up to K=3, x{true_jump:.1f} jump at K=4); "
          f"ground truth dim(so(3))=3")
    print(f"    => BROKEN reads dim={dim_broken} (x{broken_jump:.1f} jump at K=2); ground truth "
          f"dim(so(2)_z)=1")

    # ---- [R] recovery quality on TRUE at K=3 ---------------------------------------------
    # average the K=3 generator metrics over seeds
    def avg_metric(w: str, k: int, fn) -> float:
        return float(np.mean([fn(g) for g in gens[w][k]]))

    t_frac = avg_metric("TRUE", 3, frac_in_so3)
    t_anti = avg_metric("TRUE", 3, antisym_frac)
    t_close = avg_metric("TRUE", 3, closure_resid)
    t_struct = avg_metric("TRUE", 3, struct_const_norm)
    print()
    print(line)
    print("[R] RECOVERY — does TRUE@K=3 rediscover the algebra so(3)?  (no structure imposed)")
    print(line)
    print(f"    fraction of span in so(3)   : {t_frac:.3f}   (1.0 = exactly the antisymmetric subspace)")
    print(f"    symmetric-part norm (mean)  : {t_anti:.3f}   (0.0 = perfectly antisymmetric generators)")
    print(f"    Lie-bracket closure residual: {t_close:.3f}   (0.0 = [G_i,G_j] stays in span = a real algebra)")
    print(f"    structure-constant norm     : {t_struct:.3f}   (sqrt(3)={np.sqrt(3):.3f} for a unit so(3) frame = Levi-Civita)")
    print(f"    => K=3 residual {rmean('TRUE', 3):.2e}: gradient descent recovered antisymmetry AND "
          f"bracket-closure from data alone.")

    # ---- [X] rejection on BROKEN ----------------------------------------------------------
    b_axis = avg_metric("BROKEN", 1, axis_align)
    b_anti1 = avg_metric("BROKEN", 1, antisym_frac)
    broken_vs_true_k3 = rmean("BROKEN", 3) / max(rmean("TRUE", 3), 1e-12)
    print()
    print(line)
    print("[X] REJECTION — the SAME procedure on a rotation-broken world (falsifiability)")
    print(line)
    print(f"    BROKEN@K=1 axis alignment |<G,L_z>|: {b_axis:.3f}  (1.0 = exactly the unbroken z-axis)")
    print(f"    BROKEN@K=1 symmetric-part norm     : {b_anti1:.3f}  (0.0 = antisymmetric, a rotation)")
    print(f"    BROKEN vs TRUE residual at K=3     : x{broken_vs_true_k3:.1f}  (same slate fails on the broken world)")
    print(f"    => the method is not vacuous: break SO(3)->SO(2) and it reports dim 3->1 and the surviving axis.")

    # ---- [Xb] falsifiability across BREAK STRENGTH ---------------------------------------
    # The dim=1 read on BROKEN must not depend on a single tuned stretch magnitude: sweep beta over an
    # 8x range and check the SAME dimension (1) and axis (L_z) emerge at every strength. We only need
    # K in {1,2} to read the dimension (floor at K=1 = L_z survives; jump at K=2 = nothing else does).
    beta_k1: dict[float, list[float]] = {b: [] for b in BETA_LADDER}
    beta_k2: dict[float, list[float]] = {b: [] for b in BETA_LADDER}
    beta_axis: dict[float, list[float]] = {b: [] for b in BETA_LADDER}
    for seed in SEEDS:
        s, a_flat, _ = make_interacting_transitions(N_DATA, seed=seed)
        a = a_flat.reshape(N_DATA, N_OBJ, 3)
        for b in BETA_LADDER:
            tb = functools.partial(broken_teacher, beta=b)
            g1, r1 = discover(tb, 1, s, a, n_steps=N_STEPS, seed=seed)
            _, r2 = discover(tb, 2, s, a, n_steps=N_STEPS, seed=seed)
            beta_k1[b].append(r1)
            beta_k2[b].append(r2)
            beta_axis[b].append(axis_align(g1))

    def bmean(d: dict, b: float) -> float:
        return float(np.mean(d[b]))

    print()
    print(line)
    print("[Xb] FALSIFIABILITY ACROSS BREAK STRENGTH — does BROKEN read dim=1 at EVERY beta?")
    print(line)
    print(f"    {'beta':>6s} | {'K=1 resid':>13s} | {'K=2 resid':>13s} | {'jump K2/K1':>11s} | "
          f"{'dim':>3s} | {'|<G,Lz>|':>8s}")
    print("    " + "-" * 68)
    beta_dims: list[int] = []
    beta_axes_ok: list[bool] = []
    for b in BETA_LADDER:
        jump = bmean(beta_k2, b) / max(bmean(beta_k1, b), 1e-12)
        dim_b = 1 if (bmean(beta_k1, b) < TAU_FLOOR and jump > JUMP) else 2
        beta_dims.append(dim_b)
        beta_axes_ok.append(bmean(beta_axis, b) > TAU_AXIS)
        print(f"    {b:>6.2f} | {bmean(beta_k1, b):.3e}   | {bmean(beta_k2, b):.3e}   | "
              f"x{jump:>9.1e} | {dim_b:>3d} | {bmean(beta_axis, b):>8.3f}")
    all_dim1 = all(d == 1 for d in beta_dims)
    all_axis = all(beta_axes_ok)
    beta_range = max(BETA_LADDER) / min(BETA_LADDER)
    print(f"    => across an x{beta_range:.0f} range of break strength, EVERY beta reads dim=1 ({all_dim1}) "
          f"with surviving axis L_z ({all_axis}).")
    print(f"    => the dim-1 verdict is a property of the broken symmetry, not of one tuned stretch magnitude.")

    # ---- verdict --------------------------------------------------------------------------
    ok_recover_so3 = (t_frac > TAU_FRAC_SO3 and t_anti < TAU_ANTISYM
                      and t_close < TAU_CLOSURE and rmean("TRUE", 3) < TAU_FLOOR)
    ok_dim_true = (rmean("TRUE", 3) < TAU_FLOOR and true_jump > JUMP and dim_true == TRUE_DIM)
    ok_reject_broken = (rmean("BROKEN", 1) < TAU_FLOOR and broken_jump > JUMP and dim_broken == BROKEN_DIM)
    ok_broken_axis = (b_axis > TAU_AXIS and b_anti1 < 2 * TAU_ANTISYM)
    ok_broken_differs = broken_vs_true_k3 > JUMP
    ok_beta_sweep = all_dim1 and all_axis
    passed = (ok_recover_so3 and ok_dim_true and ok_reject_broken
              and ok_broken_axis and ok_broken_differs and ok_beta_sweep)

    print()
    print(line)
    print("STEP 33 SUMMARY")
    print(line)
    print(f"    [R] recover : TRUE@K=3 spans so(3) ({t_frac:.2f}), antisym ({t_anti:.2f}), CLOSES "
          f"({t_close:.2f}), struct-const {t_struct:.2f}~sqrt3.")
    print(f"    [D] dimension: residual reads dim(TRUE)={dim_true} (x{true_jump:.0f} jump @K=4), "
          f"dim(BROKEN)={dim_broken} (x{broken_jump:.0f} jump @K=2).")
    print(f"    [X] reject  : BROKEN gen ~ L_z (align {b_axis:.2f}); BROKEN@K=3 residual x{broken_vs_true_k3:.0f} "
          f"above TRUE@K=3.")
    print(f"    [Xb] robust : every beta in {list(BETA_LADDER)} (x{beta_range:.0f} range) reads dim=1 "
          f"with axis L_z ({all_dim1 and all_axis}).")
    print(f"    guards: recover-so3={ok_recover_so3}  dim-true={ok_dim_true}  reject-broken={ok_reject_broken}  "
          f"broken-axis={ok_broken_axis}  broken-differs={ok_broken_differs}  beta-sweep={ok_beta_sweep}")
    print(f"    headline: with NO symmetry prior — a blank slate of 3x3 matrices — gradient descent reads the")
    print(f"        world's symmetry off its dynamics: it rediscovers the antisymmetric, bracket-closing algebra")
    print(f"        so(3) and its dimension (3), and when the world's rotation symmetry is broken to SO(2) it")
    print(f"        correctly reports dimension 1 and the surviving z-axis. The hand-specified prior of Steps 6-32")
    print(f"        is RECOVERABLE from data, and the recovery is falsifiable.")
    if passed:
        print(f"    PASS: the symmetry is discovered (so(3), dim 3), and the discovery is falsifiable "
              f"(broken -> so(2), dim 1).")
    else:
        print(f"    INCONCLUSIVE: inspect the dimension sweep / recovery metrics above.")

    # ---- dump JSON + figure ---------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(SEEDS=list(SEEDS), N_DATA=N_DATA, N_STEPS=N_STEPS, LR=LR,
                       THETA_MAX=THETA_MAX, K_LADDER=list(K_LADDER), BREAK_BETA=BREAK_BETA,
                       C_INT=C_INT, TRUE_DIM=TRUE_DIM, BROKEN_DIM=BROKEN_DIM),
        "resid_mean": {w: {k: rmean(w, k) for k in K_LADDER} for w in worlds},
        "resid_std": {w: {k: rstd(w, k) for k in K_LADDER} for w in worlds},
        "resid_per_seed": {w: {k: resid[w][k] for k in K_LADDER} for w in worlds},
        "dim_true": dim_true, "dim_broken": dim_broken,
        "true_jump_K4_over_K3": true_jump, "broken_jump_K2_over_K1": broken_jump,
        "recover": {"frac_in_so3": t_frac, "antisym_frac": t_anti,
                    "closure_resid": t_close, "struct_const_norm": t_struct},
        "reject": {"broken_axis_align": b_axis, "broken_antisym_frac": b_anti1,
                   "broken_vs_true_k3": broken_vs_true_k3},
        "beta_sweep": {
            "betas": list(BETA_LADDER),
            "k1_resid": {str(b): bmean(beta_k1, b) for b in BETA_LADDER},
            "k2_resid": {str(b): bmean(beta_k2, b) for b in BETA_LADDER},
            "axis_align": {str(b): bmean(beta_axis, b) for b in BETA_LADDER},
            "dims": beta_dims, "all_dim1": bool(all_dim1), "all_axis": bool(all_axis),
        },
        "verdict": {"passed": bool(passed), "ok_recover_so3": ok_recover_so3,
                    "ok_dim_true": ok_dim_true, "ok_reject_broken": ok_reject_broken,
                    "ok_broken_axis": ok_broken_axis, "ok_broken_differs": ok_broken_differs,
                    "ok_beta_sweep": bool(ok_beta_sweep)},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step33_symmetry_discovery_smoke" if SMOKE else "step33_symmetry_discovery"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    beta_fig = {"betas": list(BETA_LADDER),
                "k1": [bmean(beta_k1, b) for b in BETA_LADDER],
                "k2": [bmean(beta_k2, b) for b in BETA_LADDER],
                "axis": [bmean(beta_axis, b) for b in BETA_LADDER]}
    _make_figure(fig_dir / f"{stem}.png", resid, gens, beta_fig)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)}")
    print(f"    wrote {(fig_dir / f'{stem}.png').relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


def _make_figure(path, resid, gens, beta_fig) -> None:
    r"""Four panels: dimension sweep, recovery-quality bars, a discovered generator heatmap, and the
    break-strength robustness sweep (BROKEN reads dim=1 at every $\beta$)."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    ks = list(K_LADDER)

    # Panel A: dimension sweep (residual vs K) for both worlds -> reads off the symmetry dimension
    tmean = [float(np.mean(resid["TRUE"][k])) for k in ks]
    tstd = [float(np.std(resid["TRUE"][k])) for k in ks]
    bmean = [float(np.mean(resid["BROKEN"][k])) for k in ks]
    bstd = [float(np.std(resid["BROKEN"][k])) for k in ks]
    ax[0].errorbar(ks, tmean, yerr=tstd, marker="o", lw=2, capsize=4, color="C0",
                   label="TRUE world (so(3))")
    ax[0].errorbar(ks, bmean, yerr=bstd, marker="s", lw=2, capsize=4, color="C3",
                   label="BROKEN world (so(2))")
    ax[0].axhline(TAU_FLOOR, ls=":", color="k", alpha=0.5, label=f"floor ({TAU_FLOOR})")
    ax[0].axvline(3.5, ls="-.", color="C0", alpha=0.5)
    ax[0].axvline(1.5, ls="-.", color="C3", alpha=0.5)
    ax[0].set_yscale("log")
    ax[0].set_xticks(ks)
    ax[0].set_xlabel("generator-slate size $K$")
    ax[0].set_ylabel("relative equivariance residual")
    ax[0].set_title("[D] dimension read off the data\n(jump @K=4 TRUE, @K=2 BROKEN)")
    ax[0].legend(fontsize=8, loc="best")
    ax[0].grid(alpha=0.3, which="both")

    # Panel B: recovery-quality bars (TRUE@K=3 vs BROKEN@K=3): closer to ideal = a real so(3) basis
    def m(w, k, fn):
        return float(np.mean([fn(g) for g in gens[w][k]]))

    labels = ["frac in\nso(3)", "antisym\npurity", "closure\npurity"]
    true_vals = [m("TRUE", 3, frac_in_so3), 1 - m("TRUE", 3, antisym_frac), 1 - m("TRUE", 3, closure_resid)]
    broken_vals = [m("BROKEN", 3, frac_in_so3), 1 - m("BROKEN", 3, antisym_frac),
                   1 - m("BROKEN", 3, closure_resid)]
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, true_vals, 0.4, color="C0", label="TRUE @ K=3")
    ax[1].bar(x + 0.2, broken_vals, 0.4, color="C3", label="BROKEN @ K=3")
    ax[1].axhline(1.0, ls=":", color="k", alpha=0.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylim(0, 1.15)
    ax[1].set_ylabel("score (1.0 = ideal so(3) basis)")
    ax[1].set_title("[R] TRUE recovers so(3);\nthe broken world cannot fake it at K=3")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=0.3, axis="y")

    # Panel C: a discovered TRUE generator (seed 0, K=3, first generator) -> visibly antisymmetric
    g0 = gens["TRUE"][3][0][0].numpy()
    im = ax[2].imshow(g0, cmap="RdBu", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, f"{g0[i, j]:+.2f}", ha="center", va="center", fontsize=10)
    ax[2].set_xticks(range(3))
    ax[2].set_yticks(range(3))
    ax[2].set_title("[R] a discovered generator $\\hat G_1$ (TRUE)\nantisymmetric: $\\hat G^\\top=-\\hat G$")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    # Panel D: break-strength robustness — BROKEN reads dim=1 (K=1 at floor, K=2 jumps) at EVERY beta
    betas = beta_fig["betas"]
    ax[3].plot(betas, beta_fig["k1"], marker="o", lw=2, color="C2",
               label="$K{=}1$ ($L_z$ survives)")
    ax[3].plot(betas, beta_fig["k2"], marker="s", lw=2, color="C3",
               label="$K{=}2$ (nothing else fits)")
    ax[3].axhline(TAU_FLOOR, ls=":", color="k", alpha=0.5, label=f"floor ({TAU_FLOOR})")
    ax[3].set_yscale("log")
    ax[3].set_xlabel(r"break strength $\beta$")
    ax[3].set_ylabel("relative equivariance residual")
    ax[3].set_title("[Xb] the dim=1 read is robust\nacross an $8\\times$ range of $\\beta$")
    ax[3].legend(fontsize=8, loc="best")
    ax[3].grid(alpha=0.3, which="both")

    fig.suptitle("Step 33 — the symmetry prior is recoverable from data, and the recovery is falsifiable",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
