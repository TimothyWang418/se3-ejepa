r"""Step 29 (3D arm): the controlled scaling sweep in $\mathrm{SO}(3)$ -- does augmentation $+$
brute-force scale substitute for the equivariant architecture, when the missing orientations are
genuine extrapolation?

The 2D arm (``experiments/step29_scaling_sweep.py``) held coverage partial at a half-circle
$[0,180^\circ)$ and showed: scaling MLP width $\times16$ and data $\times16$ does **not** shrink the
across-group *task* gap (it is *extrapolation*, not interpolation -- scale $\neq$ coverage), and the
*exactness* $\Delta_{\mathrm{eq}}$ sits on a **scale-independent plateau** orders of magnitude above the
Vector-Neuron float floor. This arm reruns the identical sweep in the richer group $\mathrm{SO}(3)$
(3 rotational DoF, not 1), where the Bitter-Lesson rejoinder -- "the 1-D circle was too easy; in 3D scale
will catch up" -- has its best chance.

What "partial coverage" means in 3D
-----------------------------------
In 2D the covered region is an arc $[0,\theta_{\max})$; the uncovered arc is the extrapolation set. In 3D
we use the natural analogue: a **geodesic ball** of rotation angle $\le\theta_{\max}$ about a *random axis*
(axis uniform on $S^2$, angle $\sim U[0,\theta_{\max}]$). We hold $\theta_{\max}=90^\circ$ -- a partial
ball -- so the **complement shell** (angle $\in(90^\circ,180^\circ]$, any axis) is entirely uncovered and
is *pure extrapolation*. The seen $z$-wedge $[0,90^\circ)$ lies inside the ball (its angle is $\le90^\circ$),
so the OOD/seen ratio's denominator is genuinely in-distribution -- exactly as in the 2D arm.

The two scale axes (identical to the 2D arm)
--------------------------------------------
  * **model size**  -- MLP hidden width $\in\{64,256,1024\}$ ($\approx\!2\times,11\times,160\times$ the VN's
    $3.5$k params in 3D);
  * **data**        -- number of base scenes $N\in\{256,1024,4096\}$, each re-rotated by fresh
    $\mathrm{SO}(3)$ rotations from the ball every gradient step (orientation data effectively infinite
    within coverage; $N$ scales *content* diversity).

Two metrics, two predictions (identical to Step 28 / the 2D arm)
----------------------------------------------------------------
  [1] **task metric** -- OOD/seen relMSE ratio, OOD = the uncovered shell (extrapolation). If the gap is
      missing coverage, neither bigger models nor more data close it: it stays far above the VN's
      $\times1.00$ across the whole grid.
  [2] **exactness** -- $\Delta_{\mathrm{eq}}=\max_R\lVert f(Rx)-Rf(x)\rVert/\lVert f(x)\rVert$ over random
      global $\mathrm{SO}(3)$ probes. Expectation: a **scale-independent plateau** orders of magnitude
      above the VN's weight-*independent* float floor.

A VN of fixed size is trained at each data scale as the reference: $\times1.00$ and at the float floor
regardless of $N$ -- the architecture's guarantee is scale-free.

Training budget (fair across the data axis)
-------------------------------------------
Minibatch AdamW with a **fixed gradient-step budget** (batch $256$): every cell sees the same number of
updates, so the data axis varies *content diversity* at constant optimisation budget. At $N{=}\mathrm{batch}$
a step is full-batch-with-fresh-rotations, exactly Step 28's recipe.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step29_scaling_sweep_3d.py
Smoke (~60 s):
    STEP29_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step29_scaling_sweep_3d.py
"""

import json
import math
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
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models.structured import VNLinear, VNReLU  # noqa: E402

torch.set_default_dtype(torch.float32)

N_STATE = 6
DIM = 3
SMOKE = bool(os.environ.get("STEP29_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- step8/step28 lifted to dim=3 (VN layers are dimension-agnostic and
# exactly equivariant in 3D); MLP width is the swept axis.
# --------------------------------------------------------------------------- #
class EquivariantWorld(nn.Module):
    r"""Frozen random *world*: an exactly $\mathrm{SO}(3)$-equivariant one-step map (one VN nonlinearity).

    ``forward: (B, n_state, 3), (B, 1, 3) -> (B, n_state, 3)`` with $T(Rs, Ra) = R\,T(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 8):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 3)
        return self.l2(self.a1(self.l1(x)))


class VNDynamics(nn.Module):
    r"""Equivariant **student** (dim=3, fixed size, $\sim\!3.5$k params); symmetry hard-wired and scale-free.

    ``forward: (B, n_state, 3), (B, 1, 3) -> (B, n_state, 3)`` and $T_\theta(Rs, Ra)=R\,T_\theta(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 32):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, hidden)
        self.a2 = VNReLU(hidden)
        self.l3 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 3)
        return self.l3(self.a2(self.l2(self.a1(self.l1(x)))))


class MLPDynamics(nn.Module):
    r"""Non-equivariant **baseline** on the flattened 3D state+action; ``hidden`` is the scale knob."""

    def __init__(self, n_state: int = N_STATE, hidden: int = 128):
        super().__init__()
        self.n_state = n_state
        in_dim, out_dim = n_state * DIM + DIM, n_state * DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        x = torch.cat([s.reshape(b, -1), a.reshape(b, -1)], dim=1)
        return self.net(x).reshape(b, self.n_state, DIM)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# SO(3) geometry -- mirrors step28_fair_augmentation_3d.py (self-contained).
# --------------------------------------------------------------------------- #
def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    r"""Batched skew-symmetric matrices $[\,\hat n\,]_\times$. ``(n, 3) -> (n, 3, 3)``."""
    n = v.shape[0]
    K = torch.zeros(n, 3, 3, dtype=v.dtype)
    K[:, 0, 1] = -v[:, 2]; K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2];  K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]; K[:, 2, 1] = v[:, 0]
    return K


def axis_angle_batch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    r"""Batched Rodrigues: $R=\mathbf I+\sin\theta\,[\hat n]_\times+(1-\cos\theta)[\hat n]_\times^2$.

    ``axis: (n, 3), angle: (n,) -> (n, 3, 3)``.
    """
    nrm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K = _skew_batch(axis / nrm)
    eye = torch.eye(3, dtype=axis.dtype).expand(axis.shape[0], 3, 3)
    s = torch.sin(angle).view(-1, 1, 1)
    c = (1.0 - torch.cos(angle)).view(-1, 1, 1)
    return eye + s * K + c * (K @ K)


def rand_so3(gen: torch.Generator) -> torch.Tensor:
    r"""A single random rotation (axis uniform, angle $\sim U[0,2\pi)$). ``-> (3, 3)``."""
    axis = torch.randn(3, generator=gen)
    angle = torch.full((1,), 2.0 * math.pi * torch.rand((), generator=gen).item())
    return axis_angle_batch(axis.unsqueeze(0), angle)[0]


def rot_z_batch(phi: torch.Tensor) -> torch.Tensor:
    r"""Per-sample rotation about $+z$ by ``phi`` (radians). ``(n,) -> (n, 3, 3)``."""
    z = torch.tensor([0.0, 0.0, 1.0]).expand(phi.shape[0], 3)
    return axis_angle_batch(z, phi)


def rand_so3_cap_batch(n: int, theta_max: float, gen: torch.Generator) -> torch.Tensor:
    r"""``n`` random rotations in the **geodesic ball** of radius ``theta_max`` (radians).

    Axis uniform on $S^2$ (normalised Gaussian), angle $\sim U[0,\theta_{\max}]$. ``-> (n, 3, 3)``.
    """
    axis = torch.randn(n, 3, generator=gen)
    angle = torch.rand(n, generator=gen) * theta_max
    return axis_angle_batch(axis, angle)


def rand_so3_shell_batch(n: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    r"""``n`` random rotations in the **complement shell**: angle $\sim U[\text{lo},\text{hi}]$, axis uniform.

    With ``lo`` $=\theta_{\max}$ and ``hi`` $=\pi$ this is the set of rotations strictly *outside* the
    coverage ball of radius $\theta_{\max}$ -- the pure-extrapolation OOD set. ``-> (n, 3, 3)``.
    """
    axis = torch.randn(n, 3, generator=gen)
    angle = torch.rand(n, generator=gen) * (hi - lo) + lo
    return axis_angle_batch(axis, angle)


def rotate_ps(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Per-sample rotation. ``x: (n, c, 3), R: (n, 3, 3) -> (n, c, 3)``."""
    return torch.einsum("nij,ncj->nci", R, x)


def rotate_g(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Global rotation by a single ``R``. ``x: (..., 3), R: (3, 3) -> (..., 3)``."""
    return x @ R.transpose(-1, -2)


# --------------------------------------------------------------------------- #
# Metrics (= step28 3D)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rel_mse(model: nn.Module, s, a, y) -> float:
    err = (model(s, a) - y).pow(2).mean().item()
    return err / (y.pow(2).mean().item() + 1e-12)


@torch.no_grad()
def equiv_residual(model: nn.Module, s, a, Rs) -> float:
    r"""Worst-case relative equivariance residual over a set of $\mathrm{SO}(3)$ probe rotations.

    $\Delta_{\mathrm{eq}} = \max_R \lVert f(Rs, Ra) - R\,f(s,a)\rVert_F / \lVert f(s,a)\rVert_F$.
    """
    base = model(s, a)
    den = base.pow(2).sum().item() + 1e-12
    worst = 0.0
    for R in Rs:
        lhs = model(rotate_g(s, R), rotate_g(a, R))
        rhs = rotate_g(base, R)
        worst = max(worst, ((lhs - rhs).pow(2).sum().item() / den) ** 0.5)
    return worst


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_full(model: nn.Module, s, a, y, *, steps=2000, lr=3e-3, wd=1e-4) -> nn.Module:
    r"""Full-batch AdamW on a FIXED dataset (the VN reference; data is the thin $z$-wedge)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = (model(s, a) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


def train_augmented_mb(
    model: nn.Module, s_base, a_base, target_fn, theta_max: float,
    *, steps: int, batch: int, lr=3e-3, wd=1e-4, gen: torch.Generator,
) -> nn.Module:
    r"""Minibatch AdamW with on-the-fly $\mathrm{SO}(3)$ ball augmentation, **fixed step budget**.

    Each step samples ``batch`` base scenes (uniformly, with replacement when ``batch`` > $N$), rotates them
    by fresh per-sample rotations from the geodesic ball (angle $\le\theta_{\max}$, random axis), recomputes
    the exact equivariant target, and takes one AdamW step. The step budget is fixed across data scales, so
    $N$ varies *content diversity* at constant optimisation budget.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        sb, ab = s_base[idx], a_base[idx]
        R = rand_so3_cap_batch(batch, theta_max, gen)
        s_aug, a_aug = rotate_ps(sb, R), rotate_ps(ab, R)
        with torch.no_grad():
            y_aug = target_fn(s_aug, a_aug)
        opt.zero_grad()
        loss = (model(s_aug, a_aug) - y_aug).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


# --------------------------------------------------------------------------- #
# Evaluation: seen = z-wedge [0,90) (inside the ball); OOD = complement shell (theta_cov,180] = extrapolation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_so3_extrap(model, target_fn, s_te0, a_te0, theta_cov: float, gen: torch.Generator):
    r"""Return ``(seen_relMSE, ood_relMSE, ratio)`` for the partial-coverage extrapolation test.

    ``seen`` rotates the canonical test scenes into the $z$-wedge $[0,90^\circ)$ (angle $\le90^\circ$,
    inside the coverage ball). ``ood`` rotates them by per-sample rotations from the complement shell
    (angle $\in(\theta_{\mathrm{cov}},180^\circ]$, random axis) -- strictly outside coverage, pure
    extrapolation. Content is fixed; only orientation moves.
    """
    n = s_te0.shape[0]
    # seen: z-wedge, inside the ball
    phi = torch.rand(n, generator=gen) * (math.pi / 2)
    Rz = rot_z_batch(phi)
    s_s, a_s = rotate_ps(s_te0, Rz), rotate_ps(a_te0, Rz)
    seen = rel_mse(model, s_s, a_s, target_fn(s_s, a_s))
    # ood: complement shell, outside the ball (extrapolation)
    R = rand_so3_shell_batch(n, theta_cov, math.pi, gen)
    s_o, a_o = rotate_ps(s_te0, R), rotate_ps(a_te0, R)
    ood = rel_mse(model, s_o, a_o, target_fn(s_o, a_o))
    return seen, ood, ood / (seen + 1e-12)


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)


# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 78
    torch.manual_seed(0)

    COVERAGE_DEG = 90  # partial coverage held fixed: a radius-90 ball; (90,180] shell is extrapolation
    if SMOKE:
        SEEDS, SIZES, DATAS, STEPS, BATCH, N_TEST = [0, 1], [64, 256], [256, 1024], 400, 128, 400
    else:
        SEEDS, SIZES, DATAS, STEPS, BATCH, N_TEST = [0, 1, 2, 3, 4], [64, 256, 1024], [256, 1024, 4096], 2000, 256, 2000

    print(line)
    print(f"STEP 29 (3D)  controlled scaling sweep in SO(3): can size x data substitute for the prior?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    coverage held PARTIAL at ball<={COVERAGE_DEG} deg -> shell ({COVERAGE_DEG},180] is "
          f"EXTRAPOLATION; sweep MLP width x #scenes; {len(SEEDS)} seeds")

    # --- frozen, exactly SO(3)-equivariant teacher (= step28 3D) --------------------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, DIM), torch.randn(8000, 1, DIM)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()

    def target(s, a):
        with torch.no_grad():
            return teacher(s, a) / rms

    probe_gen = torch.Generator().manual_seed(11)
    PROBE_R = [rand_so3(probe_gen) for _ in range(5)]
    s_chk, a_chk = torch.randn(64, N_STATE, DIM), torch.randn(64, 1, DIM)
    teacher_eq = equiv_residual(lambda s, a: target(s, a), s_chk, a_chk, PROBE_R)
    assert teacher_eq < 1e-4, "teacher world is not SO(3)-equivariant"

    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, DIM) * 1.5
    mu_a = torch.randn(1, DIM) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, DIM, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, DIM, generator=gen)
        return s, a

    vn_p = n_params(VNDynamics())
    mlp_ps = {h: n_params(MLPDynamics(hidden=h)) for h in SIZES}
    print(f"    VN(equivariant, fixed)={vn_p} params; "
          f"MLP sizes={ {h: mlp_ps[h] for h in SIZES} } "
          f"({ {h: round(mlp_ps[h] / vn_p, 1) for h in SIZES} }x VN)")
    print(f"    budget: {STEPS} AdamW steps, batch {BATCH}, coverage ball<={COVERAGE_DEG} deg\n")

    cov = COVERAGE_DEG * math.pi / 180.0

    # grid[(h, N)] -> lists over seeds; vn_ref[N] -> lists over seeds
    grid = {(h, N): {"ratio": [], "eq": [], "seen": [], "ood": []} for h in SIZES for N in DATAS}
    vn_ref = {N: {"ratio": [], "eq": [], "seen": []} for N in DATAS}

    for sd in SEEDS:
        data_gen = torch.Generator().manual_seed(2000 + sd)
        aug_gen = torch.Generator().manual_seed(3000 + sd)
        test_gen = torch.Generator().manual_seed(4000 + sd)

        # held-out canonical test + probe scenes (shared across all cells this seed)
        s_te0, a_te0 = aniso_inputs(N_TEST, test_gen)
        s_pr, a_pr = aniso_inputs(min(N_TEST, 1000), test_gen)

        # per-data-scale base scenes; VN reference trained on the thin z-wedge [0,90)
        for N in DATAS:
            s_c, a_c = aniso_inputs(N, data_gen)

            # VN reference (fixed size): z-wedge-only, full-batch; scale-free by construction
            phi_w = torch.rand(N, generator=data_gen) * (math.pi / 2)
            Rz_w = rot_z_batch(phi_w)
            s_w, a_w = rotate_ps(s_c, Rz_w), rotate_ps(a_c, Rz_w)
            torch.manual_seed(10 + sd)
            vn = train_full(VNDynamics(), s_w, a_w, target(s_w, a_w), steps=STEPS)
            seen, ood, ratio = eval_so3_extrap(vn, target, s_te0, a_te0, cov, test_gen)
            vn_ref[N]["ratio"].append(ratio); vn_ref[N]["seen"].append(seen)
            vn_ref[N]["eq"].append(equiv_residual(vn, s_pr, a_pr, PROBE_R))

            # MLP + partial-coverage SO(3) augmentation, swept over size
            for h in SIZES:
                torch.manual_seed(30 + sd)  # paired init across sizes for fixed (N, seed)
                mlp = train_augmented_mb(MLPDynamics(hidden=h), s_c, a_c, target, cov,
                                         steps=STEPS, batch=BATCH, gen=aug_gen)
                seen, ood, ratio = eval_so3_extrap(mlp, target, s_te0, a_te0, cov, test_gen)
                g = grid[(h, N)]
                g["ratio"].append(ratio); g["seen"].append(seen)
                g["ood"].append(ood); g["eq"].append(equiv_residual(mlp, s_pr, a_pr, PROBE_R))

    def summ(d, keys=("ratio", "eq", "seen", "ood")):
        return {k: mean_std(d[k]) for k in keys if k in d}

    grid_s = {hn: summ(grid[hn]) for hn in grid}
    vn_s = {N: summ(vn_ref[N], keys=("ratio", "eq", "seen")) for N in DATAS}

    # --- [1] task metric grid -------------------------------------------------------------
    print(line)
    print("[1] TASK METRIC -- OOD/seen relMSE ratio  (OOD = shell (90,180] = extrapolation; 1.00 = flat)")
    print(line)
    header = "    size\\data |" + "".join(f"{('N=' + str(N)):>14s}" for N in DATAS)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for h in SIZES:
        row = f"    h={h:>5d}  |"
        for N in DATAS:
            r = grid_s[(h, N)]["ratio"]
            row += f"  x{r[0]:7.2f}+-{r[1]:5.1f}"
        print(row)
    row = f"    {'VN ref':>8s}  |"
    for N in DATAS:
        row += f"  x{vn_s[N]['ratio'][0]:7.3f}      "
    print(row)
    small = grid_s[(SIZES[0], DATAS[0])]["ratio"][0]
    big = grid_s[(SIZES[-1], DATAS[-1])]["ratio"][0]
    vn_flat = max(vn_s[N]["ratio"][0] for N in DATAS)
    print(f"    => smallest cell x{small:.2f} -> largest cell x{big:.2f}; "
          f"VN flat at x{vn_flat:.3f} regardless of N.")

    # --- [2] exactness grid ---------------------------------------------------------------
    print()
    print(line)
    print("[2] EXACTNESS -- residual equivariance Delta_eq = max_R ||f(Rx)-Rf(x)|| / ||f(x)|| over SO(3)")
    print(line)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for h in SIZES:
        row = f"    h={h:>5d}  |"
        for N in DATAS:
            row += f"   {grid_s[(h, N)]['eq'][0]:9.2e} "
        print(row)
    row = f"    {'VN ref':>8s}  |"
    for N in DATAS:
        row += f"   {vn_s[N]['eq'][0]:9.2e} "
    print(row)
    vn_floor = max(vn_s[N]["eq"][0] for N in DATAS)
    best_mlp_eq = min(grid_s[(h, N)]["eq"][0] for h in SIZES for N in DATAS)  # closest aug ever gets
    eq_factor = best_mlp_eq / max(vn_floor, 1e-12)
    print(f"    => the BEST (most-equivariant) MLP cell is Delta_eq={best_mlp_eq:.2e} "
          f"= x{eq_factor:.0f} the VN floor ({vn_floor:.1e}); scale never reaches exactness.")

    # --- verdict --------------------------------------------------------------------------
    print()
    print(line)
    print("STEP 29 (3D) SUMMARY")
    print(line)
    partial_gap = small > 3.0                      # partial coverage leaves a real extrapolation gap
    exact_plateau = best_mlp_eq > 50.0 * vn_floor  # no cell reaches exactness
    task_helps = big < small                       # did scale shrink the task gap at all?
    task_closes = big < 1.5                         # did scale CLOSE it to VN level?

    print(f"    [1] task: at partial coverage the gap is EXTRAPOLATION -- scaling size x{SIZES[-1]//SIZES[0]} "
          f"and data x{DATAS[-1]//DATAS[0]} moves it x{small:.1f} -> x{big:.1f} "
          f"({'shrinks but does NOT close' if (task_helps and not task_closes) else ('CLOSES' if task_closes else 'does not shrink')}); "
          f"VN x{vn_flat:.2f} for free.")
    print(f"    [2] exactness: a SCALE-INDEPENDENT plateau -- the best MLP cell is still "
          f"x{eq_factor:.0f} the VN floor; no amount of size/data reaches exact equivariance.")
    print(f"    headline (matches the 2D arm): scale is not a substitute for the missing coverage (task) "
          f"nor for the architecture (exactness). The prior gives both, free and scale-independent.")

    assert vn_flat < 1.5, "VN must stay ~flat across SO(3) at every data scale"
    assert vn_floor < 1e-4, "VN must stay exactly SO(3)-equivariant (structural) at every data scale"
    assert partial_gap, "partial coverage must leave a real extrapolation gap to study"
    assert exact_plateau, "no (size,data) cell may reach the architecture's exact equivariance"
    print(f"\n    guards: vn-flat={vn_flat < 1.5}  vn-exact={vn_floor < 1e-4}  "
          f"partial-gap={partial_gap}  exactness-plateau={exact_plateau}  "
          f"task-shrinks={task_helps}  task-closes={task_closes}")
    print("    PASS: in SO(3) too, scale does not buy what the architecture is.")

    # --- dump JSON ------------------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "group": "SO(3)",
        "config": dict(seeds=SEEDS, sizes=SIZES, datas=DATAS, steps=STEPS, batch=BATCH,
                       coverage_deg=COVERAGE_DEG, n_test=N_TEST, n_state=N_STATE),
        "params": {"VN": vn_p, "MLP": {str(h): mlp_ps[h] for h in SIZES}},
        "grid": {f"{h}x{N}": grid_s[(h, N)] for h in SIZES for N in DATAS},
        "vn_ref": {str(N): vn_s[N] for N in DATAS},
        "derived": {"best_mlp_eq": best_mlp_eq, "vn_floor": vn_floor,
                    "eq_factor_best_vs_vn": eq_factor,
                    "ratio_small": small, "ratio_big": big,
                    "partial_gap": bool(partial_gap), "exact_plateau": bool(exact_plateau),
                    "task_helps": bool(task_helps), "task_closes": bool(task_closes)},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step29_scaling_sweep_3d_smoke.json" if SMOKE else "step29_scaling_sweep_3d.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")

    # --- figure (guarded) -----------------------------------------------------------------
    if not SMOKE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [mlp_ps[h] for h in SIZES]
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
            for N in DATAS:
                axL.plot(xs, [grid_s[(h, N)]["ratio"][0] for h in SIZES], "o-", label=f"MLP+aug, N={N}")
            axL.axhline(vn_flat, color="C2", ls="--", label="VN (exact, any N)")
            axL.set_xscale("log"); axL.set_xlabel("MLP parameters")
            axL.set_ylabel("OOD/seen relMSE ratio")
            axL.set_title(f"[1] task: scale vs the SO(3) extrapolation gap (ball<={COVERAGE_DEG} deg)")
            axL.legend(fontsize=8); axL.grid(alpha=0.3)
            for N in DATAS:
                axR.plot(xs, [grid_s[(h, N)]["eq"][0] for h in SIZES], "o-", label=f"MLP+aug, N={N}")
            axR.axhline(vn_floor, color="C2", ls="--", label="VN (exact floor)")
            axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("MLP parameters")
            axR.set_ylabel("residual equivariance  Delta_eq")
            axR.set_title("[2] exactness: scale-independent plateau (SO(3))")
            axR.legend(fontsize=8); axR.grid(alpha=0.3)
            fig.tight_layout()
            fig_path = fig_dir / "step29_scaling_sweep_3d.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"    wrote {fig_path.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    (figure skipped: {e})")

    sys.exit(0)


if __name__ == "__main__":
    main()
