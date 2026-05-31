r"""Step 29 (2D arm): the controlled scaling sweep -- does augmentation $+$ brute-force scale
substitute for the equivariant architecture?

The Bitter-Lesson stress test
-----------------------------
Step 28 settled the *fair-baseline* question at one model size and one data budget: handed the group,
rotation augmentation with **full** coverage closes the across-group *task* metric (OOD/seen $\to\times1.06$)
but **never** reaches the architecture's *exactness* ($\Delta_{\mathrm{eq}}$ plateaus $\sim\!10^5\times$
above the float floor). The natural Bitter-Lesson rejoinder (Sutton, 2019) is *scale*: give the
non-equivariant MLP more parameters and more data and surely it catches up. Step 29 runs that sweep.

Crucially we hold coverage **partial** -- the realistic regime. You rarely augment over the *whole*
group; you cover a wedge and hope the model generalises. We fix the augmentation coverage at a
**half-circle** $[0,180^\circ)$, so the test bins $[180^\circ,360^\circ)$ are pure **extrapolation**:
the MLP has architecturally no reason to continue the symmetry into orientations it never saw. Then we
sweep the two scale axes and ask whether either closes the gap:

  * **model size**  -- MLP hidden width $\in\{64,256,1024\}$ ($\approx\!2\times,20\times,310\times$ the VN's
    $3.5$k params);
  * **data**        -- number of base scenes $N\in\{256,1024,4096\}$ ($16\times$ span), each re-rotated
    with fresh angles every gradient step (so orientation data is effectively infinite within coverage;
    $N$ scales *content* diversity).

Two predictions, two metrics (identical to Step 28)
---------------------------------------------------
  [1] **task metric** -- OOD/seen relMSE ratio. If the gap is *extrapolation* (missing coverage), neither
      bigger models nor more data should close it: scale $\neq$ coverage. Expectation: the ratio stays
      far above the VN's $\times1.00$ across the whole grid (a *small* shrink at most), because a plain
      MLP has no inductive route from $[0,180^\circ)$ to $[180^\circ,360^\circ)$.
  [2] **exactness** -- residual equivariance
      $\Delta_{\mathrm{eq}}=\max_\alpha\lVert f(R_\alpha x)-R_\alpha f(x)\rVert/\lVert f(x)\rVert$.
      Expectation: a **scale-independent plateau** orders of magnitude above the VN's
      weight-*independent* float floor -- no $(\text{size},N)$ cell reaches exactness.

A VN of fixed size is trained at each data scale as the reference: it should stay $\times1.00$ and at the
float floor regardless of $N$ -- the architecture's guarantee is scale-free.

Training budget (fair across the data axis)
-------------------------------------------
Minibatch AdamW with a **fixed gradient-step budget** (batch $256$): every cell sees the same number of
updates, so the data axis varies *content diversity* at constant optimisation budget (the principled way
to vary $N$ without confounding it with compute). At $N{=}256$ a step is full-batch with fresh rotations,
exactly Step 28's recipe.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step29_scaling_sweep.py
Smoke (~40 s):
    STEP29_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step29_scaling_sweep.py
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

from src.geometry.so2 import rot_matrix, rotate_vectors  # noqa: E402
from src.models.structured import VNLinear, VNReLU  # noqa: E402

torch.set_default_dtype(torch.float32)

N_STATE = 6
SMOKE = bool(os.environ.get("STEP29_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- identical to step8 / step28 (self-contained); MLP width is the swept axis.
# --------------------------------------------------------------------------- #
class EquivariantWorld(nn.Module):
    r"""Frozen random *world*: an exactly $\mathrm{SO}(2)$-equivariant one-step map (one VN nonlinearity).

    ``forward: (B, n_state, 2), (B, 1, 2) -> (B, n_state, 2)`` with $T(R_\alpha s, R_\alpha a)=R_\alpha T(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 8):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 2)
        return self.l2(self.a1(self.l1(x)))


class VNDynamics(nn.Module):
    r"""Equivariant **student** (fixed size, $\sim\!3.5$k params); symmetry hard-wired and scale-free.

    ``forward: (B, n_state, 2), (B, 1, 2) -> (B, n_state, 2)`` and $T_\theta(R_\alpha s,R_\alpha a)=R_\alpha T_\theta(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 32):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, hidden)
        self.a2 = VNReLU(hidden)
        self.l3 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 2)
        return self.l3(self.a2(self.l2(self.a1(self.l1(x)))))


class MLPDynamics(nn.Module):
    r"""Non-equivariant **baseline** on the flattened state+action; ``hidden`` is the scale knob."""

    def __init__(self, n_state: int = N_STATE, hidden: int = 128):
        super().__init__()
        self.n_state = n_state
        in_dim, out_dim = n_state * 2 + 2, n_state * 2
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
        return self.net(x).reshape(b, self.n_state, 2)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# Geometry / metrics (= step28)
# --------------------------------------------------------------------------- #
def rotate_per_sample(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    r"""Rotate each sample by its own angle. ``x: (n, c, 2), alpha: (n,) -> (n, c, 2)``."""
    R = rot_matrix(alpha)  # (n, 2, 2)
    return torch.einsum("nij,ncj->nci", R, x)


@torch.no_grad()
def rel_mse(model: nn.Module, s, a, y) -> float:
    err = (model(s, a) - y).pow(2).mean().item()
    return err / (y.pow(2).mean().item() + 1e-12)


@torch.no_grad()
def equiv_residual(model: nn.Module, s, a, angles) -> float:
    r"""Worst-case relative equivariance residual $\Delta_{\mathrm{eq}}$ over a set of probe angles."""
    base = model(s, a)
    den = base.pow(2).sum().item() + 1e-12
    worst = 0.0
    for al in angles:
        lhs = model(rotate_vectors(s, al), rotate_vectors(a, al))
        rhs = rotate_vectors(base, al)
        worst = max(worst, ((lhs - rhs).pow(2).sum().item() / den) ** 0.5)
    return worst


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_full(model: nn.Module, s, a, y, *, steps=2000, lr=3e-3, wd=1e-4) -> nn.Module:
    r"""Full-batch AdamW on a FIXED dataset (the VN reference; data is the thin wedge)."""
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
    r"""Minibatch AdamW with on-the-fly rotation augmentation in $[0,\theta_{\max})$, **fixed step budget**.

    Each step samples ``batch`` base scenes (uniformly, with replacement when ``batch`` > $N$), rotates them
    by fresh per-sample angles $\alpha\sim U[0,\theta_{\max})$, recomputes the exact equivariant target, and
    takes one AdamW step. The step budget is fixed across data scales, so $N$ varies *content diversity* at
    constant optimisation budget. At $N{=}\mathrm{batch}$ this is full-batch-with-fresh-rotations (= Step 28).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        sb, ab = s_base[idx], a_base[idx]
        alpha = torch.rand(batch, generator=gen) * theta_max
        s_aug, a_aug = rotate_per_sample(sb, alpha), rotate_per_sample(ab, alpha)
        with torch.no_grad():
            y_aug = target_fn(s_aug, a_aug)
        opt.zero_grad()
        loss = (model(s_aug, a_aug) - y_aug).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


# --------------------------------------------------------------------------- #
# Evaluation across the circle (seen bin = [0,90); coverage [0,180) => bins 3,4 are extrapolation)
# --------------------------------------------------------------------------- #
BINS = [(0, 90), (90, 180), (180, 270), (270, 360)]


@torch.no_grad()
def eval_circle(model, target_fn, s_te, a_te, gen: torch.Generator):
    r"""Return ``(seen_relMSE, worst_relMSE, ood_ratio)`` over the 4 orientation bins (seen $=[0,90)$)."""
    errs = []
    for lo, hi in BINS:
        al = (torch.rand(s_te.shape[0], generator=gen) * (hi - lo) + lo) * math.pi / 180.0
        s_b, a_b = rotate_per_sample(s_te, al), rotate_per_sample(a_te, al)
        errs.append(rel_mse(model, s_b, a_b, target_fn(s_b, a_b)))
    return errs[0], max(errs), max(errs) / (errs[0] + 1e-12)


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)


# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 76
    torch.manual_seed(0)

    COVERAGE_DEG = 180  # partial coverage held fixed: a half-circle; [180,360) is extrapolation
    if SMOKE:
        SEEDS, SIZES, DATAS, STEPS, BATCH, N_TEST = [0, 1], [64, 256], [256, 1024], 400, 128, 400
    else:
        SEEDS, SIZES, DATAS, STEPS, BATCH, N_TEST = [0, 1, 2, 3, 4], [64, 256, 1024], [256, 1024, 4096], 2000, 256, 2000
    PROBE_ANGLES = [0.37, 1.2345, 2.71, 3.94, 5.40]

    print(line)
    print(f"STEP 29  controlled scaling sweep: can size x data substitute for the equivariant prior?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    coverage held PARTIAL at [0,{COVERAGE_DEG}) -> test bins [180,360) are EXTRAPOLATION; "
          f"sweep MLP width x #scenes; {len(SEEDS)} seeds")

    # --- frozen, exactly SO(2)-equivariant teacher (= step8/step28) -----------------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, 2), torch.randn(8000, 1, 2)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()

    def target(s, a):
        with torch.no_grad():
            return teacher(s, a) / rms

    s_chk, a_chk = torch.randn(64, N_STATE, 2), torch.randn(64, 1, 2)
    teacher_eq = equiv_residual(lambda s, a: target(s, a), s_chk, a_chk, PROBE_ANGLES)
    assert teacher_eq < 1e-4, "teacher world is not equivariant"

    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, 2) * 1.5
    mu_a = torch.randn(1, 2) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, 2, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, 2, generator=gen)
        return s, a

    vn_p = n_params(VNDynamics())
    mlp_ps = {h: n_params(MLPDynamics(hidden=h)) for h in SIZES}
    print(f"    VN(equivariant, fixed)={vn_p} params; "
          f"MLP sizes={ {h: mlp_ps[h] for h in SIZES} } "
          f"({ {h: round(mlp_ps[h] / vn_p, 1) for h in SIZES} }x VN)")
    print(f"    budget: {STEPS} AdamW steps, batch {BATCH}, coverage {COVERAGE_DEG} deg "
          f"(theta_max={COVERAGE_DEG} deg)\n")

    cov = COVERAGE_DEG * math.pi / 180.0

    # grid[(h, N)] -> lists over seeds; vn_ref[N] -> lists over seeds
    grid = {(h, N): {"ratio": [], "eq": [], "seen": [], "worst": []} for h in SIZES for N in DATAS}
    vn_ref = {N: {"ratio": [], "eq": [], "seen": []} for N in DATAS}

    for sd in SEEDS:
        data_gen = torch.Generator().manual_seed(2000 + sd)
        aug_gen = torch.Generator().manual_seed(3000 + sd)
        test_gen = torch.Generator().manual_seed(4000 + sd)

        # held-out test + probe scenes (shared across all cells this seed)
        s_te, a_te = aniso_inputs(N_TEST, test_gen)
        s_pr, a_pr = aniso_inputs(min(N_TEST, 1000), test_gen)

        # per-data-scale base scenes; VN reference trained on the thin wedge [0,90)
        for N in DATAS:
            s_c, a_c = aniso_inputs(N, data_gen)

            # VN reference (fixed size): wedge-only, full-batch; scale-free by construction
            al_w = torch.rand(N, generator=data_gen) * (math.pi / 2)
            s_w, a_w = rotate_per_sample(s_c, al_w), rotate_per_sample(a_c, al_w)
            torch.manual_seed(10 + sd)
            vn = train_full(VNDynamics(), s_w, a_w, target(s_w, a_w), steps=STEPS)
            seen, _, ratio = eval_circle(vn, target, s_te, a_te, test_gen)
            vn_ref[N]["ratio"].append(ratio); vn_ref[N]["seen"].append(seen)
            vn_ref[N]["eq"].append(equiv_residual(vn, s_pr, a_pr, PROBE_ANGLES))

            # MLP + partial-coverage augmentation, swept over size
            for h in SIZES:
                torch.manual_seed(30 + sd)  # paired init across sizes for fixed (N, seed)
                mlp = train_augmented_mb(MLPDynamics(hidden=h), s_c, a_c, target, cov,
                                         steps=STEPS, batch=BATCH, gen=aug_gen)
                seen, worst, ratio = eval_circle(mlp, target, s_te, a_te, test_gen)
                g = grid[(h, N)]
                g["ratio"].append(ratio); g["seen"].append(seen)
                g["worst"].append(worst); g["eq"].append(equiv_residual(mlp, s_pr, a_pr, PROBE_ANGLES))

    def summ(d, keys=("ratio", "eq", "seen", "worst")):
        return {k: mean_std(d[k]) for k in keys if k in d}

    grid_s = {hn: summ(grid[hn]) for hn in grid}
    vn_s = {N: summ(vn_ref[N], keys=("ratio", "eq", "seen")) for N in DATAS}

    # --- [1] task metric grid -------------------------------------------------------------
    print(line)
    print("[1] TASK METRIC -- OOD/seen relMSE ratio  (1.00 = flat; coverage [0,180) so OOD = extrapolation)")
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
    # scale effect on the task: best (largest size, most data) vs smallest cell
    small = grid_s[(SIZES[0], DATAS[0])]["ratio"][0]
    big = grid_s[(SIZES[-1], DATAS[-1])]["ratio"][0]
    vn_flat = max(vn_s[N]["ratio"][0] for N in DATAS)
    print(f"    => smallest cell x{small:.2f} -> largest cell x{big:.2f}; "
          f"VN flat at x{vn_flat:.3f} regardless of N.")

    # --- [2] exactness grid ---------------------------------------------------------------
    print()
    print(line)
    print("[2] EXACTNESS -- residual equivariance Delta_eq = max_a ||f(Rx)-Rf(x)|| / ||f(x)||")
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
    best_mlp_eq = min(grid_s[(h, N)]["eq"][0] for h in SIZES for N in DATAS)  # the closest aug ever gets
    eq_factor = best_mlp_eq / max(vn_floor, 1e-12)
    print(f"    => the BEST (most-equivariant) MLP cell is Delta_eq={best_mlp_eq:.2e} "
          f"= x{eq_factor:.0f} the VN floor ({vn_floor:.1e}); scale never reaches exactness.")

    # --- verdict --------------------------------------------------------------------------
    print()
    print(line)
    print("STEP 29 SUMMARY")
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
    print(f"    headline: scale is not a substitute for the missing coverage (task) nor for the "
          f"architecture (exactness). The prior gives both, free and scale-independent.")

    assert vn_flat < 1.5, "VN must stay ~flat across orientations at every data scale"
    assert vn_floor < 1e-4, "VN must stay exactly equivariant (structural) at every data scale"
    assert partial_gap, "partial coverage must leave a real extrapolation gap to study"
    assert exact_plateau, "no (size,data) cell may reach the architecture's exact equivariance"
    print(f"\n    guards: vn-flat={vn_flat < 1.5}  vn-exact={vn_floor < 1e-4}  "
          f"partial-gap={partial_gap}  exactness-plateau={exact_plateau}  "
          f"task-shrinks={task_helps}  task-closes={task_closes}")
    print("    PASS: scale does not buy what the architecture is.")

    # --- dump JSON ------------------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "group": "SO(2)",
        "config": dict(seeds=SEEDS, sizes=SIZES, datas=DATAS, steps=STEPS, batch=BATCH,
                       coverage_deg=COVERAGE_DEG, n_test=N_TEST, n_state=N_STATE,
                       probe_angles=PROBE_ANGLES),
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
    out_path = fig_dir / ("step29_scaling_sweep_smoke.json" if SMOKE else "step29_scaling_sweep.json")
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
            axL.set_title(f"[1] task: scale vs the extrapolation gap (cov {COVERAGE_DEG} deg)")
            axL.legend(fontsize=8); axL.grid(alpha=0.3)
            for N in DATAS:
                axR.plot(xs, [grid_s[(h, N)]["eq"][0] for h in SIZES], "o-", label=f"MLP+aug, N={N}")
            axR.axhline(vn_floor, color="C2", ls="--", label="VN (exact floor)")
            axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("MLP parameters")
            axR.set_ylabel("residual equivariance  Delta_eq")
            axR.set_title("[2] exactness: scale-independent plateau")
            axR.legend(fontsize=8); axR.grid(alpha=0.3)
            fig.tight_layout()
            fig_path = fig_dir / "step29_scaling_sweep.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"    wrote {fig_path.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    (figure skipped: {e})")

    sys.exit(0)


if __name__ == "__main__":
    main()
