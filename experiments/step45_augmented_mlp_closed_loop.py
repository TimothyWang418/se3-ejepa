r"""Step 45: does *approximate* (augmentation-bought) equivariance close an exactly
orientation-invariant loop as well as *exact* (architectural) equivariance?

The gap this closes (a reviewer's load-bearing question)
--------------------------------------------------------
Step 28 established that, *given the group*, rotation **augmentation** flattens a non-equivariant
MLP's across-group **task** ratio on a *simple* 6-vector state model (full SO(3) coverage -> OOD/seen
$\approx\times1.06$-$1.46$) but **never reaches the architecture's float-floor exactness**
($\Delta_{\mathrm{eq}}$ plateaus $\sim\!10^5\times$ above the VN floor). Step 18 established that the
*exact* VN closes an SE(3) pose loop with an orientation error invariant across the group. What was
**never run head-to-head** is the *downstream* question the paper's [C] selling point rests on:

    does an augmented MLP -- handed the same knowledge (the world is symmetric) -- still **degrade in
    the closed loop**, where the *exact* VN does not? i.e. does exactness buy a **closed-loop** win that
    approximate (augmentation) equivariance cannot?

The core paper's Limitations currently *asserts* "an MLP with $\Delta_{\mathrm{eq}}\approx0.05$ cannot
close an exactly orientation-invariant loop" -- this experiment **tests** that assertion on the *real
latent world model* (the Step 13/18 point-cloud JEPA + CEM planner), not the simple state model.

Design (multi-seed, reported as mean +- std)
--------------------------------------------
Three models, *identical* SO(3)-equivariant paired closed-loop planner of Step 18 ([E] panel):
  * VN       -- exactly SO(3)-equivariant (architectural);
  * MLP      -- no prior (the Step 18 baseline), for reference;
  * MLP+aug  -- the SAME MLP trained with full-SO(3) rotation augmentation (the Step 28 recipe: rotate
               each wedge transition by random SO(3) and add it to the training set; the teacher is
               exactly equivariant so the augmented label is exact).
A **pure-rotation orbit** ($t_{\mathrm{scale}}{=}0$) isolates the *rotation* question (translation is
handled model-independently by the closed-form centroid channel, so it is not a confound). For each of
several seeds we retrain all three and re-evaluate on a *fixed* task/orbit set, so the spread is the
model-training-seed variance. Headlines: composed $\Delta_{\mathrm{eq}}$ and the closed-loop OOD/seen
orientation ratio, per model, mean +- std, plus a pooled distribution-free sign test.

Honest reading of either outcome (reported as run, not cherry-picked)
--------------------------------------------------------------------
  * MLP+aug still degrades (pooled ratio CI excludes 1) while VN stays flat
      => exactness buys a closed-loop win approximate equivariance cannot -- supports [C].
  * MLP+aug closes the loop as flat as VN
      => approximate equivariance suffices downstream *here* -- the "[C] needs exactness" claim must be
         softened to this regime. We say so.

Run (full, 3 seeds):
    STEP45_K=96 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step45_augmented_mlp_closed_loop.py
Smoke (~2 min, 1 seed, K=6):
    STEP45_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step45_augmented_mlp_closed_loop.py
"""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# exactly-equivariant backbone + helpers (Step 13)
from step13_se3_latent_jepa import (  # noqa: E402
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    composed_equiv_err,
    rand_so3,
    rotate_points,
)
# the verbatim Step 18 paired closed-loop machinery (reused, not reimplemented)
from step18_se3_closed_loop import (  # noqa: E402
    EVAL_DTYPE,
    boot_ratio_ci,
    eval_paired_se3_exact,
    make_se3_orbit,
    make_so3_tasks,
    paired_arrays,
    paired_permutation_test,
    paired_sign_test,
)
from step10_pusht_closed_loop import n_params  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP45_SMOKE"))
NAMES = ["VN (exact)", "MLP (no aug)", "MLP+aug"]


def augment_so3(S, A, S2, n_copies: int, gen: torch.Generator):
    r"""Full-SO(3) rotation augmentation of cloud transitions (Step 28 recipe, fixed copies).

    For each of ``n_copies`` random rotations $R$ (axis uniform, angle $\in[0,2\pi)$), add
    $(RS, RA, RS_2)$ to the set. The teacher is exactly SO(3)-equivariant, so the rotated triple is a
    *valid* transition (the augmented label is exact). ``S,S2: (n,P,3)``, ``A: (n,3)``.
    """
    out_S, out_A, out_S2 = [S], [A], [S2]
    for _ in range(n_copies):
        R = rand_so3(gen)
        out_S.append(rotate_points(S, R))
        out_A.append(rotate_points(A, R))   # A is a type-1 vector: rotate_points = A @ R^T
        out_S2.append(rotate_points(S2, R))
    return torch.cat(out_S, 0), torch.cat(out_A, 0), torch.cat(out_S2, 0)


def _mean_std(xs):
    t = np.asarray(xs, dtype=np.float64)
    return float(t.mean()), (float(t.std()) if t.size > 1 else 0.0)


def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72

    if SMOKE:
        SEEDS = [0]
        N_TRAIN, EPOCHS, NCOPY = 200, 5, 4
        K, N_OOD, H_GOAL = 6, 4, 5
        T_MAX, REPLAN, W_T = 12, 6, 0.5
        cem = dict(H=8, n_samples=64, n_iters=3, n_elite=8, sigma0=0.6, w_run=0.3)
    else:
        SEEDS = [0, 1, 2]
        N_TRAIN, EPOCHS, NCOPY = 1200, 40, 12
        K = int(os.environ.get("STEP45_K", "96"))
        N_OOD, H_GOAL = 4, 6
        T_MAX, REPLAN, W_T = 18, 6, 0.5
        cem = dict(H=12, n_samples=256, n_iters=5, n_elite=25, sigma0=0.6, w_run=0.3)
    VAR_COEF = 0.1

    print(line)
    print(f"STEP 45  exact vs augmented(approx) equivariance in the closed loop  "
          f"({'SMOKE' if SMOKE else 'FULL'}; {len(SEEDS)} seed(s))")
    print(line)

    # fixed eval tasks + PURE-SO(3) orbit (t=0) across seeds -> spread = training-seed variance
    base_tasks = make_so3_tasks(K, seed=321, H_goal=H_GOAL)
    orbit = make_se3_orbit(N_OOD, seed=13, t_scale=0.0)
    gids = list(range(len(orbit)))
    St, At, _ = collect_cloud_transitions(64, seed=999)
    Rchk = rand_so3(torch.Generator().manual_seed(3))
    print(f"    {K} paired tasks (fixed); pure-SO(3) orbit (t=0): 1 seen + {N_OOD} OOD rotations; "
          f"matching equivariant planner")

    per = {nm: {"ratio": [], "eq": [], "seen": [], "ood": []} for nm in NAMES}

    for sd in SEEDS:
        S, A, S2 = collect_cloud_transitions(N_TRAIN, seed=sd)
        aug_gen = torch.Generator().manual_seed(1000 + sd)
        Sa, Aa, S2a = augment_so3(S, A, S2, NCOPY, aug_gen)
        torch.manual_seed(200 + sd)
        models = {"VN (exact)": build_eq_jepa(), "MLP (no aug)": build_mlp_jepa(), "MLP+aug": build_mlp_jepa()}
        train_jepa(models["VN (exact)"], S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=sd, log_every=999)
        train_jepa(models["MLP (no aug)"], S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=sd, log_every=999)
        train_jepa(models["MLP+aug"], Sa, Aa, S2a, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=sd, log_every=999)
        for nm, m in models.items():
            per[nm]["eq"].append(composed_equiv_err(m, St, At, Rchk))
            m_eval = m.to(EVAL_DTYPE)
            res = eval_paired_se3_exact(m_eval, base_tasks, orbit, w_t=W_T, T_max=T_MAX,
                                        replan_every=REPLAN, **cem)
            seen, ood = paired_arrays(res, gids)
            per[nm]["seen"].append(seen)
            per[nm]["ood"].append(ood)
            per[nm]["ratio"].append(float(ood.mean() / max(seen.mean(), 1e-9)))
        print(f"    seed {sd}: VN eq={per['VN (exact)']['eq'][-1]:.1e} r={per['VN (exact)']['ratio'][-1]:.3f} | "
              f"MLP eq={per['MLP (no aug)']['eq'][-1]:.1e} r={per['MLP (no aug)']['ratio'][-1]:.3f} | "
              f"aug eq={per['MLP+aug']['eq'][-1]:.1e} r={per['MLP+aug']['ratio'][-1]:.3f}")

    # ---- aggregate over seeds -------------------------------------------------------------
    print()
    print(line)
    print(f"[E] SUMMARY over {len(SEEDS)} seed(s) -- pooled closed-loop, K={K}/seed, pure-SO(3) orbit")
    print(line)
    print(f"    {'model':>14s} | {'Delta_eq mean+-std':>20s} | {'ratio mean+-std':>16s} | "
          f"{'pooled ratio CI':>18s} | pooled sign test")
    print("    " + "-" * 104)
    agg = {}
    for nm in NAMES:
        eqm, eqs = _mean_std(per[nm]["eq"])
        rm, rs = _mean_std(per[nm]["ratio"])
        seen_all = np.concatenate(per[nm]["seen"])
        ood_all = np.concatenate(per[nm]["ood"])
        diff_all = ood_all - seen_all
        prm, prlo, prhi = boot_ratio_ci(ood_all, seen_all, seed=3)
        pos, n, sp = paired_sign_test(diff_all)
        pp = paired_permutation_test(diff_all)
        print(f"    {nm:>14s} | {eqm:9.2e} +-{eqs:7.1e} | {rm:6.3f} +-{rs:5.3f} | "
              f"[{prlo:5.3f},{prhi:5.3f}] | {pos}/{n} p={sp:.2e}")
        agg[nm] = {
            "eq": [eqm, eqs], "ratio": [rm, rs], "pooled_ratio_ci": [prm, prlo, prhi],
            "sign": [pos, n, sp], "perm_p": pp, "max_d": float(np.abs(diff_all).max()),
        }

    # ---- verdict --------------------------------------------------------------------------
    print()
    print(line)
    print("STEP 45 VERDICT")
    print(line)
    vn = agg["VN (exact)"]
    aug = agg["MLP+aug"]
    noaug = agg["MLP (no aug)"]
    vn_flat = vn["pooled_ratio_ci"][2] < 1.05
    aug_degrades_ci = aug["pooled_ratio_ci"][1] > 1.0      # pooled ratio CI lower > 1
    aug_degrades_sign = aug["sign"][2] < 0.05 and aug["sign"][0] > aug["sign"][1] / 2
    aug_never_approx = aug["eq"][0] > 50.0 * max(vn["eq"][0], 1e-12)
    print(f"    VN (exact)   : ratio {vn['ratio'][0]:.3f}+-{vn['ratio'][1]:.3f} (flat={vn_flat}); "
          f"Delta_eq={vn['eq'][0]:.1e}")
    print(f"    MLP+aug      : ratio {aug['ratio'][0]:.3f}+-{aug['ratio'][1]:.3f}; "
          f"Delta_eq={aug['eq'][0]:.1e} (x{aug['eq'][0] / max(vn['eq'][0], 1e-12):.0e} the VN floor); "
          f"degrades-OOD(sign p)={aug['sign'][2]:.1e}")
    print(f"    MLP (no aug) : ratio {noaug['ratio'][0]:.3f}+-{noaug['ratio'][1]:.3f}; "
          f"Delta_eq={noaug['eq'][0]:.1e}")
    if aug_never_approx and (aug_degrades_ci or aug_degrades_sign):
        verdict = "EXACTNESS_BUYS_CLOSED_LOOP_WIN"
        print(f"    => {verdict}: on the real latent world model, full-SO(3) augmentation does NOT even "
              f"approximate equivariance (Delta_eq stays ~{aug['eq'][0]:.0f}, vs the VN's float floor), and "
              f"the augmented MLP STILL degrades in the closed loop where the exact VN is flat. Augmentation "
              f"is not a closed-loop substitute for exact equivariance here.")
    elif vn_flat and aug["pooled_ratio_ci"][2] < 1.05:
        verdict = "APPROX_SUFFICES_SOFTEN_CLAIM"
        print(f"    => {verdict}: the augmented MLP closes the loop as flat as the exact VN. Approximate "
              f"equivariance SUFFICES downstream in this regime; the '[C] needs exactness' claim must be "
              f"softened. Reported as run.")
    else:
        verdict = "EXACT_CLEANER_AUG_PARTIAL"
        print(f"    => {verdict}: the exact VN is flat and augmentation never approximates equivariance "
              f"(Delta_eq ~{aug['eq'][0]:.0f}); the augmented MLP's closed-loop ratio is no better than the "
              f"un-augmented one. Exactness is the operative property; reported with its honest spread.")

    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(SEEDS=SEEDS, N_TRAIN=N_TRAIN, EPOCHS=EPOCHS, NCOPY=NCOPY, K=K, N_OOD=N_OOD,
                       H_GOAL=H_GOAL, T_MAX=T_MAX, REPLAN=REPLAN, W_T=W_T, t_scale=0.0, cem=cem),
        "params": {"vn": n_params(models["VN (exact)"]), "mlp": n_params(models["MLP (no aug)"])},
        "per_seed": {nm: {"ratio": per[nm]["ratio"], "eq": per[nm]["eq"]} for nm in NAMES},
        "agg": agg,
        "verdict": verdict,
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step45_augmented_mlp_closed_loop_smoke.json" if SMOKE
                          else "step45_augmented_mlp_closed_loop.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
