r"""Step 85c — the CALIBRATION-ONLY baseline for ③-A (the reviewers' "do-or-die" ablation).

③-A (step85, §5.20): on 40-D Lorenz-96 the equivariant model's faithful certificate beats the dense model's inflated
certificate (lambda1 ~3.4x too big) under a fixed sensing budget. A tough reviewer asks: the inflation is suspiciously
CONSTANT (3.19/3.50/3.46 across seeds) — can you just **recalibrate** the dense model's certificate with one scalar and
close the gap? If yes, ③-A is "you need *a* faithful horizon," not "you need *equivariance*."

This file adds a third arm — **recalibrated-MLP cert** — to the cert-isolated budget frontier (forecaster fixed = conv;
only the interval changes). The recalibration is the most generous possible: set the MLP-cert interval to the MLP's own
**empirical** forecast horizon (measured on held-out rollouts — the one-parameter fix a practitioner would do). Arms:
  * conv-cert (a-priori, from the equivariant Jacobian — NO rollout/calibration data),
  * MLP-cert (raw, inflated),
  * MLP-recal (MLP cert corrected to its measured empirical horizon — needs a calibration set).

Expected (honest): MLP-recal ≈ conv-cert (the constant inflation IS correctable). The conclusion is then a **reframe,
not a refutation**: equivariance's value is that the certificate is *a-priori* (zero calibration data); a dense model
matches it only after measuring its horizon on data — exactly the ~3x-budget warm-start the cert-free adaptive baseline
already pays. Reported honestly either way. (Does NOT overwrite step85's canonical frontier JSON.)

Reuse, do NOT modify: step85 (train_one, reobserve_run_budgeted, build_true_traj, budget_frontier, _viol_at),
step79 (certificate, certified_T1_steps, empirical_forecast_horizon). float64 CPU.

Run (smoke): STEP85C_SMOKE=1 .venv/bin/python experiments/step85c_calibration_baseline.py
Writes: papers/figures/step85c_calibration_baseline.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step79_certified_control as step79  # noqa: E402
import step85_trustworthy_cert_downstream as step85  # noqa: E402

SMOKE = bool(int(os.environ.get("STEP85C_SMOKE", "0")))
step85.SMOKE = SMOKE  # propagate to step85.train_one (controls model/training size)


def run(seeds, N: int, eps: float, L: int) -> dict:
    ck = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
              n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)
    n_starts = 16 if SMOKE else 40
    per_seed = {}
    for s in seeds:
        print(f"[step85c] seed {s}: training conv + MLP at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        conv, mu_c, sd_c, _ = step85.train_one("conv", N, s)
        mlp, mu_m, sd_m, _ = step85.train_one("mlp", N, s)
        cert_c = step79.certificate(conv, mu_c, sd_c, N, eps=eps, seed=s, **ck)
        cert_m = step79.certificate(mlp, mu_m, sd_m, N, eps=eps, seed=s, **ck)
        iv_conv = step79.certified_T1_steps({"T1": cert_c.get("T1_lo") or cert_c.get("T1")})
        iv_mlp = step79.certified_T1_steps({"T1": cert_m.get("T1")})
        # The recalibration: the MLP's *measured* empirical horizon (the one-parameter, data-driven fix).
        emp_m = step79.empirical_forecast_horizon(mlp, mu_m, sd_m, N, eps=eps, n_starts=n_starts, seed=s,
                                                  max_steps=max(3 * iv_mlp, 600))
        iv_mlp_recal = max(1, int(round(emp_m["median_empirical_horizon_steps"])))

        x_seq = step85.build_true_traj(N, L, s)
        cover_conv = max(1, int(np.ceil(L / max(1, iv_conv))))
        cover_mlp = max(1, int(np.ceil(L / max(1, iv_mlp))))
        budgets = sorted({2, cover_conv, cover_mlp}
                         | {int(round(b)) for b in np.linspace(2, max(6, int(np.ceil(cover_mlp * 1.3))), 8)})
        # All arms use the SAME conv forecaster (cert-isolated); only the interval differs.
        conv_arm = step85.budget_frontier(conv, mu_c, sd_c, N, x_seq, iv_conv, eps, budgets)
        mlp_iso = step85.budget_frontier(conv, mu_c, sd_c, N, x_seq, iv_mlp, eps, budgets)
        mlp_recal = step85.budget_frontier(conv, mu_c, sd_c, N, x_seq, iv_mlp_recal, eps, budgets)
        cv = step85._viol_at(conv_arm, cover_conv)
        mv = step85._viol_at(mlp_iso, cover_conv)
        rv = step85._viol_at(mlp_recal, cover_conv)
        # Does recalibration close the gap? gap_closed if recal is much closer to conv than the raw MLP cert is.
        raw_gap = mv - cv
        recal_gap = rv - cv
        gap_closed = bool(abs(recal_gap) <= 0.5 * abs(raw_gap) + 0.02)
        per_seed[s] = {"iv_conv": iv_conv, "iv_mlp": iv_mlp, "iv_mlp_recal": iv_mlp_recal,
                       "conv_viol": cv, "mlp_raw_viol": mv, "mlp_recal_viol": rv,
                       "raw_gap": raw_gap, "recal_gap": recal_gap, "gap_closed": gap_closed}
        print(f"[step85c] seed {s}: iv conv={iv_conv} mlp={iv_mlp} mlp_recal={iv_mlp_recal} | "
              f"viol@knee conv={cv:.3f} mlp_raw={mv:.3f} mlp_recal={rv:.3f} | "
              f"raw_gap={raw_gap:+.3f} recal_gap={recal_gap:+.3f} closed={gap_closed}", file=sys.stderr)

    closed = [per_seed[s]["gap_closed"] for s in seeds]
    n_closed = int(sum(closed))
    # Interpretation: if recalibration closes the gap on most seeds, ③-A's value is A-PRIORI (no calib data), not
    # exclusivity of equivariance — an honest REFRAME. If it does NOT close, equivariance is exclusively needed.
    reframe = bool(n_closed >= int(np.ceil(2 / 3 * len(seeds))))
    verdict = {"recalibration_closes_gap": reframe, "n_closed": n_closed, "n_seeds": len(seeds),
               "raw_gaps": [per_seed[s]["raw_gap"] for s in seeds],
               "recal_gaps": [per_seed[s]["recal_gap"] for s in seeds], "eps": eps, "N": N, "L": L, "smoke": SMOKE}
    print(f"[step85c] {'REFRAME: a one-parameter recalibration CLOSES the gap on ' + str(n_closed) + '/' + str(len(seeds)) + ' seeds' if reframe else 'EQUIVARIANCE EXCLUSIVELY NEEDED: recalibration does NOT close the gap'}. "
          f"=> ③-A's value is {'the certificate being A-PRIORI (no calibration data); a dense model needs a calibration set (= the adaptive warm-start cost).' if reframe else 'that only the equivariant model gives a faithful certificate at all.'}",
          file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


if __name__ == "__main__":
    torch.manual_seed(0)
    N = int(os.environ.get("STEP85C_N", "10" if SMOKE else "40"))
    eps = float(os.environ.get("STEP85C_EPS", "0.2"))
    seeds = [int(x) for x in os.environ.get("STEP85C_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    L = int(os.environ.get("STEP85C_L", "300" if SMOKE else "1500"))
    print(f"[step85c] calibration baseline: N={N} eps={eps} L={L} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
    res = run(seeds, N, eps, L)
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"step85c_calibration_baseline{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0)
