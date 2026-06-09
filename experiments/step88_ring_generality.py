r"""Step 88 — GENERALITY of ③-A (the trustworthy-certificate budgeted re-observation win) on a SECOND high-N system:
the coupled-pendulum ring (step80). PRECHECK first (the go/no-go), then the cert-isolated budget frontier.

③-A (step85, §5.20) showed on 40-D Lorenz-96 that the Z_N-equivariant model's *faithful* certificate beats the
non-equivariant model's *inflated* certificate under a fixed sensing budget. Its precondition is the high-N
spectrum-garbage contrast (§5.16/E2): the dense model's λ1 is inflated ~3×, the equivariant model's is faithful. That
contrast is a HIGH-N phenomenon (at low N both succeed). step80 only ran the ring at N=10, where the contrast may be
absent. So this file's **precheck** trains conv (RingConv, Z_N-equiv) + MLP (RingMLP, dense) on the ring at a HIGH N
and asks: (a) is it still chaotic (true λ1>0), (b) is the MLP's λ1 inflated vs conv (the ③-A precondition), (c) is the
conv certificate calibrated? If yes → ③-A's mechanism should transfer and the full cert-isolated budget frontier is
worth running; if no → ③-A is (so far) Lorenz-96-specific at feasible N, reported honestly.

Reuse, do NOT modify: step80 (`collect_data`, `train_wm`, `certificate`, `certified_T1_steps`,
`empirical_forecast_horizon`, `true_ring_spectrum`, `attractor_traj_ring`). float64 CPU.

Run (precheck): STEP88_MODE=precheck STEP88_N=24 .venv/bin/python experiments/step88_ring_generality.py
Writes: papers/figures/step88_ring_precheck.json ; exit 0 if the ③-A precondition holds.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step80_pendulum_ring as step80  # noqa: E402

DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP88_SMOKE", "0")))


def run_precheck(N: int, eps: float, seed: int) -> dict:
    r"""Train conv + MLP on the N-pendulum ring; report true/conv/MLP leading Lyapunov exponents, the MLP inflation
    factor, and the conv calibration ratio. The ③-A precondition = chaotic AND MLP λ1 inflated (>1.5×) AND conv
    certificate calibrated (measured/certified ∈ [1/2, 2])."""
    n_traj = 1500 if SMOKE else 6000
    K = 5
    epochs = 8 if SMOKE else 60
    cert_kw = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
                   n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)

    # (a) chaos check on the TRUE ring at this N (Benettin-QR on the true field).
    lam_true = step80.true_ring_spectrum(N, n_steps=800 if SMOKE else 2500, warmup=300 if SMOKE else 800, seed=seed)
    true_l1 = float(lam_true[0])
    chaotic = bool(true_l1 > 0.03)
    print(f"[step88] N={N}: true lambda1={true_l1:.3f} (chaotic={chaotic})", file=sys.stderr)

    # (b) train conv + MLP, read each certificate's lambda1.
    data = step80.collect_data(N, n_traj, K, seed)
    conv, mu_c, sd_c, relmse_c = step80.train_wm("conv", N, data, seed, epochs=epochs, K=K)
    mlp, mu_m, sd_m, relmse_m = step80.train_wm("mlp", N, data, seed, epochs=epochs, K=K)
    cert_c = step80.certificate(conv, mu_c, sd_c, N, eps=eps, seed=seed, **cert_kw)
    cert_m = step80.certificate(mlp, mu_m, sd_m, N, eps=eps, seed=seed, **cert_kw)
    conv_l1, mlp_l1 = cert_c["lambda1"], cert_m["lambda1"]
    inflation = (mlp_l1 / conv_l1) if conv_l1 and conv_l1 > 0 else float("nan")

    # (c) conv calibration: measured (empirical) vs certified horizon.
    T1c = step80.certified_T1_steps(cert_c)
    emp_c = step80.empirical_forecast_horizon(conv, mu_c, sd_c, N, eps=eps,
                                              n_starts=16 if SMOKE else 32, seed=seed)
    H_emp_c = emp_c["median_empirical_horizon_steps"]
    ratio_c = (H_emp_c / T1c) if T1c else None
    conv_calib = bool(ratio_c is not None and 0.5 <= ratio_c <= 2.0)

    precondition = bool(chaotic and inflation > 1.5 and conv_calib)
    print(f"[step88] N={N}: true_l1={true_l1:.3f} conv_l1={conv_l1:.3f} (relMSE {relmse_c:.4f}) "
          f"mlp_l1={mlp_l1:.3f} (relMSE {relmse_m:.4f}) | inflation={inflation:.2f}x | "
          f"conv T1={T1c} Hemp={H_emp_c:.0f} ratio={ratio_c if ratio_c is None else round(ratio_c,2)} calib={conv_calib}",
          file=sys.stderr)
    verdict = {"precondition_holds": precondition, "chaotic": chaotic, "true_l1": true_l1,
               "conv_l1": conv_l1, "mlp_l1": mlp_l1, "inflation": inflation,
               "conv_T1_steps": T1c, "conv_H_emp": H_emp_c, "conv_ratio": ratio_c, "conv_calibrated": conv_calib,
               "relmse_conv": relmse_c, "relmse_mlp": relmse_m, "N": N, "eps": eps, "seed": seed, "smoke": SMOKE}
    print(f"[step88] {'PRECONDITION HOLDS — build full step88 budget frontier' if precondition else 'PRECONDITION FAILS — ③-A may be Lorenz-96-specific at this N (honest)'}: "
          f"chaotic={chaotic}, MLP inflation={inflation:.2f}x (need >1.5), conv calibrated={conv_calib}", file=sys.stderr)
    return verdict


# --------------------------------------------------------------------------------------------------------------- #
# Phase 1 — the cert-isolated FIXED-BUDGET frontier on the ring (the generality test of ③-A / step85 Phase 1).
# Forecaster held FIXED = the equivariant RingConv; re-observation interval dictated by conv-cert vs MLP-cert; budget
# swept. The MLP cert's inflated lambda1 (~2x here) gives a ~2x-too-short interval -> over-observe -> starve -> higher
# aggregate violation. Budgeted variant of step80.reobserve_run (step80 itself NOT modified): same ring-aware wm_step
# + wrap-safe error, with the budget cap added.
# --------------------------------------------------------------------------------------------------------------- #
def reobserve_run_budgeted_ring(model, mu, sd, N, y_seq_true, interval, eps, budget) -> dict:
    r"""step80.reobserve_run with a budget cap: re-observe every ``interval`` steps but only up to ``budget`` times;
    after the budget is spent, forecast open-loop to the end. Aggregate violation over all $T_{\rm total}$ steps."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    interval = max(1, int(interval)); budget = max(1, int(budget))
    y_seq_true = y_seq_true.to(DTYPE)
    T_total = y_seq_true.shape[0] - 1
    n_obs = 1
    errors = []
    with torch.no_grad():
        y_hat = y_seq_true[0].clone()
        for t in range(1, T_total + 1):
            y_hat = step80.wm_step_state(model, y_hat, N, mu, sd, None)        # ring-aware u=0 advance (handles drive)
            if (t % interval == 0) and (n_obs < budget):
                y_hat = y_seq_true[t].clone(); n_obs += 1; errors.append(0.0)
            else:
                errors.append(float(step80._forecast_relerr(y_hat, y_seq_true[t], N)))
    errors_np = np.asarray(errors, dtype=float)
    viol = float((errors_np > eps).mean()) if errors_np.size else 0.0
    return {"n_observations": n_obs, "violation_rate": viol, "interval": interval, "budget": budget,
            "covered_steps": (n_obs - 1) * interval, "T_total": T_total}


def _viol_at(front, B):
    return min(front, key=lambda r: abs(r["budget"] - B))["violation_rate"]


def run_phase1(seeds, N: int, eps: float, L: int) -> dict:
    r"""Cert-isolated budget frontier on the ring, 3 seeds. Forecaster fixed = conv; conv-cert interval vs MLP-cert
    interval; budget swept; gate G1: conv-cert aggregate violation < MLP-cert at the knee on >=2/3 seeds."""
    n_traj = 1500 if SMOKE else 6000
    K = 5
    epochs = 8 if SMOKE else 60
    cert_kw = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
                   n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)
    per_seed = {}
    for s in seeds:
        print(f"[step88.P1] seed {s}: training RingConv + RingMLP at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        data = step80.collect_data(N, n_traj, K, s)
        conv, mu_c, sd_c, _ = step80.train_wm("conv", N, data, s, epochs=epochs, K=K)
        mlp, mu_m, sd_m, _ = step80.train_wm("mlp", N, data, s, epochs=epochs, K=K)
        cert_c = step80.certificate(conv, mu_c, sd_c, N, eps=eps, seed=s, **cert_kw)
        cert_m = step80.certificate(mlp, mu_m, sd_m, N, eps=eps, seed=s, **cert_kw)
        iv_conv = step80.certified_T1_steps({"T1": cert_c.get("T1_lo") or cert_c.get("T1")})
        iv_mlp = step80.certified_T1_steps({"T1": cert_m.get("T1")})
        y_seq = step80.attractor_traj_ring(N, L, s)                          # TRUE u=0 ring trajectory (L+1, 2N+1)
        cover_conv = max(1, int(np.ceil(L / max(1, iv_conv))))
        cover_mlp = max(1, int(np.ceil(L / max(1, iv_mlp))))
        budgets = sorted({2, cover_conv, cover_mlp}
                         | {int(round(b)) for b in np.linspace(2, max(6, int(np.ceil(cover_mlp * 1.3))), 8)})
        conv_arm = [reobserve_run_budgeted_ring(conv, mu_c, sd_c, N, y_seq, iv_conv, eps, b) for b in budgets]
        mlp_iso = [reobserve_run_budgeted_ring(conv, mu_c, sd_c, N, y_seq, iv_mlp, eps, b) for b in budgets]
        cv, mv = _viol_at(conv_arm, cover_conv), _viol_at(mlp_iso, cover_conv)
        pareto = float(np.mean([c["violation_rate"] < m["violation_rate"] for c, m in zip(conv_arm, mlp_iso)]))
        per_seed[s] = {"iv_conv": iv_conv, "iv_mlp": iv_mlp, "lambda1_conv": cert_c["lambda1"],
                       "lambda1_mlp": cert_m["lambda1"], "knee_budget": cover_conv,
                       "conv_arm_viol_at_knee": cv, "mlp_iso_viol_at_knee": mv, "win_margin": mv - cv,
                       "pareto_frac": pareto,
                       "frontier": {"budgets": budgets, "conv_arm": [r["violation_rate"] for r in conv_arm],
                                    "mlp_iso": [r["violation_rate"] for r in mlp_iso]}}
        print(f"[step88.P1] seed {s}: iv_conv={iv_conv} iv_mlp={iv_mlp} knee={cover_conv} | "
              f"viol@knee conv={cv:.3f} mlp={mv:.3f} margin={mv-cv:+.3f} | Pareto {pareto:.0%}", file=sys.stderr)

    margins = [per_seed[s]["win_margin"] for s in seeds]
    wins = [m > 0.05 for m in margins]
    g1 = bool(sum(wins) >= int(np.ceil(2 / 3 * len(seeds))))
    verdict = {"G1_pass": g1, "n_win": int(sum(wins)), "n_seeds": len(seeds), "win_margins": margins,
               "eps": eps, "N": N, "L": L, "smoke": SMOKE}
    print(f"[step88.P1] {'G1 PASS — ③-A GENERALIZES to the ring' if g1 else 'G1 INCONCLUSIVE on the ring'}: "
          f"conv-cert beats MLP-cert at knee on {int(sum(wins))}/{len(seeds)} seeds (margins {[round(m,3) for m in margins]}).",
          file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = os.environ.get("STEP88_MODE", "precheck")
    eps = float(os.environ.get("STEP88_EPS", "0.2"))
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    if mode == "precheck":
        N = int(os.environ.get("STEP88_N", "8" if SMOKE else "24"))
        seed = int(os.environ.get("STEP88_SEED", "0"))
        print(f"[step88] ring generality precheck: N={N} eps={eps} seed={seed} smoke={SMOKE}", file=sys.stderr)
        res = run_precheck(N, eps, seed)
        (figdir / f"step88_ring_precheck{tag}.json").write_text(json.dumps(res, indent=2))
        raise SystemExit(0 if res["precondition_holds"] else 1)
    N = int(os.environ.get("STEP88_N", "8" if SMOKE else "24"))
    seeds = [int(x) for x in os.environ.get("STEP88_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    L = int(os.environ.get("STEP88_L", "300" if SMOKE else "1500"))
    print(f"[step88] ring generality FULL: N={N} eps={eps} L={L} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
    res = run_phase1(seeds, N, eps, L)
    (figdir / f"step88_ring_frontier{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0 if res["verdict"]["G1_pass"] else 1)
