r"""Step 90 — UQ head-to-head: how well does each uncertainty method know its own forecast's expiry, and at what cost?

The strongest methodological attack on §5.20/E12 ("why a Lyapunov certificate instead of standard UQ?") answered with
a three-way contrast on the SAME 40-D Lorenz-96 world model (the §5.20 conv), eps=0.2, 3 experiment seeds:

  * **Certificate (ours):** predicted horizon = $T_1(\epsilon)$ from the model's own Jacobian spectrum (step79
    machinery). Cost: 1 model, **zero** truth rollouts — a-priori.
  * **Ensemble disagreement (PETS/Dreamer-style):** K models, same data, different inits; predicted horizon = first
    step where the members' relative spread around the ensemble mean exceeds eps (computable WITHOUT truth). Cost:
    K× training, zero truth rollouts. Honest question: does epistemic spread track the TRUE divergence horizon?
  * **Conformal/empirical calibration:** predicted = quantile of n_cal measured divergence horizons on held-out TRUE
    rollouts. Cost: 1 model + n_cal truth-access rollouts (accuracy bought with data — the step85c lesson, formalized).

Metric per arm: calibration ratio predicted/actual (actual = measured divergence horizon of the arm's own forecaster
on fresh starts) + the cost columns. Reported honestly whichever way it lands.

Run (full):  .venv/bin/python experiments/step90_uq_baselines.py     (K=4, ~9 extra trainings, tens of minutes CPU)
Smoke:       STEP90_SMOKE=1 ... (N=10, K=2, 1 seed)
Writes: papers/figures/step90_uq_baselines.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as step74  # noqa: E402
import step79_certified_control as step79  # noqa: E402
import step85_trustworthy_cert_downstream as step85  # noqa: E402

DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP90_SMOKE", "0")))


def ensemble_spread_horizon(members, mu, sd, x0_raw, eps: float, max_steps: int = 400) -> dict:
    r"""Truth-free ensemble-disagreement horizon: roll every member (autonomous, normalized frame) from each start;
    per start, the first step where the **relative spread** $\max_j\lVert z_j-\bar z\rVert/\lVert\bar z\rVert$ exceeds
    ``eps`` (censored at ``max_steps``). Returns per-start list + median. This is the standard "trust until the
    ensemble disagrees" rule, evaluated honestly."""
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    z = torch.stack([((x0_raw.to(DTYPE) - mu) / sd) for _ in members], 0)     # (K, B, N)
    B = x0_raw.shape[0]
    crossed = [None] * B
    with torch.no_grad():
        for t in range(1, max_steps + 1):
            z = torch.stack([m(z[j]) for j, m in enumerate(members)], 0)
            zbar = z.mean(0)                                                  # (B, N)
            spread = (z - zbar).norm(dim=-1).max(0).values / zbar.norm(dim=-1).clamp_min(1e-12)
            for b in range(B):
                if crossed[b] is None and float(spread[b]) > eps:
                    crossed[b] = t
            if all(c is not None for c in crossed):
                break
    per_start = [c if c is not None else max_steps for c in crossed]
    return {"per_start": per_start, "median": float(np.median(per_start))}


def conformal_horizon(cal_horizons, alpha: float = 0.5) -> float:
    r"""Conformal-style horizon predictor: the ``alpha``-quantile of calibration-set divergence horizons (lower
    quantile = more conservative coverage). Accuracy is bought with the ``n_cal`` truth rollouts that produced them."""
    return float(np.quantile(np.asarray(cal_horizons, dtype=float), alpha))


class _MeanWM(torch.nn.Module):
    r"""Ensemble-mean forecaster (the rollout an ensemble agent would act on), autonomous one-arg signature."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], 0).mean(0)


def run(seeds, N: int, eps: float, K: int) -> dict:
    n_train = 4000 if SMOKE else 20000
    n_starts = 8 if SMOKE else 20
    n_cal = 6 if SMOKE else 10
    cert_kw = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
                   n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)
    os.environ["STEP74_SMOKE"] = "1" if SMOKE else "0"
    os.environ["STEP74_N"] = str(N)
    os.environ["STEP74_MODEL"] = "conv"
    per_seed = {}
    for s in seeds:
        print(f"[step90] seed {s}: training base conv + {K - 1} ensemble members at N={N} ...", file=sys.stderr)
        traj = step74.attractor_traj(N, n_train, s, "cpu")
        base, mu, sd, relmse = step74.train_model(traj, N, "cpu", s)
        members = [base] + [step74.train_model(traj, N, "cpu", s * 100 + j)[0] for j in range(1, K)]

        # --- certificate arm (a-priori: zero rollouts) ---
        aw = step85.AutonomousWM(base).double()
        cert = step79.certificate(aw, mu, sd, N, eps=eps, seed=s, **cert_kw)
        pred_cert = step79.certified_T1_steps(cert)
        emp0 = step79.empirical_forecast_horizon(aw, mu, sd, N, eps=eps, n_starts=n_starts, seed=s + 777,
                                                 max_steps=max(3 * pred_cert, 400))
        actual_cert = emp0["median_empirical_horizon_steps"]

        # --- ensemble arm (K models; truth-free spread rule) ---
        g = torch.Generator().manual_seed(s + 31)
        idx = torch.randperm(traj.shape[0], generator=g)[:n_starts]
        x0 = traj[idx].clone()
        ens_pred = ensemble_spread_horizon(members, mu, sd, x0, eps=eps)
        mean_wm = step85.AutonomousWM(_MeanWM(members)).double()
        emp_ens = step79.empirical_forecast_horizon(mean_wm, mu, sd, N, eps=eps, n_starts=n_starts, seed=s + 777,
                                                    max_steps=max(3 * pred_cert, 400))
        actual_ens = emp_ens["median_empirical_horizon_steps"]

        # --- conformal arm (1 model + n_cal truth rollouts) ---
        cal = step79.empirical_forecast_horizon(aw, mu, sd, N, eps=eps, n_starts=n_cal, seed=s + 555,
                                                max_steps=max(3 * pred_cert, 400))["horizons"]
        pred_conf = conformal_horizon(cal, alpha=0.5)
        actual_conf = actual_cert                                  # same forecaster, fresh starts (s+777)

        def ratio(p, a):
            return (p / a) if (p and a) else None
        per_seed[s] = {
            "relmse": relmse,
            "cert": {"pred": pred_cert, "actual": actual_cert, "ratio": ratio(pred_cert, actual_cert),
                     "models": 1, "truth_rollouts": 0},
            "ensemble": {"pred": ens_pred["median"], "actual": actual_ens,
                         "ratio": ratio(ens_pred["median"], actual_ens), "models": K, "truth_rollouts": 0},
            "conformal": {"pred": pred_conf, "actual": actual_conf, "ratio": ratio(pred_conf, actual_conf),
                          "models": 1, "truth_rollouts": n_cal}}
        for arm in ("cert", "ensemble", "conformal"):
            a = per_seed[s][arm]
            print(f"[step90] seed {s} {arm:>9}: pred={a['pred']:.0f} actual={a['actual']:.0f} "
                  f"ratio={a['ratio']:.2f} (models={a['models']}, truth={a['truth_rollouts']})", file=sys.stderr)

    # summary: |log ratio| per arm (calibration error), averaged over seeds
    summary = {}
    for arm in ("cert", "ensemble", "conformal"):
        rs = [per_seed[s][arm]["ratio"] for s in seeds if per_seed[s][arm]["ratio"]]
        summary[arm] = {"ratios": rs, "mean_abs_log_ratio": float(np.mean([abs(np.log(r)) for r in rs])),
                        "models": per_seed[seeds[0]][arm]["models"],
                        "truth_rollouts": per_seed[seeds[0]][arm]["truth_rollouts"]}
    print(f"[step90] SUMMARY |log ratio| (lower=better-calibrated): "
          f"{ {a: round(v['mean_abs_log_ratio'], 3) for a, v in summary.items()} } | "
          f"costs: cert=1 model/0 truth, ensemble={K} models/0 truth, conformal=1 model/{10} truth", file=sys.stderr)
    return {"summary": summary, "per_seed": {str(k): v for k, v in per_seed.items()},
            "eps": eps, "N": N, "K": K, "smoke": SMOKE}


if __name__ == "__main__":
    torch.manual_seed(0)
    N = int(os.environ.get("STEP90_N", "10" if SMOKE else "40"))
    eps = float(os.environ.get("STEP90_EPS", "0.2"))
    K = int(os.environ.get("STEP90_K", "2" if SMOKE else "4"))
    seeds = [int(x) for x in os.environ.get("STEP90_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    print(f"[step90] UQ head-to-head: N={N} eps={eps} K={K} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
    res = run(seeds, N, eps, K)
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    tag += os.environ.get("STEP90_TAG", "")                    # per-shard suffix (parallel single-seed runs)
    (figdir / f"step90_uq_baselines{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0)
