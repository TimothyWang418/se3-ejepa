r"""Step 85 — Structure → trustworthy certificate → downstream win (direction ③).

Design: docs/specs/2026-06-08-step85-trustworthy-cert-downstream-design.md

THIS FILE IS CURRENTLY PHASE 0 ONLY (the G0 calibration smoke / go-no-go). Phases 1 (cert-isolated fixed-budget
frontier) and 2 (package companion) are added *after* G0 passes — see the spec. The point of Phase 0: the entire
direction rests on the **equivariant** certificate being calibrated at N=40 D2 (does the conv's certified horizon
predict the conv's *actual* forecast horizon?), while the non-equivariant certificate is wrong (its $\lambda_1$ is
inflated ~3-4x, already confirmed from the cached step74 spectra). This file MEASURES that, honestly.

We reuse, do NOT modify: step74 (N=40 conv/MLP architectures + training + attractor), step78 (bootstrap CI),
step79 (`certificate`, `certified_T1_steps`, `empirical_forecast_horizon` — the D2 machinery).

Everything is float64 on CPU (the Benettin-QR Jacobian needs float64; MPS has no float64). The conv/MLP from step74
are autonomous, model(x); step79's D2 functions want model(x, u) and call model.eval(), so we wrap in a tiny
nn.Module (a bare lambda has no .eval()).

Run (smoke, validate plumbing, ~1 min): STEP85_SMOKE=1 STEP85_SEEDS=0 .venv/bin/python experiments/step85_trustworthy_cert_downstream.py
Run (real G0, 3 seeds):                 .venv/bin/python experiments/step85_trustworthy_cert_downstream.py
Knobs: STEP85_N (40), STEP85_SEEDS (0,1,2), STEP85_EPS (0.2), STEP85_SMOKE (0).
Writes: papers/figures/step85_phase0_calibration.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as step74  # noqa: E402
import step78_certified_horizon_ci as step78  # noqa: E402  (imported for parity / future phases; step79 pulls it in)
import step79_certified_control as step79  # noqa: E402

DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP85_SMOKE", "0")))


class AutonomousWM(nn.Module):
    r"""Wrap an autonomous step74 world model ``m(x)`` as an action-conditioned ``model(x, u) = m(x)`` (control ignored,
    D2 is observation-only). An ``nn.Module`` (not a bare lambda) so step79's ``model.eval()`` calls work and propagate
    to the wrapped model. The signature bridge of the spec §3."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:  # (..., N), (..., N) -> (..., N)
        return self.m(x)


def train_one(kind: str, N: int, seed: int) -> tuple[nn.Module, torch.Tensor, torch.Tensor, float]:
    r"""Train ONE autonomous step74 model (``kind in {'conv','mlp'}``) on N-D Lorenz-96, returning the AutonomousWM
    wrapper, per-site ``(mu, sd)`` normalization stats, and the one-step relMSE. Reuses ``step74.train_model`` verbatim
    (it reads ``STEP74_MODEL`` / ``STEP74_SMOKE`` / ``STEP74_N``); we only set the env knobs and the data trajectory.
    CPU + float64 throughout (Benettin-QR needs float64)."""
    os.environ["STEP74_MODEL"] = kind
    os.environ["STEP74_SMOKE"] = "1" if SMOKE else "0"
    os.environ["STEP74_N"] = str(N)
    n_train = 4000 if SMOKE else 20000
    traj = step74.attractor_traj(N, n_train, seed, "cpu")            # (n_train+1, N) float64 on-attractor states
    model, mu, sd, relmse = step74.train_model(traj, N, "cpu", seed)  # builds conv/mlp per STEP74_MODEL; trains
    return AutonomousWM(model).double(), mu.double(), sd.double(), float(relmse)


def calibration(model: nn.Module, mu, sd, N: int, eps: float, seed: int,
                cert_n_steps: int, cert_warmup: int, n_boot: int, block: int, n_starts: int) -> dict:
    r"""The G0 measurement for one (trained) model: the certified forecast horizon (map steps) vs the EMPIRICAL forecast
    horizon (map steps), and their ratio (the calibration). Pure reuse of step79.

    - ``certificate`` → $\lambda_1$ (+ bootstrap CI) → $T_1(\epsilon)$; ``certified_T1_steps`` → the units-fixed horizon
      in MAP STEPS.
    - ``empirical_forecast_horizon`` → the median map-step at which the WM's free-running rel forecast error first
      exceeds $\epsilon$ (the load-bearing validation).
    - ``ratio_self`` $= H_{\rm emp}^{\rm median}/T_1^{\rm steps}$: ``~1`` ⇒ calibrated/trustworthy; ``>>1`` ⇒ the cert is
      pessimistic (horizon longer than claimed); ``<<1`` ⇒ optimistic (Prop-8 $\delta$-bias). step79's own criterion:
      within ~2-3x ⇒ predictive.
    """
    cert = step79.certificate(model, mu, sd, N, eps=eps, n_steps=cert_n_steps, warmup=cert_warmup,
                              n_boot=n_boot, block=block, seed=seed)
    T1_steps = step79.certified_T1_steps(cert)
    emp = step79.empirical_forecast_horizon(model, mu, sd, N, eps=eps, n_starts=n_starts, seed=seed,
                                            max_steps=max(3 * T1_steps, 600))
    H_emp = emp["median_empirical_horizon_steps"]
    ratio_self = (H_emp / T1_steps) if (T1_steps and T1_steps > 0) else None
    return {"lambda1": cert["lambda1"], "lambda1_ci": cert["lambda1_ci"], "T1_time": cert["T1"],
            "T1_steps": T1_steps, "H_emp_median": H_emp, "H_emp_mean": emp["mean"],
            "H_emp_p25": emp["p25"], "H_emp_p75": emp["p75"], "n_censored": emp["n_censored"],
            "ratio_self": ratio_self}


# --------------------------------------------------------------------------------------------------------------- #
# Phase 1 — certificate-isolated FIXED-BUDGET re-observation frontier (headline, A). The forecaster is held FIXED; the
# re-observation interval is DICTATED by a certificate; the observation BUDGET is swept. A too-short interval (from an
# inflated lambda1) covers less of the episode per budget -> the tail runs open-loop -> higher aggregate violation
# (starvation). This is the budgeted variant of step79.reobserve_run (step79 itself is NOT modified): re-observe every
# `interval` steps but only up to `budget` times, then run open-loop to the end, scoring aggregate violation over the
# WHOLE episode. Holding eps + the forecaster fixed and sweeping the budget is what makes a wrong lambda1 a genuine loss
# (spec section 2: a free eps/interval sweep would relabel eps on the SAME frontier; the budget is load-bearing).
# --------------------------------------------------------------------------------------------------------------- #
def reobserve_run_budgeted(model, mu, sd, N, x_seq_true, interval, eps, budget) -> dict:
    r"""Budgeted active re-observation: forecast a TRUE trajectory ``x_seq_true`` with ``model`` under $u\equiv0$,
    re-observing (resetting to truth) every ``interval`` map steps but only up to ``budget`` times total (the initial
    observation at $t{=}0$ counts as the first); once the budget is spent the forecast runs **open-loop** to the end.
    Returns the aggregate $\epsilon$-violation rate over all $T_{\rm total}$ forecast steps (re-observation steps
    contribute error 0), plus ``covered_steps`` $=(\,$n_observations$-1)\times$``interval`` (the last re-observation
    step). Mirrors :func:`step79.reobserve_run` with the budget cap added."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    interval = max(1, int(interval)); budget = max(1, int(budget))
    x_seq_true = x_seq_true.to(DTYPE)
    T_total = x_seq_true.shape[0] - 1
    n_obs = 1                                                          # the initial observation at t=0
    errors = []
    with torch.no_grad():
        z_hat = (x_seq_true[0] - mu) / sd                             # observed: WM state = true state at t=0
        for t in range(1, T_total + 1):
            z_hat = model(z_hat, torch.zeros(N, dtype=DTYPE))         # advance WM one u=0 step
            if (t % interval == 0) and (n_obs < budget):             # re-observe ONLY while budget remains
                z_hat = (x_seq_true[t] - mu) / sd
                n_obs += 1
                errors.append(0.0)                                   # looked here => not blind, error 0
            else:                                                     # blind step (interior, or budget exhausted)
                x_hat = z_hat * sd + mu
                rel = float((x_hat - x_seq_true[t]).norm() / x_seq_true[t].norm().clamp_min(1e-12))
                errors.append(rel)
    errors_np = np.asarray(errors, dtype=float)
    viol = float((errors_np > eps).mean()) if errors_np.size else 0.0
    covered_steps = (n_obs - 1) * interval                            # last re-observation step (0 if only the initial)
    return {"n_observations": n_obs, "violation_rate": viol,
            "max_error": float(errors_np.max()) if errors_np.size else 0.0,
            "mean_error": float(errors_np.mean()) if errors_np.size else 0.0,
            "interval": interval, "budget": budget, "covered_steps": covered_steps, "T_total": T_total}


def budget_frontier(model, mu, sd, N, x_seq_true, interval, eps, budgets) -> list:
    r"""Aggregate violation vs observation budget at a FIXED interval, forecaster, and true trajectory — one
    :func:`reobserve_run_budgeted` row per budget. Sweeping the budget traces the Phase-1 frontier; called with the conv
    forecaster and the conv-cert interval vs the MLP-cert interval, the conv-cert curve Pareto-dominates."""
    return [reobserve_run_budgeted(model, mu, sd, N, x_seq_true, interval, eps, b) for b in budgets]


def adaptive_reobserve(model, mu, sd, N, x_seq_true, eps, budget, init_interval: int = 1,
                       grow: int = 1, shrink: float = 0.5) -> dict:
    r"""CERTIFICATE-FREE adaptive re-observation baseline (the red-team control). AIMD on the interval: roll open-loop;
    at each scheduled re-observation measure the error that accumulated over the window — if it stayed $\le\epsilon$,
    **grow** the interval by ``grow`` (additive increase: the window was safe, wait longer next time); if it overshot,
    **shrink** to ``max(1, int(interval*shrink))`` (multiplicative decrease). Uses ONLY information available at
    re-observation (the observed error), no certificate. Honors the same ``budget`` as the cert schedules — once spent,
    runs open-loop to the end. This baseline can match/beat the certificate given enough budget; the certificate's edge
    is the a-priori (warm-start) cadence under a TIGHT budget — the honest scope of ③ (spec rev 2026-06-09a).

    Returns the aggregate violation rate, ``n_observations``, and ``mean_interval``/``max_interval``/``intervals`` (the
    realized re-observation periods), with the same accounting as :func:`reobserve_run_budgeted` (re-observed steps
    contribute error 0)."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    budget = max(1, int(budget))
    x_seq_true = x_seq_true.to(DTYPE)
    T_total = x_seq_true.shape[0] - 1
    I = max(1, int(init_interval))
    n_obs = 1
    steps_since = 0
    errors = []
    intervals = []
    with torch.no_grad():
        z_hat = (x_seq_true[0] - mu) / sd
        for t in range(1, T_total + 1):
            z_hat = model(z_hat, torch.zeros(N, dtype=DTYPE))
            steps_since += 1
            x_hat = z_hat * sd + mu
            rel = float((x_hat - x_seq_true[t]).norm() / x_seq_true[t].norm().clamp_min(1e-12))
            if steps_since >= I and n_obs < budget:               # scheduled re-observation (budget permitting)
                intervals.append(steps_since)
                errors.append(0.0)                                # looked here => not blind
                I = (I + grow) if rel <= eps else max(1, int(I * shrink))   # AIMD on the observed window error
                z_hat = (x_seq_true[t] - mu) / sd
                n_obs += 1
                steps_since = 0
            else:                                                  # blind step (interior, or budget exhausted)
                errors.append(rel)
    errors_np = np.asarray(errors, dtype=float)
    viol = float((errors_np > eps).mean()) if errors_np.size else 0.0
    return {"n_observations": n_obs, "violation_rate": viol,
            "max_error": float(errors_np.max()) if errors_np.size else 0.0,
            "mean_error": float(errors_np.mean()) if errors_np.size else 0.0,
            "intervals": intervals,
            "mean_interval": float(np.mean(intervals)) if intervals else 0.0,
            "max_interval": int(max(intervals)) if intervals else 0,
            "budget": budget, "T_total": T_total}


def run_phase0(seeds, N: int, eps: float) -> dict:
    r"""Phase 0 / gate G0: for each seed, train conv + MLP at N, measure each one's calibration, and the cert-isolated
    cross ratio $\rho_\times = H_{\rm emp}^{\rm conv}/T_1^{\rm steps,MLP}$ (does the MLP's CERTIFICATE mistime a FAITHFUL
    forecaster's actual horizon?). Prints a table; computes the G0 verdict: conv ``ratio_self`` in the calibrated band
    [1/2, 2] on >= 2/3 seeds."""
    # Smoke vs full Benettin-QR / bootstrap / empirical budgets.
    cert_n_steps = 600 if SMOKE else 2000
    cert_warmup = 100 if SMOKE else 400
    n_boot = 80 if SMOKE else 300
    block = 30 if SMOKE else 50
    n_starts = 16 if SMOKE else 40

    per_seed = {}
    for s in seeds:
        print(f"[step85.G0] seed {s}: training conv + MLP at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        conv, mu_c, sd_c, relmse_c = train_one("conv", N, s)
        mlp, mu_m, sd_m, relmse_m = train_one("mlp", N, s)
        cal_conv = calibration(conv, mu_c, sd_c, N, eps, s, cert_n_steps, cert_warmup, n_boot, block, n_starts)
        cal_mlp = calibration(mlp, mu_m, sd_m, N, eps, s, cert_n_steps, cert_warmup, n_boot, block, n_starts)
        # cert-isolated cross ratio: faithful forecaster's actual horizon vs the MLP certificate's claimed horizon.
        cross = (cal_conv["H_emp_median"] / cal_mlp["T1_steps"]) if cal_mlp["T1_steps"] else None
        cal_conv["one_step_relmse"] = relmse_c
        cal_mlp["one_step_relmse"] = relmse_m
        per_seed[s] = {"conv": cal_conv, "mlp": cal_mlp, "cross_ratio_convEmp_over_mlpCert": cross,
                       "lambda1_inflation_mlp_over_conv": cal_mlp["lambda1"] / cal_conv["lambda1"]}
        print(f"[step85.G0] seed {s}: "
              f"conv l1={cal_conv['lambda1']:.3f} T1s={cal_conv['T1_steps']} Hemp={cal_conv['H_emp_median']:.0f} "
              f"ratio={cal_conv['ratio_self']:.2f} (relMSE {relmse_c:.3f}) | "
              f"mlp l1={cal_mlp['lambda1']:.3f} T1s={cal_mlp['T1_steps']} Hemp={cal_mlp['H_emp_median']:.0f} "
              f"ratio={cal_mlp['ratio_self']:.2f} | inflation={per_seed[s]['lambda1_inflation_mlp_over_conv']:.2f}x | "
              f"cross(convEmp/mlpCert)={cross:.2f}", file=sys.stderr)

    # G0 verdict: conv calibrated (ratio_self in [1/2, 2]) on >= 2/3 seeds.
    conv_ratios = [per_seed[s]["conv"]["ratio_self"] for s in seeds]
    conv_calibrated = [r is not None and 0.5 <= r <= 2.0 for r in conv_ratios]
    n_ok = int(sum(conv_calibrated))
    g0_pass = bool(n_ok >= int(np.ceil(2 / 3 * len(seeds))))
    inflations = [per_seed[s]["lambda1_inflation_mlp_over_conv"] for s in seeds]

    verdict = {"G0_pass": g0_pass, "n_conv_calibrated": n_ok, "n_seeds": len(seeds),
               "conv_ratios_self": conv_ratios, "conv_calibrated_band_[0.5,2]": conv_calibrated,
               "mlp_lambda1_inflation": inflations, "eps": eps, "N": N, "smoke": SMOKE}
    print(f"[step85.G0] {'G0 PASS' if g0_pass else 'G0 INCONCLUSIVE'}: conv calibrated on {n_ok}/{len(seeds)} seeds "
          f"(ratios {[round(r, 2) for r in conv_ratios]}); MLP lambda1 inflated "
          f"{[round(i, 2) for i in inflations]}x (so its certified horizon is that-many-x too short). "
          f"{'Proceed to Phase 1.' if g0_pass else 'Direction ③ rests on the conv cert being calibrated — report honestly.'}",
          file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


def build_true_traj(N: int, L: int, seed: int) -> torch.Tensor:
    r"""One long TRUE Lorenz-96 trajectory under $u\equiv0$, ``(L+1, N)`` raw states, launched from a seed-pinned
    on-attractor point. Mirrors the trajectory construction in :func:`step79.reobservation_contrast`."""
    traj = step74.attractor_traj(N, 4 * L, seed, "cpu").double()
    g = torch.Generator().manual_seed(seed + 777)
    j0 = int(torch.randint(0, traj.shape[0] - L - 1, (1,), generator=g).item())
    x_seq = torch.empty(L + 1, N, dtype=DTYPE)
    x_seq[0] = traj[j0]
    x_cur = traj[j0].clone()
    for t in range(1, L + 1):
        x_cur = step79.rk4_controlled(x_cur, torch.zeros(N, dtype=DTYPE))
        x_seq[t] = x_cur
    return x_seq


def _viol_at(front: list, B: int) -> float:
    r"""Violation rate at the frontier row whose budget is nearest ``B``."""
    return min(front, key=lambda r: abs(r["budget"] - B))["violation_rate"]


def run_phase1(seeds, N: int, eps: float, L: int) -> dict:
    r"""Phase 1 (A, headline) + Phase 2 (D, package) + the cert-free adaptive baseline. Per seed: train conv + MLP;
    read the conv certificate's **conservative** horizon ($T_1^{\rm lo}$, the sound side — spec rev 2026-06-09a) and the
    MLP certificate's (inflated, too-short) horizon; on ONE shared true trajectory sweep the observation budget for:
      * conv forecaster + conv interval  (= Agent E, and the conv-cert arm of the cert-isolated contrast),
      * conv forecaster + MLP interval   (= the cert-ISOLATED MLP arm: same forecaster, only the wrong lambda1 differs),
      * MLP forecaster  + MLP interval   (= Agent N, the realistic package),
      * cert-FREE adaptive (conv forecaster).
    Gate G1: conv-cert arm aggregate violation < MLP-cert arm at the knee budget $B^\*={\rm ceil}(L/{\rm iv\_conv})$ on
    $\ge2/3$ seeds (cert-isolated, so the gap is the lambda1 ratio)."""
    ck = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
              n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)
    per_seed = {}
    for s in seeds:
        print(f"[step85.P1] seed {s}: training conv + MLP at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        conv, mu_c, sd_c, _ = train_one("conv", N, s)
        mlp, mu_m, sd_m, _ = train_one("mlp", N, s)
        cert_c = step79.certificate(conv, mu_c, sd_c, N, eps=eps, seed=s, **ck)
        cert_m = step79.certificate(mlp, mu_m, sd_m, N, eps=eps, seed=s, **ck)
        iv_conv = step79.certified_T1_steps({"T1": cert_c.get("T1_lo") or cert_c.get("T1")})  # conservative (sound) end
        iv_mlp = step79.certified_T1_steps({"T1": cert_m.get("T1")})                          # MLP's point (too short)
        x_seq = build_true_traj(N, L, s)

        cover_conv = max(1, int(np.ceil(L / max(1, iv_conv))))     # budget for conv to just cover the episode (the knee)
        cover_mlp = max(1, int(np.ceil(L / max(1, iv_mlp))))       # budget for MLP to cover (larger, since iv_mlp short)
        budgets = sorted({2, cover_conv, cover_mlp}                          # include both knees as exact grid points
                         | {int(round(b)) for b in np.linspace(2, max(6, int(np.ceil(cover_mlp * 1.3))), 8)})

        conv_arm = budget_frontier(conv, mu_c, sd_c, N, x_seq, iv_conv, eps, budgets)   # Agent E / conv-cert arm
        mlp_iso = budget_frontier(conv, mu_c, sd_c, N, x_seq, iv_mlp, eps, budgets)     # cert-ISOLATED MLP arm
        mlp_pkg = budget_frontier(mlp, mu_m, sd_m, N, x_seq, iv_mlp, eps, budgets)      # Agent N (package)
        adapt = [adaptive_reobserve(conv, mu_c, sd_c, N, x_seq, eps, b, init_interval=1) for b in budgets]

        # G1 at the knee (conv just covers; MLP-iso still starved). Pareto across the whole sweep too.
        cv, mv = _viol_at(conv_arm, cover_conv), _viol_at(mlp_iso, cover_conv)
        pareto = float(np.mean([c["violation_rate"] < m["violation_rate"] for c, m in zip(conv_arm, mlp_iso)]))
        per_seed[s] = {
            "iv_conv": iv_conv, "iv_mlp": iv_mlp, "lambda1_conv": cert_c["lambda1"], "lambda1_mlp": cert_m["lambda1"],
            "knee_budget": cover_conv, "cover_mlp_budget": cover_mlp,
            "conv_arm_viol_at_knee": cv, "mlp_iso_viol_at_knee": mv, "win_margin": mv - cv,
            "pareto_frac_conv_below_mlp": pareto,
            "frontier": {"budgets": budgets,
                         "conv_arm": [r["violation_rate"] for r in conv_arm],
                         "mlp_iso": [r["violation_rate"] for r in mlp_iso],
                         "mlp_pkg": [r["violation_rate"] for r in mlp_pkg],
                         "adaptive": [a["violation_rate"] for a in adapt]}}
        print(f"[step85.P1] seed {s}: iv_conv={iv_conv} iv_mlp={iv_mlp} knee B*={cover_conv} | "
              f"viol@knee conv={cv:.3f} mlp(iso)={mv:.3f} margin={mv-cv:+.3f} | "
              f"Pareto(conv<mlp) {pareto:.0%}", file=sys.stderr)

    margins = [per_seed[s]["win_margin"] for s in seeds]
    wins = [m > 0.05 for m in margins]                              # clear margin (>5% of episode) at the knee
    n_win = int(sum(wins))
    g1_pass = bool(n_win >= int(np.ceil(2 / 3 * len(seeds))))
    verdict = {"G1_pass": g1_pass, "n_win": n_win, "n_seeds": len(seeds), "win_margins": margins,
               "eps": eps, "N": N, "L": L, "smoke": SMOKE}
    print(f"[step85.P1] {'G1 PASS' if g1_pass else 'G1 INCONCLUSIVE'}: conv-cert arm beats MLP-cert arm at the knee on "
          f"{n_win}/{len(seeds)} seeds (margins {[round(m, 3) for m in margins]}). Cert-isolated ⇒ the gap is the "
          f"lambda1 ratio. {'Proceed: Phase 2 package + figure + 3080.' if g1_pass else 'Report honestly.'}",
          file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


def _save(res: dict, name: str) -> None:
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"step85_{name}{tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    N = int(os.environ.get("STEP85_N", "10" if SMOKE else "40"))
    eps = float(os.environ.get("STEP85_EPS", "0.2"))
    seeds = [int(x) for x in os.environ.get("STEP85_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    phase = os.environ.get("STEP85_PHASE", "1")               # "0" = G0 calibration only; "1" = Phase 1+2 (default)
    if phase == "0":
        print(f"[step85.G0] Phase 0 calibration: N={N}, eps={eps}, seeds={seeds}, smoke={SMOKE}", file=sys.stderr)
        res = run_phase0(seeds, N, eps)
        _save(res, "phase0_calibration")
        raise SystemExit(0 if res["verdict"]["G0_pass"] else 1)
    L = int(os.environ.get("STEP85_L", "300" if SMOKE else "1500"))
    print(f"[step85.P1] Phase 1+2: N={N}, eps={eps}, L={L}, seeds={seeds}, smoke={SMOKE}", file=sys.stderr)
    res = run_phase1(seeds, N, eps, L)
    _save(res, "phase1_frontier")
    raise SystemExit(0 if res["verdict"]["G1_pass"] else 1)
