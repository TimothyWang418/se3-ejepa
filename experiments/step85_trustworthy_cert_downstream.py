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


def _save(res: dict) -> None:
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"step85_phase0_calibration{tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    N = int(os.environ.get("STEP85_N", "10" if SMOKE else "40"))
    eps = float(os.environ.get("STEP85_EPS", "0.2"))
    seeds = [int(x) for x in os.environ.get("STEP85_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    print(f"[step85.G0] Phase 0 calibration: N={N}, eps={eps}, seeds={seeds}, smoke={SMOKE}", file=sys.stderr)
    res = run_phase0(seeds, N, eps)
    _save(res)
    raise SystemExit(0 if res["verdict"]["G0_pass"] else 1)
