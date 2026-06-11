r"""P4 5-run extension — resolves the 3-seed asterisk under the stability-conditioned reading.

Registered aggregation (declared here, before the runs): qualifying run = latent_std ≥ 0.7;
a gate **PASSES at n=5 iff qualifying runs ≥ 3 AND pass-fraction among qualifying ≥ 2/3**;
fewer than 3 qualifying ⇒ INCONCLUSIVE-BY-STABILITY (reported, not retried-until-pass).
Language: runs not seeds (eq training is run-nondeterministic on MPS — record 2026-06-11);
run labels s3, s4 are bookkeeping.

Loads the 3-seed artifact's cells verbatim (no re-running of evaluated cells), adds 4 cells
(runs 3,4 × {eq, plain}), re-aggregates all gates over 5 runs.

Run: .venv/bin/python experiments/p4_5run_extension.py   (~10 min)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_spine_stage1a import boundary_from_curve, fit_growth, measured_err  # noqa: E402
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder,
    to_transitions,
)
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

NEW_RUNS = (3, 4)
ALL_RUNS = (0, 1, 2, 3, 4)
W_AUDIT, B_AUDIT, H_MAX = 16, 4, 8
EPS_MULT = (2, 4, 8, 16)
D = 128
BAND = (0.5, 2.0)
FLOOR = 0.7
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PRIOR = ROOT / "papers" / "figures" / "p4_3seed_spine.json"
OUT = ROOT / "papers" / "figures" / "p4_5run_spine.json"


def in_band(x) -> bool:
    return x is not None and BAND[0] <= x <= BAND[1]


def main() -> int:
    t0 = time.time()
    prior = json.loads(PRIOR.read_text())
    cells: dict = dict(prior["cells"])  # s0..s2 verbatim

    print("[1/3] fixed corpus + held-out ...")
    corpus = collect_weakpolicy(200, seed=0)
    obs, act, nxt = to_transitions(corpus, 200)
    ho = collect_weakpolicy(60, seed=1)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    print("[2/3] 4 new cells (runs 3,4) ...")
    for run in NEW_RUNS:
        for name, builder in (("eq", build_eq), ("plain", lambda: build_plain(w_match))):
            torch.manual_seed(run)
            m = builder()
            hist, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE,
                                   seed=run, verbose=False, return_target_encoder=True,
                                   refresh_target_cache=True)
            torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                       DATA_DIR / f"ckpt5_{name}_s{run}.pt")
            pr = Pair(tgt.eval().cpu(), m.predictor.cpu())
            rep = audit_gap(pr, fb, ach, window=W_AUDIT, k=3, burn_in=B_AUDIT, h_max=H_MAX)
            meas = measured_err(pr, fb, ach, H_MAX)
            d_mean = rep["delta"]["mean"]
            gfit = fit_growth(meas["median"])
            cal = []
            for em in EPS_MULT:
                eps = em * d_mean
                hc = boundary_from_curve(rep["certified_curve"]["err_q90"], eps)
                hm = boundary_from_curve(meas["q90"], eps)
                cal.append({"eps_mult": em, "H_cert": hc, "H_meas": hm,
                            "ratio": (hc / hm) if hm else None})
            cells[f"{name}_s{run}"] = {
                "pred_loss": hist["pred_loss"][-1], "latent_std": hist["latent_std"],
                "delta_mean": d_mean, "delta_q90": rep["delta"]["q90"],
                "delta_norm": d_mean / (max(hist["latent_std"], 1e-6) * math.sqrt(D)),
                "lambda_tangent": rep["lambda1"]["mean"],
                "measured_median": meas["median"],
                "growth": {k: gfit[k] for k in ("lambda_meas_exp_fit", "linear_rate",
                                                "r2_exp", "r2_linear")},
                "rate_ratio_cert_over_meas": (d_mean / gfit["linear_rate"])
                if gfit["linear_rate"] > 1e-9 else None,
                "c3cal": cal,
            }
            c = cells[f"{name}_s{run}"]
            print(f"    {name} run{run}: std {c['latent_std']:.3f} d {d_mean:.3f} "
                  f"d_norm {c['delta_norm']:.3f} cal {[x['ratio'] for x in c['c3cal']]}")

    print("[3/3] 5-run stability-conditioned aggregation ...")
    verdicts: dict = {}
    for name in ("eq", "plain"):
        qual = [r for r in ALL_RUNS if cells[f"{name}_s{r}"]["latent_std"] >= FLOOR]
        cal_pass = [r for r in qual
                    if all(in_band(x["ratio"]) for x in cells[f"{name}_s{r}"]["c3cal"])]
        shape_pass = [r for r in qual
                      if cells[f"{name}_s{r}"]["growth"]["r2_linear"]
                      > cells[f"{name}_s{r}"]["growth"]["r2_exp"]
                      and in_band(cells[f"{name}_s{r}"]["rate_ratio_cert_over_meas"])]
        def verdict(passing):
            if len(qual) < 3:
                return "INCONCLUSIVE-BY-STABILITY"
            return "PASS" if len(passing) / len(qual) >= 2 / 3 else "FAIL"
        verdicts[name] = {
            "qualifying_runs": qual,
            "collapse_rate": f"{len(ALL_RUNS) - len(qual)}/{len(ALL_RUNS)}",
            "c3cal": {"passing": cal_pass, "verdict": verdict(cal_pass)},
            "gpre_shape": {"passing": shape_pass, "verdict": verdict(shape_pass)},
        }
        print(f"    {name}: qual {qual} collapse {verdicts[name]['collapse_rate']} "
              f"c3cal {verdicts[name]['c3cal']['verdict']} ({cal_pass}) "
              f"shape {verdicts[name]['gpre_shape']['verdict']} ({shape_pass})")
    pairs = [(cells[f"plain_s{r}"]["delta_norm"] / cells[f"eq_s{r}"]["delta_norm"], r)
             for r in ALL_RUNS
             if cells[f"eq_s{r}"]["latent_std"] >= FLOOR
             and cells[f"plain_s{r}"]["latent_std"] >= FLOOR]
    verdicts["moat_stability_conditioned"] = {
        "per_run": {r: round(v, 2) for v, r in pairs},
        "mean": round(float(np.mean([v for v, _ in pairs])), 2) if pairs else None,
        "n": len(pairs),
    }
    print(f"    moat (conditioned): {verdicts['moat_stability_conditioned']}")

    art = {"runs": list(ALL_RUNS), "floor": FLOOR, "band": list(BAND),
           "aggregation": "qualifying>=3 AND pass-fraction>=2/3; else INCONCLUSIVE-BY-STABILITY",
           "language": "runs not seeds (eq is run-nondeterministic on MPS)",
           "cells": cells, "verdicts": verdicts,
           "wall_sec": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(art, indent=1))
    print(f"wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
