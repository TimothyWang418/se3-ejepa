r"""Wedge-quirk probe — why does in-wedge filtering IMPROVE eq? (registered mystery, ledger 06-11)

Observed: eq @ filtered-2606 δ̂ = 2.49 < eq @ unfiltered-2606 δ̂ = 3.21 (same count, same
recipe). Hypotheses:
- **H-A (angular homogeneity):** narrow absolute-angle coverage simplifies the latent geometry.
- **H-C (motion-selection bias):** windows whose angle never leaves the wedge are also windows
  where the block ROTATES LESS — dynamically quieter windows are mechanically easier to predict.

Discriminating arm: **low-rotation-matched, angle-UNRESTRICTED** — rank all 4000 windows by
within-window |Δθ| ascending, take 2606. If its δ̂ ≈ 2.49 (filtered) ⇒ H-C; if ≈ 3.21
(unfiltered-random) ⇒ H-A. n = 4, winner recipe (the quirk's original arms), δ̂ on wedge_ho_in.

Run: nohup .venv/bin/python -u experiments/p4_wedge_quirk.py > /tmp/p4_quirk.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_cpu_batch_0612 import tensorize  # noqa: E402
from experiments.p4_step1_pipeline import CHUNK, RES, build_eq, to_transitions  # noqa: E402
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from experiments.p4_wedge_lane import EQ_CFG, load_set  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_wedge_quirk.json"
N_KEEP = 2606                                       # match the filtered arm's count exactly


def main() -> int:
    art: dict = {"refs": {"eq_filtered_2606": 2.49, "eq_unfiltered_2606": 3.21},
                 "arm": "low-rotation-matched, angle-unrestricted", "cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    # within-window |dtheta| for every chunk window, transition order
    ang = torch.from_numpy(corpus["states"][:, :, 4]).double() % (2 * np.pi)
    n_ch = corpus["actions"].shape[1] // CHUNK
    dth_cols = []
    for j in range(n_ch):
        w = ang[:, j * CHUNK : j * CHUNK + CHUNK + 1]
        d = (w.max(dim=1).values - w.min(dim=1).values)
        d = torch.minimum(d, 2 * np.pi - d)
        dth_cols.append(d)
    dth = torch.stack(dth_cols, dim=1).reshape(-1)                # (4000,)
    idx = dth.argsort()[:N_KEEP]                                  # low-rotation 2606
    art["selected_dtheta_q90_deg"] = round(float(np.degrees(
        torch.quantile(dth[idx], 0.9).item())), 2)
    o_, a_, n_, x_ = obs[idx], act[idx], nxt[idx], aux_all[idx]

    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()

    for r in range(4):
        try:
            cell, pr, *_ = run_one(build_eq, EQ_CFG, r, o_, a_, n_,
                                   fb_in, ach_in, fb_small, ho_in, x_)
            if cell["stable"]:
                rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
                cell["in_delta"] = round(rep["delta"]["mean"], 3)
        except Exception as exc:  # noqa: BLE001
            cell = {"error": str(exc)[:200], "stable": False}
        art["cells"][f"lowrot_r{r}"] = {k: v for k, v in cell.items()
                                        if k in ("std", "xy", "in_delta", "stable", "error")}
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))
        print(f"  lowrot r{r}: std {cell.get('std', float('nan')):.2f} δ {cell.get('in_delta')}")

    ds = [c["in_delta"] for c in art["cells"].values() if c.get("in_delta")]
    if ds:
        med = float(np.median(ds))
        art["readout"] = {"median": round(med, 3),
                          "verdict": ("H-C (motion bias)" if abs(med - 2.49) < abs(med - 3.21)
                                      else "H-A (angular homogeneity)")}
    OUT.write_text(json.dumps(art, indent=1))
    print("READOUT:", art.get("readout"))
    print(f"QUIRK PROBE DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
