r"""Wedge v2 addendum — plain count-control (registered before launch, same review pass).

v2 left plain's δ̂ explosion confounded: plain_w2 (filtered@2606) median 10.54 vs v1 plain
(unfiltered@4000) ≈ 3.8. Two hypotheses: (a) transition COUNT (data lever), (b) wedge
RESTRICTION (angular diversity). This arm: plain recipe × UNFILTERED@2606 (the exact eq_ctrl
data), n=4. Readout (registered): if plain_ctrl ≈ 3.8 ⇒ restriction kills plain (eq exploits
restricted data via structure — the C1b-class mechanism); if ≈ 10.5 ⇒ count (sample-efficiency
moat at low N). Either way the measured 4.2× restricted-data moat stands; only attribution moves.

Run: nohup .venv/bin/python -u experiments/p4_wedge_plainctrl.py  (~30 min)
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
from experiments.p4_step1_pipeline import RES, build_eq, build_plain, pick_ladder, to_transitions  # noqa: E402
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from experiments.p4_wedge_lane import PLAIN_CFG, load_set  # noqa: E402
from experiments.p4_wedge_v2 import window_mask  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_wedge_plainctrl.json"


def main() -> int:
    art: dict = {"design": "plain recipe x UNFILTERED@2606 (eq_ctrl's exact data), n=4", "cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    mask = window_mask(corpus, 200)
    n_filt = int(mask.sum())
    rng = np.random.default_rng(0)                                # SAME control subsample as v2
    idx = torch.from_numpy(rng.choice(mask.numel(), n_filt, replace=False)).long()
    o_, a_, n_, x_ = obs[idx], act[idx], nxt[idx], aux_all[idx]
    ho_in, ho_out = load_set("wedge_ho_in"), load_set("wedge_ho_out")
    fb_in, ach_in = tensorize(ho_in)
    fb_out, ach_out = tensorize(ho_out)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    for r in range(4):
        try:
            cell, pr, m, tgt = run_one(lambda: build_plain(w_match), PLAIN_CFG, r,
                                       o_, a_, n_, fb_in, ach_in, fb_small, ho_in, x_)
            if cell["stable"]:
                for tag, fb, ach in (("in", fb_in, ach_in), ("out", fb_out, ach_out)):
                    rep = audit_gap(pr, fb, ach, window=16, k=3, burn_in=4, h_max=8)
                    cell[tag] = {"delta": round(rep["delta"]["mean"], 3)}
                cell["out_in_ratio"] = round(cell["out"]["delta"] / cell["in"]["delta"], 3)
        except Exception as exc:  # noqa: BLE001
            cell = {"error": str(exc)[:200], "stable": False}
        art["cells"][f"plain_ctrl_r{r}"] = cell
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))
        print(f"  plain_ctrl r{r}: std {cell.get('std', float('nan')):.3f} "
              f"in {cell.get('in', {}).get('delta')} out {cell.get('out', {}).get('delta')}")

    st = [c for c in art["cells"].values() if c.get("stable") and "in" in c]
    if st:
        med = float(np.median([c["in"]["delta"] for c in st]))
        art["readout"] = {"median_in_delta": round(med, 2),
                          "attribution": ("RESTRICTION kills plain (count innocent)" if med < 6
                                          else "COUNT (data lever at low N)")}
    OUT.write_text(json.dumps(art, indent=1))
    print("READOUT:", art.get("readout"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
