r"""P4 plain-cliff probe — locate plain's data-efficiency cliff in (2606, 4000) transitions.

Context (ledger 2026-06-11): plain δ̂ ≈ 9.3–10.5 at 2606 transitions, ≈ 3.8 at 4000; eq is flat
across the range. This probe brackets the cliff: counts {3000, 3500} × n=2 (plain ep40-winner
recipe, unfiltered wedge transitions, fixed-rng subsample per count), δ̂ on wedge_ho_in.
Descriptive only — no gate. Designed for the backup MacBook (CPU; auto device fallback).

Run (backup Mac): cd ~/Workspace/se3-ejepa && nohup .venv/bin/python -u \
    experiments/p4_plain_cliff.py > ~/p4_cliff.log 2>&1 &   (~3 h CPU)
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
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
COUNTS = (3000, 3500)
OUT = ROOT / "papers" / "figures" / "p4_plain_cliff.json"


def main() -> int:
    art: dict = {"design": "plain ep40 x unfiltered wedge transitions, counts x n=2", "cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    for n_count in COUNTS:
        for r in range(2):
            key = f"c{n_count}_r{r}"
            try:
                rng = np.random.default_rng(100 * r)               # fixed per run-rank
                idx = torch.from_numpy(rng.choice(obs.shape[0], n_count, replace=False)).long()
                cell, pr, *_ = run_one(lambda: build_plain(w_match), PLAIN_CFG, r,
                                       obs[idx], act[idx], nxt[idx],
                                       fb_in, ach_in, fb_small, ho_in, aux_all[idx])
                if cell["stable"]:
                    rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
                    cell["in_delta"] = round(rep["delta"]["mean"], 3)
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][key] = cell
            art["elapsed_min"] = round((time.time() - T0) / 60, 1)
            OUT.write_text(json.dumps(art, indent=1))
            print(f"  {key}: std {cell.get('std', float('nan')):.3f} in-δ {cell.get('in_delta')}")

    pts = {}
    for n_count in COUNTS:
        ds = [c["in_delta"] for k, c in art["cells"].items()
              if k.startswith(f"c{n_count}") and c.get("in_delta")]
        pts[n_count] = ds
    art["curve"] = {"2606(ref)": [9.33, 10.54], **{str(k): v for k, v in pts.items()},
                    "4000(ref)": [3.8]}
    OUT.write_text(json.dumps(art, indent=1))
    print("CURVE:", art["curve"])
    print(f"CLIFF PROBE DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
