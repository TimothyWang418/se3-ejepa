r"""Cliff-exit pinning — completes the P(fail)-vs-N figure (Mac MPS).

plain @3500 × n=6 (exit currently rests on n=2) + eq @1750 × n=3 (eq's cliff is bracketed
(1500, 2000) by 1/3-vs-0/3 — thin). Protocol = cliff-probe verbatim (unfiltered wedge
transitions, fixed-rng subsample per run-rank, δ̂ on wedge_ho_in).

Run: nohup .venv/bin/python -u experiments/p4_cliff_exit.py > /tmp/p4_cliff_exit.log 2>&1 &
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
from experiments.p4_wedge_lane import EQ_CFG, PLAIN_CFG, load_set  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_cliff_exit.json"


def main() -> int:
    art: dict = {"cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    for arm, builder, cfg, n_count, n_runs in (
            ("plain", lambda: build_plain(w_match), PLAIN_CFG, 3500, 6),
            ("eq", build_eq, EQ_CFG, 1750, 3)):
        for r in range(n_runs):
            key = f"{arm}_c{n_count}_r{r}"
            try:
                rng = np.random.default_rng(100 * r)
                idx = torch.from_numpy(rng.choice(obs.shape[0], n_count, replace=False)).long()
                cell, pr, *_ = run_one(builder, cfg, r, obs[idx], act[idx], nxt[idx],
                                       fb_in, ach_in, fb_small, ho_in, aux_all[idx])
                if cell["stable"]:
                    rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
                    cell["in_delta"] = round(rep["delta"]["mean"], 3)
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][key] = {k: v for k, v in cell.items()
                                 if k in ("std", "xy", "in_delta", "stable", "error")}
            art["elapsed_min"] = round((time.time() - T0) / 60, 1)
            OUT.write_text(json.dumps(art, indent=1))
            print(f"  {key}: std {cell.get('std', float('nan')):.2f} δ {cell.get('in_delta')}")
    print(f"CLIFF EXIT DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
