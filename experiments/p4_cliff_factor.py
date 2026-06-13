r"""T1.2 — cliff factor separation: is the low-N failure SEED-driven or SUBSAMPLE-driven?

Closes the registered rank-correlation disclosure (06-12): in the cliff cells, training seed and
subsample were bundled per run-rank, so P(fail) cells were partially rank-correlated. This 2×2×3
isolates the two factors at N ∈ {2606, 3000, 3500}:
  - **seed sweep** (which-init): FIX subsample (rng seed 0), vary training seed r=0..7
  - **subsample sweep** (which-data): FIX training seed 0, vary subsample rng=100·r, r=0..7
Readout (descriptive, no gate): variance of δ̂ explained by seed vs by subsample. If failure is
seed-dominated ⇒ it's a training-init lottery; if subsample-dominated ⇒ it's a which-episodes
effect. Either way the marginal P(fail) stands; this characterizes its structure.

Run: nohup .venv/bin/python -u experiments/p4_cliff_factor.py > /tmp/p4_cliff_factor.log 2>&1 &
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
N_VALUES = (2606, 3000, 3500)
N_REP = 8
OUT = ROOT / "papers" / "figures" / "p4_cliff_factor.json"


def main() -> int:
    art: dict = {"design": "2x2x3 seed-vs-subsample factor separation", "cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    builder = lambda: build_plain(w_match)  # noqa: E731

    def one(n_count: int, sub_seed: int, train_seed: int) -> dict:
        rng = np.random.default_rng(sub_seed)
        idx = torch.from_numpy(rng.choice(obs.shape[0], n_count, replace=False)).long()
        cell, pr, *_ = run_one(builder, PLAIN_CFG, train_seed,
                               obs[idx], act[idx], nxt[idx],
                               fb_in, ach_in, fb_small, ho_in, aux_all[idx])
        if cell.get("stable"):
            rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
            cell["in_delta"] = round(rep["delta"]["mean"], 3)
        return {k: v for k, v in cell.items() if k in ("std", "in_delta", "stable", "error")}

    for n_count in N_VALUES:
        for mode in ("seed", "subsample"):
            for r in range(N_REP):
                key = f"N{n_count}_{mode}_r{r}"
                try:
                    if mode == "seed":          # fix subsample (0), vary train seed
                        cell = one(n_count, sub_seed=0, train_seed=r)
                    else:                        # fix train seed (0), vary subsample
                        cell = one(n_count, sub_seed=100 * r, train_seed=0)
                except Exception as exc:  # noqa: BLE001
                    cell = {"error": str(exc)[:160], "stable": False}
                art["cells"][key] = cell
                art["elapsed_min"] = round((time.time() - T0) / 60, 1)
                OUT.write_text(json.dumps(art, indent=1))
                print(f"  {key}: std {cell.get('std')} δ {cell.get('in_delta')}")

    # factor analysis
    summ = {}
    for n_count in N_VALUES:
        seed_ds = [art["cells"][f"N{n_count}_seed_r{r}"].get("in_delta")
                   for r in range(N_REP)]
        sub_ds = [art["cells"][f"N{n_count}_subsample_r{r}"].get("in_delta")
                  for r in range(N_REP)]
        seed_ds = [d for d in seed_ds if d]
        sub_ds = [d for d in sub_ds if d]
        summ[str(n_count)] = {
            "seed_var": round(float(np.var(seed_ds)), 3) if len(seed_ds) > 1 else None,
            "subsample_var": round(float(np.var(sub_ds)), 3) if len(sub_ds) > 1 else None,
            "seed_fail": sum(d > 6 for d in seed_ds), "sub_fail": sum(d > 6 for d in sub_ds),
        }
    art["factor_summary"] = summ
    art["verdict"] = "see per-N seed_var vs subsample_var (larger = dominant factor)"
    OUT.write_text(json.dumps(art, indent=1))
    print("FACTOR SUMMARY:", json.dumps(summ))
    print(f"CLIFF FACTOR DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
