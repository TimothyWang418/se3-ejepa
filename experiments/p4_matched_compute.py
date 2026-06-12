r"""Matched-COMPUTE moat row (skeleton hole #2) — does compute rescue plain at low N?

The moat rows are param-matched; eq's e2cnn forward costs more per step, so eq gets MORE
compute at equal epochs — the standard reviewer attack on claim 4. This row: measure the
per-epoch wall-time ratio R = t_eq/t_plain at 2606 transitions, then train plain with
epochs = min(round(40·R), 300) (cap disclosed) × n=4 @2606 unfiltered wedge transitions.

Registered prediction: low-N failure is structural, not compute — plain's P(fail) stays
high (δ̂ ~9-11 majority) at matched compute. If compute DOES rescue it, claim 4 must be
rewritten as a compute story — either way the row pre-empts the attack.

Run: nohup .venv/bin/python -u experiments/p4_matched_compute.py > /tmp/p4_mc.log 2>&1 &
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
from experiments.p4_v16_stageA_sweep import DEVICE, run_one, state_targets  # noqa: E402
from experiments.p4_wedge_lane import EQ_CFG, PLAIN_CFG, load_set  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

T0 = time.time()
N_COUNT = 2606
OUT = ROOT / "papers" / "figures" / "p4_matched_compute.json"


def main() -> int:
    art: dict = {"design": "plain epochs scaled by measured wall-time ratio; cap 300", "cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(rng.choice(obs.shape[0], N_COUNT, replace=False)).long()
    o_, a_, n_, x_ = obs[idx], act[idx], nxt[idx], aux_all[idx]
    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    # --- measure per-epoch wall-time ratio (2 epochs each, same data/device)
    times = {}
    for name, builder in (("eq", build_eq), ("plain", lambda: build_plain(w_match))):
        torch.manual_seed(0)
        m = builder()
        t0 = time.time()
        train_jepa(m, o_, a_, n_, epochs=2, batch_size=64, device=DEVICE, seed=0,
                   verbose=False, refresh_target_cache=True)
        times[name] = (time.time() - t0) / 2
    ratio = times["eq"] / times["plain"]
    ep_matched = int(min(round(PLAIN_CFG.get("epochs", 40) * ratio), 300))
    art["timing"] = {"eq_s_per_epoch": round(times["eq"], 2),
                     "plain_s_per_epoch": round(times["plain"], 2),
                     "ratio": round(ratio, 2), "plain_epochs_matched": ep_matched,
                     "capped": ep_matched == 300}
    print(f"[ratio] eq/plain per-epoch = {ratio:.2f} -> plain epochs {ep_matched}")
    OUT.write_text(json.dumps(art, indent=1))

    cfg = dict(PLAIN_CFG)
    cfg["epochs"] = ep_matched
    for r in range(4):
        try:
            cell, pr, *_ = run_one(lambda: build_plain(w_match), cfg, r, o_, a_, n_,
                                   fb_in, ach_in, fb_small, ho_in, x_)
            if cell["stable"]:
                rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
                cell["in_delta"] = round(rep["delta"]["mean"], 3)
        except Exception as exc:  # noqa: BLE001
            cell = {"error": str(exc)[:200], "stable": False}
        art["cells"][f"plain_mc_r{r}"] = {k: v for k, v in cell.items()
                                          if k in ("std", "xy", "in_delta", "stable", "error")}
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))
        print(f"  plain_mc r{r}: std {cell.get('std', float('nan')):.2f} δ {cell.get('in_delta')}")
    ds = [c["in_delta"] for c in art["cells"].values() if c.get("in_delta")]
    art["readout"] = {"deltas": sorted(ds), "ref_plain_40ep": [4.23, 7.99, 10.66, 10.89],
                      "verdict": ("compute does NOT rescue" if ds and np.median(ds) > 6
                                  else "compute rescues — rewrite claim 4")}
    OUT.write_text(json.dumps(art, indent=1))
    print("READOUT:", art["readout"])
    print(f"MATCHED COMPUTE DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
