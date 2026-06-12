r"""kappa=0.8 rescue sweep — modern recipe family vs the expansive regime (C1a unlock attempt).

History: every kappa=0.8 attempt predates the v1.6 tuning era (aug-era recipes; eq partial
collapse std 0.562 contaminated the moat inversion read). Hypothesis-driven tuning: do the
MODERN recipes (banked aux0.5_v0.3 / old champion aux0.3 / winner v0.2) achieve stability on
kappa=0.8 data? Stage-A HEALTH ONLY (stability floor 0.7, xy content, delta-hat) — no claim
gate; >=3/4 stability for any recipe => C1a base unlocked => full treatment registered later.

Run (box): SDL_VIDEODRIVER=dummy P4_DEVICE=cuda nohup nice -n 5 .venv3/bin/python -u \
    experiments/p4_k08_rescue.py > ~/p4_k08.log 2>&1 &   (~1.5 h)
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
from experiments.p4_step1_pipeline import (  # noqa: E402
    RES, build_eq, collect_weakpolicy, to_transitions,
)
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
KAPPA = 0.8
OUT = ROOT / "papers" / "figures" / "p4_k08_rescue.json"
RECIPES = {
    "banked_aux05_v03": dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.5),
    "oldchamp_aux03_v03": dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.3),
    "winner_v02": dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0),
}


def main() -> int:
    art: dict = {"kappa": KAPPA, "design": "stage-A health only, n=4/recipe", "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] kappa=0.8 corpora ...")
    corpus = collect_weakpolicy(200, seed=0, kappa=KAPPA)
    obs, act, nxt = to_transitions(corpus, 200)
    aux_t = state_targets(corpus)
    ho = collect_weakpolicy(60, seed=1, kappa=KAPPA)
    fb, ach = tensorize(ho)
    fb_small = fb[:20].reshape(-1, 3, RES, RES).float()
    save()

    for rname, cfg in RECIPES.items():
        for r in range(4):
            try:
                cell, pr, *_ = run_one(build_eq, cfg, r, obs, act, nxt,
                                       fb, ach, fb_small, ho, aux_t)
                if cell["stable"]:
                    rep = audit_gap(pr, fb, ach, window=16, k=3, burn_in=4, h_max=8)
                    cell["delta_k08"] = round(rep["delta"]["mean"], 3)
                    cell["lambda1"] = round(rep["lambda1"]["mean"], 4)
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][f"{rname}_r{r}"] = cell
            save()
            print(f"  {rname} r{r}: std {cell.get('std', float('nan')):.2f} "
                  f"xy {cell.get('xy')} δ {cell.get('delta_k08')} λ1 {cell.get('lambda1')}")

    verd = {}
    for rname in RECIPES:
        cs = [art["cells"][f"{rname}_r{r}"] for r in range(4)]
        ok = [c for c in cs if c.get("stable")]
        verd[rname] = {"stable": f"{len(ok)}/4",
                       "unlocked": len(ok) >= 3,
                       "xy": round(float(np.mean([c["xy"] for c in ok])), 3) if ok else None}
    art["verdicts"] = verd
    save()
    print("VERDICTS:", json.dumps(verd))
    print(f"K08 RESCUE DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
