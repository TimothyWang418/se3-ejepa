r"""P4 v1.6 Stage B — n=10 claim evaluation under the frozen Stage-A winners (+ the declared
combo extension).

Declared BEFORE any claim result is seen (this docstring is the declaration):

1. **Registered winners** (Stage-A lexicographic rule, health-only):
   eq = e0.99_v0.2_l1.0; plain = e0.99_v0.04_l1.0_ep40. Each gets n=10 runs and the registered
   gates (C3-cal q90↔q90, band [0.5,2], collapse-conditioned: qualifying ≥ 5 of 10 AND
   pass-fraction ≥ 2/3, else INCONCLUSIVE-BY-STABILITY).
2. **Combo extension (declared now)**: the Stage-A grid never tested the content anchor TOGETHER
   with the strong floor. Health-check aux1.0 + v0.2 (3 runs); if stable 3/3, it receives its own
   n=10 under the same gates. BOTH recipes' results are reported regardless of outcome —
   recipe selection happened on health metrics only; no claim-based swapping.

Run: nohup .venv/bin/python experiments/p4_v16_stageB.py   (~70 min)
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

from experiments.p4_spine_stage1a import boundary_from_curve, fit_growth, measured_err  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder,
    to_transitions,
)
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
EPS_MULT = (2, 4, 8, 16)
BAND = (0.5, 2.0)
OUT = ROOT / "papers" / "figures" / "p4_v16_stageB.json"

WINNERS = {
    "eq": dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0),
    "plain": dict(ema_decay=0.99, var_coef=0.04, lr_scale=1.0, epochs=40),
}
COMBO = dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0, aux_coef=1.0)  # declared extension


def main() -> int:
    art: dict = {"declaration": "winners n=10 + combo health(3); combo n=10 iff 3/3 stable; "
                 "all reported; gates: qualifying>=5/10 AND pass>=2/3",
                 "winners": {k: str(v) for k, v in WINNERS.items()}, "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] data ...")
    corpus = collect_weakpolicy(200, seed=0)
    obs, act, nxt = to_transitions(corpus, 200)
    aux_t = state_targets(corpus)
    ho = collect_weakpolicy(60, seed=1)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK]
    fb_d = fb.double()
    fb_small = fb[:20].reshape(-1, 3, RES, RES)
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach_d = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    builders = {"eq": build_eq, "plain": (lambda: build_plain(w_match)),
                "eq_combo": build_eq}

    def claim_run(base_key: str, cfg: dict, run: int, tagprefix: str):
        try:
            cell, pr, m, tgt = run_one(builders[base_key], cfg, run, obs, act, nxt,
                                       fb_d, ach_d, fb_small, ho, aux_t)
            if cell["stable"]:
                rep = audit_gap(pr, fb_d, ach_d, window=16, k=3, burn_in=4, h_max=8)
                meas = measured_err(pr, fb_d, ach_d, 8)
                dm = rep["delta"]["mean"]
                cell["delta_norm"] = dm / (max(cell["std"], 1e-6) * (128 ** 0.5))
                cell["c3cal"] = [{"em": em,
                                  "ratio": (boundary_from_curve(rep["certified_curve"]["err_q90"], em * dm)
                                            / boundary_from_curve(meas["q90"], em * dm))
                                  if boundary_from_curve(meas["q90"], em * dm) else None}
                                 for em in EPS_MULT]
                cell["growth"] = fit_growth(meas["median"])
                torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                           DATA_DIR / f"ckpt7_{tagprefix}_r{run}.pt")
        except Exception as exc:  # noqa: BLE001
            cell = {"error": str(exc)[:200], "stable": False}
        art["cells"][f"{tagprefix}_r{run}"] = cell
        save()
        print(f"  {tagprefix} r{run}: std {cell.get('std', float('nan')):.3f} "
              f"stable {cell.get('stable')} cal {[x['ratio'] for x in cell.get('c3cal', [])]}")
        return cell

    # 1) combo health check first (cheap, decides whether it earns n=10)
    print("[1/3] combo health check (aux1.0 + v0.2, 3 runs) ...")
    combo_cells = [claim_run("eq_combo", COMBO, r, "combohealth") for r in range(3)]
    combo_stable = sum(1 for c in combo_cells if c.get("stable"))
    art["combo_health"] = f"{combo_stable}/3"
    print(f"  combo health: {art['combo_health']}  xy: {[c.get('xy') for c in combo_cells]}")

    # 2) winners at n=10
    for name, cfg in WINNERS.items():
        print(f"[2/3] {name} winner n=10 ...")
        for r in range(10):
            claim_run(name, cfg, r, f"{name}win")

    # 3) combo n=10 iff earned
    if combo_stable == 3:
        print("[3/3] combo earned n=10 ...")
        for r in range(10):
            claim_run("eq_combo", COMBO, r, "combo")

    # aggregate
    verd = {}
    for tag, n in (("eqwin", 10), ("plainwin", 10)) + ((("combo", 10),) if combo_stable == 3 else ()):
        cs = [art["cells"].get(f"{tag}_r{r}", {}) for r in range(n)]
        qual = [c for c in cs if c.get("stable")]
        cal_pass = [c for c in qual if "c3cal" in c
                    and all(x["ratio"] is not None and BAND[0] <= x["ratio"] <= BAND[1]
                            for x in c["c3cal"])]
        verd[tag] = {"stable": f"{len(qual)}/{n}",
                     "c3cal": ("INCONCLUSIVE-BY-STABILITY" if len(qual) < 5
                               else ("PASS" if len(cal_pass) / len(qual) >= 2 / 3 else "FAIL")),
                     "passing": len(cal_pass),
                     "delta_norm_mean": (round(float(np.mean([c["delta_norm"] for c in qual])), 3)
                                         if qual else None),
                     "xy_mean": (round(float(np.mean([c["xy"] for c in qual])), 3) if qual else None)}
        print(f"VERDICT {tag}: {verd[tag]}")
    art["verdicts"] = verd
    save()
    print(f"DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
