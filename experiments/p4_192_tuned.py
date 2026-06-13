r"""T1.9 — 192px TUNED-recipe sweep (closes the off-tune caveat on claim 10).

The resolution ladder (06-12) ran 192px with the 96px-tuned banked recipe → ε_task/δ̂ = 0.3
(worst, away from GO). Registered caveat: that rung measured the lever AS-AVAILABLE, not at a
192px optimum. This sweeps recipes AT 192px to settle whether the resolution-backfire survives
proper tuning. Per recipe × n=3: stability, δ̂ (audit), ε_task (D1-v2 estimator) → GO criterion
ε_task ≥ δ̂. Registered readout: if NO recipe reaches GO at 192px, claim 10's "resolution
backfires" stands even tuned; if some recipe flips to GO, the caveat upgrades to "resolution
helps under tuning" (claim 10 weakens to a recipe-conditional statement).

Run (Mac): nohup .venv/bin/python -u experiments/p4_192_tuned.py > /tmp/p4_192.log 2>&1 &
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

from experiments.p4_res_ladder import (  # noqa: E402
    TAU_POS, collect_res, eps_task_bind, to_transitions_res,
)
from experiments.p4_step1_pipeline import RES, build_eq  # noqa: E402
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
RES192 = 192
OUT = ROOT / "papers" / "figures" / "p4_192_tuned.json"
# recipe sweep AT 192px: vary the var/aux that matter for stability-content, around the banked one
RECIPES = {
    "banked": dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.5),
    "hi_var": dict(ema_decay=0.99, var_coef=0.5, lr_scale=1.0, aux_coef=0.5),
    "lo_var": dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0, aux_coef=0.5),
    "hi_aux": dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=1.0),
    "hi_ema": dict(ema_decay=0.995, var_coef=0.3, lr_scale=1.0, aux_coef=0.5),
    "lo_lr": dict(ema_decay=0.99, var_coef=0.3, lr_scale=0.5, aux_coef=0.5),
}


def main() -> int:
    art: dict = {"res": RES192, "go_criterion": "eps_task(tau) >= delta_hat (tuned at 192px)",
                 "recipes": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] 192px corpus + held-out ...")
    corpus = collect_res(200, seed=0, res=RES192)
    obs, act, nxt, _ = to_transitions_res(corpus, RES192)
    aux_t = state_targets(corpus)
    ho = collect_res(60, seed=1, res=RES192)
    _, _, _, fbh = to_transitions_res(ho, RES192)
    fb_d = fbh.double()
    a = torch.from_numpy(ho["actions"])
    from experiments.p4_step1_pipeline import CHUNK
    n_ch = a.shape[1] // CHUNK
    ach_d = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
    fb_small = fbh[:20].reshape(-1, 3, RES192, RES192).float()
    sb = torch.from_numpy(ho["states"]).double()[:, 0: n_ch * CHUNK + 1: CHUNK]
    sb = sb.reshape(-1, sb.shape[-1])
    save()

    for rname, cfg in RECIPES.items():
        cells = []
        for r in range(3):
            try:
                cell, pr, m, tgt = run_one(build_eq, cfg, r, obs, act, nxt,
                                           fb_d, ach_d, fb_small, ho, aux_t)
                if cell["stable"]:
                    rep = audit_gap(pr, fb_d, ach_d, window=16, k=3, burn_in=4, h_max=8)
                    cell["delta"] = round(rep["delta"]["mean"], 3)
                    with torch.no_grad():
                        flat = fbh.reshape(-1, 3, RES192, RES192)
                        z = torch.cat([tgt.eval()(flat[i:i + 64].float())
                                       for i in range(0, len(flat), 64)]).cpu().double()
                    cell["eps_task"] = eps_task_bind(z, sb, 60)
                    cell["go"] = bool(cell["eps_task"] and cell["eps_task"] >= cell["delta"])
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:160], "stable": False}
            cells.append({k: v for k, v in cell.items()
                          if k in ("std", "xy", "delta", "eps_task", "go", "stable", "error")})
            print(f"  {rname} r{r}: std {cell.get('std')} δ {cell.get('delta')} "
                  f"ε_task {cell.get('eps_task')} GO {cell.get('go')}")
            art["recipes"][rname] = cells
            save()

    any_go = any(c.get("go") for cs in art["recipes"].values() for c in cs)
    art["verdict"] = {
        "any_go_at_192px_tuned": any_go,
        "reading": ("resolution HELPS under tuning (claim 10 -> recipe-conditional)" if any_go
                    else "resolution backfire SURVIVES tuning (claim 10 stands, off-tune caveat closed)")}
    save()
    print("VERDICT:", art["verdict"])
    print(f"192 TUNED DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
