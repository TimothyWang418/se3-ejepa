r"""P4 night shift 2026-06-11 — autonomous EMP batch (user asleep; health metrics only throughout).

Order (declared): wait for Stage-B to free the GPU → E0.5.2 winner-neighborhood refinement
(9 cfg × 3 runs) → E0.1 data-scaling study (winner recipe × {500,1000,2000} eps × 3 runs;
fixed 20 epochs — optimizer steps scale with data, recorded as part of the treatment) →
E0.5.1 aux-anchor variant family (12 cfg × 3 runs) → E0.2 long-training (2 cfg × 1 run,
200 epochs; per-epoch std curve + endpoint probes = decay instrument lite).

Crash-immune per cell; artifact written incrementally to papers/figures/p4_nightshift_0611.json.
No claim gate is consulted anywhere (Tier 0/0.5 are health-metric tiers).
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

from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder, to_transitions,
)
from experiments.p4_v16_stageA_sweep import quick_xy, run_one, state_targets  # noqa: E402

T0 = time.time()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_nightshift_0611.json"
STAGEB = ROOT / "papers" / "figures" / "p4_v16_stageB.json"
FLOOR = 0.7

WINNER = dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0)
NEIGHBORHOOD = [dict(ema_decay=e, var_coef=v, lr_scale=1.0)
                for e in (0.985, 0.99, 0.995) for v in (0.15, 0.25, 0.3)]
AUX_FAMILY = ([dict(ema_decay=0.99, var_coef=v, lr_scale=1.0, aux_coef=c)
               for v in (0.15, 0.2, 0.3) for c in (0.3, 0.5, 1.0)]
              + [dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0, aux_coef=c, aux_theta_only=True)
                 for c in (0.5, 1.0, 2.0)])
LONGTRAIN = [dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0, epochs=200),
             dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0, aux_coef=1.0, epochs=200)]


def tag(c: dict) -> str:
    t = f"e{c['ema_decay']}_v{c['var_coef']}"
    if c.get("aux_coef"):
        t += f"_aux{c['aux_coef']}" + ("th" if c.get("aux_theta_only") else "")
    if c.get("epochs"):
        t += f"_ep{c['epochs']}"
    return t


def main() -> int:
    art: dict = {"order": "wait-stageB -> E0.5.2 -> E0.1 -> E0.5.1 -> E0.2", "blocks": {}}

    def save():
        art["elapsed_h"] = round((time.time() - T0) / 3600, 2)
        OUT.write_text(json.dumps(art, indent=1))

    print("[wait] Stage-B to finish (GPU handover) ...")
    for _ in range(240):  # up to 2h guard
        if STAGEB.exists() and "verdicts" in json.loads(STAGEB.read_text()):
            break
        time.sleep(30)
    print(f"[wait] done at {(time.time()-T0)/60:.0f} min")

    print("[setup] base data ...")
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

    def theta_targets(c: dict) -> torch.Tensor:
        s = torch.from_numpy(c["states"]).float()
        n = (s.shape[1] - 1) // CHUNK
        st = s[:, 0 : n * CHUNK : CHUNK]
        return torch.cat([torch.cos(st[..., 4:5]), torch.sin(st[..., 4:5])], -1).reshape(-1, 2)

    aux_th = theta_targets(corpus)

    def block(name: str, cfgs: list[dict], runs: int, data=None, aux_override=None):
        art["blocks"][name] = {}
        o_, a_, n_, x_ = data if data else (obs, act, nxt, aux_t)
        for cfg in cfgs:
            x_use = aux_th if cfg.get("aux_theta_only") else (aux_override if aux_override is not None else x_)
            cells = []
            for r in range(runs):
                try:
                    cell, *_ = run_one(build_eq, cfg, r, o_, a_, n_, fb_d, ach_d, fb_small, ho, x_use)
                except Exception as exc:  # noqa: BLE001
                    cell = {"error": str(exc)[:150], "stable": False}
                cells.append(cell)
            ok = [c for c in cells if c.get("stable")]
            art["blocks"][name][tag(cfg)] = {
                "cells": cells, "stable_frac": len(ok) / runs,
                "xy": float(np.mean([c["xy"] for c in ok])) if ok else None,
                "delta": float(np.mean([c["delta"] for c in ok])) if ok else None}
            print(f"  {name}/{tag(cfg)}: stable {len(ok)}/{runs} "
                  f"xy {art['blocks'][name][tag(cfg)]['xy']}")
            save()

    print("[E0.5.2] winner neighborhood ...")
    block("neighborhood", NEIGHBORHOOD, 3)

    print("[E0.1] data-scaling study (winner recipe) ...")
    art["blocks"]["datascale"] = {}
    for ctag in ("c500", "c1000", "c2000"):
        p = ROOT / "data" / "p4_step1" / f"corpus_{ctag}.npz"
        for _ in range(60):
            if p.exists():
                break
            time.sleep(30)
        if not p.exists():
            art["blocks"]["datascale"][ctag] = {"error": "corpus missing"}
            continue
        z = np.load(p)
        big = {k: z[k] for k in ("frames", "states", "actions")}
        n_ep = big["frames"].shape[0]
        o_, a_, n_ = to_transitions(big, n_ep)
        x_ = state_targets(big)
        cells = []
        for r in range(3):
            try:
                cell, *_ = run_one(build_eq, WINNER, r, o_, a_, n_, fb_d, ach_d, fb_small, ho, x_)
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:150], "stable": False}
            cells.append(cell)
        ok = [c for c in cells if c.get("stable")]
        art["blocks"]["datascale"][ctag] = {
            "cells": cells, "stable_frac": len(ok) / 3,
            "xy": float(np.mean([c["xy"] for c in ok])) if ok else None,
            "delta": float(np.mean([c["delta"] for c in ok])) if ok else None}
        print(f"  datascale/{ctag}: {art['blocks']['datascale'][ctag]['stable_frac']} "
              f"xy {art['blocks']['datascale'][ctag]['xy']}")
        save()

    print("[E0.5.1] aux variant family ...")
    block("aux_family", AUX_FAMILY, 3)

    print("[E0.2] long-training (per-epoch std curves) ...")
    art["blocks"]["longtrain"] = {}
    from src.training.jepa import train_jepa  # noqa: E402
    for cfg in LONGTRAIN:
        try:
            torch.manual_seed(0)
            m = build_eq()
            kw = dict(epochs=200, batch_size=64, device=DEVICE, seed=0, verbose=False,
                      return_target_encoder=True, refresh_target_cache=True,
                      ema_decay=cfg["ema_decay"], var_coef=cfg["var_coef"])
            if cfg.get("aux_coef"):
                kw["aux_state"] = (aux_t, cfg["aux_coef"])
            hist, tgt = train_jepa(m, obs, act, nxt, **kw)
            art["blocks"]["longtrain"][tag(cfg)] = {
                "pred_curve": hist["pred_loss"][::10], "var_curve": hist["var_loss"][::10],
                "final_std": hist["latent_std"],
                "final_xy": quick_xy(tgt, ho, fb_small)}
            print(f"  longtrain/{tag(cfg)}: final_std {hist['latent_std']:.3f} "
                  f"xy {art['blocks']['longtrain'][tag(cfg)]['final_xy']:.3f}")
        except Exception as exc:  # noqa: BLE001
            art["blocks"]["longtrain"][tag(cfg)] = {"error": str(exc)[:150]}
        save()

    print(f"NIGHT SHIFT DONE ({(time.time()-T0)/3600:.2f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
