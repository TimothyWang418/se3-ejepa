r"""P4 v1.6 Stage A — the overnight stability sweep (+ time-budgeted Stage-B auto-continuation).

Spec: step1 seed spec v1.6 block. HEALTH metrics only (stable fraction, held-out δ̂, xy content);
claim gates never consulted (construction-level separation). Equal tuning budget both bases.

- eq: 20 configs (core 12: ema×var×lr; extras 4: ema .999 / var .2 / gated / epochs 40;
  **aux family 4**: TC-WM-inspired proprio anchor, coef×var) × 3 runs.
- plain: 16 configs (core 12 + extras 4) × 3 runs.
- Per run: train → latent_std/floor → held-out δ̂ (deployable pair) → quick xy probe.
- Winner per base (registered rule): require stable fraction ≥ 2/3, then lexicographic
  (stable_frac, xy_mean, −δ̂_mean). Crash-immune per cell; artifact written incrementally.
- **Stage-B auto-continuation**: if elapsed < 3.0 h after sweeps, retrain winners × 10 runs and
  evaluate the registered claim gates at n=10 (collapse-conditioned, runs-not-seeds).

Run: nohup .venv/bin/python experiments/p4_v16_stageA_sweep.py  (~4 h)
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_spine_stage1a import boundary_from_curve, fit_growth, measured_err  # noqa: E402
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, Probe, build_eq, build_plain, circ_mask, collect_weakpolicy, fit_probe,
    pick_ladder, to_transitions,
)
from src.audit.gap_mode import audit_gap, one_step_bias  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

T0 = time.time()
BUDGET_STAGEB_START = 3.0 * 3600
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FLOOR = 0.7
D = 128
EPS_MULT = (2, 4, 8, 16)
BAND = (0.5, 2.0)
OUT = ROOT / "papers" / "figures" / "p4_v16_stageA.json"

CORE = [dict(ema_decay=e, var_coef=v, lr_scale=s)
        for e in (0.97, 0.99, 0.995) for v in (0.04, 0.1) for s in (1.0, 0.5)]
EXTRAS = [dict(ema_decay=0.999, var_coef=0.04, lr_scale=1.0),
          dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0),
          dict(ema_decay=0.99, var_coef=0.04, lr_scale=1.0, gated=True),
          dict(ema_decay=0.99, var_coef=0.04, lr_scale=1.0, epochs=40)]
AUX = [dict(ema_decay=0.99, var_coef=v, lr_scale=1.0, aux_coef=c)
       for c in (0.1, 1.0) for v in (0.04, 0.1)]


def cfg_tag(c: dict) -> str:
    t = f"e{c['ema_decay']}_v{c['var_coef']}_l{c['lr_scale']}"
    if c.get("gated"):
        t += "_gated"
    if c.get("epochs"):
        t += f"_ep{c['epochs']}"
    if c.get("aux_coef"):
        t += f"_aux{c['aux_coef']}"
    return t


def state_targets(corpus: dict) -> torch.Tensor:
    r"""Chunk-start state targets aligned with to_transitions order: (E*n_ch, 8) f32 normalized
    [ax, ay, bx, by, cosθ, sinθ, vx, vy]/scales."""
    s = torch.from_numpy(corpus["states"]).float()  # (E, T+1, 7)
    n_ch = (s.shape[1] - 1) // CHUNK
    st = s[:, 0 : n_ch * CHUNK : CHUNK]  # (E, n_ch, 7)
    out = torch.cat([st[..., 0:4] / 512.0, torch.cos(st[..., 4:5]), torch.sin(st[..., 4:5]),
                     st[..., 5:7] / 100.0], dim=-1)
    return out.reshape(-1, 8)


def quick_xy(tgt_enc, ho: dict, fb_small: torch.Tensor) -> float:
    with torch.no_grad():
        z = tgt_enc.eval().to(DEVICE)(fb_small.float().to(DEVICE)).cpu()
    tgt_enc.cpu()
    s = torch.from_numpy(ho["states"][:, ::CHUNK]).float()
    xy = (s[..., 2:4] / 512.0).reshape(-1, 2)[: z.shape[0]]
    n = z.shape[0]
    tr, ev = slice(0, int(0.7 * n)), slice(int(0.7 * n), n)
    return fit_probe(z[tr], xy[tr], z[ev], xy[ev], epochs=40)


def run_one(builder, cfg: dict, run: int, obs, act, nxt, fb_d, ach_d, fb_small, ho, aux_t):
    torch.manual_seed(run)
    m = builder()
    kw = dict(epochs=cfg.get("epochs", 20), batch_size=64, device=DEVICE, seed=run, verbose=False,
              return_target_encoder=True, refresh_target_cache=True,
              ema_decay=cfg["ema_decay"], var_coef=cfg["var_coef"],
              muon_lr=0.02 * cfg["lr_scale"], adamw_lr=1e-3 * cfg["lr_scale"],
              predictability_gated_var=cfg.get("gated", False))
    if cfg.get("aux_coef"):
        kw["aux_state"] = (aux_t, cfg["aux_coef"])
    hist, tgt = train_jepa(m, obs, act, nxt, **kw)
    pr = Pair(tgt.eval().cpu(), m.predictor.cpu())
    delta = float(one_step_bias(pr, fb_d, ach_d).mean())
    return {"std": hist["latent_std"], "delta": delta,
            "xy": quick_xy(tgt, ho, fb_small),
            "stable": bool(hist["latent_std"] >= FLOOR)}, pr, m, tgt


def main() -> int:
    art: dict = {"protocol": "v1.6 Stage A", "floor": FLOOR,
                 "language": "runs not seeds", "configs": {}, "log": []}

    def save():
        art["elapsed_h"] = round((time.time() - T0) / 3600, 2)
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
    builders = {"eq": build_eq, "plain": (lambda: build_plain(w_match))}
    grids = {"eq": CORE + EXTRAS + AUX, "plain": CORE + EXTRAS}

    for base, grid in grids.items():
        for cfg in grid:
            tag = f"{base}:{cfg_tag(cfg)}"
            cells = []
            for run in range(3):
                try:
                    cell, *_ = run_one(builders[base], cfg, run, obs, act, nxt,
                                       fb_d, ach_d, fb_small, ho, aux_t)
                    cells.append(cell)
                except Exception as exc:  # noqa: BLE001 — overnight crash immunity
                    cells.append({"error": str(exc)[:200], "stable": False})
                    art["log"].append(f"{tag} run{run} ERROR {str(exc)[:100]}")
            ok = [c for c in cells if "error" not in c]
            stable = [c for c in ok if c["stable"]]
            art["configs"][tag] = {
                "cells": cells,
                "stable_frac": len(stable) / 3,
                "xy_mean": float(np.mean([c["xy"] for c in stable])) if stable else None,
                "delta_mean": float(np.mean([c["delta"] for c in stable])) if stable else None,
            }
            s = art["configs"][tag]
            print(f"  {tag}: stable {s['stable_frac']:.2f} xy {s['xy_mean']} d {s['delta_mean']}")
            save()

    # winner per base (registered lexicographic rule)
    art["winners"] = {}
    for base in builders:
        cands = {t: c for t, c in art["configs"].items()
                 if t.startswith(base + ":") and c["stable_frac"] >= 2 / 3 and c["xy_mean"] is not None}
        if cands:
            win = max(cands, key=lambda t: (cands[t]["stable_frac"], cands[t]["xy_mean"],
                                            -(cands[t]["delta_mean"] or 1e9)))
            art["winners"][base] = win
        else:
            art["winners"][base] = None
        print(f"WINNER {base}: {art['winners'][base]}")
    save()

    # Stage-B auto-continuation if budget remains
    if (time.time() - T0) < BUDGET_STAGEB_START and all(art["winners"].values()):
        print("[Stage-B] n=10 conversion under winners ...")
        art["stageB"] = {"n": 10, "cells": {}}
        win_cfgs = {b: next(c for c in grids[b] if f"{b}:{cfg_tag(c)}" == art["winners"][b])
                    for b in builders}
        for base, cfg in win_cfgs.items():
            for run in range(10):
                try:
                    cell, pr, m, tgt = run_one(builders[base], cfg, run, obs, act, nxt,
                                               fb_d, ach_d, fb_small, ho, aux_t)
                    if cell["stable"]:
                        rep = audit_gap(pr, fb_d, ach_d, window=16, k=3, burn_in=4, h_max=8)
                        meas = measured_err(pr, fb_d, ach_d, 8)
                        dm = rep["delta"]["mean"]
                        cell["c3cal"] = [
                            {"em": em,
                             "ratio": (boundary_from_curve(rep["certified_curve"]["err_q90"], em * dm)
                                       / boundary_from_curve(meas["q90"], em * dm))
                             if boundary_from_curve(meas["q90"], em * dm) else None}
                            for em in EPS_MULT]
                        cell["growth"] = fit_growth(meas["median"])
                        torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                                   DATA_DIR / f"ckpt6_{base}_r{run}.pt")
                    art["stageB"]["cells"][f"{base}_r{run}"] = cell
                except Exception as exc:  # noqa: BLE001
                    art["stageB"]["cells"][f"{base}_r{run}"] = {"error": str(exc)[:200],
                                                                "stable": False}
                save()
        # aggregate
        verd = {}
        for base in builders:
            cs = [art["stageB"]["cells"][f"{base}_r{r}"] for r in range(10)]
            qual = [c for c in cs if c.get("stable")]
            cal_pass = [c for c in qual if "c3cal" in c
                        and all(x["ratio"] is not None and BAND[0] <= x["ratio"] <= BAND[1]
                                for x in c["c3cal"])]
            verd[base] = {
                "stable": f"{len(qual)}/10",
                "c3cal": ("INCONCLUSIVE-BY-STABILITY" if len(qual) < 5
                          else ("PASS" if len(cal_pass) / len(qual) >= 2 / 3 else "FAIL")),
                "c3cal_passing": len(cal_pass),
            }
            print(f"[Stage-B] {base}: {verd[base]}")
        art["stageB"]["verdicts"] = verd
        save()

    print(f"DONE ({(time.time()-T0)/3600:.2f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
