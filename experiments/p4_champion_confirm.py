r"""P4 champion confirmation — v0.3+aux0.3 × c2000 × n=10; C3-guar's first legal evaluation.

Declared (morning 2026-06-12, before results): champion recipe from the night shift
(e0.99_v0.3_aux0.3 — 3/3 stable, xy 0.567 at 200 eps) trained on the c2000 corpus (data lever:
content 0.613 at winner recipe), n = 10 runs. Gates evaluated on these FRESH runs:

- **C3-guar (first legal evaluation; registered 2026-06-12 pre-dawn):** certified q90 boundary ≤
  measured q90 boundary (ratio ≤ 1) at ALL ε cells, in ≥ 90% of qualifying runs.
- **C3-cal (two-sided, standing registration):** ratio ∈ [0.5, 2] all cells, ≥ 2/3 of qualifying.
- Stability rate with exact binomial CI; xy content mean.
- plain control: ep40 winner × c2000 × n=10 (same gates; the moat needs both sides on equal data).

Run: nohup .venv/bin/python experiments/p4_champion_confirm.py   (~2.5 h)
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
FLOOR = 0.7
OUT = ROOT / "papers" / "figures" / "p4_champion_confirm.json"

CHAMP = dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.3)
PLAIN = dict(ema_decay=0.99, var_coef=0.04, lr_scale=1.0, epochs=40)


def main() -> int:
    art: dict = {"champion": str(CHAMP), "plain_ctrl": str(PLAIN), "corpus": "c2000",
                 "gates": "C3-guar (<=1 all cells, >=90% qual) + C3-cal ([0.5,2], >=2/3 qual)",
                 "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] c2000 corpus + held-out ...")
    z = np.load(DATA_DIR / "corpus_c2000.npz")
    big = {k: z[k] for k in ("frames", "states", "actions")}
    obs, act, nxt = to_transitions(big, big["frames"].shape[0])
    aux_t = state_targets(big)
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
    builders = {"champ": build_eq, "plainc": (lambda: build_plain(w_match))}
    cfgs = {"champ": CHAMP, "plainc": PLAIN}

    for name in ("champ", "plainc"):
        print(f"[{name}] n=10 on c2000 ...")
        for r in range(10):
            try:
                cell, pr, m, tgt = run_one(builders[name], cfgs[name], r, obs, act, nxt,
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
                               DATA_DIR / f"ckpt8_{name}_r{r}.pt")
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][f"{name}_r{r}"] = cell
            save()
            print(f"  {name} r{r}: std {cell.get('std', float('nan')):.3f} xy {cell.get('xy')} "
                  f"cal {[x['ratio'] for x in cell.get('c3cal', [])]}")

    verd = {}
    for name in ("champ", "plainc"):
        cs = [art["cells"].get(f"{name}_r{r}", {}) for r in range(10)]
        qual = [c for c in cs if c.get("stable")]
        def ratios_ok(c, pred):
            return "c3cal" in c and all(x["ratio"] is not None and pred(x["ratio"])
                                        for x in c["c3cal"])
        guar = [c for c in qual if ratios_ok(c, lambda x: x <= 1.0 + 1e-9)]
        cal = [c for c in qual if ratios_ok(c, lambda x: BAND[0] <= x <= BAND[1])]
        verd[name] = {
            "stable": f"{len(qual)}/10",
            "C3_guar": ("INCONCLUSIVE-BY-STABILITY" if len(qual) < 5
                        else ("PASS" if len(guar) / len(qual) >= 0.9 else "FAIL")),
            "guar_passing": len(guar),
            "C3_cal": ("INCONCLUSIVE-BY-STABILITY" if len(qual) < 5
                       else ("PASS" if len(cal) / len(qual) >= 2 / 3 else "FAIL")),
            "cal_passing": len(cal),
            "xy_mean": (round(float(np.mean([c["xy"] for c in qual])), 3) if qual else None),
            "delta_norm_mean": (round(float(np.mean([c.get("delta_norm", float("nan"))
                                                     for c in qual])), 3) if qual else None),
        }
        print(f"VERDICT {name}: {verd[name]}")
    art["verdicts"] = verd
    save()
    print(f"DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
