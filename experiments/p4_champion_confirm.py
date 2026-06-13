r"""P4 champion confirmation — v0.3+aux0.3 × c2000 × n=10; C3-guar's first legal evaluation.

Declared (morning 2026-06-12, before results): champion recipe from the night shift
(e0.99_v0.3_aux0.3 — 3/3 stable, xy 0.567 at 200 eps) trained on the c2000 corpus (data lever:
content 0.613 at winner recipe), n = 10 runs. Gates evaluated on these FRESH runs:

- **C3-guar (first legal evaluation; registered 2026-06-12 pre-dawn):** certified q90 boundary ≤
  measured q90 boundary (ratio ≤ 1) at ALL ε cells, in ≥ 90% of qualifying runs.
- **C3-cal (two-sided, standing registration):** ratio ∈ [0.5, 2] all cells, ≥ 2/3 of qualifying.

AMENDMENT (registered 2026-06-11 between r1 and r2, before further data): the ratio-based code
was an unfaithful approximation of the registered text. At tight ε cells the measured q90 curve
can exceed ε at h=1 (heavy tail: q90/mean > 2), giving boundary 0 and ratio None — under the old
code a run could NEVER qualify, making C3-guar mechanically unpassable regardless of certificate
quality. Faithful mechanics: store BOTH boundaries; C3-guar compares them directly (0 ≤ 0 PASSES
— the certificate refuses a horizon the world also refuses; cert > 0 = meas FAILS — that IS
anticonservative); C3-cal uses ratios only where both boundaries > 0 (a 0/0 cell carries no
calibration information). r0/r1 were observed pre-amendment (honest flag); they are re-evaluated
from their saved ckpts under the new mechanics, not re-trained.
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
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder,
    to_transitions_lean,
)
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
EPS_MULT = (2, 4, 8, 16)
BAND = (0.5, 2.0)
FLOOR = 0.7
TAG = __import__("os").environ.get("P4_TAG", "")  # CUDA arm: P4_TAG=_cuda
NRUN = int(__import__("os").environ.get("P4_N", "10"))  # rental: P4_N=30
# P4_RECIPE: JSON override of the champ recipe (candidate stage-B reuse); default = registered champion.
# P4_ARMS: comma list, default "champ,plainc" (candidate arms may borrow the registered plainc rows
# from the same-device registered artifact -- equal-data moat row without re-spending 5h; ledgered).
_RECIPE_ENV = __import__("os").environ.get("P4_RECIPE")
ARMS = tuple(__import__("os").environ.get("P4_ARMS", "champ,plainc").split(","))
OUT = ROOT / "papers" / "figures" / f"p4_champion_confirm{TAG}.json"

CHAMP = dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.3)
if _RECIPE_ENV:
    CHAMP = json.loads(_RECIPE_ENV)
PLAIN = dict(ema_decay=0.99, var_coef=0.04, lr_scale=1.0, epochs=40)


def main() -> int:
    art: dict = {"champion": str(CHAMP), "plain_ctrl": str(PLAIN), "corpus": "c2000",
             "arm": {"tag": TAG or "_mps", "device": __import__("os").environ.get("P4_DEVICE", "default-mps")},
                 "gates": "C3-guar (<=1 all cells, >=90% qual) + C3-cal ([0.5,2], >=2/3 qual)",
                 "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] c2000 corpus + held-out ...")
    z = np.load(DATA_DIR / "corpus_c2000.npz")
    big = {k: z[k] for k in ("frames", "states", "actions")}
    obs, act, nxt = to_transitions_lean(big, big["frames"].shape[0])
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

    def audit_cell(cell: dict, pr) -> None:
        r"""Audit + dual-boundary C3 cells (amendment mechanics: store h_cert AND h_meas)."""
        rep = audit_gap(pr, fb_d, ach_d, window=16, k=3, burn_in=4, h_max=8)
        meas = measured_err(pr, fb_d, ach_d, 8)
        dm = rep["delta"]["mean"]
        cell["delta_norm"] = dm / (max(cell["std"], 1e-6) * (128 ** 0.5))
        cell["c3cal"] = []
        for em in EPS_MULT:
            hm = boundary_from_curve(meas["q90"], em * dm) or 0
            hc = boundary_from_curve(rep["certified_curve"]["err_q90"], em * dm) or 0
            cell["c3cal"].append({"em": em, "h_meas": int(hm), "h_cert": int(hc),
                                  "ratio": (hc / hm) if hm and hc else None})
        cell["growth"] = fit_growth(meas["median"])

    existing = json.loads(OUT.read_text()).get("cells", {}) if OUT.exists() else {}
    art["resumed_cells"] = [k for k, v in existing.items() if "c3cal" in v
                            and v["c3cal"] and "h_meas" in v["c3cal"][0]]

    for name in ARMS:
        print(f"[{name}] n=10 on c2000 ...")
        for r in range(NRUN):
            key = f"{name}_r{r}"
            prev = existing.get(key)
            ckp = DATA_DIR / f"ckpt8{TAG}_{name}_r{r}.pt"
            try:
                if prev and prev.get("c3cal") and "h_meas" in prev["c3cal"][0]:
                    cell = prev                                   # new-schema cell: keep
                elif prev and prev.get("stable") and ckp.exists():
                    cell = {k: v for k, v in prev.items()         # audit-refresh from ckpt
                            if k not in ("c3cal", "delta_norm", "growth")}
                    m = builders[name]()
                    ck = torch.load(ckp, map_location="cpu", weights_only=True)
                    m.load_state_dict(ck["model"])
                    m.encoder.load_state_dict(ck["target_encoder"])
                    audit_cell(cell, Pair(m.encoder.eval(), m.predictor))
                    cell["audit_refreshed"] = True
                else:
                    cell, pr, m, tgt = run_one(builders[name], cfgs[name], r, obs, act, nxt,
                                               fb_d, ach_d, fb_small, ho, aux_t)
                    if cell["stable"]:
                        audit_cell(cell, pr)
                        torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                                   DATA_DIR / f"ckpt8{TAG}_{name}_r{r}.pt")
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][key] = cell
            save()
            print(f"  {name} r{r}: std {cell.get('std', float('nan')):.3f} xy {cell.get('xy')} "
                  f"cells {[(x.get('h_cert'), x.get('h_meas')) for x in cell.get('c3cal', [])]}")

    verd = {}
    for name in ARMS:
        cs = [art["cells"].get(f"{name}_r{r}", {}) for r in range(NRUN)]
        qual = [c for c in cs if c.get("stable")]

        def cell_guar(c):
            r"""Registered text, faithfully: h_cert <= h_meas at ALL eps cells (0<=0 passes)."""
            return "c3cal" in c and all(x["h_cert"] <= x["h_meas"] for x in c["c3cal"])

        def cell_cal(c):
            r"""Band on cells where both boundaries > 0 (0/0 carries no calibration info)."""
            rs = [x["ratio"] for x in c.get("c3cal", []) if x["ratio"] is not None]
            return bool(rs) and all(BAND[0] <= x <= BAND[1] for x in rs)

        guar = [c for c in qual if cell_guar(c)]
        cal = [c for c in qual if cell_cal(c)]
        verd[name] = {
            "stable": f"{len(qual)}/{NRUN}",
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
