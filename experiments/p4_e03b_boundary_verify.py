r"""E0.3b — dual-boundary verification pass for the shape-generalization audits.

Why (review pass 2026-06-11): the CPU-batch script used the pre-amendment ratio semantics —
``None`` (measured boundary 0 at tight ε) auto-passed one-sidedness. 11/120 cells were None.
Faithful semantics require the boundary PAIR: a hidden ``h_cert > h_meas = 0`` would be a real
violation. Collection is deterministic (seed 7, CPU), so the audits are exactly reproducible;
this pass recomputes all 30 (pair × shape-set) audits storing both boundaries at every ε cell.

Run: nohup nice .venv/bin/python experiments/p4_e03b_boundary_verify.py  (~10 min CPU)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_cpu_batch_0612 import SHAPES, collect_shape, tensorize  # noqa: E402
from experiments.p4_spine_stage1a import boundary_from_curve, measured_err  # noqa: E402
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    DATA_DIR, build_eq, build_plain, collect_weakpolicy, pick_ladder,
)
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
EPS_MULT = (2, 4, 8, 16)
OUT = ROOT / "papers" / "figures" / "p4_e03b_boundaries.json"


def main() -> int:
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    pairs = {}
    for name, builder, runs in (("eq", build_eq, (0, 1, 3)),
                                ("plain", lambda: build_plain(w_match), (0, 1, 2))):
        for r in runs:
            ck = torch.load(DATA_DIR / f"ckpt7_{'eqwin' if name == 'eq' else 'plainwin'}_r{r}.pt",
                            map_location="cpu", weights_only=True)
            m = builder()
            m.load_state_dict(ck["model"])
            m.encoder.load_state_dict(ck["target_encoder"])
            pairs[f"{name}_r{r}"] = Pair(m.encoder.eval(), m.predictor)

    sets = {"T(ref)": collect_weakpolicy(30, seed=7)}
    for sname, sidx in SHAPES.items():
        sets[sname] = collect_shape(30, seed=7, shape_idx=sidx)

    art: dict = {"semantics": "faithful dual-boundary (champion amendment transplanted)",
                 "rows": {}, "violations": []}
    for sname, ho in sets.items():
        fb, ach = tensorize(ho)
        row = {}
        for pname, pr in pairs.items():
            rep = audit_gap(pr, fb, ach, window=16, k=3, burn_in=4, h_max=8, n_boot=200)
            meas = measured_err(pr, fb, ach, 8)
            dm = rep["delta"]["mean"]
            cells = []
            for em in EPS_MULT:
                hm = int(boundary_from_curve(meas["q90"], em * dm) or 0)
                hc = int(boundary_from_curve(rep["certified_curve"]["err_q90"], em * dm) or 0)
                cells.append({"em": em, "h_cert": hc, "h_meas": hm})
                if hc > hm:
                    art["violations"].append(f"{sname}/{pname}@em={em}: cert {hc} > meas {hm}")
            row[pname] = {"delta": round(dm, 3), "cells": cells,
                          "guar_faithful": all(c["h_cert"] <= c["h_meas"] for c in cells)}
        art["rows"][sname] = row
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))
        print(f"[{sname}] " + " ".join(
            f"{p}:{'✓' if v['guar_faithful'] else '✗'}" for p, v in row.items()))

    n_rows = sum(len(r) for r in art["rows"].values())
    n_pass = sum(v["guar_faithful"] for r in art["rows"].values() for v in r.values())
    print(f"FAITHFUL GUAR: {n_pass}/{n_rows} rows; violations: {len(art['violations'])}")
    if art["violations"]:
        for v in art["violations"]:
            print("  VIOLATION:", v)
    print(f"DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
