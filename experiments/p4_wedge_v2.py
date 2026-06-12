r"""P4 wedge lane v2 — leak-fixed C1b test (registered in the ledger before launch).

v1's diagnosis: only INITIAL angles were wedge-constrained; 31.1% of training timesteps drifted
out (median drift 41.7°/episode) ⇒ no power to separate the C1b hypotheses. v2 fix: training
transitions are filtered to chunk windows whose block angle stays in-wedge for ALL 6 window
frames. Eval sets unchanged (ho_in 73.3% in / ho_out 92.8% out — documented, not perfect).

Arms: eq-w2 n=8 (quorum head-room), plain-w2 n=6, eq count-control n=4 (UNFILTERED transitions
subsampled to the filtered count — separates wedge restriction from the data lever).

Gates (unchanged from v1, canonical src/audit/gates): G-W1 faithful one-sided on ho_out ≥90%
of stable eq-w2 (quorum 4); G-W2 sign gate median out/in δ̂: eq-w2 < plain-w2. Registered
prediction: C1b real ⇒ plain-w2 ratio rises materially above 1, eq-w2 stays ≈1 (transport).

Run: nohup .venv/bin/python -u experiments/p4_wedge_v2.py  (~2.5 h, MPS lane 2)
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
from experiments.p4_spine_stage1a import measured_err  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, pick_ladder, to_transitions,
)
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from experiments.p4_wedge_lane import EQ_CFG, PLAIN_CFG, load_set, transport_to_wedge  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.audit.gates import dual_boundary_cells, faithful_guar, violations  # noqa: E402

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_wedge_v2.json"
ARMS = (("eq_w2", "eq", True, 8), ("plain_w2", "plain", True, 6), ("eq_ctrl", "eq", False, 4))


def window_mask(corpus: dict, n_eps: int) -> torch.Tensor:
    r"""Chunk windows with block angle in-wedge for ALL 6 frames: (E*n_ch,) bool, transition order."""
    ang = torch.from_numpy(corpus["states"][:n_eps, :, 4]) % (2 * np.pi)   # (E, T+1)
    n_ch = corpus["actions"].shape[1] // CHUNK
    cols = [ (ang[:, j * CHUNK : j * CHUNK + CHUNK + 1] < np.pi / 2).all(dim=1)
             for j in range(n_ch) ]
    return torch.stack(cols, dim=1).reshape(-1)


def main() -> int:
    art: dict = {"design": "v2 leak-fixed (ledger-registered): in-wedge chunk filtering + count control",
                 "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] corpora + filtering ...")
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    mask = window_mask(corpus, 200)
    n_filt = int(mask.sum())
    art["filtering"] = {"kept": n_filt, "total": int(mask.numel()),
                        "frac": round(n_filt / mask.numel(), 3)}
    print(f"  filtered transitions: {n_filt}/{mask.numel()} ({art['filtering']['frac']:.1%})")
    idx_f = mask.nonzero().squeeze(1)
    rng = np.random.default_rng(0)
    idx_ctrl = torch.from_numpy(rng.choice(mask.numel(), n_filt, replace=False)).long()
    data = {"filtered": (obs[idx_f], act[idx_f], nxt[idx_f], aux_all[idx_f]),
            "control": (obs[idx_ctrl], act[idx_ctrl], nxt[idx_ctrl], aux_all[idx_ctrl])}
    save()

    ho_in, ho_out = load_set("wedge_ho_in"), load_set("wedge_ho_out")
    fb_in, ach_in = tensorize(ho_in)
    fb_out, ach_out = tensorize(ho_out)
    fb_tr, ach_tr = transport_to_wedge(ho_out)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    builders = {"eq": build_eq, "plain": (lambda: build_plain(w_match))}
    cfgs = {"eq": EQ_CFG, "plain": PLAIN_CFG}

    for arm, base, filtered, n_runs in ARMS:
        o_, a_, n_, x_ = data["filtered" if filtered else "control"]
        print(f"[{arm}] {n_runs} runs ({'filtered' if filtered else 'count-control'}) ...")
        for r in range(n_runs):
            try:
                cell, pr, m, tgt = run_one(builders[base], cfgs[base], r, o_, a_, n_,
                                           fb_in, ach_in, fb_small, ho_in, x_)
                if cell["stable"]:
                    for tag, fb, ach in (("in", fb_in, ach_in), ("out", fb_out, ach_out)):
                        rep = audit_gap(pr, fb, ach, window=16, k=3, burn_in=4, h_max=8)
                        meas = measured_err(pr, fb, ach, 8)
                        dm = rep["delta"]["mean"]
                        cells = dual_boundary_cells(meas["q90"],
                                                    rep["certified_curve"]["err_q90"], dm)
                        cell[tag] = {"delta": round(dm, 3), "cells": cells,
                                     "guar": faithful_guar(cells),
                                     "viol": len(violations(cells))}
                    if base == "eq":
                        rep_t = audit_gap(pr, fb_tr, ach_tr, window=16, k=3, burn_in=4, h_max=8)
                        cell["transported_delta"] = round(rep_t["delta"]["mean"], 3)
                    cell["out_in_ratio"] = round(cell["out"]["delta"] / cell["in"]["delta"], 3)
                    torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                               DATA_DIR / f"ckpt10_{arm}_r{r}.pt")
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][f"{arm}_r{r}"] = cell
            save()
            print(f"  {arm} r{r}: std {cell.get('std', float('nan')):.3f} "
                  f"in {cell.get('in', {}).get('delta')} out {cell.get('out', {}).get('delta')} "
                  f"ratio {cell.get('out_in_ratio')} guar {cell.get('out', {}).get('guar')}")

    verd: dict = {}
    stable = {arm: [c for k, c in art["cells"].items()
                    if k.startswith(arm) and c.get("stable") and "out" in c]
              for arm, *_ in ARMS}
    eq_s = stable["eq_w2"]
    if len(eq_s) < 4:
        verd["G_W1"] = f"INCONCLUSIVE-BY-STABILITY (stable eq_w2 {len(eq_s)}/8)"
    else:
        n_pass = sum(c["out"]["guar"] for c in eq_s)
        verd["G_W1"] = {"verdict": "PASS" if n_pass / len(eq_s) >= 0.9 else "FAIL",
                        "passing": f"{n_pass}/{len(eq_s)}"}
    ratios = {arm: sorted(c["out_in_ratio"] for c in cs) for arm, cs in stable.items()}
    if len(ratios["eq_w2"]) >= 3 and len(ratios["plain_w2"]) >= 3:
        med = {a: float(np.median(v)) for a, v in ratios.items() if v}
        rng2 = np.random.default_rng(0)
        ci = {a: [round(float(np.percentile([np.median(rng2.choice(v, len(v)))
                                             for _ in range(2000)], q)), 3) for q in (5, 95)]
              for a, v in ratios.items() if v}
        verd["G_W2"] = {"verdict": "PASS" if med["eq_w2"] < med["plain_w2"] else "FAIL",
                        "median_out_in": med, "ci90": ci, "ratios": ratios}
    else:
        verd["G_W2"] = f"INCONCLUSIVE-BY-STABILITY ({ {a: len(v) for a, v in ratios.items()} })"
    if eq_s:
        verd["orbit_transport"] = [(c["transported_delta"], c["in"]["delta"]) for c in eq_s]
    art["verdicts"] = verd
    save()
    print("VERDICTS:", json.dumps(verd)[:500])
    print(f"WEDGE V2 DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
