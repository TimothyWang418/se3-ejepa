r"""Low-N figure batch — densify the P(fail)-vs-N curve (paper figure) + eq low-N line.

Registered (descriptive, no gate): plain counts {2800, 3000, 3200} x n=6 (cliff boundary
densification; n=2 was too thin for P(fail) estimates) + eq counts {1500, 2000} x n=3 (does eq
cliff eventually? completes the figure's eq line; 2606 refs exist from wedge eq_ctrl). All on
unfiltered wedge-corpus transitions, fixed-rng subsample per (count, run) — the cliff-probe
protocol verbatim. Device knob P4_DEVICE (CUDA target; coexists nice-10 with paper2's batch).

Run (box): SDL_VIDEODRIVER=dummy P4_DEVICE=cuda nohup nice -n 10 .venv3/bin/python -u \
    experiments/p4_lown_figure_batch.py > ~/p4_lown.log 2>&1 &
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
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from experiments.p4_wedge_lane import EQ_CFG, PLAIN_CFG, load_set  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_lown_figure.json"
PLAIN_COUNTS = ((2800, 6), (3000, 6), (3200, 6))
EQ_COUNTS = ((1500, 3), (2000, 3))


def main() -> int:
    art: dict = {"cells": {}}
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    def block(arm, builder, cfg, counts):
        for n_count, n_runs in counts:
            for r in range(n_runs):
                key = f"{arm}_c{n_count}_r{r}"
                try:
                    rng = np.random.default_rng(100 * r)
                    idx = torch.from_numpy(rng.choice(obs.shape[0], n_count, replace=False)).long()
                    cell, pr, *_ = run_one(builder, cfg, r, obs[idx], act[idx], nxt[idx],
                                           fb_in, ach_in, fb_small, ho_in, aux_all[idx])
                    if cell["stable"]:
                        rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
                        cell["in_delta"] = round(rep["delta"]["mean"], 3)
                except Exception as exc:  # noqa: BLE001
                    cell = {"error": str(exc)[:200], "stable": False}
                art["cells"][key] = cell
                art["elapsed_min"] = round((time.time() - T0) / 60, 1)
                OUT.write_text(json.dumps(art, indent=1))
                print(f"  {key}: std {cell.get('std', float('nan')):.2f} δ {cell.get('in_delta')}")

    block("plain", lambda: build_plain(w_match), PLAIN_CFG, PLAIN_COUNTS)
    block("eq", build_eq, EQ_CFG, EQ_COUNTS)
    print(f"LOWN FIGURE BATCH DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
