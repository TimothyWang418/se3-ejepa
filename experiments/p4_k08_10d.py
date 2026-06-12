r"""κ=0.8 10-d anchor falsification (skeleton hole #5) — the non-Markov anchoring hypothesis.

Rescue-sweep finding: aux anchors INVERT to a liability at κ=0.8 (dose ordering aux0.5 1/4 <
aux0.3 2/4 < none 3/4). Registered hypothesis: the 7-d state target omits block momentum
(step2's underdetermination finding) — anchoring latents to an incomplete, non-Markov state
fights velocity encoding. **Falsification (registered): a 10-d momentum-complete anchor
(get_ext) restores stability.** Head-to-head on ONE fresh κ=0.8 corpus: aux0.5 × {7-d, 10-d}
targets × n=4 (same normalization builder both arms). Prediction: 10-d ≥ 3/4 stable, 7-d ≤ 2/4.

Run: nohup .venv/bin/python -u experiments/p4_k08_10d.py > /tmp/p4_k08_10d.log 2>&1 &
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
    CHUNK, RES, build_eq, make_env, to_transitions, weak_action,
)
from experiments.p4_step2_kappa_gate import get_ext  # noqa: E402
from experiments.p4_v16_stageA_sweep import run_one  # noqa: E402

T0 = time.time()
KAPPA = 0.8
RECIPE = dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.5)
OUT = ROOT / "papers" / "figures" / "p4_k08_10d.json"


def collect_ext(n_episodes: int, seed: int) -> dict:
    r"""κ=0.8 episodes recording the 10-d extended state (block momentum included)."""
    env = make_env(KAPPA)
    rng = np.random.default_rng(seed)
    frames, exts, actions = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed * 100_000 + ep)
        f = [env.render()]
        s = [get_ext(env)]
        a = []
        for _ in range(100):
            act = weak_action(env, rng)
            env.step(act)
            f.append(env.render()); s.append(get_ext(env)); a.append(act)
        frames.append(np.stack(f)); exts.append(np.stack(s)); actions.append(np.stack(a))
    env.close()
    return {"frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(exts).astype(np.float64),
            "actions": np.stack(actions).astype(np.float32)}


def targets(corpus: dict, dims: int) -> torch.Tensor:
    r"""Chunk-boundary anchor targets, z-scored per dim — SAME builder for both arms."""
    s = torch.from_numpy(corpus["states"]).float()[..., :dims]
    n_ch = (s.shape[1] - 1) // CHUNK
    st = s[:, 0 : n_ch * CHUNK : CHUNK].reshape(-1, dims)
    return (st - st.mean(0)) / (st.std(0) + 1e-6)


def main() -> int:
    art: dict = {"hypothesis": "10-d momentum-complete anchor restores stability at k=0.8",
                 "prediction": "10d >= 3/4 stable, 7d <= 2/4", "cells": {}}
    corpus = collect_ext(200, seed=0)
    obs, act, nxt = to_transitions(corpus, 200)
    ho = collect_ext(60, seed=1)
    fb, ach = tensorize(ho)
    fb_small = fb[:20].reshape(-1, 3, RES, RES).float()

    for arm, dims in (("aux7d", 7), ("aux10d", 10)):
        x_ = targets(corpus, dims)
        for r in range(4):
            try:
                cell, *_ = run_one(build_eq, RECIPE, r, obs, act, nxt,
                                   fb, ach, fb_small, ho, x_)
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][f"{arm}_r{r}"] = {k: v for k, v in cell.items()
                                           if k in ("std", "xy", "stable", "error")}
            art["elapsed_min"] = round((time.time() - T0) / 60, 1)
            OUT.write_text(json.dumps(art, indent=1))
            print(f"  {arm} r{r}: std {cell.get('std', float('nan')):.2f} "
                  f"stable {cell.get('stable')}")
    verd = {}
    for arm in ("aux7d", "aux10d"):
        ok = sum(1 for k, c in art["cells"].items() if k.startswith(arm) and c.get("stable"))
        verd[arm] = f"{ok}/4"
    s10 = int(verd["aux10d"].split("/")[0]); s7 = int(verd["aux7d"].split("/")[0])
    verd["hypothesis"] = ("SUPPORTED" if s10 >= 3 and s7 <= 2
                          else ("REFUTED" if s10 <= s7 else "MIXED"))
    art["verdict"] = verd
    OUT.write_text(json.dumps(art, indent=1))
    print("VERDICT:", verd)
    print(f"K08 10D DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
