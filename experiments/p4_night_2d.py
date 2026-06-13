r"""2D night batch (Mac MPS) — autonomous, crash-immune. Chains the 2D armoring blocks after
the running T1.2 finishes. Health + δ̂ audits inline (CPU f64, cheap at 2D scale).

Queue (numbered, after #4=T1.2 cliff-factor already running):
  #8  T1.3 finer reliability cliff: plain N∈{2400,2800,3200,3600,3800}×n=8 (publication P(fail))
  #9  T1.4 eq cliff: N∈{1200,1500,1750,2000}×n=6 (pins eq's own threshold)
  #10 T1.5 κ=0.8 C1a treatment: winner_v02 + (aux0.5+10d) bases, n=8, with δ̂ audit (expansive)

Run: nohup .venv/bin/python -u experiments/p4_night_2d.py > /tmp/night_2d.log 2>&1 &
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
    CHUNK, RES, build_eq, build_plain, make_env, pick_ladder, to_transitions, weak_action,
)
from experiments.p4_step2_kappa_gate import get_ext  # noqa: E402
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from experiments.p4_wedge_lane import EQ_CFG, PLAIN_CFG, load_set  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_night_2d.json"
T12_LOG = "/tmp/p4_cliff_factor.log"
WINNER_V02 = dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0)


def main() -> int:
    art: dict = {"queue": "#8 finer cliff / #9 eq cliff / #10 kappa=0.8 C1a", "blocks": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    # wait for T1.2 (cliff factor) to finish (up to 2.5h guard)
    print("[wait] T1.2 ...")
    for _ in range(300):
        try:
            if "CLIFF FACTOR DONE" in Path(T12_LOG).read_text():
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(30)
    print(f"[wait] done {(time.time()-T0)/60:.0f}min")

    # shared 2D corpus (wedge corpus, the cliff convention)
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_all = state_targets(corpus)
    ho_in = load_set("wedge_ho_in")
    fb_in, ach_in = tensorize(ho_in)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    def cliff_cell(builder, cfg, n_count, r):
        rng = np.random.default_rng(100 * r)
        idx = torch.from_numpy(rng.choice(obs.shape[0], n_count, replace=False)).long()
        cell, pr, *_ = run_one(builder, cfg, r, obs[idx], act[idx], nxt[idx],
                               fb_in, ach_in, fb_small, ho_in, aux_all[idx])
        if cell.get("stable"):
            rep = audit_gap(pr, fb_in, ach_in, window=16, k=3, burn_in=4, h_max=8)
            cell["in_delta"] = round(rep["delta"]["mean"], 3)
        return {k: v for k, v in cell.items() if k in ("std", "in_delta", "stable")}

    # #8 finer plain cliff
    print("[#8] finer plain cliff ...")
    art["blocks"]["finer_cliff"] = {}
    for n_count in (2400, 2800, 3200, 3600, 3800):
        cells = []
        for r in range(8):
            try:
                cells.append(cliff_cell(lambda: build_plain(w_match), PLAIN_CFG, n_count, r))
            except Exception as exc:  # noqa: BLE001
                cells.append({"error": str(exc)[:120], "stable": False})
        ds = [c["in_delta"] for c in cells if c.get("in_delta")]
        art["blocks"]["finer_cliff"][f"N{n_count}"] = {
            "cells": cells, "p_fail": round(sum(d > 6 for d in ds) / max(len(ds), 1), 2)}
        print(f"  N{n_count}: P(fail)={art['blocks']['finer_cliff'][f'N{n_count}']['p_fail']}")
        save()

    # #9 eq cliff
    print("[#9] eq cliff ...")
    art["blocks"]["eq_cliff"] = {}
    for n_count in (1200, 1500, 1750, 2000):
        cells = []
        for r in range(6):
            try:
                cells.append(cliff_cell(build_eq, EQ_CFG, n_count, r))
            except Exception as exc:  # noqa: BLE001
                cells.append({"error": str(exc)[:120], "stable": False})
        ds = [c["in_delta"] for c in cells if c.get("in_delta")]
        art["blocks"]["eq_cliff"][f"N{n_count}"] = {
            "cells": cells, "p_fail": round(sum(d > 6 for d in ds) / max(len(ds), 1), 2)}
        print(f"  eq N{n_count}: P(fail)={art['blocks']['eq_cliff'][f'N{n_count}']['p_fail']}")
        save()

    # #10 kappa=0.8 C1a treatment (winner_v02 + 10d anchor), with audit
    print("[#10] kappa=0.8 C1a ...")
    art["blocks"]["k08_c1a"] = {}

    def collect_k08(n, seed):
        env = make_env(0.8); rng = np.random.default_rng(seed)
        fr, st, ac = [], [], []
        for ep in range(n):
            env.reset(seed=seed * 100000 + ep)
            f = [env.render()]; s = [get_ext(env)]; a = []
            for _ in range(100):
                act_ = weak_action(env, rng); env.step(act_)
                f.append(env.render()); s.append(get_ext(env)); a.append(act_)
            fr.append(np.stack(f)); st.append(np.stack(s)); ac.append(np.stack(a))
        env.close()
        return {"frames": np.stack(fr).astype(np.uint8), "states": np.stack(st).astype(np.float64),
                "actions": np.stack(ac).astype(np.float32)}

    ck = collect_k08(200, 0); ho = collect_k08(60, 1)
    o2, a2, n2 = to_transitions(ck, 200)
    s = torch.from_numpy(ck["states"]).float()
    nch = (s.shape[1] - 1) // CHUNK
    aux10 = ((s[:, 0:nch * CHUNK:CHUNK, :10].reshape(-1, 10)
              - s[:, 0:nch * CHUNK:CHUNK, :10].reshape(-1, 10).mean(0))
             / (s[:, 0:nch * CHUNK:CHUNK, :10].reshape(-1, 10).std(0) + 1e-6))
    fb2, ach2 = tensorize(ho)
    fbs = fb2[:20].reshape(-1, 3, RES, RES).float()
    for name, cfg, aux in (("winner_v02", WINNER_V02, None),
                           ("aux10d", dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.5), aux10)):
        cells = []
        for r in range(8):
            try:
                cell, pr, *_ = run_one(build_eq, cfg, r, o2, a2, n2, fb2, ach2, fbs, ho,
                                       aux if aux is not None else state_targets(ck))
                if cell.get("stable"):
                    rep = audit_gap(pr, fb2, ach2, window=16, k=3, burn_in=4, h_max=8)
                    cell["delta_k08"] = round(rep["delta"]["mean"], 3)
                cells.append({k: v for k, v in cell.items()
                              if k in ("std", "xy", "delta_k08", "stable")})
            except Exception as exc:  # noqa: BLE001
                cells.append({"error": str(exc)[:120], "stable": False})
        art["blocks"]["k08_c1a"][name] = {
            "cells": cells, "stable": sum(c.get("stable", False) for c in cells)}
        print(f"  k08/{name}: stable {art['blocks']['k08_c1a'][name]['stable']}/8")
        save()

    print(f"NIGHT 2D DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
