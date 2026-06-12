r"""Integrated campaign D1 — bind τ / ε_reach / ε_task and FREEZE before any planner run.

Spec: docs/specs/2026-06-12-p4-integrated-campaign.md. Bindings (all formulas registered):
- τ from the env's OWN success test (pusht/env.py:348-352): pos_diff < 20 ∧ angle_diff < π/9.
- ε_reach(pair) = 2 δ̂(pair), δ̂ from the candidate stage-B audit cells (registered instrument).
- ε_task(pair) = largest latent radius r (grid) with q90{state error | ‖Δz‖ ≤ r} within τ on
  BOTH axes, estimated from 20k random held-out frame pairs (EMA-target latents — the same
  space the certificate curves live in).

Output: papers/figures/p4_campaign_d1.json — frozen header for D2/D3.
Run: .venv/bin/python -u experiments/p4_campaign_d1.py   (~10 min CPU)
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
    CHUNK, DATA_DIR, RES, build_eq, circ_mask, collect_weakpolicy,
)

T0 = time.time()
TAU_POS, TAU_TH = 20.0, float(np.pi / 9)            # env's own success test, verbatim
OUT = ROOT / "papers" / "figures" / "p4_campaign_d1.json"
M_PAIRS = 20_000
R_GRID = np.arange(0.5, 12.01, 0.25)   # estimator v2: fine grid in the resolution-floor regime


def main() -> int:
    cand = json.loads((ROOT / "papers/figures/p4_champion_confirm_cand.json").read_text())
    stable = {k: v for k, v in cand["cells"].items()
              if k.startswith("champ") and v.get("stable")}
    print(f"[D1] {len(stable)} stable candidate pairs")

    ho = collect_weakpolicy(60, seed=1)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    flat = f[:, ::CHUNK].reshape(-1, 3, RES, RES)
    s = torch.from_numpy(ho["states"]).double()
    n_ch = (s.shape[1] - 1) // CHUNK
    sb = s[:, 0 : n_ch * CHUNK + 1 : CHUNK].reshape(-1, s.shape[-1])
    xy, th = sb[:, 2:4], sb[:, 4]

    # Estimator v2 (D1 iteration, ledgered): random cross-episode pairs alone cannot populate
    # small-||dz|| balls (first freeze returned eps_task=None everywhere). Add ALL within-episode
    # pairs at |dt| in {1,2,3} chunks — dense small-radius coverage; criterion unchanged.
    rng = np.random.default_rng(0)
    n = len(sb)
    n_eps_ho, n_t = 60, n // 60
    wi, wj = [], []
    for e in range(n_eps_ho):
        base = e * n_t
        for t_ in range(n_t):
            for dt in (1, 2, 3):
                if t_ + dt < n_t:
                    wi.append(base + t_); wj.append(base + t_ + dt)
    ii = torch.cat([torch.tensor(wi), torch.from_numpy(rng.integers(0, n, M_PAIRS))])
    jj = torch.cat([torch.tensor(wj), torch.from_numpy(rng.integers(0, n, M_PAIRS))])
    dpos = (xy[ii] - xy[jj]).norm(dim=1).numpy()
    dth_raw = (th[ii] - th[jj]).abs().numpy() % (2 * np.pi)
    dth = np.minimum(dth_raw, 2 * np.pi - dth_raw)

    art: dict = {"tau": {"pos": TAU_POS, "theta": TAU_TH, "source": "pusht env.py:348-352"},
                 "eps_reach_formula": "2 * delta_hat(pair)", "pairs": {}}
    for key, cell in stable.items():
        r = int(key.rsplit("_r", 1)[1])
        ckp = DATA_DIR / f"ckpt8_cand_champ_r{r}.pt"
        if not ckp.exists():
            continue
        ck = torch.load(ckp, map_location="cpu", weights_only=True)
        m = build_eq()
        m.load_state_dict(ck["model"])
        m.encoder.load_state_dict(ck["target_encoder"])
        enc = m.encoder.eval()
        with torch.no_grad():
            z = torch.cat([enc(flat[i : i + 128]) for i in range(0, len(flat), 128)]).double()
        dz = (z[ii] - z[jj]).norm(dim=1).numpy()
        def bind(mult: float):
            r"""v3 tolerance ladder: primary gates at tau; 2tau/4tau = descriptive context +
            C4a backup contrast layer (registered BEFORE D2 — the resolution-floor risk was
            visible from the bindings alone, no planner data seen)."""
            out = None
            for rr in R_GRID:
                sel = dz <= rr
                if sel.sum() < 100:
                    continue
                if (np.quantile(dpos[sel], 0.9) <= TAU_POS * mult
                        and np.quantile(dth[sel], 0.9) <= TAU_TH * mult):
                    out = float(rr)
                else:
                    break
            return out

        eps_task = bind(1.0)
        delta = cell["delta_norm"] * max(cell["std"], 1e-6) * (128 ** 0.5)
        art["pairs"][f"r{r}"] = {
            "delta": round(delta, 3), "eps_reach": round(2 * delta, 3),
            "eps_task": eps_task, "eps_task_2tau": bind(2.0), "eps_task_4tau": bind(4.0),
            "xy": round(cell.get("xy", 0), 3),
        }
        v = art["pairs"][f"r{r}"]
        print(f"  r{r}: δ̂ {delta:.2f} ε_reach {2*delta:.2f} "
              f"ε_task {v['eps_task']}/{v['eps_task_2tau']}/{v['eps_task_4tau']} (τ/2τ/4τ)")

    healthy = [v["eps_task"] for v in art["pairs"].values() if v["eps_task"]]
    art["eps_task_median"] = round(float(np.median(healthy)), 3) if healthy else None
    art["frozen_at"] = "D1 complete - values immutable for D2/D3"
    art["elapsed_min"] = round((time.time() - T0) / 60, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"ε_task median (healthy): {art['eps_task_median']}")
    print(f"D1 FROZEN ({art['elapsed_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
