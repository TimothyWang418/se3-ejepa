r"""2D resolution ladder — does resolution open the planner GO regime? (E-next, post-campaign)

Campaign D4 located the binding constraint: at 96px, τ_pos = 20 env units ≈ 3.8 render px sits
at the encoder's resolution floor (ε_task ≤ 2.75 < δ̂). The data lever is measured flat
(nightshift datascale δ̂: 2.59/2.29/3.05 across c500/1000/2000). This ladder tests the
RESOLUTION lever: {96 (ref), 144, 192} px × banked recipe (κ=0) × n=4, 200 eps per res.

**GO criterion (registered now): ε_task(τ) ≥ δ̂ at some rung** — the D1 binding's sign flips.
Readouts per rung: stability, xy probe, δ̂ (audit), ε_task(τ) re-bound by the D1-v2 estimator
(within-episode pairs |Δt| ≤ 3 + random; conditional q90 vs the env's own τ). Encoder is
resolution-agnostic (PointwiseAdaptiveAvgPool — verified 96/144/192 → (B,128)).

Run (box): SDL_VIDEODRIVER=dummy P4_DEVICE=cuda nohup nice -n 5 .venv3/bin/python -u \
    experiments/p4_res_ladder.py > ~/p4_res.log 2>&1 &   (~1 h)
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

from experiments.p4_step1_pipeline import CHUNK, build_eq, make_env, weak_action  # noqa: E402
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
RUNGS = (96, 144, 192)
RECIPE = dict(ema_decay=0.99, var_coef=0.3, lr_scale=1.0, aux_coef=0.5)   # banked (kappa=0)
TAU_POS, TAU_TH = 20.0, float(np.pi / 9)
OUT = ROOT / "papers" / "figures" / "p4_res_ladder.json"


def circ_mask_res(x: torch.Tensor, res: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(res), torch.arange(res), indexing="ij")
    m = ((xx - (res - 1) / 2) ** 2 + (yy - (res - 1) / 2) ** 2
         <= ((res - 1) / 2) ** 2).to(x.dtype)
    return x * m


def collect_res(n_episodes: int, seed: int, res: int) -> dict:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    env = gym.make("swm/PushT-v1", resolution=res)
    rng = np.random.default_rng(seed)
    frames, states, actions = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        fr = [env.render()]
        st = [obs["state"]]
        ac = []
        for _ in range(100):
            a = weak_action(env, rng)
            obs, *_ = env.step(a)
            fr.append(env.render()); st.append(obs["state"]); ac.append(a)
        frames.append(np.stack(fr)); states.append(np.stack(st)); actions.append(np.stack(ac))
    env.close()
    return {"frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float64),
            "actions": np.stack(actions).astype(np.float32)}


def to_transitions_res(data: dict, res: int, batch_eps: int = 40):
    r"""Episode-batched (lean) — the eager version OOM-killed the 192px rung (9GB float
    spike + copies vs 23GB box; autopsy 06-12). Numerically identical per episode."""
    obs_l, nxt_l, fb_l = [], [], []
    a = torch.from_numpy(data["actions"])
    n_ch = a.shape[1] // CHUNK
    n_eps = data["frames"].shape[0]
    for i in range(0, n_eps, batch_eps):
        f = torch.from_numpy(data["frames"][i:i+batch_eps]).float().div_(255.0).permute(0, 1, 4, 2, 3)
        f = circ_mask_res(f.reshape(-1, 3, res, res), res).reshape(f.shape)
        obs_l.append(f[:, 0 : n_ch * CHUNK : CHUNK].reshape(-1, 3, res, res))
        nxt_l.append(f[:, CHUNK : n_ch * CHUNK + 1 : CHUNK].reshape(-1, 3, res, res))
        fb_l.append(f[:, ::CHUNK])
    act = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).reshape(-1, CHUNK * 2)
    return torch.cat(obs_l), act, torch.cat(nxt_l), torch.cat(fb_l)


def eps_task_bind(z: torch.Tensor, sb: torch.Tensor, n_eps: int) -> float | None:
    r"""D1-v2 estimator verbatim (within-episode pairs |Δt|≤3 + 20k random)."""
    n = len(sb); n_t = n // n_eps
    rng = np.random.default_rng(0)
    wi, wj = [], []
    for e in range(n_eps):
        for t_ in range(n_t):
            for dt in (1, 2, 3):
                if t_ + dt < n_t:
                    wi.append(e * n_t + t_); wj.append(e * n_t + t_ + dt)
    ii = torch.cat([torch.tensor(wi), torch.from_numpy(rng.integers(0, n, 20_000))])
    jj = torch.cat([torch.tensor(wj), torch.from_numpy(rng.integers(0, n, 20_000))])
    dz = (z[ii] - z[jj]).norm(dim=1).numpy()
    dpos = (sb[ii, 2:4] - sb[jj, 2:4]).norm(dim=1).numpy()
    dth_raw = (sb[ii, 4] - sb[jj, 4]).abs().numpy() % (2 * np.pi)
    dth = np.minimum(dth_raw, 2 * np.pi - dth_raw)
    out = None
    for rr in np.arange(0.5, 12.01, 0.25):
        sel = dz <= rr
        if sel.sum() < 100:
            continue
        if np.quantile(dpos[sel], 0.9) <= TAU_POS and np.quantile(dth[sel], 0.9) <= TAU_TH:
            out = float(rr)
        else:
            break
    return out


def main() -> int:
    art: dict = {"go_criterion": "eps_task(tau) >= delta_hat at some rung", "rungs": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    if OUT.exists():                                   # rung-level resume
        art.update(json.loads(OUT.read_text()))
        art.pop("verdict", None)
    for res in RUNGS:
        done = art["rungs"].get(str(res), [])
        if len([c for c in done if "std" in c or "error" in c]) >= 4:
            print(f"[res {res}] resumed from artifact, skip")
            continue
        print(f"[res {res}] collect ...")
        corpus = collect_res(200, seed=0, res=res)
        obs, act, nxt, _ = to_transitions_res(corpus, res)
        aux_t = state_targets(corpus)
        ho = collect_res(60, seed=1, res=res)
        obs_h, _, _, fbh = to_transitions_res(ho, res)
        fb_d = fbh.double()
        a1 = torch.from_numpy(ho["actions"])
        n_ch = a1.shape[1] // CHUNK
        ach_d = a1[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
        fb_small = fbh[:20].reshape(-1, 3, res, res).float()
        sb = torch.from_numpy(ho["states"]).double()[:, 0 : n_ch * CHUNK + 1 : CHUNK]
        sb = sb.reshape(-1, sb.shape[-1])
        cells = []
        for r in range(4):
            try:
                cell, pr, m, tgt = run_one(build_eq, RECIPE, r, obs, act, nxt,
                                           fb_d, ach_d, fb_small, ho, aux_t)
                if cell["stable"]:
                    rep = audit_gap(pr, fb_d, ach_d, window=16, k=3, burn_in=4, h_max=8)
                    cell["delta"] = round(rep["delta"]["mean"], 3)
                    with torch.no_grad():
                        flat = fbh.reshape(-1, 3, res, res)
                        z = torch.cat([tgt.eval()(flat[i:i+128].float())
                                       for i in range(0, len(flat), 128)]).cpu().double()
                    cell["eps_task"] = eps_task_bind(z, sb, 60)
                    cell["go"] = bool(cell["eps_task"] and cell["eps_task"] >= cell["delta"])
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            cells.append({k: v for k, v in cell.items()
                          if k in ("std", "xy", "delta", "eps_task", "go", "stable", "error")})
            print(f"  res{res} r{r}: std {cell.get('std', float('nan')):.2f} "
                  f"δ̂ {cell.get('delta')} ε_task {cell.get('eps_task')} GO {cell.get('go')}")
            art["rungs"][str(res)] = cells
            save()
    n_go = sum(1 for cs in art["rungs"].values() for c in cs if c.get("go"))
    art["verdict"] = {"any_go": n_go > 0, "n_go_cells": n_go}
    save()
    print("VERDICT:", art["verdict"])
    print(f"RES LADDER DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
