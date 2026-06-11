r"""P4-spine Stage-1b — the planner side: $H^{\text{plan}}$ boundary, crossover, $\hat\theta^\ast$.

Spec: docs/specs/2026-06-10-p4-spine-ep43-seed.md §2 (planner-side boundary; θ̂* elicitation).
Seed-0 descriptive (gates need 3 seeds); registered design decisions:

- **Fixed-goal windows**: the Stage-1a held-out set has per-episode random goals that were not
  stored — re-resetting would redraw the goal and change the goal-zone pixels under the encoder.
  Stage-1b therefore uses a FRESH fixed-goal collection (seed 2, `FIXED_GOAL` shared with the
  G-training pipeline), and computes its OWN measured model curve on these windows so the
  crossover comparison is apples-to-apples.
- **Planner** = subgoal-conditioned short-window CEM in the deployable latent space: minimize
  $\lVert f^{(H)}(z_0, a_{1..H}) - z^\ast \rVert$ over $H$ action chunks (K=64, 3 iters, elites 8),
  execute in the TRUE env, encode the reached frame, reach distance to $z^\ast$.
- **Readouts** (units: H in f-chunks; ε in $\hat\delta_f$ multiples per base):
  (i) reach-rate(H, ε_reach) for H ∈ {1,2,3,5,8}, ε_reach ∈ {2,4,8}·δ̂ →
  $H^{\text{plan}}_{\max}$; (ii) crossover vs the same-windows model boundary; (iii) θ̂*: at
  H=3, plan toward NOISED subgoals ($z^\ast + \nu\,\hat\delta\,u$, $u$ random unit), reach-rate
  to the TRUE subgoal at ε_reach=4δ̂ vs ν ∈ {0,1,2,4,8,16} → θ̂* = max ν with rate ≥ 0.5.

Run: .venv/bin/python experiments/p4_spine_stage1b.py   (~5-10 min)
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, make_env, pick_ladder, weak_action,
)
from experiments.p4_step10_g_trainer import FIXED_GOAL  # noqa: E402

SEED = 2
N_EP = 30
EP_LEN = 100
H_GRID = (1, 2, 3, 5, 8)
EPS_MULT = (2, 4, 8)
NU_GRID = (0, 1, 2, 4, 8, 16)
N_WIN = 30
K, ITERS, ELITE = 64, 3, 8
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_spine_stage1b.json"


def collect_fixed_goal(n_episodes: int, seed: int) -> dict:
    env = make_env(None)
    rng = np.random.default_rng(seed)
    frames, states, actions = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed * 100_000 + ep)
        env.reset(options={"goal_state": FIXED_GOAL,
                           "state": np.asarray(env.unwrapped._get_obs(), dtype=np.float64)})
        f = [env.render()]
        s = [np.asarray(env.unwrapped._get_obs(), dtype=np.float64)]
        a = []
        for _ in range(EP_LEN):
            act = weak_action(env, rng)
            ob, _r, _t, _tr, _ = env.step(act)
            f.append(env.render())
            s.append(ob["state"])
            a.append(act)
        frames.append(np.stack(f)); states.append(np.stack(s)); actions.append(np.stack(a))
    env.close()
    return {"frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float64),
            "actions": np.stack(actions).astype(np.float32)}


class Deploy:
    r"""Deployable pair (ckpt3): target encoder for encoding, f64 predictor for rollouts."""

    def __init__(self, name: str, builder):
        ck = torch.load(DATA_DIR / f"ckpt3_{name}_f200.pt", map_location="cpu", weights_only=True)
        m = builder()
        m.load_state_dict(ck["model"])
        m.encoder.load_state_dict(ck["target_encoder"])
        self.enc = m.encoder.eval().to(DEVICE)
        self.pred = copy.deepcopy(m.predictor).double().eval()
        for p in self.pred.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames_u8: np.ndarray) -> torch.Tensor:  # (..., 96,96,3) -> (..., D) f64
        f = torch.from_numpy(frames_u8).float().div_(255.0)
        lead = f.shape[:-3]
        f = circ_mask(f.reshape(-1, RES, RES, 3).permute(0, 3, 1, 2))
        z = self.enc(f.to(DEVICE)).cpu().double()
        return z.reshape(*lead, -1)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        r"""z0 (B, D) f64; acts (B, H, 10) f64 -> final latent (B, D)."""
        z = z0
        for h in range(acts.shape[1]):
            z = self.pred(z, acts[:, h])
        return z


@torch.no_grad()
def cem_plan(dp: Deploy, z0: torch.Tensor, z_star: torch.Tensor, H: int, rng: torch.Generator):
    r"""Plan H chunks toward z_star; returns best action plan (H, 10)."""
    mu = torch.zeros(H, CHUNK * 2, dtype=torch.float64)
    sig = torch.full_like(mu, 0.5)
    for _ in range(ITERS):
        cand = (mu + sig * torch.randn(K, *mu.shape, dtype=torch.float64, generator=rng)).clamp_(-1, 1)
        zf = dp.rollout(z0.expand(K, -1), cand)
        cost = (zf - z_star).norm(dim=-1)
        idx = cost.argsort()[:ELITE]
        mu, sig = cand[idx].mean(0), cand[idx].std(0) + 1e-3
    return mu


def execute(env, state7: np.ndarray, plan: torch.Tensor) -> np.ndarray:
    env.reset(options={"goal_state": FIXED_GOAL, "state": state7})
    for h in range(plan.shape[0]):
        for c in range(CHUNK):
            env.step(plan[h, 2 * c : 2 * c + 2].float().numpy())
    return env.render()


def main() -> int:
    t0 = time.time()
    art: dict = {"seed": SEED, "n_ep": N_EP, "H_grid": list(H_GRID), "eps_mult": list(EPS_MULT),
                 "nu_grid": list(NU_GRID), "cem": {"K": K, "iters": ITERS, "elite": ELITE},
                 "units": "H in f-chunks; eps and nu in delta_f multiples per base"}

    print(f"[1/4] fixed-goal collection ({N_EP} eps, seed {SEED}) ...")
    data = collect_fixed_goal(N_EP, SEED)
    fb_idx = np.arange(0, EP_LEN + 1, CHUNK)            # boundary frame indices
    n_ch = len(fb_idx) - 1

    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    env = make_env(None)
    g_cpu = torch.Generator().manual_seed(SEED)

    for name, builder in (("eq", build_eq), ("plain_match", lambda: build_plain(w_match))):
        print(f"[2/4] {name}: encode boundaries + per-window planning ...")
        dp = Deploy(name, builder)
        zb = dp.encode(data["frames"][:, fb_idx])        # (E, n_ch+1, D)
        acts = torch.from_numpy(data["actions"]).double()
        ach = acts[:, : n_ch * CHUNK].reshape(N_EP, n_ch, CHUNK * 2)

        # same-windows measured model curve (crossover reference) + delta on these windows
        deltas = (dp.rollout(zb[:, :-1].reshape(-1, zb.shape[-1]),
                             ach.reshape(-1, 1, CHUNK * 2)) - zb[:, 1:].reshape(-1, zb.shape[-1])
                  ).norm(dim=-1)
        delta_f = float(deltas.median())
        meas_med = []
        for H in H_GRID:
            errs = []
            for t0_ in range(0, n_ch - H + 1):
                zf = dp.rollout(zb[:, t0_], ach[:, t0_ : t0_ + H])
                errs.append((zf - zb[:, t0_ + H]).norm(dim=-1))
            meas_med.append(float(torch.cat(errs).median()))

        # windows for planning: spread over episodes/starts
        rng_np = np.random.default_rng(SEED)
        wins = [(int(e), int(t)) for e, t in zip(
            rng_np.integers(0, N_EP, N_WIN), rng_np.integers(0, n_ch - max(H_GRID), N_WIN))]

        reach = {}
        for H in H_GRID:
            d_reach = []
            for (e, t) in wins:
                plan = cem_plan(dp, zb[e, t].unsqueeze(0), zb[e, t + H].unsqueeze(0), H, g_cpu)
                fr = execute(env, data["states"][e, fb_idx[t]], plan)
                z_r = dp.encode(fr[None])[0]
                d_reach.append(float((z_r - zb[e, t + H]).norm()))
            reach[H] = d_reach
            print(f"      H={H}: reach median {np.median(d_reach):.2f} (delta_f {delta_f:.2f})")

        # theta* elicitation at H=3
        H3 = 3
        theta = {}
        for nu in NU_GRID:
            ok = 0
            for (e, t) in wins:
                u = torch.randn(zb.shape[-1], dtype=torch.float64, generator=g_cpu)
                z_noisy = zb[e, t + H3] + nu * delta_f * u / u.norm()
                plan = cem_plan(dp, zb[e, t].unsqueeze(0), z_noisy.unsqueeze(0), H3, g_cpu)
                fr = execute(env, data["states"][e, fb_idx[t]], plan)
                z_r = dp.encode(fr[None])[0]
                ok += int(float((z_r - zb[e, t + H3]).norm()) <= 4 * delta_f)
            theta[nu] = ok / len(wins)
            print(f"      theta* sweep nu={nu}: reach-rate {theta[nu]:.2f}")

        art[name] = {
            "delta_f_windows": delta_f,
            "measured_median_curve": meas_med,
            "reach_median": {H: float(np.median(v)) for H, v in reach.items()},
            "H_plan_max": {em: max([0] + [H for H in H_GRID
                           if np.mean(np.array(reach[H]) <= em * delta_f) >= 0.5])
                           for em in EPS_MULT},
            "H_model_max": {em: max([0] + [H for H, m in zip(H_GRID, meas_med)
                           if m <= em * delta_f]) for em in EPS_MULT},
            "theta_star_sweep": theta,
            "theta_star": max([0] + [nu for nu in NU_GRID if theta[nu] >= 0.5]),
        }
        print(f"      H_plan_max {art[name]['H_plan_max']}  H_model_max {art[name]['H_model_max']}"
              f"  theta* {art[name]['theta_star']}")

    env.close()
    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"[4/4] wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
