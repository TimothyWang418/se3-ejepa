r"""Step 75 — robust MBRL via ensemble-CEM, attacking the FetchPush task-win failure (Experiment 17 / 4b).

The 4b diagnosis (`step73 --diagnose`) was **model exploitation**: the world model is accurate on-distribution
(object-readout RMSE $<\!1$cm) but the CEM, an optimizer over the whole action disk, finds the model's
*off*-distribution blind spots — actions it wrongly predicts reach the goal — so closed-loop success is $\sim\!0$ at
every horizon, for both the equivariant and the dense stack (architecture-agnostic; nothing to do with the certificate).

The standard fix (PETS; Chua et al., 2018, arXiv:1805.12114) is an **ensemble with a disagreement penalty**: train $K$
world models (different seeds), and in the CEM cost add $\lambda\cdot\mathrm{disagreement}$, where disagreement is the
across-model variance of the predicted object position along the rollout. Where a single model has an exploitable blind
spot, the *others* disagree — so the penalty pushes the plan back onto the data-supported manifold the ensemble agrees
on. We keep every model $\mathbb{Z}_N$/$\mathrm{SO}(2)$-**equivariant** (reusing step73's `EquivariantWM` +
`EquivGoalHead`), so the orbit-flatness certificate is preserved and, if competence emerges, the seen-vs-OOD task-win
test can finally discriminate (equivariant flat vs a scene-blind baseline degrading).

Honest scope. This is a genuine bet: model exploitation is a hard MBRL failure, and an ensemble + disagreement penalty
may still not reach FetchPush competence at $1$-GPU scale from a scripted-data world model. The gate is competence-first:
if the ensemble controller does not clear the $\sim\!0.07$ give-away floor in-distribution, we report INCONCLUSIVE (no
working control to make the task-win test meaningful), exactly as for 4b — the certificate (Exp 16/17) stands regardless.

Run:
  competence (the gate):  STEP75_SMOKE=1 .venv/bin/python experiments/step75_ensemble_mbrl.py --compete --device cpu
  full task-win attempt:  .venv/bin/python experiments/step75_ensemble_mbrl.py --realenv --device cuda --seed 0   (3080)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fetchpush_symmetry as sym  # noqa: E402
from step73_fetchpush_planning import (  # noqa: E402
    EquivGoalHead, EquivPlanner, EquivariantWM, _rot_np, collect_scripted_transitions, prep_equiv,
    scripted_policy_success, train_planner,
)

SMOKE = bool(int(os.environ.get("STEP75_SMOKE", "0")))


def train_ensemble(K: int, obs, act, nxt, device: str, epochs: int):
    r"""Train $K$ equivariant planning stacks (WM + goal head) on the SAME goal-directed data, different seeds."""
    planners = []
    for k in range(K):
        p = EquivPlanner(EquivariantWM(latent_dim=128, hidden=64), EquivGoalHead(latent_dim=128))
        train_planner(p, obs, act, nxt, prep_equiv, epochs=epochs, device=device, seed=100 + k)
        p.wm.eval(); p.head.eval()
        planners.append(p)
    return planners


def encode_all(planners, obs_np: np.ndarray, device: str):
    return [p.encode(obs_np, device)[0] for p in planners]


@torch.no_grad()
def ensemble_cem(planners, z0_list, goal_xy, *, noise_rot: float = 0.0, H: int = 12, n_samples: int = 256,
                 n_iters: int = 5, n_elite: int = 32, sigma0: float = 0.4, w_run: float = 0.2, disagree: float = 2.0,
                 gen=None) -> torch.Tensor:
    r"""G-equivariant CEM whose cost is the ENSEMBLE-mean goal distance + ``disagree`` * across-model disagreement.
    Each model rolls out its own latent under the shared candidate actions; the disagreement penalty (variance of the
    $K$ predicted object positions) blocks exploitation of any single model's off-distribution blind spot."""
    device, dtype = z0_list[0].device, z0_list[0].dtype
    g = goal_xy.to(device=device, dtype=dtype)
    Rn = torch.tensor(_rot_np(noise_rot), dtype=dtype, device=device) if noise_rot else None
    mean = torch.zeros(H, 2, device=device, dtype=dtype)
    sigma = torch.full((H,), sigma0, device=device, dtype=dtype)
    z0b = [z.unsqueeze(0).expand(n_samples, -1).contiguous() for z in z0_list]
    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 2, generator=gen, dtype=dtype).to(device)
        if Rn is not None:
            eps = torch.einsum("ij,nhj->nhi", Rn, eps)
        cand = mean.unsqueeze(0) + sigma.view(1, H, 1) * eps
        cand = cand / cand.norm(dim=-1, keepdim=True).clamp_min(1.0)
        z = [zb.clone() for zb in z0b]
        cost = torch.zeros(n_samples, device=device)
        for h in range(H):
            preds = []
            for k, p in enumerate(planners):
                z[k] = p.step(z[k], cand[:, h])
                preds.append(p.readout(z[k]))                      # (n_samples, 2)
            P = torch.stack(preds, 0)                              # (K, n_samples, 2)
            d = ((P.mean(0) - g) ** 2).sum(-1)                     # ensemble-mean goal distance
            disag = P.var(0).sum(-1) if len(planners) > 1 else torch.zeros_like(d)
            step_cost = d + disagree * disag
            cost = cost + (w_run * step_cost if h < H - 1 else step_cost)
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        dev = elite - mean.unsqueeze(0)
        sigma = (dev.pow(2).sum(-1).mean(0) / 2).sqrt().clamp_min(1e-3)
    return mean


@torch.no_grad()
def ensemble_success(planners, env_id, device, *, thetas, n_episodes, seed, T_max=50, replan_every=10, cem_kw=None):
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    cem_kw = cem_kw or {}
    env = gym.make(env_id)
    succ = {}
    for th in thetas:
        hits = 0
        for ep in range(n_episodes):
            o, info = env.reset(seed=seed + 1000 * ep)
            done_success = False
            t = 0
            while t < T_max:
                obs = np.asarray(o["observation"]); goal3 = np.asarray(o["desired_goal"])
                obs_r = sym.rotate_obs(obs, th); goal_r = sym.rotate_goal(goal3, th)[:2]
                z0_list = encode_all(planners, obs_r[None], device)
                plan = ensemble_cem(planners, z0_list, torch.tensor(goal_r, dtype=torch.float32, device=device),
                                    noise_rot=th, gen=torch.Generator().manual_seed(5000 + ep), **cem_kw)
                for k in range(min(replan_every, plan.shape[0], T_max - t)):
                    a_xy = sym.rotate_action(np.append(plan[k].cpu().numpy(), [0.0, 0.0]), -th)
                    o, _, term, trunc, info = env.step(a_xy.astype(np.float32))
                    done_success = done_success or bool(info.get("is_success", 0.0))
                    t += 1
                    if term or trunc:
                        break
                if term or trunc:
                    break
            hits += int(done_success)
        succ[float(th)] = hits / n_episodes
        print(f"[step75]   theta={np.degrees(th):5.1f} deg  success {succ[float(th)]:.2f}", file=sys.stderr)
    env.close()
    return succ


def _data(env_id, seed):
    n_tr = 1500 if SMOKE else 12000
    return collect_scripted_transitions(env_id, n_tr, seed)


def run_compete(env_id, device, seed) -> int:
    r"""The gate: does the ensemble controller clear the ~0.07 give-away floor IN-DISTRIBUTION (theta=0)? If not,
    the task-win test is meaningless (4b's lesson) and we stop honestly."""
    K = 3 if SMOKE else 5
    epochs = 6 if SMOKE else 40
    cem_kw = dict(H=8, n_samples=128, n_iters=3, disagree=2.0) if SMOKE else dict(H=12, n_samples=256, n_iters=5, disagree=2.0)
    n_ep = 5 if SMOKE else 30
    sp = scripted_policy_success(env_id, 10 if SMOKE else 40, seed)
    print(f"[step75] scripted-policy sanity {sp:.2f}; training {K}-model equivariant ensemble on scripted data ...",
          file=sys.stderr)
    obs, act, nxt = _data(env_id, seed)
    planners = train_ensemble(K, obs, act, nxt, device, epochs)
    s = ensemble_success(planners, env_id, device, thetas=[0.0], n_episodes=n_ep, seed=seed, cem_kw=cem_kw)
    competent = bool(s[0.0] >= 0.2)
    print(f"[step75] ensemble in-dist success {s[0.0]:.2f} (K={K}, disagreement penalty). "
          f"{'COMPETENT — the task-win test can run (--realenv).' if competent else 'INCONCLUSIVE: ensemble-CEM still below the 0.2 competence floor — model exploitation not beaten at this scale; certificate (Exp 16/17) stands.'}",
          file=sys.stderr)
    return 0 if competent else 1


def run_realenv(env_id, device, seed, tag) -> int:
    import json
    K = 3 if SMOKE else 5
    epochs = 6 if SMOKE else 40
    thetas = [0.0, np.pi / 2] if SMOKE else [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    n_ep = 3 if SMOKE else 30
    cem_kw = dict(H=8, n_samples=128, n_iters=3, disagree=2.0) if SMOKE else dict(H=12, n_samples=256, n_iters=5, disagree=2.0)
    obs, act, nxt = _data(env_id, seed)
    planners = train_ensemble(K, obs, act, nxt, device, epochs)
    s = ensemble_success(planners, env_id, device, thetas=thetas, n_episodes=n_ep, seed=seed, cem_kw=cem_kw)
    drop = s[0.0] - min(s.values())
    competent = bool(s[0.0] >= 0.2)
    res = {"competent": competent, "seen": s[0.0], "ood_drop": drop, "by_theta": s, "K": K, "seed": seed,
           "smoke": SMOKE, "env": env_id}
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step75_ensemble_realenv{tag}.json").write_text(json.dumps(res, indent=2))
    verdict = ("TASK WIN (equivariant ensemble holds across orientations)" if competent and drop < 0.1 else
               ("INCONCLUSIVE: not competent in-distribution — model exploitation not beaten at 1-GPU scale"
                if not competent else "competent but check OOD spread"))
    print(f"[step75] realenv: seen {s[0.0]:.2f} OOD-drop {drop:.2f} (K={K} equivariant ensemble). {verdict}.",
          file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Step 75 — robust MBRL (ensemble-CEM) for the FetchPush task win")
    p.add_argument("--compete", action="store_true", help="the gate: in-dist competence of the ensemble controller")
    p.add_argument("--realenv", action="store_true", help="full seen-vs-OOD is_success with the ensemble controller")
    p.add_argument("--env", default="FetchPush-v4")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if a.compete:
        return run_compete(a.env, a.device, a.seed)
    if a.realenv:
        return run_realenv(a.env, a.device, a.seed, a.tag)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
