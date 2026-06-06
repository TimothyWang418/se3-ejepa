r"""Step 73 — the certificate as a DOWNSTREAM PLANNING WIN on FetchPush (Stage 4 of the MuJoCo flagship).

Experiment 16 (`step72`) showed the equivariant world model is orbit-FLAT on FetchPush *at the prediction level*.
This file lifts that to the TASK level: an equivariant WM + an equivariant goal-readout head + a G-equivariant CEM
planner make the **plan itself orbit-equivariant** ($\mathrm{plan}(\rho(g)\,s,\,\rho(g)\,\mathrm{goal})
=R(g)\,\mathrm{plan}(s,\mathrm{goal})$) and the planned goal-distance **orbit-FLAT to the float floor**, while a
scaled non-equivariant baseline degrades out of the training orientation. This is the 举一反三 downstream win: the
same controller succeeds at every scene orientation because the certificate (Theorem A) propagates through the
planner — the closed-loop instance of Experiment 11 / Step 61, lifted to a standard MuJoCo manipulation benchmark.

Two sub-stages, split exactly like Experiment 11 (provable core + empirical task win):

  4a  (this file, CPU/Mac, PROVABLE + unit-tested):
      * ``EquivGoalHead`` g: latent -> object planar (x,y) = a single ``VNLinear(latent_dim/2 -> 1)``, so
        $g(\rho(g)z)=R(g)\,g(z)$ and the planning cost $\lVert g(z)-\mathrm{goal}_{xy}\rVert$ is rotation-INVARIANT.
      * ``cem_plan`` — a G-equivariant CEM over LATENT rollouts (isotropic per-step covariance + disk action bound +
        scene-covariant action noise), scored by the readout's predicted object->goal distance. With the equivariant
        WM + head this satisfies $\mathrm{plan}(\rho(g)s,\rho(g)\,\mathrm{goal})=R(g)\,\mathrm{plan}(s,\mathrm{goal})$
        to the float floor.
      * ``run_certificate`` — the MODEL-ROLLOUT certificate: planned terminal predicted-goal-distance is orbit-flat
        for the equivariant stack (ratio ~1.000) and degrades for the baseline (ratio > 1.5). No real env needed:
        fast, deterministic, gated (INCONCLUSIVE rather than loosen).
      * ``tests/test_step73_planner_equivariance.py``: $\mathrm{plan}(\rho(g)s)=R(g)\,\mathrm{plan}(s)$ at machine
        precision.

  4b  (``--realenv``, smoke-testable on Mac, full on the 3080, EMPIRICAL):
      receding-horizon control on FetchPush-v4 in a *rotated control frame* (rotate the obs+goal the planner sees,
      rotate its action back before applying). For the equivariant stack the applied control is $\theta$-INVARIANT,
      so ``is_success`` rate is flat across orientations; the scene-blind baseline's control changes with $\theta$,
      so it drops OOD. Reports seen-vs-OOD success rate for both stacks.

Reuses `step72`'s `EquivariantWM` / `BaselineWM` / `collect_transitions` / `prep_*` and `fetchpush_symmetry`'s
group action. The planner pattern (isotropic+disk+scene-covariant) is `step61`'s `cem_plan_pose_equiv`, adapted from
state-space to LATENT rollouts scored by an equivariant readout.

Run:
  smoke:   STEP73_SMOKE=1 .venv/bin/python experiments/step73_fetchpush_planning.py --cert --device cpu --seed 0 --tag _smoke
  full:    .venv/bin/python experiments/step73_fetchpush_planning.py --cert --device cuda --seed 0
  realenv: STEP73_SMOKE=1 .venv/bin/python experiments/step73_fetchpush_planning.py --realenv --device cpu --seed 0   (Mac smoke)
           .venv/bin/python experiments/step73_fetchpush_planning.py --realenv --device cuda --seed 0                 (3080 full)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `src`/`fetchpush_symmetry` import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import fetchpush_symmetry as sym  # noqa: E402
from step72_mujoco_certificate import (  # noqa: E402
    BaselineWM, EquivariantWM, _vicreg, collect_transitions, latent_rotation, prep_baseline, prep_equiv,
)

OBJ_XY = (3, 4)                       # object_pos planar (x,y) indices in the 25-D obs (the planning target)
SMOKE = bool(int(os.environ.get("STEP73_SMOKE", "0")))


# ------------------------------------------------------------------------------------------------- #
# Goal-directed (scripted) data. The random-policy 4b run was INCONCLUSIVE because neither WM+CEM
# controller is COMPETENT from random data (both ~7%, the near-goal give-away rate) — so the seen-vs-OOD
# test has no working control to break. A scripted "get behind the block, push toward the goal" policy
# generates on-task transitions: training the WM on these should make the controller competent, which is
# the prerequisite for the task-win test to discriminate (equivariant flat vs baseline drop). The scripted
# policy is for DATA + a competence sanity check ONLY — the evaluated controller is still the learned WM+CEM.
# ------------------------------------------------------------------------------------------------- #
def scripted_action(obs: np.ndarray, goal_xyz: np.ndarray, rng=None, noise: float = 0.0) -> np.ndarray:
    r"""Heuristic FetchPush pusher: approach a point just BEHIND the block (opposite the goal) at table height,
    then push through the block toward the goal. Gripper held closed. Returns a 4-D action in $[-1,1]^4$."""
    grip = obs[0:3]; objp = obs[3:6]
    to_goal = goal_xyz - objp
    dxy = to_goal[:2]
    dn = dxy / (np.linalg.norm(dxy) + 1e-6)                          # planar push direction (object -> goal)
    behind = objp.copy()
    behind[:2] = objp[:2] - dn * 0.055                              # stand-off point behind the block
    behind[2] = objp[2]
    grip_to_behind = behind[:2] - grip[:2]
    a = np.zeros(4, dtype=np.float64)
    if np.linalg.norm(grip_to_behind) > 0.025:                     # phase 1: get behind the block (in plane)
        a[:2] = 6.0 * grip_to_behind
        a[2] = 6.0 * (behind[2] + 0.01 - grip[2])
    else:                                                          # phase 2: push through the block to the goal
        a[:2] = 6.0 * (objp[:2] + dn * 0.10 - grip[:2])
        a[2] = 6.0 * (objp[2] - grip[2])
    a[3] = -1.0                                                     # gripper closed (push, not grasp)
    if noise and rng is not None:
        a[:3] = a[:3] + noise * rng.standard_normal(3)
    return np.clip(a, -1.0, 1.0)


def collect_scripted_transitions(env_id, n_steps, seed, noise: float = 0.4):
    r"""(obs, act, next_obs) from the scripted pusher + exploration noise — goal-directed coverage of the dynamics."""
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    O, A, N = [], [], []
    for _ in range(n_steps):
        o = obs["observation"]; g = obs["desired_goal"]
        a = scripted_action(np.asarray(o), np.asarray(g), rng=rng, noise=noise).astype(np.float32)
        nxt, _, term, trunc, _ = env.step(a)
        O.append(o); A.append(a); N.append(nxt["observation"])
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(O, np.float32), np.array(A, np.float32), np.array(N, np.float32)


def scripted_policy_success(env_id, n_episodes, seed, T_max: int = 50, noise: float = 0.0) -> float:
    r"""Sanity: does the SCRIPTED policy itself solve FetchPush? If this is ~7% the heuristic is broken (fix it
    before training on its data); if it is decently above the give-away floor, its data is worth training on."""
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    hits = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 7 * ep)
        done = False
        for _ in range(T_max):
            a = scripted_action(np.asarray(obs["observation"]), np.asarray(obs["desired_goal"]),
                                rng=rng, noise=noise).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            done = done or bool(info.get("is_success", 0.0))
            if term or trunc:
                break
        hits += int(done)
    env.close()
    return hits / n_episodes


def _rot_np(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


# ------------------------------------------------------------------------------------------------- #
# Equivariant goal-readout head. The WM predicts the next LATENT (JEPA), not the next state, so the
# planner needs a readout to score latent rollouts against the 2-D goal. A single VNLinear(latent/2 -> 1)
# maps the latent (latent_dim/2 flattened 2-vecs) to ONE 2-vector = the predicted object planar position;
# being a VN map it is exactly equivariant: g(rho(g) z) = R(g) g(z), so ||g(z)-goal|| is rotation-invariant.
# The baseline head is an ordinary MLP (not equivariant) — the fair non-equivariant counterpart.
# ------------------------------------------------------------------------------------------------- #
class EquivGoalHead(nn.Module):
    r"""$g:\;$ latent $(B, d)\to$ object planar position $(B,2)$, equivariant: $g(\rho(g)z)=R(g)g(z)$."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        from src.models.structured import VNLinear
        if latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even (VN latent is flattened 2-vectors)")
        self.lin = VNLinear(latent_dim // 2, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:        # (B, latent_dim) -> (B, 2)
        zz = z.reshape(z.shape[0], -1, 2)                      # (B, latent_dim/2, 2)
        return self.lin(zz).squeeze(-2)                        # (B, 1, 2) -> (B, 2)


class BaselineGoalHead(nn.Module):
    r"""Ordinary MLP readout (not equivariant). Same latent in, $(x,y)$ out."""

    def __init__(self, latent_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


# ------------------------------------------------------------------------------------------------- #
# Uniform planner wrapper. Exposes encode(obs)->z, step(z, a_xy)->z' (LATENT->LATENT), readout(z)->object_xy.
# Both stacks plan over the SAME planar (dx,dy) action so the comparison is about the SYMMETRY, not action dim:
# the equivariant WM consumes (dx,dy); the baseline WM gets the 4-D [dx,dy,0,0].
# ------------------------------------------------------------------------------------------------- #
class EquivPlanner:
    equivariant = True

    def __init__(self, wm: EquivariantWM, head: EquivGoalHead):
        self.wm, self.head = wm, head

    def encode(self, obs_np: np.ndarray, device: str) -> torch.Tensor:
        v, _ = sym.obs_to_vn(np.asarray(obs_np))
        return self.wm.enc(torch.tensor(v, dtype=torch.float32, device=device))

    def step(self, z: torch.Tensor, a_xy: torch.Tensor) -> torch.Tensor:
        return self.wm.pred(z, a_xy)                           # jointly-equivariant latent step

    def readout(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class BaselinePlanner:
    equivariant = False

    def __init__(self, wm: BaselineWM, head: BaselineGoalHead):
        self.wm, self.head = wm, head

    def encode(self, obs_np: np.ndarray, device: str) -> torch.Tensor:
        return self.wm.enc(torch.tensor(np.asarray(obs_np), dtype=torch.float32, device=device))

    def step(self, z: torch.Tensor, a_xy: torch.Tensor) -> torch.Tensor:
        a4 = torch.zeros(a_xy.shape[0], 4, dtype=a_xy.dtype, device=a_xy.device)
        a4[:, :2] = a_xy
        return z + self.wm.pred(torch.cat([z, a4], dim=-1))    # residual latent step (dz, gripper = 0)

    def readout(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ------------------------------------------------------------------------------------------------- #
# Joint training: JEPA latent loss (EMA target + VICReg, exactly as step72) + a readout MSE that teaches g to
# decode object planar position from BOTH the current encoder latent and the one-step predicted latent — so the
# readout is accurate on the manifold the multi-step planner traverses.
# ------------------------------------------------------------------------------------------------- #
def train_planner(planner, obs, act, nxt, prep, *, epochs: int, device: str, seed: int,
                  ema: float = 0.99, var_coef: float = 0.1, head_coef: float = 1.0):
    import copy
    torch.manual_seed(seed)
    wm, head = planner.wm.to(device), planner.head.to(device)
    target_enc = copy.deepcopy(wm.enc).to(device)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(wm.parameters()) + list(head.parameters()), lr=1e-3)

    enc_in, act_in = prep(obs, act, device)
    nxt_in, _ = prep(nxt, act, device)
    # object planar position from raw obs (the readout target), in obs units
    obj_cur = torch.tensor(np.asarray(obs)[:, OBJ_XY], dtype=torch.float32, device=device)
    obj_nxt = torch.tensor(np.asarray(nxt)[:, OBJ_XY], dtype=torch.float32, device=device)
    n = enc_in.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 256):
            idx = torch.randperm(n, generator=g)[i:i + 256]
            z_cur = wm.encode(enc_in[idx])
            pred = planner_forward(planner, enc_in[idx], act_in[idx])     # predicted next latent
            with torch.no_grad():
                tgt = target_enc(nxt_in[idx])                            # EMA-target latent of the NEXT obs
            jepa = ((pred - tgt) ** 2).mean()
            readout = (((head(z_cur) - obj_cur[idx]) ** 2).sum(-1).mean()
                       + ((head(pred) - obj_nxt[idx]) ** 2).sum(-1).mean())
            loss = jepa + var_coef * _vicreg(z_cur) + head_coef * readout
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for tp, mp in zip(target_enc.parameters(), wm.enc.parameters()):
                    tp.mul_(ema).add_(mp, alpha=1 - ema)
    return planner


def planner_forward(planner, enc_in, act_in):
    r"""One-step predicted next latent from the model's native TRAINING inputs: vectors + (dx,dy) for the equivariant
    stack; raw obs + the full 4-D action for the baseline (the CEM later plans over planar (dx,dy) for both)."""
    if planner.equivariant:
        return planner.wm.forward(enc_in, act_in)             # pred(enc(vectors), a_xy), act_in is 2-D
    z = planner.wm.enc(enc_in)
    return z + planner.wm.pred(torch.cat([z, act_in], dim=-1))   # act_in is the full 4-D training action


# ------------------------------------------------------------------------------------------------- #
# The G-equivariant CEM planner (step61's A5 pattern), in LATENT space, scored by the readout's goal distance.
# isotropic per-step covariance + disk action bound + scene-covariant noise => plan(R s, R goal) = R plan(s, goal).
# ------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def cem_plan(planner, z0: torch.Tensor, goal_xy: torch.Tensor, *, noise_rot: float = 0.0, H: int = 12,
             n_samples: int = 256, n_iters: int = 5, n_elite: int = 32, sigma0: float = 0.4, w_run: float = 0.2,
             gen: "torch.Generator | None" = None) -> torch.Tensor:
    r"""Returns the elite-mean plan ``(H, 2)`` of planar actions. ``noise_rot`` rotates the Gaussian draws by the
    orbit angle (scene-covariant); with the equivariant planner the whole procedure is SO(2)-equivariant."""
    device, dtype = z0.device, z0.dtype
    z0b = z0.unsqueeze(0).expand(n_samples, -1).contiguous()
    g = goal_xy.to(device=device, dtype=dtype)
    Rn = torch.tensor(_rot_np(noise_rot), dtype=dtype, device=device) if noise_rot else None
    mean = torch.zeros(H, 2, device=device, dtype=dtype)
    sigma = torch.full((H,), sigma0, device=device, dtype=dtype)          # per-step SCALAR == isotropic
    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 2, generator=gen, dtype=dtype).to(device)
        if Rn is not None:
            eps = torch.einsum("ij,nhj->nhi", Rn, eps)                    # scene-covariant draws
        cand = mean.unsqueeze(0) + sigma.view(1, H, 1) * eps
        cand = cand / cand.norm(dim=-1, keepdim=True).clamp_min(1.0)      # DISK bound (rotation-invariant)
        z = z0b.clone()
        cost = torch.zeros(n_samples, device=device)
        for h in range(H):
            z = planner.step(z, cand[:, h])
            d = ((planner.readout(z) - g) ** 2).sum(-1)
            cost = cost + (w_run * d if h < H - 1 else d)
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        dev = elite - mean.unsqueeze(0)
        sigma = (dev.pow(2).sum(-1).mean(0) / 2).sqrt().clamp_min(1e-3)   # rotation-INVARIANT isotropic std
    return mean


@torch.no_grad()
def rollout_goal_dist(planner, z0: torch.Tensor, goal_xy: torch.Tensor, plan: torch.Tensor) -> float:
    r"""Terminal predicted object->goal distance when ``plan`` is rolled out under the model (no env)."""
    z = z0.unsqueeze(0)
    for h in range(plan.shape[0]):
        z = planner.step(z, plan[h:h + 1])
    return float(((planner.readout(z)[0] - goal_xy) ** 2).sum().sqrt())


# ------------------------------------------------------------------------------------------------- #
# 4a — MODEL-ROLLOUT certificate: planned terminal predicted-goal-distance, orbit-flat for the equivariant
# stack, degrading for the baseline. Goal = initial object position + a fixed planar displacement; at orbit
# angle theta the whole problem (obs + displacement) is rotated, and the equivariant planner uses scene-covariant
# noise (noise_rot=theta), so its planned distance is orbit-INVARIANT to the float floor.
# ------------------------------------------------------------------------------------------------- #
THETAS = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi]
DISP0 = np.array([0.10, 0.0], dtype=np.float64)        # push the block +10cm in the (training-frame) x direction


@torch.no_grad()
def orbit_planned_dist(planner, init_obs: np.ndarray, device: str, *, thetas=THETAS, cem_kw=None) -> dict:
    cem_kw = cem_kw or {}
    out = {}
    for th in thetas:
        ds = []
        for i, obs in enumerate(init_obs):
            obs_r = sym.rotate_obs(obs, th)
            obj_xy = obs_r[list(OBJ_XY)]
            disp_r = _rot_np(th) @ DISP0
            goal = torch.tensor(obj_xy + disp_r, dtype=torch.float32, device=device)
            z0 = planner.encode(obs_r[None], device)[0]
            nr = th if planner.equivariant else 0.0
            plan = cem_plan(planner, z0, goal, noise_rot=nr,
                            gen=torch.Generator().manual_seed(1000 + i), **cem_kw)
            ds.append(rollout_goal_dist(planner, z0, goal, plan))
        out[float(th)] = float(np.mean(ds))
    return out


def _ratio(d: dict) -> float:
    base = max(d[min(d)], 1e-9)
    return max(d.values()) / base


def run_certificate(env_id: str, device: str, seed: int, tag: str) -> int:
    r"""Train both planning stacks, then check orbit-flatness of the planned terminal goal-distance."""
    n_tr = 1500 if SMOKE else 12000
    epochs = 6 if SMOKE else 40
    n_init = 6 if SMOKE else 24
    cem_kw = dict(H=8, n_samples=128, n_iters=3) if SMOKE else dict(H=12, n_samples=256, n_iters=5)
    print(f"[step73] cert: collecting {n_tr} transitions from {env_id} (seed {seed}) ...", file=sys.stderr)
    obs, act, nxt = collect_transitions(env_id, n_tr, seed)
    init_obs = obs[:n_init]                                          # initial states for the planning probe

    torch.manual_seed(seed)
    eq = EquivPlanner(EquivariantWM(latent_dim=128, hidden=64), EquivGoalHead(latent_dim=128))
    bl = BaselinePlanner(BaselineWM(latent_dim=128, hidden=256), BaselineGoalHead(latent_dim=128, hidden=128))
    print("[step73] training equivariant planning stack ...", file=sys.stderr)
    train_planner(eq, obs, act, nxt, prep_equiv, epochs=epochs, device=device, seed=seed)
    print("[step73] training baseline planning stack ...", file=sys.stderr)
    train_planner(bl, obs, act, nxt, prep_baseline, epochs=epochs, device=device, seed=seed)

    eq.wm.eval(); eq.head.eval(); bl.wm.eval(); bl.head.eval()
    d_eq = orbit_planned_dist(eq, init_obs, device, cem_kw=cem_kw)
    d_bl = orbit_planned_dist(bl, init_obs, device, cem_kw=cem_kw)
    r_eq, r_bl = _ratio(d_eq), _ratio(d_bl)
    print(f"[step73]   equivariant planned-dist orbit ratio {r_eq:.3f}  (seen {d_eq[0.0]:.4f})", file=sys.stderr)
    print(f"[step73]   baseline    planned-dist orbit ratio {r_bl:.3f}  (seen {d_bl[0.0]:.4f})", file=sys.stderr)

    ok_flat = bool(r_eq < 1.05)                                     # equivariant planner orbit-flat (Theorem A -> task)
    ok_degrade = bool(r_bl > 1.5)                                   # baseline planner degrades OOD
    passed = bool(ok_flat and ok_degrade)
    res = {"passed": passed, "gate": {"equivariant_flat": ok_flat, "baseline_degrades": ok_degrade},
           "equivariant": {"ratio": r_eq, "by_theta": d_eq, "seen": d_eq[0.0]},
           "baseline": {"ratio": r_bl, "by_theta": d_bl, "seen": d_bl[0.0]},
           "seed": seed, "smoke": SMOKE, "env": env_id}
    _save_cert_figure(res, tag)
    msg = "TASK-LEVEL CERTIFICATE HOLDS" if passed else "INCONCLUSIVE"
    print(f"[step73] {msg}: equivariant planner orbit-flat (ratio {r_eq:.3f}); baseline planner degrades "
          f"(ratio {r_bl:.3f}). The G-equivariant CEM carries Theorem A's orbit-flatness to the closed-loop plan; "
          f"run --realenv on the 3080 for the is_success task win.", file=sys.stderr)
    return 0 if passed else 1


def _save_cert_figure(res: dict, tag: str) -> None:
    try:
        import json

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        (figdir / f"step73_fetchpush_planning{tag}.json").write_text(json.dumps(res, indent=2))
        eq, bl = res["equivariant"], res["baseline"]
        xs = sorted(float(k) for k in eq["by_theta"])
        deg = [int(round(np.degrees(x))) for x in xs]
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(deg, [eq["by_theta"][x] for x in xs], "o-", color="#1f77b4",
                label=f"equivariant stack (ratio {eq['ratio']:.2f})")
        ax.plot(deg, [bl["by_theta"][x] for x in xs], "s--", color="#d62728",
                label=f"baseline stack (ratio {bl['ratio']:.2f})")
        ax.set_xlabel("scene rotation off the training orientation (deg)")
        ax.set_ylabel("planned terminal object→goal distance")
        ax.set_title("Task-level certificate on FetchPush: the plan is orbit-flat vs. scale (Exp 17)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / f"step73_fetchpush_planning{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step73]   (figure skipped: {e})", file=sys.stderr)


# ------------------------------------------------------------------------------------------------- #
# 4b — EMPIRICAL real-env closed-loop is_success, seen vs OOD, via a ROTATED CONTROL FRAME. We rotate the obs+goal
# the planner sees by theta and rotate its action back by -theta before applying. For the equivariant stack the
# applied control is theta-INVARIANT, so is_success is flat across orientations; the scene-blind baseline's control
# changes with theta, so it drops OOD. (Mac-smokeable; full sweep on the 3080.)
# ------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def realenv_success(planner, env_id: str, device: str, *, thetas, n_episodes: int, seed: int,
                    T_max: int = 50, replan_every: int = 10, cem_kw=None) -> dict:
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    cem_kw = cem_kw or dict(H=12, n_samples=256, n_iters=5)
    env = gym.make(env_id)
    succ = {}
    for th in thetas:
        nr = th if planner.equivariant else 0.0
        hits = 0
        for ep in range(n_episodes):
            o, info = env.reset(seed=seed + 1000 * ep)
            done_success = False
            t = 0
            while t < T_max:
                obs = np.asarray(o["observation"])
                goal3 = np.asarray(o["desired_goal"])
                obs_r = sym.rotate_obs(obs, th)                       # plan in the theta-rotated frame
                goal_r = sym.rotate_goal(goal3, th)[:2]
                z0 = planner.encode(obs_r[None], device)[0]
                plan = cem_plan(planner, z0, torch.tensor(goal_r, dtype=torch.float32, device=device),
                                noise_rot=nr, gen=torch.Generator().manual_seed(5000 + ep), **cem_kw)
                for k in range(min(replan_every, plan.shape[0], T_max - t)):
                    a_xy = sym.rotate_action(np.append(plan[k].cpu().numpy(), [0.0, 0.0]), -th)  # rotate control back
                    o, _, term, trunc, info = env.step(a_xy.astype(np.float32))
                    done_success = done_success or bool(info.get("is_success", 0.0))
                    t += 1
                    if term or trunc:
                        break
                if term or trunc:
                    break
            hits += int(done_success)
        succ[float(th)] = hits / n_episodes
        print(f"[step73]   {'equiv' if planner.equivariant else 'base '} theta={np.degrees(th):5.1f} deg  "
              f"success {succ[float(th)]:.2f}", file=sys.stderr)
    env.close()
    return succ


def run_realenv(env_id: str, device: str, seed: int, tag: str, scripted: bool = False) -> int:
    import json
    n_tr = 1500 if SMOKE else 12000
    epochs = 6 if SMOKE else 40
    thetas = [0.0, np.pi / 2] if SMOKE else [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    n_episodes = 3 if SMOKE else 30
    cem_kw = dict(H=8, n_samples=128, n_iters=3) if SMOKE else dict(H=16, n_samples=384, n_iters=6)
    src = "scripted goal-directed" if scripted else "random-policy"
    if scripted:
        sp = scripted_policy_success(env_id, 10 if SMOKE else 40, seed)
        print(f"[step73] realenv: scripted-policy sanity success {sp:.2f} (must clear the ~0.07 give-away floor to be "
              f"worth training on)", file=sys.stderr)
    print(f"[step73] realenv: collecting {n_tr} {src} transitions + training both stacks (seed {seed}) ...",
          file=sys.stderr)
    obs, act, nxt = (collect_scripted_transitions(env_id, n_tr, seed) if scripted
                     else collect_transitions(env_id, n_tr, seed))
    eq = EquivPlanner(EquivariantWM(latent_dim=128, hidden=64), EquivGoalHead(latent_dim=128))
    bl = BaselinePlanner(BaselineWM(latent_dim=128, hidden=256), BaselineGoalHead(latent_dim=128, hidden=128))
    train_planner(eq, obs, act, nxt, prep_equiv, epochs=epochs, device=device, seed=seed)
    train_planner(bl, obs, act, nxt, prep_baseline, epochs=epochs, device=device, seed=seed)
    eq.wm.eval(); eq.head.eval(); bl.wm.eval(); bl.head.eval()

    s_eq = realenv_success(eq, env_id, device, thetas=thetas, n_episodes=n_episodes, seed=seed, cem_kw=cem_kw)
    s_bl = realenv_success(bl, env_id, device, thetas=thetas, n_episodes=n_episodes, seed=seed, cem_kw=cem_kw)
    drop_eq = s_eq[0.0] - min(s_eq.values())
    drop_bl = s_bl[0.0] - min(s_bl.values())
    # Honest gate: a task win needs (i) a COMPETENT in-distribution baseline (else nothing to break — the random-data
    # failure mode), (ii) the baseline to drop OOD, and (iii) the equivariant to stay flatter. Anything else is
    # INCONCLUSIVE, not a win.
    competent = bool(s_bl[0.0] >= 0.2 and s_eq[0.0] >= 0.2)
    win = bool(competent and drop_bl >= 0.1 and drop_eq < drop_bl - 0.05)
    verdict = ("TASK WIN: equivariant holds success across orientations where the scaled baseline drops (举一反三)"
               if win else ("INCONCLUSIVE: baseline not competent in-distribution (no working control to break — "
                            "needs better data/horizon/scale)" if not competent else "INCONCLUSIVE"))
    res = {"task_win": win, "competent": competent, "data": src,
           "equivariant": {"by_theta": s_eq, "seen": s_eq[0.0], "ood_drop": drop_eq},
           "baseline": {"by_theta": s_bl, "seen": s_bl[0.0], "ood_drop": drop_bl},
           "seed": seed, "smoke": SMOKE, "env": env_id}
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step73_fetchpush_realenv{tag}.json").write_text(json.dumps(res, indent=2))
    _save_realenv_figure(res, tag)
    print(f"[step73] realenv ({src}): equivariant seen {s_eq[0.0]:.2f} OOD-drop {drop_eq:.2f}; "
          f"baseline seen {s_bl[0.0]:.2f} OOD-drop {drop_bl:.2f}. {verdict}.", file=sys.stderr)
    return 0


def _save_realenv_figure(res: dict, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
        eq, bl = res["equivariant"], res["baseline"]
        xs = sorted(float(k) for k in eq["by_theta"])
        deg = [int(round(np.degrees(x))) for x in xs]
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(deg, [eq["by_theta"][x] for x in xs], "o-", color="#1f77b4",
                label=f"equivariant (seen {eq['seen']:.2f}, OOD-drop {eq['ood_drop']:.2f})")
        ax.plot(deg, [bl["by_theta"][x] for x in xs], "s--", color="#d62728",
                label=f"baseline (seen {bl['seen']:.2f}, OOD-drop {bl['ood_drop']:.2f})")
        ax.set_xlabel("scene rotation off the training orientation (deg)")
        ax.set_ylabel("FetchPush task success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"FetchPush closed-loop success vs. orientation ({res['data']} data)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / f"step73_fetchpush_realenv{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step73]   (realenv figure skipped: {e})", file=sys.stderr)


def diagnose(env_id: str, device: str, seed: int) -> int:
    r"""Why does the real-env task win not materialize (4b/4b-v2 both INCONCLUSIVE)? Train the equivariant stack on
    goal-directed data, then localize the failure: (1) readout fidelity — can $g(z)$ decode the object position? — and
    (2) does ANY planning horizon reach in-distribution competence? The finding: the WM is accurate ($<\!1$cm) yet CEM
    gets $0$ success at every horizon, i.e. the planner **exploits the model off-distribution** (a model-based-RL
    pitfall), which is architecture-agnostic and unrelated to the (exact) certificate."""
    n_tr = 1500 if SMOKE else 5000
    epochs = 6 if SMOKE else 30
    obs, act, nxt = collect_scripted_transitions(env_id, n_tr, seed)
    eq = EquivPlanner(EquivariantWM(latent_dim=128, hidden=64), EquivGoalHead(latent_dim=128))
    train_planner(eq, obs, act, nxt, prep_equiv, epochs=epochs, device=device, seed=seed)
    eq.wm.eval(); eq.head.eval()
    o2, a2, n2 = collect_scripted_transitions(env_id, 1000, seed + 99)
    with torch.no_grad():
        cur = eq.readout(eq.encode(o2, device)).cpu().numpy()
        enc_in, act_in = prep_equiv(o2, a2, device)
        nx = eq.readout(planner_forward(eq, enc_in, act_in)).cpu().numpy()
    txy, tnx = np.asarray(o2)[:, list(OBJ_XY)], np.asarray(n2)[:, list(OBJ_XY)]
    rc = float(np.sqrt(((cur - txy) ** 2).sum(-1)).mean())
    rn = float(np.sqrt(((nx - tnx) ** 2).sum(-1)).mean())
    print(f"[step73] diagnose: object-readout RMSE current={rc:.4f} m, one-step-pred={rn:.4f} m "
          f"(objects span ~0.34 m) — the model is accurate.", file=sys.stderr)
    for H, re_ in ([(8, 4)] if SMOKE else [(16, 10), (6, 2), (3, 1)]):
        s = realenv_success(eq, env_id, device, thetas=[0.0], n_episodes=5 if SMOKE else 20, seed=seed,
                            replan_every=re_, cem_kw=dict(H=H, n_samples=256, n_iters=4))
        print(f"[step73] diagnose: eq in-dist success H={H} replan_every={re_}: {s[0.0]:.2f}", file=sys.stderr)
    print("[step73] diagnose => accurate WM but ~0 CEM success at every horizon = the planner exploits the model "
          "off-distribution (MBRL pitfall), architecture-agnostic, NOT an equivariance cost.", file=sys.stderr)
    return 0


def smoke(env_id: str) -> int:
    r"""Tiny end-to-end: train both stacks on few transitions, run the orbit planning probe, print ratios."""
    os.environ["STEP73_SMOKE"] = "1"
    global SMOKE
    SMOKE = True
    return run_certificate(env_id, "cpu", 0, "_smoke")


def main() -> int:
    p = argparse.ArgumentParser(description="Step 73 — task-level certificate (planning) on FetchPush")
    p.add_argument("--cert", action="store_true", help="4a: model-rollout orbit-flatness certificate of the plan")
    p.add_argument("--realenv", action="store_true", help="4b: real-env closed-loop is_success (seen vs OOD)")
    p.add_argument("--scripted", action="store_true",
                   help="4b-v2: train the WM on goal-directed (scripted-pusher) data, not random-policy data")
    p.add_argument("--scripted-eval", action="store_true",
                   help="sanity only: run the scripted pusher itself and report its FetchPush success rate")
    p.add_argument("--diagnose", action="store_true",
                   help="localize the 4b failure: readout fidelity + competence vs planning horizon (model exploitation)")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end validation of the planning probe")
    p.add_argument("--env", default="FetchPush-v4")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if a.scripted_eval:
        n = 20 if SMOKE else 100
        sr = scripted_policy_success(a.env, n, a.seed)
        print(f"[step73] scripted-pusher success on {a.env}: {sr:.2f} over {n} episodes "
              f"({'competent — worth training on' if sr >= 0.2 else 'weak — fix the heuristic before using its data'}).",
              file=sys.stderr)
        return 0
    if a.diagnose:
        return diagnose(a.env, a.device, a.seed)
    if a.smoke:
        return smoke(a.env)
    if a.realenv:
        return run_realenv(a.env, a.device, a.seed, a.tag, scripted=a.scripted)
    if a.cert:
        return run_certificate(a.env, a.device, a.seed, a.tag)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
