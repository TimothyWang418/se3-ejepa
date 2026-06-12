r"""Step 100 — S2(leg-exchange)-equivariant world model ON walker-walk (the bridge experiment).

Spec (frozen before any run): `docs/specs/2026-06-12-step100-walker-s2-equivariant-seed.md`.

Part 1 (this file, MODE=collect|gsym): the group action, the data collector (official TD-MPC2
walker-walk-1 policy), and the pre-registered **G-SYM probe** — the measured leg-exchange defect of the
TRUE environment's transition map on collected states:

    $\mathrm{defect}(s,a) = \lVert T(\rho s,\rho a) - \rho T(s,a)\rVert_2 / \lVert T(s,a)\rVert_2$,

estimated by resetting the env to the swapped physics state (qpos/qvel with leg blocks exchanged).
If the walker XML is exactly leg-symmetric this is 0 to integrator noise; the measured value is
$\epsilon_{\mathrm{world}}$ and governs the experiment per E3's approximate-symmetry law.

The S2 action (dm_control walker, planar):
  obs (24) = orientations (14 = 7 bodies x (cos,sin): torso, R thigh/leg/foot, L thigh/leg/foot),
             height (1), velocity (9 = rootx, rootz, rooty, R hip/knee/ankle, L hip/knee/ankle)
  rho_obs: swap orientation blocks [2:8] <-> [8:14]; swap velocity blocks [18:21] <-> [21:24]
           (i.e. velocity local idx 3:6 <-> 6:9); torso/height/root untouched.
  act (6) = (R hip, knee, ankle, L hip, knee, ankle); rho_act: swap [0:3] <-> [3:6].
  qpos (9) = (rootz?, rootx, rooty, R hip/knee/ankle, L hip/knee/ankle) — swap [3:6] <-> [6:9];
  qvel (9) likewise.

Run:  MODE=gsym    .venv/bin/python experiments/step100_walker_s2.py     (probe, ~2 min)
      MODE=collect .venv/bin/python experiments/step100_walker_s2.py     (60 episodes -> npz)
Writes: papers/figures/step100_gsym_probe.json | data/step100_walker_transitions.npz
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89  # noqa: E402  (policy slices + _flat_obs)

ROOT = Path(__file__).resolve().parent.parent
MODE = os.environ.get("MODE", "gsym")
N_EP = int(os.environ.get("STEP100_EPISODES", "60"))
SEED0 = int(os.environ.get("STEP100_ENVSEED", "0"))

# ---------- the S2 action ----------

OBS_SWAP = list(range(24))
OBS_SWAP[2:8], OBS_SWAP[8:14] = list(range(8, 14)), list(range(2, 8))          # orientation leg blocks
OBS_SWAP[18:21], OBS_SWAP[21:24] = list(range(21, 24)), list(range(18, 21))    # velocity leg blocks
ACT_SWAP = [3, 4, 5, 0, 1, 2]
Q_SWAP = [0, 1, 2, 6, 7, 8, 3, 4, 5]                                           # qpos/qvel (9,)


def rho_obs(x: np.ndarray) -> np.ndarray:
    return x[..., OBS_SWAP]


def rho_act(a: np.ndarray) -> np.ndarray:
    return a[..., ACT_SWAP]


def rho_q(q: np.ndarray) -> np.ndarray:
    return q[..., Q_SWAP]


# ---------- env + official policy ----------

def make_env():
    from dm_control import suite
    return suite.load("walker", "walk", task_kwargs={"random": SEED0})


def load_policy():
    """Official TD-MPC2 walker-walk-1 prior pi(z(obs)) via step89's slices (deterministic tanh-mean)."""
    ck = ROOT / "models/tdmpc2/walker-walk-1.pt"
    if not ck.exists():
        import subprocess
        ck.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(ck),
                        "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/walker-walk-1.pt"],
                       check=True)
    sl = s89.load_tdmpc2_slices(torch.load(ck, map_location="cpu", weights_only=True),
                                action_dim=6, obs_dim=24)
    @torch.no_grad()
    def pi(obs_np: np.ndarray) -> np.ndarray:
        z = sl.encoder(torch.as_tensor(obs_np, dtype=torch.float64).unsqueeze(0))
        return torch.tanh(sl.pi_mean(z)).squeeze(0).numpy()
    return pi


# ---------- G-SYM probe ----------

def gsym_probe(n_states: int = 200) -> dict:
    env, pi = make_env(), load_policy()
    ts = env.reset()
    physics = env.physics
    defects, raw = [], []
    rng = np.random.default_rng(0)
    steps_done = 0
    while len(defects) < n_states:
        obs = s89._flat_obs(ts).numpy()
        a = pi(obs)
        qpos, qvel = physics.data.qpos.copy(), physics.data.qvel.copy()
        # branch 1: T(s, a)
        ts = env.step(a)
        nq, nv = physics.data.qpos.copy(), physics.data.qvel.copy()
        # branch 2: T(rho s, rho a), then pull back
        with physics.reset_context():
            physics.data.qpos[:] = rho_q(qpos)
            physics.data.qvel[:] = rho_q(qvel)
        env.step(rho_act(a))
        sq, sv = physics.data.qpos.copy(), physics.data.qvel.copy()
        num = np.linalg.norm(np.concatenate([rho_q(sq) - nq, rho_q(sv) - nv]))
        den = np.linalg.norm(np.concatenate([nq, nv])) + 1e-12
        defects.append(num / den)
        raw.append(num)
        # restore branch-1 state to continue along the nominal trajectory
        with physics.reset_context():
            physics.data.qpos[:] = nq
            physics.data.qvel[:] = nv
        steps_done += 1
        if steps_done % 250 == 249:
            ts = env.reset()
    d = np.array(defects)
    out = {"n_states": int(len(d)), "defect_median": float(np.median(d)), "defect_p90": float(np.percentile(d, 90)),
           "defect_max": float(d.max()), "abs_median": float(np.median(raw)),
           "verdict_exact": bool(np.median(d) < 1e-6),
           "gate": "G-SYM: epsilon_world = median defect; <0.05 required for the exact-symmetry reading"}
    (ROOT / "papers/figures/step100_gsym_probe.json").write_text(json.dumps(out, indent=1))
    print(f"[step100:gsym] median={out['defect_median']:.2e} p90={out['defect_p90']:.2e} "
          f"max={out['defect_max']:.2e} exact={out['verdict_exact']}", file=sys.stderr)
    return out


# ---------- collector ----------

def collect():
    env, pi = make_env(), load_policy()
    obs_l, act_l, nxt_l = [], [], []
    for ep in range(N_EP):
        ts = env.reset()
        obs = s89._flat_obs(ts).numpy()
        for t in range(1000):
            a = pi(obs)
            ts = env.step(a)
            nxt = s89._flat_obs(ts).numpy()
            obs_l.append(obs); act_l.append(a); nxt_l.append(nxt)
            obs = nxt
        print(f"[step100:collect] ep {ep+1}/{N_EP} ({len(obs_l)} transitions)", file=sys.stderr)
    out = ROOT / "data/step100_walker_transitions.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, obs=np.array(obs_l, dtype=np.float32),
                        act=np.array(act_l, dtype=np.float32), nxt=np.array(nxt_l, dtype=np.float32))
    print(f"[step100:collect] wrote {out} ({len(obs_l)} transitions)", file=sys.stderr)


if __name__ == "__main__":
    if MODE == "gsym":
        gsym_probe()
    elif MODE == "collect":
        collect()
    else:
        raise SystemExit(f"unknown MODE={MODE}")
