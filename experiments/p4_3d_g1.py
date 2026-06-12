r"""3D G1 — conversion validation (G1a) + data-only timescales (G1.2). Box CPU, no renderer.

Protocol spec: docs/specs/2026-06-11-p4-3d-protocol-v1.md §5. Constants from the G0-banked
controller config (pos ±0.1 m, rot ±0.1 rad, normalized, delta, root frame).

G1a (controller-faithful local conversion + open-loop validation, 20 demos):
  reference tcp poses via state-replay (obs_mode="state" — no Vulkan path at all);
  $a_t = [\mathrm{clip}(\Delta p_t / 0.1), \mathrm{clip}(\omega_t / 0.1), g_t]$ with
  $\omega_t = \log(q_{t+1} q_t^{-1})$ (axis-angle, root frame), gripper from qpos mapping;
  open-loop execution under pd_ee_delta_pose from the same initial state; **gate (registered):
  median tcp tracking error < 10% of median per-step tcp displacement.**

G1.2 (data-only): per-control-step tcp displacement norms (medians/dispersion) — the stride
binding's denominator inputs.

Run (box): nohup .venv3d/bin/python -u experiments/p4_3d_g1.py > ~/p4_3d_g1.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T0 = time.time()
N_VAL = 20
POS_SCALE, ROT_SCALE = 0.1, 0.1
GRIP_LO, GRIP_HI = -0.01, 0.04
DEMO_DIR = Path.home() / "maniskill_demos" / "PegInsertionSide-v1"
OUT = ROOT / "papers" / "figures" / "p4_3d_g1.json"


def load_states(g):
    import h5py
    if isinstance(g, h5py.Dataset):
        return np.array(g)
    return {k: load_states(g[k]) for k in g}


def index_state(s, t):
    import torch
    if isinstance(s, np.ndarray):
        return torch.as_tensor(s[t])
    return {k: index_state(v, t) for k, v in s.items()}


def tcp_of(env) -> np.ndarray:
    p = env.unwrapped.agent.tcp.pose
    q = np.concatenate([np.asarray(p.p).reshape(-1), np.asarray(p.q).reshape(-1)])
    return q.astype(np.float64)                                   # (7,) xyz + wxyz


def main() -> int:
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    from transforms3d.quaternions import qinverse, qmult, quat2axangle

    art: dict = {"constants": {"pos_scale": POS_SCALE, "rot_scale": ROT_SCALE,
                               "gate": "median tracking err < 10% median step displacement"},
                 "episodes": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    traj = sorted(DEMO_DIR.rglob("*.h5"))[0]
    meta = json.loads(traj.with_suffix(".json").read_text())
    env = gym.make("PegInsertionSide-v1", obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu")

    all_disp, all_err, ep_verdicts = [], [], []
    with h5py.File(traj, "r") as f:
        for ep in meta["episodes"][:N_VAL]:
            tid = f"traj_{ep['episode_id']}"
            states = load_states(f[tid]["env_states"])
            n = len(np.atleast_1d(next(iter(states.values()))
                    if isinstance(states, dict) else states)) - 1
            n = len(f[tid]["actions"])
            # --- reference tcp trajectory via state-replay (no renderer)
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            ref = []
            grip_q = []
            for t in range(n + 1):
                env.unwrapped.set_state_dict(index_state(states, t))
                ref.append(tcp_of(env))
                qpos = np.asarray(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
                grip_q.append(float(qpos[-1]))                     # gripper joint
            ref = np.stack(ref)                                   # (n+1, 7)
            disp = np.linalg.norm(np.diff(ref[:, :3], axis=0), axis=1)
            all_disp.extend(disp.tolist())
            # --- convert to ee-delta actions
            acts = []
            for t in range(n):
                dp = (ref[t + 1, :3] - ref[t, :3]) / POS_SCALE
                ax, ang = quat2axangle(qmult(ref[t + 1, 3:7], qinverse(ref[t, 3:7])))
                om = np.asarray(ax) * ang / ROT_SCALE
                g = 2 * (grip_q[t + 1] - GRIP_LO) / (GRIP_HI - GRIP_LO) - 1
                acts.append(np.clip(np.concatenate([dp, om, [g]]), -1, 1).astype(np.float32))
            # --- open-loop execution from the same initial state
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            env.unwrapped.set_state_dict(index_state(states, 0))
            errs = []
            for t in range(n):
                env.step(acts[t])
                errs.append(float(np.linalg.norm(tcp_of(env)[:3] - ref[t + 1, :3])))
            all_err.extend(errs)
            med_e, med_d = float(np.median(errs)), float(np.median(disp))
            ep_verdicts.append(med_e < 0.1 * med_d)
            art["episodes"][tid] = {"n": n, "med_err": round(med_e, 5),
                                    "med_disp": round(med_d, 5),
                                    "ratio": round(med_e / max(med_d, 1e-9), 3),
                                    "pass": bool(ep_verdicts[-1])}
            print(f"  {tid}: med_err {med_e:.4f} med_disp {med_d:.4f} "
                  f"ratio {med_e/max(med_d,1e-9):.2f} {'✓' if ep_verdicts[-1] else '✗'}")
            save()
    env.close()

    art["G1a"] = {"pass_eps": f"{sum(ep_verdicts)}/{len(ep_verdicts)}",
                  "overall_ratio": round(float(np.median(all_err)) /
                                         max(float(np.median(all_disp)), 1e-9), 3),
                  "verdict": "PASS" if (np.median(all_err)
                                        < 0.1 * np.median(all_disp)) else "FAIL"}
    art["G1_2"] = {"tcp_step_disp": {"median": round(float(np.median(all_disp)), 5),
                                     "q10": round(float(np.quantile(all_disp, 0.1)), 5),
                                     "q90": round(float(np.quantile(all_disp, 0.9)), 5)}}
    save()
    print("G1a:", art["G1a"])
    print("G1.2:", art["G1_2"])
    print(f"G1 DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
