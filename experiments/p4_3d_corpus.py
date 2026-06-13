r"""3D corpus production — closed-loop replays + same-pass point clouds (overnight, box).

Per the 06-12 scope decision: coverage-grade corpus (self-consistent (cloud, a, cloud') tuples
by construction), goal rows out of scope. Mechanism = G1a-v2 closed-loop conversion; rendering
= llvmpipe CPU (the WSL boundary's sanctioned path, G0-measured).

Output shards: data/p4_3d/corpus_NNN.npz, each ≤ 25 episodes: clouds (E, T+1, 1024, 3) f32,
actions (E, T, 7) f32, tcp (E, T+1, 7) f64, success flags. N_EPS=200 first tranche (G2 smoke
scale); the full-1000 decision waits for G2's appetite.

Run (box): nohup nice -n 10 /home/whb/se3-ejepa/.venv3d/bin/python -u \
    /home/whb/se3-ejepa/experiments/p4_3d_corpus.py > /home/whb/p4_3d_corpus.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_3d_g1 import (  # noqa: E402
    DEMO_DIR, GRIP_HI, GRIP_LO, POS_SCALE, ROT_SCALE, index_state, load_states, tcp_of,
)

T0 = time.time()
EP_START = int(os.environ.get("P4_EP_START", "0"))
EP_END = int(os.environ.get("P4_EP_END", "200"))
N_PTS = 1024
SHARD = 25
OUT_DIR = ROOT / "data" / "p4_3d"
META = ROOT / "papers" / "figures" / "p4_3d_corpus_meta.json"


def cloud_of(env, rng: np.random.Generator) -> np.ndarray:
    obs = env.unwrapped.get_obs()
    pc = obs["pointcloud"]["xyzw"]
    pc = pc[0] if pc.ndim == 3 else pc
    pc = np.asarray(pc.cpu() if hasattr(pc, "cpu") else pc)
    pc = pc[pc[:, 3] > 0.5][:, :3]
    idx = rng.choice(len(pc), N_PTS, replace=len(pc) < N_PTS)
    return pc[idx].astype(np.float32)


def main() -> int:
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    from transforms3d.quaternions import qinverse, qmult, quat2axangle

    meta_art: dict = {"mechanism": "closed-loop ee-delta replay + same-pass clouds",
                      "n_pts": N_PTS, "shards": []}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    traj = sorted(DEMO_DIR.rglob("*.h5"))[0]
    meta = json.loads(traj.with_suffix(".json").read_text())
    env = gym.make("PegInsertionSide-v1", obs_mode="pointcloud",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                   render_backend="cpu")
    rng = np.random.default_rng(0)

    shard_c, shard_a, shard_t = [], [], []
    shard_id = EP_START // SHARD
    with h5py.File(traj, "r") as f:
        ep_slice = meta["episodes"][EP_START:EP_END]
        for k, ep in enumerate(ep_slice):
            tid = f"traj_{ep['episode_id']}"
            states = load_states(f[tid]["env_states"])
            n = len(f[tid]["actions"])
            # pass 1 (state-replay): reference tcp + gripper
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            ref, grip_q = [], []
            for t in range(n + 1):
                env.unwrapped.set_state_dict(index_state(states, t))
                ref.append(tcp_of(env))
                qpos = np.asarray(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
                grip_q.append(float(qpos[-1]))
            ref = np.stack(ref)
            # pass 2: closed-loop replay, clouds + executed actions recorded
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            env.unwrapped.set_state_dict(index_state(states, 0))
            clouds = [cloud_of(env, rng)]
            tcps = [tcp_of(env)]
            acts = []
            for t in range(n):
                cur = tcps[-1]
                dp = (ref[t + 1, :3] - cur[:3]) / POS_SCALE
                ax, ang = quat2axangle(qmult(ref[t + 1, 3:7], qinverse(cur[3:7])))
                om = np.asarray(ax) * ang / ROT_SCALE
                g = 2 * (grip_q[t + 1] - GRIP_LO) / (GRIP_HI - GRIP_LO) - 1
                a = np.clip(np.concatenate([dp, om, [g]]), -1, 1).astype(np.float32)
                env.step(a)
                acts.append(a)
                clouds.append(cloud_of(env, rng))
                tcps.append(tcp_of(env))
            # pad to uniform length within shard via truncation convention (min len later);
            # here store per-episode arrays in object shards -> save ragged via npz per episode
            shard_c.append(np.stack(clouds))
            shard_a.append(np.stack(acts))
            shard_t.append(np.stack(tcps))
            if len(shard_c) >= SHARD or k == len(ep_slice) - 1:
                pth = OUT_DIR / f"corpus3d_{shard_id:03d}.npz"
                np.savez_compressed(
                    pth,
                    **{f"c{i}": c for i, c in enumerate(shard_c)},
                    **{f"a{i}": a for i, a in enumerate(shard_a)},
                    **{f"t{i}": t_ for i, t_ in enumerate(shard_t)},
                    n=len(shard_c))
                meta_art["shards"].append({"file": pth.name, "eps": len(shard_c)})
                meta_art["elapsed_min"] = round((time.time() - T0) / 60, 1)
                META.write_text(json.dumps(meta_art, indent=1))
                print(f"[shard {shard_id}] saved {len(shard_c)} eps "
                      f"({(time.time() - T0) / 60:.0f} min, ep {k + 1}/{N_EPS})")
                shard_c, shard_a, shard_t = [], [], []
                shard_id += 1
    env.close()
    meta_art["done"] = True
    META.write_text(json.dumps(meta_art, indent=1))
    print(f"CORPUS3D DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
