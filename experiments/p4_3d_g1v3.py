r"""3D G1a-v3 — task-success fitness of closed-loop replays (gate re-scoped, rationale ledgered).

Corpus self-consistency holds by construction (closed-loop actions pair with their own frames);
the remaining requirement is TASK VALIDITY. Gate (registered in the v2 FAIL entry):
**closed-loop replay task-success ≥ 80% of the demos' own success rate** (20 demos; success =
env's info["success"] at any step in the final quarter of the episode, matching the env's own
evaluation; demos' own success measured by state-replay readout of the same flag).

Run (box): nohup /home/whb/se3-ejepa/.venv3d/bin/python -u \
    /home/whb/se3-ejepa/experiments/p4_3d_g1v3.py > /home/whb/p4_3d_g1v3.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_3d_g1 import (  # noqa: E402
    DEMO_DIR, GRIP_HI, GRIP_LO, N_VAL, POS_SCALE, ROT_SCALE, index_state, load_states, tcp_of,
)

T0 = time.time()
OUT = ROOT / "papers" / "figures" / "p4_3d_g1v3.json"


def succ_of(env) -> bool:
    try:
        ev = env.unwrapped.evaluate()
        s = ev.get("success", False)
        return bool(s.item() if hasattr(s, "item") else s)
    except Exception:
        return False


def main() -> int:
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    from transforms3d.quaternions import qinverse, qmult, quat2axangle

    art: dict = {"gate": "replay success >= 80% of demos' own success", "episodes": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    traj = sorted(DEMO_DIR.rglob("*.h5"))[0]
    meta = json.loads(traj.with_suffix(".json").read_text())
    env = gym.make("PegInsertionSide-v1", obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                   render_backend="cpu")

    demo_succ, replay_succ = [], []
    with h5py.File(traj, "r") as f:
        for ep in meta["episodes"][:N_VAL]:
            tid = f"traj_{ep['episode_id']}"
            states = load_states(f[tid]["env_states"])
            n = len(f[tid]["actions"])
            # demo's own success: state-replay the FINAL state and evaluate
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            ref, grip_q = [], []
            d_succ = False
            for t in range(n + 1):
                env.unwrapped.set_state_dict(index_state(states, t))
                ref.append(tcp_of(env))
                qpos = np.asarray(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
                grip_q.append(float(qpos[-1]))
                if t >= 3 * n // 4 and succ_of(env):
                    d_succ = True
            ref = np.stack(ref)
            demo_succ.append(d_succ)
            # closed-loop replay, success flag in final quarter
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            env.unwrapped.set_state_dict(index_state(states, 0))
            r_succ = False
            for t in range(n):
                cur = tcp_of(env)
                dp = (ref[t + 1, :3] - cur[:3]) / POS_SCALE
                ax, ang = quat2axangle(qmult(ref[t + 1, 3:7], qinverse(cur[3:7])))
                om = np.asarray(ax) * ang / ROT_SCALE
                g = 2 * (grip_q[t + 1] - GRIP_LO) / (GRIP_HI - GRIP_LO) - 1
                env.step(np.clip(np.concatenate([dp, om, [g]]), -1, 1).astype(np.float32))
                if t >= 3 * n // 4 and succ_of(env):
                    r_succ = True
            replay_succ.append(r_succ)
            art["episodes"][tid] = {"demo_success": d_succ, "replay_success": r_succ}
            print(f"  {tid}: demo {'✓' if d_succ else '✗'} replay {'✓' if r_succ else '✗'}")
            save()
    env.close()
    ds, rs = sum(demo_succ), sum(replay_succ)
    art["G1a_v3"] = {"demo_success": f"{ds}/{len(demo_succ)}",
                     "replay_success": f"{rs}/{len(replay_succ)}",
                     "ratio": round(rs / max(ds, 1), 3),
                     "verdict": "PASS" if rs >= 0.8 * ds and ds > 0 else "FAIL"}
    save()
    print("G1a_v3:", art["G1a_v3"])
    print(f"G1V3 DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
