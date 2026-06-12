r"""3D G1a-v2 — CLOSED-LOOP conversion validation (registered after v1's honest FAIL 0/20).

v1 mechanism of failure: pd_ee_delta_pose anchors each target on the ACHIEVED pose; offline
deltas (ref[t+1] ⊖ ref[t]) compound controller lag — ratio 27.3. v2 (the official CLI's own
mechanism): compute each action ONLINE against the achieved pose,

$$ a_t = [\mathrm{clip}((p^{\mathrm{ref}}_{t+1} - p^{\mathrm{ach}}_t)/0.1),\;
          \mathrm{clip}(\log(q^{\mathrm{ref}}_{t+1} (q^{\mathrm{ach}}_t)^{-1})/0.1),\; g_{t+1}] $$

— each step corrects toward the reference; error is bounded by one-step PD tracking, no
compounding. GATE (unchanged, registered): median tcp tracking error < 10% of median per-step
displacement, 20 demos. PASS ⇒ corpus production unlocks (same loop + pointcloud rendering).

Run (box): nohup /home/whb/se3-ejepa/.venv3d/bin/python -u \
    /home/whb/se3-ejepa/experiments/p4_3d_g1v2.py > /home/whb/p4_3d_g1v2.log 2>&1 &
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
OUT = ROOT / "papers" / "figures" / "p4_3d_g1v2.json"


def main() -> int:
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    from transforms3d.quaternions import qinverse, qmult, quat2axangle

    art: dict = {"design": "closed-loop conversion (delta vs ACHIEVED pose, official mechanism)",
                 "gate": "median tracking err < 10% median step displacement", "episodes": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    traj = sorted(DEMO_DIR.rglob("*.h5"))[0]
    meta = json.loads(traj.with_suffix(".json").read_text())
    env = gym.make("PegInsertionSide-v1", obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                   render_backend="cpu")

    all_disp, all_err, verdicts = [], [], []
    with h5py.File(traj, "r") as f:
        for ep in meta["episodes"][:N_VAL]:
            tid = f"traj_{ep['episode_id']}"
            states = load_states(f[tid]["env_states"])
            n = len(f[tid]["actions"])
            # pass 1: reference tcp + gripper qpos via state-replay
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            ref, grip_q = [], []
            for t in range(n + 1):
                env.unwrapped.set_state_dict(index_state(states, t))
                ref.append(tcp_of(env))
                qpos = np.asarray(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
                grip_q.append(float(qpos[-1]))
            ref = np.stack(ref)
            disp = np.linalg.norm(np.diff(ref[:, :3], axis=0), axis=1)
            all_disp.extend(disp.tolist())
            # pass 2: CLOSED-LOOP conversion + execution
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            env.unwrapped.set_state_dict(index_state(states, 0))
            errs, acts = [], []
            for t in range(n):
                cur = tcp_of(env)                                  # achieved pose
                dp = (ref[t + 1, :3] - cur[:3]) / POS_SCALE
                ax, ang = quat2axangle(qmult(ref[t + 1, 3:7], qinverse(cur[3:7])))
                om = np.asarray(ax) * ang / ROT_SCALE
                g = 2 * (grip_q[t + 1] - GRIP_LO) / (GRIP_HI - GRIP_LO) - 1
                a = np.clip(np.concatenate([dp, om, [g]]), -1, 1).astype(np.float32)
                env.step(a)
                acts.append(a)
                errs.append(float(np.linalg.norm(tcp_of(env)[:3] - ref[t + 1, :3])))
            all_err.extend(errs)
            med_e, med_d = float(np.median(errs)), float(np.median(disp))
            ok = med_e < 0.1 * med_d
            verdicts.append(ok)
            art["episodes"][tid] = {"med_err": round(med_e, 5), "med_disp": round(med_d, 5),
                                    "ratio": round(med_e / max(med_d, 1e-9), 3), "pass": bool(ok)}
            print(f"  {tid}: ratio {med_e / max(med_d, 1e-9):.3f} {'✓' if ok else '✗'}")
            save()
    env.close()
    art["G1a_v2"] = {"pass_eps": f"{sum(verdicts)}/{len(verdicts)}",
                     "overall_ratio": round(float(np.median(all_err)) /
                                            max(float(np.median(all_disp)), 1e-9), 3),
                     "verdict": "PASS" if (np.median(all_err)
                                           < 0.1 * np.median(all_disp)) else "FAIL"}
    save()
    print("G1a_v2:", art["G1a_v2"])
    print(f"G1V2 DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
