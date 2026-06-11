r"""P4 3D lane G0 contract — box-side (WSL/RTX3080): env contract -> demo replay -> transition
tensors -> CUDA JEPA smoke. Runs AFTER the model gates (tests/test_p4_3d_g0.py) pass.

Blocks (all crash-immune, artifact saved incrementally):
  A. env contract dump: PegInsertionSide-v1 @ obs_mode=pointcloud, control_mode=pd_ee_delta_pose,
     render_backend=cpu (llvmpipe — the WSL boundary measured 2026-06-11). Records obs tree,
     action space, controller frame config, sim/control freq. The *frame* entry decides whether
     the equivariant action story holds as-is (we need base/root-frame vectors).
  B. demo metadata: episode count, source control mode, length stats.
  C. replay 3 demos -> point clouds: primary = official CLI (mani_skill.trajectory.replay_trajectory
     with --target-control-mode conversion); fallback = manual state-replay with tcp-derived
     approximate ee-delta actions (flagged "approx" in the artifact). Output: downsampled
     (1024-pt) chunked transition tensors, CHUNK=1 for G0 (3D stride is a protocol decision
     deferred until the spectral timescale is measured — v1.2's stride-5 was PushT@10Hz-specific).
  D. CUDA JEPA smoke: VN pair (EMA) on the G0 transitions, 200 steps; pred_loss start->end,
     latent std, tcp-position probe R^2 (content sanity), ms/step on the 3080.

GPU_LANE.lock protocol: written at start, removed at end (shared-lane spec rule 3).

Run (box): cd ~/se3-ejepa && source .venv3d/bin/activate && python experiments/p4_3d_g0_contract.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vn_jepa import VNDGCNNEncoder, VNPredictor  # noqa: E402

T0 = time.time()
DEMO_DIR = Path.home() / "maniskill_demos" / "PegInsertionSide-v1"
DATA_DIR = ROOT / "data" / "p4_3d"
OUT = ROOT / "papers" / "figures" / "p4_3d_g0_contract.json"
LOCK = Path.home() / "GPU_LANE.lock"
N_PTS = 1024
N_DEMOS = 3

art: dict = {"blocks": {}}


def save() -> None:
    art["elapsed_min"] = round((time.time() - T0) / 60, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=1, default=str))


def shapes(tree, prefix="") -> dict:
    out = {}
    for k, v in tree.items():
        if hasattr(v, "items"):
            out.update(shapes(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = str(getattr(v, "shape", type(v).__name__))
    return out


def block_a() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make("PegInsertionSide-v1", obs_mode="pointcloud",
                   control_mode="pd_ee_delta_pose", render_backend="cpu")
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    ctrl = u.agent.controller.controllers if hasattr(u.agent.controller, "controllers") else {}
    cfgs = {k: {a: str(getattr(c.config, a)) for a in dir(c.config)
                if not a.startswith("_") and not callable(getattr(c.config, a))}
            for k, c in ctrl.items()}
    art["blocks"]["A_contract"] = {
        "obs_tree": shapes(obs),
        "action_space": str(env.action_space),
        "controller_configs": cfgs,
        "sim_freq": getattr(u, "sim_freq", None), "control_freq": getattr(u, "control_freq", None),
    }
    env.close()
    save()
    print(f"[A] action {art['blocks']['A_contract']['action_space']}")
    arm = cfgs.get("arm", {})
    print(f"[A] arm frame: {arm.get('frame', '?')}  bounds: {arm.get('pos_lower','?')}..")


def block_b() -> Path:
    h5s = sorted(DEMO_DIR.rglob("*.h5"))
    assert h5s, f"no demo h5 under {DEMO_DIR}"
    meta_p = h5s[0].with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    eps = meta.get("episodes", [])
    lens = [e.get("elapsed_steps", 0) for e in eps]
    art["blocks"]["B_demos"] = {
        "file": str(h5s[0]), "n_episodes": len(eps),
        "source_control_mode": meta.get("env_info", {}).get("env_kwargs", {}).get("control_mode"),
        "len_mean": float(np.mean(lens)) if lens else None,
        "len_max": int(np.max(lens)) if lens else None,
    }
    save()
    print(f"[B] {len(eps)} demos, source control {art['blocks']['B_demos']['source_control_mode']},"
          f" mean len {art['blocks']['B_demos']['len_mean']}")
    return h5s[0]


def replay_cli(traj: Path) -> Path | None:
    r"""Official replay with control-mode conversion; returns output h5 or None."""
    cmd = [sys.executable, "-m", "mani_skill.trajectory.replay_trajectory",
           "--traj-path", str(traj), "--obs-mode", "pointcloud",
           "--target-control-mode", "pd_ee_delta_pose", "--save-traj",
           "--count", str(N_DEMOS), "--sim-backend", "physx_cpu",
           "--render-backend", "cpu"]
    t0 = time.time()
    # inherit env (venv python is absolute); unknown flag / cuda:0 crash both -> rc!=0 -> fallback
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    cand = sorted(traj.parent.glob("*.pointcloud.pd_ee_delta_pose*.h5"))
    art["blocks"]["C_replay"] = {"mode": "cli", "rc": r.returncode,
                                 "min": round((time.time() - t0) / 60, 1),
                                 "stderr_tail": r.stderr[-400:] if r.returncode else ""}
    save()
    return cand[-1] if (r.returncode == 0 and cand) else None


def replay_manual(traj: Path) -> dict:
    r"""Fallback: state-replay + approximate ee-delta actions from consecutive tcp poses.

    $\Delta p_t = p_{t+1} - p_t$ (root frame); $\omega_t = \log(R_{t+1} R_t^\top)$ via
    quaternion difference (axis-angle, root frame). Flagged approx: ignores controller
    normalization — good enough to validate plumbing, NOT for training claims.
    """
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    from transforms3d.quaternions import qinverse, qmult, quat2axangle

    def load_states(g):
        if isinstance(g, h5py.Dataset):
            return np.array(g)
        return {k: load_states(g[k]) for k in g}

    def index_state(s, t):
        if isinstance(s, np.ndarray):
            return torch.as_tensor(s[t])
        return {k: index_state(v, t) for k, v in s.items()}

    env = gym.make("PegInsertionSide-v1", obs_mode="pointcloud",
                   control_mode="pd_ee_delta_pose", render_backend="cpu")
    meta = json.loads(traj.with_suffix(".json").read_text())
    clouds, tcps, acts, ep_bounds = [], [], [], []
    with h5py.File(traj, "r") as f:
        for ep in meta["episodes"][:N_DEMOS]:
            tid = f"traj_{ep['episode_id']}"
            states = load_states(f[tid]["env_states"])              # nested groups -> dict tree
            n = len(f[tid]["actions"])
            env.reset(seed=ep["episode_seed"], options={"reconfigure": True})
            start = len(clouds)
            for t in range(n + 1):
                env.unwrapped.set_state_dict(index_state(states, t))
                o = env.unwrapped.get_obs()
                pc = o["pointcloud"]["xyzw"]
                pc = pc[0] if pc.ndim == 3 else pc
                pc = np.asarray(pc.cpu() if hasattr(pc, "cpu") else pc)
                pc = pc[pc[:, 3] > 0.5][:, :3]                       # valid points only
                idx = np.random.default_rng(t).choice(len(pc), N_PTS, replace=len(pc) < N_PTS)
                clouds.append(pc[idx].astype(np.float32))
                tcp = np.asarray(o["extra"]["tcp_pose"].cpu()
                                 if hasattr(o["extra"]["tcp_pose"], "cpu")
                                 else o["extra"]["tcp_pose"]).reshape(-1)
                tcps.append(tcp.astype(np.float64))
            for t in range(start, start + n):
                dp = tcps[t + 1][:3] - tcps[t][:3]
                ax, ang = quat2axangle(qmult(tcps[t + 1][3:7], qinverse(tcps[t][3:7])))
                acts.append(np.concatenate([dp, ax * ang, [0.0]]).astype(np.float32))
            ep_bounds.append((start, start + n))
    env.close()
    art["blocks"]["C_replay"] = {"mode": "manual-approx", "n_frames": len(clouds),
                                 "n_eps": len(ep_bounds)}
    save()
    return {"clouds": np.stack(clouds), "tcp": np.stack(tcps),
            "actions": np.stack(acts), "ep_bounds": ep_bounds}


def parse_cli_h5(p: Path) -> dict:
    import h5py
    clouds, tcps, acts, ep_bounds = [], [], [], []
    with h5py.File(p, "r") as f:
        for tid in sorted(f.keys())[:N_DEMOS]:
            xyzw = np.array(f[tid]["obs"]["pointcloud"]["xyzw"])     # (T+1, P, 4)
            tcp = np.array(f[tid]["obs"]["extra"]["tcp_pose"])       # (T+1, 7)
            a = np.array(f[tid]["actions"])                          # (T, 7)
            start = len(clouds)
            for t in range(len(xyzw)):
                pc = xyzw[t]
                pc = pc[pc[:, 3] > 0.5][:, :3]
                idx = np.random.default_rng(t).choice(len(pc), N_PTS, replace=len(pc) < N_PTS)
                clouds.append(pc[idx].astype(np.float32))
            tcps.extend(tcp.astype(np.float64))
            acts.extend(a.astype(np.float32))
            ep_bounds.append((start, start + len(a)))
    return {"clouds": np.stack(clouds), "tcp": np.stack(tcps),
            "actions": np.stack(acts), "ep_bounds": ep_bounds}


def block_d(d: dict) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # transitions, CHUNK=1: (cloud_t, a_t, cloud_{t+1}) within episodes
    oi, ai, ni = [], [], []
    k = 0
    for s, e in d["ep_bounds"]:
        for t in range(s, e):
            oi.append(t); ai.append(k); ni.append(t + 1)
            k += 1
    x = torch.from_numpy(d["clouds"]).to(dev)
    a = torch.from_numpy(d["actions"]).to(dev)
    a_dim = a.shape[1]
    enc = VNDGCNNEncoder(c_vec=16, k=16, width=32).to(dev)
    pred = VNPredictor(16, 48, a_inv_dim=a_dim - 6).to(dev)
    import copy
    tgt = copy.deepcopy(enc)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-4)
    oi_t, ni_t = torch.tensor(oi), torch.tensor(ni)
    losses, t_train = [], time.time()
    for step in range(200):
        sel = torch.randint(0, len(ai), (32,))
        z = enc(x[oi_t[sel]])
        with torch.no_grad():
            zt = tgt(x[ni_t[sel]])
        loss = (pred(z, a[sel]) - zt).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, po in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(0.99).add_(po, alpha=0.01)
        losses.append(loss.item())
    ms = (time.time() - t_train) / 200 * 1000
    with torch.no_grad():
        z_all = torch.cat([tgt(x[i:i + 64]) for i in range(0, len(x), 64)])
    std = z_all.std(dim=0).mean().item()
    # tcp-position probe (ridge, in-sample smoke): content sanity only
    zc = (z_all - z_all.mean(0)).cpu().double()
    y = torch.from_numpy(d["tcp"][:, :3]) - torch.from_numpy(d["tcp"][:, :3]).mean(0)
    w = torch.linalg.lstsq(zc.T @ zc + 1e-3 * torch.eye(zc.shape[1]), zc.T @ y).solution
    r2 = (1 - ((zc @ w - y).pow(2).sum(0) / y.pow(2).sum(0))).mean().item()
    art["blocks"]["D_jepa_smoke"] = {
        "device": dev, "n_transitions": len(ai), "action_dim": int(a_dim),
        "pred_loss": {"first10": round(float(np.mean(losses[:10])), 4),
                      "last10": round(float(np.mean(losses[-10:])), 4)},
        "latent_std": round(std, 4), "tcp_probe_r2": round(r2, 3), "ms_per_step": round(ms, 1)}
    save()
    print(f"[D] dev={dev} loss {art['blocks']['D_jepa_smoke']['pred_loss']} std {std:.3f} "
          f"tcpR2 {r2:.3f} {ms:.0f}ms/step")


def main() -> int:
    LOCK.write_text(f"p4_3d_g0 pid={os.getpid()} eta=30min\n")
    try:
        for fn, name in ((block_a, "A"), ):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                art["blocks"][f"{name}_error"] = repr(exc)[:300]; save(); raise
        traj = block_b()
        out = None
        try:
            out = replay_cli(traj)
        except Exception as exc:  # noqa: BLE001
            art["blocks"]["C_cli_error"] = repr(exc)[:300]; save()
        d = parse_cli_h5(out) if out else replay_manual(traj)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA_DIR / "g0_transitions.npz", clouds=d["clouds"],
                            tcp=d["tcp"], actions=d["actions"],
                            ep_bounds=np.array(d["ep_bounds"]))
        art["blocks"]["C_tensors"] = {"clouds": str(d["clouds"].shape),
                                      "actions": str(d["actions"].shape)}
        save()
        block_d(d)
        print(f"G0 CONTRACT DONE ({(time.time() - T0) / 60:.1f} min)")
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
