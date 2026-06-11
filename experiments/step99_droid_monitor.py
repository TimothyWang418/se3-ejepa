r"""Step 99 — the V-JEPA 2-AC monitor on REAL ROBOT DATA (droid_100), conditional gates per the pre-registered spec.

Spec (written before step98's verdict was known): `docs/specs/2026-06-10-step99-droid-monitor-seed.md`. Claim wording
红线: this is **real-robot DATA, offline monitoring** — the monitor is sensor-only/passive, so offline replay of
logged episodes is a faithful instantiation of the deployment, never "real-robot deployment/control".

Data: `lerobot/droid_100` (100 real Franka episodes, 15 fps, LeRobot v3: one parquet + one mp4 per camera with all
episodes concatenated in row order). Camera: `observation.images.exterior_image_1_left` (spec rule: first exterior
key). Model step = `SUB` raw frames (SUB from the official droid-256px-8f config's fps if parseable, else 4,
disclosed). Telemetry: actions from logged states via the authors' `poses_to_diff`; pose stream from the same log.

Monitor: state = per-frame token block (256x1408, layer-normed); read = encode frame; between reads forecast with
the AC predictor under telemetry; staleness = relative L2 on flattened normed tokens; theta=0.2; k in {1,2,4,8} +
k=24 censoring probe; n_ep=20. Fault arm: telemetry zeroed from mid-episode (descriptive unless G8-S active).

Run (box, GPU): ~/se3-ejepa/.venv/bin/python experiments/step99_droid_monitor.py
Writes: papers/figures/step99_droid_monitor.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step98_vjepa2_audit as s98  # noqa: E402  (load_model, encode_frame, HUB_DIR)

ROOT = Path(__file__).resolve().parent.parent
SMOKE = bool(int(os.environ.get("STEP99_SMOKE", "0")))
DROID = Path(os.environ.get("STEP99_DROID", str(Path.home() / "droid_100")))
CAM = "observation.images.exterior_image_1_left"
THETA = 0.2
K_LIST = [1, 2, 4, 8]
TPF = s98.TOKENS_PER_FRAME


def read_sub_from_config() -> int:
    r"""SUB = round(dataset_fps / training_fps) from the official config when parseable; else 4 (disclosed)."""
    try:
        import re
        cfg = (s98.HUB_DIR / "configs/train/vitg16/droid-256px-8f.yaml").read_text()
        m = re.search(r"fps:\s*(\d+)", cfg)
        if m:
            return max(1, round(15 / int(m.group(1))))
    except Exception:
        pass
    return 4


def load_episodes(n_ep: int, T: int, sub: int):
    r"""(frames_uint8 (T+1,H,W,3), states (T+1,7)) per episode at model-step cadence. The mp4 is AV1-encoded
    (cv2's bundled ffmpeg cannot decode it) -> PyAV with its bundled dav1d software decoder, ONE sequential pass
    over the single concatenated video, picking the needed global frame indices (row order == frame order,
    verified: 32212 rows == 32212 frames, `index` contiguous)."""
    import pandas as pd
    df = pd.read_parquet(DROID / "data/chunk-000/file-000.parquet",
                         columns=["episode_index", "frame_index", "observation.state", "index"])
    need = T * sub + 1
    chosen = []
    for ep, g in df.groupby("episode_index"):
        if len(g) >= need:
            chosen.append((int(ep), g.sort_values("frame_index").iloc[:need:sub]))
        if len(chosen) >= n_ep:
            break
    assert len(chosen) == n_ep, f"only {len(chosen)} episodes with >= {need} frames"
    wanted = {}
    for ep, rows in chosen:
        for gi in rows["index"].to_numpy():
            wanted[int(gi)] = None
    import av
    with av.open(str(DROID / f"videos/{CAM}/chunk-000/file-000.mp4")) as container:
        gi = 0
        for frame in container.decode(video=0):
            if gi in wanted:
                wanted[gi] = frame.to_ndarray(format="rgb24")
            gi += 1
            if all(v is not None for v in wanted.values()):
                break
    missing = [k for k, v in wanted.items() if v is None]
    assert not missing, f"{len(missing)} frames undecoded"
    out = []
    for ep, rows in chosen:
        idxs = rows["index"].to_numpy()
        states = np.stack(rows["observation.state"].to_numpy())
        frames = np.stack([wanted[int(gi)] for gi in idxs])
        out.append((ep, frames, states.astype(np.float32)))
    return out


def monitor_episode(pred, z_true, acts, states, k, fault_t=None):
    r"""Latent-space replay (step97-isomorphic). z_true: (T+1, TPF*1408) normed; acts: (T,1,7); states: (T,1,7)."""
    T = len(acts)
    z_hat = z_true[0].clone()
    invalid, flags, reads = 0, [], 0
    one_step = []
    with torch.no_grad():
        for t in range(T):
            a = acts[t] if (fault_t is None or t < fault_t) else torch.zeros_like(acts[t])
            zp = pred(z_hat.view(1, TPF, 1408).cuda(), a.cuda(), states[t].cuda())[:, -TPF:]
            z_hat = F.layer_norm(zp, (zp.size(-1),)).reshape(-1).cpu()
            rel = float((z_hat - z_true[t + 1]).norm() / z_true[t + 1].norm().clamp_min(1e-12))
            if k == 1:
                one_step.append(rel)
            stale = rel > THETA
            if stale:
                invalid += 1
            if (t + 1) % k == 0:
                reads += 1
                if stale:
                    flags.append(t)
                z_hat = z_true[t + 1].clone()
    return invalid / T, (len(flags) / max(1, reads)), flags, one_step


def crossings_probe(pred, z_true, acts, states, k=24):
    r"""Per-window first crossing (censored at k+1)."""
    T = len(acts)
    z_hat = z_true[0].clone()
    cur, s_in, cross = None, 0, []
    with torch.no_grad():
        for t in range(T):
            zp = pred(z_hat.view(1, TPF, 1408).cuda(), acts[t].cuda(), states[t].cuda())[:, -TPF:]
            z_hat = F.layer_norm(zp, (zp.size(-1),)).reshape(-1).cpu()
            s_in += 1
            rel = float((z_hat - z_true[t + 1]).norm() / z_true[t + 1].norm().clamp_min(1e-12))
            if cur is None and rel > THETA:
                cur = s_in
            if (t + 1) % k == 0:
                cross.append(cur if cur is not None else k + 1)
                z_hat = z_true[t + 1].clone()
                cur, s_in = None, 0
    return cross


def main() -> int:
    n_ep = int(os.environ.get("STEP99_EPISODES", "3" if SMOKE else "20"))
    T = 20 if SMOKE else 60
    sub = read_sub_from_config()
    print(f"[step99] sub={sub} (model step = {sub} raw frames @15fps)", file=sys.stderr)
    enc, pred = s98.load_model()
    s98.encode_frame._enc = enc
    sys.path.insert(0, str(s98.HUB_DIR / "notebooks"))
    from utils.mpc_utils import poses_to_diff  # type: ignore

    eps = load_episodes(n_ep, T, sub)
    print(f"[step99] {len(eps)} episodes loaded (cam={CAM})", file=sys.stderr)
    inv = {k: [] for k in K_LIST}
    flagrate = {k: [] for k in K_LIST}
    one_steps, crossings, pre_post = [], [], []
    for ei, (ep, frames, states) in enumerate(eps):
        z_true = []
        for fr in frames:
            z = s98.encode_frame(enc, fr)
            z_true.append(F.layer_norm(z, (z.size(-1),)).reshape(-1).cpu())
        z_true = torch.stack(z_true)
        acts = [torch.tensor(poses_to_diff(states[t], states[t + 1]), dtype=torch.float32).view(1, 1, 7)
                for t in range(T)]
        sts = [torch.tensor(states[t], dtype=torch.float32).view(1, 1, 7) for t in range(T)]
        for k in K_LIST:
            iv, fr_rate, _, ones = monitor_episode(pred, z_true, acts, sts, k)
            inv[k].append(iv)
            flagrate[k].append(fr_rate)
            if k == 1:
                one_steps += ones
        crossings += crossings_probe(pred, z_true, acts, sts, k=24 if not SMOKE else T)
        tf = T // 2
        _, _, flags, _ = monitor_episode(pred, z_true, acts, sts, 2, fault_t=tf)
        pre = sum(1 for f in flags if f < tf) / max(1, tf // 2)
        post = sum(1 for f in flags if f >= tf) / max(1, (T - tf) // 2)
        pre_post.append((pre, post))
        print(f"[step99] ep{ep}: inv@k=" + " ".join(f"{k}:{inv[k][-1]:.2f}" for k in K_LIST) +
              f" | fault pre/post {pre:.2f}/{post:.2f}", file=sys.stderr)

    c = np.array(crossings, dtype=float)
    med_cross = float(np.median(c))
    cens_frac = float((c > (24 if not SMOKE else T)).mean())
    med_one = float(np.median(one_steps))
    sub_class = ("stable" if (cens_frac >= 0.5 and med_one < THETA / 2) else
                 "bias" if (med_one >= THETA / 2 or med_cross <= 3) else "mixed")
    cert = json.loads((ROOT / "papers/figures/step98_vjepa2_audit.json").read_text())
    expansive = cert["verdict"].startswith("EXPANSIVE")
    T1 = next((r["T1_steps"] for r in cert["cert_rows"] if r["eps"] == THETA), None)
    out = {"theta": THETA, "k_list": K_LIST, "n_ep": n_ep, "T": T, "sub": sub, "camera": CAM,
           "episodes": [int(e[0]) for e in eps],
           "invalid_by_k": {str(k): float(np.mean(inv[k])) for k in K_LIST},
           "flag_rate_by_k": {str(k): float(np.mean(flagrate[k])) for k in K_LIST},
           "one_step_rel_err_median": med_one, "crossing_median": med_cross, "censored_frac": cens_frac,
           "telemetry_fault_pre_post": [float(np.mean([p for p, _ in pre_post])),
                                        float(np.mean([q for _, q in pre_post]))],
           "certificate": {"verdict": cert["verdict"], "lambda1": cert["lambda1"], "T1_at_theta": T1},
           "subclass_rule": "stable iff censored>=50% AND one-step<theta/2; bias iff one-step>=theta/2 OR crossing<=3",
           "measured_subclass": sub_class}
    # --- conditional gate (one branch activates; spec pre-registered) ---
    if expansive and T1 is not None:
        g8 = (T1 / 1.5 <= med_cross <= 1.5 * T1)
        out["branch"], out["G8"] = "G8-E (expansive pricing)", bool(g8)
    elif sub_class == "stable":
        iv24 = float(np.mean([crossings_probe.__defaults__]))  # placeholder never hit: stable needs k24 invalid
        g8 = (cens_frac >= 0.8)
        out["branch"], out["G8"] = "G8-S (free monitoring)", bool(g8)
    elif sub_class == "bias":
        g8 = (all(out["invalid_by_k"][str(k)] > 0.25 for k in K_LIST if k >= 2)
              and all(out["flag_rate_by_k"][str(k)] >= 0.35 for k in K_LIST if k >= 2))
        out["branch"], out["G8"] = "G8-B (do-not-deploy, step97-isomorphic)", bool(g8)
    else:
        out["branch"], out["G8"] = "mixed (no gate, INCONCLUSIVE)", None
    print(f"[step99] certificate={cert['verdict'][:30]}... subclass={sub_class} branch={out['branch']} "
          f"G8={out['G8']} | one-step med={med_one:.3f} crossing med={med_cross} censored={cens_frac:.2f}",
          file=sys.stderr)
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step99_droid_monitor{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(out, indent=2))
    return 0 if out["G8"] in (True, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
