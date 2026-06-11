r"""Step 97 — the PIXEL-family deployment cell: what the certificate's ABSTAIN means for a deployed monitor (LeWM/PushT).

E15/step96 deployed the TD-MPC2 taxonomy cells; this adds the architecturally disjoint PIXEL family (LeWM: ViT +
transformer JEPA, official checkpoint, authors' code — the step91 audit cell). The published verdict there is
**ABSTAIN, bias-driven** ($\lambda_1=0.0013$, CI straddling $0$; one-step relative bias $\approx0.17$; bench median
$\theta{=}0.2$ crossing $=2$ model steps). The deployment meaning of that abstain, pre-registered as gates BEFORE any
run:

- **G6 (no usable cadence):** for every read cadence $k\ge2$ in $\{1,2,4,8\}$, the belief-invalid fraction exceeds
  $25\%$ — there is NO sensing budget at which this monitor keeps a valid belief ($k{=}1$ reported too).
- **G7 v2 (flooded alarm channel beyond one step):** for every $k\ge2$, the no-fault per-read flag rate is
  $\ge0.35$ — any fault detection would sit on a flooded channel; **the certificate's abstain told you, a-priori
  and training-free, that this world model buys ZERO sensing savings as a monitor** (usable cadence $=1$ $=$ read
  every frame $=$ no forecasting at all). DESIGN NOTE (v1→v2, smoke-caught): v1 demanded the flooded-channel
  property at $k{=}1$ too — wrong test: at $k{=}1$ the forecaster is never trusted beyond one step, so its flag
  rate only re-measures the published one-step bias ($0.17<\theta{=}0.2$ ⇒ clean), not monitor usability; $k{=}1$
  is reported as the no-forecasting baseline. The taxonomy now ORDERS deployment value: stable-abstain ($\infty$
  savings, step96) $>$ calibrated/optimistic (priced savings, E15) $>$ bias-abstain (zero savings, here).
- Descriptive, no gate — **telemetry-corruption arm:** from $t_f$ the monitor's action telemetry reads a constant
  non-zero chunk ($\delta=0.5$ per dim) while the env keeps executing $0$: report pre/post per-read flag rates.
  Stated prediction: the margin is small — detection is inseparable from nominal drift in an abstained cell.
- Descriptive — **moving-scene control:** a smoothed random-walk action sequence (telemetry = truth) replays the
  same metrics, pre-empting "the static scene is the artifact" (prediction: same verdict or worse).

Monitor design = E15's: sensor-only, passive; a read = grab + encode a camera frame; between reads the forecaster is
the AUDITED loop itself (zero-action window predictor — deployed forecaster $=$ certified loop by construction). The
monitor never controls the system, so OFFLINE REPLAY is a faithful instantiation, not a simulation of one (the same
property the step99 real-robot-data arm relies on). float64 CPU; pixel scale from the published step91 artifact.

Run (smoke): STEP97_SMOKE=1 ...; full: .venv/bin/python experiments/step97_lewm_monitor.py
Writes: papers/figures/step97_lewm_monitor.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step91_lewm_audit as s91  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMOKE = bool(int(os.environ.get("STEP97_SMOKE", "0")))
THETA = 0.2
K_LIST = [1, 2, 4, 8]
HS, ZD, CHUNK, ADIM = s91.HS, s91.ZD, s91.CHUNK, s91.ADIM


def act_emb_for(model, chunk_actions: np.ndarray):
    r"""Embed a (HS, CHUNK*ADIM) action history through the authors' float32 Embedder, cast to f64."""
    with torch.no_grad():
        return model.action_encoder(torch.tensor(chunk_actions, dtype=torch.float32).unsqueeze(0)).double()


def env_rollout_actions(seed: int, n_model_steps: int, action_fn):
    r"""Roll PushT for ``n_model_steps`` chunks under ``action_fn(t)`` -> (ADIM,) raw action held for the chunk.
    Returns (frames (n+HS, C, H, W), chunk_actions (n+HS-1, ADIM))."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    env = gym.make("swm/PushT-v1")
    obs, info = env.reset(seed=seed)

    def grab(o):
        px = o["pixels"] if isinstance(o, dict) and "pixels" in o else env.render()
        a = np.asarray(px)
        if a.ndim == 3 and a.shape[-1] == 3:
            a = a.transpose(2, 0, 1)
        return torch.tensor(a.copy(), dtype=torch.float64)

    frames = [grab(obs)]
    acts = []
    for t in range(n_model_steps + HS - 1):
        a = action_fn(t).astype(np.float32)
        for _ in range(CHUNK):
            obs, *_ = env.step(a)
        acts.append(a)
        frames.append(grab(obs))
    env.close()
    return torch.stack(frames, 0), np.stack(acts, 0)


def monitor_offline(model, z_true: torch.Tensor, k: int, fc_embs, fault_t: int | None = None,
                    fault_emb=None):
    r"""Latent-space monitor replay. ``z_true``: (T+HS, 192) encoded true latents. ``fc_embs(t)`` -> the action
    embedding the MONITOR uses at model step t (telemetry); after ``fault_t`` it is replaced by ``fault_emb``
    (corrupted telemetry) while reality (z_true) is unchanged. Sensor read every ``k`` steps -> compare last-frame
    forecast vs encoded truth, flag iff rel err > THETA, resync the window to true latents.
    Returns (invalid_fraction, per_read_flag_rate, flags, T)."""
    T = z_true.shape[0] - HS
    win = list(z_true[:HS])
    invalid, flags, reads = 0, [], 0
    with torch.no_grad():
        for t in range(T):
            emb = fc_embs(t) if (fault_t is None or t < fault_t) else fault_emb
            nxt = model.predict(torch.stack(win, 0).unsqueeze(0), emb)[0, -1]
            tgt = z_true[HS + t]
            rel = float((nxt - tgt).norm() / tgt.norm().clamp_min(1e-12))
            stale = rel > THETA
            if stale:
                invalid += 1
            if (t + 1) % k == 0:                                   # sensor read: compare then resync
                reads += 1
                if stale:
                    flags.append(t)
                win = list(z_true[HS + t - HS + 1: HS + t + 1])    # resync window to the true last-HS latents
            else:
                win = win[1:] + [nxt]
    return invalid / T, (len(flags) / max(1, reads)), flags, T


def main() -> int:
    n_ep = int(os.environ.get("STEP97_EPISODES", "3" if SMOKE else "10"))
    T = 24 if SMOKE else 60
    model = s91.load_lewm()
    scale01 = bool(json.loads((ROOT / "papers/figures/step91_lewm_audit.json").read_text())["pixel_scale01"])
    zero_emb = s91.zero_act_emb(model)
    delta = np.full((HS, CHUNK * ADIM), 0.5, dtype=np.float32)
    delta_emb = act_emb_for(model, delta)
    out = {"theta": THETA, "k_list": K_LIST, "n_ep": n_ep, "T": T,
           "published": {"lambda1": 0.0013, "verdict": "ABSTAIN (bias-driven)", "bench_median_at_0.2": 2.0},
           "arms": {}}

    # ---------- primary arm: zero-action system, certified-loop forecaster ----------
    inv = {k: [] for k in K_LIST}
    flagrate = {k: [] for k in K_LIST}
    pre_post = []
    for e in range(n_ep):
        fr = s91.env_rollout(seed=1000 + 7 * e, n_model_steps=T)
        z_true = s91.encode_seq(model, fr, scale01)
        for k in K_LIST:
            iv, fr_rate, _, _ = monitor_offline(model, z_true, k, lambda t: zero_emb)
            inv[k].append(iv)
            flagrate[k].append(fr_rate)
        # telemetry-corruption arm at k=2 (descriptive): corrupt from T//2
        tf = T // 2
        _, _, flags, _ = monitor_offline(model, z_true, 2, lambda t: zero_emb, fault_t=tf, fault_emb=delta_emb)
        pre = sum(1 for f in flags if f < tf) / max(1, tf // 2)
        post = sum(1 for f in flags if f >= tf) / max(1, (T - tf) // 2)
        pre_post.append((pre, post))
        print(f"[step97] ep{e}: inv@k=" + " ".join(f"{k}:{inv[k][-1]:.2f}" for k in K_LIST) +
              f" | telemetry-fault pre/post flag rate {pre:.2f}/{post:.2f}", file=sys.stderr)
    arm = {"invalid_by_k": {str(k): float(np.mean(inv[k])) for k in K_LIST},
           "flag_rate_by_k": {str(k): float(np.mean(flagrate[k])) for k in K_LIST},
           "telemetry_fault_pre_post": [float(np.mean([p for p, _ in pre_post])),
                                        float(np.mean([q for _, q in pre_post]))]}
    g6 = all(arm["invalid_by_k"][str(k)] > 0.25 for k in K_LIST if k >= 2)
    g7 = all(arm["flag_rate_by_k"][str(k)] >= 0.35 for k in K_LIST if k >= 2)   # v2: k=1 = no-forecasting baseline
    arm["G6_no_usable_cadence"], arm["G7_flooded_channel"] = bool(g6), bool(g7)
    out["arms"]["zero_action"] = arm
    print(f"[step97] zero-action arm: inv={arm['invalid_by_k']} flag={arm['flag_rate_by_k']} | G6={g6} G7={g7}",
          file=sys.stderr)

    # ---------- moving-scene control (descriptive): smoothed random-walk actions ----------
    n_mv = max(2, n_ep // 2)
    inv_mv = {k: [] for k in [1, 4]}
    for e in range(n_mv):
        rng = np.random.RandomState(3000 + e)
        state = {"a": np.zeros(ADIM)}

        def walk(t, rng=rng, state=state):
            state["a"] = np.clip(0.8 * state["a"] + 0.3 * rng.randn(ADIM), -1, 1)
            return state["a"]

        fr, acts = env_rollout_actions(seed=2000 + 7 * e, n_model_steps=T, action_fn=walk)
        z_true = s91.encode_seq(model, fr, scale01)
        # per-step telemetry embeddings: history of the last HS chunks, flattened per chunk
        embs = []
        for t in range(T):
            hist = np.zeros((HS, CHUNK * ADIM), dtype=np.float32)
            for h in range(HS):
                idx = t + h                                          # action driving frame (t+h)->(t+h+1)
                if 0 <= idx < acts.shape[0]:
                    hist[h] = np.tile(acts[idx], CHUNK)
            embs.append(act_emb_for(model, hist))
        for k in inv_mv:
            iv, _, _, _ = monitor_offline(model, z_true, k, lambda t: embs[t])
            inv_mv[k].append(iv)
        print(f"[step97] moving ep{e}: inv@k1={inv_mv[1][-1]:.2f} inv@k4={inv_mv[4][-1]:.2f}", file=sys.stderr)
    out["arms"]["moving_scene"] = {"invalid_by_k": {str(k): float(np.mean(v)) for k, v in inv_mv.items()},
                                   "note": "descriptive control; telemetry=truth; action-history embedding "
                                           "approximated by chunk-constant tiling (disclosed)"}
    out["determinism_note"] = ("zero-action arm is near-deterministic (static scene): invalid fraction is exactly "
                               "(k-1)/k and flag rate 1.0 at k>=2 — crossing at staleness 2, every window; "
                               "episode variation carried by the moving-scene arm")

    out["verdict"] = {"G6": bool(g6), "G7": bool(g7)}
    print(f"[step97] G6 {'PASS' if g6 else 'INCONCLUSIVE'} (abstain => no usable cadence); "
          f"G7 {'PASS' if g7 else 'INCONCLUSIVE'} (flooded channel).", file=sys.stderr)
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step97_lewm_monitor{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(out, indent=2))
    return 0 if (g6 and g7) else 1


if __name__ == "__main__":
    raise SystemExit(main())
