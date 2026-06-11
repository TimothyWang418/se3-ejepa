r"""wma-step1 — ATM audit of official LeWM (PushT): first WMA report-card column (instrument II).

Spec (pre-registered): docs/specs/2026-06-11-wma-step1-atm-selfimpl-seed.md
Instrument: src/audit/atm.py — ATM self-implementation (arXiv:2606.09028; official code
unreleased as of 2026-06-11), certified on synthetic gates G1a-c in tests/test_wma_step1.py.
Loader / encoding / G0b pixel-scale check: reused from step91 (G3 infra parity by construction).

Behavior policy: single-env replica of swm's PushT ``WeakPolicy``
(third_party/stable-worldmodel/stable_worldmodel/envs/pusht/expert_policy.py): relative
displacement targets, clipped to ``dist_constraint`` px around the block — the LeWM
ecosystem's own contact-seeking collection policy (the upstream class assumes a vector
env, hence the 8-line local replica; logic verbatim). Seeded per episode.

Action-window convention (source-verified in swm ``data/buffer.py::_gather_clip``): obs
strided by ``frameskip``; actions dense, reshaped ``(history_len, frameskip*adim)`` — the
LAST row is the chunk driving $F_t \to F_{t+1}$. Probe target = that 10-d chunk
$c_t \in [-1,1]^{10}$.

Gates:
  G2 (pattern, NOT numeric replication): healthy-regime $|G_{T\to P}| \le 0.5$ and
     $L_{\mathrm{sym}} \le 2.0$; ATM Table 6 LeWM/PushT reference (raw): $D_{T,T}=0.1829$,
     $|G_{T\to P}|=0.0372$, $L_{\mathrm{sym}}=0.2263$ — magnitude deviation disclosed, ungated
     (probe hyperparams / data distribution / action normalization are paper-unstated dof).
  G3 (infra parity): replay step91's pre-registered G0b one-step pixel-scale check verbatim
     (same seed, same code path) and diff against the recorded step91 JSON.

Run (smoke): WMA1_SMOKE=1 .venv/bin/python experiments/wma_step1_atm_probe.py
Writes: papers/figures/wma_step1_atm_lewm{_smoke}.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from step91_lewm_audit import (  # noqa: E402
    ADIM, CHUNK, DTYPE, HS, ZD, encode_seq, env_rollout, load_lewm, zero_act_emb,
)

from src.audit.atm import atm_audit  # noqa: E402

SMOKE = bool(int(os.environ.get("WMA1_SMOKE", "0")))
DIST_CONSTRAINT = 100.0   # swm WeakPolicy default
EP_SEED0 = 1000
SPLIT_SEED = 0
TEST_FRAC_EP = 0.2

# ATM reference values — Chen, arXiv:2606.09028, Table 6, LeWM/PushT row (reported x100)
ATM_TABLE6_LEWM_PUSHT = {"D_TT": 0.1829, "abs_G_TP": 0.0372, "L_sym": 0.2263}


def weak_action(base_env, rng: np.random.Generator, dist: float = DIST_CONSTRAINT) -> np.ndarray:
    r"""Verbatim single-env replica of swm WeakPolicy.get_action (expert_policy.py L66-80).

    Sample $u \sim U[-1,1]^2$, map to a position target ``agent.pos + u * action_scale``,
    clip into the ``dist``-px box around the block (contact seeking), then re-normalize to
    the relative action space and clip to $[-1, 1]^2$.
    """
    agent = np.asarray(base_env.agent.position, dtype=np.float64)
    a = rng.uniform(-1.0, 1.0, size=(2,)) * base_env.action_scale + agent
    bp = np.array((base_env.block.position.x, base_env.block.position.y), dtype=np.float64)
    a = np.clip(a, bp - dist, bp + dist)
    a = (a - agent) / base_env.action_scale
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def grab(obs, env) -> torch.Tensor:
    """Frame fetch, identical logic to step91's grab(): obs['pixels'] if present else render()."""
    px = obs["pixels"] if isinstance(obs, dict) and "pixels" in obs else env.render()
    a = np.asarray(px)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.transpose(2, 0, 1)
    return torch.tensor(a.copy(), dtype=DTYPE)


def collect_episode(seed: int, n_model_steps: int):
    """Roll one WeakPolicy episode; return frames (n+1, C, H, W) and chunks (n, CHUNK*ADIM)."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/* envs)
    env = gym.make("swm/PushT-v1")
    base = env.unwrapped
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames = [grab(obs, env)]
    chunks = []
    done = False
    for _ in range(n_model_steps):
        acts = []
        for _ in range(CHUNK):
            a = weak_action(base, rng)
            obs, _, term, trunc, info = env.step(a)
            acts.append(a)
            if term or trunc:
                done = True
                break
        if done:
            break
        frames.append(grab(obs, env))
        chunks.append(np.concatenate(acts, dtype=np.float32))
    env.close()
    return torch.stack(frames, 0), np.stack(chunks, 0) if chunks else np.zeros((0, CHUNK * ADIM)), done


def main() -> int:
    t0 = time.time()
    n_ep, n_step = (4, 12) if SMOKE else (132, 40)
    audit_kw = (dict(widths=(256,), main_width=256, probe_seeds=(0,), max_steps=400)
                if SMOKE else {})

    print("[wma1] loading official LeWM (PushT) ...", file=sys.stderr)
    model = load_lewm()
    act_dim = CHUNK * ADIM

    # ---- G3: replay step91's G0b pixel-scale check verbatim (same seed, same code path) ----
    fr0 = env_rollout(seed=0, n_model_steps=6)
    errs = {}
    for scale01 in (True, False):
        z = encode_seq(model, fr0, scale01)
        win = z[:HS].unsqueeze(0)
        with torch.no_grad():
            pred = model.predict(win, zero_act_emb(model))[0, -1]
        errs[scale01] = float((pred - z[HS]).norm() / z[HS].norm().clamp_min(1e-12))
    scale01 = errs[True] <= errs[False]
    rec_path = ROOT / "papers/figures/step91_lewm_audit.json"
    g3 = {"one_step_errs_replay": {str(k).lower(): v for k, v in errs.items()},
          "scale01": bool(scale01)}
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        rec_errs = rec.get("one_step_errs", {})
        diffs = [abs(errs[k] - rec_errs.get(str(k).lower(), np.nan)) for k in (True, False)]
        g3.update({"step91_recorded": rec_errs,
                   "max_abs_diff": float(np.nanmax(diffs)),
                   "scale01_match": bool(scale01 == rec.get("pixel_scale01")),
                   "pass": bool(np.nanmax(diffs) < 1e-6 and scale01 == rec.get("pixel_scale01"))})
    else:
        g3.update({"step91_recorded": None, "pass": None,
                   "note": "step91 JSON not found; parity replay reported without diff"})
    print(f"[wma1] G3 parity: errs={ {str(k): round(v, 6) for k, v in errs.items()} } "
          f"pass={g3.get('pass')}", file=sys.stderr)

    # ---- collect WeakPolicy transitions ----
    print(f"[wma1] collecting {n_ep} episodes x {n_step} model-steps (WeakPolicy) ...",
          file=sys.stderr)
    Zt, Znext, Zwin, Awin, Atgt, ep_ids = [], [], [], [], [], []
    n_dropped = 0
    for ep in range(n_ep):
        frames, chunks, done_early = collect_episode(EP_SEED0 + ep, n_step)
        n = chunks.shape[0]
        if n < HS:                                   # need >= HS chunks for one transition
            n_dropped += 1
            continue
        with torch.no_grad():
            z = encode_seq(model, frames, scale01)   # (n+1, ZD), f64
        for t in range(HS - 1, n):                   # transition t -> t+1
            Zt.append(z[t])
            Znext.append(z[t + 1])
            Zwin.append(z[t - HS + 1:t + 1])         # (HS, ZD): z_{t-2}, z_{t-1}, z_t
            Awin.append(chunks[t - HS + 1:t + 1])    # (HS, 10): rows c_{t-2}, c_{t-1}, c_t
            Atgt.append(chunks[t])                   # 10-d chunk driving t -> t+1
            ep_ids.append(ep)
        if (ep + 1) % 20 == 0:
            print(f"[wma1]   {ep + 1}/{n_ep} episodes, {len(Atgt)} transitions "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)
    Zt, Znext = torch.stack(Zt, 0), torch.stack(Znext, 0)
    Zwin = torch.stack(Zwin, 0)                                  # (N, HS, ZD) f64
    Awin = torch.tensor(np.stack(Awin, 0))                       # (N, HS, 10)
    Atgt = torch.tensor(np.stack(Atgt, 0), dtype=torch.float32)  # (N, 10)
    ep_ids = np.asarray(ep_ids)
    N = Zt.shape[0]
    print(f"[wma1] {N} transitions from {len(set(ep_ids.tolist()))} episodes "
          f"({n_dropped} dropped)", file=sys.stderr)

    # ---- model-predicted endpoints zhat_{t+1} = Pred(window, action rows)[:, -1] ----
    print("[wma1] predicting zhat_{t+1} (batched) ...", file=sys.stderr)
    Zpred = torch.empty(N, ZD, dtype=DTYPE)
    bs = 256
    with torch.no_grad():
        for lo in range(0, N, bs):
            zb = Zwin[lo:lo + bs]
            ab = model.action_encoder(Awin[lo:lo + bs].float()).double()
            Zpred[lo:lo + bs] = model.predict(zb, ab)[:, -1]

    # ---- episode-level split ----
    eps_unique = np.unique(ep_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(eps_unique)
    n_test_ep = max(1, int(round(TEST_FRAC_EP * eps_unique.shape[0])))
    test_eps = set(eps_unique[:n_test_ep].tolist())
    test_mask = np.isin(ep_ids, list(test_eps))
    train_idx = torch.tensor(np.where(~test_mask)[0])
    test_idx = torch.tensor(np.where(test_mask)[0])

    # ---- ATM audit (full registered protocol unless SMOKE) ----
    print("[wma1] running ATM audit ...", file=sys.stderr)
    res = atm_audit(Zt, Znext, Zpred, Atgt, train_idx=train_idx, test_idx=test_idx, **audit_kw)

    r = res.readouts
    g2_pass = bool(abs(r["G_TP"]) <= 0.5 and r["L_sym"] <= 2.0)
    one_step_rel = float(((Zpred - Znext).norm(dim=-1) /
                          Znext.norm(dim=-1).clamp_min(1e-12)).median())
    out = {
        "model": "lewm-pusht (official ckpt, authors' code, strict load; reused step91 loader)",
        "instrument": "ATM self-implementation (arXiv:2606.09028; official code unreleased)",
        "atm": res.to_dict(),
        "gates": {
            "G2": {"pass": g2_pass, "abs_G_TP": abs(r["G_TP"]), "L_sym": r["L_sym"],
                   "thresholds": {"abs_G_TP": 0.5, "L_sym": 2.0},
                   "reference_atm_table6_lewm_pusht": ATM_TABLE6_LEWM_PUSHT,
                   "D_TT_ratio_vs_paper": res.D[("T", "T")] / ATM_TABLE6_LEWM_PUSHT["D_TT"],
                   "note": "pattern gate; magnitude deviation disclosed, ungated (probe "
                           "hyperparams / data distribution / action normalization are "
                           "paper-unstated degrees of freedom)"},
            "G3": g3,
        },
        "data": {"env": "swm/PushT-v1", "policy": "WeakPolicy replica (dist_constraint=100)",
                 "episodes": int(len(set(ep_ids.tolist()))), "dropped": n_dropped,
                 "n_transitions": int(N), "model_steps_per_ep": n_step,
                 "split": {"test_frac_ep": TEST_FRAC_EP, "split_seed": SPLIT_SEED,
                           "n_train": int(train_idx.shape[0]), "n_test": int(test_idx.shape[0])},
                 "action_convention": "target = last action row of swm window "
                                      "(buffer._gather_clip): 10-d chunk driving F_t->F_{t+1}",
                 "pixel_scale01": bool(scale01), "ep_seed0": EP_SEED0},
        "one_step_rel_err_median": one_step_rel,
        "wall_seconds": round(time.time() - t0, 1),
        "smoke": SMOKE,
    }
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"wma_step1_atm_lewm{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[wma1] D={ {f'{i}{j}': round(v, 4) for (i, j), v in res.D.items()} }", file=sys.stderr)
    print(f"[wma1] D_norm={ {f'{i}{j}': round(v, 4) for (i, j), v in res.D_norm.items()} }",
          file=sys.stderr)
    print(f"[wma1] readouts={ {k: round(v, 4) for k, v in r.items()} }", file=sys.stderr)
    print(f"[wma1] G2 pass={g2_pass}  G3 pass={g3.get('pass')}  "
          f"wall={out['wall_seconds']}s -> wrote wma_step1_atm_lewm{tag}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
