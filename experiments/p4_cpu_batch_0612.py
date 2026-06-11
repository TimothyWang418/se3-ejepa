r"""P4 CPU batch 2026-06-12 — runs PARALLEL to the MPS champion confirmation (CPU-only by design).

Blocks (declared):
1. **E0.3 shape-generalization audits** (Tier 0, the free moat experiment): collect held-out
   episodes per unseen shape {L, Z, square, +} (the env's block.shape variation, default-frozen
   at T — all historical data is T-only, so these are honestly OOD); audit the EXISTING stable
   Stage-B pairs (ckpt7 eqwin/plainwin) on each shape. Readouts (descriptive, no gate): δ̂ per
   shape vs on-T, and the C3-guar-style one-sidedness on OOD shapes. Encoders run CPU f32;
   audits CPU f64 (the audit convention anyway).
2. **c5000 corpus collection** (data-scaling extension beyond the 2000-ep knee-search).
3. **Wedge-lane data prep** (Tier 1 ready-up): wedge corpus (initial block angle ∈ [0°,90°),
   200 eps) + held-out in-wedge (60) + held-out OUT-wedge (60, angles ∈ [90°,360°)) — via
   reset(options={'state': sampled}) so no env modification.

Shape selection: variation_space['block']['shape'].value is set directly and VERIFIED by a
render-difference assert against the T render (empirical guard against a silent no-op).

Run: nohup nice .venv/bin/python experiments/p4_cpu_batch_0612.py   (~1.5 h CPU)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_spine_stage1a import boundary_from_curve, measured_err  # noqa: E402
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, make_env,
    pick_ladder, weak_action,
)
from src.audit.gap_mode import audit_gap  # noqa: E402

T0 = time.time()
SHAPES = {"L": 1, "Z": 3, "square": 4, "+": 7}
EPS_MULT = (2, 4, 8, 16)
OUT = ROOT / "papers" / "figures" / "p4_cpu_batch_0612.json"


def collect_shape(n_episodes: int, seed: int, shape_idx: int) -> dict:
    r"""WeakPolicy episodes with the block shape pinned; render-difference guard."""
    env = make_env(None)
    env.reset(seed=0)
    ref = env.render().copy()  # T-shape reference
    env.unwrapped.variation_space["block"]["shape"].value = shape_idx
    env.reset(seed=0)
    assert np.abs(env.render().astype(int) - ref.astype(int)).sum() > 1000, \
        f"shape set to {shape_idx} did not change the render — selection is a no-op"
    rng = np.random.default_rng(seed)
    frames, states, actions = [], [], []
    for ep in range(n_episodes):
        env.unwrapped.variation_space["block"]["shape"].value = shape_idx
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        f = [env.render()]
        s = [obs["state"]]
        a = []
        for _ in range(100):
            act = weak_action(env, rng)
            obs, *_ = env.step(act)
            f.append(env.render())
            s.append(obs["state"])
            a.append(act)
        frames.append(np.stack(f)); states.append(np.stack(s)); actions.append(np.stack(a))
    env.close()
    return {"frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float64),
            "actions": np.stack(actions).astype(np.float32)}


def collect_wedge(n_episodes: int, seed: int, in_wedge: bool) -> dict:
    r"""Episodes whose INITIAL block angle is wedge-restricted (set via reset options.state)."""
    env = make_env(None)
    rng = np.random.default_rng(seed)
    frames, states, actions = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        s0 = np.asarray(obs["state"], dtype=np.float64)
        ang = (rng.uniform(0, np.pi / 2) if in_wedge
               else rng.uniform(np.pi / 2, 2 * np.pi))
        s0[4] = ang
        env.reset(options={"state": s0})
        f = [env.render()]
        s = [np.asarray(env.unwrapped._get_obs(), dtype=np.float64)]
        a = []
        for _ in range(100):
            act = weak_action(env, rng)
            obs, *_ = env.step(act)
            f.append(env.render())
            s.append(obs["state"])
            a.append(act)
        frames.append(np.stack(f)); states.append(np.stack(s)); actions.append(np.stack(a))
    env.close()
    return {"frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float64),
            "actions": np.stack(actions).astype(np.float32)}


def tensorize(ho: dict):
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).double()
    return fb, ach


def main() -> int:
    art: dict = {"blocks": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    # --- block 3 first (cheap, unblocks Tier 1): wedge data prep
    print("[wedge] corpora ...")
    for tag, n, inw, seed in (("wedge_corpus", 200, True, 0),
                              ("wedge_ho_in", 60, True, 1),
                              ("wedge_ho_out", 60, False, 1)):
        d = collect_wedge(n, seed, inw)
        np.savez_compressed(DATA_DIR / f"{tag}.npz", **d)
        print(f"  {tag}: {d['frames'].shape}")
    art["blocks"]["wedge_prep"] = "saved"
    save()

    # --- block 2: c5000
    print("[c5000] collecting ...")
    d = collect_weakpolicy(5000, seed=0)
    np.savez_compressed(DATA_DIR / "corpus_c5000.npz", **d)
    art["blocks"]["c5000"] = str(d["frames"].shape)
    del d
    save()

    # --- block 1: shape-generalization audits on existing stable Stage-B pairs
    print("[shapes] audits on ckpt7 pairs (CPU) ...")
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    pairs = {}
    for name, builder, runs in (("eq", build_eq, (0, 1, 3)), ("plain", lambda: build_plain(w_match), (0, 1, 2))):
        for r in runs:
            ck = torch.load(DATA_DIR / f"ckpt7_{'eqwin' if name=='eq' else 'plainwin'}_r{r}.pt",
                            map_location="cpu", weights_only=True)
            m = builder()
            m.load_state_dict(ck["model"])
            m.encoder.load_state_dict(ck["target_encoder"])
            pairs[f"{name}_r{r}"] = Pair(m.encoder.eval(), m.predictor)
    art["blocks"]["shape_transfer"] = {}
    # on-T reference held-out
    ref = collect_weakpolicy(30, seed=7)
    sets = {"T(ref)": ref}
    for sname, sidx in SHAPES.items():
        sets[sname] = collect_shape(30, seed=7, shape_idx=sidx)
    for sname, ho in sets.items():
        fb, ach = tensorize(ho)
        row = {}
        for pname, pr in pairs.items():
            rep = audit_gap(pr, fb, ach, window=16, k=3, burn_in=4, h_max=8, n_boot=200)
            meas = measured_err(pr, fb, ach, 8)
            dm = rep["delta"]["mean"]
            ratios = []
            for em in EPS_MULT:
                hm = boundary_from_curve(meas["q90"], em * dm)
                hc = boundary_from_curve(rep["certified_curve"]["err_q90"], em * dm)
                ratios.append((hc / hm) if hm else None)
            row[pname] = {"delta": round(dm, 3),
                          "guar_onesided": all(r is None or r <= 1.0 + 1e-9 for r in ratios),
                          "ratios": [round(r, 2) if r else None for r in ratios]}
        art["blocks"]["shape_transfer"][sname] = row
        print(f"  {sname}: " + "  ".join(f"{k} d={v['delta']}" for k, v in row.items()))
        save()

    print(f"CPU BATCH DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
