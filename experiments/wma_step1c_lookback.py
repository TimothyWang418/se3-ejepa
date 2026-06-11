r"""wma-step1c — LOOK-BACK pass over step1/1b (adversarial self-review, paper3 culture).

Three suspicions raised in review, three direct tests:

(i) **Run-to-run reproducibility** (MPS lesson from paper3, 2026-06-11: same-seed eq runs
    irreproducible on MPS). Our stack is CPU f64 encode + CPU f32 seeded probes — claimed
    deterministic. VERIFY, don't assume: recompute the width-256 cells end-to-end (fresh
    collection + fresh probes) and diff against the recorded step1 JSON.

(ii) **Distribution sensitivity, first-hand** (fix of the F2 overclaim): step1 compared our
    WeakPolicy numbers against ATM Table 6 — but their Table 6 checkpoints are their own
    retrained LeWM-style models (controlled-experiment hosts for AITS variants), NOT
    established to be the official HF ckpt; that comparison spans {ckpt provenance x data
    distribution x probe details}. Direct test instead: SAME official ckpt, SAME pipeline,
    TWO registered behavior policies (WeakPolicy vs uniform-random) — any material shift
    in readouts establishes distribution-dependence with zero reliance on the paper.

(iii) **Imprint control without the roll(1) asterisk**: step1b's permuted action a' = roll(1)
    mostly equals the SAME episode's c_{t-1} (dataset is episode-ordered), so "environment-
    independent" was inaccurate (the sanity cell D_TT(a')=1.001 rescued the conclusion).
    Redo with a seeded cross-episode derangement (no fixed points, predominantly cross-
    episode pairs) and report the cross-episode fraction.

Run: .venv/bin/python experiments/wma_step1c_lookback.py
Writes: papers/figures/wma_step1c_lookback.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from step91_lewm_audit import ADIM, CHUNK, DTYPE, HS, ZD, encode_seq, load_lewm  # noqa: E402
from wma_step1_atm_probe import EP_SEED0, SPLIT_SEED, TEST_FRAC_EP, collect_episode, grab  # noqa: E402
from wma_step1b_controls import predict_batch  # noqa: E402

from src.audit.atm import atm_audit  # noqa: E402

AUDIT_KW = dict(widths=(256,), main_width=256, probe_seeds=(0, 1, 2))   # width-256 lane
N_EP, N_STEP = 132, 40


def collect_random_episode(seed: int, n_model_steps: int):
    """Uniform-random U[-1,1]^2 behavior (the other registered policy), same plumbing."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    env = gym.make("swm/PushT-v1")
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames = [grab(obs, env)]
    chunks = []
    done = False
    for _ in range(n_model_steps):
        acts = []
        for _ in range(CHUNK):
            a = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
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
    return torch.stack(frames, 0), (np.stack(chunks, 0) if chunks
                                    else np.zeros((0, CHUNK * ADIM), dtype=np.float32)), done


def build_dataset(model, collector, scale01=True):
    Zt, Znext, Zwin, Awin, Atgt, ep_ids = [], [], [], [], [], []
    for ep in range(N_EP):
        frames, chunks, _ = collector(EP_SEED0 + ep, N_STEP)
        n = chunks.shape[0]
        if n < HS:
            continue
        with torch.no_grad():
            z = encode_seq(model, frames, scale01)
        for t in range(HS - 1, n):
            Zt.append(z[t]); Znext.append(z[t + 1]); Zwin.append(z[t - HS + 1:t + 1])
            Awin.append(chunks[t - HS + 1:t + 1]); Atgt.append(chunks[t]); ep_ids.append(ep)
    return (torch.stack(Zt), torch.stack(Znext), torch.stack(Zwin),
            torch.tensor(np.stack(Awin)), torch.tensor(np.stack(Atgt), dtype=torch.float32),
            np.asarray(ep_ids))


def ep_split(ep_ids: np.ndarray):
    eps_unique = np.unique(ep_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(eps_unique)
    n_test = max(1, int(round(TEST_FRAC_EP * eps_unique.shape[0])))
    mask = np.isin(ep_ids, eps_unique[:n_test].tolist())
    return torch.tensor(np.where(~mask)[0]), torch.tensor(np.where(mask)[0])


def main() -> int:
    t0 = time.time()
    model = load_lewm()
    out = {}

    # ---------- (i) + (ii) arm 1: WeakPolicy lane, fresh end-to-end ----------
    print("[wma1c] (i) reproducibility: fresh WeakPolicy lane ...", file=sys.stderr)
    Zt, Znext, Zwin, Awin, Atgt, ep_ids = build_dataset(model, collect_episode)
    tr, te = ep_split(ep_ids)
    Zpred = predict_batch(model, Zwin, Awin)
    res_weak = atm_audit(Zt, Znext, Zpred, Atgt, train_idx=tr, test_idx=te, **AUDIT_KW)

    rec = json.loads((ROOT / "papers/figures/wma_step1_atm_lewm.json").read_text())
    rec256 = rec["atm"]["per_width"]["256"]["median"]
    fresh = {f"D_{i}{j}": v for (i, j), v in res_weak.D.items()}
    max_rel = max(abs(fresh[k] - rec256[k]) / rec256[k] for k in fresh)
    out["repro"] = {"recorded_w256": rec256, "fresh_w256": fresh,
                    "max_rel_diff": max_rel,
                    "verdict": "REPRODUCIBLE (CPU lane)" if max_rel < 1e-6 else
                               f"NON-DETERMINISM at rel {max_rel:.2e} — investigate"}
    print(f"[wma1c] (i) max rel diff vs recorded = {max_rel:.2e} -> {out['repro']['verdict']}",
          file=sys.stderr)

    # ---------- (ii) arm 2: uniform-random lane, everything else frozen ----------
    print("[wma1c] (ii) uniform-random lane ...", file=sys.stderr)
    Zt2, Znext2, Zwin2, Awin2, Atgt2, ep_ids2 = build_dataset(model, collect_random_episode)
    tr2, te2 = ep_split(ep_ids2)
    Zpred2 = predict_batch(model, Zwin2, Awin2)
    res_rand = atm_audit(Zt2, Znext2, Zpred2, Atgt2, train_idx=tr2, test_idx=te2, **AUDIT_KW)
    out["distribution_sensitivity"] = {
        "weak": {"D_norm": {f"{i}{j}": v for (i, j), v in res_weak.D_norm.items()},
                 "readouts": res_weak.readouts, "action_var": res_weak.action_var,
                 "n": int(ep_ids.shape[0])},
        "random": {"D_norm": {f"{i}{j}": v for (i, j), v in res_rand.D_norm.items()},
                   "readouts": res_rand.readouts, "action_var": res_rand.action_var,
                   "n": int(ep_ids2.shape[0])},
        "note": "same official ckpt, same pipeline, only behavior policy differs "
                "(both policies registered in the step1 seed)"}
    print(f"[wma1c] (ii) weak:   Dn_TT={res_weak.D_norm[('T','T')]:.3f} "
          f"Dn_PP={res_weak.D_norm[('P','P')]:.3f} L_sym={res_weak.readouts['L_sym']:.2f}",
          file=sys.stderr)
    print(f"[wma1c] (ii) random: Dn_TT={res_rand.D_norm[('T','T')]:.3f} "
          f"Dn_PP={res_rand.D_norm[('P','P')]:.3f} L_sym={res_rand.readouts['L_sym']:.2f}",
          file=sys.stderr)

    # ---------- (iii) imprint with cross-episode derangement ----------
    print("[wma1c] (iii) imprint, cross-episode derangement ...", file=sys.stderr)
    N = Zt.shape[0]
    g = np.random.default_rng(2026)
    perm = g.permutation(N)
    fixed = np.where(perm == np.arange(N))[0]
    if fixed.size:                                       # repair fixed points by cyclic swap
        perm[fixed] = np.roll(perm[fixed], 1) if fixed.size > 1 else perm[(fixed + 1) % N]
    cross_ep_frac = float((ep_ids[perm] != ep_ids).mean())
    Aperm = Atgt[torch.tensor(perm)]
    Awin_p = Awin.clone()
    Awin_p[:, -1] = Aperm
    Zpred_p = predict_batch(model, Zwin, Awin_p)
    res_imp = atm_audit(Zt, Znext, Zpred_p, Aperm, train_idx=tr, test_idx=te, **AUDIT_KW)
    dn = res_imp.D_norm
    out["imprint_derangement"] = {
        "cross_episode_fraction": cross_ep_frac,
        "Dnorm_TT_sanity": dn[("T", "T")], "Dnorm_PP_imprint": dn[("P", "P")],
        "imprint_confirmed": bool(dn[("T", "T")] > 0.9 and dn[("P", "P")] < 0.7),
        "step1b_roll1_reference": {"Dnorm_TT": 1.001, "Dnorm_PP": 0.435,
                                   "caveat": "roll(1) a' mostly = same-episode c_{t-1}"}}
    print(f"[wma1c] (iii) cross-ep {cross_ep_frac:.2%}: Dn_TT(a')={dn[('T','T')]:.3f} "
          f"Dn_PP(a')={dn[('P','P')]:.3f} confirmed={out['imprint_derangement']['imprint_confirmed']}",
          file=sys.stderr)

    out["wall_seconds"] = round(time.time() - t0, 1)
    (ROOT / "papers/figures/wma_step1c_lookback.json").write_text(json.dumps(out, indent=2))
    print(f"[wma1c] wrote wma_step1c_lookback.json  wall={out['wall_seconds']}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
