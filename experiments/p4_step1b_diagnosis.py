r"""P4-step1b — two diagnoses before protocol v1.2 is registered.

(1) **aug shortcut diagnosis**: the seed-0 grid's aug base showed pred_loss → 0 with content-free
    latents. Hypothesis: the encoder encodes the AUGMENTATION ANGLE itself (o/o₂ share the angle ⇒
    it is the most predictable high-variance signal; JEPA's two losses are both satisfied by it).
    Test: probe each base's latent for an *applied known rotation angle* — if aug reads the angle
    (high $R^2_{\text{angle}}$) while reading no state (xy/θ ≈ 0), the diagnosis is confirmed.
    Control contrast: eq ALSO reads the global rotation (the regular rep contains the freq-1
    irrep — that is *by design*) but retains state content; the shortcut signature is
    angle-WITHOUT-content.

(2) **θ-unreadability discriminator**: G3 failed (θ linearly unreadable in every base). Competing
    hypotheses: (b) undertraining (20 epochs) vs (a) the WeakPolicy corpus barely rotates the
    block ⇒ JEPA never needs θ. Test: eq@20 retrained at 60 epochs — if θ-$R^2$ stays ≤ 0, (b) is
    dead and v1.2 goes corpus surgery (oracle frames, blocked on the G0c solver fix). Also logs
    the corpus' own within/across-episode θ variation (the direct check of (a)).

Bases re-materialized by seed determinism (collection prefix is bit-identical for the same seed;
training seeds identical; the overnight grid predates checkpointing).

Run: .venv/bin/python experiments/p4_step1b_diagnosis.py   (~10 min, MPS)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_step1_pipeline import (  # noqa: E402
    DEVICE, RES, Probe, augment_x4, build_eq, build_plain, collect_weakpolicy, fit_probe,
    pick_ladder, probe_targets, rotate_images, to_transitions,
)
from src.training.jepa import train_jepa  # noqa: E402

SEED = 0
N_TRAIN_EP = 20          # the @20 fraction (collection prefix identical to the grid's)
N_PROBE_EP = 6
OUT = ROOT / "papers" / "figures" / "p4_step1b_diagnosis.json"


def encode_eps(model, frames_np: np.ndarray) -> torch.Tensor:
    f = torch.from_numpy(frames_np).float().div_(255.0).permute(0, 1, 4, 2, 3)
    with torch.no_grad():
        z = model.encode(f.reshape(-1, 3, RES, RES).to(DEVICE)).cpu()
    return z.reshape(f.shape[0], f.shape[1], -1)


def state_probes(model, probe_data: dict) -> dict:
    z = encode_eps(model, probe_data["frames"])
    xy, cs, _ = probe_targets(probe_data["states"])
    zf = z.reshape(-1, z.shape[-1])
    n = zf.shape[0]
    tr = slice(0, int(0.7 * n))
    ev = slice(int(0.7 * n), n)
    return {
        "xy_r2": fit_probe(zf[tr], xy.reshape(-1, 2)[tr], zf[ev], xy.reshape(-1, 2)[ev]),
        "theta_r2": fit_probe(zf[tr], cs.reshape(-1, 2)[tr], zf[ev], cs.reshape(-1, 2)[ev]),
    }


def angle_probe(model, probe_data: dict, n_sample: int = 600, seed: int = 0) -> dict:
    r"""Probe the latent of ROTATED frames for the applied angle.

    Two target sets: $(\cos\theta, \sin\theta)$ (full angle) and $(\cos 4\theta, \sin 4\theta)$
    (angle mod $\pi/2$ — the zero-padding corner wedge's symmetry class: if the aug base collapsed
    onto the rotation's black-corner artifact, it reads mod-4 but not the full angle)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.from_numpy(probe_data["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    flat = f.reshape(-1, 3, RES, RES)
    idx = torch.randperm(flat.shape[0], generator=g)[:n_sample]
    zs, y1, y4 = [], [], []
    with torch.no_grad():
        for i in idx:
            th = float(torch.rand((), generator=g) * 2 * math.pi)
            rot = rotate_images(flat[i : i + 1], th)
            zs.append(model.encode(rot.to(DEVICE)).cpu().squeeze(0))
            y1.append(torch.tensor([math.cos(th), math.sin(th)]))
            y4.append(torch.tensor([math.cos(4 * th), math.sin(4 * th)]))
    z = torch.stack(zs)
    n = z.shape[0]
    tr, ev = slice(0, int(0.7 * n)), slice(int(0.7 * n), n)
    out = {}
    for tag, ys in (("angle_r2", torch.stack(y1)), ("angle4_r2", torch.stack(y4))):
        out[tag] = fit_probe(z[tr], ys[tr], z[ev], ys[ev])
    return out


def main() -> int:
    t0 = time.time()
    art: dict = {"seed": SEED, "n_train_ep": N_TRAIN_EP, "device": DEVICE}

    print("[1/4] collection (deterministic prefix) ...")
    data = collect_weakpolicy(N_TRAIN_EP + N_PROBE_EP, seed=SEED)
    corpus = {k: v[:N_TRAIN_EP] for k, v in data.items()}
    probe_data = {k: v[N_TRAIN_EP:] for k, v in data.items()}
    # direct check of hypothesis (a): how much does theta actually vary in the corpus?
    th = corpus["states"][..., 4]
    within = np.array([np.ptp(th[e]) for e in range(th.shape[0])])
    art["corpus_theta"] = {
        "within_episode_range_deg_median": float(np.degrees(np.median(within))),
        "within_episode_range_deg_q90": float(np.degrees(np.quantile(within, 0.9))),
        "across_episode_std_deg": float(np.degrees(th[:, 0].std())),
    }
    print(f"      theta variation: {art['corpus_theta']}")

    obs, act, nxt = to_transitions(corpus, N_TRAIN_EP)
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    print("[2/4] re-materialize bases (seed-deterministic) ...")
    bases = {}
    torch.manual_seed(SEED)
    m = build_plain(w_match)
    o, a, n = augment_x4(obs, act, nxt, seed=SEED)
    train_jepa(m, o, a, n, epochs=5, batch_size=64, device=DEVICE, seed=SEED, verbose=False)
    bases["aug@20"] = m
    torch.manual_seed(SEED)
    m = build_eq()
    train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE, seed=SEED, verbose=False)
    bases["eq@20/e20"] = m
    torch.manual_seed(SEED)
    m = build_plain(w_match)
    train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE, seed=SEED, verbose=False)
    bases["plain@20/e20"] = m
    print("[3/4] the 60-epoch discriminator (hypothesis b) ...")
    torch.manual_seed(SEED)
    m = build_eq()
    train_jepa(m, obs, act, nxt, epochs=60, batch_size=64, device=DEVICE, seed=SEED, verbose=False)
    bases["eq@20/e60"] = m

    print("[4/4] probes: state (xy, theta) + APPLIED-ANGLE ...")
    art["probes"] = {}
    for name, model in bases.items():
        model.eval().to(DEVICE)
        rep = state_probes(model, probe_data)
        rep.update(angle_probe(model, probe_data))
        art["probes"][name] = rep
        model.cpu()
        print(f"      {name:<14} xy {rep['xy_r2']:+.3f}  theta {rep['theta_r2']:+.3f}  "
              f"ANGLE {rep['angle_r2']:+.3f}  ANGLE4 {rep['angle4_r2']:+.3f}")

    aug, e60 = art["probes"]["aug@20"], art["probes"]["eq@20/e60"]
    art["diagnosis_aug_shortcut"] = bool(
        (aug["angle_r2"] > 0.5 or aug["angle4_r2"] > 0.5) and aug["xy_r2"] < 0.2
    )
    art["hypothesis_b_undertraining_alive"] = bool(e60["theta_r2"] > 0.0)
    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"aug-shortcut confirmed: {art['diagnosis_aug_shortcut']}   "
          f"undertraining-hypothesis alive: {art['hypothesis_b_undertraining_alive']}   "
          f"({art['wall_sec']}s) -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
