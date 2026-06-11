r"""P4-step1c — direct verification of the step1b '4-cluster collapse' diagnosis.

step1b INFERRED (from per-copy angle drawing + the loss/variance/content signature) that the aug
base's latents cluster by augmentation copy. This script verifies it directly: re-materialize
aug@20 (seed-deterministic), encode each of the 4 copies of the training set, and compute the
between-copy / within-copy variance ratio (ANOVA-style) plus a copy-identity linear probe.

Confirmed iff between/within is large (copy identity dominates the latent) AND copy identity is
linearly readable (~100% accuracy). Refuted again iff the ratio is O(1).

Run: .venv/bin/python experiments/p4_step1c_cluster_check.py  (~2 min, MPS)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_step1_pipeline import (  # noqa: E402
    DEVICE, RES, augment_x4, build_eq, build_plain, collect_weakpolicy, pick_ladder,
    to_transitions,
)
from src.training.jepa import train_jepa  # noqa: E402

SEED = 0
OUT = ROOT / "papers" / "figures" / "p4_step1c_cluster_check.json"


def main() -> int:
    t0 = time.time()
    data = collect_weakpolicy(20, seed=SEED)
    obs, act, nxt = to_transitions(data, 20)
    n = obs.shape[0]

    w = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    torch.manual_seed(SEED)
    model = build_plain(w)
    o4, a4, n4 = augment_x4(obs, act, nxt, seed=SEED)  # 4 copies, deterministic angles
    train_jepa(model, o4, a4, n4, epochs=5, batch_size=64, device=DEVICE, seed=SEED, verbose=False)
    model.eval().to(DEVICE)

    with torch.no_grad():
        z = model.encode(o4.to(DEVICE)).cpu()  # (4n, D), copies stacked in order
    copies = z.reshape(4, n, -1)
    mu_c = copies.mean(dim=1)                          # (4, D) per-copy means
    within = copies.var(dim=1, unbiased=False).mean()  # mean within-copy variance
    between = mu_c.var(dim=0, unbiased=False).mean()   # variance of copy means
    ratio = float(between / within)

    # copy-identity linear probe (multinomial logistic, closed-form-ish via torch)
    y = torch.arange(4).repeat_interleave(n)
    perm = torch.randperm(4 * n, generator=torch.Generator().manual_seed(1))
    tr, ev = perm[: 3 * n], perm[3 * n :]
    clf = torch.nn.Linear(z.shape[1], 4)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(z[tr]), y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((clf(z[ev]).argmax(1) == y[ev]).float().mean())

    art = {
        "seed": SEED, "n_per_copy": int(n),
        "between_within_variance_ratio": ratio,
        "copy_identity_probe_acc": acc,
        "verdict": "CONFIRMED" if (ratio > 10 and acc > 0.95) else
                   ("PARTIAL" if (ratio > 1 and acc > 0.5) else "REFUTED"),
        "wall_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(art, indent=1))
    print(f"between/within variance ratio: {ratio:.2f}   copy-identity probe acc: {acc:.3f}   "
          f"VERDICT: {art['verdict']}   ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
