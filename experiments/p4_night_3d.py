r"""3D night batch (3080) — autonomous, crash-immune. Executes manifest Tier-0 training blocks
after the current G2 fleet frees the GPU. Health metrics only (stage-A); audits run separately
on CPU. Every block writes its own artifact incrementally.

Queue (numbered):
  #5  G2 plain baseline (PointNet, n=6) — the 3D moat's other side (T0.2)
  #6  3D data-scaling VN: {3,6,9,11 shards}×n=4 — does the low-N moat replicate in 3D? (T0.3)
  #7  3D data-scaling plain: same grid — 3D reliability cliff comparison (T0.4)

Param-match: PointNet width chosen so its param count ≈ VNDGCNNEncoder(c_vec=32,k=16,width=64).
Run (box worktree): PYTHONPATH=. P4_DEVICE=cuda setsid .venv3/bin/python -u \
    experiments/p4_night_3d.py > ~/night_3d.log 2>&1 &
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.plain_point import PlainPredictor, PointNetEncoder  # noqa: E402
from src.models.vn_jepa import VNDGCNNEncoder, VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

T0 = time.time()
DEVICE = os.environ.get("P4_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_DIR = ROOT / "data" / "p4_3d"
OUT = ROOT / "papers" / "figures" / "p4_night_3d.json"
C_VEC, K_NN, WIDTH = 32, 16, 64
RECIPE = dict(ema_decay=0.99, var_coef=0.3, aux_coef=0.5)


class VNJEPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VNDGCNNEncoder(c_vec=C_VEC, k=K_NN, width=WIDTH)
        self.predictor = VNPredictor(C_VEC, 3 * C_VEC, a_inv_dim=1)
        self.latent_dim = self.encoder.latent_dim


class PlainJEPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_vn = sum(p.numel() for p in VNJEPA().encoder.parameters())
        w = 256
        while sum(p.numel() for p in PointNetEncoder(192, w).parameters()) < n_vn and w < 700:
            w += 16                                   # match VN encoder param count from below
        self.encoder = PointNetEncoder(192, w)
        self.predictor = PlainPredictor(192, 7, 256)
        self.latent_dim = 192
        self.width = w


def load_shards(k: int | None = None) -> tuple[list, list, list]:
    clouds, acts, tcps = [], [], []
    shards = sorted(CORPUS_DIR.glob("corpus3d_*.npz"))
    if k:
        shards = shards[:k]
    for p in shards:
        z = np.load(p)
        for i in range(int(z["n"])):
            clouds.append(z[f"c{i}"]); acts.append(z[f"a{i}"]); tcps.append(z[f"t{i}"])
    return clouds, acts, tcps


def transitions(clouds, acts, tcps):
    o, a, n, x = [], [], [], []
    for c, ac, t_ in zip(clouds, acts, tcps):
        o.append(torch.from_numpy(c[:-1])); n.append(torch.from_numpy(c[1:]))
        a.append(torch.from_numpy(ac)); x.append(torch.from_numpy(t_[:-1, :3]).float())
    return torch.cat(o), torch.cat(a), torch.cat(n), torch.cat(x)


def probe_r2(z, y):
    zc = (z - z.mean(0)).double(); yc = (y - y.mean(0)).double()
    w = torch.linalg.lstsq(zc.T @ zc + 1e-3 * torch.eye(zc.shape[1], dtype=torch.float64),
                           zc.T @ yc).solution
    return float((1 - ((zc @ w - yc).pow(2).sum(0) / yc.pow(2).sum(0))).mean())


def train_one(make_model, obs, act, nxt, aux_t, obs_h, tcp_h, seed):
    torch.manual_seed(seed)
    m = make_model()
    hist, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE,
                           seed=seed, verbose=False, return_target_encoder=True,
                           refresh_target_cache=True, ema_decay=RECIPE["ema_decay"],
                           var_coef=RECIPE["var_coef"], var_field_size=3,
                           aux_state=(aux_t, RECIPE["aux_coef"]))
    tgt = tgt.eval().to("cpu")
    with torch.no_grad():
        z = torch.cat([tgt(obs_h[i:i + 64]) for i in range(0, len(obs_h), 64)])
    return {"std": round(float(z.std(0).mean()), 3),
            "tcp_r2": round(probe_r2(z, tcp_h), 3),
            "final_loss": round(float(hist["pred_loss"][-1]), 5),
            "stable": bool(z.std(0).mean() >= 0.7)}


def main() -> int:
    art: dict = {"queue": "#5 plain baseline / #6 VN data-scale / #7 plain data-scale",
                 "device": DEVICE, "blocks": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    # wait for the G2 fleet to free the GPU (up to 3h guard)
    print("[wait] G2 fleet GPU handover ...")
    for _ in range(360):
        try:
            import subprocess
            n = subprocess.run(["pgrep", "-fc", "p4_3d_g2"], capture_output=True, text=True)
            if int(n.stdout.strip() or 0) == 0:
                break
        except Exception:  # noqa: BLE001
            break
        time.sleep(30)
    print(f"[wait] done at {(time.time()-T0)/60:.0f} min")

    full = load_shards()
    obs, act, nxt, aux_t = transitions(full[0][:-int(len(full[0]) / 8)], full[1][:-int(len(full[1]) / 8)], full[2][:-int(len(full[2]) / 8)])
    n_ho = max(1, len(full[0]) // 8)
    obs_h, _, _, tcp_h = transitions(full[0][-n_ho:], full[1][-n_ho:], full[2][-n_ho:])

    # #5 plain baseline n=6
    print("[#5] plain baseline ...")
    art["blocks"]["plain_baseline"] = {"width": PlainJEPA().width, "cells": []}
    for r in range(6):
        try:
            c = train_one(PlainJEPA, obs, act, nxt, aux_t, obs_h, tcp_h, r)
        except Exception as exc:  # noqa: BLE001
            c = {"error": str(exc)[:160], "stable": False}
        art["blocks"]["plain_baseline"]["cells"].append(c)
        print(f"  plain r{r}: {c}"); save()

    # #6/#7 data-scaling (shard counts -> episode counts)
    for tag, make in (("vn_datascale", VNJEPA), ("plain_datascale", PlainJEPA)):
        print(f"[#6/7] {tag} ...")
        art["blocks"][tag] = {}
        for k in (3, 6, 9, 11):
            cl, ac, tc = load_shards(k)
            nh = max(1, len(cl) // 8)
            o, a, n, x = transitions(cl[:-nh], ac[:-nh], tc[:-nh])
            oh, _, _, th = transitions(cl[-nh:], ac[-nh:], tc[-nh:])
            cells = []
            for r in range(4):
                try:
                    c = train_one(make, o, a, n, x, oh, th, r)
                except Exception as exc:  # noqa: BLE001
                    c = {"error": str(exc)[:160], "stable": False}
                cells.append(c)
            art["blocks"][tag][f"k{k}"] = cells
            print(f"  {tag} k{k}: stable {sum(c.get('stable', False) for c in cells)}/4"); save()

    print(f"NIGHT 3D DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
