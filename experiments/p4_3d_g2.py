r"""3D G2 — VN-JEPA first real training (stage-A health, n=6; protocol v1 + A1/A2).

Per 3D protocol spec: VNDGCNNEncoder(c_vec=32, k=16, width=64) → latent [vec(V)(96); I(96)];
VNPredictor; recipe transplant = banked family (EMA 0.99, var_coef 0.3, var_field_size=3
uniform — A2 disclosure, aux = tcp xyz coef 0.5); CHUNK=1 @20 Hz (stride binding pending);
train via the standard `train_jepa` harness (shape-agnostic row indexing; refresh flag N/A for
VN but kept ON harmlessly). Corpus: closed-loop replay shards (coverage-grade, A2). Held-out =
last shard. Stage-A health per A2: per-dim std AND field(3)-norm std (VN floor calibration
TBD on these very distributions), tcp-xyz probe R², δ̂ audit per stable run.

Run (box): nohup nice -n 5 /home/whb/se3-ejepa/.venv3/bin/python -u \
    /home/whb/se3-ejepa/experiments/p4_3d_g2.py > /home/whb/p4_3d_g2.log 2>&1 &
NOTE: .venv3 (torch cu124) — VN stack is plain PyTorch; mani_skill NOT needed for training.
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

from src.models.vn_jepa import VNDGCNNEncoder, VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

T0 = time.time()
DEVICE = os.environ.get("P4_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
N_RUNS = 6
RECIPE = dict(ema_decay=0.99, var_coef=0.3, aux_coef=0.5)
C_VEC, K_NN, WIDTH = 32, 16, 64
CORPUS_DIR = ROOT / "data" / "p4_3d"
OUT = ROOT / "papers" / "figures" / "p4_3d_g2.json"


class VNJEPA(nn.Module):
    r"""train_jepa-compatible wrapper: .encoder / .predictor over point clouds."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = VNDGCNNEncoder(c_vec=C_VEC, k=K_NN, width=WIDTH)
        self.predictor = VNPredictor(C_VEC, 3 * C_VEC, a_inv_dim=1)


def load_shards() -> tuple[list, list, list]:
    clouds, acts, tcps = [], [], []
    for p in sorted(CORPUS_DIR.glob("corpus3d_*.npz")):
        z = np.load(p)
        n = int(z["n"])
        for i in range(n):
            clouds.append(z[f"c{i}"]); acts.append(z[f"a{i}"]); tcps.append(z[f"t{i}"])
    return clouds, acts, tcps


def transitions(clouds: list, acts: list, tcps: list):
    o_l, a_l, n_l, x_l = [], [], [], []
    for c, a, t_ in zip(clouds, acts, tcps):
        o_l.append(torch.from_numpy(c[:-1]))
        n_l.append(torch.from_numpy(c[1:]))
        a_l.append(torch.from_numpy(a))
        x_l.append(torch.from_numpy(t_[:-1, :3]).float())
    return (torch.cat(o_l), torch.cat(a_l), torch.cat(n_l), torch.cat(x_l))


def field_norm_std(z: torch.Tensor) -> float:
    r"""Per-field(3) norm std, mean over fields — the A2 VN-floor candidate metric."""
    f = z.reshape(z.shape[0], -1, 3)
    return float(f.norm(dim=-1).std(dim=0).mean())


def probe_r2(z: torch.Tensor, y: torch.Tensor) -> float:
    zc = (z - z.mean(0)).double(); yc = (y - y.mean(0)).double()
    w = torch.linalg.lstsq(zc.T @ zc + 1e-3 * torch.eye(zc.shape[1], dtype=torch.float64),
                           zc.T @ yc).solution
    return float((1 - ((zc @ w - yc).pow(2).sum(0) / yc.pow(2).sum(0))).mean())


def main() -> int:
    art: dict = {"recipe": str(RECIPE), "arch": f"c_vec={C_VEC},k={K_NN},w={WIDTH}",
                 "device": DEVICE, "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    clouds, acts, tcps = load_shards()
    n_ho = max(1, len(clouds) // 8)
    print(f"[G2] {len(clouds)} eps loaded; held-out {n_ho}")
    obs, act, nxt, aux_t = transitions(clouds[:-n_ho], acts[:-n_ho], tcps[:-n_ho])
    obs_h, act_h, nxt_h, tcp_h = transitions(clouds[-n_ho:], acts[-n_ho:], tcps[-n_ho:])
    art["n_transitions"] = int(obs.shape[0])
    save()

    for r in range(N_RUNS):
        try:
            torch.manual_seed(r)
            m = VNJEPA()
            t0 = time.time()
            hist, tgt = train_jepa(
                m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE, seed=r,
                verbose=False, return_target_encoder=True, refresh_target_cache=True,
                ema_decay=RECIPE["ema_decay"], var_coef=RECIPE["var_coef"],
                var_field_size=3, aux_state=(aux_t, RECIPE["aux_coef"]))
            tgt = tgt.eval().to("cpu")
            with torch.no_grad():
                z = torch.cat([tgt(obs_h[i:i + 64]) for i in range(0, len(obs_h), 64)])
            cell = {"std": round(float(z.std(dim=0).mean()), 3),
                    "field_norm_std": round(field_norm_std(z), 3),
                    "tcp_r2": round(probe_r2(z, tcp_h), 3),
                    "pred_loss_final": round(float(hist["pred_loss"][-1]), 5),
                    "train_min": round((time.time() - t0) / 60, 1)}
            cell["stable_provisional"] = bool(cell["std"] >= 0.7)
            torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                       CORPUS_DIR / f"ckpt3d_g2_r{r}.pt")
        except Exception as exc:  # noqa: BLE001
            cell = {"error": str(exc)[:300], "stable_provisional": False}
        art["cells"][f"r{r}"] = cell
        save()
        print(f"  r{r}: std {cell.get('std')} fns {cell.get('field_norm_std')} "
              f"tcpR2 {cell.get('tcp_r2')} loss {cell.get('pred_loss_final')} "
              f"({cell.get('train_min')}min) {cell.get('error', '')[:60]}")

    ok = [c for c in art["cells"].values() if c.get("stable_provisional")]
    art["stage_A"] = {"stable_provisional": f"{len(ok)}/{N_RUNS}",
                      "tcp_r2": sorted(round(c["tcp_r2"], 3) for c in ok if "tcp_r2" in c),
                      "note": "VN floor calibration TBD on these distributions (A2)"}
    save()
    print("STAGE A:", art["stage_A"])
    print(f"G2 DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
