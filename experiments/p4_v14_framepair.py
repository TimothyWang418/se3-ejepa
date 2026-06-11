r"""P4 v1.4 — the frame-pair experiment: does the moat return when velocity is observable?

Spec: step1 seed spec, v1.4 amendment (registered 2026-06-11). The Stage-2κ finding: block
velocity is unobservable in single frames ⇒ non-Markov latent dynamics at κ>0 drown both bases;
normalized δ̂ INVERTED at κ=0.8 (eq 1.6× worse), confounded with eq's partial variance collapse.

**The registered question:** with velocity observable (adjacent-frame pair, 0.1 s apart,
channel-stacked — both frames rotate together so equivariance is trivially preserved) and recipe
stability checked, *which side of 1× does the normalized δ̂ ratio land on, per regime?*

Pre-declared outcome map (all three publishable, none requires loosening anything):
- ratio ≫ 1 both regimes → the moat was an observability artifact; equivariance wins both.
- ratio ≫ 1 at κ=0 only → the advantage is genuinely regime-scoped.
- ratio ≤ 1 at κ=0.8 still → momentum dynamics resist the equivariant constraint — the sharpest
  honest outcome; reported as such.

Cells: {eq, plain} × {κ=0, κ=0.8}, frame-pair input (6-ch), #9 fix on, κ=0 hyperparameters frozen
(fairness rule), registered normalizer (per-dim latent_std × √D), zsens + gap audit + measured
curves + G-pre at κ=0.8.

Run: .venv/bin/python experiments/p4_v14_framepair.py   (~12 min)
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

from experiments.p4_spine_stage1a import boundary_from_curve, fit_growth, measured_err  # noqa: E402
from experiments.p4_spine_stage2_kappa08 import Pair, zsens  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, circ_mask, collect_weakpolicy,
)
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.models.cn_regular import CNRegularPredictor  # noqa: E402
from src.models.eqjepa import ConvEncoder, EqJEPA, SteerableEncoder  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

SEED = 0
KAPPAS = (0.0, 0.8)
W_AUDIT, B_AUDIT, H_MAX = 16, 4, 8
EPS_MULT = (2, 4, 8, 16)
D = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_v14_framepair.json"


def build_eq6() -> EqJEPA:
    enc = SteerableEncoder(in_channels=6, latent_dim=D, n_rot=16, width=8)
    pred = CNRegularPredictor(latent_dim=D, action_dim=2 * CHUNK, n_rot=16, hidden_fields=32)
    return EqJEPA(latent_dim=D, action_dim=2 * CHUNK, encoder=enc, predictor=pred)


def build_plain6(width: int) -> EqJEPA:
    return EqJEPA(latent_dim=D, action_dim=2 * CHUNK, encoder=ConvEncoder(6, D, width=width))


def pick_w(target: int) -> int:
    cands = {w: sum(p.numel() for p in build_plain6(w).parameters()) for w in (8, 16, 24, 32, 48)}
    return min(cands, key=lambda w: abs(cands[w] - target))


def pairs_at(frames_u8: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    r"""(E, T+1, H, W, 3) uint8 → masked f32 pairs (E, len(idx), 6, H, W) at boundary indices,
    pairing each boundary frame with its PREVIOUS env frame (t−1; t=0 pairs with itself —
    velocities are genuinely zero at reset)."""
    f = torch.from_numpy(frames_u8).float().div_(255.0).permute(0, 1, 4, 2, 3)  # (E,T+1,3,H,W)
    prev = f[:, np.maximum(idx - 1, 0)]
    cur = f[:, idx]
    x = torch.cat([prev, cur], dim=2)  # (E, n, 6, H, W)
    return circ_mask(x.reshape(-1, 6, RES, RES)).reshape(x.shape)


def transitions_v14(data: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.from_numpy(data["actions"])
    n_ch = a.shape[1] // CHUNK
    starts = np.arange(0, n_ch * CHUNK, CHUNK)
    ends = starts + CHUNK
    obs = pairs_at(data["frames"], starts).reshape(-1, 6, RES, RES)
    nxt = pairs_at(data["frames"], ends).reshape(-1, 6, RES, RES)
    act = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).reshape(-1, CHUNK * 2)
    return obs, act, nxt


def main() -> int:
    t0 = time.time()
    art: dict = {"seed": SEED, "input": "frame-pair (t-1, t) channel-stacked, 6-ch",
                 "normalizer": "per-dim latent_std * sqrt(D)", "units": "per f-chunk"}
    w6 = pick_w(sum(p.numel() for p in build_eq6().parameters()))
    art["plain_width"] = w6

    for kappa in KAPPAS:
        ktag = f"k{kappa:g}"
        print(f"=== regime kappa={kappa} ===")
        corpus = collect_weakpolicy(200, seed=SEED, kappa=kappa if kappa > 0 else None)
        obs, act, nxt = transitions_v14(corpus)
        ho = collect_weakpolicy(60, seed=1, kappa=kappa if kappa > 0 else None)
        a = torch.from_numpy(ho["actions"])
        n_ch = a.shape[1] // CHUNK
        bidx = np.arange(0, n_ch * CHUNK + 1, CHUNK)
        fb = pairs_at(ho["frames"], bidx).double()
        ach = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()

        for name, builder in (("eq", build_eq6), ("plain", lambda: build_plain6(w6))):
            torch.manual_seed(SEED)
            m = builder()
            hist, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE,
                                   seed=SEED, verbose=False, return_target_encoder=True,
                                   refresh_target_cache=True)
            torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                       DATA_DIR / f"ckpt4_{name}_{ktag}.pt")
            pr = Pair(tgt.eval().cpu(), m.predictor.cpu())
            rep = audit_gap(pr, fb, ach, window=W_AUDIT, k=3, burn_in=B_AUDIT, h_max=H_MAX)
            meas = measured_err(pr, fb, ach, H_MAX)
            std = hist["latent_std"]
            d_norm = rep["delta"]["mean"] / (std * math.sqrt(D))
            gfit = fit_growth(meas["median"])
            lam_t = rep["lambda1"]["mean"]
            cell = {
                "pred_loss": hist["pred_loss"][-1], "latent_std": std,
                "delta_raw": rep["delta"]["mean"], "delta_norm": d_norm,
                "lambda1_tangent": {k: rep["lambda1"][k] for k in ("mean", "ci_lo", "ci_hi")},
                "zsens": zsens(pr.predictor),
                "measured_median": meas["median"], "growth_fit": gfit,
                "gpre_r": (gfit["lambda_meas_exp_fit"] / lam_t) if abs(lam_t) > 1e-6 else None,
                "c3cal_q90": [{
                    "eps_mult": em,
                    "H_cert": boundary_from_curve(rep["certified_curve"]["err_q90"], em * rep["delta"]["mean"]),
                    "H_meas": boundary_from_curve(meas["q90"], em * rep["delta"]["mean"]),
                } for em in EPS_MULT],
            }
            art[f"{name}@{ktag}"] = cell
            print(f"  {name}@{ktag}: pred {cell['pred_loss']:.4f} std {std:.3f} "
                  f"d_raw {cell['delta_raw']:.3f} d_norm {d_norm:.3f} "
                  f"lam_t {lam_t:+.4f} zsens {cell['zsens']['label']} r {cell['gpre_r']}")

    for ktag in ("k0", "k0.8"):
        r = art[f"plain@{ktag}"]["delta_norm"] / art[f"eq@{ktag}"]["delta_norm"]
        art[f"moat_ratio_{ktag}"] = r  # >1 means eq better (normalized)
        print(f"MOAT (normalized plain/eq) @ {ktag}: {r:.2f}x {'(eq better)' if r > 1 else '(eq worse)'}")

    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
