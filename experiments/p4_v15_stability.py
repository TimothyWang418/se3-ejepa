r"""P4 v1.5 — the stability pass: isotypic-aware variance floor vs Step-64 gating, on the worst
collapse cell (eq/6-ch frame-pair @ κ=0, std 0.307 under v1.4).

Spec: step1 seed spec, v1.5 block (H-v1.5a isotropy conflict / H-v1.5b gated-var). Sweep
{per_dim, per_field(16)} × {gated off, on}; **gate: floor-stat ≥ 0.7**; content pulse = xy probe
(a stable-but-empty latent is no good — the aug-collapse lesson). If a non-baseline recipe wins,
the v1.4 four-cell moat question re-runs under it in the same artifact.

Loss-compatibility assert (run first): the per-field floor statistic is invariant under the fiber
permutation action (roll) — the variance term cannot break equivariance.

Run: .venv/bin/python experiments/p4_v15_stability.py   (~15-20 min)
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
    CHUNK, DATA_DIR, RES, Probe, circ_mask, collect_weakpolicy, fit_probe, probe_targets,
)
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_v14_framepair import build_eq6, build_plain6, pairs_at, pick_w, transitions_v14  # noqa: E402
from src.audit.gap_mode import one_step_bias  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

SEED = 0
N_ROT = 16
GATE_FLOOR = 0.7
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_v15_stability.json"

RECIPES = [
    {"tag": "per_dim",        "var_field_size": 0,     "predictability_gated_var": False},
    {"tag": "per_dim+gated",  "var_field_size": 0,     "predictability_gated_var": True},
    {"tag": "per_field",      "var_field_size": N_ROT, "predictability_gated_var": False},
    {"tag": "per_field+gated","var_field_size": N_ROT, "predictability_gated_var": True},
]


def floor_invariance_assert() -> float:
    r"""Per-field floor statistic invariant under fiber roll (the rho action)."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 128, generator=g)
    def stat(zz):
        zf = zz.view(64, 128 // N_ROT, N_ROT)
        return (zf - zf.mean(0, keepdim=True)).pow(2).mean(0).mean(-1).clamp_min(1e-12).sqrt()
    base = stat(z)
    worst = 0.0
    for s in range(1, N_ROT):
        zr = torch.roll(z.view(64, -1, N_ROT), shifts=s, dims=-1).reshape(64, 128)
        worst = max(worst, float((stat(zr) - base).abs().max()))
    assert worst < 1e-6, f"per-field floor not roll-invariant: {worst:.2e}"
    return worst


def xy_probe_pairs(enc, ho: dict, bidx: np.ndarray) -> float:
    r"""Content pulse: block-(x,y) probe on boundary PAIR latents (target encoder)."""
    fb = pairs_at(ho["frames"], bidx).float()
    E, T = fb.shape[0], fb.shape[1]
    with torch.no_grad():
        z = enc.eval().to(DEVICE)(fb.reshape(-1, 6, RES, RES).to(DEVICE)).cpu()
    enc.cpu()
    z = z.reshape(E, T, -1)
    xy, _, _ = probe_targets(ho["states"][:, bidx])
    zf, yf = z.reshape(-1, z.shape[-1]), xy.reshape(-1, 2)
    n = zf.shape[0]
    tr, ev = slice(0, int(0.7 * n)), slice(int(0.7 * n), n)
    return fit_probe(zf[tr], yf[tr], zf[ev], yf[ev])


def run_cell(builder, recipe: dict, obs, act, nxt, ho, bidx, fb_d, ach_d) -> dict:
    torch.manual_seed(SEED)
    m = builder()
    hist, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE, seed=SEED,
                           verbose=False, return_target_encoder=True, refresh_target_cache=True,
                           var_field_size=recipe["var_field_size"],
                           predictability_gated_var=recipe["predictability_gated_var"])
    pr = Pair(tgt.eval().cpu(), m.predictor.cpu())
    delta = float(one_step_bias(pr, fb_d, ach_d).mean())
    std, fstat = hist["latent_std"], hist["latent_floor_stat"]
    norm = delta / (max(std, 1e-6) * math.sqrt(128))
    return {"pred_loss": hist["pred_loss"][-1], "latent_std": std, "floor_stat": fstat,
            "delta_raw": delta, "delta_norm": norm,
            "xy_r2": xy_probe_pairs(tgt, ho, bidx),
            "pass_floor": bool(fstat >= GATE_FLOOR)}


def main() -> int:
    t0 = time.time()
    art: dict = {"seed": SEED, "gate_floor": GATE_FLOOR,
                 "floor_invariance_max_err": floor_invariance_assert()}
    print(f"[0] per-field floor roll-invariance: {art['floor_invariance_max_err']:.2e}  PASS")

    print("[1] data (kappa=0, frame pairs) ...")
    corpus = collect_weakpolicy(200, seed=SEED)
    obs, act, nxt = transitions_v14(corpus)
    ho = collect_weakpolicy(60, seed=1)
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    bidx = np.arange(0, n_ch * CHUNK + 1, CHUNK)
    fb_d = pairs_at(ho["frames"], bidx).double()
    ach_d = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()

    print("[2] sweep on the worst cell (eq/6ch@k0) ...")
    art["sweep"] = {}
    for r in RECIPES:
        cell = run_cell(build_eq6, r, obs, act, nxt, ho, bidx, fb_d, ach_d)
        art["sweep"][r["tag"]] = cell
        print(f"    {r['tag']:<16} std {cell['latent_std']:.3f} floor {cell['floor_stat']:.3f} "
              f"pred {cell['pred_loss']:.4f} d_norm {cell['delta_norm']:.3f} "
              f"xy {cell['xy_r2']:+.3f} {'PASS' if cell['pass_floor'] else 'fail'}")

    qual = {t: c for t, c in art["sweep"].items() if c["pass_floor"]}
    if qual:
        winner = max(qual, key=lambda t: qual[t]["xy_r2"])
    else:
        winner = max(art["sweep"], key=lambda t: art["sweep"][t]["floor_stat"])
    art["winner"] = winner
    print(f"[3] winner: {winner}")

    if winner != "per_dim":
        print("[4] re-ask the v1.4 moat question under the winner (4 cells) ...")
        wr = next(r for r in RECIPES if r["tag"] == winner)
        w6 = pick_w(sum(p.numel() for p in build_eq6().parameters()))
        art["moat_v15"] = {}
        for kappa in (0.0, 0.8):
            kk = kappa if kappa > 0 else None
            corpus_k = corpus if kappa == 0.0 else collect_weakpolicy(200, seed=SEED, kappa=kk)
            ok, ak, nk = (obs, act, nxt) if kappa == 0.0 else transitions_v14(corpus_k)
            ho_k = ho if kappa == 0.0 else collect_weakpolicy(60, seed=1, kappa=kk)
            a_k = torch.from_numpy(ho_k["actions"])
            fb_k = pairs_at(ho_k["frames"], bidx).double()
            ach_k = a_k[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
            for name, builder in (("eq", build_eq6), ("plain", lambda: build_plain6(w6))):
                cell = run_cell(builder, wr, ok, ak, nk, ho_k, bidx, fb_k, ach_k)
                art["moat_v15"][f"{name}@k{kappa:g}"] = cell
                print(f"    {name}@k{kappa:g}: std {cell['latent_std']:.3f} floor {cell['floor_stat']:.3f} "
                      f"d_norm {cell['delta_norm']:.3f} xy {cell['xy_r2']:+.3f} "
                      f"{'PASS' if cell['pass_floor'] else 'fail'}")
        for kt in ("k0", "k0.8"):
            e, p = art["moat_v15"][f"eq@{kt}"], art["moat_v15"][f"plain@{kt}"]
            art[f"moat_ratio_{kt}"] = p["delta_norm"] / e["delta_norm"]
            print(f"    MOAT (norm plain/eq) @{kt}: {art[f'moat_ratio_{kt}']:.2f}x "
                  f"(floors: eq {'OK' if e['pass_floor'] else 'FAIL'}, plain {'OK' if p['pass_floor'] else 'FAIL'})")

    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
