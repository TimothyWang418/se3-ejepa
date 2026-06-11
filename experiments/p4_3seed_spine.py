r"""P4 3-seed — the conversion blade: seed-0 shapes become (or fail to become) claims.

Banked configuration (proposal §7.5): single-frame, κ=0, protocol v1.2, #9 fix on. Train seeds
{0,1,2} × {eq, plain_match} on the FIXED corpus (collection seed 0); audits on the FIXED held-out
set (collection seed 1). **Protocol freeze in force: no new knobs, no new diagnostics — this run
only evaluates already-registered gates.**

Gates evaluated (first real evaluation, spine spec as amended):
- **C3-cal-static (quantile-matched):** $H^{q90}_{\text{cert}}(\epsilon) / H^{q90}_{\text{meas}}
  (\epsilon) \in [0.5, 2]$ across the ε grid {2,4,8,16}·δ̂, ≥2/3 seeds, per base. (Stage-1b
  showed planner ≥ model everywhere at κ=0 ⇒ the whole grid is model-limited; stated.)
- **G-pre shape (neutral regime):** measured median curve fits linear better than exponential
  (r²_lin > r²_exp) AND certified linear rate conservative within [0.5,2]× of measured rate,
  ≥2/3 seeds (eq base; plain reported).
- **Moat (descriptive headline):** normalized δ̂ ratio plain/eq across seeds, mean ± range.

Run: .venv/bin/python experiments/p4_3seed_spine.py   (~15 min)
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
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder,
    to_transitions,
)
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

SEEDS = (0, 1, 2)
W_AUDIT, B_AUDIT, H_MAX = 16, 4, 8
EPS_MULT = (2, 4, 8, 16)
D = 128
BAND = (0.5, 2.0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_3seed_spine.json"


def in_band(x: float) -> bool:
    return BAND[0] <= x <= BAND[1]


def main() -> int:
    t0 = time.time()
    art: dict = {"seeds": list(SEEDS), "config": "banked: single-frame kappa=0, v1.2, #9 fix",
                 "band": list(BAND), "units": "per f-chunk"}

    print("[1/3] fixed corpus + held-out ...")
    corpus = collect_weakpolicy(200, seed=0)
    obs, act, nxt = to_transitions(corpus, 200)
    ho = collect_weakpolicy(60, seed=1)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]

    print("[2/3] 6 cells: train + audit per (seed, base) ...")
    cells: dict = {}
    for seed in SEEDS:
        for name, builder in (("eq", build_eq), ("plain", lambda: build_plain(w_match))):
            torch.manual_seed(seed)
            m = builder()
            hist, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE,
                                   seed=seed, verbose=False, return_target_encoder=True,
                                   refresh_target_cache=True)
            torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                       DATA_DIR / f"ckpt5_{name}_s{seed}.pt")
            pr = Pair(tgt.eval().cpu(), m.predictor.cpu())
            rep = audit_gap(pr, fb, ach, window=W_AUDIT, k=3, burn_in=B_AUDIT, h_max=H_MAX)
            meas = measured_err(pr, fb, ach, H_MAX)
            d_mean = rep["delta"]["mean"]
            gfit = fit_growth(meas["median"])
            cal = []
            for em in EPS_MULT:
                eps = em * d_mean
                hc = boundary_from_curve(rep["certified_curve"]["err_q90"], eps)
                hm = boundary_from_curve(meas["q90"], eps)
                cal.append({"eps_mult": em, "H_cert": hc, "H_meas": hm,
                            "ratio": (hc / hm) if hm else None})
            cells[f"{name}_s{seed}"] = {
                "pred_loss": hist["pred_loss"][-1], "latent_std": hist["latent_std"],
                "delta_mean": d_mean, "delta_q90": rep["delta"]["q90"],
                "delta_norm": d_mean / (max(hist["latent_std"], 1e-6) * math.sqrt(D)),
                "lambda_tangent": rep["lambda1"]["mean"],
                "measured_median": meas["median"],
                "growth": {k: gfit[k] for k in ("lambda_meas_exp_fit", "linear_rate",
                                                "r2_exp", "r2_linear")},
                "rate_ratio_cert_over_meas": (d_mean / gfit["linear_rate"])
                if gfit["linear_rate"] > 1e-9 else None,
                "c3cal": cal,
            }
            c = cells[f"{name}_s{seed}"]
            print(f"    {name} s{seed}: std {c['latent_std']:.3f} d {d_mean:.3f} "
                  f"d_norm {c['delta_norm']:.3f} lin_rate {gfit['linear_rate']:.3f} "
                  f"r2lin {gfit['r2_linear']:.3f}/r2exp {gfit['r2_exp']:.3f} "
                  f"cal {[x['ratio'] for x in cal]}")
    art["cells"] = cells

    print("[3/3] gate aggregation ...")
    verdicts: dict = {}
    for name in ("eq", "plain"):
        # C3-cal: all eps ratios in band, per seed; pass if >=2/3 seeds
        seed_pass = []
        for s in SEEDS:
            ratios = [x["ratio"] for x in cells[f"{name}_s{s}"]["c3cal"]]
            seed_pass.append(all(r is not None and in_band(r) for r in ratios))
        verdicts[f"c3cal_static_{name}"] = {"per_seed": seed_pass,
                                            "PASS": sum(seed_pass) >= 2}
        # G-pre shape: linear beats exp AND certified rate conservative in band
        shape_pass = []
        for s in SEEDS:
            g = cells[f"{name}_s{s}"]["growth"]
            rr = cells[f"{name}_s{s}"]["rate_ratio_cert_over_meas"]
            shape_pass.append(bool(g["r2_linear"] > g["r2_exp"] and rr and in_band(rr)))
        verdicts[f"gpre_shape_{name}"] = {"per_seed": shape_pass, "PASS": sum(shape_pass) >= 2}
        # stability floor
        verdicts[f"stability_{name}"] = {
            "per_seed": [cells[f"{name}_s{s}"]["latent_std"] >= 0.7 for s in SEEDS]}
    moat = [cells[f"plain_s{s}"]["delta_norm"] / cells[f"eq_s{s}"]["delta_norm"] for s in SEEDS]
    verdicts["moat_normalized_plain_over_eq"] = {
        "per_seed": [round(m, 2) for m in moat],
        "mean": round(float(np.mean(moat)), 2),
        "range": [round(min(moat), 2), round(max(moat), 2)],
    }
    art["verdicts"] = verdicts
    for k, v in verdicts.items():
        print(f"    {k}: {v}")

    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
