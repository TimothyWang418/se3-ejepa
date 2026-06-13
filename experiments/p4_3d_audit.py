r"""3D certificate-side audit (C3-3D) — δ̂ + λ̂₁ + certified curve + faithful gate cells, on the
point-cloud VN pair vs the plain baseline. The 3D counterpart of the 2D gap-mode audit (which
takes image frames; this one takes latents from a cloud encoder, reusing window_exponent's JVP).

Per stable pair: encode held-out cloud transitions → latents (f64); δ̂ = median one-step latent
error ‖pred(z_t,a_t) − z_{t+1}‖; λ̂₁ via window_exponent (Benettin JVP, encoder-agnostic) on
motion windows; certified curve Err(H) = δ̂·Σ_{t<H} e^{λ̂₁ t}; faithful boundary cells via the
canonical src/audit/gates.py (H_cert ≤ H_meas one-sided guarantee). ε grid = {2,4,8,16}×δ̂.

VN n=3 + plain n=3 (banked recipe), 200-eps corpus. Saves ckpts (the night batch didn't).
Run (box tmux): cd /home/whb/p4_wt && PYTHONPATH=. P4_DEVICE=cuda python3 -u \
    experiments/p4_3d_audit.py > /home/whb/audit3d.log 2>&1
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

from experiments.p4_night_3d import (  # noqa: E402
    PlainJEPA, VNJEPA, load_shards, transitions,
)
from src.audit.gap_mode import window_exponent  # noqa: E402
from src.audit.gates import dual_boundary_cells, faithful_guar, violations  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

T0 = time.time()
DEVICE = os.environ.get("P4_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
EPS_MULT = (2, 4, 8, 16)
H_MAX = 8
OUT = ROOT / "papers" / "figures" / "p4_3d_audit.json"
CKPT = ROOT / "data" / "p4_3d"
RECIPE = dict(ema_decay=0.99, var_coef=0.3, aux_coef=0.5)


def boundary(curve, eps):
    h = 0
    for i, v in enumerate(curve):
        if v <= eps:
            h = i + 1
        else:
            break
    return h


def audit_pair(enc, pred, ep_clouds, ep_acts):
    r"""δ̂ + λ̂₁ + REAL multi-step measured curve on episode-structured held-out clouds (f64).

    ep_clouds: list of (T+1, N, 3) per episode; ep_acts: list of (T, A). Measured H-step error =
    ‖rollout_H(z_t, a_{t:t+H}) − z_{t+H}‖ (true H-ahead latent), the honest measured side.
    """
    enc = enc.double().eval()
    pred = copy.deepcopy(pred).double().eval()
    for p in pred.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def encode(x):
        x = torch.from_numpy(x).double() if isinstance(x, np.ndarray) else x.double()
        return torch.cat([enc(x[i:i + 64]) for i in range(0, len(x), 64)])

    with torch.no_grad():
        zseq = [encode(c) for c in ep_clouds]                # list of (T+1, D)
        aseq = [torch.from_numpy(a).double() if isinstance(a, np.ndarray) else a.double()
                for a in ep_acts]
        # δ̂: one-step error pooled over all episodes
        step = []
        for z, a in zip(zseq, aseq):
            zp = pred(z[:-1], a)
            step.append((zp - z[1:]).norm(dim=1))
        step_err = torch.cat(step)
        delta = {"mean": float(step_err.mean()), "q50": float(step_err.median()),
                 "q90": float(step_err.quantile(0.9))}
        # measured H-step: roll out and compare to true z_{t+H}
        meas = {H: [] for H in range(1, H_MAX + 1)}
        for z, a in zip(zseq, aseq):
            T = a.shape[0]
            for t in range(T):
                zc = z[t:t + 1]
                for H in range(1, min(H_MAX, T - t) + 1):
                    zc = pred(zc, a[t + H - 1:t + H])
                    meas[H].append((zc - z[t + H:t + H + 1]).norm().item())
        meas_q90 = [float(np.quantile(meas[H], 0.9)) if meas[H] else float("inf")
                    for H in range(1, H_MAX + 1)]

    # λ̂₁: Benettin JVP on the latent sequences (motion windows)
    def pred_fn(zz, aa):
        return pred(zz, aa)
    lams = []
    for z, a in zip(zseq, aseq):
        if a.shape[0] >= 8:
            for s in range(0, a.shape[0] - 8, 4):
                try:
                    lams.append(window_exponent(pred_fn, z[s], a[s:s + 8], k=3, burn_in=2))
                except Exception:  # noqa: BLE001
                    pass
    lam1 = float(np.median(lams)) if lams else 0.0
    cert_q90 = [float(delta["q90"] * sum(np.exp(lam1 * t) for t in range(H)))
                for H in range(1, H_MAX + 1)]
    cells = dual_boundary_cells(meas_q90, cert_q90, delta["mean"], EPS_MULT)
    return {"delta": {k: round(v, 3) for k, v in delta.items()}, "lambda1": round(lam1, 4),
            "n_lam": len(lams), "cells": cells, "guar": faithful_guar(cells),
            "viol": len(violations(cells))}


def main() -> int:
    art: dict = {"design": "3D C3 certificate audit (VN vs plain), faithful gates", "pairs": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    cl, ac, tc = load_shards(8)                              # 200 eps
    nh = max(1, len(cl) // 8)
    obs, act, nxt, aux = transitions(cl[:-nh], ac[:-nh], tc[:-nh])
    ep_clouds_h, ep_acts_h = cl[-nh:], ac[-nh:]              # episode-structured held-out

    for name, make in (("vn", VNJEPA), ("plain", PlainJEPA)):
        art["pairs"][name] = []
        for r in range(3):
            try:
                torch.manual_seed(r)
                m = make()
                _, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=48, device=DEVICE,
                                    seed=r, verbose=False, return_target_encoder=True,
                                    refresh_target_cache=True, ema_decay=RECIPE["ema_decay"],
                                    var_coef=RECIPE["var_coef"], var_field_size=3,
                                    aux_state=(aux, RECIPE["aux_coef"]))
                tgt = tgt.eval().to("cpu")
                torch.save({"target": tgt.state_dict(), "pred": m.predictor.state_dict()},
                           CKPT / f"ckpt3d_audit_{name}_r{r}.pt")
                res = audit_pair(tgt, m.predictor.to("cpu"), ep_clouds_h, ep_acts_h)
            except Exception as exc:  # noqa: BLE001
                res = {"error": str(exc)[:200]}
            art["pairs"][name].append(res)
            print(f"  {name} r{r}: δ̂ {res.get('delta',{}).get('mean')} λ1 {res.get('lambda1')} "
                  f"guar {res.get('guar')} viol {res.get('viol')}")
            save()

    vn = [p for p in art["pairs"]["vn"] if "guar" in p]
    art["verdict"] = {
        "vn_faithful_guar": f"{sum(p['guar'] for p in vn)}/{len(vn)}" if vn else "none",
        "vn_delta_median": round(float(np.median([p["delta"]["mean"] for p in vn])), 3) if vn else None,
        "plain_delta_median": round(float(np.median([p["delta"]["mean"] for p in art["pairs"]["plain"] if "delta" in p])), 3) if art["pairs"]["plain"] else None,
    }
    save()
    print("VERDICT:", art["verdict"])
    print(f"3D AUDIT DONE ({(time.time()-T0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
