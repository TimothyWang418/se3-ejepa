r"""P4-spine Stage-2κ — the κ=0.8 lane: bases, zsens pre-gate, audits, C3-regime seed-0 preview.

Spec: docs/specs/2026-06-10-p4-spine-ep43-seed.md (§2 regimes; C3-zsens; G-pre).
The expansive point from step2's gate (env $\hat\lambda_1^{\text{env}} = +0.088$/env-step at
κ=0.8 ⇒ ×5 = **+0.44/f-chunk** for cross-reference — conversion stated, never silent).

Registered expectations (declared BEFORE the run, honest priors, not gates):

- The κ=0 eq predictor trained to $J \equiv I$ (purely action-driven). If the κ=0.8 eq predictor
  does the same, its tangent $\hat\lambda_1$ is *structural* zero while the TRUE dynamics are
  expansive ⇒ **G-pre should fire FAILED-BY-SCOPE for eq at κ=0.8** — that outcome is the
  C3-zsens pre-gate doing its job, and it opens the registered open question (how to train a
  z-sensitive equivariant predictor), not a burial.
- plain learned mild z-dependence at κ=0; at κ=0.8 it may pick up more.

Protocol: v1.1/v1.2 throughout (WeakPolicy corpus regenerates at κ=0.8 by construction;
hyperparameters frozen at the κ=0 settings — the proposal's fairness rule). Bases trained with
``refresh_target_cache=True`` (#9 fix) and saved as deployable pairs (ckpt3 format, ``_k08``).

Run: .venv/bin/python experiments/p4_spine_stage2_kappa08.py   (~20 min; collection dominates)
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_spine_stage1a import (  # noqa: E402
    boundary_from_curve, fit_growth, instrument_bias, measured_err,
)
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder,
    to_transitions,
)
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

KAPPA = 0.8
SEED = 0
W_AUDIT, B_AUDIT = 16, 4
H_MAX = 8
EPS_MULT = (2, 4, 8, 16)
ENV_LAMBDA_PER_CHUNK = 0.088 * CHUNK  # step2 gate, converted at the comparison site
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_spine_stage2_kappa08.json"


class Pair(torch.nn.Module):
    def __init__(self, enc, pred):
        super().__init__()
        self.encoder = enc
        self.pred = pred

    def encode(self, px):
        return self.encoder(px)

    @property
    def predictor(self):
        return self.pred


def zsens(predictor, d: int = 128, n_dirs: int = 8) -> dict:
    r"""C3-zsens pre-gate probe: $\lVert \partial\Delta/\partial z\, v \rVert$ over random unit v."""
    pred = copy.deepcopy(predictor).double().eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    z = torch.randn(d, dtype=torch.float64)
    a = torch.empty(CHUNK * 2, dtype=torch.float64).uniform_(-1, 1)
    f = lambda zz: pred(zz.unsqueeze(0), a.unsqueeze(0)).squeeze(0) - zz  # noqa: E731
    norms = []
    for _ in range(n_dirs):
        v = torch.randn(d, dtype=torch.float64)
        v /= v.norm()
        _, jv = torch.func.jvp(f, (z,), (v,))
        norms.append(float(jv.norm()))
    return {"mean": float(np.mean(norms)), "max": float(np.max(norms)),
            "label": "STRUCTURAL-IDENTITY" if max(norms) < 1e-9 else "z-sensitive"}


def main() -> int:
    t0 = time.time()
    art: dict = {"kappa": KAPPA, "seed": SEED, "units": "per f-chunk (1 chunk = 5 env steps)",
                 "env_lambda_per_chunk_ref": ENV_LAMBDA_PER_CHUNK,
                 "audit": {"window": W_AUDIT, "burn_in": B_AUDIT}}

    print(f"[1/6] training corpus at kappa={KAPPA} (200 eps, seed {SEED}) ...")
    corpus = collect_weakpolicy(200, seed=SEED, kappa=KAPPA)
    obs, act, nxt = to_transitions(corpus, 200)
    print(f"      done ({time.time()-t0:.0f}s)")

    print("[2/6] train bases (frozen kappa=0 hyperparams, #9 fix on) ...")
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    pairs = {}
    for name, builder in (("eq", build_eq), ("plain_match", lambda: build_plain(w_match))):
        torch.manual_seed(SEED)
        m = builder()
        hist, tgt = train_jepa(m, obs, act, nxt, epochs=20, batch_size=64, device=DEVICE,
                               seed=SEED, verbose=False, return_target_encoder=True,
                               refresh_target_cache=True)
        torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                   DATA_DIR / f"ckpt3_{name}_f200_k08.pt")
        pairs[name] = Pair(tgt.eval().cpu(), m.predictor.cpu())
        art.setdefault("train", {})[name] = {"pred_loss": hist["pred_loss"][-1],
                                             "latent_std": hist["latent_std"]}
        print(f"      {name}: pred_loss {hist['pred_loss'][-1]:.5f}  latent_std {hist['latent_std']:.3f}")

    print("[3/6] C3-zsens pre-gate ...")
    for name, pr in pairs.items():
        art.setdefault("zsens", {})[name] = zsens(pr.predictor)
        print(f"      {name}: {art['zsens'][name]}")

    print(f"[4/6] held-out collection at kappa={KAPPA} (60 eps, seed 1) ...")
    ho = collect_weakpolicy(60, seed=1, kappa=KAPPA)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).double()

    print("[5/6] instrument bias (planted, spine settings) ...")
    art["instrument_bias"] = instrument_bias(fb, ach)

    print("[6/6] audits + measured curves + C3-cal/G-pre seed-0 shapes ...")
    for name, pr in pairs.items():
        rep = audit_gap(pr, fb, ach, window=W_AUDIT, k=3, burn_in=B_AUDIT, h_max=H_MAX)
        meas = measured_err(pr, fb, ach, H_MAX)
        d_mean = rep["delta"]["mean"]
        cal = []
        for em in EPS_MULT:
            eps = em * d_mean
            cal.append({"eps_mult": em,
                        "H_cert_q90": boundary_from_curve(rep["certified_curve"]["err_q90"], eps),
                        "H_meas_q90": boundary_from_curve(meas["q90"], eps),
                        "H_meas_med": boundary_from_curve(meas["median"], eps)})
        gfit = fit_growth(meas["median"])
        lam_t = rep["lambda1"]["mean"]
        gpre_r = (gfit["lambda_meas_exp_fit"] / lam_t) if abs(lam_t) > 1e-6 else None
        art[name] = {
            "delta": {k: rep["delta"][k] for k in ("mean", "q50", "q90", "ci_lo", "ci_hi")},
            "lambda1_tangent": rep["lambda1"],
            "measured_median": meas["median"], "measured_q90": meas["q90"],
            "growth_fit": gfit, "gpre_r": gpre_r, "c3cal_seed0": cal,
        }
        print(f"      {name}: delta {d_mean:.3f}  lam_tangent {lam_t:+.4f}  "
              f"lam_meas {gfit['lambda_meas_exp_fit']:+.4f}  r {gpre_r}  "
              f"r2_lin {gfit['r2_linear']:.3f} r2_exp {gfit['r2_exp']:.3f}")
        print(f"        meas median: {[round(v,2) for v in meas['median']]}")
        print(f"        C3-cal: {cal}")

    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
