r"""P4-spine Stage-1a — the first certified-vs-measured comparison on real bases (seed-0, descriptive).

Spec: docs/specs/2026-06-10-p4-spine-ep43-seed.md (§2 model-side boundary; §6 G-pre). This is the
audit half of Stage 1: NO planner yet (Stage-1b adds subgoal-conditioned CEM, crossover, θ̂*).
**No gate is decided here** (C3-cal needs 3 seeds; this is the seed-0 shape pass) — but every
registered readout is computed exactly as the gates will consume it.

Per (base ∈ {eq, plain_match} @200, regime = static κ=0):

1. **Instrument-bias calibration at spine settings** (W=16, B=4 — shorter than G-I's deployed
   W=40/B=10 because v1.2 episodes hold only 20 f-chunks): planted linear loops at
   λ ∈ {−0.05, 0, +0.08}; offsets recorded and quoted alongside every λ̂.
2. **Gap audit** (`audit_gap`, held-out data, never trained on): (δ̂, λ̂₁, ε̂_max) + certified
   curves (δ̂-mean and δ̂-q90 variants), H in f-chunks 1..8 (= 5..40 env steps).
3. **Measured rollout error** Err_f^meas(H): f^(H) from encoded chunk-boundary states under
   recorded action chunks vs encoded true future; median + q90 over all windows.
4. **G-pre diagnostic** (linearization-neighborhood, step99 lesson): exponential-vs-linear fit of
   the measured median curve; in the neutral regime the certificate predicts LINEAR growth δ̂·H —
   the diagnostic here is the growth SHAPE (the r-ratio form activates in the expansive regime).
5. **C3-cal seed-0 shapes**: H*_q90(ε) vs H^meas_max(ε) for ε ∈ {2,4,8,16}·δ̂_mean.

Units: H in f-chunks throughout this artifact (1 f-chunk = 5 env steps; spine spec §1).

Run: .venv/bin/python experiments/p4_spine_stage1a.py   (~5-10 min)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, pick_ladder,
)
from src.audit.gap_mode import audit_gap, loop_exponents  # noqa: E402

SEED_DATA = 1            # held-out: disjoint from the training collection (seed 0)
N_EP = 60
W_AUDIT, B_AUDIT = 16, 4  # spine settings (episodes hold 20 f-chunks)
STARTS = (0, 2, 4)
H_MAX = 8                # f-chunks (= 40 env steps)
EPS_MULT = (2, 4, 8, 16)
OUT = ROOT / "papers" / "figures" / "p4_spine_stage1a.json"
DATA_DIR = ROOT / "data" / "p4_step1"


# ------------------------------------------------------------------ planted calibration (W=16,B=4)
class PlantedLinear(torch.nn.Module):
    def __init__(self, lam1: float, d: int = 16, in_dim: int = 3 * RES * RES):
        super().__init__()
        gen = torch.Generator().manual_seed(1)
        self.register_buffer("P", torch.randn(in_dim, d, generator=gen, dtype=torch.float64) / in_dim**0.5)
        diag = torch.full((d,), lam1 - 0.1, dtype=torch.float64)
        diag[0] = lam1
        self.register_buffer("A", torch.diag(torch.exp(diag)))

    def encode(self, px):
        return px.flatten(1).double() @ self.P

    def predictor(self, z, a):
        return z @ self.A.T


def instrument_bias(frames: torch.Tensor, actions: torch.Tensor) -> dict:
    out = {}
    for lam in (-0.05, 0.0, 0.08):
        lams = loop_exponents(PlantedLinear(lam), frames, actions, window=W_AUDIT, k=3,
                              burn_in=B_AUDIT, starts=STARTS)
        out[str(lam)] = {"recovered": float(lams.mean()), "offset": float(lams.mean() - lam)}
    return out


# ------------------------------------------------------------------ measured rollout error
@torch.no_grad()
def measured_err(model, frames: torch.Tensor, actions: torch.Tensor, h_max: int) -> dict:
    r"""Err_f^meas(H) = || f^(H)(z_t, a-chunks) − z_{t+H} || over all valid windows; per-H median/q90.

    Encoding in the model's native dtype; rollout chain in f64 (same convention as the audit).
    """
    import copy

    dt = next(model.parameters()).dtype
    E, T1 = frames.shape[0], frames.shape[1]  # T1 = n_chunks + 1 boundary frames
    z = model.encode(frames.reshape(-1, *frames.shape[2:]).to(dt)).reshape(E, T1, -1).double()
    pred = copy.deepcopy(model.predictor).double()
    for p in pred.parameters():
        p.requires_grad_(False)
    errs = {h: [] for h in range(1, h_max + 1)}
    n_ch = T1 - 1
    for t0 in range(0, n_ch - 1):
        cur = z[:, t0]
        for h in range(1, h_max + 1):
            if t0 + h > n_ch:
                break
            cur = pred(cur, actions[:, t0 + h - 1].double())
            errs[h].append((cur - z[:, t0 + h]).norm(dim=-1))
    return {
        "median": [float(torch.cat(errs[h]).median()) for h in range(1, h_max + 1) if errs[h]],
        "q90": [float(torch.cat(errs[h]).quantile(0.9)) for h in range(1, h_max + 1) if errs[h]],
    }


def boundary_from_curve(curve: list[float], eps: float) -> int:
    r"""max{H : curve[H-1] <= eps}; 0 if even H=1 exceeds."""
    h = 0
    for i, v in enumerate(curve, start=1):
        if v <= eps:
            h = i
    return h


def fit_growth(curve: list[float]) -> dict:
    r"""Exponential vs linear fit of the measured median curve (G-pre, neutral-regime form).

    Both models carry an intercept (review fix 2026-06-11: the original no-intercept linear model
    made the exp-vs-linear comparison meaningless whenever Err(1) has a floor)."""
    h = np.arange(1, len(curve) + 1, dtype=float)
    y = np.asarray(curve, dtype=float)
    exp_coef = np.polyfit(h, np.log(np.maximum(y, 1e-12)), 1)   # ln y = lam*h + c
    lin_coef = np.polyfit(h, y, 1)                              # y = a*h + b
    ss = lambda r: float((r**2).sum())  # noqa: E731
    r2 = lambda r: 1.0 - ss(r) / max(ss(y - y.mean()), 1e-12)  # noqa: E731
    return {
        "lambda_meas_exp_fit": float(exp_coef[0]),
        "linear_rate": float(lin_coef[0]),
        "linear_intercept": float(lin_coef[1]),
        "r2_exp": r2(y - np.exp(np.polyval(exp_coef, h))),
        "r2_linear": r2(y - np.polyval(lin_coef, h)),
    }


def main() -> int:
    t0 = time.time()
    art: dict = {"seed_data": SEED_DATA, "n_ep": N_EP, "regime": "static (kappa=0)",
                 "audit": {"window": W_AUDIT, "burn_in": B_AUDIT, "starts": list(STARTS)},
                 "units": "H in f-chunks (1 f-chunk = 5 env steps)"}

    print(f"[1/5] held-out collection: {N_EP} episodes (seed {SEED_DATA}) ...")
    data = collect_weakpolicy(N_EP, seed=SEED_DATA)
    f = torch.from_numpy(data["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK].double()                                    # (E, 21, 3, 96, 96) boundaries
    a = torch.from_numpy(data["actions"])
    n_ch = a.shape[1] // CHUNK
    ach = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).double()  # (E, 20, 10)

    print("[2/5] instrument-bias calibration at spine settings (W=16, B=4) ...")
    art["instrument_bias"] = instrument_bias(fb, ach)
    print(f"      {art['instrument_bias']}")

    print("[3/5] load bases from ckpt2 (DEPLOYABLE pair: EMA-target encoder + predictor) ...")

    class DeployPair(torch.nn.Module):
        r"""The pairing the predictor was trained against — train_jepa regresses onto the
        EMA-target encoder's latents, so the deployable world model is (target encoder,
        predictor). Stage-1a's first run used the online encoder from the v1.2 checkpoints and
        measured δ̂ ≈ the full latent scale (15× inflated; superseded artifact kept in git)."""

        def __init__(self, encoder, predictor):
            super().__init__()
            self.encoder = encoder       # named `encoder` so eps_max sees encode_geometric
            self.pred = predictor

        def encode(self, px):
            return self.encoder(px)

        @property
        def predictor(self):
            return self.pred

    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    bases = {}
    for name, builder in (("eq", build_eq), ("plain_match", lambda: build_plain(w_match))):
        # ckpt3 = trained with refresh_target_cache=True (the e2cnn pairing fix; #9 equality
        # gates E-I/II/III all PASS — in-process == reloaded exactly). ckpt2 era is quarantined.
        ck = torch.load(DATA_DIR / f"ckpt3_{name}_f200.pt", map_location="cpu", weights_only=True)
        m = builder()
        m.load_state_dict(ck["model"])
        import copy as _copy

        tgt = _copy.deepcopy(m.encoder)
        tgt.load_state_dict(ck["target_encoder"])
        bases[name] = DeployPair(tgt, m.predictor).eval()

    for name, model in bases.items():
        print(f"[4/5] {name}: gap audit + measured curves ...")
        rep = audit_gap(model, fb, ach, window=W_AUDIT, k=3, burn_in=B_AUDIT, h_max=H_MAX)
        meas = measured_err(model, fb, ach, H_MAX)
        d_mean = rep["delta"]["mean"]
        eps_grid = [m_ * d_mean for m_ in EPS_MULT]
        cal = []
        for em, eps in zip(EPS_MULT, eps_grid):
            h_cert = boundary_from_curve(rep["certified_curve"]["err_q90"], eps)
            h_meas = boundary_from_curve(meas["median"], eps)
            cal.append({"eps_mult": em, "eps": eps, "H_cert_q90": h_cert, "H_meas": h_meas,
                        "ratio": (h_cert / h_meas) if h_meas else None})
        art[name] = {
            "delta": rep["delta"], "lambda1": rep["lambda1"],
            "eps_max": rep["eps_max"] if rep["eps_max"] is None else
            {k: rep["eps_max"][k] for k in ("max", "grid_subgroup_max", "offgrid_max")},
            "certified_q90": rep["certified_curve"]["err_q90"],
            "measured_median": meas["median"], "measured_q90": meas["q90"],
            "gpre_growth_fit": fit_growth(meas["median"]),
            "c3cal_seed0": cal,
        }
        print(f"      delta {d_mean:.4f}  lambda1 {rep['lambda1']['mean']:+.4f} "
              f"[{rep['lambda1']['ci_lo']:+.4f},{rep['lambda1']['ci_hi']:+.4f}]/chunk")
        print(f"      meas median Err(H): {[round(v,3) for v in meas['median']]}")
        print(f"      G-pre fit: {art[name]['gpre_growth_fit']}")
        print(f"      C3-cal: {cal}")

    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"[5/5] wrote {OUT.name}  ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
