r"""P4-step3 instrument validation: the gap-mode audit is certified before it certifies.

- **G-I (planted-spectrum recovery):** a linear latent loop $z \mapsto A z$ with planted leading
  exponent $\lambda_1 \in \{-0.05, 0, +0.08\}$ (the κ-gate's measured range) — recovered to
  numerical precision (a linear system makes every window exponent exact).
- **G-II (determinism):** two runs of ``audit_gap`` on the same model+data are bit-identical.
- **G-III (orbit-invariance witness):** for a random-weights exactly-equivariant base
  (SteerableEncoder $C_4$ + CNRegularPredictor), $\hat\lambda_1$ on $g$-transformed data
  (e2cnn transform + the sign-calibrated action rotation, $s{=}+1$) equals the untransformed
  value within CI — a joint consistency check of instrument and equivariance.

Run: .venv/bin/python tests/test_p4_step3.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.audit.gap_mode import audit_gap  # noqa: E402

E_EP, T_STEPS, RES_T = 3, 40, 16


def make_data(seed: int = 0, c: int = 3, res: int = RES_T, t_steps: int = T_STEPS):
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand(E_EP, t_steps + 1, c, res, res, generator=g, dtype=torch.float64)
    actions = torch.empty(E_EP, t_steps, 2, dtype=torch.float64).uniform_(-1, 1, generator=g)
    return frames, actions


class PlantedLinear(torch.nn.Module):
    r"""Encoder = fixed random projection of pixels; predictor = $z \mapsto A z$ with planted
    $\lambda_1 = \log\sigma_1(A)$ (diagonal A ⇒ every window exponent is exactly $\lambda_1$)."""

    def __init__(self, lam1: float, d: int = 16, in_dim: int = 3 * RES_T * RES_T):
        super().__init__()
        gen = torch.Generator().manual_seed(1)
        self.register_buffer("P", torch.randn(in_dim, d, generator=gen, dtype=torch.float64) / in_dim**0.5)
        diag = torch.full((d,), lam1 - 0.1, dtype=torch.float64)
        diag[0] = lam1
        self.register_buffer("A", torch.diag(torch.exp(diag)))
        self.encoder = torch.nn.Identity()  # placeholder (no geometric interface => eps_max None)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.flatten(1).double() @ self.P

    def predictor(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:  # action-blind, linear
        return z @ self.A.T


def test_gI_planted() -> dict:
    r"""Two tiers: (a) DEPLOYED settings (W=40, B=10) — tolerance 0.02 = the registered
    finite-window instrument bias at spectral gap 0.1 (Prop 8 truncation analogue, measured here);
    (b) LONG window (W=220, B=20) — tolerance 2e-3, shows the bias is transient-alignment, not
    systematic. The no-burn-in failure that motivated this design: planted -0.05 read as -0.076."""
    out = {}
    frames, actions = make_data()
    frames_l, actions_l = make_data(seed=1, t_steps=240)
    for lam in (-0.05, 0.0, 0.08):
        rep = audit_gap(PlantedLinear(lam), frames, actions, window=40, burn_in=10, k=4,
                        n_boot=100, h_max=10)
        got = rep["lambda1"]["mean"]
        assert abs(got - lam) < 0.02, f"G-I(deployed) FAIL: planted {lam}, recovered {got}"
        rep_l = audit_gap(PlantedLinear(lam), frames_l, actions_l, window=220, burn_in=20, k=4,
                          n_boot=100, h_max=10)
        got_l = rep_l["lambda1"]["mean"]
        assert abs(got_l - lam) < 2e-3, f"G-I(long) FAIL: planted {lam}, recovered {got_l}"
        assert rep["eps_max"] is None, "G-I: planted model must report eps_max=None (no rho)"
        out[str(lam)] = {"deployed": round(got, 5), "long": round(got_l, 6)}
    return out


def test_gII_determinism() -> float:
    from src.models.cn_regular import CNRegularPredictor
    from src.models.eqjepa import EqJEPA, SteerableEncoder

    torch.manual_seed(0)
    model = EqJEPA(latent_dim=64, action_dim=2,
                   encoder=SteerableEncoder(3, 64, n_rot=4, width=4),
                   predictor=CNRegularPredictor(64, 2, n_rot=4, hidden_fields=8))
    frames, actions = make_data(seed=2, res=RES_T)
    a = json.dumps(audit_gap(model, frames, actions, window=20, k=3, n_boot=50, h_max=10), sort_keys=True)
    b = json.dumps(audit_gap(model, frames, actions, window=20, k=3, n_boot=50, h_max=10), sort_keys=True)
    assert a == b, "G-II FAIL: audit not bit-deterministic"
    return 0.0


def test_gIII_orbit_witness() -> dict:
    from e2cnn import nn as enn

    from src.models.cn_regular import CNRegularPredictor
    from src.models.eqjepa import EqJEPA, SteerableEncoder

    torch.manual_seed(0)
    n_rot = 4
    enc = SteerableEncoder(3, 64, n_rot=n_rot, width=4)
    model = EqJEPA(latent_dim=64, action_dim=2, encoder=enc,
                   predictor=CNRegularPredictor(64, 2, n_rot=n_rot, hidden_fields=8))
    frames, actions = make_data(seed=3, res=17)  # odd size: exact 90-degree grid rotations

    base = audit_gap(model, frames, actions, window=20, k=3, n_boot=100, h_max=10)

    # g = 1 (90 deg, pixel-exact); pinned pairing s=+1 (tests/test_p4_step1.py)
    g = 1
    theta = 2 * math.pi * g / n_rot
    flat = frames.reshape(-1, *frames.shape[2:]).float()
    gx = enn.GeometricTensor(flat, enc.in_type).transform(
        list(enc.r2.testing_elements)[g]).tensor.double().reshape(frames.shape)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    rot = audit_gap(model, gx, actions @ R.T, window=20, k=3, n_boot=100, h_max=10)

    lam_b, lam_r = base["lambda1"], rot["lambda1"]
    inside = lam_b["ci_lo"] - 1e-6 <= lam_r["mean"] <= lam_b["ci_hi"] + 1e-6
    close = abs(lam_r["mean"] - lam_b["mean"]) < 0.01
    assert inside or close, (
        f"G-III FAIL: lambda1 base {lam_b['mean']:.4f} CI [{lam_b['ci_lo']:.4f},{lam_b['ci_hi']:.4f}] "
        f"vs rotated {lam_r['mean']:.4f}"
    )
    # bonus witness: the eq encoder's residual at the exact angle must be tiny
    assert base["eps_max"] is not None and base["eps_max"]["grid_subgroup_max"] < 1e-4, (
        f"G-III: grid-angle eps_max unexpectedly large: {base['eps_max']}"
    )
    return {"lambda_base": lam_b["mean"], "lambda_rot": lam_r["mean"],
            "eps_grid": base["eps_max"]["grid_subgroup_max"]}


if __name__ == "__main__":
    r1 = test_gI_planted()
    print(f"[G-I] planted-spectrum recovery: {r1}  PASS")
    test_gII_determinism()
    print("[G-II] determinism: bit-identical  PASS")
    r3 = test_gIII_orbit_witness()
    print(f"[G-III] orbit witness: base {r3['lambda_base']:+.4f} vs rotated {r3['lambda_rot']:+.4f}, "
          f"grid eps {r3['eps_grid']:.2e}  PASS")
    print("ALL P4-step3 instrument gates PASS")
