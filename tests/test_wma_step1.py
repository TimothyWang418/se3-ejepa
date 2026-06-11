r"""wma-step1 — ATM instrument certification gates (pre-registered).

Spec: docs/specs/2026-06-11-wma-step1-atm-selfimpl-seed.md  (P4-D / WMA)
Instrument: src/audit/atm.py — self-implementation of the Action-Consistency Transfer
Matrix (Chen, arXiv:2606.09028; official code unreleased as of 2026-06-11).

Gates (synthetic, fully controlled linear world $z' = Az + Ba + \sigma\eta$):

- **G1a (perfect predictor):** $F$ = true mean map $\Rightarrow$
  $|G_{T\to P}| < 0.05$, $L_{\mathrm{sym}} < 0.2$, $\tilde D_{T,T} < 0.1$.
- **G1b (broken predictor):** $F$ = frozen noise independent of $(z, a)$ $\Rightarrow$
  $\tilde D_{P,P} \ge 0.7$ and $D_{T,P} / D_{T,T} \ge 5$.
- **G1c (shortcut injection):** $F$ writes $2a$ verbatim into the last $d_a$ latent
  coordinates (a $P$-domain-only action code; the $T$ domain has no such channel)
  $\Rightarrow$ $L_{\mathrm{sym}} \ge 10\times$ the G1a value, with the AITS-P signature
  ($D_{P,P}$ collapsed, $D_{P,T}$ high — the code does not transfer back to $T$).

Plus closed-form correctness tests for the feature constructor and the matrix readouts
(eqs. 4, 6, 8-9 of arXiv:2606.09028) and the scale-free normalization $\tilde D_{i,j} =
D_{i,j} / \mathbb{E}\|a - \bar a\|_2^2$ (this tool's protocol addition).
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit.atm import ATMResult, atm_audit, build_transition_features, compute_readouts  # noqa: E402

# Light probe settings for the synthetic gates (the LeWM run uses the full registered
# protocol: widths {64,256,1024} x 3 probe seeds x <=2000 steps).
FAST = dict(widths=(256,), main_width=256, probe_seeds=(0,), max_steps=800)


# ---------------------------------------------------------------------------
# synthetic linear world  z' = A z + B a + sigma * eta
# ---------------------------------------------------------------------------

def _linear_world(n: int = 4000, dz: int = 16, da: int = 4, sigma: float = 0.003, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    A = 0.9 * torch.linalg.qr(torch.randn(dz, dz, generator=g, dtype=torch.float64))[0].float()
    B = torch.randn(dz, da, generator=g) / math.sqrt(da)
    z = torch.randn(n, dz, generator=g)
    a = torch.rand(n, da, generator=g) * 2.0 - 1.0          # U[-1, 1]^da
    eta = torch.randn(n, dz, generator=g)
    z_next = z @ A.T + a @ B.T + sigma * eta
    return z, a, z_next, A, B


# ---------------------------------------------------------------------------
# closed-form correctness
# ---------------------------------------------------------------------------

def test_features_are_exact_concat():
    z = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    z2 = torch.tensor([[3.0, 5.0], [1.0, 1.0]])
    xi = build_transition_features(z, z2)
    expect = torch.tensor([[1.0, 2.0, 3.0, 5.0, 2.0, 3.0],
                           [0.0, -1.0, 1.0, 1.0, 1.0, 2.0]])
    assert xi.shape == (2, 6)
    assert torch.allclose(xi, expect)


def test_readouts_match_hand_computation():
    # Hand-pick D values; eps small enough not to matter at this scale.
    D = {("T", "T"): 0.2, ("T", "P"): 0.3, ("P", "T"): 0.9, ("P", "P"): 0.1}
    r = compute_readouts(D, eps=0.0)
    assert r["G_TP"] == pytest.approx((0.3 - 0.2) / 0.2)            # 0.5
    assert r["G_PT"] == pytest.approx((0.9 - 0.1) / 0.1)            # 8.0
    assert r["I_diag"] == pytest.approx(abs(0.2 - 0.1) / (0.5 * (0.2 + 0.1)))   # 2/3
    assert r["L_sym"] == pytest.approx(2.0 / 3.0 + abs(0.5 - 8.0))  # I_diag + |G_TP - G_PT|


def test_normalization_is_variance_of_action():
    # tilde-D divides by E||a - abar||^2 (sum over dims), computed on the TRAIN split:
    # predicting the train-mean action must give tilde-D ~= 1.
    z, a, z_next, A, B = _linear_world(n=2000)
    res = atm_audit(z, z_next, z_next, a, **FAST)                   # P == T (degenerate ok)
    per_dim_var = 1.0 / 3.0                                          # Var U[-1,1]
    assert res.action_var == pytest.approx(a.shape[1] * per_dim_var, rel=0.1)
    for key, val in res.D.items():
        assert res.D_norm[key] == pytest.approx(val / res.action_var, rel=1e-6)


# ---------------------------------------------------------------------------
# pre-registered gates G1a / G1b / G1c
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def g1a_result() -> ATMResult:
    z, a, z_next, A, B = _linear_world()
    z_pred = z @ A.T + a @ B.T                                       # true mean map
    return atm_audit(z, z_next, z_pred, a, **FAST)


def test_g1a_perfect_predictor(g1a_result):
    r = g1a_result
    assert abs(r.readouts["G_TP"]) < 0.05, f"G_TP={r.readouts['G_TP']}"
    assert r.readouts["L_sym"] < 0.2, f"L_sym={r.readouts['L_sym']}"
    assert r.D_norm[("T", "T")] < 0.1, f"Dtt_norm={r.D_norm[('T', 'T')]}"


def test_g1b_broken_predictor():
    z, a, z_next, A, B = _linear_world()
    g = torch.Generator().manual_seed(123)
    z_pred = torch.randn(z.shape[0], z.shape[1], generator=g)        # frozen noise, indep of (z, a)
    r = atm_audit(z, z_next, z_pred, a, **FAST)
    assert r.D_norm[("P", "P")] >= 0.7, f"Dpp_norm={r.D_norm[('P', 'P')]}"
    ratio = r.D[("T", "P")] / r.D[("T", "T")]
    assert ratio >= 5.0, f"D_TP/D_TT={ratio}"


def test_g1c_shortcut_injection(g1a_result):
    z, a, z_next, A, B = _linear_world(sigma=0.05)
    da = a.shape[1]
    z_pred = z @ A.T
    z_pred = torch.cat([z_pred[:, :-da], 2.0 * a], dim=1)            # verbatim action code, P only
    r = atm_audit(z, z_next, z_pred, a, **FAST)
    # gate: matrix-level imbalance explodes vs the healthy G1a reference
    assert r.readouts["L_sym"] >= 10.0 * max(g1a_result.readouts["L_sym"], 1e-3), (
        f"L_sym={r.readouts['L_sym']} vs G1a {g1a_result.readouts['L_sym']}")
    # AITS-P signature: P-domain action code decodes in P but does not transfer back to T
    assert r.D[("P", "P")] < r.D[("T", "T")], "expected collapsed D_PP"
    assert r.D[("P", "T")] > 10.0 * r.D[("P", "P")], (
        f"D_PT={r.D[('P', 'T')]} vs D_PP={r.D[('P', 'P')]}")


# ---------------------------------------------------------------------------
# result object is JSON-serializable (report-card plumbing)
# ---------------------------------------------------------------------------

def test_result_serializes(g1a_result):
    out = g1a_result.to_dict()
    assert isinstance(out, dict)
    assert "D" in out and "readouts" in out and "per_width" in out
    import json
    json.dumps(out)                                                   # must not raise
