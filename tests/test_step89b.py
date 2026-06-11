"""Tests for step89b — zoo expansion: FLAT->FUSED translator + frozen 84-cell map statistics.

The translator handles the zoo's second export generation (separate Linear/LayerNorm modules at
interleaved indices, including the `_dynamics.0.{i}.*` inner-wrapper nesting with a trailing
top-level `_dynamics.1.*` LayerNorm — the collision that broke single-index keying).
The statistics tests re-derive every number cited in the E13 expansion paragraph from the artifacts.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import step89b_audit_expansion as s89b  # noqa: E402


# ---------- translator ----------

def test_fused_checkpoint_passes_through_untouched():
    sd = {"_encoder.state.0.weight": torch.randn(8, 4), "_encoder.state.0.bias": torch.randn(8),
          "_encoder.state.0.ln.weight": torch.randn(8), "_encoder.state.0.ln.bias": torch.randn(8)}
    assert s89b._maybe_translate_old_format(sd) is sd


def test_flat_linear_ln_pairs_fuse_in_order():
    # FLAT: Linear at 1, LayerNorm at 2, Linear at 4, LayerNorm at 5 (activations occupy 0/3 — stateless)
    sd = {"_encoder.state.1.weight": torch.randn(8, 4), "_encoder.state.1.bias": torch.randn(8),
          "_encoder.state.2.weight": torch.ones(8), "_encoder.state.2.bias": torch.zeros(8),
          "_encoder.state.4.weight": torch.randn(6, 8), "_encoder.state.4.bias": torch.randn(6),
          "_encoder.state.5.weight": torch.ones(6), "_encoder.state.5.bias": torch.zeros(6)}
    out = s89b._maybe_translate_old_format(sd)
    assert set(out) == {"_encoder.state.0.weight", "_encoder.state.0.bias",
                        "_encoder.state.0.ln.weight", "_encoder.state.0.ln.bias",
                        "_encoder.state.1.weight", "_encoder.state.1.bias",
                        "_encoder.state.1.ln.weight", "_encoder.state.1.ln.bias"}
    assert torch.equal(out["_encoder.state.0.weight"], sd["_encoder.state.1.weight"])
    assert torch.equal(out["_encoder.state.1.ln.weight"], sd["_encoder.state.5.weight"])


def test_dynamics_nested_wrapper_with_trailing_toplevel_layernorm():
    # The collision case: `_dynamics.0.{i}.*` inner wrapper + trailing `_dynamics.1.*` top-level LN.
    # Single stripped-index keying collided (inner 1 vs top-level 1); tuple-path keying must not.
    sd = {"_dynamics.0.0.weight": torch.randn(16, 10), "_dynamics.0.0.bias": torch.randn(16),
          "_dynamics.0.1.weight": torch.ones(16), "_dynamics.0.1.bias": torch.zeros(16),
          "_dynamics.0.3.weight": torch.randn(12, 16), "_dynamics.0.3.bias": torch.randn(12),
          "_dynamics.1.weight": torch.ones(12), "_dynamics.1.bias": torch.zeros(12)}
    out = s89b._maybe_translate_old_format(sd)
    # paths sort (0,0)<(0,1)<(0,3)<(1,): Linear0+LN, Linear1+trailing-LN -> fused 0 and 1
    assert torch.equal(out["_dynamics.0.weight"], sd["_dynamics.0.0.weight"])
    assert torch.equal(out["_dynamics.0.ln.weight"], sd["_dynamics.0.1.weight"])
    assert torch.equal(out["_dynamics.1.weight"], sd["_dynamics.0.3.weight"])
    assert torch.equal(out["_dynamics.1.ln.weight"], sd["_dynamics.1.weight"])
    assert "_dynamics.1.ln.bias" in out


def test_unpaired_linear_keeps_no_ln():
    sd = {"_pi.0.weight": torch.randn(4, 4), "_pi.0.bias": torch.randn(4)}
    out = s89b._maybe_translate_old_format(sd)
    assert set(out) == {"_pi.0.weight", "_pi.0.bias"}


# ---------- frozen 84-cell statistics (every number in the E13 expansion paragraph) ----------

@pytest.fixture(scope="module")
def cells():
    out = {}
    for name in ("step89_pretrained_audit.json", "step89b_audit_expansion.json"):
        f = ROOT / "papers/figures" / name
        if not f.exists():
            pytest.skip(f"{name} not present")
        for k, v in json.loads(f.read_text()).items():
            if not k.startswith("_") and "error" not in v:
                out[k] = v
    return out


def _at(v, eps=0.2):
    return next(r for r in v["cert_rows"] if abs(r["eps"] - eps) < 1e-9)


def test_map_size_and_abstain_split(cells):
    assert len(cells) == 84
    abstain = [v for v in cells.values() if not (v["lambda1"] > 0 and v["lambda1_ci"][0] > 0)]
    stable = [v for v in abstain if v["measured"]["0.2"]["n_censored"] >= 10]
    assert len(abstain) == 42 and len(stable) == 13


def test_inband_cells_span_lambda_axis(cells):
    exp = {k: v for k, v in cells.items() if v["lambda1"] > 0 and v["lambda1_ci"][0] > 0}
    assert len(exp) == 42
    lam_inband = [v["lambda1"] for v in exp.values()
                  if _at(v)["ratio_measured_over_certified"] and 2 / 3 <= _at(v)["ratio_measured_over_certified"] <= 1.5]
    assert min(lam_inband) < 0.05 and max(lam_inband) > 0.30   # spans the axis -> lambda does not predict calibration


def test_growth_set_cells_are_calibrated(cells):
    import numpy as np
    exp = [v for v in cells.values() if v["lambda1"] > 0 and v["lambda1_ci"][0] > 0]
    bias_dom = [v for v in exp if _at(v)["measured_median"] <= 3]
    growth5 = [_at(v)["ratio_measured_over_certified"] for v in exp if _at(v)["measured_median"] >= 5]
    assert len(bias_dom) == 24
    assert len(growth5) == 15
    assert abs(float(np.median(growth5)) - 0.95) < 0.005
    assert sum(1 for r in growth5 if 2 / 3 <= r <= 1.5) == 10


def test_excluded_by_rule_listed_and_disjoint(cells):
    meta = json.loads((ROOT / "papers/figures/step89b_audit_expansion.json").read_text())["_meta"]
    assert len(meta["excluded_by_rule"]) == 11
    assert set(meta["excluded_by_rule"]).isdisjoint({k.rsplit("-", 1)[0] for k in cells})


# ---------- step89c: measured-column thickening (100 starts/cell) ----------

def test_step89c_n100_regime_stats(cells):
    import numpy as np
    f = ROOT / "papers/figures/step89c_measured_n100.json"
    if not f.exists():
        pytest.skip("step89c artifact not present")
    c = json.loads(f.read_text())
    ok = {k: v for k, v in c.items() if "error" not in v}
    assert len(ok) == 84 and len(ok) == len(c)
    exp = [(c[k]["measured_n100"]["0.2"]["median"], c[k]["ratio_n100"]["0.2"])
           for k, v in cells.items() if v["lambda1"] > 0 and v["lambda1_ci"][0] > 0]
    assert sum(1 for m, _ in exp if m <= 3) == 25                  # bias-dominated
    g5 = [r for m, r in exp if m >= 5]
    assert len(g5) == 15
    assert abs(float(np.median(g5)) - 0.943) < 0.005               # growth-side calibration
    assert sum(1 for r in g5 if 2 / 3 <= r <= 1.5) == 8
