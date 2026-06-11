r"""Artifact-consistency tests for step98 (V-JEPA 2-AC certificate) and step99 (DROID real-robot-data monitor)."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
S98 = ROOT / "papers/figures/step98_vjepa2_audit.json"
S99 = ROOT / "papers/figures/step99_droid_monitor.json"


@pytest.mark.skipif(not S98.exists(), reason="artifact missing")
def test_step98_certificate_consistent():
    d = json.loads(S98.read_text())
    l1s = [d["runs"][s]["lambda1"] for s in d["runs"]]
    # 2026-06-11 thickened to five Q-seeds; the registered rule generalizes to the extreme pair
    assert len(l1s) >= 2 and len({np.sign(x) for x in l1s}) == 1
    assert (max(l1s) - min(l1s)) / max(abs(x) for x in l1s) < 0.30
    assert abs(l1s[0] - l1s[1]) / max(map(abs, l1s)) < 0.30 and d["q_seed_stable"]
    assert d["lambda1"] == pytest.approx(float(np.mean(l1s)), rel=1e-9)
    lo, hi = d["lambda1_ci_envelope"]
    assert lo == min(d["runs"][s]["ci"][0] for s in d["runs"])
    assert hi == max(d["runs"][s]["ci"][1] for s in d["runs"])
    assert d["verdict"].startswith("EXPANSIVE") == (d["lambda1"] > 0 and lo > 0)
    t1 = [r for r in d["cert_rows"] if r["eps"] == 0.2][0]["T1_steps"]
    assert t1 == pytest.approx(np.log(5.0) / d["lambda1"], rel=1e-9)
    assert d["d_state"] == 360448


@pytest.mark.skipif(not (S98.exists() and S99.exists()), reason="artifacts missing")
def test_step99_branch_and_gate_reproducible():
    d = json.loads(S99.read_text())
    cert = json.loads(S98.read_text())
    # provenance: step99 consumed the certificate at its run time = the mean of Q-seeds 0,1
    # (the 2026-06-11 five-seed thickening shifts the artifact mean by <1%; DROID gates never bind on it)
    two_seed_mean = (cert["runs"]["0"]["lambda1"] + cert["runs"]["1"]["lambda1"]) / 2
    assert d["certificate"]["lambda1"] == pytest.approx(two_seed_mean, rel=1e-9)
    assert abs(d["certificate"]["lambda1"] - cert["lambda1"]) / cert["lambda1"] < 0.01
    T1 = d["certificate"]["T1_at_theta"]
    # branch selection per the frozen spec: certificate EXPANSIVE -> G8-E pricing band
    assert d["branch"].startswith("G8-E")
    in_band = T1 / 1.5 <= d["crossing_median"] <= 1.5 * T1
    assert in_band == d["G8"] and d["G8"] is False                  # fails as registered (honest)
    # pre-registered sub-classification rule re-derived
    sub = ("stable" if (d["censored_frac"] >= 0.5 and d["one_step_rel_err_median"] < 0.1) else
           "bias" if (d["one_step_rel_err_median"] >= 0.1 or d["crossing_median"] <= 3) else "mixed")
    assert sub == d["measured_subclass"] == "bias"


@pytest.mark.skipif(not S99.exists(), reason="artifact missing")
def test_step99_mechanism_numbers_internally_consistent():
    d = json.loads(S99.read_text())
    m = d["mechanism"]
    curve = np.array(m["median_err_vs_staleness"])
    assert (np.diff(curve) > -1e-9).all()                           # monotone growth of the median curve
    xs = np.arange(1, len(curve) + 1)
    slope = float(np.polyfit(xs, np.log(curve), 1)[0])
    assert slope == pytest.approx(m["growth_log_slope"], rel=1e-6)
    # the headline gap: growth slope far below the certified lambda1; bias floor at native step motion
    assert m["growth_log_slope"] < 0.5 * d["certificate"]["lambda1"]
    assert d["one_step_rel_err_median"] < m["copy_baseline_err_median"]   # predictor beats copying
    assert abs(d["one_step_rel_err_median"] - m["consecutive_latent_dist_median"]) < 0.15
