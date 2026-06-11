r"""Tests for Proposition 6′ (prefactor = splitting conditioning) and its artifacts (step65b, step95, step96).

Layers:
1. Pure-math identities behind Prop 6′: $\lVert\Pi\rVert=1/\sin\theta$ for an oblique projector, the orthogonal
   case giving exactly $1$, and the horizon haircut $\log\kappa/\lambda$ — verified on random instances, not trusted.
2. Live re-runs of the fast measured claims (step65b's Schur placement + zero leakage; step95 part A's law) at
   reduced size — these take seconds and re-derive the gates from scratch.
3. Artifact consistency: the committed step95/step96 JSONs' gates are reproducible from their own raw numbers.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

S95 = ROOT / "papers/figures/step95_prefactor_angles.json"
S96 = ROOT / "papers/figures/step96_taxonomy_monitor.json"


def test_projector_norm_equals_inverse_sine_angle():
    rng = np.random.default_rng(7)
    for _ in range(50):
        d = rng.integers(2, 6)
        V = rng.standard_normal((d, d))
        while abs(np.linalg.det(V)) < 1e-3:
            V = rng.standard_normal((d, d))
        W = np.linalg.inv(V)
        P = np.outer(V[:, 0], W[0])                                  # projector onto span(v0) along span(v1..)
        assert np.allclose(P @ P, P, atol=1e-8)
        # minimal principal angle between span(v0) and span(v1..): sin(theta) = dist(v0_hat, complement)
        v0 = V[:, 0] / np.linalg.norm(V[:, 0])
        Q, _ = np.linalg.qr(V[:, 1:])
        sin_theta = np.linalg.norm(v0 - Q @ (Q.T @ v0))
        assert np.linalg.norm(P, 2) == pytest.approx(1.0 / sin_theta, rel=1e-6)


def test_orthogonal_projector_norm_is_one_and_haircut_identity():
    # orthogonal splitting -> constant exactly 1
    P = np.diag([1.0, 0.0, 0.0])
    assert np.linalg.norm(P, 2) == pytest.approx(1.0)
    # haircut: crossing of eps*kappa*e^{lam T} = R shifts by log(kappa)/lam
    lam, eps, R = 0.7, 1e-3, 10.0
    T0 = math.log(R / eps) / lam
    for kappa in [1.5, 5.0, 20.0]:
        Tk = math.log(R / (eps * kappa)) / lam
        assert (T0 - Tk) == pytest.approx(math.log(kappa) / lam, rel=1e-12)


def test_step65b_live_schur_placement_and_leakage():
    import step65b_isotypic_blocks as b
    p = b.part_p(n_draws=20)
    assert p["G_P_placement_forced"]
    s = b.part_s(T_max=12)
    assert s["G_S1_per_block_growth"] and s["G_S2_zero_leakage"]


def test_step95_live_part_a_law_attained():
    import step95_prefactor_angles as a
    res = a.part_a()
    assert res["G_A1_orthogonal_constant_1"]
    assert res["G_A2_oblique_matches_projector_norm"]
    assert res["G_A3_horizon_shift_log_kappa"]
    # the orthogonal case is the matching-constant statement: kappa == 1 exactly
    assert res["cases"][0]["kappa_analytic"] == pytest.approx(1.0)


@pytest.mark.skipif(not S95.exists(), reason="step95 artifact missing")
def test_step95_artifact_gates_consistent():
    d = json.loads(S95.read_text())
    a = d["A_controlled"]
    assert a["G_A1_orthogonal_constant_1"] and a["G_A2_oblique_matches_projector_norm"]
    for c in a["cases"]:
        assert c["rel_err"] < 0.01
    for cell in ["walker-walk-1", "cheetah-run-3"]:
        v = d["C_pretrained"][cell]
        assert v["kappa1_q25"] <= v["kappa1_median"] <= v["kappa1_q75"] <= v["kappa1_max"]
        assert v["kappa1_median"] > 1.0


@pytest.mark.skipif(not S96.exists(), reason="step96 artifact missing")
def test_step96_artifact_gates_reproducible():
    d = json.loads(S96.read_text())
    c = d["cells"]
    for s in ["finger-spin-2", "finger-spin-3"]:
        v = c[s]
        g3 = v["invalid_at_k24"] <= 0.05 and v["fault_recall_at_k8"] >= 0.95 and v["median_delay"] <= 8
        assert g3 == v["G3"] and v["G3"]
        assert v["censored_frac"] > 0.9                              # the stable-abstain signature
    h = c["hopper-hop-1"]
    assert (h["bench_median"] / 1.5 <= h["insitu_median"] <= h["bench_median"] * 1.5) == h["G4"]
    f1 = c["finger-spin-1"]
    assert (abs(f1["ratio_insitu"] - f1["ratio_bench"]) <= 0.25) == f1["G5"]
    assert all(d["verdict"].values())
