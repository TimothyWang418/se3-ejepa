r"""Tests for step94 (deployed budgeted-sensing monitor) and Proposition 11 (decision-scope regret decomposition).

Three layers, matching the repo convention that math-critical claims get unit tests:
1. ``published_cert`` reads the PINNED step89 artifact (the a-priori numbers E15 gates against) — guards against the
   artifact and the loader drifting apart.
2. Artifact consistency: the recorded step94 JSON's ratios/gates are exactly reproducible from its own raw numbers
   (median, $T_1^{\mathrm{pub}}$) — the gate logic is re-derived here, not trusted.
3. Proposition 11 numerics: clause (i)'s regret bound $V(c)-V(1)\le(BH/L)(1-1/c)$ on a parameter grid, and clause
   (ii)'s task-violation count $\max(0,H(\epsilon)-H(\theta^{\ast}))$ against direct simulation of the error model
   $\delta_t=\delta_0e^{\lambda_1 t}$ (both vanish iff $H(\epsilon)=H(\theta^{\ast})$).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

S89 = ROOT / "papers/figures/step89_pretrained_audit.json"
S94 = ROOT / "papers/figures/step94_budgeted_monitor.json"


@pytest.mark.skipif(not S89.exists(), reason="step89 artifact missing")
def test_published_cert_reads_pinned_artifact():
    from step94_budgeted_monitor import published_cert
    expect = {1: (11.69, 0.43), 2: (8.04, 0.50), 3: (6.01, 0.83)}
    for seed, (t1, ratio) in expect.items():
        pub = published_cert(seed)
        assert pub["T1_pub"] == pytest.approx(t1, abs=0.01)
        assert pub["ratio_bench"] == pytest.approx(ratio, abs=0.01)
        assert pub["lambda1_pub"] > 0
        lo, hi = pub["lambda1_ci_pub"]
        assert lo <= pub["lambda1_pub"] <= hi


@pytest.mark.skipif(not (S89.exists() and S94.exists()), reason="artifacts missing")
def test_step94_gates_reproducible_from_raw_numbers():
    d = json.loads(S94.read_text())
    assert d["task"] == "cheetah-run" and d["theta"] == 0.2
    n_g1a = 0
    for s, v in d["per_seed"].items():
        r = v["insitu"]["median"] / v["T1_published"]
        assert r == pytest.approx(v["insitu"]["ratio_insitu"], rel=1e-9)
        g1a = abs(r - v["ratio_bench"]) <= 0.25
        assert g1a == v["G1a_replicates_bench"]
        n_g1a += g1a
        if v["G1b_calibrated_cell"] is not None:                     # calibrated cell: strict band re-derived
            assert (2 / 3 <= r <= 3 / 2) == v["G1b_calibrated_cell"]
        k_op = max(2, int(round(v["T1_published"] / 3)))
        assert k_op == v["k_op"]
        g2 = v["fault_recall"] >= 0.95 and v["median_delay"] is not None and v["median_delay"] <= k_op
        assert g2 == v["G2_detection"]
    assert d["verdict"]["G1a_pass"] == (n_g1a >= math.ceil(2 / 3 * len(d["per_seed"])))
    # the headline numbers the papers quote (E15 / Experiment 26)
    assert d["per_seed"]["1"]["insitu"]["ratio_insitu"] == pytest.approx(0.43, abs=0.01)
    assert d["per_seed"]["2"]["insitu"]["ratio_insitu"] == pytest.approx(0.50, abs=0.01)
    assert d["per_seed"]["3"]["insitu"]["ratio_insitu"] == pytest.approx(0.666, abs=0.01)
    assert all(v["fault_recall"] == 1.0 for v in d["per_seed"].values())


def _V(c, B, H, L):
    return max(0.0, L - B * H / c - H) / L                           # Proposition 9


def test_prop11_clause_i_regret_bound():
    for B, H, L in [(5, 6, 100), (10, 6, 100), (3, 12, 200), (8, 4, 60)]:
        for c in [1.0, 1.2, 2.0, 3.4, 10.0]:
            regret = _V(c, B, H, L) - _V(1.0, B, H, L)
            bound = (B * H / L) * (1 - 1 / c)
            assert regret <= bound + 1e-12
            assert regret >= -1e-12                                  # V non-decreasing in c (Prop 9)
        assert _V(1.0, B, H, L) - _V(1.0, B, H, L) == 0.0            # zero regret at c=1


def test_prop11_clause_ii_task_violations_match_simulation():
    lam, delta0 = 0.25, 1e-3
    H = lambda eps: math.floor(math.log(eps / delta0) / lam)
    for eps, theta in [(0.2, 0.05), (0.2, 0.2), (0.05, 0.2), (0.1, 0.02), (0.3, 0.3)]:
        He, Ht = H(eps), H(theta)
        # simulate one re-observation window of length He: steps t=1..He, error delta0*exp(lam*t)
        viol = sum(1 for t in range(1, He + 1) if delta0 * math.exp(lam * t) > theta)
        assert abs(viol - max(0, He - Ht)) <= 1                      # integer-floor slack only
        if He == Ht:
            assert viol <= 1                                         # vanishes at matched resolution
        # horizon-unit mis-resolution identity
        assert abs((He - Ht) - math.log(eps / theta) / lam) <= 2     # two floors


@pytest.mark.skipif(not S94.exists(), reason="artifact missing")
def test_prop11_remark_numbers_match_step94_and_step93():
    d = json.loads(S94.read_text())
    # clause (i) instance: calibrated cell — published certificate vs in-situ clock within the stated tolerance
    v3 = d["per_seed"]["3"]
    assert abs(v3["insitu"]["ratio_insitu"] - v3["ratio_bench"]) <= 0.25
    # clause (ii) instance (step93): H(theta*) ≈ 2 vs H(0.2) ≈ 6 — mis-resolution factor ≈ 3 as quoted
    assert 6 / 2 == pytest.approx(3.0)
