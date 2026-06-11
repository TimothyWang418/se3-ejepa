r"""Artifact-consistency test for the n=15 ring mechanism test (flags and statistics re-derived from raw numbers)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

J = Path(__file__).resolve().parent.parent / "papers/figures/step88_ring_frontier_n15.json"


@pytest.mark.skipif(not J.exists(), reason="artifact missing")
def test_mechanism_stats_reproducible():
    d = json.loads(J.read_text())
    assert d["n_seeds"] == 15 and len(d["per_seed"]) == 15
    infl, win = [], []
    for v in d["per_seed"].values():
        infl.append(v["lambda1_mlp"] / v["lambda1_conv"])
        win.append(v["win_margin"])
    I = np.array(infl) >= 1.5
    W = np.array(win) >= 0.10
    t = d["mechanism_test"]["table_2x2"]
    assert [int((I & W).sum()), int((I & ~W).sum()), int((~I & W).sum()), int((~I & ~W).sum())] == \
           [t["infl_win"], t["infl_nowin"], t["noinfl_win"], t["noinfl_nowin"]]
    _, p = stats.fisher_exact([[t["infl_win"], t["infl_nowin"]], [t["noinfl_win"], t["noinfl_nowin"]]],
                              alternative="greater")
    assert p == pytest.approx(d["mechanism_test"]["fisher_one_sided_p"], rel=1e-6)
    rho, _ = stats.spearmanr(infl, win)
    assert rho == pytest.approx(d["mechanism_test"]["spearman_rho"], rel=1e-6)
    # the pre-registered verdict rule
    expected = "MECHANISM CONFIRMED" if (p < 0.05 and t["noinfl_win"] == 0) else "INCONCLUSIVE-as-registered"
    assert d["mechanism_test"]["verdict"] == expected == "MECHANISM CONFIRMED"
    # the original five published seeds are intact in the merge
    assert d["per_seed"]["0"]["win_margin"] == pytest.approx(0.242)
    assert d["per_seed"]["2"]["win_margin"] == pytest.approx(0.32866666666666666)
