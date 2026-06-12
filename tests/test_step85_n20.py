"""E12 n=20 thickening: the merged-margin artifact is the ledger of the 20/20 result."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "papers/figures/step85_n20_merged.json"


def test_merged_margins_n20_all_win():
    d = json.loads(MERGED.read_text())
    m = np.array(d["win_margins"])
    assert d["n_seeds"] == len(m) == 20
    assert (m > 0).all() and d["all_win"]
    assert 0.40 < m.min() and m.max() < 0.62
    assert abs(np.median(m) - d["median"]) < 1e-12
    lo, hi = d["ci95_median"]
    assert lo > 0.45 and hi < 0.56                      # the cited bootstrap CI [0.48, 0.54]


def test_step85c_recalibration_closes_n10():
    d = json.loads((ROOT / "papers/figures/step85c_n10_merged.json").read_text())
    assert d["n_seeds"] == 10 and d["gap_closed_count"] == 10
    assert 0.4 <= d["raw_gap_med"] <= 0.6 and abs(d["recal_gap_med"]) < 0.12


def test_catchup_factor_n20_brackets_prediction():
    import glob
    base = json.loads((ROOT / "papers/figures/step85_phase1_frontier.json").read_text())
    ps = dict(base["per_seed"])
    for f in sorted(glob.glob(str(ROOT / "papers/figures/step85_phase1_frontier_seed*.json"))):
        ps.update(json.loads(Path(f).read_text())["per_seed"])
    fac = np.array([v["cover_mlp_budget"] / v["knee_budget"] for v in ps.values()
                    if v.get("cover_mlp_budget") and v.get("knee_budget")])
    assert len(fac) == 20
    assert abs(float(np.median(fac)) - 2.97) < 0.05
    assert fac.min() > 2.0 and fac.max() < 4.5          # brackets the predicted c~3.4
