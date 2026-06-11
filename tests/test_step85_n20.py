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
