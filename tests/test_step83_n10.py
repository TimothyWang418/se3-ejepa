"""E2 pillar at n=10: the merged crossover artifact is the ledger — zero overlap at N=40."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "papers/figures/step83_n10_merged.json"


def test_crossover_n10_zero_overlap():
    d = json.loads(MERGED.read_text())
    cells = d["cells"]
    for N, archs in cells.items():
        for a in ("conv", "mlp", "gru"):
            assert len(archs[a]) == 10, (N, a)
            assert len({r["seed"] for r in archs[a]}) == 10
    c40 = cells["40"]
    conv = [r["spectrum_r2"] for r in c40["conv"]]
    mlp = [r["spectrum_r2"] for r in c40["mlp"]]
    gru = [r["spectrum_r2"] for r in c40["gru"]]
    assert min(conv) > 0.97                       # all ten hold
    assert max(mlp) < 0 and max(gru) < 0          # all twenty dense-family fits collapse
    assert min(conv) > max(mlp) + 1.0             # zero overlap, with margin
    assert abs(float(np.median(mlp)) + 1.378) < 0.01
