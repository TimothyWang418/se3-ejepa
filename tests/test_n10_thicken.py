"""Ledger tests for the 2026-06-12 completionist thickening: step90 n=10 and step92 100-start column."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_step90_n10_near_parity():
    d = json.loads((ROOT / "papers/figures/step90_n10_merged.json").read_text())
    s = d["summary"]
    assert d["n_seeds"] == 10 and all(s[k]["n"] == 10 for k in ("cert", "ensemble", "conformal"))
    mal = {k: s[k]["mean_abs_log_ratio"] for k in s}
    assert 0.15 < mal["conformal"] < mal["ensemble"] < mal["cert"] < 0.26   # ordering kept, gaps narrow
    assert mal["cert"] - mal["conformal"] < 0.06                            # near-parity at n=10


def test_step92_ladder_start_count_stable():
    d = json.loads((ROOT / "papers/figures/step92_scale_sweep.json").read_text())
    cells = {v["cell"]: v for v in d["cells"]}
    signs = [cells[c]["lambda1"] > 0 for c in ("mt30-1M", "mt30-5M", "mt30-19M", "mt30-48M", "mt30-317M")]
    assert signs == [False, True, True, False, True]                        # non-monotonic regime flips
    r = {c: [x for x in cells[c]["rows"] if abs(x["eps"] - 0.2) < 1e-9][0]["ratio"]
         for c in ("mt30-5M", "mt30-19M", "mt30-317M")}
    assert abs(r["mt30-5M"] - 0.39) < 0.01 and abs(r["mt30-19M"] - 1.91) < 0.01 and abs(r["mt30-317M"] - 1.22) < 0.01
