"""E0/step82 at n=10 (completionist batch): merged tightness ledger across 4 seed-batches."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_step82_n10_tightness_and_soundness():
    d = json.loads((ROOT / "papers/figures/step82_n10_merged.json").read_text())
    assert d["n_seeds_total"] == 10
    cat = d["per_system"]["CatMap(learned)"]
    assert 1.17 <= cat["tightness_min"] <= 1.18 and 1.25 <= cat["tightness_max"] <= 1.27   # cited range
    assert d["per_system"]["CatMap(linear,true)"]["tightness_max"] < 1.0001                # exact on true map
    hen = d["per_system"]["Henon(true)"]
    assert 3.1 < hen["tightness_min"] and hen["tightness_max"] < 3.25                        # ~3x conservative
    assert d["per_system"]["Henon(true)"]["cone_margin"] < 0                                 # negative cone -> abstain
    assert all(v == 1.0 for v in d["G2_soundness_coverage_by_batch"].values())               # never over-promises
