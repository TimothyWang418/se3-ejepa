"""step100 walker-S2 bridge — n=10 negative-result ledger (INCONCLUSIVE-BY-LEVEL-DOMINANCE).

Pins the honest verdict: G-EQ passes at full strength (eq equivariance defect literally 0, dense ~1),
but every arm's measured horizon is level-set (median 1 at eps=0.2) so the structure-vs-dense
certificate comparison never opens at this training scale — and the v2 escalation did not break it.
Also pins the G-SYM finding (the walker is exactly S2-equivariant).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_gsym_walker_exactly_s2_equivariant():
    d = json.loads((ROOT / "papers/figures/step100_gsym_probe.json").read_text())
    assert d["verdict_exact"] and d["defect_median"] < 1e-6        # measured fact: walker is exactly S2


def test_step100_n10_level_dominance_verdict():
    d = json.loads((ROOT / "papers/figures/step100_walker_s2_results.json").read_text())
    cells = {k: v for k, v in d.items() if "-" in k and "error" not in v}
    assert len(cells) == 30
    for arm in ("eq", "dense", "aug"):
        ks = [f"{arm}-{s}" for s in range(10)]
        assert all(k in cells for k in ks)
        # level-dominance: every arm pins measured median = 1 at eps=0.2 (no growth-set precision reached)
        meds = {next(r for r in cells[k]["rows"] if r["eps"] == 0.2)["measured_median"] for k in ks}
        assert meds == {1.0}, (arm, meds)
    # G-EQ at full strength: eq exact, dense not
    assert max(cells[f"eq-{s}"]["enc_defect"] for s in range(10)) == 0.0
    assert min(cells[f"dense-{s}"]["enc_defect"] for s in range(10)) > 0.1
