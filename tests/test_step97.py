r"""Artifact-consistency tests for step97 (pixel-family deployment cell: bias-abstain => zero sensing savings)."""
import json
from pathlib import Path

import pytest

S97 = Path(__file__).resolve().parent.parent / "papers/figures/step97_lewm_monitor.json"


@pytest.mark.skipif(not S97.exists(), reason="artifact missing")
def test_step97_gates_reproducible_from_raw_numbers():
    d = json.loads(S97.read_text())
    arm = d["arms"]["zero_action"]
    inv, fr = arm["invalid_by_k"], arm["flag_rate_by_k"]
    assert (all(inv[str(k)] > 0.25 for k in [2, 4, 8])) == arm["G6_no_usable_cadence"] and arm["G6_no_usable_cadence"]
    assert (all(fr[str(k)] >= 0.35 for k in [2, 4, 8])) == arm["G7_flooded_channel"] and arm["G7_flooded_channel"]
    # k=1 is the no-forecasting baseline: clean, and NOT part of the gates (v2 design note)
    assert inv["1"] <= 0.25 and fr["1"] < 0.35
    # the deterministic staleness-2 structure: invalid fraction == (k-1)/k
    for k in [2, 4, 8]:
        assert inv[str(k)] == pytest.approx((k - 1) / k, abs=0.02)
    # telemetry fault inseparable from drift (flooded channel)
    pre, post = arm["telemetry_fault_pre_post"]
    assert pre >= 0.9 and post >= 0.9
    # moving-scene control reproduces the verdict
    assert d["arms"]["moving_scene"]["invalid_by_k"]["4"] > 0.25
