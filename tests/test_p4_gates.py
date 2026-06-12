r"""Gate-mechanics equivalence + semantics tests for the canonical src/audit/gates.py.

- GE-I: canonical boundary_from_curve == historical stage1a implementation on random curves.
- GE-II: faithful_guar semantics table — (0,0) passes, cert>meas fails, censored (8,8) passes.
- GE-III: cal_band excludes non-positive cells; empty-evaluable returns False.

Run: .venv/bin/python tests/test_p4_gates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_spine_stage1a import boundary_from_curve as hist_boundary  # noqa: E402
from src.audit.gates import (  # noqa: E402
    boundary_from_curve, cal_band, dual_boundary_cells, faithful_guar, violations,
)


def ge1_equivalence() -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        curve = np.abs(rng.normal(1, 1, size=8)).cumsum() * rng.uniform(0.1, 3)
        eps = float(rng.uniform(0, curve[-1] * 1.5))
        a, b = boundary_from_curve(curve, eps), hist_boundary(curve, eps)
        assert a == b, f"GE-I FAIL: canonical {a} != historical {b} (eps={eps}, curve={curve})"
    print("GE-I   PASS  500 random curves, canonical == historical")


def ge2_guar_semantics() -> None:
    mk = lambda hc, hm: {"em": 2, "h_cert": hc, "h_meas": hm, "ratio": None}  # noqa: E731
    assert faithful_guar([mk(0, 0)]), "(0,0) must pass"
    assert faithful_guar([mk(8, 8)]), "censored (8,8) must pass (caveat registered)"
    assert not faithful_guar([mk(1, 0)]), "cert>meas=0 must FAIL (anticonservative)"
    assert not faithful_guar([mk(5, 3)]), "cert>meas must FAIL"
    assert violations([mk(1, 0), mk(0, 0)]) == [mk(1, 0)]
    print("GE-II  PASS  guar semantics table")


def ge3_cal_semantics() -> None:
    cells = dual_boundary_cells([0.5, 1.0, 2.0], [1.0, 2.0, 4.0], delta_mean=0.3,
                                eps_mult=(1, 4, 16))
    # em=1: eps=0.3 < 0.5 -> both 0 -> ratio None; em=4: eps=1.2 -> meas 2, cert 1
    assert cells[0]["ratio"] is None and cells[0]["h_meas"] == 0
    assert not cal_band([{"em": 2, "h_cert": 0, "h_meas": 0, "ratio": None}]), \
        "no evaluable cell -> False"
    assert cal_band([{"em": 2, "h_cert": 2, "h_meas": 3, "ratio": 2 / 3}])
    print("GE-III PASS  cal semantics + dual cells")


if __name__ == "__main__":
    ge1_equivalence()
    ge2_guar_semantics()
    ge3_cal_semantics()
    print("ALL GATE TESTS PASS")
