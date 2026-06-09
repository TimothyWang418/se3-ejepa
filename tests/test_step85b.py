r"""Guard for Step 85b — full-spectrum allocation (direction ③, extension C).

Fast unit tests for the new pure logic ``allocate`` (split a fixed total observation budget across chaoticity regimes
by certified-need weights). The allocation experiment itself reuses the tested ``step85.reobserve_run_budgeted`` per
regime, so only the budget split needs its own tests. (design: docs/specs/2026-06-08-step85b-spectrum-allocation-seed.md)
"""
from experiments import step85b_spectrum_allocation as s85b


def test_allocate_proportional_exact():
    r"""Budget splits proportionally to the (normalized) weight vector when it divides evenly."""
    assert s85b.allocate([0.1, 0.2, 0.3, 0.4], 100) == [10, 20, 30, 40]
    assert s85b.allocate([0.1, 0.2, 0.3, 0.4], 10) == [1, 2, 3, 4]


def test_allocate_preserves_total_via_largest_remainder():
    r"""The split always sums to EXACTLY the total budget (largest-remainder rounding) — required for a matched-budget
    comparison of conv-allocation vs MLP-allocation."""
    for w, B in [([0.11, 0.2, 0.29, 0.39], 37), ([0.25, 0.25, 0.25, 0.25], 10), ([0.7, 0.1, 0.1, 0.1], 13)]:
        b = s85b.allocate(w, B)
        assert sum(b) == B and len(b) == len(w)


def test_allocate_unnormalized_weights_ok():
    r"""Weights need not be pre-normalized (lambda1(F) values are passed raw)."""
    b = s85b.allocate([1.0, 2.0, 3.0, 4.0], 100)       # same proportions as [.1,.2,.3,.4]
    assert b == [10, 20, 30, 40]


def test_allocate_flatter_weights_shift_budget_to_easy_regimes():
    r"""The C mechanism: a FLATTER weight vector (the MLP's range-compressed spectrum) moves budget away from the
    hardest regime toward the easiest, relative to a steeper (true/conv) vector — at the same total."""
    steep = s85b.allocate([0.11, 0.20, 0.29, 0.39], 100)   # true-like
    flat = s85b.allocate([0.20, 0.22, 0.26, 0.32], 100)    # MLP-like (compressed)
    assert flat[0] > steep[0]      # easiest regime gets MORE under the flat (MLP) allocation
    assert flat[-1] < steep[-1]    # hardest regime gets LESS -> it starves -> higher violation
