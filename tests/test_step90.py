r"""Guard for Step 90 — UQ head-to-head: certificate vs ensemble-disagreement vs conformal horizon.

Stub-level tests for the two NEW pure-logic pieces: the ensemble-spread horizon (first step where member spread
exceeds eps — computable WITHOUT truth) and the conformal-quantile horizon predictor. The expensive arms (training
K members) are exercised by the experiment itself.
"""
import torch

from experiments import step90_uq_baselines as s90

DT = torch.float64


class _Lin(torch.nn.Module):
    r"""Stub forecaster x -> a*x (autonomous, model(x, u) signature)."""

    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, x, u=None):
        return self.a * x


def test_ensemble_spread_horizon_known_crossing():
    r"""Two members growing as $1.1^t$ and $0.9^t$ from the same start: relative spread vs the ensemble mean is
    computable in closed form; the first eps-crossing matches a brute-force scan."""
    m1, m2 = _Lin(1.1).double(), _Lin(0.9).double()
    x0 = torch.ones(2, 4, dtype=DT)                              # 2 starts, N=4
    mu = torch.zeros(4, dtype=DT); sd = torch.ones(4, dtype=DT)
    h = s90.ensemble_spread_horizon([m1, m2], mu, sd, x0, eps=0.2, max_steps=60)
    # brute force
    import math
    for t in range(1, 61):
        a, b = 1.1 ** t, 0.9 ** t
        mean = (a + b) / 2
        spread = max(abs(a - mean), abs(b - mean)) / mean        # state is uniform => relative spread is scalar
        if spread > 0.2:
            expect = t
            break
    assert h["per_start"] == [expect, expect]
    assert h["median"] == float(expect)


def test_conformal_quantile_predictor():
    r"""Conformal-style horizon = the alpha-quantile of calibration horizons (lower quantile = conservative)."""
    cal = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    assert s90.conformal_horizon(cal, alpha=0.5) == 19.0          # median
    lo = s90.conformal_horizon(cal, alpha=0.1)
    assert lo <= 12                                               # conservative lower quantile
