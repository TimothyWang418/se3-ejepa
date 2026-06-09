r"""Guard for Step 85 — structure → trustworthy certificate → downstream win (direction ③).

Fast, training-free unit tests for the NEW Phase-1 logic in
``experiments/step85_trustworthy_cert_downstream.py`` (design:
docs/specs/2026-06-08-step85-trustworthy-cert-downstream-design.md, revision 2026-06-09a). The headline mechanism is
**budget starvation**: an agent that re-observes too often (a too-short interval, e.g. from the non-equivariant model's
inflated $\lambda_1$) exhausts a finite observation budget early and runs the episode tail open-loop, raising the
aggregate violation rate. We test that bookkeeping deterministically with a drift stub: truth is held constant and the
world model drifts by a fixed vector, so the open-loop forecast error after $k$ steps is exactly $k\lVert d\rVert/
\lVert x_0\rVert$ and the empirical horizon $k^\*$ is known in closed form.
"""
import numpy as np
import torch
import torch.nn as nn

from experiments import step85_trustworthy_cert_downstream as s85

DT = torch.float64


class _DriftWM(nn.Module):
    r"""Stub world model: ``m(x) = x + d`` (a fixed drift ``d``), action ignored. An ``nn.Module`` so ``.eval()`` works.
    Rolled open-loop from a constant truth $x_0$, the forecast after $k$ steps is $x_0 + k d$, so the relative error is
    $k\lVert d\rVert/\lVert x_0\rVert$ — linear and exactly predictable."""

    def __init__(self, drift: torch.Tensor):
        super().__init__()
        self.register_buffer("d", drift.to(DT))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + self.d


def _drift_setup(N=4, x0_val=1.0, drift0=0.1, T_total=30):
    r"""Constant-truth + drift-stub fixture. ||x0|| = x0_val*sqrt(N); ||d|| = drift0. With eps=0.2 below, the empirical
    horizon k* = floor-ish of eps*||x0||/||d|| + 1. Defaults: ||x0||=2, ||d||=0.1, eps=0.2 -> rel err = 0.05k, first
    EXCEEDS 0.2 at k*=5."""
    x0 = torch.full((N,), x0_val, dtype=DT)
    drift = torch.zeros(N, dtype=DT)
    drift[0] = drift0
    x_seq_true = x0.repeat(T_total + 1, 1)                 # (T_total+1, N) constant true trajectory
    mu = torch.zeros(N, dtype=DT)
    sd = torch.ones(N, dtype=DT)                            # identity normalization
    return _DriftWM(drift), mu, sd, N, x_seq_true


def test_budgeted_reobserve_shorter_interval_starves_more():
    r"""Core mechanism: at a FIXED budget, a SHORTER re-observation interval covers less of the episode (the last
    re-observation lands earlier), so the open-loop tail is longer and the aggregate violation HIGHER. This is exactly
    why the MLP's inflated-lambda1 (too-short) interval loses under budget."""
    model, mu, sd, N, x_seq = _drift_setup()
    eps = 0.2
    # both intervals < k*=5 so covered windows are violation-free; only the starved tail violates.
    short = s85.reobserve_run_budgeted(model, mu, sd, N, x_seq, interval=2, eps=eps, budget=4)
    long = s85.reobserve_run_budgeted(model, mu, sd, N, x_seq, interval=4, eps=eps, budget=4)
    assert short["violation_rate"] > long["violation_rate"]
    assert short["violation_rate"] > 0.0 and long["violation_rate"] > 0.0   # both starve (budget too small to cover 30)
    # covered_steps = (budget-1)*interval : the last re-observation step.
    assert short["covered_steps"] == 6 and long["covered_steps"] == 12


def test_budgeted_reobserve_enough_budget_covers_episode():
    r"""With enough budget at a sound interval (< k*), the whole episode is covered and aggregate violation ~ 0."""
    model, mu, sd, N, x_seq = _drift_setup(T_total=30)
    out = s85.reobserve_run_budgeted(model, mu, sd, N, x_seq, interval=4, eps=0.2, budget=8)
    assert out["covered_steps"] == 28                       # last reset at t=28, tail (29,30) errors 0.05,0.10 < 0.2
    assert out["violation_rate"] < 0.05                     # ~zero: the episode is covered


def test_budget_frontier_more_budget_weakly_lowers_violation():
    r"""The Phase-1 frontier: aggregate violation is (weakly) monotone DECREASING in observation budget — more budget
    covers more of the episode. Returns one row per budget."""
    model, mu, sd, N, x_seq = _drift_setup(T_total=30)
    rows = s85.budget_frontier(model, mu, sd, N, x_seq, interval=4, eps=0.2, budgets=[2, 4, 8])
    viols = [r["violation_rate"] for r in rows]
    assert viols[0] >= viols[1] >= viols[2]                 # monotone non-increasing in budget
    assert [r["budget"] for r in rows] == [2, 4, 8]


def test_adaptive_reobserve_learns_efficient_cadence_without_certificate():
    r"""The red-team baseline: a CERTIFICATE-FREE adaptive agent (AIMD on the interval — grow when a window stayed under
    $\epsilon$, shrink when it overshot) learns an efficient, safe cadence from the error it observes at re-observation.
    On the drift stub (true horizon $k^\*{=}5$) it grows the interval up from the timid init toward $k^\*$, so it uses
    far fewer observations than always-observing (interval 1) while keeping violations bounded. This is what makes the
    certificate's a-priori cadence only a WARM-START advantage (it pays off in the tight-budget/few-shot regime), the
    honest scope of direction ③."""
    model, mu, sd, N, x_seq = _drift_setup(T_total=60)
    eps = 0.22                                               # off the window boundary (rel=0.05k): k=4 -> 0.20 clearly
    #                                                          stays under (grow), k=5 -> 0.25 overshoots (shrink), k*=5
    out = s85.adaptive_reobserve(model, mu, sd, N, x_seq, eps=eps, budget=60,
                                 init_interval=1, grow=1, shrink=0.5)
    always = s85.reobserve_run_budgeted(model, mu, sd, N, x_seq, interval=1, eps=eps, budget=60)
    assert out["n_observations"] < always["n_observations"]   # learned to WAIT: fewer looks than always-observe
    assert out["max_interval"] >= 5 and out["mean_interval"] >= 3.0   # grew the interval toward k*=5 from init=1
    assert out["violation_rate"] <= 0.2                      # adapts down on overshoot -> stays safe-ish


# --- guard tests on the dependencies the design rests on (pass immediately: existing/known behavior) --- #

def test_conv_is_zn_equivariant_and_mlp_is_not():
    r"""The structural fact the whole paper rests on: the cyclic-conv world model is EXACTLY $\mathbb{Z}_N$-equivariant
    ($f(\mathrm{roll}(x,s))=\mathrm{roll}(f(x),s)$ to float round-off), the dense MLP is not. Equivariant layers MUST
    carry an equivariance test (project policy)."""
    from experiments import step74_lorenz96_spectrum as s74
    N, s = 12, 3
    torch.manual_seed(0)
    conv = s74.L96CyclicConv(N).double().eval()
    mlp = s74.L96MLP(N).double().eval()
    x = torch.randn(N, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(conv(torch.roll(x, s)), torch.roll(conv(x), s), atol=1e-9)        # exact equivariance
        assert not torch.allclose(mlp(torch.roll(x, s)), torch.roll(mlp(x), s), atol=1e-6)      # MLP breaks it


def test_certified_T1_steps_units_inflation_and_abstain():
    r"""The units fix + the MLP mechanism: an inflated (here $2\times$) $\lambda_1$ halves the certified horizon
    $T_1=\log(1/\epsilon)/\lambda_1$ (converted to map steps via $\Delta t_{\rm map}$); an abstaining certificate
    ($T_1{=}$None) returns the most-conservative 1 step."""
    import math
    from experiments import step79_certified_control as s79, step74_lorenz96_spectrum as s74
    eps, lam = 0.2, 1.6
    t1a = math.log(1.0 / eps) / lam
    t1b = math.log(1.0 / eps) / (2.0 * lam)                  # double lambda1 (the MLP inflation) -> half the horizon
    sa = s79.certified_T1_steps({"T1": t1a})
    sb = s79.certified_T1_steps({"T1": t1b})
    assert sa == max(1, round(t1a / s74.DTMAP))              # units fix: T1 / dt_map, not round(T1)
    assert abs(sb - sa / 2.0) <= 1                            # inflated lambda1 -> ~half certified horizon
    assert s79.certified_T1_steps({"T1": None}) == 1         # abstain -> most conservative (1 step)
