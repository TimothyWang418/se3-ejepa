r"""Guard for Step 86 — certified safety / catastrophe-avoidance bound (direction ①).

The experiment reuses step79's tested machinery (controlled dynamics, equivariant WM/planner, certificate); the new
piece is the open-loop-segment safety loop, validated by the smoke run. Here we guard the load-bearing structural
property (the planner the safety agent uses is exactly $\mathbb{Z}_N$-orbit-equivariant) and the run contract.
"""
import torch

from experiments import step79_certified_control as s79
from experiments import step86_certified_safety as s86

DT = torch.float64


def test_equivariant_planner_is_zn_orbit_equivariant():
    r"""The safety agent plans with step79.plan_control, which must be EXACTLY $\mathbb{Z}_N$-orbit-equivariant:
    $\mathrm{plan}(\mathrm{roll}(x_0,s))=\mathrm{roll}(\mathrm{plan}(x_0),s)$ to float round-off (project policy:
    equivariant components carry an equivariance test)."""
    N, s, H = 8, 3, 4
    model = s79.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT)
    sd = torch.ones(N, dtype=DT)
    g = torch.Generator().manual_seed(0)
    x0 = s79.F_FORCE + 0.5 * torch.randn(N, generator=g, dtype=DT)
    u = s79.plan_control(model, x0, mu, sd, H, u_max=4.0, n_iter=10, lr=0.2, seed=0)
    u_roll = s79.plan_control(model, torch.roll(x0, s), mu, sd, H, u_max=4.0, n_iter=10, lr=0.2, seed=0)
    assert torch.allclose(u_roll, torch.roll(u, s, dims=-1), atol=1e-9)


def test_safety_run_contract():
    r"""safety_run (re-observe estimate every tau; plan modest depth from estimate; apply to truth) returns the expected
    keys, a boolean escape flag, never runs past n_steps, and re-observes at least once."""
    N = 8
    model = s79.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT)
    sd = torch.ones(N, dtype=DT)
    x0 = s79.F_FORCE + 2.0 * torch.randn(N, dtype=DT)
    out = s86.safety_run(model, mu, sd, x0, tau=3, n_steps=12, sigma=2.0, H_plan=3, n_iter=5)
    assert {"escaped", "max_dev", "steps", "reobs", "tau"} <= set(out)
    assert isinstance(out["escaped"], bool) and out["steps"] <= 12 and out["tau"] == 3 and out["reobs"] >= 1
