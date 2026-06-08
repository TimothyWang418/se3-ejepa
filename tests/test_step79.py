r"""Guard for Step 79 — certified-control co-demonstration on controlled Lorenz-96."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step79_certified_control as s79  # noqa: E402
DT = torch.float64


def test_controlled_dynamics_is_ZN_equivariant() -> None:
    torch.manual_seed(0); N = 12
    x = torch.randn(N, dtype=DT); u = torch.randn(N, dtype=DT)
    for s in (1, 3, 7):
        lhs = s79.l96_controlled_rhs(torch.roll(x, s), torch.roll(u, s))
        rhs = torch.roll(s79.l96_controlled_rhs(x, u), s)
        assert torch.allclose(lhs, rhs, atol=1e-12), f"shift {s}: field not Z_N-equivariant"
        mlhs = s79.rk4_controlled(torch.roll(x, s), torch.roll(u, s))
        mrhs = torch.roll(s79.rk4_controlled(x, u), s)
        assert torch.allclose(mlhs, mrhs, atol=1e-12), f"shift {s}: RK4 map not equivariant"
    print("PASS: controlled Lorenz-96 field and RK4 map are Z_N-equivariant.")


def test_world_model_conv_is_equivariant_mlp_is_not() -> None:
    torch.manual_seed(0); N = 12
    conv = s79.L96ControlledConv(N).double()
    mlp = s79.L96ControlledMLP(N).double()
    x = torch.randn(N, dtype=DT); u = torch.randn(N, dtype=DT)
    for s in (1, 5):
        c_lhs = conv(torch.roll(x, s), torch.roll(u, s))
        c_rhs = torch.roll(conv(x, u), s, dims=-1)
        assert torch.allclose(c_lhs, c_rhs, atol=1e-10), f"conv shift {s}: not equivariant"
    m_lhs = mlp(torch.roll(x, 1), torch.roll(u, 1))
    m_rhs = torch.roll(mlp(x, u), 1, dims=-1)
    assert not torch.allclose(m_lhs, m_rhs, atol=1e-6), "MLP baseline should NOT be Z_N-equivariant"
    print("PASS: controlled conv WM is Z_N-equivariant; MLP baseline is not.")


def test_planner_is_orbit_equivariant() -> None:
    torch.manual_seed(0); N = 12
    model = s79.make_equivariant_wm(N).double()       # untrained equivariant WM is enough (equivariant by construction)
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    x0 = torch.randn(N, dtype=DT)
    for s in (1, 5):
        u_a = s79.plan_control(model, x0, mu, sd, H=6, seed=0)
        u_b = s79.plan_control(model, torch.roll(x0, s), mu, sd, H=6, seed=0)
        assert torch.allclose(u_b, torch.roll(u_a, s, dims=-1), atol=1e-6), \
            f"shift {s}: planner not orbit-equivariant (max diff {(u_b-torch.roll(u_a,s,-1)).abs().max():.2e})"
    print("PASS: planner commutes with the Z_N shift (exactly orbit-equivariant).")


def test_closed_loop_control_is_orbit_flat() -> None:
    torch.manual_seed(0); N = 12
    model = s79.make_equivariant_wm(N).double()      # untrained equivariant WM: planner still exactly equivariant
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    x0 = torch.randn(N, dtype=DT)
    of = s79.orbit_flatness(model, x0, mu, sd, H=5, n_steps=8, s=3, u_max=4.0, n_iter=10)
    assert of["control_mismatch"] < 1e-8, f"closed-loop control not orbit-flat: mismatch {of['control_mismatch']:.2e}"
    assert abs(of["ratio"] - 1.0) < 1e-6, f"closed-loop cost ratio {of['ratio']:.6f} not ~1.000"
    print(f"PASS: closed-loop control orbit-flat (mismatch {of['control_mismatch']:.1e}, ratio {of['ratio']:.6f}).")


def test_stabilize_run_is_orbit_equivariant() -> None:
    # Phase 5a: stabilize_run wraps the exactly-equivariant closed-loop MPC, so a cyclic shift of the near-F start x0
    # produces an exactly shifted applied-control sequence (and trajectory) on the equivariant WM. Untrained WM suffices
    # (the planner is equivariant by construction). This guards that the Phase-5a metrics layer did not break the orbit
    # symmetry the co-demonstration's configuration axis rests on.
    torch.manual_seed(0); N = 12
    model = s79.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    s79.stabilize_run._sigma = 2.0
    x0 = s79.F_FORCE + 2.0 * torch.randn(N, dtype=DT)         # a near-F start
    for s in (1, 5):
        a = s79.stabilize_run(model, mu, sd, x0, H=4, n_steps=6, u_max=4.0, n_iter=10, seed=0)
        b = s79.stabilize_run(model, mu, sd, torch.roll(x0, s), H=4, n_steps=6, u_max=4.0, n_iter=10, seed=0)
        mismatch = (b["controls"] - torch.roll(a["controls"], s, dims=-1)).norm(dim=-1).max()
        assert float(mismatch) < 1e-8, f"shift {s}: stabilize_run controls not orbit-equivariant (max {float(mismatch):.2e})"
        traj_mismatch = (b["traj"] - torch.roll(a["traj"], s, dims=-1)).norm(dim=-1).max()
        assert float(traj_mismatch) < 1e-8, f"shift {s}: stabilize_run traj not orbit-equivariant (max {float(traj_mismatch):.2e})"
    print("PASS: stabilize_run is exactly Z_N-orbit-equivariant in the near-F start (controls + traj).")


def test_certified_T1_steps_units_conversion() -> None:
    # Phase 5b UNITS FIX: certificate()['T1'] is in TIME units; the certified horizon in MAP STEPS is T1/DTMAP, NOT
    # round(T1). With DTMAP=0.01, a T1 of ~3 time units must convert to ~300 steps (NOT 3). Guards the load-bearing
    # conversion the whole re-observation decision rests on. Also: abstain (T1=None) -> 1, and the >=1 clamp holds.
    import step74_lorenz96_spectrum as s74
    assert abs(s74.DTMAP - 0.01) < 1e-12, f"this test assumes DTMAP=0.01, got {s74.DTMAP}"
    steps = s79.certified_T1_steps({"T1": 3.0701})
    assert steps == round(3.0701 / s74.DTMAP) == 307, f"expected 307 map steps, got {steps}"   # NOT round(3.07)=3
    assert steps != 3, "units bug: certified_T1_steps must NOT collapse a ~3-time-unit horizon to ~3 steps"
    assert s79.certified_T1_steps({"T1": None}) == 1, "abstain (T1=None) must fall back to 1 step"
    assert s79.certified_T1_steps({"T1": 0.001}) == 1, "horizon must be clamped to >= 1 step"
    print(f"PASS: certified_T1_steps converts TIME->MAP STEPS (T1=3.07 -> {steps} steps, not 3).")


def test_reobserve_run_interval_monotonicity() -> None:
    # Phase 5b: reobserve_run with interval=1 re-observes EVERY step (never blind) -> ~zero violations and
    # n_observations ~= T_total; with interval=T_total it observes only at the ends -> the MOST violations. The WM here
    # is the (untrained) equivariant conv; its u=0 map differs from the true dynamics, so a long blind forecast WILL
    # accumulate error -> the monotonicity is structural (more re-observation => no more violations than less).
    torch.manual_seed(0); N = 12
    model = s79.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    T_total = 60
    # A nontrivial true trajectory (uncontrolled field) the WM must forecast.
    import step74_lorenz96_spectrum as s74
    x_seq = s74.attractor_traj(N, T_total, 0, "cpu").double()[:T_total + 1]   # (T_total+1, N)
    r_obs1 = s79.reobserve_run(model, mu, sd, N, x_seq, interval=1, eps=0.01)
    r_full = s79.reobserve_run(model, mu, sd, N, x_seq, interval=T_total, eps=0.01)
    # interval=1: observed every step -> all recorded errors are 0 -> zero violations; n_obs = T_total + 1 (incl. t=0).
    assert r_obs1["violation_rate"] == 0.0, f"interval=1 should have zero violations, got {r_obs1['violation_rate']}"
    assert r_obs1["n_observations"] == T_total + 1, \
        f"interval=1 should observe every step (n_obs={T_total + 1}), got {r_obs1['n_observations']}"
    # interval=T_total observes the fewest times and must have the HIGHEST violation rate of the two.
    assert r_full["n_observations"] < r_obs1["n_observations"], "interval=T_total must observe fewer times"
    assert r_full["violation_rate"] >= r_obs1["violation_rate"], \
        f"longer interval must not have FEWER violations ({r_full['violation_rate']} vs {r_obs1['violation_rate']})"
    print(f"PASS: reobserve_run monotone — interval=1 viol={r_obs1['violation_rate']:.2f} "
          f"(n_obs={r_obs1['n_observations']})  <=  interval={T_total} viol={r_full['violation_rate']:.2f} "
          f"(n_obs={r_full['n_observations']}).")


def test_true_controlled_map_certificate_is_chaotic() -> None:
    import numpy as np
    import step78_certified_horizon_ci as s78
    import step74_lorenz96_spectrum as s74
    torch.manual_seed(0); N = 12
    traj = s74.attractor_traj(N, 400, 0, "cpu").double()
    mu, sd = traj.mean(0), traj.std(0) + 1e-8
    x0 = (traj[len(traj)//2] - mu) / sd
    g = lambda xn: (s79.rk4_controlled(xn * sd + mu, torch.zeros_like(xn)) - mu) / sd
    logR = s78.qr_logR_series(g, x0, n_steps=1200, warmup=200)
    point, lo, hi = s78.bootstrap_spectrum_ci(logR, s74.DTMAP, n_boot=400, block=40, seed=0)
    assert lo[0] > 0, f"true controlled map (u=0) must be chaotic; lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]"
    assert abs(float(point.sum()) - (-N)) / N < 0.05, "Liouville sum(lambda)=-N must hold for the controlled field at u=0"
    print(f"PASS: true controlled-map certificate is chaotic (lambda1 CI>0) and honors Liouville.")


if __name__ == "__main__":
    test_controlled_dynamics_is_ZN_equivariant()
    test_world_model_conv_is_equivariant_mlp_is_not()
    test_planner_is_orbit_equivariant()
    test_closed_loop_control_is_orbit_flat()
    test_stabilize_run_is_orbit_equivariant()
    test_certified_T1_steps_units_conversion()
    test_reobserve_run_interval_monotonicity()
    test_true_controlled_map_certificate_is_chaotic()
    print("step79 phase-0+1+3+4+5a+5b guard PASS.")
