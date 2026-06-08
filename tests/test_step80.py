r"""Guard for Step 80 — pendulum-ring corroborator of the certified-control co-demonstration (class-lift).

Fast unit tests only (NO training): the three exactness statements of result 1 (controlled ring dynamics, conv WM,
gradient planner all $\mathbb{Z}_N$-equivariant; MLP baseline not), the closed-loop orbit-flatness (result C), the
units-fix conversion ``certified_T1_steps``, the re-observation monotonicity, and a light check that the TRUE ring's
$u{=}0$ certificate is chaotic ($\lambda_1$ CI $>0$) and honors the ring's Liouville anchor $\sum\lambda=-N\gamma$.
Mirrors ``tests/test_step79.py``, adapted to the phase-augmented periodic-encoding ring."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step80_pendulum_ring as s80  # noqa: E402
DT = torch.float64


def test_controlled_ring_dynamics_is_ZN_equivariant() -> None:
    # Result 1(a): the phase-augmented ring field + RK4 map are EXACTLY Z_N-equivariant under the cyclic site shift
    # (which acts on the theta/omega blocks and leaves the global drive phase fixed) jointly with a shift of the control.
    torch.manual_seed(0); N = 8
    y = s80.attractor_traj_ring(N, 1, 0)[0]                    # an on-attractor phase-augmented state
    u = 0.5 * torch.randn(N, dtype=DT)
    for s in (1, 3, 5):
        lhs = s80.ring_rhs(s80.roll_state(y, N, s), N, u=torch.roll(u, s))
        rhs = s80.roll_state(s80.ring_rhs(y, N, u=u), N, s)
        assert torch.allclose(lhs, rhs, atol=1e-12), f"shift {s}: ring field not Z_N-equivariant"
        mlhs = s80.rk4_ring(s80.roll_state(y, N, s), N, torch.roll(u, s))
        mrhs = s80.roll_state(s80.rk4_ring(y, N, u), N, s)
        assert torch.allclose(mlhs, mrhs, atol=1e-12), f"shift {s}: RK4 ring map not equivariant"
    print("PASS: controlled pendulum-ring field and RK4 map are Z_N-equivariant.")


def test_world_model_conv_is_equivariant_mlp_is_not() -> None:
    # Result 1(b): the circular-conv WM (over [cos th, sin th, omega, u, cos psi, sin psi], drive global) is EXACTLY
    # Z_N-equivariant; the dense-MLP baseline is NOT. The drive (cos psi, sin psi) is global => shift-invariant.
    torch.manual_seed(0); N = 8
    conv = s80.RingConv(N).double()
    mlp = s80.RingMLP(N).double()
    y = s80.attractor_traj_ring(N, 1, 0)[0]
    feat, drive = s80.encode_state(y, N)                       # feat: (3,N), drive: (2,)
    u = 0.5 * torch.randn(N, dtype=DT)
    with torch.no_grad():
        for s in (1, 4):
            c_lhs = conv(torch.roll(feat, s, dims=-1), torch.roll(u, s), drive)
            c_rhs = torch.roll(conv(feat, u, drive), s, dims=-1)
            assert torch.allclose(c_lhs, c_rhs, atol=1e-10), \
                f"conv shift {s}: not equivariant (max {(c_lhs - c_rhs).abs().max():.2e})"
        m_lhs = mlp(torch.roll(feat, 1, dims=-1), torch.roll(u, 1), drive)
        m_rhs = torch.roll(mlp(feat, u, drive), 1, dims=-1)
        assert not torch.allclose(m_lhs, m_rhs, atol=1e-6), "MLP baseline should NOT be Z_N-equivariant"
    print("PASS: controlled conv ring WM is Z_N-equivariant; MLP baseline is not.")


def test_wm_step_state_is_ZN_equivariant() -> None:
    # The full-state WM step (encode -> WM -> decode atan2 -> advance drive phase) must commute with the site shift,
    # so rollouts/certificates inherit exact orbit-equivariance. Untrained equivariant conv suffices.
    torch.manual_seed(0); N = 8
    model = s80.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    y = s80.attractor_traj_ring(N, 1, 0)[0]
    u = 0.3 * torch.randn(N, dtype=DT)
    with torch.no_grad():
        for s in (1, 4):
            lhs = s80.wm_step_state(model, s80.roll_state(y, N, s), N, mu, sd, torch.roll(u, s))
            rhs = s80.roll_state(s80.wm_step_state(model, y, N, mu, sd, u), N, s)
            assert torch.allclose(lhs, rhs, atol=1e-10), \
                f"wm_step_state shift {s}: not equivariant (max {(lhs - rhs).abs().max():.2e})"
    print("PASS: wm_step_state (full phase-augmented WM step) is Z_N-equivariant.")


def test_planner_is_orbit_equivariant() -> None:
    # Result 1(c): plan_control (gradient MPC from u=0, invariant stabilization cost) is EXACTLY orbit-equivariant.
    torch.manual_seed(0); N = 8
    model = s80.make_equivariant_wm(N).double()               # untrained equivariant WM is enough (by construction)
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    y0 = s80.attractor_traj_ring(N, 1, 0)[0]
    for s in (1, 4):
        u_a = s80.plan_control(model, y0, mu, sd, H=5, seed=0)
        u_b = s80.plan_control(model, s80.roll_state(y0, N, s), mu, sd, H=5, seed=0)
        assert torch.allclose(u_b, torch.roll(u_a, s, dims=-1), atol=1e-6), \
            f"shift {s}: planner not orbit-equivariant (max diff {(u_b - torch.roll(u_a, s, -1)).abs().max():.2e})"
    print("PASS: ring planner commutes with the Z_N shift (exactly orbit-equivariant).")


def test_closed_loop_control_is_orbit_flat() -> None:
    # Result C: the equivariant-planner closed-loop control on the TRUE ring is orbit-flat to machine precision.
    torch.manual_seed(0); N = 8
    model = s80.make_equivariant_wm(N).double()               # untrained equivariant WM: planner still exact
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    y0 = s80.attractor_traj_ring(N, 1, 0)[0]
    of = s80.orbit_flatness(model, y0, mu, sd, H=4, n_steps=6, s=3, u_max=0.5, n_iter=10)
    assert of["control_mismatch"] < 1e-8, f"closed-loop control not orbit-flat: mismatch {of['control_mismatch']:.2e}"
    assert abs(of["ratio"] - 1.0) < 1e-6, f"closed-loop cost ratio {of['ratio']:.6f} not ~1.000"
    print(f"PASS: ring closed-loop control orbit-flat (mismatch {of['control_mismatch']:.1e}, ratio {of['ratio']:.6f}).")


def test_certified_T1_steps_units_conversion() -> None:
    # The UNITS FIX: certificate()['T1'] is in TIME units; the certified horizon in MAP STEPS is T1/DTMAP, NOT round(T1).
    # With the ring DTMAP=0.02, a T1 of ~3.07 time units must convert to round(3.07/0.02)=154 steps, NOT 3.
    assert abs(s80.DTMAP - 0.02) < 1e-12, f"this test assumes the ring DTMAP=0.02, got {s80.DTMAP}"
    steps = s80.certified_T1_steps({"T1": 3.0701})
    assert steps == round(3.0701 / s80.DTMAP) == 154, f"expected 154 map steps, got {steps}"   # NOT round(3.07)=3
    assert steps != 3, "units bug: certified_T1_steps must NOT collapse a ~3-time-unit horizon to ~3 steps"
    assert s80.certified_T1_steps({"T1": None}) == 1, "abstain (T1=None) must fall back to 1 step"
    assert s80.certified_T1_steps({"T1": 0.001}) == 1, "horizon must be clamped to >= 1 step"
    print(f"PASS: certified_T1_steps converts TIME->MAP STEPS (T1=3.07 -> {steps} steps, not 3).")


def test_reobserve_run_interval_monotonicity() -> None:
    # D2 mechanism: reobserve_run with interval=1 re-observes EVERY step (never blind) -> zero violations and
    # n_observations = T_total+1; with interval=T_total it observes only at the ends -> the MOST violations. The WM
    # here is the untrained equivariant conv; its u=0 map differs from the true ring, so a long blind forecast WILL
    # accumulate error -> monotonicity is structural (more re-observation => no more violations than less).
    torch.manual_seed(0); N = 8
    model = s80.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    T_total = 60
    y_seq = s80.attractor_traj_ring(N, T_total, 0)[:T_total + 1]   # (T_total+1, 2N+1) true ring trajectory
    r_obs1 = s80.reobserve_run(model, mu, sd, N, y_seq, interval=1, eps=0.01)
    r_full = s80.reobserve_run(model, mu, sd, N, y_seq, interval=T_total, eps=0.01)
    assert r_obs1["violation_rate"] == 0.0, f"interval=1 should have zero violations, got {r_obs1['violation_rate']}"
    assert r_obs1["n_observations"] == T_total + 1, \
        f"interval=1 should observe every step (n_obs={T_total + 1}), got {r_obs1['n_observations']}"
    assert r_full["n_observations"] < r_obs1["n_observations"], "interval=T_total must observe fewer times"
    assert r_full["violation_rate"] >= r_obs1["violation_rate"], \
        f"longer interval must not have FEWER violations ({r_full['violation_rate']} vs {r_obs1['violation_rate']})"
    print(f"PASS: reobserve_run monotone — interval=1 viol={r_obs1['violation_rate']:.2f} "
          f"(n_obs={r_obs1['n_observations']})  <=  interval={T_total} viol={r_full['violation_rate']:.2f} "
          f"(n_obs={r_full['n_observations']}).")


def test_true_ring_certificate_is_chaotic_and_honors_liouville() -> None:
    # The TRUE ring (u=0) phase-augmented map must be chaotic (lambda1 CI > 0) and honor the ring's Liouville anchor:
    # div f = sum_i d(dom_i)/d(omega_i) = -N*gamma (theta/psi rows add 0), so sum(lambda) = -N*gamma exactly. Light QR.
    import step78_certified_horizon_ci as s78
    torch.manual_seed(0); N = 8
    y0 = s80.attractor_traj_ring(N, 1, 0)[0].detach()
    g = lambda y: s80.rk4_ring(y, N, None)                    # TRUE autonomous (u=0) ring Delta t-map
    logR = s78.qr_logR_series(g, y0, n_steps=900, warmup=200)
    point, lo, hi = s78.bootstrap_spectrum_ci(logR, s80.DTMAP, n_boot=300, block=40, seed=0)
    assert lo[0] > 0, f"true ring map (u=0) must be chaotic; lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]"
    target = -N * s80.GAMMA
    assert abs(float(point.sum()) - target) / abs(target) < 0.05, \
        f"Liouville sum(lambda)={float(point.sum()):.3f} must match -N*gamma={target:.3f}"
    print(f"PASS: true ring certificate is chaotic (lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]>0) and honors Liouville "
          f"(sum {float(point.sum()):.3f} ~ -N*gamma {target:.3f}).")


if __name__ == "__main__":
    test_controlled_ring_dynamics_is_ZN_equivariant()
    test_world_model_conv_is_equivariant_mlp_is_not()
    test_wm_step_state_is_ZN_equivariant()
    test_planner_is_orbit_equivariant()
    test_closed_loop_control_is_orbit_flat()
    test_certified_T1_steps_units_conversion()
    test_reobserve_run_interval_monotonicity()
    test_true_ring_certificate_is_chaotic_and_honors_liouville()
    print("step80 corroborator guard PASS.")
