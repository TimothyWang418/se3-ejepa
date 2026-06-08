r"""Guard for Step 81 — double-pendulum ($\mathbb{Z}_2$ reflection) corroborator of the certified-horizon decision.

Fast unit tests only (NO training): the three exactness statements of result 1 (controlled DP dynamics, frame-averaged
WM, gradient planner all $\mathbb{Z}_2$-equivariant under the reflection $g\cdot x=-x$, $g\cdot u=-u$; the bare-MLP
baseline not), the closed-loop reflection-orbit-flatness on the 2-element orbit (result C), the units-fix conversion
``certified_T1_steps``, the re-observation monotonicity, the energy-conservation sanity at $u{=}0$, and a light check
that the TRUE DP's $u{=}0$ certificate is chaotic ($\lambda_1$ CI $>0$) and honors the conservative Liouville anchor
$\sum\lambda=-2\gamma\approx 0$. Mirrors ``tests/test_step80.py``, adapted to the $\mathbb{Z}_2$ reflection +
frame-averaged WM + conservative double pendulum."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step81_double_pendulum as s81  # noqa: E402
DT = torch.float64


def test_controlled_dp_dynamics_is_Z2_equivariant() -> None:
    # Result 1(a): the DP field + RK4 map are EXACTLY Z_2-equivariant under the reflection g.x = -x jointly with the
    # control reflection g.u = -u, i.e. f(-x,-u) = -f(x,u) and rk4(-x,-u) = -rk4(x,u). The field is odd in (x,u); RK4
    # preserves the odd symmetry. Checked with and without light damping (which is also odd in omega).
    torch.manual_seed(0)
    x = s81.attractor_traj_dp(0, 1, 0)[0]                      # an on-shell state
    u = 0.5 * torch.randn(2, dtype=DT)
    for gamma in (0.0, 0.05):
        lhs = s81.dp_rhs(s81.reflect_state(x), gamma=gamma, u=s81.reflect_control(u))
        rhs = s81.reflect_state(s81.dp_rhs(x, gamma=gamma, u=u))
        assert torch.allclose(lhs, rhs, atol=1e-10), f"gamma={gamma}: DP field not Z_2-equivariant"
        mlhs = s81.rk4_dp(s81.reflect_state(x), s81.reflect_control(u), gamma=gamma)
        mrhs = s81.reflect_state(s81.rk4_dp(x, u, gamma=gamma))
        assert torch.allclose(mlhs, mrhs, atol=1e-10), f"gamma={gamma}: RK4 DP map not Z_2-equivariant"
    print("PASS: controlled double-pendulum field and RK4 map are Z_2-equivariant (g.x=-x, g.u=-u).")


def test_encode_state_is_Z2_equivariant() -> None:
    # The periodic encoding intertwines the state reflection g.x=-x with the fixed feature sign flip P (cos EVEN; sin,
    # omega ODD): encode(-x) = P @ encode(x). This is what frame-averaging averages over.
    torch.manual_seed(0)
    x = torch.randn(5, 4, dtype=DT)
    lhs = s81.encode_state(s81.reflect_state(x))
    rhs = s81.encode_state(x) @ s81.P_REFLECT.T
    assert torch.allclose(lhs, rhs, atol=1e-12), f"encode not Z_2-equivariant (max {(lhs - rhs).abs().max():.2e})"
    # P is an orthogonal involution (P^2 = I, P^T = P).
    assert torch.allclose(s81.P_REFLECT @ s81.P_REFLECT, torch.eye(6, dtype=DT), atol=1e-12)
    print("PASS: periodic encoding is Z_2-equivariant (encode(-x) = P encode(x)); P is an involution.")


def test_world_model_frameavg_is_equivariant_baseline_is_not() -> None:
    # Result 1(b): the frame-averaged WM is EXACTLY Z_2-equivariant (the average over {e,g} is g-invariant), so
    # f(P feat, -u) = P f(feat, u) to machine precision; the bare-MLP baseline is NOT.
    torch.manual_seed(0)
    equi = s81.DPFrameAvg().double()
    base = s81.DPBaseMLP().double()
    x = s81.attractor_traj_dp(0, 1, 0)[0]
    feat = s81.encode_state(x)
    u = 0.5 * torch.randn(2, dtype=DT)
    P = s81.P_REFLECT
    with torch.no_grad():
        e_lhs = equi(feat @ P.T, -u); e_rhs = equi(feat, u) @ P.T
        assert torch.allclose(e_lhs, e_rhs, atol=1e-12), \
            f"frame-avg WM not Z_2-equivariant (max {(e_lhs - e_rhs).abs().max():.2e})"
        b_lhs = base(feat @ P.T, -u); b_rhs = base(feat, u) @ P.T
        assert not torch.allclose(b_lhs, b_rhs, atol=1e-6), "bare-MLP baseline should NOT be Z_2-equivariant"
    print("PASS: frame-averaged DP WM is Z_2-equivariant (~1e-12); bare-MLP baseline is not.")


def test_wm_step_state_is_Z2_equivariant() -> None:
    # The full-state WM step (encode -> WM -> decode atan2) must commute with the reflection, so rollouts/certificates
    # inherit exact orbit-equivariance: wm_step_state(-x, -u) = -wm_step_state(x, u). Untrained frame-avg WM suffices.
    torch.manual_seed(0)
    model = s81.make_equivariant_wm().double()
    mu = torch.zeros(2, dtype=DT); sd = torch.ones(2, dtype=DT)
    x = s81.attractor_traj_dp(0, 1, 0)[0]
    u = 0.3 * torch.randn(2, dtype=DT)
    with torch.no_grad():
        lhs = s81.wm_step_state(model, s81.reflect_state(x), mu, sd, s81.reflect_control(u))
        rhs = s81.reflect_state(s81.wm_step_state(model, x, mu, sd, u))
        assert torch.allclose(lhs, rhs, atol=1e-10), \
            f"wm_step_state not Z_2-equivariant (max {(lhs - rhs).abs().max():.2e})"
    print("PASS: wm_step_state (full-state WM step) is Z_2-equivariant.")


def test_planner_is_orbit_equivariant() -> None:
    # Result 1(c): plan_control (gradient MPC from u=0, Z_2-invariant stabilization cost toward the g-fixed rest state
    # theta=0,omega=0) is EXACTLY reflection-orbit-equivariant: plan(-x0) = -plan(x0).
    torch.manual_seed(0)
    model = s81.make_equivariant_wm().double()                # untrained frame-avg WM is enough (equivariant by const.)
    mu = torch.zeros(2, dtype=DT); sd = torch.ones(2, dtype=DT)
    x0 = s81.attractor_traj_dp(0, 1, 0)[0]
    u_a = s81.plan_control(model, x0, mu, sd, H=5, seed=0)
    u_b = s81.plan_control(model, s81.reflect_state(x0), mu, sd, H=5, seed=0)
    assert torch.allclose(u_b, s81.reflect_control(u_a), atol=1e-6), \
        f"planner not orbit-equivariant (max diff {(u_b - s81.reflect_control(u_a)).abs().max():.2e})"
    print("PASS: DP planner commutes with the Z_2 reflection (plan(-x0) = -plan(x0)).")


def test_closed_loop_control_is_orbit_flat() -> None:
    # Result C: the equivariant-planner closed-loop control on the TRUE DP is reflection-orbit-flat to machine precision
    # on the 2-element orbit {x0, -x0} (applied controls negate exactly; the even cost is invariant).
    torch.manual_seed(0)
    model = s81.make_equivariant_wm().double()                # untrained frame-avg WM: planner still exact
    mu = torch.zeros(2, dtype=DT); sd = torch.ones(2, dtype=DT)
    x0 = s81.attractor_traj_dp(0, 1, 0)[0]
    of = s81.orbit_flatness(model, x0, mu, sd, H=4, n_steps=6, u_max=0.5, n_iter=10)
    assert of["control_mismatch"] < 1e-8, f"closed-loop control not orbit-flat: mismatch {of['control_mismatch']:.2e}"
    assert abs(of["ratio"] - 1.0) < 1e-6, f"closed-loop cost ratio {of['ratio']:.6f} not ~1.000"
    print(f"PASS: DP closed-loop control reflection-orbit-flat (mismatch {of['control_mismatch']:.1e}, "
          f"ratio {of['ratio']:.6f}).")


def test_certified_T1_steps_units_conversion() -> None:
    # The UNITS FIX: certificate()['T1'] is in TIME units; the certified horizon in MAP STEPS is T1/DTMAP, NOT round(T1).
    # With the DP DTMAP=0.02, a T1 of ~3.07 time units must convert to round(3.07/0.02)=154 steps, NOT 3.
    assert abs(s81.DTMAP - 0.02) < 1e-12, f"this test assumes the DP DTMAP=0.02, got {s81.DTMAP}"
    steps = s81.certified_T1_steps({"T1": 3.0701})
    assert steps == round(3.0701 / s81.DTMAP) == 154, f"expected 154 map steps, got {steps}"   # NOT round(3.07)=3
    assert steps != 3, "units bug: certified_T1_steps must NOT collapse a ~3-time-unit horizon to ~3 steps"
    assert s81.certified_T1_steps({"T1": None}) == 1, "abstain (T1=None) must fall back to 1 step"
    assert s81.certified_T1_steps({"T1": 0.001}) == 1, "horizon must be clamped to >= 1 step"
    print(f"PASS: certified_T1_steps converts TIME->MAP STEPS (T1=3.07 -> {steps} steps, not 3).")


def test_reobserve_run_interval_monotonicity() -> None:
    # D2 mechanism: reobserve_run with interval=1 re-observes EVERY step (never blind) -> zero violations and
    # n_observations = T_total+1; with interval=T_total it observes only at the ends -> the MOST violations. The WM
    # here is the untrained frame-avg WM; its u=0 map differs from the true DP, so a long blind forecast accumulates
    # error -> monotonicity is structural (more re-observation => no more violations than less).
    torch.manual_seed(0)
    model = s81.make_equivariant_wm().double()
    mu = torch.zeros(2, dtype=DT); sd = torch.ones(2, dtype=DT)
    T_total = 60
    x_seq = s81.attractor_traj_dp(0, T_total, 0)[:T_total + 1]   # (T_total+1, 4) true DP trajectory
    r_obs1 = s81.reobserve_run(model, mu, sd, 0, x_seq, interval=1, eps=0.01)
    r_full = s81.reobserve_run(model, mu, sd, 0, x_seq, interval=T_total, eps=0.01)
    assert r_obs1["violation_rate"] == 0.0, f"interval=1 should have zero violations, got {r_obs1['violation_rate']}"
    assert r_obs1["n_observations"] == T_total + 1, \
        f"interval=1 should observe every step (n_obs={T_total + 1}), got {r_obs1['n_observations']}"
    assert r_full["n_observations"] < r_obs1["n_observations"], "interval=T_total must observe fewer times"
    assert r_full["violation_rate"] >= r_obs1["violation_rate"], \
        f"longer interval must not have FEWER violations ({r_full['violation_rate']} vs {r_obs1['violation_rate']})"
    print(f"PASS: reobserve_run monotone — interval=1 viol={r_obs1['violation_rate']:.2f} "
          f"(n_obs={r_obs1['n_observations']})  <=  interval={T_total} viol={r_full['violation_rate']:.2f} "
          f"(n_obs={r_full['n_observations']}).")


def test_energy_conservation_at_u0() -> None:
    # EOM correctness: the conservative DP (u=0, gamma=0) conserves energy under RK4 to ~1e-5 over 5 time units (a
    # standard sanity check that the double-pendulum equations of motion are coded correctly). The chaotic shell at
    # E~1.25-2.2 makes this a nontrivial check (a wrong sign/term would leak energy fast).
    torch.manual_seed(0)
    x = s81.attractor_traj_dp(0, 1, 0)[0]
    E0 = float(s81.dp_energy(x))
    n = int(round(5.0 / s81.DTMAP))                            # 5 time units of map steps
    for _ in range(n):
        x = s81.rk4_dp(x, None)                                # u=0, gamma=0
    drift = abs(float(s81.dp_energy(x)) - E0)
    assert drift < 1e-3, f"energy drift {drift:.2e} over 5t too large — check the DP equations of motion"
    print(f"PASS: DP conserves energy at u=0 (drift {drift:.2e} over 5t, E0={E0:.3f}) — EOM correct.")


def test_true_dp_certificate_is_chaotic_and_honors_liouville() -> None:
    # The TRUE DP (u=0) map must be chaotic (lambda1 CI > 0) and honor the conservative Liouville anchor: div f =
    # sum_j d(dom_j)/d(omega_j) = -2*gamma (theta rows add 0 on the diagonal), so sum(lambda) = -2*gamma = 0 at
    # gamma=0. Light QR. (A 4-D system with one positive + one near-zero + symplectic-paired negatives.)
    import step78_certified_horizon_ci as s78
    torch.manual_seed(0)
    x0 = s81.attractor_traj_dp(0, 1, 0)[0].detach()
    g = lambda x: s81.rk4_dp(x, None)                          # TRUE autonomous (u=0) DP Delta t-map
    logR = s78.qr_logR_series(g, x0, n_steps=900, warmup=300)
    point, lo, hi = s78.bootstrap_spectrum_ci(logR, s81.DTMAP, n_boot=300, block=40, seed=0)
    assert lo[0] > 0, f"true DP map (u=0) must be chaotic; lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]"
    target = -2.0 * s81.GAMMA                                  # = 0 at gamma=0 (conservative)
    assert abs(float(point.sum()) - target) < 0.05, \
        f"Liouville sum(lambda)={float(point.sum()):.3f} must match -2*gamma={target:.3f} (conservative anchor)"
    print(f"PASS: true DP certificate is chaotic (lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]>0) and honors Liouville "
          f"(sum {float(point.sum()):.3f} ~ -2*gamma {target:.3f}).")


if __name__ == "__main__":
    test_controlled_dp_dynamics_is_Z2_equivariant()
    test_encode_state_is_Z2_equivariant()
    test_world_model_frameavg_is_equivariant_baseline_is_not()
    test_wm_step_state_is_Z2_equivariant()
    test_planner_is_orbit_equivariant()
    test_closed_loop_control_is_orbit_flat()
    test_certified_T1_steps_units_conversion()
    test_reobserve_run_interval_monotonicity()
    test_energy_conservation_at_u0()
    test_true_dp_certificate_is_chaotic_and_honors_liouville()
    print("step81 corroborator guard PASS.")
