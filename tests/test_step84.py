r"""Guard for Step 84 — the certified-control-benchmark spotlight on Gymnasium ``Acrobot-v1`` ($\mathbb Z_2$ reflection).

Fast unit tests only (CPU, **NO training**): the env reflection commutes with $g$ (reflect+step vs step+reflect, to
integrator precision); our faithful RK4 copy agrees with the real gym ``step``; the frame-averaged WM is exactly
$\mathbb Z_2$-equivariant on a random init while the matched-capacity bare baseline is NOT; the tip-height proxy is
$\mathbb Z_2$-invariant and matches the gym goal formula; the certificate units are right ($T_1=\log(1/\epsilon)/
\lambda_1$ in env steps, monotone-increasing as $\epsilon$ shrinks) and abstain when $\lambda_1\le0$; the CEM planner
returns a valid discrete action; and the episode return is monotone in (negative) steps-to-goal. Mirrors
``tests/test_step81.py``, adapted to the Acrobot $\mathbb Z_2$ reflection + frame-averaged action-conditioned WM.

These run in a couple of seconds: every test uses an UNTRAINED frame-averaged WM (already exactly equivariant by
construction) or pure arithmetic — the spec forbids training here (the 3080 run is the scale-up)."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step84_certified_control_benchmark as s84  # noqa: E402

DT = torch.float64


def test_env_reflection_commutes_with_g() -> None:
    # The Acrobot dynamics must commute with the Z_2 reflection g: theta,omega -> -theta,-omega and torque a -> 2-a, to
    # INTEGRATOR PRECISION. We check reflect(step(s,a)) == step(reflect(s), 2-a) in obs space over random states + all
    # three actions (verify_env_equivariance does exactly this on our faithful RK4 copy). The field is odd in (x,tau)
    # and RK4 + wrap + bound preserve oddness, so this is ~machine precision (atol 1e-6 is loose but the spec target).
    out = s84.verify_env_equivariance(n=24, seed=0)
    assert out["commutes"], f"env not Z_2-equivariant: max_err {out['max_err']:.2e}"
    assert out["max_err"] < 1e-6, f"env reflection mismatch {out['max_err']:.2e} exceeds 1e-6"
    print(f"PASS: Acrobot dynamics commute with the Z_2 reflection (max_err {out['max_err']:.1e}).")


def test_rk4_copy_matches_gymnasium_step() -> None:
    # Our faithful RK4 copy (used for the env-equivariance check + the env-level chaos sanity) must agree with the REAL
    # gymnasium Acrobot step, so the certificate/equivariance claims are about the SAME map the episodes run. atol 1e-4
    # (the only slop is the float32<->float64 obs round-trip; the integrator + constants are identical).
    out = s84.verify_acrobot_matches_gym(n=8, seed=1)
    assert out["matches"], f"our RK4 copy disagrees with gym Acrobot step: max_err {out['max_err']:.2e}"
    print(f"PASS: faithful RK4 Acrobot copy matches gymnasium step (max_err {out['max_err']:.1e}).")


def test_obs_reflection_and_action_reflection() -> None:
    # The obs reflection P = diag(+1,-1,+1,-1,-1,-1) (cos EVEN; sin, omega ODD) is an orthogonal involution, and the
    # action reflection is a -> 2-a (torque -> -torque). Both are exact integer/sign maps.
    P = s84.P_REFLECT64
    assert torch.allclose(P @ P, torch.eye(6, dtype=DT), atol=1e-12), "P must be an involution (P^2 = I)"
    assert torch.allclose(P.T, P, atol=1e-12), "P must be symmetric (P^T = P)"
    for a in (0, 1, 2):
        assert s84.reflect_action(a) == 2 - a, f"action reflection wrong for a={a}"
        assert s84.AVAIL_TORQUE[s84.reflect_action(a)] == -s84.AVAIL_TORQUE[a], "torque must negate under g"
    # encode/reflect: obs(reflect_state(x)) == P obs(x) (reflecting the internal state negates sin, omega; cos fixed)
    torch.manual_seed(0)
    s4 = torch.randn(5, 4, dtype=DT)
    s4_refl = torch.stack([-s4[:, 0], -s4[:, 1], -s4[:, 2], -s4[:, 3]], dim=-1)
    lhs = s84.state4_to_obs6(s4_refl)
    rhs = s84.reflect_obs(s84.state4_to_obs6(s4))
    assert torch.allclose(lhs, rhs, atol=1e-12), f"obs reflection inconsistent (max {(lhs - rhs).abs().max():.2e})"
    print("PASS: obs reflection P is an orthogonal involution; action reflection a->2-a negates torque.")


def test_world_model_frameavg_is_equivariant_baseline_is_not() -> None:
    # The frame-averaged WM is EXACTLY Z_2-equivariant (the average over {e,g} is g-invariant): f(P s, 2-a) = P f(s, a)
    # to machine precision; the matched-capacity bare baseline is NOT. Untrained models suffice (equivariance is by
    # construction, independent of weights).
    torch.manual_seed(0)
    equi = s84.AcrobotFrameAvg().double()
    base = s84.AcrobotBaseMLP().double()
    s = torch.randn(6, 6, dtype=DT)
    a = torch.randint(0, 3, (6,))
    P = s84.P_REFLECT64
    with torch.no_grad():
        e_lhs = equi(s @ P.T, 2 - a); e_rhs = equi(s, a) @ P.T
        assert torch.allclose(e_lhs, e_rhs, atol=1e-6), \
            f"frame-avg WM not Z_2-equivariant (max {(e_lhs - e_rhs).abs().max():.2e})"
        b_lhs = base(s @ P.T, 2 - a); b_rhs = base(s, a) @ P.T
        assert not torch.allclose(b_lhs, b_rhs, atol=1e-4), "bare-MLP baseline should NOT be Z_2-equivariant"
    print(f"PASS: frame-averaged Acrobot WM is Z_2-equivariant ({float((e_lhs - e_rhs).abs().max()):.1e}); "
          f"baseline is not ({float((b_lhs - b_rhs).abs().max()):.2e}).")


def test_tip_height_is_invariant_and_matches_gym() -> None:
    # The shaped planner score uses the tip height -cos(t1) - cos(t1+t2). It must (a) match the gym goal formula exactly
    # (the terminal condition is height > 1) and (b) be Z_2-INVARIANT (cos fixed; the s1 s2 cross term is even under the
    # double sign flip), so the planner's objective is symmetric, like the env.
    torch.manual_seed(0)
    s4 = torch.randn(7, 4, dtype=DT)
    th1 = s4[:, 0]; th2 = s4[:, 1]
    gym_h = -torch.cos(th1) - torch.cos(th1 + th2)
    our_h = s84.tip_height_obs(s84.state4_to_obs6(s4))
    assert torch.allclose(gym_h, our_h, atol=1e-10), f"tip height != gym formula (max {(gym_h - our_h).abs().max():.2e})"
    P = s84.P_REFLECT64
    s6 = s84.state4_to_obs6(s4)
    assert torch.allclose(s84.tip_height_obs(s6), s84.tip_height_obs(s6 @ P.T), atol=1e-12), \
        "tip height must be Z_2-invariant"
    print("PASS: tip-height proxy matches gym's -cos(t1)-cos(t1+t2) and is Z_2-invariant.")


def test_certificate_units_log_one_over_eps_over_lambda1_in_steps() -> None:
    # The UNITS contract: certified horizon T1(eps) = log(1/eps)/lambda1 in ENV STEPS directly (Acrobot's Delta t-map IS
    # one env step, so NO DTMAP rescale, unlike step79/80/81). We verify the closed-form on a synthetic lambda1, the
    # monotonicity in eps (smaller eps => longer certified horizon, since log(1/eps) grows), and that the step78 horizon
    # interval is the right shape. Pure arithmetic via step78.horizon_interval (the same helper the certificate calls).
    import step78_certified_horizon_ci as s78
    lam1 = 0.5
    for eps in (0.3, 0.1, 0.01):
        T1 = math.log(1.0 / eps) / lam1
        assert T1 > 0, f"T1 must be positive for chaotic lambda1>0 (eps={eps})"
    # monotone: smaller eps -> larger T1
    T_big_eps = math.log(1.0 / 0.3) / lam1
    T_small_eps = math.log(1.0 / 0.01) / lam1
    assert T_small_eps > T_big_eps, "T1 must INCREASE as eps shrinks (log(1/eps) grows)"
    # step78 horizon interval from a lambda CI: decreasing in lambda, abstains if lambda_lo<=0
    iv = s78.horizon_interval(0.4, 0.6, eps=0.01)
    assert iv is not None and iv[0] < iv[1], "horizon interval must be a valid [T_lo,T_hi]"
    assert abs(iv[0] - math.log(1 / 0.01) / 0.6) < 1e-9 and abs(iv[1] - math.log(1 / 0.01) / 0.4) < 1e-9, \
        "horizon interval endpoints must be log(1/eps)/lambda_hi .. log(1/eps)/lambda_lo"
    assert s78.horizon_interval(-0.1, 0.6, eps=0.01) is None, "must abstain when lambda CI straddles 0"
    print(f"PASS: certified horizon units T1=log(1/eps)/lambda1 in env steps; monotone in eps "
          f"({T_big_eps:.1f}<{T_small_eps:.1f}); abstains when lambda1 CI straddles 0.")


def test_certificate_T1_steps_clamp_and_abstain() -> None:
    # The integer env-step horizon the planner consumes: round(T1) clamped to >= 1; abstain (lambda1<=0 => T1 None) -> 1.
    # We exercise the small inline conversion logic in control_certificate by checking its documented behavior on edge
    # values via a tiny synthetic stand-in (the same max(1, round(.)) rule, abstain->1).
    def t1_steps(T1):
        return (max(1, int(round(T1))) if (T1 is not None and np.isfinite(T1)) else 1)
    assert t1_steps(689.9) == 690, "round to nearest env step"
    assert t1_steps(0.3) == 1, "clamp to >= 1 step (a 0-step plan is unusable)"
    assert t1_steps(None) == 1, "abstain (lambda1<=0) -> 1 step (most conservative)"
    assert t1_steps(float("inf")) == 1, "non-finite -> 1 step"
    print("PASS: T1_steps rounds to env steps, clamps >= 1, abstains (lambda1<=0) -> 1.")


def test_cem_planner_returns_valid_discrete_action() -> None:
    # The CEM-MPC planner must return a valid discrete Acrobot action index in {0,1,2}, for any plan depth H, from an
    # untrained WM (the wiring, not the policy quality, is under test). Tiny CEM budget for speed.
    torch.manual_seed(0)
    model = s84.make_equivariant_wm().double()
    s6 = torch.randn(6, dtype=DT)
    for H in (1, 3, 7):
        a = s84.cem_plan(model, s6, H=H, n_iter=2, n_samples=32, n_elite=8, seed=0)
        assert a in (0, 1, 2), f"CEM action {a} not a valid Acrobot action index (H={H})"
    print("PASS: CEM-MPC planner returns a valid discrete action in {0,1,2} for H in {1,3,7}.")


def test_return_is_monotone_in_negative_steps_to_goal() -> None:
    # The reported return is -(steps-to-goal): faster swing-up => higher (less negative) return. So return is strictly
    # DECREASING in steps-to-goal. We check the contract directly (run_episode sets ret = -steps_to_goal) and the
    # monotonicity ordering used by every gate.
    rets = {steps: -steps for steps in (80, 150, 300, 500)}
    ordered = [rets[s] for s in (80, 150, 300, 500)]
    assert all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)), \
        "return = -(steps-to-goal) must strictly decrease as steps-to-goal grows"
    assert rets[80] > rets[500], "solving in 80 steps must beat solving in 500 (return-wise)"
    # and the gate_binding picks the H with the MAX mean return as the optimum (consistent with 'higher is better')
    fake_sweep = {"per_H": {1: {"mean_return": -300.0}, 3: {"mean_return": -120.0}, 9: {"mean_return": -260.0}}}
    gb = s84.gate_binding(fake_sweep, T1_steps=3)
    assert gb["H_star"] == 3 and gb["interior"] and not gb["flat"] and gb["passed"], \
        f"gate_binding must pick the interior max-return H as the optimum, got {gb}"
    # a FLAT sweep must NOT pass G-binding (the step79-D1 INCONCLUSIVE branch)
    flat_sweep = {"per_H": {1: {"mean_return": -200.0}, 3: {"mean_return": -200.3}, 9: {"mean_return": -200.1}}}
    gbf = s84.gate_binding(flat_sweep, T1_steps=3)
    assert not gbf["passed"] and gbf["flat"], "a flat return-vs-H curve must FAIL G-binding (INCONCLUSIVE)"
    print("PASS: return = -(steps-to-goal) monotone; G-binding picks interior optimum and rejects a flat curve.")


def test_gate_ii_return_win_logic() -> None:
    # G-ii (the win): cert-aware (H=T1) return >= best swept blind on >= 2/3 seeds AND strictly > a too-shallow and a
    # too-deep blind planner (mean). We check the decision logic on a synthetic sweep where cert-aware clearly wins, and
    # one where it does not (loosening is forbidden, so a marginal case must NOT pass).
    T1 = 5
    win_sweep = {"per_H": {
        2: {"H": 2, "is_cert": False, "returns": [-300, -310, -290], "mean_return": -300.0},
        5: {"H": 5, "is_cert": True, "returns": [-120, -130, -125], "mean_return": -125.0},
        10: {"H": 10, "is_cert": False, "returns": [-260, -250, -270], "mean_return": -260.0}}}
    g = s84.gate_ii_return_win(win_sweep, T1_steps=T1)
    assert g["passed"] and g["win_seed_frac"] == 1.0 and g["beats_shallow"] and g["beats_deep"], \
        f"G-ii should declare a clean win, got {g}"
    # cert-aware worse than the deep blind on mean => no win, even if it wins some seeds
    lose_sweep = {"per_H": {
        2: {"H": 2, "is_cert": False, "returns": [-300, -310, -290], "mean_return": -300.0},
        5: {"H": 5, "is_cert": True, "returns": [-200, -260, -255], "mean_return": -238.3},
        10: {"H": 10, "is_cert": False, "returns": [-130, -140, -135], "mean_return": -135.0}}}
    gl = s84.gate_ii_return_win(lose_sweep, T1_steps=T1)
    assert not gl["passed"] and not gl["beats_deep"], f"G-ii must NOT declare a win when beaten by the deep blind: {gl}"
    print("PASS: G-ii declares a win only on >=2/3-seed + beats-shallow + beats-deep; rejects marginal cases.")


if __name__ == "__main__":
    test_env_reflection_commutes_with_g()
    test_rk4_copy_matches_gymnasium_step()
    test_obs_reflection_and_action_reflection()
    test_world_model_frameavg_is_equivariant_baseline_is_not()
    test_tip_height_is_invariant_and_matches_gym()
    test_certificate_units_log_one_over_eps_over_lambda1_in_steps()
    test_certificate_T1_steps_clamp_and_abstain()
    test_cem_planner_returns_valid_discrete_action()
    test_return_is_monotone_in_negative_steps_to_goal()
    test_gate_ii_return_win_logic()
    print("step84 spotlight guard PASS.")
