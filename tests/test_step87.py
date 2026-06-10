r"""Guard for Step 87 Stage B — the cert-gated MBRL online loop (direction ②).

Fast, training-free tests for the NEW Stage-B primitives in ``experiments/step87_cert_gated_mbrl.py``: actor-driven
data collection, per-sample pathwise returns (the differentiable objective the cert-gate truncates), warm-start WM
refit, deterministic true-env evaluation, and the single-arm online loop (with the certificate injected via ``H_fn``
so tests never run Benettin-QR). Stage A (Actor/Critic/actor_objective) is exercised by its own diagnostic run.
"""
import torch

from experiments import step79_certified_control as s79
from experiments import step87_cert_gated_mbrl as s87

DT = torch.float64


def _tiny(N=8, seed=0):
    torch.manual_seed(seed)
    wm = s79.make_equivariant_wm(N).double()
    mu = torch.zeros(N, dtype=DT)
    sd = torch.ones(N, dtype=DT)
    actor = s87.Actor(N, hidden=32)
    critic = s87.Critic(N, hidden=32)
    return wm, mu, sd, actor, critic, N


def test_collect_segments_contract():
    r"""Actor-driven collection matches step79.collect_data's tuple contract: normalized starts/targets, RAW bounded
    controls, and exact env-step accounting (n_traj * K true-dynamics steps)."""
    wm, mu, sd, actor, critic, N = _tiny()
    n, K, umax = 6, 3, 1.0
    starts, controls, targets, env_steps = s87.collect_segments(actor, mu, sd, N, n_traj=n, K=K, seed=0, u_max=umax)
    assert starts.shape == (n, N) and controls.shape == (n, K, N) and targets.shape == (n, K, N)
    assert float(controls.abs().max()) <= umax + 1e-9          # actions clamped to the control authority
    assert env_steps == n * K                                   # every collected transition costs one TRUE env step
    assert torch.isfinite(starts).all() and torch.isfinite(targets).all()


def test_pathwise_returns_differentiable_and_gated():
    r"""pathwise_returns gives per-sample (B,) returns DIFFERENTIABLE w.r.t. the actor (the pathwise mechanism), and the
    gate H_g controls the rollout depth (deeper gate => different value; H_g=0 => pure critic bootstrap)."""
    wm, mu, sd, actor, critic, N = _tiny()
    for p in wm.parameters():
        p.requires_grad_(False)
    z0 = torch.randn(5, N, dtype=DT)
    J3 = s87.pathwise_returns(wm, actor, critic, z0, mu, sd, H_g=3)
    assert J3.shape == (5,)
    J3.mean().backward()
    gnorm = sum(float((p.grad ** 2).sum()) for p in actor.parameters() if p.grad is not None)
    assert gnorm > 0.0                                          # gradient flows through the WM rollout to the actor
    with torch.no_grad():
        J0 = s87.pathwise_returns(wm, actor, critic, z0, mu, sd, H_g=0)
    assert J0.shape == (5,) and not torch.allclose(J0, J3.detach())   # H_g=0 = critic-only bootstrap, differs from H_g=3


def test_wm_refit_warm_start_changes_params_in_place():
    r"""wm_refit continues training the SAME wm instance (warm start) on actor-collected data; parameters move."""
    wm, mu, sd, actor, critic, N = _tiny()
    data = s87.collect_segments(actor, mu, sd, N, n_traj=8, K=3, seed=1, u_max=1.0)
    before = [p.detach().clone() for p in wm.parameters()]
    out = s87.wm_refit(wm, data[:3], mu, sd, epochs=2, K=3, lr=1e-3)
    assert out is wm                                            # in-place warm start, not a fresh model
    moved = any(not torch.allclose(b, p.detach()) for b, p in zip(before, wm.parameters()))
    assert moved


def test_eval_actor_deterministic_and_gradfree():
    r"""eval_actor rolls the TRUE dynamics under the MEAN action: same seed => identical return; no grads accumulate."""
    wm, mu, sd, actor, critic, N = _tiny()
    r1 = s87.eval_actor(actor, mu, sd, N, n_eval=2, T_eval=10, seed=7)
    r2 = s87.eval_actor(actor, mu, sd, N, n_eval=2, T_eval=10, seed=7)
    assert isinstance(r1, float) and r1 == r2
    assert all(p.grad is None for p in actor.parameters())


def test_run_arm_history_contract():
    r"""One tiny-config arm iteration: history rows carry strictly increasing env_steps, the eval return, and the gate
    actually used; the injected H_fn (certificate stub) is honored."""
    wm, mu, sd, actor, critic, N = _tiny()
    hist = s87.run_arm(N, mu, sd, seed=0, H_fn=lambda wm_, mu_, sd_, k: 3, n_iters=2,
                       n_traj=4, K=3, M_policy=2, batch_img=8, n_eval=1, T_eval=8, wm_epochs=1)
    assert len(hist) == 2
    assert hist[0]["env_steps"] < hist[1]["env_steps"]
    for row in hist:
        assert {"iter", "env_steps", "eval_return", "H_g"} <= set(row)
        assert row["H_g"] == 3                                  # the injected gate is what the loop used
