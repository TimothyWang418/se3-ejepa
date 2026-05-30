r"""SE(3) equivariance of the Step 18 *planner* (the missing half of the closed-loop theorem).

Step 18's [E] result -- that the closed-loop orientation error is *exactly* invariant under a
global $(R,t)\in\mathrm{SE}(3)$ -- rests on a planner claim that the four model-equivariance
tests do NOT cover:

$$ \mathrm{plan}\big(R x_0+t,\;R x_g+t\big) \;=\; R\cdot\mathrm{plan}(x_0,x_g), $$

i.e. the CEM-MPC :func:`step18.latent_cem_plan_iso` commutes with the group action on the
state+goal. This holds only because every CEM operation was made equivariant -- isotropic
$\sigma$, the unit-**ball** clamp (not a box), exploration noise pre-rotated by $R$, a latent
cost $\lVert\hat z_H-z_g\rVert^2$ (orthogonal $\rho$), and a translation-invariant +
closed-form centroid cost. Swap any one back (a box clamp, a *diagonal* $\sigma$) and the
identity breaks at generic $R$ -- exactly why Step 18's [S] panel drifts. This test is the
structural regression guard for that claim.

Determinism. CEM is stochastic, so we re-seed the *same* generator for the base run and the
$(R,t)$ run; with the noise pre-rotated by $R$ the two candidate sets satisfy
$\varepsilon_g=R\varepsilon_{\rm base}$ bit-for-bit (up to float32), so identical costs select
identical elites and the elite means satisfy $\bar a_g=R\bar a_{\rm base}$. We check both a pure
rotation ($t=0$) and a rotation with a large translation ($t$ far from 0) to confirm the
encoder's translation-invariance carries through the planner. A non-equivariant MLP world model
is the control: with the SAME equivariant planner its plan does NOT commute (residual $\gg$ floor),
so the test is not vacuously satisfied. We verify at init AND after a real training run, on CPU
(MPS's SVD/float path is less precise; the closed-loop readout uses Kabsch SVD).

Run:
    .venv/bin/python tests/test_planner_equivariance.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    rand_so3,
    rotate_points,
    teacher_step,
)
from step18_se3_closed_loop import (  # noqa: E402
    centroid,
    kabsch_rotation,
    latent_cem_plan_iso,
    rotation_angle_deg,
)
from src.training.jepa import train_jepa  # noqa: E402

torch.set_default_dtype(torch.float32)

# small, tie-unlikely CEM config (keeps the unit test fast and deterministic)
_CEM = dict(H=6, n_samples=64, n_iters=3, n_elite=8, sigma0=0.6, w_run=0.3)
_TOL_VN = 1e-3     # float32 + einsum-rotated noise + Kabsch-free plan compare
_TOL_MLP = 1e-2    # the non-equivariant control must miss by >> the VN floor


def _make_task(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""One reachable ``(X0, Xg)``: a wedge cloud and a short teacher rollout from it."""
    S, _, _ = collect_cloud_transitions(1, seed=seed)
    X0 = S[:1]
    X = X0.clone()
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(5):
        a = (torch.randn(1, 3, generator=g) * 0.6).clamp(-1, 1)
        X = teacher_step(X, a)
    return X0, X


@torch.no_grad()
def _plan_equiv_residual(
    model, X0: torch.Tensor, Xg: torch.Tensor, R: torch.Tensor, t: torch.Tensor, *,
    seed: int, w_t: float = 0.5,
) -> float:
    r"""$\max\lvert R\cdot\mathrm{plan}(x_0,x_g) - \mathrm{plan}(Rx_0+t,Rx_g+t)\rvert$ (0 if equivariant)."""
    tt = t.reshape(1, 1, 3)
    # base run
    zg = model.encoder(Xg)
    cg = centroid(Xg)
    gen_b = torch.Generator().manual_seed(seed)
    plan_base = latent_cem_plan_iso(model, X0, zg, cg=cg, R_noise=None, w_t=w_t, gen=gen_b, **_CEM)
    # (R,t) run: SAME seed, noise pre-rotated by R
    X0g, Xgg = rotate_points(X0, R) + tt, rotate_points(Xg, R) + tt
    zg_g = model.encoder(Xgg)
    cg_g = centroid(Xgg)
    gen_g = torch.Generator().manual_seed(seed)
    plan_g = latent_cem_plan_iso(model, X0g, zg_g, cg=cg_g, R_noise=R, w_t=w_t, gen=gen_g, **_CEM)
    return (rotate_points(plan_base, R) - plan_g).abs().max().item()


def _worst_residual(model, *, n_rot: int = 5, post: bool = False) -> float:
    r"""Worst plan-equivariance residual over ``n_rot`` random $(R,t)$, both $t=0$ and $t\neq0$."""
    gen = torch.Generator().manual_seed(99 if post else 7)
    worst = 0.0
    for k in range(n_rot):
        X0, Xg = _make_task(seed=1000 + k + (500 if post else 0))
        R = rand_so3(gen)
        for t in (torch.zeros(3), torch.tensor([10.0, -8.0, 6.0])):
            worst = max(worst, _plan_equiv_residual(model, X0, Xg, R, t, seed=2000 + k))
    return worst


def test_equivariant_planner_commutes_with_se3() -> None:
    r"""``plan(g.state, g.goal) == g.plan(state, goal)`` for the VN model; MLP control fails."""
    torch.manual_seed(0)

    # ---- VN at init -----------------------------------------------------------
    eq = build_eq_jepa()
    init_res = _worst_residual(eq)
    assert init_res < _TOL_VN, f"equivariant planner not equivariant at init: {init_res:.2e}"

    # ---- VN after a real training run (property must survive optimisation) -----
    S, A, S2 = collect_cloud_transitions(300, seed=0)
    train_jepa(eq, S, A, S2, epochs=12, batch_size=128, var_coef=0.1, seed=0, log_every=10**9)
    post_res = _worst_residual(eq, post=True)
    assert post_res < _TOL_VN, f"planner equivariance broke AFTER training: {post_res:.2e}"

    # ---- non-equivariant control: SAME planner, MLP world model, must FAIL ------
    mlp = build_mlp_jepa()
    train_jepa(mlp, S, A, S2, epochs=12, batch_size=128, var_coef=0.1, seed=0, log_every=10**9)
    mlp_res = _worst_residual(mlp, post=True)
    assert mlp_res > _TOL_MLP, (
        f"MLP control unexpectedly commutes ({mlp_res:.2e}); test may be vacuous"
    )

    print(f"PASS: VN plan residual init={init_res:.2e} -> post-train={post_res:.2e} (< {_TOL_VN:.0e}); "
          f"MLP control={mlp_res:.2e} (> {_TOL_MLP:.0e}).")
    print("=> the SE(3)-equivariant planner commutes with global (R,t): plan(g.x)=g.plan(x), "
          "the planner half of Step 18's [E] closed-loop theorem.")


def test_kabsch_geodesic_readout_is_se3_invariant() -> None:
    r"""The Kabsch orientation error and centroid distance are invariant under global $(R,t)$."""
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(5)
    worst_ang, worst_pos = 0.0, 0.0
    for _ in range(6):
        XT = torch.randn(24, 3, generator=gen) * torch.tensor([1.0, 0.6, 0.35])
        Xg = XT + torch.randn(24, 3, generator=gen) * 0.15        # a near, non-rigid perturbation
        ang0 = rotation_angle_deg(kabsch_rotation(XT, Xg))
        pos0 = float((XT.mean(0) - Xg.mean(0)).norm())
        R = rand_so3(gen)
        t = torch.tensor([7.0, -3.0, 5.0])
        XTg, Xgg = XT @ R.T + t, Xg @ R.T + t
        ang1 = rotation_angle_deg(kabsch_rotation(XTg, Xgg))
        pos1 = float((XTg.mean(0) - Xgg.mean(0)).norm())
        worst_ang = max(worst_ang, abs(ang1 - ang0))
        worst_pos = max(worst_pos, abs(pos1 - pos0))
    assert worst_ang < 1e-2, f"Kabsch orientation error not SE(3)-invariant: d={worst_ang:.2e} deg"
    assert worst_pos < 1e-3, f"centroid distance not SE(3)-invariant: d={worst_pos:.2e}"
    print(f"PASS: Kabsch readout SE(3)-invariant: max|d angle|={worst_ang:.2e} deg, "
          f"max|d pos|={worst_pos:.2e} (the conjugation-invariance the [E] theorem uses).")


if __name__ == "__main__":
    test_equivariant_planner_commutes_with_se3()
    test_kabsch_geodesic_readout_is_se3_invariant()
