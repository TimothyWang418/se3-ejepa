r"""SO(2) equivariance of the Step 61 *pose-control planner* (the task-level certificate's load-bearing claim).

Step 61's [B] result -- that closed-loop pose control is *exactly* orbit-invariant (model-rollout ratio
$1.000$) -- rests on the planner satisfying

$$ \mathrm{plan}\big(R_\beta s,\;R_\beta g\big) \;=\; R_\beta\cdot\mathrm{plan}(s,g), $$

i.e. :func:`step61.cem_plan_pose_equiv` commutes with the SO(2) action on the state+goal. This holds only because
every CEM operation was made equivariant: an **isotropic** per-step $\sigma$ computed as the RMS *norm* of the
deviation from the per-step vector mean (NOT a pooled per-axis ``std(dim=(0,2))``, which subtracts a scalar pooled
mean and is scrambled by rotation -- the silent break this test guards), a **disk** action bound
$\lVert a\rVert\le1$ (not the box $[-1,1]^2$), exploration noise pre-rotated by $R_\beta$, and an SO(2)-invariant
pose cost. Swap any one back and the identity breaks at a generic angle -- which is exactly the difference between
Step 61's $1.000$ model-rollout ratio and the noisy closed loops of Steps 10/12.

Determinism. CEM is stochastic, so we re-seed the *same* generator for the base run and the $R_\beta$ run; with the
noise pre-rotated by $R_\beta$ the candidate sets satisfy $\varepsilon_\beta=R_\beta\varepsilon_{\rm base}$ up to
float32, so identical (invariant) costs select identical elites and the elite means satisfy
$\bar a_\beta=R_\beta\bar a_{\rm base}$. The non-equivariant MLP world model is the control: with the SAME
equivariant planner its plan does NOT commute, so the test is not vacuously satisfied. The equivariance is
architectural, so it must hold both at init and after a real training run.

Run:
    .venv/bin/python tests/test_step61.py
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from step10_pusht_closed_loop import MLPForwardModelPushT, rot_np, rotate_packed_torch  # noqa: E402
from step59_pusht_certificate import RichVNForwardModelPushT  # noqa: E402
from step61_closed_loop_certificate import cem_plan_pose_equiv, model_rollout_pose_err  # noqa: E402

torch.set_default_dtype(torch.float32)

# small, tie-unlikely CEM config (keeps the unit test fast and deterministic)
_CEM = dict(H=8, n_samples=96, n_iters=3, n_elite=12, sigma0=0.7)
_TOL_EQ = 1e-3      # float32 + einsum-rotated noise + disk clamp over a few CEM iters
_TOL_MLP = 1e-2     # the non-equivariant control must miss by >> the equivariant floor


def _rand_packed(gen: torch.Generator) -> np.ndarray:
    r"""A random valid packed PushT state ``(8,)`` (block_dir a unit vector)."""
    p = torch.randn(8, generator=gen) * 0.3
    bd = p[6:8]
    p[6:8] = bd / bd.norm().clamp_min(1e-6)
    return p.numpy().astype(np.float32)


def _rand_goal(gen: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    r"""A random scaled goal position and a unit goal direction."""
    gpos = (torch.randn(2, generator=gen) * 0.1).numpy().astype(np.float32)
    a = float(torch.randn(1, generator=gen))
    gdir = np.array([math.cos(a), math.sin(a)], dtype=np.float32)
    return gpos, gdir


@torch.no_grad()
def _plan_equiv_residual(model, s0, gpos, gdir, beta, *, seed: int) -> float:
    r"""$\max\lvert R_\beta\cdot\mathrm{plan}(s,g) - \mathrm{plan}(R_\beta s,R_\beta g)\rvert$ (0 if equivariant)."""
    R = rot_np(beta)
    Rt = torch.tensor(R, dtype=torch.float32)
    gen_b = torch.Generator().manual_seed(seed)
    plan_base = cem_plan_pose_equiv(model, s0, gpos, gdir, noise_rot=0.0, gen=gen_b, **_CEM)
    # rotated run: SAME seed, state+goal+noise rotated by beta
    s0r = rotate_packed_torch(torch.tensor(s0).unsqueeze(0), beta)[0].numpy()
    gpr = (R @ gpos).astype(np.float32)
    gdr = (R @ gdir).astype(np.float32)
    gen_r = torch.Generator().manual_seed(seed)
    plan_rot = cem_plan_pose_equiv(model, s0r, gpr, gdr, noise_rot=beta, gen=gen_r, **_CEM)
    plan_base_rot = (R @ plan_base.T).T                                  # R_beta . plan(s,g)
    return float(np.abs(plan_base_rot - plan_rot).max())


def _worst_residual(model, *, seed0: int) -> float:
    r"""Worst plan-equivariance residual over a few random tasks and orbit angles."""
    gen = torch.Generator().manual_seed(seed0)
    worst = 0.0
    for k in range(4):
        s0 = _rand_packed(gen)
        gpos, gdir = _rand_goal(gen)
        for deg in (37.0, 90.0, 211.0):
            worst = max(worst, _plan_equiv_residual(model, s0, gpos, gdir, math.radians(deg), seed=2000 + k))
    return worst


def test_pose_planner_commutes_with_so2() -> None:
    r"""``plan(R.s, R.g) == R.plan(s, g)`` for the equivariant model; MLP control fails (non-vacuous)."""
    torch.manual_seed(0)

    eq = RichVNForwardModelPushT(hidden=32)
    init_res = _worst_residual(eq, seed0=7)
    assert init_res < _TOL_EQ, f"equivariant pose planner not equivariant at init: {init_res:.2e}"

    # property must survive a real optimisation step (it is architectural, but we check)
    opt = torch.optim.Adam(eq.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    S = torch.randn(128, 8, generator=g); S[:, 6:8] /= S[:, 6:8].norm(dim=-1, keepdim=True)
    A = torch.randn(128, 2, generator=g)
    S2 = S + 0.05 * torch.randn(128, 8, generator=g)
    for _ in range(40):
        opt.zero_grad(); ((eq.step(S, A) - S2) ** 2).mean().backward(); opt.step()
    post_res = _worst_residual(eq, seed0=11)
    assert post_res < _TOL_EQ, f"pose-planner equivariance broke AFTER training: {post_res:.2e}"

    # non-equivariant control: SAME planner, MLP world model, must FAIL
    mlp = MLPForwardModelPushT(hidden=128)
    mlp_res = _worst_residual(mlp, seed0=7)
    assert mlp_res > _TOL_MLP, f"MLP control unexpectedly commutes ({mlp_res:.2e}); test may be vacuous"

    print(f"PASS: equivariant pose-plan residual init={init_res:.2e} -> post-train={post_res:.2e} "
          f"(< {_TOL_EQ:.0e}); MLP control={mlp_res:.2e} (> {_TOL_MLP:.0e}).")
    print("=> the SO(2)-equivariant CEM commutes with R_beta: plan(R.s)=R.plan(s) -- the planner half of "
          "Step 61's exact (x1.000) task-level certificate, and the guard for the isotropic-sigma fix.")


def test_model_rollout_pose_error_is_orbit_invariant() -> None:
    r"""The model-rollout terminal pose error is orbit-invariant for the equivariant model (the EXACT
    certificate), and varies for the MLP (non-vacuous)."""
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(3)
    eq = RichVNForwardModelPushT(hidden=32)
    mlp = MLPForwardModelPushT(hidden=128)

    def orbit_spread(model) -> float:
        s0 = _rand_packed(gen)
        gpos, gdir = _rand_goal(gen)
        errs = []
        for deg in (0.0, 37.0, 90.0, 180.0, 270.0):
            beta = math.radians(deg)
            R = rot_np(beta)
            s0r = rotate_packed_torch(torch.tensor(s0).unsqueeze(0), beta)[0].numpy()
            gpr, gdr = (R @ gpos).astype(np.float32), (R @ gdir).astype(np.float32)
            plan = cem_plan_pose_equiv(model, s0r, gpr, gdr, noise_rot=beta,
                                       gen=torch.Generator().manual_seed(123), **_CEM)
            task = {"state_packed": s0r, "goal_px": gpr * 256.0 + np.array([255.5, 255.5]),
                    "goal_angle": math.atan2(gdr[1], gdr[0])}
            ang, _ = model_rollout_pose_err(model, task, plan)
            errs.append(ang)
        return max(errs) / max(min(errs), 1e-9) if min(errs) > 1e-6 else max(errs) - min(errs)

    eq_spread = orbit_spread(eq)
    mlp_spread = orbit_spread(mlp)
    assert eq_spread < 1.02, f"equivariant model-rollout pose error not orbit-invariant: ratio {eq_spread:.4f}"
    assert mlp_spread > 1.05, f"MLP model-rollout pose error unexpectedly flat ({mlp_spread:.4f}); test vacuous"
    print(f"PASS: model-rollout pose-error orbit ratio: equivariant x{eq_spread:.4f} (< 1.02, the EXACT "
          f"task-level certificate) vs MLP x{mlp_spread:.3f} (> 1.05).")


if __name__ == "__main__":
    test_pose_planner_commutes_with_so2()
    test_model_rollout_pose_error_is_orbit_invariant()
