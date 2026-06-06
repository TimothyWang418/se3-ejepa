r"""Guard for Step 73 / Stage 4a — the FetchPush planning stack is SO(2)-equivariant end-to-end, so the plan it
produces is orbit-equivariant: $\mathrm{plan}(\rho(g)\,s,\,\rho(g)\,\mathrm{goal}) = R(g)\,\mathrm{plan}(s,\mathrm{goal})$.

This is the architectural guarantee behind the task-level certificate (Experiment 17): if the encoder, predictor,
goal-readout head, AND the CEM search are all equivariant, then rotating the scene rotates the planned action
sequence identically — so the closed-loop behaviour (and hence task success) is orbit-INVARIANT, before any training.
We verify three links and then the composed planner:

  (i)   EquivGoalHead: g(rho(g) z) = R(g) g(z)                          [readout is equivariant]
  (ii)  the latent step + readout compose equivariantly                 [rollout is equivariant]
  (iii) cem_plan(rho(g) z0, R(g) goal; noise_rot=g) = R(g) cem_plan(z0, goal; noise_rot=0)   [SEARCH is equivariant]

(iii) is the load-bearing one: it needs the isotropic per-step covariance + disk action bound + scene-covariant
noise of the G-equivariant CEM. A per-axis covariance or box bound would silently break it (the bug step61 fixed).

Run:  .venv/bin/python tests/test_step73_planner_equivariance.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from step73_fetchpush_planning import EquivGoalHead, EquivPlanner, cem_plan  # noqa: E402
from step72_mujoco_certificate import EquivariantWM, latent_rotation  # noqa: E402


def _R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def test_goal_head_is_equivariant() -> None:
    torch.manual_seed(0)
    head = EquivGoalHead(latent_dim=64).double().eval()
    theta = 0.7
    z = torch.randn(16, 64, dtype=torch.float64)
    with torch.no_grad():
        g0 = head(z)
        g1 = head(latent_rotation(z, theta))
        g0_rot = g0 @ _R(theta).T
        err = (g1 - g0_rot).abs().max().item()
    assert err < 1e-9, f"EquivGoalHead not SO(2)-equivariant: max err {err:.2e}"
    print(f"PASS (i): goal-readout head g(rho z) = R g(z) — max err {err:.1e}.")


def test_rollout_readout_composes_equivariantly() -> None:
    torch.manual_seed(0)
    wm = EquivariantWM(latent_dim=64, hidden=32).double().eval()
    head = EquivGoalHead(latent_dim=64).double().eval()
    planner = EquivPlanner(wm, head)
    theta = 0.9
    z = torch.randn(8, 64, dtype=torch.float64)
    a = torch.randn(8, 2, dtype=torch.float64)
    R = _R(theta)
    with torch.no_grad():
        # one latent step + readout, on the rotated vs unrotated input
        g_plain = head(planner.step(z, a))
        g_rot = head(planner.step(latent_rotation(z, theta), a @ R.T))
        err = (g_rot - g_plain @ R.T).abs().max().item()
    assert err < 1e-9, f"step+readout not jointly equivariant: max err {err:.2e}"
    print(f"PASS (ii): readout(step(rho z, R a)) = R readout(step(z, a)) — max err {err:.1e}.")


def test_cem_search_is_equivariant() -> None:
    r"""The whole CEM search: same generator seed, rotated scene+goal+noise => rotated plan, to the float floor."""
    torch.manual_seed(0)
    wm = EquivariantWM(latent_dim=64, hidden=32).double().eval()
    head = EquivGoalHead(latent_dim=64).double().eval()
    planner = EquivPlanner(wm, head)
    theta = 1.1
    R = _R(theta)
    z0 = torch.randn(64, dtype=torch.float64)
    goal = torch.randn(2, dtype=torch.float64)
    kw = dict(H=6, n_samples=128, n_iters=3, n_elite=16)

    # unrotated plan
    plan0 = cem_plan(planner, z0, goal, noise_rot=0.0, gen=torch.Generator().manual_seed(123), **kw)
    # rotated scene (rho z0), rotated goal (R goal), scene-covariant noise (noise_rot=theta), SAME gen seed
    z0_rot = latent_rotation(z0.unsqueeze(0), theta)[0]
    goal_rot = goal @ R.T
    plan1 = cem_plan(planner, z0_rot, goal_rot, noise_rot=theta, gen=torch.Generator().manual_seed(123), **kw)

    plan0_rot = plan0 @ R.T                                    # R applied to each (dx,dy)
    err = (plan1 - plan0_rot).abs().max().item()
    assert err < 1e-9, f"CEM search not SO(2)-equivariant: max err {err:.2e} (check isotropic cov + disk bound)"
    print(f"PASS (iii): cem_plan(rho z0, R goal; noise_rot=theta) = R cem_plan(z0, goal) — max err {err:.1e}.")
    print("=> the whole planning stack is orbit-equivariant: rotating the scene rotates the plan identically, so "
          "closed-loop task behaviour is orbit-invariant by construction (Theorem A at the task level).")


if __name__ == "__main__":
    test_goal_head_is_equivariant()
    test_rollout_readout_composes_equivariantly()
    test_cem_search_is_equivariant()
