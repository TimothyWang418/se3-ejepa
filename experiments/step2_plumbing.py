r"""Step 2 plumbing test: prove EqJEPA plugs into SWM's planning stack.

This does NOT train anything. It verifies the *integration surface* end to end:

  1. ``EqJEPA.get_cost`` honours the solver contract: returns a ``(B, S)`` tensor.
  2. A ``CEMSolver`` wrapping EqJEPA produces an action plan from one observation.
  3. ``World.evaluate(episodes=...)`` drives the full
     World -> WorldModelPolicy -> CEMSolver -> EqJEPA.get_cost -> action -> step
     loop in *episodic* mode (no dataset download needed).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step2_plumbing.py
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

# make `src` importable when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.solver.cem import CEMSolver  # noqa: E402

from src.models.eqjepa import EqJEPA  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"  # tiny model; CPU avoids MPS edge cases for a plumbing test

    # image transform: uint8 HWC -> float32 CHW in [0,1]
    # (_prepare_info already permutes to CHW; ToImage/ToDtype just set dtype/scale)
    img_tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    banner("[A] Build World (PushT, episodic mode, 64x64 render)")
    world = swm.World("swm/PushT-v1", num_envs=2, image_shape=(64, 64))
    world.reset(seed=0)
    action_dim = world.envs.single_action_space.shape[-1]
    print(f"    num_envs={world.num_envs}  action_dim={action_dim}")
    print(f"    action_space={world.envs.single_action_space}")

    banner("[B] Build EqJEPA model")
    model = EqJEPA(in_channels=3, latent_dim=128, action_dim=action_dim).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    EqJEPA params: {n_params:,}  (encoder + predictor)")

    # ---- Layer 1: direct get_cost contract --------------------------------
    banner("[C] Layer 1 — direct get_cost contract")
    B, S, H = 2, 7, 5
    fake_info = {
        # solver-expanded layout: (B, S, T, C, H, W)
        "pixels": torch.rand(B, S, 1, 3, 64, 64),
        "goal": torch.rand(B, S, 1, 3, 64, 64),
    }
    candidates = torch.randn(B, S, H, action_dim)
    cost = model.get_cost(fake_info, candidates)
    assert cost.shape == (B, S), f"expected (B,S)=({B},{S}), got {tuple(cost.shape)}"
    assert torch.isfinite(cost).all(), "cost has non-finite entries"
    print(f"    get_cost -> shape {tuple(cost.shape)}  finite={bool(torch.isfinite(cost).all())}")
    print(f"    cost[0]: {cost[0].tolist()}")
    print("    PASS: returns finite (B, S) cost tensor")

    # ---- Layer 2: one CEM solve -------------------------------------------
    banner("[D] Layer 2 — one CEM solve from a real observation")
    config = swm.PlanConfig(
        horizon=5, receding_horizon=5, action_block=1, warm_start=False
    )
    solver = CEMSolver(
        model=model, num_samples=30, n_steps=3, topk=10, device=device, seed=0
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process={},  # no normalizers needed for the plumbing test
        transform={"pixels": img_tf, "goal": img_tf},
    )
    world.set_policy(policy)  # configures the solver against the action space

    # build a prepared info_dict from the current world state and solve once
    prepared = policy._prepare_info(world.infos)
    out = solver.solve(prepared, init_action=None)
    actions = out["actions"]
    print(f"    solver returned actions shape={tuple(actions.shape)}  costs={out['costs']}")
    assert actions.shape[0] == world.num_envs
    assert actions.shape[-1] == action_dim
    print("    PASS: CEM produced an action plan")

    # ---- Layer 3: full episodic evaluate ----------------------------------
    banner("[E] Layer 3 — full World.evaluate (episodic, no dataset)")
    world.reset(seed=0)
    # keep episodes short so it finishes fast on CPU
    metrics = world.evaluate(episodes=2, seed=0)
    print(f"    metrics: {metrics}")
    assert "success_rate" in metrics, "evaluate did not return success_rate"
    print("    PASS: full World -> policy -> solver -> get_cost -> step loop ran")

    world.close()
    banner("STEP 2 PLUMBING: ALL LAYERS PASSED")
    print("EqJEPA satisfies the Costable protocol and drives the SWM planning stack.")
    print("Note: model is UNTRAINED, so success_rate ~ random is expected here.")


if __name__ == "__main__":
    main()
