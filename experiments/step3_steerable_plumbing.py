r"""Step 3 plumbing test: prove the *steerable* EqJEPA still drives SWM's stack.

Identical wiring to ``step2_plumbing.py`` but swaps the ordinary ``ConvEncoder``
for the $C_N$-steerable :class:`SteerableEncoder`. The point is a controlled
single-variable change: everything downstream (predictor, ``get_cost`` contract,
CEM solver, ``World.evaluate``) is unchanged, so if this passes we know the
equivariant encoder is a drop-in for the planning stack.

Equivariance itself is proven separately in ``tests/test_eqjepa_equivariance.py``;
this script only checks the *integration surface* still holds with e2cnn layers.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step3_steerable_plumbing.py
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.solver.cem import CEMSolver  # noqa: E402

from src.models.eqjepa import EqJEPA, SteerableEncoder  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"  # e2cnn convs are CPU-friendly; tiny model

    img_tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    banner("[A] Build World (PushT, episodic mode, 64x64 render)")
    world = swm.World("swm/PushT-v1", num_envs=2, image_shape=(64, 64))
    world.reset(seed=0)
    action_dim = world.envs.single_action_space.shape[-1]
    print(f"    num_envs={world.num_envs}  action_dim={action_dim}")

    banner("[B] Build STEERABLE EqJEPA model (C_4-equivariant encoder)")
    latent_dim = 128
    encoder = SteerableEncoder(in_channels=3, latent_dim=latent_dim, n_rot=4)
    model = EqJEPA(latent_dim=latent_dim, action_dim=action_dim, encoder=encoder).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"    EqJEPA params: {n_params:,}  (steerable encoder: {n_enc:,})")

    # ---- Layer 1: direct get_cost contract --------------------------------
    banner("[C] Layer 1 - direct get_cost contract")
    B, S, H = 2, 7, 5
    fake_info = {
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
    banner("[D] Layer 2 - one CEM solve from a real observation")
    config = swm.PlanConfig(
        horizon=5, receding_horizon=5, action_block=1, warm_start=False
    )
    solver = CEMSolver(
        model=model, num_samples=30, n_steps=3, topk=10, device=device, seed=0
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process={},
        transform={"pixels": img_tf, "goal": img_tf},
    )
    world.set_policy(policy)

    prepared = policy._prepare_info(world.infos)
    out = solver.solve(prepared, init_action=None)
    actions = out["actions"]
    print(f"    solver returned actions shape={tuple(actions.shape)}  costs={out['costs']}")
    assert actions.shape[0] == world.num_envs
    assert actions.shape[-1] == action_dim
    print("    PASS: CEM produced an action plan")

    # ---- Layer 3: full episodic evaluate ----------------------------------
    banner("[E] Layer 3 - full World.evaluate (episodic, no dataset)")
    world.reset(seed=0)
    metrics = world.evaluate(episodes=2, seed=0)
    print(f"    metrics: {metrics}")
    assert "success_rate" in metrics, "evaluate did not return success_rate"
    print("    PASS: full World -> policy -> solver -> get_cost -> step loop ran")

    world.close()
    banner("STEP 3 PLUMBING: ALL LAYERS PASSED (steerable encoder)")
    print("Steerable EqJEPA is a drop-in for the SWM planning stack.")
    print("Note: model is UNTRAINED, so success_rate ~ random is expected here.")


if __name__ == "__main__":
    main()
