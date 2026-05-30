r"""Step 4: train EqJEPA and run the ordinary-vs-steerable A/B under rotation.

Controlled experiment. A single variable changes between the two runs — the
**encoder** (ordinary :class:`ConvEncoder` vs $C_4$-steerable
:class:`SteerableEncoder`). Everything else is identical: the same collected
PushT transitions, the same residual predictor, the same JEPA objective, and the
same optimiser *policy* (``ndim>=2`` -> Muon, else AdamW).

What it shows:
  1. Both models train without latent collapse (latent_std stays > 0).
  2. The steerable encoder's planning cost is ~invariant to global $90^\circ$
     scene rotations; the ordinary CNN's cost drifts substantially.
  3. Muon-trained steerable encoder is STILL exactly equivariant — hard
     basis-constrained equivariance is optimiser-agnostic (clarifies the scope
     of the Symmetry-Compatible-Optimizer concern, which targets *soft*
     /augmentation-based equivariance).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step4_train_ab.py
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
from e2cnn import nn as enn  # noqa: E402

import stable_worldmodel as swm  # noqa: E402

from src.models.eqjepa import ConvEncoder, EqJEPA, SteerableEncoder  # noqa: E402
from src.training.jepa import collect_transitions, fov_cost_drift, train_jepa  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


@torch.no_grad()
def trained_encoder_equivariance(encoder, size: int = 65) -> dict:
    r"""Worst $\max|E(g x)-\rho(g)E(x)|$ over $C_4$ for a *trained* SteerableEncoder."""
    encoder.eval()
    x = torch.randn(2, 3, size, size)
    base = encoder.encode_geometric(x)
    gx = enn.GeometricTensor(x, encoder.in_type)
    worst = 0.0
    for g in encoder.r2.testing_elements:
        left = encoder.encode_geometric(gx.transform(g).tensor)
        right = base.transform(g)
        worst = max(worst, (left.tensor - right.tensor).abs().max().item())
    return {"worst_c4_residual": worst}


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"  # e2cnn is CPU-reliable; keeps the A/B on equal footing
    size = 65  # odd -> torch.rot90 is an exact, centred pixel permutation
    latent_dim, action_dim = 128, 2

    banner("[A] Build World + collect shared PushT transitions (random policy)")
    world = swm.World("swm/PushT-v1", num_envs=4, image_shape=(size, size))
    n_trans = 1200
    obs, act, nxt = collect_transitions(world, n_trans, seed=0)
    world.close()
    n_train = int(0.8 * obs.shape[0])
    tr = slice(0, n_train)
    te = slice(n_train, obs.shape[0])
    print(f"    collected {obs.shape[0]} transitions  obs={tuple(obs.shape)}  act={tuple(act.shape)}")
    print(f"    train={n_train}  held-out={obs.shape[0] - n_train}")

    # identical training hyperparameters for both arms
    hp = dict(
        epochs=20, batch_size=64, muon_lr=0.02, adamw_lr=1e-3,
        ema_decay=0.99, var_coef=1.0, device=device, seed=0,  # strong anti-collapse
    )

    banner("[B] Train ORDINARY EqJEPA (ConvEncoder)")
    torch.manual_seed(0)
    ord_model = EqJEPA(
        latent_dim=latent_dim, action_dim=action_dim,
        encoder=ConvEncoder(in_channels=3, latent_dim=latent_dim),
    )
    ord_hist = train_jepa(ord_model, obs[tr], act[tr], nxt[tr], **hp)

    banner("[C] Train STEERABLE EqJEPA (C_4 SteerableEncoder)")
    torch.manual_seed(0)
    steer_model = EqJEPA(
        latent_dim=latent_dim, action_dim=action_dim,
        encoder=SteerableEncoder(in_channels=3, latent_dim=latent_dim, n_rot=4),
    )
    steer_hist = train_jepa(steer_model, obs[tr], act[tr], nxt[tr], **hp)

    banner("[D] FoV A/B — relative drift of planning cost under 90 deg rotations")
    ord_drift = fov_cost_drift(ord_model, obs[te], device=device, seed=1)
    steer_drift = fov_cost_drift(steer_model, obs[te], device=device, seed=1)
    print("    rotation |  ordinary drift  |  steerable drift")
    print("    ---------+------------------+-----------------")
    for ang in (90, 180, 270):
        print(f"    {ang:4d} deg |   {ord_drift['drift'][ang]:12.4e} |  {steer_drift['drift'][ang]:12.4e}")
    ord_worst = max(ord_drift["drift"][a] for a in (90, 180, 270))
    steer_worst = max(steer_drift["drift"][a] for a in (90, 180, 270))
    print(f"    worst    |   {ord_worst:12.4e} |  {steer_worst:12.4e}")

    banner("[E] Post-training equivariance of the steerable encoder (Muon)")
    eq = trained_encoder_equivariance(steer_model.encoder, size=size)
    print(f"    worst C_4 residual after training = {eq['worst_c4_residual']:.2e}")

    banner("STEP 4 SUMMARY")
    print(f"    latent_std  ordinary={ord_hist['latent_std']:.3f}  steerable={steer_hist['latent_std']:.3f}  (>0 => no collapse)")
    print(f"    pred_loss   ordinary={ord_hist['pred_loss'][-1]:.4f}  steerable={steer_hist['pred_loss'][-1]:.4f}")
    print(f"    cost drift  ordinary={ord_worst:.3e}  steerable={steer_worst:.3e}")
    print(f"    steerable equivariance after Muon training = {eq['worst_c4_residual']:.2e}")
    ratio = ord_worst / max(steer_worst, 1e-12)
    print(f"    => steerable cost is {ratio:.1f}x more rotation-stable than ordinary")
    assert steer_worst < 1e-3, f"steerable cost drift too high: {steer_worst:.2e}"
    assert eq["worst_c4_residual"] < 1e-3, "training broke steerable equivariance"
    print("    PASS: equivariant encoder gives a rotation-invariant planning cost,")
    print("          preserved exactly through Muon training.")


if __name__ == "__main__":
    main()
