r"""Post-**training** SE(3) equivariance of the 3D latent world model.

The other four equivariance tests check the property *at initialisation* (random
weights). Project policy (CLAUDE.md) requires verifying it **after training too**:
optimisation must not silently break the symmetry -- e.g. by someone later inserting a
non-equivariant layer, a bias on an $\ell=1$ field, or a coordinate-dependent
normalisation. Because the encoder ($E(Rx)=\rho(R)E(x)$ with $\rho$ orthogonal) and the
VN predictor ($f(\rho(R)z,Ra)=\rho(R)f(z,a)$) are equivariant *by construction*, the
composed residual is forced to the float floor regardless of the weights -- so this test
is a structural regression guard: it stays green only while the architecture stays exactly
SE(3)-equivariant through a real Muon/AdamW + EMA-target + VICReg training run.

We also assert a **non**-equivariant MLP-encoder control has a large residual, so the
test is not vacuously satisfied (it would catch a broken "equivariant" model).

Run:
    .venv/bin/python tests/test_post_training_equivariance.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from e3nn import o3  # noqa: E402
from torch import nn  # noqa: E402

from src.models.eqjepa import EqJEPA, LatentPredictor  # noqa: E402
from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.models.structured import VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

N_POINTS = 24
N_OUT_VEC = 8
LATENT_DIM = 3 * N_OUT_VEC   # 24 (divisible by 3 for VNPredictor dim=3)
ACTION_DIM = 3


def rotate_points(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""$v \mapsto R v$ on every vector. ``(...,3)->(...,3)``."""
    return x @ R.transpose(-1, -2)


def rotate_latent(z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""$\rho(R)$ = block-diagonal copies of $R$ on a flattened stack of 3-vectors."""
    b = z.shape[0]
    return rotate_points(z.reshape(b, -1, 3), R).reshape(b, -1)


@torch.no_grad()
def composed_equiv_err(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor) -> float:
    r"""End-to-end residual $\max\lvert\rho(R)f(E(x),a)-f(E(Rx),Ra)\rvert$."""
    lhs = rotate_latent(model.predictor(model.encoder(S), A), R)
    rhs = model.predictor(model.encoder(rotate_points(S, R)), rotate_points(A, R))
    return (lhs - rhs).abs().max().item()


def _worst_over_rotations(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, n: int = 5) -> float:
    return max(composed_equiv_err(model, S, A, o3.rand_matrix()) for _ in range(n))


def _equivariant_teacher(S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""A tiny exactly-SO(3) teacher (drift + torque) so training is non-trivial."""
    xt = S - S.mean(dim=1, keepdim=True)
    a = A[:, None, :]
    return S + 0.15 * a + 0.15 * torch.cross(a.expand_as(xt), xt, dim=-1)


def _synthetic_transitions(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(n, N_POINTS, 3, generator=g) * torch.tensor([1.0, 0.55, 0.3])
    A = (torch.randn(n, ACTION_DIM, generator=g) * 0.6).clamp(-1, 1)
    return S, A, _equivariant_teacher(S, A)


class _MLPEncoder(nn.Module):
    r"""Non-equivariant control: flattens raw coords (no centering, no prior)."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.net = nn.Sequential(
            nn.Linear(N_POINTS * 3, 128), nn.ReLU(inplace=True), nn.Linear(128, LATENT_DIM)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,P,3)->(B,D)
        return self.net(x.flatten(1))


def _build_eq() -> EqJEPA:
    enc = SE3PointEncoder(n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    pred = VNPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=32, dim=3)
    return EqJEPA(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, encoder=enc, predictor=pred)


def _build_mlp() -> EqJEPA:
    enc = _MLPEncoder()
    pred = LatentPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=128)
    return EqJEPA(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, encoder=enc, predictor=pred)


def test_se3_jepa_equivariance_survives_training() -> None:
    r"""The composed SE(3) residual stays at the float floor after a real training run."""
    torch.manual_seed(0)
    S, A, S2 = _synthetic_transitions(400, seed=0)
    St, At, _ = _synthetic_transitions(64, seed=7)

    eq = _build_eq()
    init_err = _worst_over_rotations(eq, St, At)
    assert init_err < 1e-4, f"equivariant model not equivariant at init: {init_err:.2e}"

    train_jepa(eq, S, A, S2, epochs=12, batch_size=128, var_coef=0.1, seed=0, log_every=10**9)

    post_err = _worst_over_rotations(eq, St, At)
    assert post_err < 1e-4, f"SE(3) equivariance broke AFTER training: worst={post_err:.2e}"

    # control: the non-equivariant MLP encoder must FAIL the same check (test is not vacuous)
    mlp = _build_mlp()
    train_jepa(mlp, S, A, S2, epochs=12, batch_size=128, var_coef=0.1, seed=0, log_every=10**9)
    mlp_err = _worst_over_rotations(mlp, St, At)
    assert mlp_err > 1e-2, f"MLP control unexpectedly equivariant ({mlp_err:.2e}); test may be vacuous"

    print(f"PASS: VN composed residual init={init_err:.2e} -> post-train={post_err:.2e} (< 1e-4); "
          f"MLP control={mlp_err:.2e} (>> floor).")
    print("=> the 3D latent world model stays EXACTLY SE(3)-equivariant through optimisation.")


if __name__ == "__main__":
    test_se3_jepa_equivariance_survives_training()
