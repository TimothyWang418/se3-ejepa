r"""Exact SE(3) equivariance of the 3D point-cloud world model (Step 6).

This is the 3D analogue of ``test_structured_equivariance.py``. There the
coordinate encoder was exactly SO(2)-equivariant for *every* planar angle; here
the :class:`~src.models.se3.SE3PointEncoder` is exactly **SE(3)**-equivariant on
$\mathbb{R}^3$ point clouds — for arbitrary rotations $R\in\mathrm{SO}(3)$ (no
grid, no interpolation floor) and arbitrary translations $t$.

We verify four things, each to the float floor (~$10^{-5}$):

1. **Translation invariance**: $E(x+t) = E(x)$ (the cloud is centred).
2. **SO(3) equivariance of the encoder**: $E(R\,x) = \rho(R)\,E(x)$, with
   $\rho(R)$ block-diagonal copies of $R$ acting on the $\ell=1$ vector latent.
3. **Joint equivariance of encoder + predictor**: feeding the equivariant latent
   and a type-1 action into :class:`~src.models.structured.VNPredictor`
   (``dim=3``), the predicted next latent rotates the same way —
   $f_\phi(\rho(R)z, R a) = \rho(R) f_\phi(z, a)$ — so the **whole** latent world
   model is SE(3)-equivariant, not just the encoder.
4. **Cost invariance**: the JEPA cost $\lVert E(x_a)-E(x_b)\rVert$ is unchanged by
   a joint rotation, because $\rho(R)$ is orthogonal.

Run:
    .venv/bin/python tests/test_se3_equivariance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.models.structured import VNPredictor  # noqa: E402


def rotate_points(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Apply $v \mapsto R v$ to every vector in ``x: (..., 3)`` (right-multiply)."""
    return x @ R.transpose(-1, -2)


@torch.no_grad()
def translation_error(enc: SE3PointEncoder, pos: torch.Tensor) -> float:
    r"""$\max\lvert E(x+t) - E(x)\rvert$ for a random global shift $t$."""
    t = torch.randn(1, 1, 3)
    return (enc(pos + t) - enc(pos)).abs().max().item()


@torch.no_grad()
def equivariance_error(enc: SE3PointEncoder, pos: torch.Tensor, R: torch.Tensor) -> float:
    r"""$\max\lvert E(R x) - R\,E(x)\rvert$ on the $\ell=1$ vector latent."""
    z = enc.encode_vectors(pos)  # (B, n_vec, 3)
    left = enc.encode_vectors(rotate_points(pos, R))  # E(R.x)
    right = rotate_points(z, R)  # R.E(x)
    return (left - right).abs().max().item()


@torch.no_grad()
def joint_rollout_error(
    enc: SE3PointEncoder, pred: VNPredictor, pos: torch.Tensor, a: torch.Tensor, R: torch.Tensor
) -> float:
    r"""$\max\lvert f(E(Rx), Ra) - R\,f(E(x), a)\rvert$ — whole-model equivariance."""
    z = enc(pos)
    z_next = pred(z, a)  # (B, latent_dim)
    b = z_next.shape[0]

    z_rot = enc(rotate_points(pos, R))
    a_rot = rotate_points(a.reshape(b, -1, 3), R).reshape(b, -1)
    z_next_rot = pred(z_rot, a_rot)

    # rotate the reference next-latent and compare
    ref = rotate_points(z_next.reshape(b, -1, 3), R).reshape(b, -1)
    return (z_next_rot - ref).abs().max().item()


@torch.no_grad()
def cost_invariance_error(enc: SE3PointEncoder, xa, xb, R: torch.Tensor) -> float:
    base = (enc(xa) - enc(xb)).norm().item()
    rot = (enc(rotate_points(xa, R)) - enc(rotate_points(xb, R))).norm().item()
    return abs(rot - base)


def main() -> None:
    torch.manual_seed(0)
    enc = SE3PointEncoder(n_out_vec=8, lmax=2, mul=8).eval()
    latent_dim = enc.latent_dim  # 24
    pred = VNPredictor(latent_dim=latent_dim, action_dim=3, hidden=32, dim=3).eval()

    B, N = 6, 24
    pos = torch.randn(B, N, 3)
    posb = torch.randn(B, N, 3)
    action = torch.randn(B, 3)

    print("SE(3) point-cloud world model — exact equivariance on R^3\n")
    print(f"    encoder latent_dim = {latent_dim} ({enc.n_out_vec} type-1 vectors)\n")

    worst = 0.0
    print("    test                              | error")
    print("    ----------------------------------+----------")

    e_trans = translation_error(enc, pos)
    worst = max(worst, e_trans)
    print(f"    translation invariance E(x+t)=E(x)|  {e_trans:.2e}")

    # several random rotations to be sure it is not an accident of one R
    eq_worst = 0.0
    ci_worst = 0.0
    joint_worst = 0.0
    for _ in range(5):
        R = o3.rand_matrix()
        eq_worst = max(eq_worst, equivariance_error(enc, pos, R))
        ci_worst = max(ci_worst, cost_invariance_error(enc, pos, posb, R))
        joint_worst = max(joint_worst, joint_rollout_error(enc, pred, pos, action, R))
    worst = max(worst, eq_worst, ci_worst, joint_worst)
    print(f"    SO(3) equivariance E(Rx)=R.E(x)   |  {eq_worst:.2e}")
    print(f"    joint enc+pred rollout (whole WM) |  {joint_worst:.2e}")
    print(f"    cost invariance ||E(xa)-E(xb)||   |  {ci_worst:.2e}")

    print(f"\n    worst over all checks = {worst:.2e}")
    assert worst < 1e-4, f"SE(3) equivariance broke: worst={worst:.2e}"
    print("PASS: encoder AND predictor are exactly SE(3)-equivariant on R^3")
    print("=> the WHOLE 3D latent world model is SE(3)-equivariant — Step 6 payoff.")


def test_se3_world_model_equivariance() -> None:
    """pytest entry point: encoder + predictor SE(3) assertions on $\\mathbb{R}^3$."""
    main()


if __name__ == "__main__":
    main()
