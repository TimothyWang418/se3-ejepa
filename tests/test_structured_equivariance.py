r"""Exact *continuous* SO(2) equivariance of the structured-state encoder.

Contrast with ``test_eqjepa_equivariance.py`` (the pixel encoder): there, only
$90^\circ$ rotations were float-exact and $45^\circ$ hit a ~$10^{-1}$
interpolation floor. Here the encoder acts on coordinates, so equivariance holds
to the float floor (~$10^{-6}$) for *arbitrary* angles $\alpha$ — including the
$37^\circ$, $45^\circ$, $123.4^\circ$ that wreck the pixel path.

We verify, for random continuous angles,
$$ E_\theta(R(\alpha)\cdot x) = \rho(\alpha)\,E_\theta(x), \qquad
   \rho(\alpha)\ \text{block-diagonal rotations (orthogonal)}, $$
and the consequent rotation-invariance of the JEPA cost $\lVert E(x_a)-E(x_b)\rVert$.

Run:
    .venv/bin/python tests/test_structured_equivariance.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.geometry.so2 import rotate_vectors  # noqa: E402
from src.models.structured import StructuredStateEncoder  # noqa: E402


@torch.no_grad()
def equivariance_error(enc: StructuredStateEncoder, x: torch.Tensor, alpha: float) -> float:
    r"""$\max\lvert E(R_\alpha x) - R_\alpha E(x)\rvert$ on the vector latent."""
    z = enc.encode_vectors(x)  # (B, D/2, 2)
    left = enc.encode_vectors(rotate_vectors(x, alpha))  # E(R.x)
    right = rotate_vectors(z, alpha)  # R.E(x)
    return (left - right).abs().max().item()


@torch.no_grad()
def cost_invariance_error(enc: StructuredStateEncoder, xa, xb, alpha: float) -> float:
    base = (enc(xa) - enc(xb)).norm().item()
    rot = (enc(rotate_vectors(xa, alpha)) - enc(rotate_vectors(xb, alpha))).norm().item()
    return abs(rot - base)


def main() -> None:
    torch.manual_seed(0)
    enc = StructuredStateEncoder(n_vec=6, latent_dim=128, hidden=64).eval()
    x = torch.randn(8, 6, 2)
    xb = torch.randn(8, 6, 2)

    print("Structured-state encoder — exact continuous SO(2) equivariance\n")
    angles = {
        "0":     0.0,
        "37deg": math.radians(37.0),
        "45deg": math.radians(45.0),     # the angle the pixel path cannot do
        "90deg": math.radians(90.0),
        "123.4deg": math.radians(123.4),
        "random": 2 * math.pi * torch.rand(()).item(),
    }
    worst = 0.0
    for name, a in angles.items():
        e = equivariance_error(enc, x, a)
        ci = cost_invariance_error(enc, x, xb, a)
        worst = max(worst, e, ci)
        print(f"    alpha={name:9s}: max|E(R.x)-R.E(x)| = {e:.2e}   cost drift = {ci:.2e}")

    print(f"\n    worst over all angles = {worst:.2e}")
    assert worst < 1e-4, f"continuous SO(2) equivariance broke: worst={worst:.2e}"
    print("PASS: coordinate encoder is exactly SO(2)-equivariant for ALL angles")
    print("=> no interpolation floor — this is the thesis payoff of the coordinate path.")


def test_continuous_so2_equivariance() -> None:
    """pytest entry point: arbitrary-angle SO(2) coordinate-encoder assertions."""
    main()


if __name__ == "__main__":
    main()
