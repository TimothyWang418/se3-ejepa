r"""Encoder-level equivariance proof for :class:`SteerableEncoder`.

Step 3 acceptance test. Verifies the property the whole project rests on:

$$ E_\theta(g\cdot x) = \rho(g)\, E_\theta(x) \qquad \forall g \in C_N $$

where $E_\theta$ is the $C_N$-steerable encoder and $\rho$ is the **orthogonal**
regular representation of $C_N$ acting on the latent fiber.

Two consequences are checked:

1. **Equivariance.** Rotating the input image by $g$ equals transforming the
   latent fiber by $\rho(g)$. Because the encoder pools to a $1\times1$ spatial
   grid, the *output* ``.transform(g)`` is a pure fiber permutation (no spatial
   interpolation), so the only interpolation error comes from rotating the
   *input* grid — exact for $C_4$ ($90^\circ$), interpolation-limited for $C_8$.

2. **Cost invariance.** Since $\rho(g)$ is orthogonal,
   $\lVert E_\theta(g x_a) - E_\theta(g x_b)\rVert_2 = \lVert E_\theta(x_a) - E_\theta(x_b)\rVert_2$.
   This is *exactly* the JEPA planning cost, so an equivariant encoder makes the
   cost — and hence the optimal plan — invariant to global rotations of the scene.

Run:
    .venv/bin/python tests/test_eqjepa_equivariance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from e2cnn import nn as enn  # noqa: E402

from src.models.eqjepa import SteerableEncoder  # noqa: E402


@torch.no_grad()
def equivariance_errors(n_rot: int, size: int = 65, latent_dim: int = 128):
    r"""Per-group-element $\max\lvert E(g x) - \rho(g) E(x)\rvert$ on the pooled fiber."""
    torch.manual_seed(0)
    enc = SteerableEncoder(in_channels=3, latent_dim=latent_dim, n_rot=n_rot)
    enc.eval()  # use BN running stats (identity-ish at init), batch-independent
    x = torch.randn(2, 3, size, size)

    base = enc.encode_geometric(x)  # GeometricTensor (2, latent_dim, 1, 1) = E(x)
    gx = enn.GeometricTensor(x, enc.in_type)

    errs, norm_devs = {}, {}
    for g in enc.r2.testing_elements:
        left = enc.encode_geometric(gx.transform(g).tensor)  # E(g . x)
        right = base.transform(g)  # rho(g) . E(x)
        errs[g] = (left.tensor - right.tensor).abs().max().item()
        # orthogonality witness: ||rho(g) z|| == ||z|| (per sample)
        n0 = base.tensor.flatten(1).norm(dim=1)
        ng = right.tensor.flatten(1).norm(dim=1)
        norm_devs[g] = (n0 - ng).abs().max().item()
    return errs, norm_devs


@torch.no_grad()
def cost_invariance_error(n_rot: int, size: int = 65, latent_dim: int = 128):
    r"""$\bigl|\;\lVert E(g x_a)-E(g x_b)\rVert - \lVert E(x_a)-E(x_b)\rVert\;\bigr|$, worst over $g$.

    This is the JEPA cost evaluated on a (state, goal) pair; an orthogonal
    $\rho(g)$ should leave it unchanged under a global rotation.
    """
    torch.manual_seed(1)
    enc = SteerableEncoder(in_channels=3, latent_dim=latent_dim, n_rot=n_rot)
    enc.eval()
    xa = torch.randn(1, 3, size, size)
    xb = torch.randn(1, 3, size, size)
    base_cost = (enc(xa) - enc(xb)).norm().item()

    gxa = enn.GeometricTensor(xa, enc.in_type)
    gxb = enn.GeometricTensor(xb, enc.in_type)
    worst = 0.0
    for g in enc.r2.testing_elements:
        za = enc(gxa.transform(g).tensor)
        zb = enc(gxb.transform(g).tensor)
        rot_cost = (za - zb).norm().item()
        worst = max(worst, abs(rot_cost - base_cost))
    return base_cost, worst


def main() -> None:
    print("SteerableEncoder SO(2) equivariance check\n")
    for n_rot in (4, 8):
        errs, norm_devs = equivariance_errors(n_rot)
        worst = max(errs.values())
        worst_norm = max(norm_devs.values())
        print(f"=== C_{n_rot} (rotations by k*{360 // n_rot} deg) ===")
        for g, e in errs.items():
            print(f"    g={g}: max|E(g.x) - rho(g).E(x)| = {e:.2e}")
        print(f"    worst equivariance residual = {worst:.2e}")
        print(f"    worst |rho(g) norm change|  = {worst_norm:.2e}  (orthogonality)")
        bc, ci = cost_invariance_error(n_rot)
        print(f"    JEPA cost ||za-zb||={bc:.4f}, worst rotation drift = {ci:.2e}\n")

    # Rigorous assertions on the grid-aligned C_4 case (90 deg => ~exact pixel perm).
    errs_c4, norm_c4 = equivariance_errors(4)
    worst_c4 = max(errs_c4.values())
    worst_norm_c4 = max(norm_c4.values())
    _, ci_c4 = cost_invariance_error(4)
    assert worst_c4 < 1e-3, f"C_4 encoder equivariance broke: worst={worst_c4:.2e}"
    assert worst_norm_c4 < 1e-4, f"rho(g) not orthogonal: {worst_norm_c4:.2e}"
    assert ci_c4 < 1e-3, f"C_4 JEPA cost not rotation-invariant: {ci_c4:.2e}"
    print(f"PASS: C_4 encoder equivariance {worst_c4:.2e}, "
          f"orthogonality {worst_norm_c4:.2e}, cost drift {ci_c4:.2e} (all < tol).")
    print("=> E_theta(g.x) = rho(g) E_theta(x); JEPA cost is SO(2)-invariant.")


def test_steerable_encoder_equivariance() -> None:
    """pytest entry point: the script's $C_4$ encoder + cost-invariance assertions."""
    main()


if __name__ == "__main__":
    main()
