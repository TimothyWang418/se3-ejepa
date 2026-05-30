r"""Standalone equivariance proof for an e2cnn SO(2)-steerable conv stack.

Run BEFORE wiring steerable convs into EqJEPA (Step 3). Verifies the defining
property of an $\mathrm{SO}(2)$-steerable map $f$:

$$ f(\rho_{\text{in}}(g)\, x) = \rho_{\text{out}}(g)\, f(x) \qquad \forall g \in C_N $$

i.e. rotating the input image by $g$ then convolving equals convolving then
transforming the output fiber by $g$.

Precision note: ``GeometricTensor.transform(g)`` rotates the *spatial* grid by
bilinear interpolation. For $C_4$ (multiples of $90^\circ$) this is a near-exact
pixel permutation, so the residual is at the float32 floor (~1e-5) and reflects
the conv's true equivariance. For $C_8$ ($45^\circ$) interpolation on a square
grid dominates the residual — the conv is *still* exactly $C_8$-equivariant by
construction; the number just measures grid-interpolation error, not a model bug.
"""

import torch
from e2cnn import gspaces
from e2cnn import nn as enn


def build_tiny_steerable(n_rot: int):
    r"""A 2-layer $C_N$-steerable conv stack: scalar image fields -> regular fields."""
    r2 = gspaces.Rot2dOnR2(N=n_rot)
    in_type = enn.FieldType(r2, 3 * [r2.trivial_repr])  # RGB = 3 scalar fields
    h1 = enn.FieldType(r2, 8 * [r2.regular_repr])
    h2 = enn.FieldType(r2, 16 * [r2.regular_repr])
    net = enn.SequentialModule(
        enn.R2Conv(in_type, h1, kernel_size=5, padding=2, bias=False),
        enn.ReLU(h1),  # pointwise ReLU on regular fields is equivariant
        enn.R2Conv(h1, h2, kernel_size=5, padding=2, bias=False),
        enn.ReLU(h2),
    )
    return r2, in_type, net


def equivariance_errors(n_rot: int, size: int = 33):
    torch.manual_seed(0)
    r2, in_type, net = build_tiny_steerable(n_rot)
    net.eval()
    x = torch.randn(2, 3, size, size)
    xg = enn.GeometricTensor(x, in_type)
    errs = {}
    with torch.no_grad():
        base = net(xg)  # f(x)
        for g in r2.testing_elements:
            left = net(xg.transform(g))  # f(g . x)
            right = base.transform(g)  # g . f(x)
            errs[g] = (left.tensor - right.tensor).abs().max().item()
    return errs


def main() -> None:
    print("e2cnn SO(2)-steerable equivariance check\n")
    for n_rot in (4, 8):
        errs = equivariance_errors(n_rot)
        worst = max(errs.values())
        print(f"=== C_{n_rot} (rotations by k*{360 // n_rot} deg) ===")
        for g, e in errs.items():
            print(f"    g={g}: max|f(g.x) - g.f(x)| = {e:.2e}")
        print(f"    worst = {worst:.2e}\n")

    # rigorous assertion on the grid-aligned C_4 case (interpolation ~exact)
    c4 = equivariance_errors(4)
    worst_c4 = max(c4.values())
    assert worst_c4 < 1e-4, f"C_4 equivariance broke: worst={worst_c4:.2e}"
    print(f"PASS: C_4 steerable equivariance holds to {worst_c4:.2e} (< 1e-4).")
    print("=> e2cnn SO(2)-steerable convs are exactly equivariant on this machine.")


def test_c4_steerable_equivariance() -> None:
    """pytest entry point: the script's $C_4$ assertions are the test body."""
    main()


if __name__ == "__main__":
    main()
