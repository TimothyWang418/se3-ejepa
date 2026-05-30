r"""Step 27 — the **tensor-product (bilinear) message** primitive.

Step 24 diagnosed a hard ceiling: the vanilla Vector-Neuron layers
(:class:`VNLinear`, :class:`VNReLU`) are **degree-1 homogeneous**, so a VN predictor
can only take *linear combinations* of its input vectors and can never form a
**product** such as the interaction torque $\omega=\hat r\times a$. This test certifies
the fix — :class:`VNTensorProduct` (a degree-2 cross-product layer) and
:class:`VNTPPredictor` (a latent predictor wired from it) — against three properties:

1. **SO(3) equivariance** (to the float floor) of both the bare layer and the predictor,
   $f(Rx)=R\,f(x)$ and $f(\rho(R)z, Ra)=\rho(R)\,f(z,a)$;
2. a **degree-2 homogeneity witness** $f(\lambda x)=\lambda^2 f(x)$ that *separates* the
   tensor-product layer from a degree-1 :class:`VNLinear` ($f(\lambda x)=\lambda f(x)$) —
   this is the missing primitive made concrete;
3. the **pseudovector / improper-rotation** sign flip $f(Qx)=\det(Q)\,Q f(x)=-Q f(x)$ for
   $Q\in\mathrm O(3)\setminus\mathrm{SO}(3)$ — documenting that the layer is $\mathrm{SO}(3)$-
   but deliberately **not** $\mathrm O(3)$-equivariant, exactly matching the cross-product
   teacher of Step 24.

Run:
    .venv/bin/python tests/test_step27_tensor_product.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.models.structured import (  # noqa: E402
    VNLinear,
    VNTensorProduct,
    VNTPPredictor,
)

torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------- #
# self-contained SO(3) / O(3) helpers (pure torch, no heavy imports)
# --------------------------------------------------------------------------- #
def rand_so3(gen: torch.Generator) -> torch.Tensor:
    r"""A Haar-random **proper** rotation $R\in\mathrm{SO}(3)$ via QR. ``-> (3, 3)``.

    QR of a Gaussian matrix gives an orthogonal $Q$ with $\det Q=\pm1$; flipping the
    last column by $\det Q$ projects onto $\mathrm{SO}(3)$ ($\det=+1$).
    """
    a = torch.randn(3, 3, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def rand_improper(gen: torch.Generator) -> torch.Tensor:
    r"""A random **improper** orthogonal matrix $Q\in\mathrm O(3)$, $\det Q=-1$. ``-> (3, 3)``."""
    q = rand_so3(gen)
    q[:, -1] = -q[:, -1]  # one reflection -> det = -1
    return q


def rotate_channels(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Apply $v\mapsto Rv$ to every type-1 vector. ``(..., 3) -> (..., 3)``."""
    return x @ R.transpose(-1, -2)


# --------------------------------------------------------------------------- #
# 1. SO(3) equivariance of the bare tensor-product layer
# --------------------------------------------------------------------------- #
@torch.no_grad()
def tp_equivariance_error(layer: VNTensorProduct, x: torch.Tensor, R: torch.Tensor) -> float:
    left = layer(rotate_channels(x, R))    # f(R.x)
    right = rotate_channels(layer(x), R)   # R.f(x)
    return (left - right).abs().max().item()


# --------------------------------------------------------------------------- #
# 2. degree-2 homogeneity witness: f(λx) = λ^2 f(x)  (vs VNLinear degree-1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def homogeneity_degree(layer: torch.nn.Module, x: torch.Tensor, lam: float) -> float:
    r"""Empirical degree $d$ from $\lVert f(\lambda x)\rVert = \lambda^{d}\lVert f(x)\rVert$."""
    base = layer(x).norm().item()
    scaled = layer(lam * x).norm().item()
    return torch.log(torch.tensor(scaled / base)).item() / torch.log(torch.tensor(lam)).item()


# --------------------------------------------------------------------------- #
# 3. pseudovector: f(Qx) = det(Q) Q f(x) for improper Q
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pseudovector_errors(layer: VNTensorProduct, x: torch.Tensor, Q: torch.Tensor):
    r"""Return ``(err_pseudo, err_naive)``: the residual against $\det(Q)Qf(x)$ and against $Qf(x)$."""
    left = layer(rotate_channels(x, Q))         # f(Q.x)
    detQ = torch.det(Q).item()
    pseudo = detQ * rotate_channels(layer(x), Q)  # det(Q) Q f(x)  -- correct for a pseudovector
    naive = rotate_channels(layer(x), Q)          # Q f(x)         -- the (wrong) true-vector law
    return (left - pseudo).abs().max().item(), (left - naive).abs().max().item()


# --------------------------------------------------------------------------- #
# 4. SO(3) equivariance of the full VNTPPredictor (latent + action)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predictor_equivariance_error(
    pred: VNTPPredictor, z: torch.Tensor, a: torch.Tensor, R: torch.Tensor
) -> float:
    b = z.shape[0]
    z_rot = rotate_channels(z.reshape(b, -1, 3), R).reshape(b, -1)   # ρ(R) z
    a_rot = rotate_channels(a.reshape(b, -1, 3), R).reshape(b, -1)   # R a
    left = pred(z_rot, a_rot)                                        # f(ρ(R)z, Ra)
    right = rotate_channels(pred(z, a).reshape(b, -1, 3), R).reshape(b, -1)  # ρ(R) f(z, a)
    return (left - right).abs().max().item()


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    print("Step 27 — tensor-product (bilinear) message: equivariance + degree witness\n")

    # ---- bare layer: SO(3) equivariance over many random rotations ----------
    layer = VNTensorProduct(in_channels=10, out_channels=8).eval()
    x = torch.randn(16, 10, 3, generator=gen)
    worst_equiv = max(tp_equivariance_error(layer, x, rand_so3(gen)) for _ in range(20))
    print(f"  [1] VNTensorProduct  max|f(R.x)-R.f(x)| over 20 rotations = {worst_equiv:.2e}")
    assert worst_equiv < 1e-4, f"VNTensorProduct broke SO(3) equivariance: {worst_equiv:.2e}"

    # ---- degree witness: TP is degree-2, VNLinear is degree-1 ---------------
    lin = VNLinear(10, 8).eval()
    deg_tp = homogeneity_degree(layer, x, lam=2.0)
    deg_lin = homogeneity_degree(lin, x, lam=2.0)
    print(f"  [2] homogeneity degree:  VNTensorProduct = {deg_tp:.4f} (want 2)   "
          f"VNLinear = {deg_lin:.4f} (want 1)")
    assert abs(deg_tp - 2.0) < 1e-3, f"TP is not degree-2: {deg_tp:.4f}"
    assert abs(deg_lin - 1.0) < 1e-3, f"VNLinear is not degree-1: {deg_lin:.4f}"

    # ---- pseudovector: improper rotation flips the sign --------------------
    Q = rand_improper(gen)
    err_pseudo, err_naive = pseudovector_errors(layer, x, Q)
    print(f"  [3] improper Q (det={torch.det(Q).item():+.2f}):  "
          f"|f(Q.x)-det(Q)Q f(x)| = {err_pseudo:.2e}   |f(Q.x)-Q f(x)| = {err_naive:.2e}")
    assert err_pseudo < 1e-4, f"TP is not a clean pseudovector: {err_pseudo:.2e}"
    assert err_naive > 1e-2, ("TP wrongly behaves as a true vector under reflection "
                              f"(err_naive={err_naive:.2e}); it should flip sign")

    # ---- full predictor: SO(3) equivariance (latent 48 = 16 vecs, action 6 = 2 vecs) ----
    pred = VNTPPredictor(latent_dim=48, action_dim=6, hidden=32, dim=3).eval()
    z = torch.randn(8, 48, generator=gen)
    a = torch.randn(8, 6, generator=gen)
    worst_pred = max(predictor_equivariance_error(pred, z, a, rand_so3(gen)) for _ in range(20))
    print(f"  [4] VNTPPredictor    max|f(ρz,Ra)-ρf(z,a)| over 20 rotations = {worst_pred:.2e}")
    assert worst_pred < 1e-4, f"VNTPPredictor broke SO(3) equivariance: {worst_pred:.2e}"

    print("\nPASS: the tensor-product message is degree-2, SO(3)-equivariant, and a pseudovector")
    print("=> supplies the bilinear primitive the degree-1 VN predictor structurally lacked (Step 24).")


def test_tp_layer_so3_equivariance() -> None:
    """pytest: VNTensorProduct + VNTPPredictor SO(3) equivariance, degree-2 witness, pseudovector."""
    main()


if __name__ == "__main__":
    main()
