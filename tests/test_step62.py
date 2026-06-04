r"""$C_4$ equivariance of the Step 62 *pixel-latent* predictor (the load-bearing claim of the pixel certificate).

Step 62 lifts the predictability certificate to a learned PIXEL latent. The exact orbit-flatness of its multi-step
latent rollout rests on the latent predictor being $C_4$-equivariant:

$$ f\big(\rho(g)\,z,\ \sigma(g)\,a\big) \;=\; \rho(g)\, f(z, a), $$

where $\rho$ is the regular representation of $C_4$ on the latent and $\sigma=\mathrm{irrep}(1)$ acts on the $2$D
action as a rotating vector. :class:`SteerableLatentPredictor` builds this from two $1\times1$ steerable convolutions
on ``regular_fields ⊕ irrep(1)``, so the identity holds to the float floor by construction. This test is the
structural guard for that claim; an ordinary :class:`LatentPredictor` is the non-vacuous control (it does NOT
commute). The equivariance is architectural, so it must hold both at init and after a real optimisation step.

Run:  .venv/bin/python tests/test_step62.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from src.models.eqjepa import LatentPredictor  # noqa: E402
from step62_pixel_latent_certificate import SteerableLatentPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)

LATENT_DIM, N_ROT, B = 32, 4, 16
_TOL_EQ = 1e-4      # architectural C_4 equivariance (1x1 steerable conv) -> float floor
_TOL_MLP = 1e-2     # the ordinary control must miss by >> the equivariant floor


def _equiv_residual(pred, *, seed: int) -> float:
    r"""$\max\lVert \rho(g)f(z,a) - f(\rho(g)z,\sigma(g)a)\rVert$ over the $90^\circ$ element of $C_4$."""
    from e2cnn import nn as enn

    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, LATENT_DIM, generator=g)
    a = torch.randn(B, 2, generator=g)
    elt = list(pred.r2.testing_elements)[1]                       # 90-degree rotation
    with torch.no_grad():
        lhs = enn.GeometricTensor(pred(z, a).reshape(B, LATENT_DIM, 1, 1), pred.lat_t).transform(elt).tensor.reshape(B, LATENT_DIM)
        zt = enn.GeometricTensor(z.reshape(B, LATENT_DIM, 1, 1), pred.lat_t).transform(elt).tensor.reshape(B, LATENT_DIM)
        at = enn.GeometricTensor(a.reshape(B, 2, 1, 1), pred.act_t).transform(elt).tensor.reshape(B, 2)
        rhs = pred(zt, at)
    return float((lhs - rhs).abs().max())


def test_pixel_latent_predictor_is_c4_equivariant() -> None:
    r"""``f(rho(g)z, sigma(g)a) == rho(g) f(z,a)`` for the steerable predictor; ordinary MLP control fails."""
    torch.manual_seed(0)
    pred = SteerableLatentPredictor(LATENT_DIM, n_rot=N_ROT)
    init_res = _equiv_residual(pred, seed=1)
    assert init_res < _TOL_EQ, f"steerable latent predictor not C_4-equivariant at init: {init_res:.2e}"

    # property must survive a real optimisation step (architectural, but we check)
    opt = torch.optim.Adam(pred.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, LATENT_DIM, generator=g)
    a = torch.randn(64, 2, generator=g)
    tgt = torch.randn(64, LATENT_DIM, generator=g)
    for _ in range(30):
        opt.zero_grad(); ((pred(z, a) - tgt) ** 2).mean().backward(); opt.step()
    post_res = _equiv_residual(pred, seed=2)
    assert post_res < _TOL_EQ, f"predictor C_4 equivariance broke AFTER training: {post_res:.2e}"

    # non-equivariant control: ordinary residual MLP must NOT commute (else the test is vacuous)
    mlp = LatentPredictor(LATENT_DIM, action_dim=2)
    gm = torch.Generator().manual_seed(3)
    zc = torch.randn(B, LATENT_DIM, generator=gm); ac = torch.randn(B, 2, generator=gm)
    from e2cnn import nn as enn
    elt = list(pred.r2.testing_elements)[1]
    with torch.no_grad():
        lhs = enn.GeometricTensor(mlp(zc, ac).reshape(B, LATENT_DIM, 1, 1), pred.lat_t).transform(elt).tensor.reshape(B, LATENT_DIM)
        zt = enn.GeometricTensor(zc.reshape(B, LATENT_DIM, 1, 1), pred.lat_t).transform(elt).tensor.reshape(B, LATENT_DIM)
        at = enn.GeometricTensor(ac.reshape(B, 2, 1, 1), pred.act_t).transform(elt).tensor.reshape(B, 2)
        mlp_res = float((lhs - mlp(zt, at)).abs().max())
    assert mlp_res > _TOL_MLP, f"ordinary MLP control unexpectedly commutes ({mlp_res:.2e}); test may be vacuous"

    print(f"PASS: steerable latent predictor C_4-equivariant init={init_res:.2e} -> post-train={post_res:.2e} "
          f"(< {_TOL_EQ:.0e}); ordinary MLP control={mlp_res:.2e} (> {_TOL_MLP:.0e}).")
    print("=> f(rho(g)z, sigma(g)a) = rho(g) f(z,a): the predictor half of the exact pixel-latent certificate.")


if __name__ == "__main__":
    test_pixel_latent_predictor_is_c4_equivariant()
