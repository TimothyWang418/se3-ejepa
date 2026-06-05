r"""$C_4$ equivariance of the Step 64 *frame-averaged* pixel encoder and predictor (the load-bearing claim of the
frame-averaging route to a flat-AND-competitive pixel certificate).

Step 64 closes the §7 open problem by making a *plain* CNN / MLP exactly $C_4$-equivariant through frame averaging
(Reynolds operator) rather than steerable layers:

$$ E(g\!\cdot\!o)=\rho(g)\,E(o), \qquad f\big(\rho(g)\,z,\ \sigma(g)\,a\big)=\rho(g)\,f(z,a), $$

with $\rho=I_{n_{\mathrm{inv}}}\oplus(R_k\otimes I_{n_{\mathrm{vec}}})$ orthogonal and $\sigma(g_k)=R_k$ on the $2$D
action. Because the construction averages over the whole group, these identities hold to the floating-point floor by
construction — at init AND after a real optimisation step (frame averaging commutes with gradient descent on the
backbone). The non-equivariant controls (the bare backbone / bare MLP) must NOT commute, else the test is vacuous.

Run:  .venv/bin/python tests/test_step64.py
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

from step64_frame_averaged_pixel import (  # noqa: E402
    FrameAveragedEncoder,
    FrameAveragedPredictor,
    _rot_vec,
)

torch.set_default_dtype(torch.float32)

B = 16
_TOL_EQ = 1e-4      # architectural C_4 equivariance via frame averaging -> float floor
_TOL_CTRL = 1e-2    # the non-averaged control must miss by >> the equivariant floor


def _enc_residual(enc: FrameAveragedEncoder, *, seed: int) -> float:
    r"""$\max_{k}\lVert E(\mathrm{rot90}^k o) - \rho(g_k)E(o)\rVert$ over the full $C_4$ orbit."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(B, 3, 65, 65, generator=g)
    with torch.no_grad():
        z = enc(o)
        worst = 0.0
        for k in range(4):
            zr = enc(torch.rot90(o, k, dims=(-2, -1)))
            inv = z[:, : enc.n_inv]
            vec = _rot_vec(z[:, enc.n_inv :].reshape(B, enc.n_vec, 2), k)
            rho_z = torch.cat([inv, vec.reshape(B, 2 * enc.n_vec)], dim=1)
            worst = max(worst, float((zr - rho_z).abs().max()))
    return worst


def _pred_residual(pred: FrameAveragedPredictor, *, seed: int) -> float:
    r"""$\max_{k}\lVert \rho(g_k)f(z,a) - f(\rho(g_k)z,\sigma(g_k)a)\rVert$ over the full $C_4$ orbit."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, pred.latent_dim, generator=g)
    a = torch.randn(B, 2, generator=g)
    with torch.no_grad():
        worst = 0.0
        for k in range(4):
            lhs = pred._rho(pred(z, a), k)
            rhs = pred(pred._rho(z, k), _rot_vec(a, k))
            worst = max(worst, float((lhs - rhs).abs().max()))
    return worst


def test_frame_averaged_encoder_is_c4_equivariant() -> None:
    r"""``E(g.o) == rho(g) E(o)`` to the float floor; the bare CNN backbone control fails."""
    torch.manual_seed(0)
    enc = FrameAveragedEncoder()
    init_res = _enc_residual(enc, seed=1)
    assert init_res < _TOL_EQ, f"FA encoder not C_4-equivariant at init: {init_res:.2e}"

    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    o = torch.randn(32, 3, 65, 65, generator=g)
    tgt = torch.randn(32, enc.latent_dim, generator=g)
    for _ in range(15):
        opt.zero_grad(); ((enc(o) - tgt) ** 2).mean().backward(); opt.step()
    post_res = _enc_residual(enc, seed=2)
    assert post_res < _TOL_EQ, f"FA encoder C_4 equivariance broke AFTER training: {post_res:.2e}"

    # control: the bare backbone (no frame averaging) must NOT commute with rho
    o2 = torch.randn(B, 3, 65, 65, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        z = enc.backbone(o2)
        zr = enc.backbone(torch.rot90(o2, 1, dims=(-2, -1)))
        inv = z[:, : enc.n_inv]
        vec = _rot_vec(z[:, enc.n_inv :].reshape(B, enc.n_vec, 2), 1)
        rho_z = torch.cat([inv, vec.reshape(B, 2 * enc.n_vec)], dim=1)
        ctrl = float((zr - rho_z).abs().max())
    assert ctrl > _TOL_CTRL, f"bare backbone control unexpectedly commutes ({ctrl:.2e}); test may be vacuous"

    print(f"PASS: FA encoder C_4-equivariant init={init_res:.2e} -> post-train={post_res:.2e} "
          f"(< {_TOL_EQ:.0e}); bare-backbone control={ctrl:.2e} (> {_TOL_CTRL:.0e}).")


def test_frame_averaged_predictor_is_c4_equivariant() -> None:
    r"""``f(rho(g)z, sigma(g)a) == rho(g) f(z,a)`` to the float floor; the bare MLP control fails."""
    torch.manual_seed(0)
    pred = FrameAveragedPredictor()
    init_res = _pred_residual(pred, seed=1)
    assert init_res < _TOL_EQ, f"FA predictor not C_4-equivariant at init: {init_res:.2e}"

    opt = torch.optim.Adam(pred.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, pred.latent_dim, generator=g)
    a = torch.randn(64, 2, generator=g)
    tgt = torch.randn(64, pred.latent_dim, generator=g)
    for _ in range(30):
        opt.zero_grad(); ((pred(z, a) - tgt) ** 2).mean().backward(); opt.step()
    post_res = _pred_residual(pred, seed=2)
    assert post_res < _TOL_EQ, f"FA predictor C_4 equivariance broke AFTER training: {post_res:.2e}"

    # control: the bare MLP (no frame averaging), wrapped as a residual, must NOT commute
    zc = torch.randn(B, pred.latent_dim, generator=torch.Generator().manual_seed(3))
    ac = torch.randn(B, 2, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        def bare(z_, a_):
            return z_ + pred.mlp(torch.cat([z_, a_], dim=1))
        lhs = pred._rho(bare(zc, ac), 1)
        rhs = bare(pred._rho(zc, 1), _rot_vec(ac, 1))
        ctrl = float((lhs - rhs).abs().max())
    assert ctrl > _TOL_CTRL, f"bare MLP control unexpectedly commutes ({ctrl:.2e}); test may be vacuous"

    print(f"PASS: FA predictor C_4-equivariant init={init_res:.2e} -> post-train={post_res:.2e} "
          f"(< {_TOL_EQ:.0e}); bare-MLP control={ctrl:.2e} (> {_TOL_CTRL:.0e}).")
    print("=> frame averaging gives exact E and f equivariance from PLAIN nets: the basis of the flat-and-competitive pixel certificate.")


if __name__ == "__main__":
    test_frame_averaged_encoder_is_c4_equivariant()
    test_frame_averaged_predictor_is_c4_equivariant()
