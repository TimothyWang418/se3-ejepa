r"""Guard for Step 72 / Stage 2a — the FetchPush equivariant world model is SO(2)-equivariant end-to-end.

Composes Stage 1 (rotate_obs / obs_to_vn) with the VN encoder+predictor (src/models/structured.py): rotating the scene
rotates the latent the same way, and rotating (latent, action) rotates the predicted next latent the same way. This is
the architectural guarantee behind Theorem A's orbit-flatness on this task — verified before any training result.

Run:  .venv/bin/python tests/test_step72_wm_equivariance.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fetchpush_symmetry as sym  # noqa: E402
from step72_mujoco_certificate import EquivariantWM, latent_rotation  # noqa: E402


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def test_fetchpush_wm_is_so2_equivariant() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    wm = EquivariantWM(latent_dim=64, hidden=32).double().eval()
    theta = 0.7

    obs = rng.standard_normal((12, sym.OBS_DIM))
    v0, _ = sym.obs_to_vn(obs)                                   # (12, N_VEC, 2)
    v1, _ = sym.obs_to_vn(sym.rotate_obs(obs, theta))            # rotated scene
    V0 = torch.tensor(v0, dtype=torch.float64)
    V1 = torch.tensor(v1, dtype=torch.float64)

    with torch.no_grad():
        # (i) encoder equivariance: E(rho(g) obs) == rho_latent(g) E(obs)
        z0 = wm.encode(V0)
        z1 = wm.encode(V1)
        z0_rot = latent_rotation(z0, theta)
        enc_err = (z1 - z0_rot).abs().max().item()
        assert enc_err < 1e-9, f"encoder not SO(2)-equivariant on FetchPush obs: max err {enc_err:.2e}"

        # (ii) joint predictor equivariance: f(rho z, R a) == rho f(z, a)
        a_xy = torch.tensor(rng.standard_normal((12, 2)), dtype=torch.float64)
        R = _rot2(theta)
        zt0 = wm.pred(z0, a_xy)
        zt1 = wm.pred(latent_rotation(z0, theta), a_xy @ R.T)
        pred_err = (zt1 - latent_rotation(zt0, theta)).abs().max().item()
        assert pred_err < 1e-9, f"predictor not jointly SO(2)-equivariant: max err {pred_err:.2e}"

    print(f"PASS: FetchPush equivariant WM is SO(2)-equivariant — encoder err {enc_err:.1e}, "
          f"joint predictor err {pred_err:.1e} (machine precision).")
    print("=> rotating the scene (and action) rotates the latent identically — Theorem A's orbit-flatness is "
          "architectural for this task, before any training.")


if __name__ == "__main__":
    test_fetchpush_wm_is_so2_equivariant()
