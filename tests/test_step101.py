"""step101 — the S₂-equivariant TD-MPC2 latent module: G-EQ at machine precision (the bridge's core)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import step101_eq_world_model as s101  # noqa: E402


def test_g_eq_machine_precision():
    res, ok = s101.g_eq_test(seed=0)
    assert ok
    for k, v in res.items():
        assert v <= 1e-5, (k, v)            # encoder/dynamics equivariant, reward invariant, pi equivariant


def test_rho_z_is_involution_and_swaps_legs():
    import torch
    torch.manual_seed(1)
    z = torch.randn(8, s101.LAT)
    assert torch.allclose(s101.rho_z(s101.rho_z(z)), z)        # involution
    inv, r, l = z[:, :s101.INV_D], z[:, s101.INV_D:s101.INV_D + s101.LEG_D], z[:, s101.INV_D + s101.LEG_D:]
    rz = s101.rho_z(z)
    assert torch.allclose(rz[:, :s101.INV_D], inv)             # invariant block fixed
    assert torch.allclose(rz[:, s101.INV_D:s101.INV_D + s101.LEG_D], l)   # R<-L
