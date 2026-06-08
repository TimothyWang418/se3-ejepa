r"""Guard for Step 79 — certified-control co-demonstration on controlled Lorenz-96."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step79_certified_control as s79  # noqa: E402
DT = torch.float64


def test_controlled_dynamics_is_ZN_equivariant() -> None:
    torch.manual_seed(0); N = 12
    x = torch.randn(N, dtype=DT); u = torch.randn(N, dtype=DT)
    for s in (1, 3, 7):
        lhs = s79.l96_controlled_rhs(torch.roll(x, s), torch.roll(u, s))
        rhs = torch.roll(s79.l96_controlled_rhs(x, u), s)
        assert torch.allclose(lhs, rhs, atol=1e-12), f"shift {s}: field not Z_N-equivariant"
        mlhs = s79.rk4_controlled(torch.roll(x, s), torch.roll(u, s))
        mrhs = torch.roll(s79.rk4_controlled(x, u), s)
        assert torch.allclose(mlhs, mrhs, atol=1e-12), f"shift {s}: RK4 map not equivariant"
    print("PASS: controlled Lorenz-96 field and RK4 map are Z_N-equivariant.")


def test_world_model_conv_is_equivariant_mlp_is_not() -> None:
    torch.manual_seed(0); N = 12
    conv = s79.L96ControlledConv(N).double()
    mlp = s79.L96ControlledMLP(N).double()
    x = torch.randn(N, dtype=DT); u = torch.randn(N, dtype=DT)
    for s in (1, 5):
        c_lhs = conv(torch.roll(x, s), torch.roll(u, s))
        c_rhs = torch.roll(conv(x, u), s, dims=-1)
        assert torch.allclose(c_lhs, c_rhs, atol=1e-10), f"conv shift {s}: not equivariant"
    m_lhs = mlp(torch.roll(x, 1), torch.roll(u, 1))
    m_rhs = torch.roll(mlp(x, u), 1, dims=-1)
    assert not torch.allclose(m_lhs, m_rhs, atol=1e-6), "MLP baseline should NOT be Z_N-equivariant"
    print("PASS: controlled conv WM is Z_N-equivariant; MLP baseline is not.")


def test_planner_is_orbit_equivariant() -> None:
    torch.manual_seed(0); N = 12
    model = s79.make_equivariant_wm(N).double()       # untrained equivariant WM is enough (equivariant by construction)
    mu = torch.zeros(N, dtype=DT); sd = torch.ones(N, dtype=DT)
    x0 = torch.randn(N, dtype=DT)
    for s in (1, 5):
        u_a = s79.plan_control(model, x0, mu, sd, H=6, seed=0)
        u_b = s79.plan_control(model, torch.roll(x0, s), mu, sd, H=6, seed=0)
        assert torch.allclose(u_b, torch.roll(u_a, s, dims=-1), atol=1e-6), \
            f"shift {s}: planner not orbit-equivariant (max diff {(u_b-torch.roll(u_a,s,-1)).abs().max():.2e})"
    print("PASS: planner commutes with the Z_N shift (exactly orbit-equivariant).")


def test_true_controlled_map_certificate_is_chaotic() -> None:
    import numpy as np
    import step78_certified_horizon_ci as s78
    import step74_lorenz96_spectrum as s74
    torch.manual_seed(0); N = 12
    traj = s74.attractor_traj(N, 400, 0, "cpu").double()
    mu, sd = traj.mean(0), traj.std(0) + 1e-8
    x0 = (traj[len(traj)//2] - mu) / sd
    g = lambda xn: (s79.rk4_controlled(xn * sd + mu, torch.zeros_like(xn)) - mu) / sd
    logR = s78.qr_logR_series(g, x0, n_steps=1200, warmup=200)
    point, lo, hi = s78.bootstrap_spectrum_ci(logR, s74.DTMAP, n_boot=400, block=40, seed=0)
    assert lo[0] > 0, f"true controlled map (u=0) must be chaotic; lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]"
    assert abs(float(point.sum()) - (-N)) / N < 0.05, "Liouville sum(lambda)=-N must hold for the controlled field at u=0"
    print(f"PASS: true controlled-map certificate is chaotic (lambda1 CI>0) and honors Liouville.")


if __name__ == "__main__":
    test_controlled_dynamics_is_ZN_equivariant()
    test_world_model_conv_is_equivariant_mlp_is_not()
    test_planner_is_orbit_equivariant()
    test_true_controlled_map_certificate_is_chaotic()
    print("step79 phase-0+1+3 guard PASS.")
