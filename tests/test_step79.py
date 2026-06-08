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


if __name__ == "__main__":
    test_controlled_dynamics_is_ZN_equivariant()
    print("step79 phase-0 guard PASS.")
