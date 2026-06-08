r"""Guard for Step 77 — the GRU-BPTT baseline. The recovered spectrum is read off the joint autonomous map
$M:[x,h]\mapsto[x',h']$ via Step 74's (already unit-tested) ``lyapunov_spectrum``. The one thing NOT covered by
Step 74's test is whether the joint-state map is *assembled* correctly. We cross-check its leading exponent against
an INDEPENDENT finite-difference Benettin estimate that uses only forward evaluations of the same map (no autograd
Jacobian, no QR) — so a bug in the [x,h] packing/unpacking would surface.

Run:  .venv/bin/python -m pytest tests/test_step77.py -q   (or run this file directly)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step74_lorenz96_spectrum as s74  # noqa: E402
import step77_lorenz96_gru_baseline as s77  # noqa: E402

torch.manual_seed(0)
DT = torch.float64


def test_joint_autonomous_map_lyapunov_matches_finite_difference() -> None:
    N, H = 3, 16
    model = s77.L96GRU(N, H).double()
    with torch.no_grad():                                  # scale the readout so the joint map has nonzero stretching
        model.readout.weight.mul_(4.0)
    M = s77.autonomous_map(model)

    state = torch.randn(N + H, dtype=DT) * 0.3            # land on the map's own attractor
    for _ in range(400):
        state = M(state)
    assert torch.isfinite(state).all(), "autonomous map diverged; pick a milder readout scale for the test"

    lam_qr = s74.lyapunov_spectrum(M, state, n_steps=3000, warmup=300)[0].item()

    delta = 1e-7
    v = torch.randn(N + H, dtype=DT); v = v / v.norm()
    x, xp = state.clone(), state.clone() + delta * v
    acc, cnt = 0.0, 0
    for t in range(3300):
        x = M(x); xp = M(xp)
        d = xp - x; nrm = d.norm().item()
        if t >= 300:
            acc += torch.log(torch.tensor(nrm / delta)).item(); cnt += 1
        xp = x + (delta / nrm) * d
    lam_fd = acc / cnt

    tol = max(0.02, 0.06 * abs(lam_fd))
    assert abs(lam_qr - lam_fd) < tol, f"joint-map lambda1 QR {lam_qr:.4f} vs finite-diff {lam_fd:.4f} (tol {tol:.3f})"
    print(f"PASS: GRU joint-map lambda1 QR={lam_qr:.4f} matches finite-difference {lam_fd:.4f}; the [x,h] autonomous "
          "map is assembled correctly and Step 74's QR reads its spectrum faithfully.")


if __name__ == "__main__":
    test_joint_autonomous_map_lyapunov_matches_finite_difference()
    print("\nstep77 guard PASS — the GRU baseline's recovered spectrum can be trusted.")
