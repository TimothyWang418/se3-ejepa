r"""Guard for Step 76 — the ESN reservoir baseline. Two correctness checks that the *spectrum* it reports can be
trusted (without them, a wrong analytic Jacobian or QR loop would silently produce a wrong R^2 and either fake or
falsely sink the structure-beats-scale comparison):

  (1) the **analytic autonomous Jacobian** ``auto_jacobian`` equals ``torch.autograd``'s Jacobian of ``auto_step``
      to float64 precision (the Jacobian is what the Lyapunov QR integrates);
  (2) the **thin-frame Benettin–QR leading exponent** ``esn_lyapunov_topk(...)[0]`` matches an INDEPENDENT
      finite-difference Benettin estimate that uses only ``auto_step`` (no analytic Jacobian, no QR) — cross-checking
      the whole spectrum pipeline against a method that shares none of its code.

Run:  .venv/bin/python -m pytest tests/test_step76.py -q   (or run this file directly)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step76_lorenz96_reservoir_baseline as s76  # noqa: E402

torch.manual_seed(0)
DT = torch.float64


def test_analytic_jacobian_matches_autograd() -> None:
    r"""$J(r)=(1-\alpha)I+\alpha\,\mathrm{diag}(\mathrm{sech}^2 z)W_c$ must equal the autograd Jacobian of the map."""
    for alpha in (0.6, 1.0):
        esn = s76.build_esn(N=4, Dr=16, rho=1.2, sigma_in=1.0, alpha=alpha, density=0.3, seed=1)
        Wout = (torch.rand(esn["Dr"], esn["N"], dtype=DT) - 0.5) * 0.3
        Wc = s76.autonomous_Wc(esn, Wout)
        r = torch.randn(esn["Dr"], dtype=DT) * 0.5
        J_ana = s76.auto_jacobian(esn, Wc, r)
        J_ad = torch.autograd.functional.jacobian(lambda x: s76.auto_step(esn, Wc, x), r)
        err = (J_ana - J_ad).abs().max().item()
        assert err < 1e-10, f"alpha={alpha}: analytic Jacobian disagrees with autograd by {err:.2e}"
    print("PASS (1): analytic autonomous Jacobian == autograd Jacobian to <1e-10 (float64), alpha in {0.6, 1.0}.")


def test_qr_leading_exponent_matches_finite_difference() -> None:
    r"""Independent check of the leading Lyapunov exponent: a finite-difference Benettin estimate using ONLY
    ``auto_step`` must agree with the thin-frame-QR ``esn_lyapunov_topk[0]`` (which uses the analytic Jacobian)."""
    esn = s76.build_esn(N=3, Dr=100, rho=3.0, sigma_in=1.0, alpha=1.0, density=1.0, seed=2)
    Wout = torch.zeros(esn["Dr"], esn["N"], dtype=DT)          # Wc = W => a clean autonomous chaotic reservoir
    # (high-gain tanh reservoir, Sompolinsky regime: robustly chaotic, lambda1 ~ 0.25/step)
    Wc = s76.autonomous_Wc(esn, Wout)

    # land on the attractor
    r = torch.randn(esn["Dr"], dtype=DT) * 0.1
    for _ in range(500):
        r = s76.auto_step(esn, Wc, r)

    # (A) thin-frame QR leading exponent (dt_map = 1 => per-step units)
    lam_qr = s76.esn_lyapunov_topk(esn, Wc, r, k=5, n_steps=4000, warmup=200, dt_map=1.0, seed=3)[0].item()

    # (B) finite-difference Benettin single-vector estimate, using only auto_step
    delta = 1e-7
    v = torch.randn(esn["Dr"], dtype=DT); v = v / v.norm()
    x, xp = r.clone(), r.clone() + delta * v
    acc, cnt = 0.0, 0
    for t in range(4200):
        x = s76.auto_step(esn, Wc, x)
        xp = s76.auto_step(esn, Wc, xp)
        d = xp - x
        nrm = d.norm().item()
        if t >= 200:
            acc += torch.log(torch.tensor(nrm / delta)).item()
            cnt += 1
        xp = x + (delta / nrm) * d                            # renormalize the perturbation to delta
    lam_fd = acc / cnt

    assert lam_qr > 0.05, f"test reservoir should be chaotic; got lambda1_qr={lam_qr:.3f}"
    rel = abs(lam_qr - lam_fd) / abs(lam_fd)
    assert rel < 0.06, f"QR lambda1 {lam_qr:.4f} vs finite-diff {lam_fd:.4f} disagree by {rel:.1%} (>6%)"
    print(f"PASS (2): QR lambda1={lam_qr:.4f} matches finite-difference lambda1={lam_fd:.4f} (rel {rel:.1%}); "
          "the analytic-Jacobian thin-frame QR spectrum pipeline is correct.")


if __name__ == "__main__":
    test_analytic_jacobian_matches_autograd()
    test_qr_leading_exponent_matches_finite_difference()
    print("\nstep76 guards PASS — the reservoir baseline's recovered spectrum can be trusted.")
