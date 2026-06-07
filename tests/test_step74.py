r"""Guard for Step 74 — the Lyapunov-spectrum estimator is correct on the TRUE Lorenz-96 system, anchored by a
rigorous physical invariant. Lorenz-96's vector-field divergence is $\nabla\!\cdot f=-N$ (the quadratic terms have
zero diagonal; only the $-x_i$ damping contributes), so by Liouville the whole spectrum sums to $-N$ exactly:
$\sum_{j=1}^N\lambda_j=-N$. We verify the Benettin-QR estimator reproduces this to a few percent, and that the system
is genuinely chaotic ($\lambda_1>0$) with the expected count of positive exponents. This validates the estimator before
any *learned*-model spectrum is trusted (the step74 experiment).

Run:  .venv/bin/python tests/test_step74.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

import step74_lorenz96_spectrum as s74  # noqa: E402


def test_true_spectrum_obeys_liouville_and_is_chaotic() -> None:
    torch.manual_seed(0)
    N = 12
    traj = s74.attractor_traj(N, 200, seed=0, device="cpu", burn=1500)
    x0 = traj[-1]
    lam = s74.lyapunov_spectrum(s74.true_map, x0, n_steps=1500, warmup=300)

    total = float(lam.sum())
    # (i) Liouville: sum of exponents equals the divergence -N, exactly in the continuum; QR recovers it to a few %.
    assert abs(total - (-N)) / N < 0.05, f"sum(lambda)={total:.3f} should be ~ -N={-N} (Liouville); rel-err too large"
    # (ii) genuinely chaotic: a positive leading exponent.
    assert float(lam[0]) > 0.3, f"lambda_1={float(lam[0]):.3f} should be clearly positive (chaotic Lorenz-96, F=8)"
    # (iii) a high-dimensional chaotic spectrum: several positive exponents (not just one).
    npos = int((lam > 0).sum())
    assert npos >= 2, f"expected several positive exponents for N={N} chaotic Lorenz-96, got {npos}"
    # (iv) ordered descending.
    assert bool((lam[:-1] >= lam[1:] - 1e-9).all()), "spectrum must be sorted descending"
    print(f"PASS: Lorenz-96 (N={N}) spectrum — sum={total:.2f} vs Liouville -N={-N} (rel-err "
          f"{abs(total+N)/N:.1%}); lambda_1={float(lam[0]):.3f}; {npos} positive exponents.")
    print("=> the Benettin-QR estimator is correct (anchored by an exact physical invariant); the learned-model "
          "spectrum it returns in step74 can be trusted.")


if __name__ == "__main__":
    test_true_spectrum_obeys_liouville_and_is_chaotic()
