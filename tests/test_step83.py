r"""Guard for Step 83 — the $R^2(N)$ crossover sweep. NO training happens here (the sweep itself is the long compute
job); these are fast unit checks on the three load-bearing pieces the crossover rests on:

  (1) the full-spectrum $R^2$ metric (``s83._r2``) equals a *hand-computed* value on a toy spectrum, and obeys its
      definitional anchors ($R^2=1$ for a perfect match; $R^2=0$ for the mean predictor);
  (2) the **true** Lorenz-96 spectrum obeys Liouville $\sum_j\lambda_j=-N$ at a small $N$ (the physical invariant that
      anchors every R^2 in the sweep — the score is only meaningful if the *reference* spectrum is correct);
  (3) the three architectures (conv / MLP / GRU) **instantiate at each swept $N$** with the expected parameter
      scaling: the $\mathbb{Z}_N$-conv's parameter count is **independent of $N$** (weight sharing across the cyclic
      group — the whole point of equivariance), while the dense MLP's input/output layers scale like $N$.

Run:  .venv/bin/python -m pytest tests/test_step83.py -q   (or run this file directly)
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import step74_lorenz96_spectrum as s74  # noqa: E402
import step77_lorenz96_gru_baseline as s77  # noqa: E402
import step83_rsquared_crossover as s83  # noqa: E402


def test_r2_metric_matches_hand_computation() -> None:
    r"""$R^2 = 1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$ with $\mathrm{SS_{res}}=\lVert\hat\lambda-\lambda\rVert^2$ and
    $\mathrm{SS_{tot}}=\lVert\lambda-\bar\lambda\rVert^2$. Hand-check on a 4-point toy spectrum."""
    lt = np.array([2.0, 1.0, -1.0, -2.0])           # mean 0, SS_tot = 4+1+1+4 = 10
    ll = np.array([1.5, 1.0, -1.0, -1.5])           # residuals .5,0,0,.5  -> SS_res = .25+0+0+.25 = .5
    expected = 1.0 - 0.5 / 10.0                      # = 0.95
    got = s83._r2(ll, lt)
    assert abs(got - expected) < 1e-12, f"R^2 metric {got} != hand value {expected}"

    # definitional anchors
    assert abs(s83._r2(lt.copy(), lt) - 1.0) < 1e-12, "perfect recovery must give R^2 = 1"
    mean_pred = np.full_like(lt, lt.mean())
    assert abs(s83._r2(mean_pred, lt) - 0.0) < 1e-12, "mean predictor must give R^2 = 0"
    # a spectrum worse than the mean predictor gives R^2 < 0 (the regime the MLP/GRU fall into at high N)
    assert s83._r2(np.array([-5.0, -5.0, 5.0, 5.0]), lt) < 0.0, "anti-correlated guess must give R^2 < 0"
    print(f"PASS: R^2 metric — toy spectrum gives {got:.4f} (hand {expected:.4f}); anchors 1 / 0 / <0 all hold.")


def test_true_lorenz96_spectrum_obeys_liouville_small_N() -> None:
    r"""The reference spectrum every R^2 is scored against must be correct: Lorenz-96's divergence is $-N$, so by
    Liouville $\sum_j\lambda_j=-N$ exactly (continuum); Step 74's QR recovers it to a few %. Checked at small N=8
    (cheap, no training)."""
    torch.manual_seed(0)
    N = 8
    traj = s74.attractor_traj(N, 200, seed=0, device="cpu", burn=1500)
    lam = s74.lyapunov_spectrum(s74.true_map, traj[-1], n_steps=1500, warmup=300)
    total = float(lam.sum())
    assert abs(total - (-N)) / N < 0.05, f"sum(lambda)={total:.3f} should be ~ -N={-N} (Liouville); rel-err too large"
    assert float(lam[0]) > 0.2, f"lambda_1={float(lam[0]):.3f} should be positive (chaotic Lorenz-96, F=8)"
    assert bool((lam[:-1] >= lam[1:] - 1e-9).all()), "spectrum must be sorted descending"
    print(f"PASS: true Lorenz-96 (N={N}) spectrum sum={total:.3f} vs Liouville -N={-N} "
          f"(rel-err {abs(total + N) / N:.1%}); lambda_1={float(lam[0]):.3f}.")


def test_architectures_instantiate_at_each_N_with_right_scaling() -> None:
    r"""All three architectures must build at every swept $N$ and map $(\cdot,N)\!\to\!(\cdot,N)$. The
    $\mathbb{Z}_N$-conv shares weights across the cyclic group, so its parameter count is **independent of $N$**; the
    dense MLP's first/last layers scale linearly in $N$. No training — just shape/param-count checks."""
    import os

    Ns = [12, 20, 28, 40]
    conv_pcounts, mlp_pcounts = [], []
    for N in Ns:
        # conv (Z_N-equivariant) — circular convs, weights shared across the group
        os.environ["STEP74_MODEL"] = "conv"
        conv, kind_c = s74._build(N, "cpu")
        assert "Z_N" in kind_c or "equivariant" in kind_c
        # dense MLP baseline
        os.environ["STEP74_MODEL"] = "mlp"
        mlp, kind_m = s74._build(N, "cpu")
        assert "MLP" in kind_m
        # GRU (Step 77)
        gru = s77.L96GRU(N, hidden=32).double()

        x = torch.randn(4, N, dtype=torch.float64)
        assert conv(x).shape == (4, N), f"conv output shape wrong at N={N}"
        assert mlp(x).shape == (4, N), f"mlp output shape wrong at N={N}"
        h = torch.zeros(4, 32, dtype=torch.float64)
        xo, ho = gru(x, h)
        assert xo.shape == (4, N) and ho.shape == (4, 32), f"gru output shapes wrong at N={N}"

        conv_pcounts.append(sum(p.numel() for p in conv.parameters()))
        mlp_pcounts.append(sum(p.numel() for p in mlp.parameters()))

    # (i) conv param count is INDEPENDENT of N (equivariance = weight sharing across the cyclic group).
    assert len(set(conv_pcounts)) == 1, f"Z_N-conv param count must not depend on N, got {dict(zip(Ns, conv_pcounts))}"
    # (ii) dense MLP param count scales (strictly increases) with N — its Jacobian is the unconstrained N x N block.
    assert all(mlp_pcounts[i] < mlp_pcounts[i + 1] for i in range(len(mlp_pcounts) - 1)), \
        f"dense-MLP param count must grow with N, got {dict(zip(Ns, mlp_pcounts))}"
    # sanity: each extra dim adds `hidden` weights to the input layer and `hidden`+1 (weight+bias) to the output
    # layer of L96MLP (in: Linear(N,hidden); out: Linear(hidden,N)), i.e. exactly (2*hidden + 1) params per dim.
    hidden = 128 if s74.SMOKE else 256
    gap = mlp_pcounts[1] - mlp_pcounts[0]
    per_dim = 2 * hidden + 1
    assert gap == (Ns[1] - Ns[0]) * per_dim, f"MLP N-scaling off: gap {gap} != {(Ns[1]-Ns[0])*per_dim}"
    print(f"PASS: archs instantiate at N={Ns}; Z_N-conv params N-independent ({conv_pcounts[0]}); "
          f"dense-MLP params scale with N ({dict(zip(Ns, mlp_pcounts))}).")


if __name__ == "__main__":
    test_r2_metric_matches_hand_computation()
    test_true_lorenz96_spectrum_obeys_liouville_small_N()
    test_architectures_instantiate_at_each_N_with_right_scaling()
    print("\nstep83 guard PASS — R^2 metric, true-spectrum Liouville anchor, and per-N architecture scaling all hold.")
