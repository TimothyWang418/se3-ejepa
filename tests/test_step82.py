r"""Guard for Step 82 — certified predictability horizon FROM the (learned) model.

Fast, training-free unit tests for the cone / adapted-metric certificate (Theorem B', see
``experiments/step82_certified_horizon_from_model.py`` and docs/specs/2026-06-08-certified-horizon-from-model-design.md).
Phase A covers the rigorous certificate on the TRUE Henon map (exact Jacobian-Lipschitz constant $L_J=2.8$):
the Henon Jacobian (A1), the Theorem-B' closed form ``t_guar`` (A2), the constant adapted-metric solve (A3), the
Lipschitz sample->continuum bridge (A4), and the make-or-break ``run_true_henon`` wiring + Gate G1 (A5).
"""
import math

import numpy as np
from experiments import step82_certified_horizon_from_model as s82


def test_henon_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.uniform(-1.0, 1.0, size=2)
        J = s82.henon_jac(s)                       # analytic 2x2
        fd = np.zeros((2, 2))
        eps = 1e-6
        for j in range(2):
            sp = s.copy(); sp[j] += eps
            sm = s.copy(); sm[j] -= eps
            fd[:, j] = (s82.henon_map(sp) - s82.henon_map(sm)) / (2 * eps)
        assert np.max(np.abs(J - fd)) < 1e-7


def test_henon_jacobian_lipschitz_is_exact():
    # D phi depends only on x via the (0,0) entry = -2a x ; Lip = 2a = 2.8
    assert abs(s82.HENON_JAC_LIP - 2.8) < 1e-12


def test_t_guar_matches_closed_form_and_is_monotone():
    Lam, kappa, eps_res = 1.5, 4.0, 1.0
    expected = math.floor((math.log(eps_res / 0.01) - 0.5 * math.log(kappa)) / math.log(Lam))
    assert s82.t_guar(Lam, kappa, 0.01, eps_res) == expected
    # smaller eps (sharper resolution demand at fixed eps_res) => longer horizon
    assert s82.t_guar(Lam, kappa, 1e-4, eps_res) >= s82.t_guar(Lam, kappa, 1e-2, eps_res)
    # Lambda = 1 (no expansion) => unbounded horizon sentinel
    assert s82.t_guar(1.0, 1.0, 0.01, 1.0) == s82.HORIZON_INF


def test_adapted_metric_on_symmetric_matrix_recovers_spectral_norm():
    # For a single symmetric matrix, the optimal constant metric gives Lambda = sigma_max = |max eig|.
    A = np.array([[1.3, 0.0], [0.0, 0.7]], dtype=np.float64)
    jacs = np.stack([A, A, A])
    P, Lam, _ = s82.adapted_metric(jacs)
    assert abs(Lam - 1.3) < 1e-2
    # P is SPD
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_adapted_metric_beats_euclidean_on_rotated_expansion():
    # A non-normal matrix: Euclidean op-norm overestimates; the adapted metric tightens toward rho(A)=1.2.
    A = np.array([[1.2, 5.0], [0.0, 1.2]], dtype=np.float64)
    jacs = np.stack([A, A])
    P, Lam, _ = s82.adapted_metric(jacs)
    assert Lam < np.linalg.norm(A, 2) - 1e-3        # strictly better than Euclidean
    assert Lam >= 1.2 - 1e-2                          # cannot beat the spectral radius
