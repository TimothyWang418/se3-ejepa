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


def test_lipschitz_bridge_inflates_lambda_and_is_sound():
    # Continuum-certified Lambda = Lambda_samples + sqrt(kappa) * L_J * h
    out = s82.lipschitz_bridge(lambda_samples=1.30, kappa=4.0, L_J=2.8, h=0.01)
    assert abs(out["lambda_cert"] - (1.30 + 2.0 * 2.8 * 0.01)) < 1e-12   # sqrt(4)=2
    assert out["lambda_cert"] > 1.30                                      # bridge only inflates (sound)


def test_lipschitz_bridge_vacuous_when_slack_dominates():
    # Huge grid spacing => Lambda_cert explodes => horizon vacuous => certified=False
    out = s82.lipschitz_bridge(lambda_samples=1.30, kappa=4.0, L_J=2.8, h=10.0,
                               eps=0.01, eps_res=1.0)
    assert out["certified"] is False


def test_run_true_henon_smoke_is_sound_and_non_vacuous():
    # n_samples=2000: the make-or-break Gate G1 (non-vacuous AND beats-Euclidean) passes here. G1 depends on the
    # covering radius h via the bridge slack sqrt(kappa)*L_J*h: an under-sampled cloud (n<=1500) has h~0.06-0.09 so the
    # inflated Lambda_cert loses to the Euclidean bound (beats_euclidean=False) -- a real covering-radius effect, NOT a
    # loosening. By n=2000 (h~0.04) the adapted metric beats Euclidean and the certificate is non-vacuous; the full
    # run() default n=4000 is denser still. This is a fast (~3s) wiring smoke of that PASS, not a re-tuned threshold.
    out = s82.run_true_henon(n_samples=2000, seed=0, eps=0.01, eps_res=1.0)
    # certificate is sound by construction; on the TRUE map at adequate resolution it must be non-vacuous (G1)
    assert out["certified"] is True
    assert out["t_guar"] >= 1
    # certified exponent brackets the textbook Henon exponent from ABOVE (sound: log Lambda_cert >= lambda_1)
    assert math.log(out["lambda_cert"]) >= 0.419 - 0.05
