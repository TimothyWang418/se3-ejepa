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
