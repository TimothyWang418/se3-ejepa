r"""Guard for Step 65 / Proposition 6: the $\epsilon$-approximately-equivariant rollout's orbit-error-variation equals
$\epsilon\,e^{\lambda_j T}$ exactly (the matching lower bound that makes the certified horizon tight), and the exactly
equivariant model is orbit-flat to the float floor.

This pins the math behind the paper's central horizon claim: structure reaches an unbounded horizon, approximate
structure does not. Run:  .venv/bin/python tests/test_step65.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from step65_horizon_tightness import LAMBDAS, T_MAX, orbit_variation_per_channel  # noqa: E402


def test_proposition6_lower_bound_is_exact() -> None:
    r"""$|\Delta\mathrm{Err}_T| = \epsilon\,e^{\lambda_j T}$ per channel (tightness); $\epsilon{=}0$ is flat."""
    eps = 1e-3
    var = orbit_variation_per_channel(eps, np.random.default_rng(0))     # (T_MAX+1, d)
    Ts = np.arange(T_MAX + 1)
    analytic = eps * (np.exp(LAMBDAS)[None, :] ** Ts[:, None])
    rel = np.abs(var - analytic) / np.maximum(analytic, 1e-300)
    assert rel.max() < 1e-6, f"orbit-variation deviates from eps*e^(lambda T): max rel-err {rel.max():.2e}"

    # exact-equivariant model (eps=0) is orbit-flat at every horizon
    var0 = orbit_variation_per_channel(0.0, np.random.default_rng(0))
    assert var0.max() < 1e-9, f"exact-equivariant model not orbit-flat: {var0.max():.2e}"

    # expansive channels blow up; conserved/contractive channels stay bounded -> the central claim
    exp_j = int(np.argmax(LAMBDAS))                       # most expansive
    cons_j = int(np.argmin(LAMBDAS))                      # most contractive (lambda<0)
    assert var[-1, exp_j] > var[0, exp_j] * 10, "expansive channel did not grow with horizon"
    assert var[-1, cons_j] <= var[0, cons_j] + 1e-12, "conserved/contractive channel should not grow"

    print(f"PASS: |dErr_T| = eps*e^(lambda T) to {rel.max():.1e}; exact-equiv flat to {var0.max():.1e}; "
          f"expansive grows x{var[-1, exp_j]/max(var[0, exp_j],1e-18):.0f}, conserved bounded.")
    print("=> Proposition 6: approximate equivariance is horizon-limited; only exact structure reaches infinity.")


if __name__ == "__main__":
    test_proposition6_lower_bound_is_exact()
