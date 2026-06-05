r"""Guard for Step 69 / certificate-driven active inference (③-2): on the $\mathbb Z_2^k$ configuration axis, the
certificate-driven explorer certifies the *entire* group in exactly $k$ observations (the generator basis), while an
error-curiosity explorer lured by noisy distractors does not. Pins the math behind "expand your certified region" as
a noise-immune epistemic drive.

Run:  .venv/bin/python tests/test_step69.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import step69_certificate_driven_exploration as s69  # noqa: E402


def test_certificate_driven_certifies_basis_in_k_steps() -> None:
    r"""Certificate-driven certifies all $2^k$ in $k$ steps; error-curiosity (chasing noisy distractors) lags badly."""
    K = s69.K
    rng = np.random.default_rng(0)
    errors_gen = rng.uniform(0.3, 0.6, size=K)
    errors_dist = rng.uniform(0.8, 1.2, size=s69.N_DISTRACT)         # distractors look most surprising to curiosity

    cert = s69.run_agent("certificate", errors_gen, errors_dist, 2 * K, np.random.default_rng(1))
    err = s69.run_agent("error", errors_gen, errors_dist, 2 * K, np.random.default_rng(2))

    # certificate-driven reaches full certification at exactly budget = k (the independent generator basis)
    assert abs(cert[K - 1] - 1.0) < 1e-12, f"certificate-driven did not certify all 2^{K} in k steps: {cert[K-1]}"
    assert cert[K - 2] < 1.0, "certificate-driven certified everything BEFORE k steps (impossible for a basis)"
    # subgroup_size doubles per independent generator
    assert s69.subgroup_size(np.array([True, True, False] + [False] * (K - 3))) == 4

    # error-curiosity, lured by the high-error distractors, is far behind at the same budget
    assert err[K - 1] < 0.5, f"error-curiosity unexpectedly kept pace ({err[K-1]:.2f}); the distractor lure failed"
    assert cert[K - 1] - err[K - 1] > 0.4, "certificate-driven did not dominate error-curiosity at budget k"

    print(f"PASS: certificate-driven certifies all 2^{K}={1<<K} in exactly k={K} steps (frac {cert[K-1]:.2f}); "
          f"error-curiosity only {err[K-1]:.2f} at the same budget (lured by noisy distractors).")
    print("=> 'expand your certified region' is a noise-immune epistemic drive (certificate-driven active inference).")


if __name__ == "__main__":
    test_certificate_driven_certifies_basis_in_k_steps()
