r"""Guard for Step 78 — the statistical certified horizon. Two checks that the error bars and the abstention rule are
sound (without them the "confidence interval" could be miscalibrated, faking either certification or abstention):

  (1) the horizon interval $T(\epsilon)=\log(1/\epsilon)/\lambda$ is *decreasing* in $\lambda$ and the certificate
      *abstains* (returns None) exactly when the $\lambda$ CI straddles or sits at/below $0$ (no finite chaotic horizon);
  (2) the block-bootstrap spectrum CI is *calibrated* on the TRUE Lorenz-96 map: $\lambda_1$'s CI is strictly positive
      (chaotic), the CI brackets the point estimate, and the estimator honors the Liouville anchor $\sum\lambda=-N$.

Run:  .venv/bin/python -m pytest tests/test_step78.py -q   (or run this file directly)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step74_lorenz96_spectrum as s74  # noqa: E402
import step78_certified_horizon_ci as s78  # noqa: E402


def test_horizon_interval_decreasing_and_abstains() -> None:
    L = np.log(1.0 / s78.EPS)
    iv = s78.horizon_interval(0.5, 1.0)                     # positive, sign-stable lambda CI
    assert iv is not None
    assert abs(iv[0] - L / 1.0) < 1e-9 and abs(iv[1] - L / 0.5) < 1e-9   # T decreasing in lambda
    assert iv[0] < iv[1]
    assert s78.horizon_interval(-0.1, 0.3) is None          # straddles 0 -> abstain
    assert s78.horizon_interval(-1.0, -0.2) is None         # all <=0 -> no finite chaotic horizon -> abstain
    print("PASS (1): horizon interval is decreasing in lambda and abstains iff the lambda CI is not sign-stable-positive.")


def test_bootstrap_ci_calibrated_on_true_map() -> None:
    N = 12
    traj = s74.attractor_traj(N, 400, 0, "cpu").double()
    mu, sd = traj.mean(0), traj.std(0) + 1e-8
    x0 = (traj[len(traj) // 2] - mu) / sd
    logR = s78.qr_logR_series(lambda xn: (s74.true_map(xn * sd + mu) - mu) / sd, x0, n_steps=1200, warmup=200)
    point, lo, hi = s78.bootstrap_spectrum_ci(logR, s74.DTMAP, n_boot=400, block=40, seed=0)

    assert lo[0] > 0, f"lambda1 CI must be strictly positive (chaotic); got [{lo[0]:.3f},{hi[0]:.3f}]"
    assert bool((lo <= point + 1e-9).all() and (point <= hi + 1e-9).all()), "CI must bracket the point estimate"
    sum_point = float(point.sum())
    assert abs(sum_point - (-N)) / N < 0.05, f"sum(lambda)={sum_point:.2f} must honor Liouville -N={-N} (<5%)"
    print(f"PASS (2): bootstrap CI calibrated — lambda1 CI [{lo[0]:.3f},{hi[0]:.3f}]>0, brackets the point estimate, "
          f"and sum(lambda)={sum_point:.2f} honors Liouville -N={-N}.")


if __name__ == "__main__":
    test_horizon_interval_decreasing_and_abstains()
    test_bootstrap_ci_calibrated_on_true_map()
    print("\nstep78 guards PASS — the statistical certificate's error bars and abstention rule are sound.")
