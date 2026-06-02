r"""Horizon × resolution unit test (Step 52, paper2 §2 / P2 — Theorem B).

paper2's third axis: a learned predictor's per-channel certified horizon obeys
$T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$. We guard the two falsifiable cores at smoke scale:

1. **The learned model recovers the chaotic Lyapunov exponent.** The angle-doubling channel $z\mapsto z^2$ has
   $\lambda=\ln2\approx0.69$; the rollout-error growth rate the model produces should land near it.
2. **Slow ⇒ long horizon; chaotic ⇒ short.** At a fixed resolution $\epsilon$ the chaotic "fine detail" is
   certifiable for far fewer steps than every slow/invariant channel — the predictability staircase / 推背图 law.

[Smoke-scale; the full 3-seed result is `experiments/step52_horizon_resolution.py`.]

Run:  .venv/bin/python tests/test_step52_horizon_resolution.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step52_horizon_resolution import (  # noqa: E402
    CH, Predictor, certified_horizon, fit_lyapunov, make_pairs, rollout_errors, train,
)


def _trained_rollout(seed=0, horizon=50):
    torch.set_default_dtype(torch.float64)               # step52's math is float64 (robust to test import order)
    torch.manual_seed(seed)
    S, S2 = make_pairs(3000, seed=seed)
    model = Predictor()
    train(model, S, S2, epochs=20, seed=seed)
    s0 = make_pairs(1500, seed=777)[0]
    return rollout_errors(model, s0, horizon)


def test_model_recovers_chaotic_lyapunov_exponent() -> None:
    errs = _trained_rollout()
    lam = fit_lyapunov(errs["chaotic"])
    print(f"chaotic Lyapunov exponent: measured {lam:+.3f} vs true ln2 = {math.log(2.0):+.3f}")
    assert 0.45 < lam < 0.95, f"learned model should recover λ≈ln2≈0.69 for the doubling channel, got {lam:+.3f}"


def test_slow_channels_certify_far_longer_than_chaotic() -> None:
    errs = _trained_rollout()
    eps = 0.05
    T = {g: certified_horizon(errs[g], eps) for g in CH}
    slow_min = min(T["conserved"], T["contracting"], T["rotor (neutral)"])
    print(f"certified horizons @ε={eps}: chaotic={T['chaotic']} | slowest-slow={slow_min} "
          f"(conserved={T['conserved']}, contracting={T['contracting']}, rotor={T['rotor (neutral)']})")
    assert T["chaotic"] < slow_min, "the chaotic 'detail' must die first (shortest certified horizon)"
    assert slow_min >= 3 * max(T["chaotic"], 1), (
        f"slow/invariant channels should certify ≥3× longer than chaotic: {slow_min} vs {T['chaotic']}"
    )


def main() -> None:
    print("Step 52 — horizon × resolution (Theorem B): T_j(ε) ~ log(1/ε)/λ_j\n")
    test_model_recovers_chaotic_lyapunov_exponent()
    print()
    test_slow_channels_certify_far_longer_than_chaotic()
    print("\nPASS: the learned model recovers the chaotic Lyapunov exponent and the certified-horizon ordering")
    print("      slow/invariant ≫ chaotic holds — the predictability staircase, Theorem B.")


if __name__ == "__main__":
    main()
