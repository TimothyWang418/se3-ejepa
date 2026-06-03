r"""A runnable predictability certificate (Algorithm 1 of the paper).

This turns the theory — Lemma 1 (composition closure), Theorem A (exact configuration certificate), Lemma 2
(necessity), and Theorem B (spectral horizon $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$) — into a **procedure**:
given a trained equivariant world model, *measure* its two certificate ingredients and *emit* the certified region
across the three axes (configuration $\langle S\rangle$, horizon $T$, resolution $\epsilon$).

The certificate is not just a lens: ``certify(...)`` is something you run. Its inputs are exactly the two
quantities every experiment already measures —

  1. the per-generator encoder/predictor equivariance residual $\epsilon_{\max}=\max_i\lVert E(g_i x)-\rho(g_i)E(x)\rVert$
     (the configuration-axis slack — by Lemma 1, $k$ generator checks certify the whole monoid $\langle S\rangle$);
  2. the predictor-Jacobian singular values $\{\sigma_j\}$ at the operating point (the per-channel multipliers
     $e^{\lambda_j}$ of Theorem B).

and its output is the certified region: whether the configuration axis is certified (residual $\le$ tolerance),
and the per-channel horizon $T_j(\epsilon)$ at the requested resolution $\epsilon$ ($\infty$ for contractive
$\lambda_j\le0$ channels, $\log(1/\epsilon)/\lambda_j$ otherwise).

Demo (wires Step 47's *measured* Jacobian spectrum into the procedure):
    .venv/bin/python -m src.certify
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Certificate:
    r"""The certified region emitted by :func:`certify`, across the three axes."""

    eps: float                              # requested resolution
    eps_max: float                          # max per-generator equivariance residual (configuration slack)
    residual_tol: float                     # tolerance below which the configuration axis is certified
    config_certified: bool                  # eps_max <= residual_tol  =>  whole monoid certified (Lemma 1)
    n_generators: int                       # k: checks that certify the exponential monoid
    horizons: list[float]                   # per-channel certified horizon T_j(eps); inf if contractive
    n_unbounded: int                        # channels certified to *all* horizons (lambda_j <= 0)
    min_finite_horizon: float | None        # the binding (shortest finite) channel horizon, or None

    def summary(self) -> str:
        cfg = ("CONFIGURATION: certified over the entire generated monoid ⟨S⟩ "
               f"from {self.n_generators} generator checks (residual {self.eps_max:.1e} ≤ {self.residual_tol:.0e})"
               ) if self.config_certified else (
               f"CONFIGURATION: NOT certified — generator residual {self.eps_max:.1e} > tol {self.residual_tol:.0e} "
               "(in Theorem B's approximate regime)")
        if self.min_finite_horizon is None:
            hor = (f"HORIZON×RESOLUTION: all {self.n_unbounded} channels contractive (λ≤0) ⇒ certified to every "
                   f"horizon at ε={self.eps:g}")
        elif self.min_finite_horizon < 1.0:
            hor = (f"HORIZON×RESOLUTION at ε={self.eps:g}: {self.n_unbounded} channels unbounded (λ≤0); the binding "
                   f"expansive channel is certified to <1 step (T={self.min_finite_horizon:.2f}; strongly expansive "
                   "— its fine detail is uncertifiable at this resolution)")
        else:
            hor = (f"HORIZON×RESOLUTION at ε={self.eps:g}: {self.n_unbounded} channels unbounded (λ≤0); the binding "
                   f"expansive channel is certified to T={self.min_finite_horizon:.1f} steps")
        return cfg + "\n" + hor


def certify(
    equivariance_residuals: dict[str, float],
    jacobian_singular_values: list[float] | tuple[float, ...],
    eps: float,
    *,
    residual_tol: float = 1e-3,
    dt: float = 1.0,
) -> Certificate:
    r"""Emit the predictability certificate for a trained equivariant world model.

    Parameters
    ----------
    equivariance_residuals
        Per-generator residual ``{name: ||E(g·x) − ρ(g)E(x)||}`` (or the joint encoder+predictor residual). By
        Lemma 1 these $k$ checks certify the whole monoid $\langle S\rangle$; the max is Theorem B's
        $\epsilon_{\max}$.
    jacobian_singular_values
        Singular values $\{\sigma_j\}$ of the predictor Jacobian at the operating point. Channel $j$ has Lyapunov
        exponent $\lambda_j=\log\sigma_j/\Delta t$.
    eps
        Requested resolution (error tolerance) for the horizon axis.
    residual_tol
        Configuration-axis tolerance; ``eps_max ≤ residual_tol`` certifies the exact (Theorem A) regime.
    dt
        Time step relating a one-step multiplier $\sigma_j$ to a rate $\lambda_j=\log\sigma_j/\Delta t$.

    Returns
    -------
    Certificate
        The certified region across configuration × horizon × resolution.
    """
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must lie in (0, 1)")
    eps_max = max(equivariance_residuals.values()) if equivariance_residuals else 0.0
    config_certified = eps_max <= residual_tol

    horizons: list[float] = []
    n_unbounded = 0
    for sigma in jacobian_singular_values:
        lam = math.log(max(float(sigma), 1e-12)) / dt          # Lyapunov exponent of this channel
        if lam <= 0.0:
            horizons.append(math.inf)                          # contractive/conserved ⇒ certified to all horizons
            n_unbounded += 1
        else:
            horizons.append(math.log(1.0 / eps) / lam)         # Theorem B: T_j(ε) ~ log(1/ε)/λ_j
    finite = [h for h in horizons if math.isfinite(h)]
    return Certificate(
        eps=eps,
        eps_max=eps_max,
        residual_tol=residual_tol,
        config_certified=config_certified,
        n_generators=len(equivariance_residuals),
        horizons=horizons,
        n_unbounded=n_unbounded,
        min_finite_horizon=min(finite) if finite else None,
    )


def _demo() -> None:
    r"""Run the certificate on Step 47's *measured* Jacobian spectrum, if available."""
    figpath = Path(__file__).resolve().parent.parent / "papers" / "figures" / "step47_certificate.json"
    if figpath.exists():
        data = json.loads(figpath.read_text())
        svs = data.get("jacobian_spectrum")
        if svs:
            # Step 47 verifies Theorem A architecturally to ~1.2e-7 on its generators (SE(3) rotation + S_n perm).
            residuals = {"SE(3) rotation": 1.2e-7, "S_n permutation": 1.2e-7}
            for eps in (0.1, 0.01):
                cert = certify(residuals, svs, eps)
                print(f"\n=== certify(Step-47 model, ε={eps}) ===\n{cert.summary()}")
            return
    # fallback synthetic spectrum (1 conserved, 1 contracting, 1 chaotic)
    print("(step47_certificate.json not found; synthetic demo)")
    print(certify({"g1": 5e-7}, [1.0, 0.75, 2.0], 0.01).summary())


if __name__ == "__main__":
    _demo()
