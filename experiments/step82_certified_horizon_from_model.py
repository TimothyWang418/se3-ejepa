r"""Step 82 — a CERTIFIED predictability horizon read FROM the (learned) model, not measured.

Design: docs/specs/2026-06-08-certified-horizon-from-model-design.md; plan:
docs/plans/2026-06-08-step82-certified-horizon-from-model.md. This file is **Phase A**: a rigorous,
sound-by-construction cone / adapted-metric certificate, validated on the TRUE Henon map (exact Jacobian, so the
Jacobian-field Lipschitz constant is *exactly* $L_J=2a=2.8$ — no neural-net slop).

**Theorem B' (cone / adapted-metric certified horizon).** Let $\hat\phi:\mathcal U\to\mathcal U$ be $C^1$ on a compact
forward-invariant $\mathcal U\subset\mathbb R^d$. If a constant SPD metric $P\succ0$ and a constant $\Lambda\ge1$ satisfy
$$ D\hat\phi(z)^\top P\,D\hat\phi(z)\ \preceq\ \Lambda^2 P\qquad\forall z\in\mathcal U, \tag{LMI}$$
then with $V(z,v)=v^\top P v$ we get $V(\hat\phi(z),D\hat\phi(z)v)\le\Lambda^2 V(z,v)$; iterating and converting back to the
Euclidean norm with $\kappa=\mathrm{cond}(P)$ gives $\lVert D\hat\phi^T(z)v\rVert^2\le\kappa\Lambda^{2T}\lVert v\rVert^2$,
hence $\lambda_1(\hat\phi)\le\log\Lambda$ and a horizon computed from $\hat\phi$ ALONE:
$$ T_{\text{guar}}(\epsilon)=\Big\lfloor\tfrac{\log(\epsilon_{\text{res}}/\epsilon)-\tfrac12\log\kappa}{\log\Lambda}\Big\rfloor. $$
The optimal $P$ drives $\Lambda\to e^{\lambda_1}$ — the cone absorbs the rotation of the expanding direction that makes the
naive $\sup_z\lVert D\hat\phi(z)\rVert$ bound useless, so the certificate is *tight, not merely sound*.

**Sample -> continuum bridge (what makes it a certificate, not "checked on a grid").** We verify (LMI) on a finite
attractor point cloud $\{z_i\}$ (covering radius $h$) and inflate the sampled metric op-norm to a continuum-sound bound:
$\Lambda^{\text{cert}}=\Lambda_{\text{samples}}+\sqrt\kappa\,L_J\,h$ (a constant $P$ contributes $L_P=0$). The bridge ONLY
inflates $\Lambda$, so $\log\Lambda^{\text{cert}}\ge\lambda_1$ stays an UPPER bound — sound by construction.

**Honest scope (load-bearing, not hidden).** The certificate is a-priori and deterministic about the *model's*
error-amplification horizon. The link to the TRUE system's horizon ($T_{\text{guar}}\le T_{\text{true}}$) is the
misspecification gap, which is provably **un-certifiable a-priori**; on the true Henon map here there is no
misspecification, so the certificate is a fully rigorous bound on the map's own exponent. Gate **G1** (make-or-break) is
that the bridged $\Lambda^{\text{cert}}$ is **non-vacuous** ($T_{\text{guar}}\ge1$) AND **beats the trivial Euclidean
bound** at the target $(\epsilon,\epsilon_{\text{res}})$. If a single constant $P$ cannot tame the widely-varying Henon
Jacobian and the certificate comes back vacuous, that is a real, reportable finding — we print INCONCLUSIVE and never
loosen anything to force a pass (a smooth field $P(z)$ would be a separate decision).

Reuse, do NOT modify: ``experiments/step71_multichaos_horizon.py`` (``henon_step``, ``SYSTEMS["Henon"]``,
``on_attractor_trajs`` for the attractor point cloud).

Run (Phase A make-or-break):
    .venv/bin/python -c "import experiments.step82_certified_horizon_from_model as s; import json; \
        print(json.dumps(s.run_true_henon(), indent=2))"
"""
import math

import numpy as np

DTYPE = np.float64
HENON_A, HENON_B = 1.4, 0.3
# D phi(s) = [[-2a x, 1], [b, 0]] depends only on x; the only varying entry is -2a x with slope 2a in x, and the
# Jacobian's spectral norm is 1-Lipschitz in that entry, so Lip(z -> D phi) = 2a = 2.8 EXACTLY.
HENON_JAC_LIP = 2.0 * HENON_A          # = 2.8

# Sentinel for a non-expanding certified map (log Lambda <= 0): the certified horizon is unbounded. A large finite
# integer keeps t_guar's return type a plain int (vs math.inf), so callers can compare/min/round without special-casing.
HORIZON_INF = 10**9


def t_guar(Lambda, kappa, eps, eps_res):
    r"""Theorem B' certified horizon (in MAP STEPS) from the verified constants alone:
    $T_{\text{guar}}=\lfloor(\log(\epsilon_{\text{res}}/\epsilon)-\tfrac12\log\kappa)/\log\Lambda\rfloor$.

    A perturbation of size $\epsilon$ is amplified by at most $\sqrt\kappa\,\Lambda^T$ (Theorem B'), so it first may reach
    the resolution $\epsilon_{\text{res}}$ at the floor above. If $\Lambda\le1$ the metric certifies NO net expansion, so
    the horizon is unbounded -> :data:`HORIZON_INF`. Returns a non-negative ``int`` (clamped at 0)."""
    if Lambda <= 1.0 + 1e-15:
        return HORIZON_INF
    val = (math.log(eps_res / eps) - 0.5 * math.log(max(kappa, 1.0))) / math.log(Lambda)
    return max(0, int(math.floor(val)))


def henon_map(s):
    r"""True Henon map $\phi(x,y)=(1-a x^2+y,\;b x)$ with $(a,b)=(1.4,0.3)$. ``s: (..., 2) -> (..., 2)``, float64."""
    s = np.asarray(s, dtype=DTYPE)
    x, y = s[..., 0], s[..., 1]
    return np.stack([1.0 - HENON_A * x * x + y, HENON_B * x], axis=-1)


def henon_jac(s):
    r"""Exact Jacobian $D\phi(s)=\begin{pmatrix}-2a x & 1\\ b & 0\end{pmatrix}$ (depends only on $x$). Returns a
    ``(2, 2)`` float64 array."""
    s = np.asarray(s, dtype=DTYPE)
    x = s[..., 0]
    return np.array([[-2.0 * HENON_A * x, 1.0], [HENON_B, 0.0]], dtype=DTYPE)
