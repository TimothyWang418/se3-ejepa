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
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# step71 supplies the Henon map, its on-attractor sampler, and the documented exponent. We REUSE its
# ``henon_step`` / ``SYSTEMS["Henon"]`` / ``on_attractor_trajs`` for the attractor point cloud; do NOT modify it.
import experiments.step71_multichaos_horizon as step71  # noqa: E402

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


# --------------------------------------------------------------------------------------------------------------- #
# Theorem-B' constant adapted metric (a global common-Lyapunov metric). We seek a single SPD $P$ and the smallest
# $\Lambda$ with $D_i^\top P D_i\preceq\Lambda^2 P$ for every sampled Jacobian $D_i$. Writing $P=P^{1/2}P^{1/2}$, this is
# $\Lambda(P)=\max_i\lVert P^{1/2}D_i P^{-1/2}\rVert_2$ — the worst metric operator norm — which we minimize over $P$.
# We parameterize $P=CC^\top$ by the log-Cholesky factor $C$ (lower-triangular, positive diagonal via $\exp$) so the
# search is unconstrained and $P\succ0$ automatically, and minimize with derivative-free Nelder-Mead (d=2 is tiny and
# the objective is non-smooth at norm ties — a gradient method would be brittle here). This is the SDP-free stand-in
# for the LMI; the constant metric contributes $L_P=0$ to the Lipschitz bridge (the cleanest possible certificate).
# --------------------------------------------------------------------------------------------------------------- #
def _metric_opnorm(L_flat, jacs, d):
    r"""Worst-case metric operator norm $\Lambda(P)=\max_i\lVert P^{1/2}D_i P^{-1/2}\rVert_2$ for $P=CC^\top$ encoded by
    the log-Cholesky vector ``L_flat`` (lower-triangular fill; diagonal exponentiated for positivity). Returns
    ``(Lambda, P)``.

    We evaluate $\Lambda(P)$ via the **symmetric-definite generalized eigenproblem**
    $\Lambda(P)^2=\max_i\lambda_{\max}\big(D_i^\top P D_i,\;P\big)$, which is *exactly* the metric operator norm
    (verified to $10^{-8}$ over random SPD $P$) and is what the LMI $D_i^\top P D_i\preceq\Lambda^2 P$ literally asserts.
    This is numerically stable even for the highly-sheared, ill-conditioned $P$ the optimum needs on a *defective*
    (non-normal) Jacobian — unlike explicitly forming $P^{-1/2}$, which can throw a spurious ``LinAlgError`` on a probe
    near $\mathrm{cond}(P)\sim10^{16}$. A probe that is not numerically PD returns a large finite penalty so the
    derivative-free optimizer simply rejects it (never crashes); soundness is unaffected — this only evaluates a
    candidate $P$, and the final certified $\Lambda$ is always the honest op-norm of the returned $P$."""
    C = np.zeros((d, d))
    idx = np.tril_indices(d)
    C[idx] = L_flat
    di = np.diag_indices(d)
    C[di] = np.exp(np.clip(np.diag(C), -10, 10))     # positive diagonal => C full-rank => P = C C^T SPD
    P = C @ C.T
    try:
        # Lambda(P)^2 = max_i lambda_max(D_i^T P D_i, P): the generalized (symmetric-definite) eigenproblem == the
        # squared operator norm in the P-metric. eigh returns ascending eigenvalues; take the largest, clamp >= 0.
        lam2 = max(float(eigh(D.T @ P @ D, P, eigvals_only=True)[-1]) for D in jacs)
    except np.linalg.LinAlgError:
        return 1e12, P                               # non-PD probe => huge finite penalty (optimizer rejects it)
    return math.sqrt(max(lam2, 0.0)), P


def adapted_metric(jacs, succ_jacs=None, mode="constant"):
    r"""Solve for a constant adapted metric $P$ minimizing $\Lambda$ s.t. $D_i^\top P D_i\preceq\Lambda^2 P$ on all
    sampled Jacobians (the Theorem-B' common-Lyapunov metric). Returns ``(P, Lambda, fun)`` where ``P`` is SPD with
    trace normalized to $d$ (cosmetic; leaves $\mathrm{cond}(P)=\kappa$ and $\Lambda$ unchanged), ``Lambda`` is the
    certified sample op-norm, and ``fun`` is the optimizer's final objective.

    ``succ_jacs``/``mode`` are reserved for a future smooth field $P(z)$ (which would also need the successor Jacobians
    $D\hat\phi(\hat\phi(z))$ and an $L_P$); the constant-$P$ path uses only ``jacs`` and has $L_P=0$.

    Shapes: ``jacs: (n, d, d) -> (P: (d, d), Lambda: float, fun: float)``.
    """
    jacs = np.asarray(jacs, dtype=DTYPE)
    d = jacs.shape[-1]
    x0 = np.zeros(d * (d + 1) // 2)                  # P = I initial (zero log-Cholesky => unit diagonal, zero off-diag)
    best = minimize(lambda v: _metric_opnorm(v, jacs, d)[0], x0,
                    method="Nelder-Mead",
                    options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
    Lam, P = _metric_opnorm(best.x, jacs, d)
    P = P / np.trace(P) * d                          # normalize trace = d (cosmetic; cond(P) and Lambda unchanged)
    return P, float(Lam), float(best.fun)


def lipschitz_bridge(lambda_samples, kappa, L_J, h, L_P=0.0, eps=0.01, eps_res=1.0):
    r"""Inflate the sampled metric op-norm to a **continuum-sound** bound, then check non-vacuity (G1, part 1).

    The LMI is verified only on the finite point cloud $\{z_i\}$; to certify ALL of the $h$-covered set we bound how much
    $\lVert P^{1/2}D\hat\phi(z)P^{-1/2}\rVert$ can grow between samples. With $L_J=\mathrm{Lip}(z\mapsto D\hat\phi)$ and a
    constant metric ($L_P=0$),
    $$\Lambda^{\text{cert}}=\Lambda_{\text{samples}}+\underbrace{\sqrt\kappa\,L_J\,h+L_P\,h}_{\text{slack}}\ \ge\
       \Lambda_{\text{samples}},$$
    so the bridge ONLY inflates $\Lambda$ — $\log\Lambda^{\text{cert}}\ge\lambda_1$ stays an upper bound (sound by
    construction; it can never make the horizon spuriously long). The certificate is **non-vacuous** iff the resulting
    $T_{\text{guar}}(\epsilon)\ge1$.

    Returns ``dict(lambda_cert, slack, horizon, certified)`` (``certified`` = ``horizon >= 1`` only — the beats-Euclidean
    half of G1 is checked by the caller, which knows the Euclidean bound)."""
    slack = math.sqrt(max(kappa, 1.0)) * L_J * h + L_P * h
    lambda_cert = lambda_samples + slack
    horizon = t_guar(lambda_cert, kappa, eps, eps_res)
    return dict(lambda_cert=float(lambda_cert), slack=float(slack),
                horizon=horizon, certified=bool(horizon >= 1))


def _covering_radius(pts):
    r"""Covering radius $h$ of the sample = $\max_i$ nearest-neighbour distance among ``pts`` (a sound covering radius
    for the point cloud: every sample lies within $h$ of another sample). ``pts: (n, d) -> float``."""
    d, _ = cKDTree(pts).query(pts, k=2)              # k=2: self (dist 0) + nearest distinct neighbour
    return float(np.max(d[:, 1]))


def run_true_henon(n_samples=4000, seed=0, eps=0.01, eps_res=1.0):
    r"""**Phase-A make-or-break:** the rigorous cone / adapted-metric certificate on the TRUE Henon map and Gate G1.

    Pipeline: sample an on-attractor point cloud (reusing :func:`step71.on_attractor_trajs`); take the EXACT Jacobian
    $D\phi(z_i)$ at each point; solve for the constant adapted metric $(P,\Lambda_{\text{samples}})$
    (:func:`adapted_metric`); read $\kappa=\mathrm{cond}(P)$ and the covering radius $h$ (:func:`_covering_radius`);
    inflate to the continuum-sound $\Lambda^{\text{cert}}=\Lambda_{\text{samples}}+\sqrt\kappa\,L_J\,h$ with the EXACT
    $L_J=$ :data:`HENON_JAC_LIP` (:func:`lipschitz_bridge`); and compute $T_{\text{guar}}(\epsilon)$.

    **Gate G1 (make-or-break, never loosened):** ``certified`` is True iff the certificate is BOTH non-vacuous
    ($T_{\text{guar}}\ge1$) AND beats the trivial Euclidean bound $\max_i\lVert D_i\rVert_2+L_J h$ (a constant *identity*
    metric, $\kappa=1$). The certificate is sound by construction (the bridge only inflates $\Lambda$), so
    $\log\Lambda^{\text{cert}}\ge\lambda_1\approx0.419$ always holds. If a single constant $P$ cannot tame the
    widely-varying Henon Jacobian and the result is vacuous, that is a REAL finding — reported, not loosened.

    Returns a dict with ``lambda_samples``, ``lambda_cert``, ``kappa``, ``h``, ``slack``, ``t_guar`` (the certified
    horizon), ``euclid_bound``, ``euclid_t_guar`` (the Euclidean baseline horizon), ``beats_euclidean``, ``certified``
    (the G1 verdict), and the inputs ``eps``/``eps_res``."""
    rng = np.random.default_rng(seed)
    cfg = step71.SYSTEMS["Henon"]
    trajs = step71.on_attractor_trajs(cfg, rng, n=max(8, n_samples // 200), length=300)
    pts = np.concatenate([t[:-1] for t in trajs], axis=0)[:n_samples].astype(DTYPE)
    jacs = np.stack([henon_jac(s) for s in pts])

    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts)
    br = lipschitz_bridge(lam_samples, kappa, HENON_JAC_LIP, h, eps=eps, eps_res=eps_res)

    # Trivial Euclidean baseline: the constant IDENTITY metric (kappa = 1), whose continuum bound is
    # max_i ||D_i||_2 + L_J h (no cone to absorb the rotation of the expanding direction). Its certified horizon is the
    # bar the adapted metric must beat (the second half of G1).
    euclid_bound = float(max(np.linalg.norm(D, 2) for D in jacs)) + HENON_JAC_LIP * h
    euclid_t_guar = t_guar(euclid_bound, 1.0, eps, eps_res)
    beats_euclidean = bool(br["lambda_cert"] < euclid_bound)

    return dict(system="Henon(true)", lambda_samples=lam_samples, lambda_cert=br["lambda_cert"],
                kappa=kappa, h=h, slack=br["slack"], t_guar=br["horizon"],
                euclid_bound=euclid_bound, euclid_t_guar=euclid_t_guar, beats_euclidean=beats_euclidean,
                certified=bool(br["certified"] and beats_euclidean), eps=eps, eps_res=eps_res)
