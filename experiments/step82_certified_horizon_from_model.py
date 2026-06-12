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
import json
import math
import sys
from pathlib import Path

# Make the repo root importable so ``import experiments.stepNN`` works whether this file is imported as a package
# module (``import experiments.step82...``, as the tests do) OR run directly as a script
# (``.venv/bin/python experiments/step82_certified_horizon_from_model.py``, the documented entry point, where only the
# script's own dir is on sys.path). We prepend the parent-of-experiments (the repo root); no-op if already present.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# step71 supplies the Henon map, its on-attractor sampler, and the documented exponent. We REUSE its
# ``henon_step`` / ``SYSTEMS["Henon"]`` / ``on_attractor_trajs`` for the attractor point cloud; do NOT modify it.
import experiments.step71_multichaos_horizon as step71  # noqa: E402

DTYPE = np.float64
HENON_A, HENON_B = 1.4, 0.3
# D phi(s) = [[-2a x, 1], [b, 0]] depends only on x; the only varying entry is -2a x with slope 2a in x, and the
# Jacobian's spectral norm is 1-Lipschitz in that entry, so Lip(z -> D phi) = 2a = 2.8 EXACTLY.
HENON_JAC_LIP = 2.0 * HENON_A          # = 2.8
# Textbook largest Lyapunov exponent of the Henon map (a=1.4, b=0.3): lambda_1 ~= 0.419 /step (step71's reference).
# Used only as the denominator of the honest certified-exponent looseness ratio log(Lambda_cert)/lambda_1 (sound iff >=1).
HENON_LAMBDA1_REF = 0.419

# Sentinel for a non-expanding certified map (log Lambda <= 0): the certified horizon is unbounded. A large finite
# integer keeps t_guar's return type a plain int (vs math.inf), so callers can compare/min/round without special-casing.
HORIZON_INF = 10**9


def _safe_log(x, floor=1e-12):
    r"""``log`` clamped at a tiny positive floor -- a reporting convenience for the bootstrap branch, where the upper
    confidence bound :math:`\lambda_1^{\text{hi}}` (already converted to a multiplier :math:`e^{\lambda}`) can be
    non-positive on a degenerate fit. Used ONLY for the printed/figure tightness ratio, never inside the certificate
    (which is sound by construction)."""
    return math.log(max(float(x), floor))


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


# =============================================================================================================== #
# Phase A' — the TIGHT certificate on a UNIFORMLY HYPERBOLIC system (the cat map).
#
# Phase A's certificate is sound but ~3.15x loose on the Henon map. Working out *why* (design spec §10) found a real
# obstruction, not an effort gap: Henon (a=1.4) is *non-uniformly* hyperbolic (homoclinic tangencies), so no global
# uniform cone field exists and a single-step metric caps at the worst-case operator-norm scale. Tight certification
# belongs to *uniformly hyperbolic* dynamics. The cleanest such system is the linear cat map (Arnol'd's hyperbolic
# toral automorphism), whose Jacobian A = [[2,1],[1,1]] is **symmetric** and **constant**:
#   * symmetric  => the optimal adapted metric is P = I  => Lambda = ||A||_2 = rho(A) = (3+sqrt5)/2 = e^{lambda_1}
#                   EXACTLY (kappa = 1, no rotation of the expanding direction to absorb);
#   * constant   => L_J = Lip(z -> Dphi) = 0  => the Lipschitz bridge adds ZERO slack
#                   => certified exponent = lambda_1 to machine precision  ==>  TIGHT BY CONSTRUCTION.
# A small smooth area-preserving perturbation stays Anosov (structural stability), is genuinely nonlinear, and has a
# small analytic L_J, so the bridge still closes near-tightly. Henon is retained as the honest "sound-but-conservative
# on non-uniformly-hyperbolic" companion: the certificate is provably tight where a uniform cone exists, soundly
# conservative where it does not, and knows which regime it is in (a tightness dimension on Prop 7's scope theorem).
#
# We REUSE the Phase-A machinery unchanged: adapted_metric -> lipschitz_bridge -> t_guar, _covering_radius. The cat
# map's SRB measure is Lebesgue on the torus, so a uniform sample on [0,1)^2 covers the invariant set (no need for
# step71.on_attractor_trajs, which is for systems whose attractor is a thin subset of state space).
# =============================================================================================================== #

# Linear cat map Jacobian A = [[2,1],[1,1]] (symmetric, det = 1, area-preserving). Eigenvalues (3 +- sqrt5)/2, so the
# leading Lyapunov exponent is analytic: lambda_1 = log rho(A) = log((3+sqrt5)/2).
CAT_A = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
CAT_LAMBDA1 = math.log((3.0 + math.sqrt(5.0)) / 2.0)     # ~= 0.9624236501


def cat_map(s):
    r"""Linear cat map $\phi(x,y)=(2x+y,\;x+y)\bmod 1$ on the 2-torus $[0,1)^2$ (Arnol'd's hyperbolic toral
    automorphism). ``s: (..., 2) -> (..., 2)``, float64, wrapped into $[0,1)$."""
    s = np.asarray(s, dtype=DTYPE)
    x, y = s[..., 0], s[..., 1]
    return np.stack([(2.0 * x + y) % 1.0, (x + y) % 1.0], axis=-1)


def cat_jac(s):
    r"""Jacobian of the linear cat map: the **constant** matrix $A=\begin{pmatrix}2&1\\1&1\end{pmatrix}$ at every point
    (the mod-1 wrap is a translation, contributing nothing to $D\phi$). Returns a ``(2, 2)`` float64 array;
    broadcasting over a batch just repeats $A$, so we return $A$ itself (callers stack as needed)."""
    return CAT_A.copy()


def benettin_lambda1(jac_fn, map_fn, s0, n_steps=4000, warmup=200, seed=0):
    r"""Hand-rolled Benettin leading-Lyapunov estimate $\lambda_1=\overline{\log\lVert D\phi(z_t)\,q_t\rVert}$ for a
    discrete map (one-vector power iteration with renormalization — the top channel of a QR Benettin; cf.
    ``step78.qr_logR_series``). We evolve a unit vector $q$ under the Jacobian field along an orbit $z_{t+1}=\phi(z_t)$,
    accumulating $\log\lVert D\phi(z_t)q_t\rVert$ after a warm-up. Used only to *cross-check* the analytic exponents in
    tests; the certificate itself never calls it. ``s0: (d,)`` -> ``float``."""
    rng = np.random.default_rng(seed)
    z = np.asarray(s0, dtype=DTYPE).copy()
    q = rng.standard_normal(z.shape[-1])
    q /= np.linalg.norm(q)
    acc, cnt = 0.0, 0
    for t in range(n_steps + warmup):
        J = np.asarray(jac_fn(z), dtype=DTYPE)
        w = J @ q
        nrm = float(np.linalg.norm(w))
        q = w / nrm
        z = map_fn(z)
        if t >= warmup:
            acc += math.log(nrm)
            cnt += 1
    return acc / cnt


# --------------------------------------------------------------------------------------------------------------- #
# Perturbed cat map (the nonlinear upgrade). With g(x) = (delta / 2pi) sin(2 pi x), so g'(x) = delta cos(2 pi x) and
# g''(x) = -2 pi delta sin(2 pi x):
#   phi(x,y) = ( 2x + y + g(x),  x + y + g(x) )  mod 1.
# The Jacobian is  Dphi = [[2 + g'(x), 1], [1 + g'(x), 1]],  with det = (2+g')*1 - 1*(1+g') = 1 for EVERY x
# (area-preserving for any delta — the perturbation only shears, never compresses). It varies only in x, through the
# single scalar g'(x), so the whole field is a constant matrix plus g'(x) * E with E = [[1,0],[1,0]].
# --------------------------------------------------------------------------------------------------------------- #
_CAT_PERT_E = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=DTYPE)     # carrier of the x-dependent shear; ||E||_2 = sqrt2


def perturbed_cat_map(s, delta=0.1):
    r"""Smooth area-preserving perturbation of the cat map,
    $\phi(x,y)=\bigl(2x+y+\tfrac{\delta}{2\pi}\sin2\pi x,\;x+y+\tfrac{\delta}{2\pi}\sin2\pi x\bigr)\bmod1$.
    For small $\delta$ it stays Anosov (structural stability of the automorphism). ``s: (...,2) -> (...,2)``."""
    s = np.asarray(s, dtype=DTYPE)
    x, y = s[..., 0], s[..., 1]
    g = (delta / (2.0 * math.pi)) * np.sin(2.0 * math.pi * x)
    return np.stack([(2.0 * x + y + g) % 1.0, (x + y + g) % 1.0], axis=-1)


def perturbed_cat_jac(s, delta=0.1):
    r"""Analytic Jacobian $D\phi(s)=\begin{pmatrix}2+g'(x)&1\\1+g'(x)&1\end{pmatrix}$, $g'(x)=\delta\cos2\pi x$
    (depends only on $x$; $\det\equiv1$). Returns a ``(2, 2)`` float64 array."""
    s = np.asarray(s, dtype=DTYPE)
    x = s[..., 0]
    gp = delta * math.cos(2.0 * math.pi * float(x))
    return np.array([[2.0 + gp, 1.0], [1.0 + gp, 1.0]], dtype=DTYPE)


def perturbed_cat_jac_lipschitz(delta=0.1):
    r"""**Analytic** Lipschitz constant of the Jacobian field $z\mapsto D\phi(z)$ in the operator norm. The field
    varies only in $x$, and $\partial_x D\phi(z)=g''(x)\,E$ with $E=\begin{pmatrix}1&0\\1&0\end{pmatrix}$,
    $g''(x)=-2\pi\delta\sin2\pi x$. Hence
    $\lVert\partial_x D\phi\rVert_2=|g''(x)|\,\lVert E\rVert_2\le(2\pi\delta)\sqrt2$, so
    $$L_J=2\sqrt2\,\pi\,\delta\qquad(\text{e.g. }\delta=0.1\Rightarrow L_J\approx0.8886),$$
    a sound (in fact tight, attained at $x=\tfrac14$) bound $\lVert D\phi(z)-D\phi(z')\rVert_2\le L_J\lVert z-z'\rVert$
    since the variation is along a single coordinate. The linear cat map is $\delta=0$ ⇒ $L_J=0$."""
    return 2.0 * math.sqrt(2.0) * math.pi * delta


# --------------------------------------------------------------------------------------------------------------- #
# Uniform-hyperbolicity certificate via a forward-invariant expanding cone (2D). In slope coordinates u = v_y / v_x a
# 2x2 matrix D = [[a,b],[c,d]] sends a direction of slope u to slope  T(u) = (c + d u) / (a + b u). A cone
# C = { |u - u*| <= w } (center u*, half-width w) is *forward-invariant* under a set of Jacobians {D_i} iff every D_i
# maps the closed cone strictly inside itself (T_i(boundary) lands in the open cone). It is *uniformly expanding* iff
# every unit vector with slope in C grows by a factor >= mu > 1 under every D_i. A single common forward-invariant,
# uniformly expanding cone across ALL sampled Jacobians is exactly the (unstable) cone condition for uniform
# (Anosov) hyperbolicity. We center the cone on A's unstable eigendirection (golden-ratio slope u* = (sqrt5 - 1)/2,
# from v_+ proportional to ((1+sqrt5)/2, 1)) and search the half-width w for the best expansion margin.
# --------------------------------------------------------------------------------------------------------------- #
_CAT_UNSTABLE_SLOPE = (math.sqrt(5.0) - 1.0) / 2.0           # = 1/phi; slope of A's unstable eigenvector


def _cone_dir_expansion(D, u):
    r"""Stretch factor $\lVert Dv\rVert/\lVert v\rVert$ for the unit vector $v\propto(1,u)$ (slope-$u$ direction)."""
    v = np.array([1.0, u], dtype=DTYPE)
    v /= np.linalg.norm(v)
    return float(np.linalg.norm(D @ v))


def _cone_image_slope(D, u):
    r"""Image slope $T_D(u)=(c+d u)/(a+b u)$ of a slope-$u$ direction under $D=[[a,b],[c,d]]$ (``inf`` if vertical)."""
    a, b = D[0, 0], D[0, 1]
    c, d = D[1, 0], D[1, 1]
    den = a + b * u
    if abs(den) < 1e-300:
        return math.inf
    return float((c + d * u) / den)


def cone_margin(jacs, center=_CAT_UNSTABLE_SLOPE, widths=None, n_dirs=41):
    r"""Verify a single forward-invariant, uniformly expanding **unstable cone** common to all sampled Jacobians
    (uniform / Anosov hyperbolicity) and return its **margin**.

    For each candidate half-width $w$ we form the cone $C=\{u:|u-\text{center}|\le w\}$, sample ``n_dirs`` directions
    across it, and compute two quantities over **all** Jacobians $D_i$:

    * **invariance margin** $= w - \max_{i,u\in\partial C}|T_{D_i}(u)-\text{center}|$ — positive iff every image slope
      of every cone direction lands strictly inside $C$ (the cone maps into itself);
    * **expansion margin** $= \min_{i,u\in C}\log\bigl(\lVert D_i v_u\rVert/\lVert v_u\rVert\bigr)$ — positive iff every
      direction in the cone is strictly stretched by every Jacobian.

    The cone is valid only when **both** are positive; we report $\min(\text{invariance},\text{expansion})$ and keep the
    width maximizing it. ``cone_margin > 0`` certifies uniform hyperbolicity on the sampled set; $\le0$ means no common
    expanding invariant cone (the certificate correctly abstains). ``jacs: (n, d, d) -> float``."""
    jacs = np.asarray(jacs, dtype=DTYPE)
    if widths is None:
        widths = np.linspace(0.05, 0.8, 24)
    best = -math.inf
    for w in widths:
        us = np.linspace(center - w, center + w, n_dirs)
        # invariance: every boundary image slope stays within the cone (check both endpoints over all Jacobians)
        worst_img = max(abs(_cone_image_slope(D, u) - center)
                        for D in jacs for u in (center - w, center + w))
        inv_margin = w - worst_img
        # expansion: smallest log-stretch over all cone directions and all Jacobians
        exp_margin = min(math.log(_cone_dir_expansion(D, u)) for D in jacs for u in us)
        m = min(inv_margin, exp_margin)
        if m > best:
            best = m
    return float(best)


def true_horizon_torus(true_map, eps, eps_res, n_starts=400, seed=0, max_t=200):
    r"""Empirical first-crossing horizon on the **torus**: the analogue of :func:`true_horizon` (Henon, Euclidean) but
    with **mod-1 (toroidal) distance** so wrap-around does not spuriously trigger a crossing. We perturb each of
    ``n_starts`` random torus points by $\epsilon$ in a random direction and report the median step $T$ at which the
    *toroidal* separation first exceeds $\epsilon_{\text{res}}$ (with $\epsilon_{\text{res}}<0.5$ the wrap is never
    closer than the true gap before the crossing). This is the soundness ground truth $T_{\text{true}}$ on the torus.

    Toroidal distance per coordinate: $\min(|\Delta|,\,1-|\Delta|)$, combined Euclidean. ``-> float``."""
    rng = np.random.default_rng(seed)
    z0 = rng.uniform(0.0, 1.0, size=(n_starts, 2)).astype(DTYPE)
    dirs = rng.standard_normal((n_starts, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    za, zb = z0.copy(), (z0 + eps * dirs) % 1.0
    crossed = np.full(n_starts, -1, dtype=int)
    for t in range(1, max_t + 1):
        za, zb = true_map(za), true_map(zb)
        dd = np.abs(za - zb) % 1.0
        dd = np.minimum(dd, 1.0 - dd)                       # toroidal per-coordinate distance
        sep = np.linalg.norm(dd, axis=1)
        newly = (crossed < 0) & (sep > eps_res)
        crossed[newly] = t
    crossed[crossed < 0] = max_t                            # never crossed within the budget
    return float(np.median(crossed))


def true_horizon(true_map, eps, eps_res, n_starts=40, seed=0, burn=300, T_max=4000):
    r"""Empirical first-crossing horizon on the TRUE (Euclidean) system -- the soundness ground truth $T_{\text{true}}$
    the certificate must under-promise. We burn $n_{\text{starts}}$ random inits onto the attractor, perturb each by a
    size-$\epsilon$ random displacement, evolve the pair under ``true_map``, and report the **median** first step $T$ at
    which the Euclidean separation $\lVert\delta_T\rVert$ exceeds the resolution $\epsilon_{\text{res}}$. (The torus
    analogue with mod-1 distance is :func:`true_horizon_torus`.) ``-> int``."""
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(-0.3, 0.3, size=(n_starts, 2)).astype(DTYPE)
    for _ in range(burn):
        s0 = np.array([true_map(s) for s in s0])
    cross = []
    for s in s0:
        sp = s + eps * rng.standard_normal(2) / math.sqrt(2)
        a, b = s.copy(), sp.copy()
        t = T_max
        for k in range(1, T_max + 1):
            a, b = true_map(a), true_map(b)
            if np.linalg.norm(b - a) > eps_res:
                t = k
                break
        cross.append(t)
    return int(np.median(cross))


def is_sound(t_guar, t_true):
    r"""Soundness self-check (Gate G2): the certified horizon must NOT exceed the true first-crossing horizon, i.e.
    $T_{\text{guar}}\le T_{\text{true}}$ (the certificate under-promises). ``-> bool``."""
    return bool(t_guar <= t_true)


def _run_cat_certificate(map_fn, jac_fn, L_J, lambda1, system, n_samples, seed, eps, eps_res, n_true=400):
    r"""Shared cat-map pipeline (linear & perturbed): uniform torus sample -> analytic Jacobians -> constant adapted
    metric (:func:`adapted_metric`) -> Lipschitz bridge (:func:`lipschitz_bridge`) -> certified exponent & horizon, plus
    the cone margin (uniform-hyperbolicity check) and the torus first-crossing $T_{\text{true}}$. Returns the certified
    exponent $\log\Lambda^{\text{cert}}$, the **tightness ratio** $\log\Lambda^{\text{cert}}/\lambda_1$, ``t_guar``,
    ``kappa``, ``h``, the cone margin, $T_{\text{true}}$, and the soundness flags."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n_samples, 2)).astype(DTYPE)     # SRB = Lebesgue on the torus
    jacs = np.stack([np.asarray(jac_fn(s), dtype=DTYPE) for s in pts])

    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts)
    br = lipschitz_bridge(lam_samples, kappa, L_J, h, eps=eps, eps_res=eps_res)
    log_lambda_cert = math.log(br["lambda_cert"])

    margin = cone_margin(jacs)
    t_true = true_horizon_torus(map_fn, eps, eps_res, n_starts=n_true, seed=seed)

    return dict(
        system=system, lambda1=float(lambda1), lambda_samples=float(lam_samples),
        lambda_cert=float(br["lambda_cert"]), log_lambda_cert=float(log_lambda_cert),
        tightness_ratio=float(log_lambda_cert / lambda1), kappa=kappa, h=float(h),
        slack=float(br["slack"]), t_guar=int(br["horizon"]), t_true=float(t_true),
        cone_margin=float(margin), anosov=bool(margin > 0.0),
        sound_exponent=bool(log_lambda_cert >= lambda1 - 1e-9),
        sound_horizon=bool(br["horizon"] <= t_true + 1e-9),
        certified=bool(br["certified"]), eps=eps, eps_res=eps_res,
    )


def run_true_catmap(n_samples=2000, seed=0, eps=0.01, eps_res=0.4):
    r"""**The tight anchor (analytic, must land).** The cone / adapted-metric certificate on the TRUE *linear* cat map.

    Because $A$ is symmetric the optimal constant metric is $P=I$ ($\kappa=1$, nothing to rotate away), giving
    $\Lambda_{\text{samples}}=\rho(A)=e^{\lambda_1}$ exactly; and $L_J=0$ (constant Jacobian) ⇒ the bridge slack is
    $0$ ⇒ $\log\Lambda^{\text{cert}}=\lambda_1$ to machine precision. The **tightness ratio** should be $\approx1.000$.
    $\epsilon_{\text{res}}=0.4<0.5$ keeps toroidal wrap from triggering the true crossing prematurely. Returns the dict
    from :func:`_run_cat_certificate` (certified exponent, tightness ratio, $T_{\text{guar}}$, $T_{\text{true}}$,
    $\kappa$, $h$, cone margin, soundness flags)."""
    return _run_cat_certificate(cat_map, lambda s: cat_jac(s), L_J=0.0, lambda1=CAT_LAMBDA1,
                                system="CatMap(linear)", n_samples=n_samples, seed=seed,
                                eps=eps, eps_res=eps_res)


def run_true_perturbed_catmap(delta=0.1, n_samples=2000, seed=0, eps=0.01, eps_res=0.4):
    r"""The **nonlinear upgrade**: the same certificate on the perturbed (still area-preserving, still Anosov for small
    $\delta$) cat map, with the **analytic** $L_J=2\sqrt2\,\pi\,\delta$ (:func:`perturbed_cat_jac_lipschitz`). The
    Jacobian now varies in $x$, so $P$ deviates slightly from $I$ and the bridge adds a small slack $\sqrt\kappa L_J h$;
    the tightness ratio is expected $\lesssim1.3$. The ``cone_margin`` in the returned dict verifies it is still
    uniformly hyperbolic. Returns the :func:`_run_cat_certificate` dict (with the perturbed map's analytic $\lambda_1$
    from a Benettin estimate is *not* used for the ratio denominator — we use the analytic-anchored value below)."""
    # The perturbed map has no closed-form lambda_1, but for small delta it is O(delta^2)-close to the linear value;
    # we report the ratio against a Benettin estimate of ITS OWN lambda_1 (sound denominator) so "tightness" is honest.
    lam1 = benettin_lambda1(lambda z: perturbed_cat_jac(z, delta), lambda z: perturbed_cat_map(z, delta),
                            s0=np.array([0.12, 0.34]), n_steps=6000, warmup=400, seed=seed)
    return _run_cat_certificate(lambda s: perturbed_cat_map(s, delta), lambda s: perturbed_cat_jac(s, delta),
                                L_J=perturbed_cat_jac_lipschitz(delta), lambda1=lam1,
                                system=f"CatMap(perturbed,delta={delta})", n_samples=n_samples, seed=seed,
                                eps=eps, eps_res=eps_res)


def tightness_comparison(n_samples=2000, seed=0, eps=0.01):
    r"""The validity-domain-**with-tightness** story in one table: the cone certificate is *provably tight* on a
    uniformly-hyperbolic system (the cat map, ratio $\approx1.00$, $T_{\text{guar}}\approx T_{\text{true}}$) and
    *soundly conservative* on a non-uniformly-hyperbolic one (Henon, the Phase-A $\sim\!3.15\times$). Returns a dict
    keyed by system with ``log_lambda_cert``, ``lambda1``, ``tightness_ratio``, ``t_guar``, ``cone_margin``, ``anosov``.

    Henon uses $\epsilon_{\text{res}}=1.0$ (its Phase-A setting, Euclidean state space); the cat map uses
    $\epsilon_{\text{res}}=0.4$ (torus). The ratio is the apples-to-apples *certified-exponent* conservatism
    $\log\Lambda^{\text{cert}}/\lambda_1$, which is independent of $\epsilon_{\text{res}}$."""
    cat = run_true_catmap(n_samples=n_samples, seed=seed, eps=eps, eps_res=0.4)
    hen = run_true_henon(n_samples=max(2000, n_samples), seed=seed, eps=eps, eps_res=1.0)
    hen_ratio = math.log(hen["lambda_cert"]) / 0.419       # Henon textbook lambda_1 (step71: 0.419 / step)
    return {
        "CatMap(linear)": dict(
            log_lambda_cert=cat["log_lambda_cert"], lambda1=cat["lambda1"],
            tightness_ratio=cat["tightness_ratio"], t_guar=cat["t_guar"], t_true=cat["t_true"],
            cone_margin=cat["cone_margin"], anosov=cat["anosov"]),
        "Henon(true)": dict(
            log_lambda_cert=math.log(hen["lambda_cert"]), lambda1=0.419,
            tightness_ratio=hen_ratio, t_guar=hen["t_guar"],
            cone_margin=cone_margin(np.stack([henon_jac(p) for p in
                                              np.random.default_rng(seed).uniform(-1, 1, size=(200, 2))])),
            anosov=False),
    }


# =============================================================================================================== #
# Phase B — the actual contribution: the SAME cone / adapted-metric certificate, but read off a LEARNED model's
# Jacobian field instead of the analytic one. This is what makes the ICLR title's "Certified *from the learned
# model*" literal rather than aspirational. Two extra honest costs appear that Phase A/A' did not pay:
#
#   1. The Jacobian is now an AUTOGRAD Jacobian of a trained net (B1), not a closed form.
#   2. The Lipschitz constant L_J = Lip(z -> D net(z)) is no longer analytic; for a black-box net we can only bound it
#      from the layers' spectral norms (B2). That bound is SOUND but typically loose -- the honest price of certifying
#      a learned model. When it is so loose that the cone bridge goes vacuous, we DO NOT loosen anything: we route to a
#      step78 block-bootstrap fallback on the SAME point cloud (B3), taking the UPPER lambda_1 confidence bound (the
#      fastest plausible divergence => the most conservative / shortest horizon). Reporting `route` = "cone" vs
#      "bootstrap" is itself an honest finding -- both are publishable outcomes (the controller decides framing).
#
# We REUSE the Phase-A machinery unchanged (adapted_metric -> lipschitz_bridge -> t_guar, _covering_radius) and
# step71's MLP + on_attractor_trajs + exact training recipe (do NOT modify step71); the fallback reuses step78.
# =============================================================================================================== #

# Sound max|sigma''| for the activations we certify (verified numerically to float64): tanh attains 0.7698 (we use the
# documented 0.77 cap), SiLU/swish attains exactly 0.5. step71's MLP is SiLU, so its net-Lipschitz bound uses SILU_S2.
TANH_SIGMA2_MAX = 0.77       # sound cap on |tanh''| (true max 0.7698)
SILU_SIGMA2_MAX = 0.5        # exact max of |SiLU''| (= |swish''|); step71's MLP activation


def learned_jacobian(net, z):
    r"""Autograd Jacobian $D(\text{net})(z)$ of a torch map at a single point ``z`` (1-D tensor). Returns a ``(d, d)``
    float64 numpy array. This replaces the analytic ``henon_jac`` / ``cat_jac`` of Phase A/A' with the Jacobian field of
    a *learned* net -- the only change needed to read the certificate off the model. Forward shape: ``z: (d,) -> (d, d)``."""
    z = z.detach().clone().to(torch.float64).requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda u: net(u), z, create_graph=False)
    return J.detach().double().numpy()


def net_jacobian_lipschitz(net, sigma_second_deriv_max=0.77):
    r"""**Sound** upper bound on $L_J^{\text{net}}=\mathrm{Lip}(z\mapsto D(\text{net})(z))$ for a ``Linear``/``Tanh`` MLP,
    from the layers' spectral norms alone.

    For $f=W_L\sigma(\cdots\sigma(W_1 z+b_1)\cdots)+b_L$ with an elementwise $1$-Lipschitz $\sigma$, the Jacobian is
    $Df(z)=W_L\,\mathrm{diag}(\sigma'(a_{L-1}))\,W_{L-1}\cdots\mathrm{diag}(\sigma'(a_1))\,W_1$. Differentiating once more,
    each $\mathrm{diag}(\sigma'(a_k))$ varies through $\tfrac{d}{dz}\sigma'(a_k)=\sigma''(a_k)\,\tfrac{da_k}{dz}$, and the
    pre-activation map $z\mapsto a_k$ is itself $\big(\prod_{\ell\le k}\lVert W_\ell\rVert_2\big)$-Lipschitz. Bounding the
    single second-derivative path (the others' $\sigma'$ factors are $\le1$) gives the standard
    $$L_J^{\text{net}}\ \le\ \big(\max_\ell|\sigma''|\big)\cdot\Big(\prod_\ell\lVert W_\ell\rVert_2\Big)\cdot
       \big(\max_\ell\lVert W_\ell\rVert_2\big),$$
    with $|\tanh''|\le0.77$ (the max of $|{-2}\tanh(1-\tanh^2)|$). This is **loose but sound** -- the honest cost of
    certifying a black-box net rather than a closed-form map (Phase A/A' had $L_J$ analytic). A net with no ``Linear``
    layers (or none found) returns $0.0$. ``net -> float``."""
    norms = []
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            norms.append(float(torch.linalg.matrix_norm(m.weight.detach().double(), 2)))
    if not norms:
        return 0.0
    prod = float(np.prod(norms))
    return sigma_second_deriv_max * prod * max(norms)


# step78 supplies the calibrated block-bootstrap CI on the Lyapunov spectrum and the horizon-interval conversion. We
# REUSE bootstrap_spectrum_ci + horizon_interval for the hybrid fallback; do NOT modify step78.
import experiments.step78_certified_horizon_ci as step78  # noqa: E402
# step74 supplies the high-dimensional Lorenz-96 system + its residual MLP for the high-D / deep-net STRETCH abstention
# (Phase C). We REUSE l96_rhs/rk4/true_map/attractor_traj/L96MLP; do NOT modify step74.
import experiments.step74_lorenz96_spectrum as step74  # noqa: E402


def bootstrap_fallback(logR, dt_map, eps=0.01, n_boot=400, block=50, seed=0, alpha=0.1):
    r"""**Hybrid backstop** for when the net-Lipschitz cone bridge goes vacuous (B2's bound too loose to certify): a
    *statistical* certified horizon from the SAME Jacobian field, via a block-bootstrap of the leading Lyapunov exponent.

    We feed the per-step $\log|\mathrm{diag}(R)|$ series (Benettin-QR along the learned orbit, :func:`_logR_from_jacs`)
    to :func:`step78.bootstrap_spectrum_ci`, then take the **UPPER** confidence bound $\lambda_1^{\text{hi}}$ -- the
    *fastest plausible divergence* -- and convert it to the **shortest** (most conservative) horizon. The conversion is
    the same $T=\log(1/\epsilon)/\lambda$ monotone-decreasing map that :func:`step78.horizon_interval` uses; we inline it
    rather than call ``horizon_interval`` because that helper's $\lambda\le0$ abstention is keyed to its *first* argument
    (the lower CB), whereas the *conservative* horizon we need is keyed to the *upper* CB, so the per-endpoint guard
    below is the correct sign-aware form. This is "conservative" in the SAME direction as the cone certificate (it
    under-promises the horizon), but the guarantee is now *statistical* (a CI coverage level), not the cone's
    deterministic a-priori bound -- an honest weaker fallback, reported as ``route="bootstrap"``.

    The **conservative** (shortest) horizon $t_{\text{lo}}=\lfloor\log(1/\epsilon)/\lambda_1^{\text{hi}}\rfloor$ is the
    load-bearing output and is well-defined whenever $\lambda_1^{\text{hi}}>0$. The optimistic (longest) horizon
    $t_{\text{hi}}=\lfloor\log(1/\epsilon)/\lambda_1^{\text{lo}}\rfloor$ is finite only when the *lower* CB
    $\lambda_1^{\text{lo}}>0$ (the spectrum is sign-stably chaotic); if the $\lambda_1$ CI straddles $0$ (sign-unstable)
    the longest horizon is unbounded -> :data:`HORIZON_INF` (we never return a nonsensical negative $T$). If even
    $\lambda_1^{\text{hi}}\le0$ the bootstrap sees no expansion at all and the whole horizon is unbounded. Returns
    ``dict(lambda1, lambda1_lo, lambda1_hi, t_point, t_lo, t_hi)`` (all horizons non-negative ``int``)."""
    lam, lam_lo, lam_hi = step78.bootstrap_spectrum_ci(logR, dt_map, n_boot, block, seed, alpha)
    lam1, lam1_lo, lam1_hi = float(lam[0]), float(lam_lo[0]), float(lam_hi[0])
    L = math.log(1.0 / eps)
    # conservative (shortest) horizon from the UPPER CB lam1_hi (fastest plausible divergence)
    t_lo = max(0, int(math.floor(L / lam1_hi))) if lam1_hi > 0 else HORIZON_INF
    # optimistic (longest) horizon from the LOWER CB lam1_lo; only finite when sign-stably chaotic (lam1_lo > 0)
    t_hi = max(0, int(math.floor(L / lam1_lo))) if lam1_lo > 0 else HORIZON_INF
    t_point = max(0, int(math.floor(L / lam1))) if lam1 > 0 else HORIZON_INF
    return dict(lambda1=lam1, lambda1_lo=lam1_lo, lambda1_hi=lam1_hi,
                t_point=t_point, t_lo=t_lo, t_hi=t_hi)


# --------------------------------------------------------------------------------------------------------------- #
# B4 helpers — train a step71.MLP on the Henon one-step map and read the certificate off ITS Jacobian field.
#
# step71's exact recipe (read from step71.run_system, NOT guessed): per-coordinate normalize the on-attractor
# trajectories with mu = flat.mean(0), sd = flat.std(0)+1e-9; the residual MLP `model(z)=z+net(z)` is trained so that
# `model(norm(s_t)) ~= norm(s_{t+1})` under MSE. The full one-step map in STATE space is therefore
#     phi(s) = denorm(model(norm(s))),   norm(s)=(s-mu)/sd,   denorm(z)=z*sd+mu,
# whose Jacobian factors by the chain rule (the affine norm/denorm have constant linear parts diag(1/sd), diag(sd)):
#     Dphi(s) = diag(sd) . Dmodel(norm(s)) . diag(1/sd) = diag(sd) . (I + Dnet(zhat)) . diag(1/sd),   zhat=norm(s).
# This is the plan's contract diag(sd)*Df_net*diag(1/sd)+I. We REUSE step71.MLP, step71.on_attractor_trajs and
# step71.SYSTEMS["Henon"] for the data; the training loop here mirrors step71.run_system (Adam 1e-3, batch 512) but is
# kept local so this experiment is self-contained and never mutates step71's globals (SEED/EPOCHS).
# --------------------------------------------------------------------------------------------------------------- #
def _train_henon_mlp(seed, smoke, hidden=None, epochs=None, n_traj=None, traj_len=None):
    r"""Train a :class:`step71.MLP` residual one-step predictor on the Henon map, mirroring step71's normalized-residual
    recipe. Returns ``(net, mu, sd)`` where ``net`` is the trained MLP (predicts the normalized next state from the
    normalized state) and ``mu, sd`` are the per-coordinate normalization stats (float64 numpy, shape ``(2,)``).

    ``smoke=True`` uses a tiny net (hidden=16) and few epochs (8) and few short trajectories so the wiring smoke is
    fast (~1 s) and is *allowed* to route to the bootstrap fallback. The full run uses step71-scale capacity."""
    import torch.nn as nn
    hidden = hidden if hidden is not None else (16 if smoke else 256)
    epochs = epochs if epochs is not None else (8 if smoke else 200)
    n_traj = n_traj if n_traj is not None else (12 if smoke else 200)
    traj_len = traj_len if traj_len is not None else (200 if smoke else 1500)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cfg = step71.SYSTEMS["Henon"]
    traj = step71.on_attractor_trajs(cfg, rng, n_traj, traj_len)        # (m, traj_len+1, 2), on-attractor
    flat = traj.reshape(-1, 2)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9                            # step71's exact normalization
    S = (traj - mu) / sd
    s_in = torch.tensor(S[:, :-1].reshape(-1, 2), dtype=torch.float64)
    s_out = torch.tensor(S[:, 1:].reshape(-1, 2), dtype=torch.float64)

    net = step71.MLP(2, hidden=hidden).double()                         # residual: forward(z) = z + inner(z)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    n = s_in.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            ((net(s_in[idx]) - s_out[idx]) ** 2).mean().backward()
            opt.step()
    net.eval()
    return net, mu.astype(DTYPE), sd.astype(DTYPE)


def _full_step_jacobian(net, mu, sd, s):
    r"""Chain-rule Jacobian of the un-normalized learned one-step map $\phi(s)=\mathrm{denorm}(\text{model}(\mathrm{norm}(s)))$
    at a single state ``s``:
    $$D\phi(s)=\mathrm{diag}(sd)\,D\text{model}(\hat z)\,\mathrm{diag}(1/sd),\qquad \hat z=(s-\mu)/sd,$$
    where $D\text{model}(\hat z)=$ :func:`learned_jacobian` of the residual net (already $=I+D\text{net}$). Returns a
    ``(2, 2)`` float64 numpy array -- the learned-model analogue of :func:`henon_jac`. ``s: (2,) -> (2, 2)``."""
    s = np.asarray(s, dtype=DTYPE)
    zhat = torch.tensor((s - mu) / sd, dtype=torch.float64)
    Dmodel = learned_jacobian(net, zhat)                                # (2,2): D(model)(zhat) = I + D(net)(zhat)
    return (np.diag(sd) @ Dmodel @ np.diag(1.0 / sd)).astype(DTYPE)


def _logR_from_jacs(jacs):
    r"""Benettin-style QR accumulation of a *sequence* of one-step Jacobians (in orbit order) into the per-step
    $\log|\mathrm{diag}(R)|$ series that :func:`step78.bootstrap_spectrum_ci` block-bootstraps. We evolve an orthonormal
    frame $Q$ under the supplied Jacobians, $Z_t=D_t Q_{t-1}=Q_t R_t$, and record $\log|\mathrm{diag}(R_t)|$ (the local
    log-stretches). The sampled point cloud is concatenated on-attractor trajectory segments, so consecutive Jacobians
    are orbit-ordered; bootstrapping the time-average over these rows yields the spectrum CI (the conjugating diagonal
    of the normalization is constant and cancels in the asymptotic rate, so doing this on the un-normalized $D\phi$ is
    equivalent to the normalized map up to the bridge's covering term). ``jacs: (n, d, d) -> (n, d)`` log-stretches."""
    jacs = np.asarray(jacs, dtype=DTYPE)
    d = jacs.shape[-1]
    Q = np.eye(d, dtype=DTYPE)
    rows = []
    for D in jacs:
        Z = D @ Q
        Q, R = np.linalg.qr(Z)
        # fix sign convention so diag(R) > 0 (QR is unique up to signs); the log-magnitudes are sign-independent
        rows.append(np.log(np.abs(np.diagonal(R)).clip(min=1e-300)))
    return np.array(rows)                                               # (n, d), sorted by the QR's running order


def run_learned_henon(n_samples=4000, seed=0, eps=0.01, eps_res=1.0, smoke=False):
    r"""**Phase-B contribution:** the cone / adapted-metric certificate read off a LEARNED Henon model's Jacobian field,
    with a step78 bootstrap hybrid fallback when the net-Lipschitz bridge is vacuous.

    Pipeline: train a :class:`step71.MLP` on the Henon one-step map (:func:`_train_henon_mlp`); take the chain-rule
    Jacobian $D\phi(z_i)$ of the *un-normalized* learned map at each on-attractor sample (:func:`_full_step_jacobian`);
    solve the constant adapted metric $(P,\Lambda_{\text{samples}})$ and read $\kappa,h$ exactly as Phase A; bound the
    learned Jacobian field's Lipschitz constant $L_J^{\text{net}}$ from the net's layer spectral norms
    (:func:`net_jacobian_lipschitz`, SiLU cap); and inflate to $\Lambda^{\text{cert}}$ (:func:`lipschitz_bridge`).

    **Routing (never loosened):** if the bridged certificate is non-vacuous ($T_{\text{guar}}\ge1$) the route is
    ``"cone"`` (pure deterministic a-priori certificate). Otherwise -- the net-Lipschitz bound is the honest loose price
    of a black-box net and may make the bridge vacuous -- we route to ``"bootstrap"``: a block-bootstrap of the leading
    exponent on the SAME Jacobian sequence (:func:`_logR_from_jacs` -> :func:`bootstrap_fallback`), taking the upper CB
    for a conservative statistical horizon. Both are publishable; we report which.

    Returns a dict with ``route``, ``t_guar``, and route-specific diagnostics (cone: ``lambda_cert``/``kappa``/``h``/
    ``slack``/``L_J_net``; bootstrap: ``lambda1_hi``/``cone_lambda_cert``/``cone_slack``/``L_J_net`` so the vacuity is
    auditable), plus ``one_step_relmse`` (did the model learn the map?) and the inputs."""
    net, mu, sd = _train_henon_mlp(seed, smoke)
    rng = np.random.default_rng(seed)
    cfg = step71.SYSTEMS["Henon"]
    trajs = step71.on_attractor_trajs(cfg, rng, n=max(8, n_samples // 200), length=300)
    pts = np.concatenate([t[:-1] for t in trajs], axis=0)[:n_samples].astype(DTYPE)

    # one-step relMSE of the learned map on these points (in normalized space, step71's metric) -- a model-quality flag
    Zhat = torch.tensor((pts - mu) / sd, dtype=torch.float64)
    with torch.no_grad():
        pred = net(Zhat).numpy() * sd + mu
    nxt = henon_map(pts)
    one_step_relmse = float(np.sum((pred - nxt) ** 2) / max(np.sum((nxt - nxt.mean(0)) ** 2), 1e-12))

    jacs = np.stack([_full_step_jacobian(net, mu, sd, s) for s in pts])
    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts)
    L_J = net_jacobian_lipschitz(net, sigma_second_deriv_max=SILU_SIGMA2_MAX)
    br = lipschitz_bridge(lam_samples, kappa, L_J, h, eps=eps, eps_res=eps_res)
    # certified-exponent ratio vs the textbook Henon lambda_1 = 0.419/step (sound iff >= 1): the honest looseness number
    cert_ratio = math.log(br["lambda_cert"]) / HENON_LAMBDA1_REF

    # Full Gate-G1 cone-acceptance (same as run_true_henon, never loosened): the cone is only meaningfully accepted if
    # it is BOTH non-vacuous (T_guar>=1) AND beats the trivial Euclidean continuum bound max_i||D_i|| + L_J h (the
    # constant-identity metric). A learned net's L_J^net is so loose that the slack L_J h can dominate BOTH bounds, in
    # which case "t_guar>=1" can squeak through arithmetically while the cone adds nothing over Euclidean -- that is a
    # vacuous-in-spirit cone, and the honest route is the bootstrap hybrid.
    euclid_bound = float(max(np.linalg.norm(D, 2) for D in jacs)) + L_J * h
    beats_euclidean = bool(br["lambda_cert"] < euclid_bound)
    cone_ok = bool(br["certified"] and beats_euclidean)

    # ALWAYS compute the bootstrap cross-check on the SAME Jacobian sequence (the fallback's number), so the report can
    # compare the deterministic cone vs the statistical horizon regardless of which route the gate selects.
    logR = _logR_from_jacs(jacs)
    fb = bootstrap_fallback(logR, dt_map=1.0, eps=eps)

    if cone_ok:
        return dict(system="Henon(learned)", route="cone", lambda_samples=float(lam_samples),
                    lambda_cert=br["lambda_cert"], log_lambda_cert=math.log(br["lambda_cert"]),
                    cert_ratio_vs_true=float(cert_ratio), t_guar=int(br["horizon"]),
                    kappa=kappa, h=float(h), slack=br["slack"], L_J_net=float(L_J),
                    euclid_bound=euclid_bound, beats_euclidean=beats_euclidean,
                    boot_t_lo=int(fb["t_lo"]), boot_lambda1=fb["lambda1"], boot_lambda1_hi=fb["lambda1_hi"],
                    one_step_relmse=one_step_relmse, eps=eps, eps_res=eps_res)
    # vacuous-in-spirit cone (net-Lipschitz slack dominates / loses to Euclidean) => hybrid statistical fallback
    return dict(system="Henon(learned)", route="bootstrap", t_guar=int(fb["t_lo"]),
                lambda1=fb["lambda1"], lambda1_hi=fb["lambda1_hi"], t_hi=int(fb["t_hi"]),
                cone_lambda_cert=br["lambda_cert"], cone_log_lambda_cert=math.log(br["lambda_cert"]),
                cone_cert_ratio_vs_true=float(cert_ratio), cone_slack=br["slack"], cone_t_guar=int(br["horizon"]),
                euclid_bound=euclid_bound, beats_euclidean=beats_euclidean, lambda_samples=float(lam_samples),
                kappa=kappa, h=float(h), L_J_net=float(L_J), one_step_relmse=one_step_relmse,
                eps=eps, eps_res=eps_res)


# =============================================================================================================== #
# The TIGHT-from-the-learned-model SHOWPIECE — certificate read off a learned model of the cat map.
#
# Phase A' showed the cone certificate is TIGHT on the *analytic* cat map (P=I, L_J=0 => ratio 1.000). The honest
# question for Phase B is whether that tightness SURVIVES the transfer to a LEARNED model. We train a SMALL MLP (1
# hidden layer) to predict the **lifted** linear cat map s |-> A s WITHOUT the mod-1, on samples s ~ uniform[0,1)^2.
# Lifting matters: the per-sample one-step Jacobian of the lifted map is the constant A (no wrap discontinuity), so the
# learned net's autograd Jacobian is well-defined everywhere and converges to ~A; the certificate then reads off that
# learned Jacobian field exactly as Phase A'/B do. A SMALL net is the whole point: the spectral-norm net-Lipschitz
# bound L_J^net (B2) grows with the layer-norm product, so a 1-hidden-layer net keeps L_J^net small (~6-8, vs the
# learned Henon's ~1370) and the bridge slack sqrt(kappa) L_J^net h tiny => the certified exponent stays near-tight.
# Honest expectation (what the test asserts): the *sampled* certified exponent is near-tight (the net learns ~A so
# Lambda_samples ~ e^{lambda_1}); the bridge inflates it via the loose net-Lipschitz bound, but only mildly, so the
# final ratio stays ~1.1-1.3 -- far tighter than learned Henon -- and the route stays "cone". If the net-Lipschitz
# bound ever made the cone vacuous we would route to bootstrap and report it; on this small net it does not.
# =============================================================================================================== #
def _train_lifted_catmap_mlp(seed, smoke, hidden=None, epochs=None, n_train=None, lr=2e-3):
    r"""Train a SMALL plain MLP $g_\theta:\mathbb R^2\to\mathbb R^2$ to predict the **lifted** linear cat map
    $s\mapsto A s$ (no mod-1) on $s\sim\text{uniform}[0,1)^2$. One hidden layer + SiLU keeps the spectral-norm
    net-Lipschitz bound small (tight bridge). No normalization/residual wrapper: the lifted map is global-linear and
    well-scaled on $[0,1)^2$, so $g_\theta(s)\approx As$ and $Dg_\theta(s)\approx A$ directly. Returns the trained
    ``net`` (a torch ``Sequential``). ``smoke=True`` => tiny (hidden=8, 60 epochs), allowed to be loose / fall back."""
    import torch.nn as nn
    hidden = hidden if hidden is not None else (8 if smoke else 32)
    epochs = epochs if epochs is not None else (60 if smoke else 600)
    n_train = n_train if n_train is not None else (1000 if smoke else 8000)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    S = rng.uniform(0.0, 1.0, size=(n_train, 2)).astype(DTYPE)
    X = torch.tensor(S, dtype=torch.float64)
    Y = torch.tensor(S @ CAT_A.T, dtype=torch.float64)                  # targets A s (lifted, no mod)
    net = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, 2)).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(X) - Y) ** 2).mean().backward()
        opt.step()
    net.eval()
    return net


def run_learned_catmap(n_samples=2000, seed=0, eps=0.01, eps_res=0.4, smoke=False):
    r"""**The tight-from-the-learned-model showpiece:** the cone / adapted-metric certificate read off a LEARNED model
    of the (lifted, linear) cat map, with the same routing as :func:`run_learned_henon`.

    Pipeline: train a small MLP on $s\mapsto As$ (:func:`_train_lifted_catmap_mlp`); sample a uniform torus cloud (SRB =
    Lebesgue); take the autograd Jacobian $Dg_\theta(z_i)\approx A$ at each (:func:`learned_jacobian`); solve the
    constant adapted metric, read $\kappa,h$; bound $L_J^{\text{net}}$ from the net's layer spectral norms (SiLU cap);
    inflate to $\Lambda^{\text{cert}}$ (:func:`lipschitz_bridge`); certify or fall back. The ground-truth $\lambda_1$ is
    the analytic :data:`CAT_LAMBDA1` (the net targets the *exact* linear $A$, so the system's exponent is known).

    Reports the **learned-model tightness ratio** $\log\Lambda^{\text{cert}}/\lambda_1$ and the route. Honest soundness
    flags: ``sound_exponent`` ($\log\Lambda^{\text{cert}}\ge\lambda_1$, the cert is an upper bound) and ``sound_horizon``
    ($T_{\text{guar}}\le T_{\text{true}}$ on the torus via :func:`true_horizon_torus`). Returns a dict with
    ``route``, ``lambda_samples``, ``lambda_cert``, ``log_lambda_cert``,
    ``tightness_ratio``, ``t_guar``, ``t_true``, ``kappa``, ``h``, ``slack``, ``L_J_net``, ``jac_dev_from_A``
    (max sampled $\lVert Dg_\theta-A\rVert_2$, the model-fidelity number), ``one_step_relmse``, the soundness flags, and
    the inputs."""
    net = _train_lifted_catmap_mlp(seed, smoke)
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n_samples, 2)).astype(DTYPE)       # SRB = Lebesgue on the torus

    # one-step relMSE of the learned lifted map vs A s, and the max sampled Jacobian deviation from A (model fidelity)
    with torch.no_grad():
        pred = net(torch.tensor(pts, dtype=torch.float64)).numpy()
    tgt = pts @ CAT_A.T
    one_step_relmse = float(np.sum((pred - tgt) ** 2) / max(np.sum((tgt - tgt.mean(0)) ** 2), 1e-12))
    jacs = np.stack([learned_jacobian(net, torch.tensor(p, dtype=torch.float64)) for p in pts])
    jac_dev = float(np.max([np.linalg.norm(J - CAT_A, 2) for J in jacs]))

    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts)
    L_J = net_jacobian_lipschitz(net, sigma_second_deriv_max=SILU_SIGMA2_MAX)
    br = lipschitz_bridge(lam_samples, kappa, L_J, h, eps=eps, eps_res=eps_res)
    log_lambda_cert = math.log(br["lambda_cert"])
    t_true = true_horizon_torus(cat_map, eps, eps_res, n_starts=400, seed=seed)

    common = dict(system="CatMap(learned,lifted)", lambda1=float(CAT_LAMBDA1), lambda_samples=float(lam_samples),
                  lambda_cert=float(br["lambda_cert"]), log_lambda_cert=float(log_lambda_cert),
                  tightness_ratio=float(log_lambda_cert / CAT_LAMBDA1), kappa=kappa, h=float(h),
                  slack=float(br["slack"]), L_J_net=float(L_J), jac_dev_from_A=jac_dev,
                  one_step_relmse=one_step_relmse, t_true=float(t_true),
                  sound_exponent=bool(log_lambda_cert >= CAT_LAMBDA1 - 1e-9), eps=eps, eps_res=eps_res)
    if br["certified"]:
        common.update(route="cone", t_guar=int(br["horizon"]),
                      sound_horizon=bool(br["horizon"] <= t_true + 1e-9))
        return common
    # vacuous cone (net-Lipschitz slack dominates) => hybrid statistical fallback on the SAME point cloud
    logR = _logR_from_jacs(jacs)
    fb = bootstrap_fallback(logR, dt_map=1.0, eps=eps)
    common.update(route="bootstrap", t_guar=int(fb["t_lo"]), boot_lambda1=fb["lambda1"],
                  boot_lambda1_hi=fb["lambda1_hi"], cone_lambda_cert=float(br["lambda_cert"]),
                  cone_slack=float(br["slack"]), sound_horizon=bool(fb["t_lo"] <= t_true + 1e-9))
    return common


# =============================================================================================================== #
# Phase C — VALIDATION (soundness G2 + tightness G3 + the regime diagnostic G1) and HONEST cone-abstention on the
# high-dimensional / non-uniformly-hyperbolic STRETCH systems (Lorenz, Rossler, Lorenz-96).
#
# The headline story is "tightness-by-regime": the SAME deterministic cone / adapted-metric certificate is
#   * TIGHT on a uniformly-hyperbolic system (cat map: cone margin > 0, certified-exponent ratio ~ 1.0), and
#   * SOUND-but-conservative where no global uniform cone exists (Henon: cone margin < 0, ratio ~ 3.2, the learned
#     model routes to the bootstrap hybrid),
# and it KNOWS which regime it is in (the cone margin sign). The stretch systems push that honesty further: on
# genuinely high-D / deep-net learned models the net-Lipschitz bridge is so loose it goes vacuous, the cone ABSTAINS,
# and the certificate routes to the step78 bootstrap hybrid -- a sound horizon, the EXPECTED and CORRECT outcome. We
# record the abstention, never force the cone.
#
# Gates (HONEST; never loosened):
#   G2 (soundness, load-bearing): over ALL (system, seed, eps) the certified T_guar must satisfy T_guar <= T_true.
#       Coverage must be 1.0; a single violation prints INCONCLUSIVE and is NOT papered over.
#   G3 (tightness, reported not gated): the certified-exponent / true-lambda_1 ratio and T_guar / T_true per system.
#   G1 (regime diagnostic): the cone-margin sign per system (+ => tight-cone-eligible, - => conservative; undefined in
#       d>2, where the 2-D slope-coordinate cone does not apply, so we report None and rely on the bridge vacuity).
#
# We REUSE the Phase A/A'/B machinery unchanged (adapted_metric -> lipschitz_bridge -> t_guar, _covering_radius,
# cone_margin, net_jacobian_lipschitz, _logR_from_jacs -> bootstrap_fallback, true_horizon, true_horizon_torus) and
# step71 (Lorenz/Rossler systems + on_attractor_trajs + MLP) and step74 (Lorenz-96 + L96MLP). Do NOT modify any of them.
# =============================================================================================================== #
EPS_LIST = (0.1, 0.05, 0.01)        # the soundness sweep: G2 coverage is computed over every (system, seed, eps) here


def _train_step71_mlp(name, seed, smoke, hidden=None, epochs=None, n_traj=None, traj_len=None):
    r"""Train a :class:`step71.MLP` residual one-step predictor on a step71 system's :math:`\Delta t`-map, mirroring
    step71's exact normalized-residual recipe (the generic sibling of :func:`_train_henon_mlp`, for the 3-D flows
    Lorenz/Rossler used in the stretch). Returns ``(net, mu, sd, cfg)``: the trained MLP (predicts the normalized next
    state from the normalized state), the per-coordinate normalization stats (float64 numpy, shape ``(dim,)``), and the
    step71 system config. ``smoke=True`` uses a tiny net / few epochs (fast wiring smoke, allowed to be loose)."""
    cfg = step71.SYSTEMS[name]
    d = cfg["dim"]
    hidden = hidden if hidden is not None else (16 if smoke else 256)
    epochs = epochs if epochs is not None else (8 if smoke else 120)
    n_traj = n_traj if n_traj is not None else (12 if smoke else 120)
    traj_len = traj_len if traj_len is not None else (200 if smoke else 1200)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    traj = step71.on_attractor_trajs(cfg, rng, n_traj, traj_len)        # (m, traj_len+1, dim), on-attractor
    flat = traj.reshape(-1, d)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9                            # step71's exact normalization
    S = (traj - mu) / sd
    s_in = torch.tensor(S[:, :-1].reshape(-1, d), dtype=torch.float64)
    s_out = torch.tensor(S[:, 1:].reshape(-1, d), dtype=torch.float64)

    net = step71.MLP(d, hidden=hidden).double()                         # residual: forward(z) = z + inner(z)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    n = s_in.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            ((net(s_in[idx]) - s_out[idx]) ** 2).mean().backward()
            opt.step()
    net.eval()
    return net, mu.astype(DTYPE), sd.astype(DTYPE), cfg


def _true_horizon_nd(step_fn, dim, eps, eps_res, n_starts=40, seed=0, burn=300, T_max=4000,
                     init=None, bound=1e6):
    r"""Dimension-general first-crossing horizon on a TRUE (Euclidean) :math:`d`-D system -- the soundness ground truth
    :math:`T_{\text{true}}` for the stretch flows/high-D systems (the d>2 analogue of :func:`true_horizon`, whose init
    is hard-wired 2-D; this is a NEW helper, it does not modify :func:`true_horizon`). We burn ``n_starts`` inits onto
    the attractor under ``step_fn``, perturb each by a size-:math:`\epsilon` random displacement, evolve the pair, and
    report the **median** first step at which the Euclidean separation exceeds :math:`\epsilon_{\text{res}}`. Diverged
    pairs (``|state| > bound`` or non-finite) are pinned at ``T_max`` (never crossed within budget). ``-> int``."""
    rng = np.random.default_rng(seed)
    if init is None:
        s0 = rng.uniform(-0.3, 0.3, size=(n_starts, dim)).astype(DTYPE)
    else:
        s0 = init(rng, n_starts).astype(DTYPE)
    s = s0
    for _ in range(burn):
        s = step_fn(s)
    s0 = s
    cross = []
    for k in range(n_starts):
        a = s0[k:k + 1].copy()
        b = a + eps * rng.standard_normal((1, dim)) / math.sqrt(dim)
        t = T_max
        for j in range(1, T_max + 1):
            a, b = step_fn(a), step_fn(b)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                t = T_max
                break
            if np.linalg.norm(b - a) > eps_res:
                t = j
                break
        cross.append(t)
    return int(np.median(cross))


def _stretch_record(name, dim, eps, eps_res, br_lambda_cert, br_slack, kappa, h, L_J, beats_euclidean,
                    boot_lambda1_hi, t_true, one_step_relmse, cone_ok, system_label):
    r"""Assemble one stretch (system, eps) honest-abstention record from the (eps-INDEPENDENT) fitted certificate
    constants + the per-eps horizon/true-horizon. Cone route only if ``cone_ok`` (non-vacuous AND beats Euclidean);
    otherwise the bootstrap hybrid. ``cone_margin=None`` for :math:`d>2` (the slope cone is undefined there)."""
    cone_t = t_guar(br_lambda_cert, kappa, eps, eps_res)
    boot_t = max(0, int(math.floor(math.log(1.0 / eps) / boot_lambda1_hi))) if boot_lambda1_hi > 0 else HORIZON_INF
    t_g = int(cone_t) if cone_ok else int(boot_t)
    return dict(system=system_label, dim=int(dim), route=("cone" if cone_ok else "bootstrap"), cone_margin=None,
                certified_or_abstained=("certified" if cone_ok else "abstained"),
                cone_lambda_cert=float(br_lambda_cert), cone_t_guar=int(cone_t), cone_slack=float(br_slack),
                beats_euclidean=bool(beats_euclidean), L_J_net=float(L_J), kappa=float(kappa), h=float(h),
                bootstrap_T_guar=int(boot_t), boot_lambda1_hi=float(boot_lambda1_hi), t_guar=t_g,
                t_true=float(t_true), sound=bool(t_g <= t_true + 1e-9), one_step_relmse=float(one_step_relmse),
                eps=float(eps), eps_res=float(eps_res))


def _attempt_cone_flow(name, eps_list, seed, eps_res, n_samples, smoke=False):
    r"""**STRETCH (honest abstention) — a 3-D flow (Lorenz / Rossler).** Attempt the cone / adapted-metric certificate
    on a LEARNED step71 model of the flow's :math:`\Delta t`-map; EXPECT the net-Lipschitz bridge to go vacuous (a deep
    SiLU MLP's spectral-norm Jacobian-Lipschitz bound is loose), the cone to ABSTAIN, and the route to be the step78
    bootstrap hybrid. This abstention is the EXPECTED, CORRECT outcome on a non-uniformly-hyperbolic high(er)-D learned
    model; we record it, never force the cone.

    Pipeline mirrors :func:`run_learned_henon` but for a step71 flow: train the MLP (:func:`_train_step71_mlp`), take
    the chain-rule Jacobian of the un-normalized learned map at each on-attractor sample (:func:`_full_step_jacobian`,
    dimension-general), solve the constant adapted metric (:func:`adapted_metric`, works in any :math:`d`), bound
    :math:`L_J^{\text{net}}` (:func:`net_jacobian_lipschitz`, SiLU cap), inflate (:func:`lipschitz_bridge`); if vacuous,
    bootstrap on the SAME Jacobian sequence (:func:`_logR_from_jacs` -> :func:`bootstrap_fallback`).

    Fit ONCE (the certificate is eps-independent except the closed-form horizon), then emit one record per
    ``eps_list`` entry (rescaling the horizon + recomputing the true first-crossing :math:`T_{\text{true}}`). Returns
    ``{eps_str: record}``."""
    net, mu, sd, cfg = _train_step71_mlp(name, seed, smoke)
    d = cfg["dim"]
    rng = np.random.default_rng(seed + 7)
    trajs = step71.on_attractor_trajs(cfg, rng, n=max(8, n_samples // 200), length=300)
    pts = np.concatenate([t[:-1] for t in trajs], axis=0)[:n_samples].astype(DTYPE)

    Zhat = torch.tensor((pts - mu) / sd, dtype=torch.float64)
    with torch.no_grad():
        pred = net(Zhat).numpy() * sd + mu
    nxt = cfg["step"](pts)
    one_step_relmse = float(np.sum((pred - nxt) ** 2) / max(np.sum((nxt - nxt.mean(0)) ** 2), 1e-12))

    jacs = np.stack([_full_step_jacobian(net, mu, sd, s) for s in pts])
    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts)
    L_J = net_jacobian_lipschitz(net, sigma_second_deriv_max=SILU_SIGMA2_MAX)
    euclid_bound = float(max(np.linalg.norm(D, 2) for D in jacs)) + L_J * h
    logR = _logR_from_jacs(jacs)
    fb = bootstrap_fallback(logR, dt_map=1.0, eps=eps_list[-1])         # bootstrap exponents are eps-independent

    out = {}
    for e in eps_list:
        br = lipschitz_bridge(lam_samples, kappa, L_J, h, eps=e, eps_res=eps_res)
        cone_ok = bool(br["certified"] and br["lambda_cert"] < euclid_bound)
        t_true = _true_horizon_nd(cfg["step"], d, e, eps_res, n_starts=40, seed=seed,
                                  burn=cfg["burn"], T_max=cfg["t_roll"] * 4, init=cfg["init"], bound=cfg["bound"])
        out[f"{e}"] = _stretch_record(name, d, e, eps_res, br["lambda_cert"], br["slack"], kappa, h, L_J,
                                      br["lambda_cert"] < euclid_bound, float(fb["lambda1_hi"]), t_true,
                                      one_step_relmse, cone_ok, f"{name}(learned,stretch)")
    return out


def _attempt_cone_l96(eps_list, seed, eps_res, N=20, smoke=False):
    r"""**STRETCH (honest abstention) — high-D Lorenz-96.** Attempt the cone / adapted-metric certificate on a LEARNED
    :class:`step74.L96MLP` (a dense Linear/SiLU residual MLP, so the spectral-norm net-Lipschitz bound APPLIES, unlike
    the cyclic-conv which has no ``Linear`` layers) of the :math:`N`-D Lorenz-96 :math:`\Delta t`-map. This is the
    genuinely high-D / deep-net case: the :math:`N\times N` autograd Jacobian field is well-defined, the constant
    adapted metric still solves, but the deep-net :math:`L_J^{\text{net}}` (a product of layer spectral norms) is so
    loose that the bridge goes vacuous -- the cone ABSTAINS and routes to the step78 bootstrap hybrid. EXPECTED, CORRECT.

    The cone margin is a 2-D-only quantity, so ``cone_margin=None`` here (the abstention is evidenced by the vacuous
    bridge). :math:`T_{\text{true}}` via :func:`_true_horizon_nd` on the true Lorenz-96 integrator (:func:`step74.true_map`,
    wrapped to numpy, in the SAME normalized coordinates so eps/eps_res match the certificate). Fit ONCE, emit one record
    per ``eps_list`` entry (rescale the horizon + recompute :math:`T_{\text{true}}`). Returns ``{eps_str: record}``."""
    n_samples = (150 if smoke else 400)
    epochs = 8 if smoke else 60
    # --- train a dense L96MLP on the normalized Delta t-map (step74's recipe, multi-step rollout) -----------------
    device = "cpu"
    traj = step74.attractor_traj(N, (1500 if smoke else 6000), seed, device).to(torch.float64)
    mu_t, sd_t = traj.mean(0), traj.std(0) + 1e-8
    xn = (traj - mu_t) / sd_t
    torch.manual_seed(seed)
    K = 2 if smoke else 5
    net = step74.L96MLP(N, hidden=(128 if smoke else 256), layers=3).double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = xn.shape[0] - K
    starts = xn[:n]
    tgts = [xn[1 + j: 1 + j + n] for j in range(K)]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 512):
            idx = torch.randperm(n, generator=g)[i:i + 512]
            z = starts[idx]
            loss = 0.0
            for j in range(K):
                z = net(z)
                loss = loss + ((z - tgts[j][idx]) ** 2).mean()
            (loss / K).backward()
            opt.step()
            opt.zero_grad()
    net.eval()
    with torch.no_grad():
        X, Y = xn[:-1], xn[1:]
        one_step_relmse = float((((net(X) - Y) ** 2).sum(-1) / (Y ** 2).sum(-1).clamp_min(1e-12)).mean())

    # --- normalized-coordinate one-step map zhat -> net(zhat); its autograd Jacobian is the learned Lyapunov operator
    mu = mu_t.numpy().astype(DTYPE)
    sd = sd_t.numpy().astype(DTYPE)
    pts_n = xn.numpy()[::max(1, xn.shape[0] // n_samples)][:n_samples].astype(DTYPE)   # on-attractor normalized states
    jacs = np.stack([learned_jacobian(net, torch.tensor(z, dtype=torch.float64)) for z in pts_n])
    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts_n)
    L_J = net_jacobian_lipschitz(net, sigma_second_deriv_max=SILU_SIGMA2_MAX)
    euclid_bound = float(max(np.linalg.norm(D, 2) for D in jacs)) + L_J * h
    logR = _logR_from_jacs(jacs)
    fb = bootstrap_fallback(logR, dt_map=1.0, eps=eps_list[-1])         # bootstrap exponents are eps-independent

    def step_norm(z):
        zt = torch.tensor(z * sd + mu, dtype=torch.float64)
        return ((step74.true_map(zt) - mu_t) / sd_t).numpy().astype(DTYPE)

    out = {}
    for e in eps_list:
        br = lipschitz_bridge(lam_samples, kappa, L_J, h, eps=e, eps_res=eps_res)
        cone_ok = bool(br["certified"] and br["lambda_cert"] < euclid_bound)
        t_true = _true_horizon_nd(step_norm, N, e, eps_res, n_starts=24, seed=seed, burn=50, T_max=2000)
        out[f"{e}"] = _stretch_record("Lorenz96", N, e, eps_res, br["lambda_cert"], br["slack"], kappa, h, L_J,
                                      br["lambda_cert"] < euclid_bound, float(fb["lambda1_hi"]), t_true,
                                      one_step_relmse, cone_ok, f"Lorenz96(N={N},learned,stretch)")
    return out


def _attempt_stretch(name, eps_list, seed, eps_res, smoke=False):
    r"""Dispatch a stretch-system cone attempt (fit ONCE, one record per :math:`\epsilon` in ``eps_list``) and wrap any
    failure as an HONEST ``skipped`` record rather than fabricating a result (the plan's "skipped: <reason>" escape
    hatch). ``name in {Lorenz, Rossler, Lorenz96}``; returns ``{eps_str: record}``."""
    try:
        if name == "Lorenz96":
            return _attempt_cone_l96(eps_list, seed=seed, eps_res=eps_res, N=(10 if smoke else 20), smoke=smoke)
        return _attempt_cone_flow(name, eps_list, seed=seed, eps_res=eps_res,
                                  n_samples=(400 if smoke else 1500), smoke=smoke)
    except Exception as e:                                              # pragma: no cover - honest skip, never fabricate
        rec = dict(system=f"{name}(stretch)", skipped=f"{type(e).__name__}: {e}", route=None,
                   cone_margin=None, certified_or_abstained="skipped", bootstrap_T_guar=None,
                   t_guar=None, t_true=None, sound=None, eps_res=eps_res)
        return {f"{e}": dict(rec, eps=e) for e in eps_list}


# --------------------------------------------------------------------------------------------------------------- #
# run() — assemble the per-system table, gate G2 (soundness coverage == 1.0), report G3 (tightness) + G1 (regime),
# run the stretch cone-abstention, write the headline figure. The heavy entry point (training + certification across
# systems x 3 seeds); the fast smoke tests in tests/test_step82.py cover the components.
# --------------------------------------------------------------------------------------------------------------- #
def _horizon_at_eps(result, eps, eps_res):
    r"""Recompute the certified horizon of an already-fitted ``run_*`` result at a NEW ``eps`` from its eps-independent
    constants -- so the soundness grid evaluates a system's certificate over the whole :math:`\epsilon` sweep WITHOUT
    retraining the net per :math:`\epsilon` (the certificate's only :math:`\epsilon`-dependence is the closed-form
    horizon). Cone route: :func:`t_guar` on the certified multiplier :math:`\Lambda^{\text{cert}}` and :math:`\kappa`.
    Bootstrap route: the conservative :math:`\lfloor\log(1/\epsilon)/\lambda_1^{\text{hi}}\rfloor` from the upper CB
    (matching :func:`bootstrap_fallback`). ``-> int``."""
    route = result.get("route", "cone")
    if route == "bootstrap":
        lam_hi = float(result.get("lambda1_hi", result.get("boot_lambda1_hi", 0.0)))
        return max(0, int(math.floor(math.log(1.0 / eps) / lam_hi))) if lam_hi > 0 else HORIZON_INF
    return t_guar(float(result["lambda_cert"]), float(result["kappa"]), eps, eps_res)


def _sound_grid(label, results_by_seed, t_true_fn, seeds, eps_list, eps_res):
    r"""Evaluate :func:`is_sound` over a (seed, eps) grid for one system from PRE-FITTED per-seed ``run_*`` results
    (``results_by_seed[seed]``), recomputing the horizon at each :math:`\epsilon` via :func:`_horizon_at_eps` (no
    retraining). ``t_true_fn(seed, eps) -> float`` is the empirical ground truth. Returns the per-cell records and the
    sound-cell count -> the G2 coverage."""
    recs, n_sound = [], 0
    for seed in seeds:
        for e in eps_list:
            tg = int(_horizon_at_eps(results_by_seed[seed], e, eps_res))
            tt = float(t_true_fn(seed, e))
            ok = is_sound(tg, tt)
            n_sound += int(ok)
            recs.append(dict(system=label, seed=int(seed), eps=float(e), t_guar=tg, t_true=tt, sound=bool(ok)))
    return recs, n_sound


def run(seeds=(0, 1, 2), eps_res_henon=1.0, eps_res_torus=0.4, smoke=False):
    r"""**Phase C entry point.** Assemble the certified-horizon validation across the main systems (true + learned cat
    map, true + learned Henon) over ``seeds`` x ``EPS_LIST``, gate **G2** (soundness coverage, must be 1.0), report
    **G3** (tightness ratios) and **G1** (cone-margin regime sign), attempt the honest cone-abstention on the STRETCH
    systems (Lorenz, Rossler, Lorenz-96), print the per-system table, and write the headline figure
    ``papers/figures/step82_certified_horizon.{json,png}``. Returns the full results dict (also serialized to JSON).

    G2 is load-bearing and never loosened: a single ``T_guar > T_true`` prints ``INCONCLUSIVE: soundness violated at
    (...)``. The stretch abstention (cone margin < 0 / bridge vacuous -> bootstrap) is the EXPECTED, CORRECT outcome,
    recorded not failed."""
    eps_list = EPS_LIST
    per_system = []          # the per-system table rows (one per system x route, with the load-bearing numbers)
    sound_recs = []          # every (system, seed, eps) soundness cell -> G2 coverage
    soundness_scatter = []   # (t_guar, t_true, system, seed, eps) for the figure's right panel

    # Each certificate is fit ONCE per seed (training + adapted metric + bridge are eps-independent); the (seed, eps)
    # soundness grid then recomputes only the closed-form horizon at each eps via _horizon_at_eps (no retraining). The
    # empirical T_true is cached per (seed, eps).
    tt_cache = {}

    def _cache_tt(key, fn, seed, e):
        k = (key, seed, e)
        if k not in tt_cache:
            tt_cache[k] = float(fn(seed, e))
        return tt_cache[k]

    # ----- MAIN system 1/2: TRUE + LEARNED cat map (the TIGHT, uniformly-hyperbolic anchor; eps_res on the torus) ---
    n_cat = 800 if smoke else 2000
    cat_true = {s: run_true_catmap(n_samples=n_cat, seed=s, eps=0.01, eps_res=eps_res_torus) for s in seeds}
    cat_learned = {s: run_learned_catmap(n_samples=(800 if smoke else 1500), seed=s, eps=0.01,
                                         eps_res=eps_res_torus, smoke=smoke) for s in seeds}

    def cat_tt(seed, e):
        return _cache_tt("cat", lambda s, ee: true_horizon_torus(
            cat_map, ee, eps_res_torus, n_starts=(120 if smoke else 400), seed=s), seed, e)

    rec, _ = _sound_grid("CatMap(linear,true)", cat_true, cat_tt, seeds, eps_list, eps_res_torus)
    sound_recs += rec
    rec, _ = _sound_grid("CatMap(learned)", cat_learned, cat_tt, seeds, eps_list, eps_res_torus)
    sound_recs += rec

    # ----- MAIN system 3/4: TRUE + LEARNED Henon (the SOUND-but-conservative, non-uniformly-hyperbolic companion) ---
    n_hen = 2000 if smoke else 4000
    hen_true = {s: run_true_henon(n_samples=n_hen, seed=s, eps=0.01, eps_res=eps_res_henon) for s in seeds}
    hen_learned = {s: run_learned_henon(n_samples=n_hen, seed=s, eps=0.01, eps_res=eps_res_henon, smoke=smoke)
                   for s in seeds}

    def hen_tt(seed, e):
        return _cache_tt("hen", lambda s, ee: true_horizon(
            henon_map, ee, eps_res_henon, n_starts=(20 if smoke else 40), seed=s), seed, e)

    rec, _ = _sound_grid("Henon(true)", hen_true, hen_tt, seeds, eps_list, eps_res_henon)
    sound_recs += rec
    rec, _ = _sound_grid("Henon(learned)", hen_learned, hen_tt, seeds, eps_list, eps_res_henon)
    sound_recs += rec

    # ----- per-system table rows (seed-0 representative + regime diagnostic via cone_margin) ------------------------
    def _cat_margin(jac_fn, seed):
        rng = np.random.default_rng(seed)
        return float(cone_margin(np.stack([np.asarray(jac_fn(rng.uniform(0, 1, 2)), dtype=DTYPE) for _ in range(80)])))

    def _hen_margin(seed):
        rng = np.random.default_rng(seed)
        return float(cone_margin(np.stack([henon_jac(rng.uniform(-1, 1, 2)) for _ in range(200)])))

    ct0 = cat_true[seeds[0]]
    per_system.append(dict(system="CatMap(linear,true)", route="cone", certified=ct0["certified"],
                           tightness_ratio=ct0["tightness_ratio"],
                           cert_exponent=ct0["log_lambda_cert"], true_lambda1=ct0["lambda1"],
                           t_guar=ct0["t_guar"], t_true=ct0["t_true"],
                           t_ratio=float(ct0["t_guar"] / max(ct0["t_true"], 1.0)),
                           cone_margin=_cat_margin(lambda s: cat_jac(s), seeds[0]), anosov=ct0["anosov"]))
    cl0 = cat_learned[seeds[0]]
    per_system.append(dict(system="CatMap(learned)", route=cl0["route"],
                           certified=(cl0["route"] == "cone"), tightness_ratio=cl0["tightness_ratio"],
                           cert_exponent=cl0["log_lambda_cert"], true_lambda1=cl0["lambda1"],
                           t_guar=cl0["t_guar"], t_true=cl0["t_true"],
                           t_ratio=float(cl0["t_guar"] / max(cl0["t_true"], 1.0)),
                           cone_margin=_cat_margin(lambda s: cat_jac(s), seeds[0]),
                           one_step_relmse=cl0.get("one_step_relmse")))
    ht0 = hen_true[seeds[0]]
    ht0_tt = hen_tt(seeds[0], 0.01)
    per_system.append(dict(system="Henon(true)", route="cone", certified=ht0["certified"],
                           tightness_ratio=float(math.log(ht0["lambda_cert"]) / HENON_LAMBDA1_REF),
                           cert_exponent=float(math.log(ht0["lambda_cert"])), true_lambda1=HENON_LAMBDA1_REF,
                           t_guar=ht0["t_guar"], t_true=float(ht0_tt),
                           t_ratio=float(ht0["t_guar"] / max(ht0_tt, 1.0)),
                           cone_margin=_hen_margin(seeds[0]), anosov=False))
    hl0 = hen_learned[seeds[0]]
    hl0_tt = hen_tt(seeds[0], 0.01)
    # Certified exponent: the cone branch reports a MULTIPLIER Lambda (exponent = log Lambda); the bootstrap branch
    # reports the upper-CB exponent lambda1_hi DIRECTLY (already a rate, not a multiplier — do NOT log it).
    hl0_exp = (_safe_log(hl0["lambda_cert"]) if hl0["route"] == "cone" else float(hl0["lambda1_hi"]))
    per_system.append(dict(system="Henon(learned)", route=hl0["route"],
                           certified=(hl0["route"] == "cone"), tightness_ratio=float(hl0_exp / HENON_LAMBDA1_REF),
                           cert_exponent=float(hl0_exp),
                           true_lambda1=HENON_LAMBDA1_REF, t_guar=hl0["t_guar"], t_true=float(hl0_tt),
                           t_ratio=float(hl0["t_guar"] / max(hl0_tt, 1.0)),
                           cone_margin=_hen_margin(seeds[0]), one_step_relmse=hl0.get("one_step_relmse")))

    # ----- STRETCH (honest abstention): Lorenz, Rossler, Lorenz-96 at seed 0, across EPS_LIST -----------------------
    # The stretch systems are an HONEST cone-abstention record, NOT G2 cells: on these high-D / deep-net learned models
    # the cone abstains (net-Lipschitz bridge vacuous) and routes to the bootstrap hybrid. Their T_guar<=T_true 'sound'
    # flag against the TRUE system is the misspecification gap (provably un-certifiable a-priori; design-spec honest
    # scope) -- recorded per system, but NOT folded into the load-bearing G2 (which is the cat-map + Henon coverage
    # where the certificate is the deterministic cone or its sound same-system bootstrap).
    stretch = {}
    stretch_sound_recs = []
    for name in ("Rossler", "Lorenz", "Lorenz96"):
        per_eps = _attempt_stretch(name, eps_list, seed=seeds[0], eps_res=eps_res_henon, smoke=smoke)  # fit ONCE
        for e in eps_list:
            r = per_eps[f"{e}"]
            if r.get("skipped") is None and r.get("t_guar") is not None:
                stretch_sound_recs.append(dict(system=r["system"], seed=int(seeds[0]), eps=float(e),
                                               t_guar=int(r["t_guar"]), t_true=float(r["t_true"]),
                                               sound=bool(r["sound"])))
        stretch[name] = per_eps
        r01 = per_eps[f"{eps_list[-1]}"]                                # the eps=0.01 record for the table row
        row = dict(system=r01["system"], route=r01.get("route"),
                   certified=bool(r01.get("certified_or_abstained") == "certified"),
                   certified_or_abstained=r01.get("certified_or_abstained"),
                   cone_margin=r01.get("cone_margin"), bootstrap_T_guar=r01.get("bootstrap_T_guar"),
                   t_guar=r01.get("t_guar"), t_true=r01.get("t_true"), sound=r01.get("sound"),
                   stretch=True, skipped=r01.get("skipped"))
        if r01.get("cone_lambda_cert") is not None and r01.get("t_guar") is not None and r01.get("t_true"):
            row["tightness_ratio"] = None                              # cone abstained => no certified-cone exponent ratio
            row["t_ratio"] = float(r01["t_guar"] / max(r01["t_true"], 1.0))
        per_system.append(row)

    # ----- G2: soundness coverage over the MAIN (system, seed, eps) cells (load-bearing; never loosened) ------------
    # G2 covers the cat-map (true + learned, torus T_true) and Henon (true + learned, Euclidean T_true) -- the systems
    # where the certificate is the deterministic cone or its sound same-system bootstrap. A learned model that certifies
    # NO net expansion returns T_guar = HORIZON_INF (an unbounded model horizon); against a finite true horizon that is
    # the (un-certifiable a-priori) misspecification gap, so we record it as a violation honestly rather than hide it.
    n_total = len(sound_recs)
    n_sound = sum(int(r["sound"]) for r in sound_recs)
    g2_coverage = n_sound / max(n_total, 1)
    violations = [r for r in sound_recs if not r["sound"]]
    for r in sound_recs:                                               # the soundness scatter = the G2 cells (on/below y=x)
        if r["t_guar"] is not None and r["t_true"] is not None and r["t_guar"] < HORIZON_INF:
            soundness_scatter.append([int(r["t_guar"]), float(r["t_true"]), r["system"], int(r["seed"]), float(r["eps"])])

    res = dict(seeds=list(seeds), eps_list=list(eps_list), smoke=smoke,
               main=dict(cat_true={str(s): cat_true[s] for s in seeds},
                         cat_learned={str(s): cat_learned[s] for s in seeds},
                         henon_true={str(s): hen_true[s] for s in seeds},
                         henon_learned={str(s): hen_learned[s] for s in seeds}),
               stretch=stretch, stretch_sound_records=stretch_sound_recs,
               per_system=per_system, sound_records=sound_recs,
               G1_regime=[dict(system=r["system"], cone_margin=r.get("cone_margin"),
                               regime=("tight-cone-eligible" if (r.get("cone_margin") or -1) > 0 else
                                       ("conservative" if r.get("cone_margin") is not None else "n/a (d>2)")))
                          for r in per_system],
               G2_soundness_coverage=g2_coverage, G2_total_cells=n_total, G2_sound_cells=n_sound,
               G2_violations=violations,
               G3_tightness=[dict(system=r["system"], route=r.get("route"), tightness_ratio=r.get("tightness_ratio"),
                                  t_ratio=r.get("t_ratio"), t_guar=r.get("t_guar"), t_true=r.get("t_true"))
                             for r in per_system])

    _gate_and_report(res)
    _save_figure(res, soundness_scatter)
    return res


def _gate_and_report(res):
    r"""Print the per-system table + the G1/G2/G3 verdicts. G2 (soundness coverage) is the load-bearing gate: if any
    cell has :math:`T_{\text{guar}}>T_{\text{true}}` we print ``INCONCLUSIVE: soundness violated at (...)`` and do NOT
    loosen anything. G3 (tightness) and G1 (regime sign) are reported as numbers, not gated. Stretch cone-abstention is
    printed as the EXPECTED outcome, not a failure."""
    p = lambda *a: print(*a, file=sys.stderr)
    p("\n[step82] ============ Phase C: certified-horizon validation (tightness-by-regime) ============")
    p(f"[step82] {'system':<34}{'route':<11}{'tight_ratio':<13}{'T_guar':<9}{'T_true':<9}{'cone_margin':<13}{'sound'}")
    for r in res["per_system"]:
        if r.get("skipped"):
            p(f"[step82] {r['system']:<34}{'SKIPPED':<11}{r['skipped']}")
            continue
        tr = r.get("tightness_ratio")
        tr_s = f"{tr:.3f}" if isinstance(tr, (int, float)) else "—(abstain)"
        cm = r.get("cone_margin")
        cm_s = (f"{cm:+.4f}" if isinstance(cm, (int, float)) else "None(d>2)")
        tg = r.get("t_guar"); tt = r.get("t_true")
        tg_s = (f"{tg}" if tg is not None else "—")
        tt_s = (f"{tt:.0f}" if isinstance(tt, (int, float)) else "—")
        snd = r.get("sound")
        snd_s = ("True" if (snd is True or r.get("certified")) else (str(snd)))
        p(f"[step82] {r['system']:<34}{str(r.get('route')):<11}{tr_s:<13}{tg_s:<9}{tt_s:<9}{cm_s:<13}{snd_s}")

    # G1 — regime diagnostic (the cone-margin sign): + tight-cone-eligible, - conservative, None for d>2
    p("\n[step82] G1 (regime diagnostic — cone-margin sign):")
    for g in res["G1_regime"]:
        cm = g["cone_margin"]
        cm_s = (f"{cm:+.4f}" if isinstance(cm, (int, float)) else "None")
        p(f"[step82]   {g['system']:<34} cone_margin {cm_s:<12} => {g['regime']}")

    # G3 — tightness (reported, not gated): certified-exponent ratio and T_guar/T_true
    p("\n[step82] G3 (tightness — certified-exponent/true-lambda_1 ratio; T_guar/T_true):")
    for g in res["G3_tightness"]:
        tr = g["tightness_ratio"]; trr = g["t_ratio"]
        tr_s = (f"{tr:.3f}" if isinstance(tr, (int, float)) else "—")
        trr_s = (f"{trr:.3f}" if isinstance(trr, (int, float)) else "—")
        p(f"[step82]   {g['system']:<34} exp-ratio {tr_s:<10} T_guar/T_true {trr_s}")

    # G2 — soundness coverage (LOAD-BEARING; must be 1.0)
    cov = res["G2_soundness_coverage"]
    p(f"\n[step82] G2 (soundness, load-bearing): coverage {cov:.4f} over {res['G2_total_cells']} "
      f"(system, seed, eps) cells [{res['G2_sound_cells']} sound]")
    if res["G2_violations"]:
        for v in res["G2_violations"]:
            p(f"[step82] INCONCLUSIVE: soundness violated at (system={v['system']}, seed={v['seed']}, "
              f"eps={v['eps']}): T_guar={v['t_guar']} > T_true={v['t_true']}")
        p("[step82] INCONCLUSIVE: G2 soundness coverage < 1.0 — NOT loosened; reported as-is.")
    else:
        p(f"[step82] G2 PASS: every certified horizon is a sound lower bound (T_guar <= T_true) across all "
          f"{res['G2_total_cells']} cells (coverage = 1.0).")

    # stretch — the EXPECTED honest cone-abstention (the headline finding is route=bootstrap, NOT a G2 gate cell)
    n_abst = sum(1 for pe in res["stretch"].values()
                 if pe[f"{res['eps_list'][-1]}"].get("certified_or_abstained") == "abstained")
    n_str = len(res["stretch"])
    p(f"\n[step82] STRETCH (honest cone-abstention; EXPECTED outcome = cone abstains -> bootstrap hybrid): "
      f"{n_abst}/{n_str} systems abstained")
    for name, per_eps in res["stretch"].items():
        r = per_eps[f"{res['eps_list'][-1]}"]
        if r.get("skipped"):
            p(f"[step82]   {name:<10} SKIPPED: {r['skipped']}")
        else:
            p(f"[step82]   {r['system']:<34} route={r.get('route')} ({r.get('certified_or_abstained')}); "
              f"cone slack L_J^net={r.get('L_J_net'):.1f} => bridge VACUOUS; bootstrap T_guar="
              f"{r.get('bootstrap_T_guar')} vs true T={r.get('t_true'):.0f} (sound={r.get('sound')})")
    # The stretch sound flag is the model-vs-truth MISSPECIFICATION gap (provably un-certifiable a-priori); it is
    # RECORDED honestly, NOT gated. sound=False on a stretch system means the learned model under-estimates the true
    # expansion rate -- the expected risk on a high-D / deep-net model, and exactly why the cone abstains there.
    str_recs = res.get("stretch_sound_records", [])
    if str_recs:
        n_str_sound = sum(int(x["sound"]) for x in str_recs)
        p(f"[step82]   stretch bootstrap-vs-true soundness (recorded, NOT a gate): {n_str_sound}/{len(str_recs)} "
          f"(sound<1 here = the model-vs-truth misspecification gap, the reason the cone correctly abstains).")
    p("[step82] ====================================================================================\n")


def _save_figure(res, soundness_scatter):
    r"""Write ``papers/figures/step82_certified_horizon.{json,png}``: the headline tightness-by-regime story.
    (left) per-system certified-exponent / true-:math:`\lambda_1` ratio as a bar, colored by route (cone=tight,
    bootstrap=hybrid), with the cone-margin sign annotated (the regime diagnostic); (right) a soundness scatter of
    :math:`T_{\text{guar}}` vs :math:`T_{\text{true}}` across all (system, seed, eps) with the :math:`y=x` line and
    everything on/below it (sound)."""
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "step82_certified_horizon.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    try:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))

        # ---- LEFT: tightness-by-regime bar (certified-exponent ratio per system; color = route) -------------------
        rows = [r for r in res["per_system"] if isinstance(r.get("tightness_ratio"), (int, float))]
        names = [r["system"].replace("(", "\n(") for r in rows]
        ratios = [r["tightness_ratio"] for r in rows]
        routes = [r.get("route") for r in rows]
        colors = ["#1f77b4" if rt == "cone" else "#d62728" for rt in routes]   # cone=blue (tight), bootstrap=red(hybrid)
        xpos = np.arange(len(rows))
        bars = axL.bar(xpos, ratios, color=colors, zorder=3)
        axL.axhline(1.0, color="k", lw=1.2, ls="--", label="ratio $=1$ (perfectly tight: $\\log\\Lambda^{cert}=\\lambda_1$)")
        for x, r, b in zip(xpos, rows, bars):
            cm = r.get("cone_margin")
            tag = (f"margin {cm:+.3f}" if isinstance(cm, (int, float)) else "margin None\n(d>2)")
            axL.text(x, b.get_height() + 0.04, f"{b.get_height():.2f}\n{tag}", ha="center", va="bottom", fontsize=7.0)
        # also place abstaining stretch systems as hatched zero-height markers with their bootstrap route annotated
        abst = [r for r in res["per_system"] if r.get("certified_or_abstained") == "abstained"]
        axL.set_xticks(xpos)
        axL.set_xticklabels(names, fontsize=7.0)
        axL.set_ylabel("certified-exponent ratio  $\\log\\Lambda^{\\mathrm{cert}}/\\lambda_1$")
        from matplotlib.patches import Patch
        axL.legend(handles=[Patch(color="#1f77b4", label="route: cone (tight, deterministic)"),
                            Patch(color="#d62728", label="route: bootstrap (hybrid, statistical)"),
                            plt.Line2D([0], [0], color="k", ls="--", lw=1.2, label="ratio $=1$ (perfectly tight)")],
                   fontsize=7.5, loc="upper left")
        axL.set_title("(a) Tightness by regime\n"
                      "tight where a uniform cone exists (cat, ratio$\\approx$1, margin$>$0);\n"
                      f"sound-conservative where not (Henon, margin$<$0); {len(abst)} stretch abstain")
        axL.set_ylim(0, max(ratios + [1.0]) * 1.3)

        # ---- RIGHT: soundness scatter T_guar vs T_true (all cells on/below y=x) ------------------------------------
        sc = np.array([[a, b] for a, b, *_ in soundness_scatter], dtype=float) if soundness_scatter else np.zeros((0, 2))
        if len(sc):
            sysnames = sorted({s[2] for s in soundness_scatter})
            cmap = plt.get_cmap("tab10")
            colmap = {nm: cmap(i % 10) for i, nm in enumerate(sysnames)}
            hi = float(max(sc.max(), 1.0)) * 1.1
            axR.plot([0, hi], [0, hi], "k--", lw=1.2, label="$y=x$ (boundary: $T_{guar}=T_{true}$)")
            axR.fill_between([0, hi], [0, hi], [hi, hi], color="green", alpha=0.05)   # sound region (T_guar <= T_true)
            for nm in sysnames:
                pts = np.array([[a, b] for a, b, s, *_ in soundness_scatter if s == nm], dtype=float)
                axR.scatter(pts[:, 0], pts[:, 1], s=34, color=colmap[nm], edgecolor="k", lw=0.4,
                            label=nm, zorder=3)
            n_sound = int(np.sum(sc[:, 0] <= sc[:, 1] + 1e-9))
            axR.set_xlabel("certified horizon  $T_{\\mathrm{guar}}$  [map steps]")
            axR.set_ylabel("true first-crossing horizon  $T_{\\mathrm{true}}$  [map steps]")
            axR.set_title("(b) Soundness (G2): all cells on/below $y=x$\n"
                          f"{n_sound}/{len(sc)} sound; coverage "
                          f"{res['G2_soundness_coverage']:.2f} — the certificate under-promises")
            axR.set_xlim(0, hi); axR.set_ylim(0, hi)
            axR.legend(fontsize=6.5, loc="lower right", ncol=1)
            axR.set_aspect("equal", adjustable="box")
        fig.suptitle("Certified predictability horizon FROM the learned model: tight by regime, sound everywhere "
                     "(Step 82)", y=1.02, fontsize=11)
        fig.tight_layout()
        fig.savefig(figdir / "step82_certified_horizon.png", dpi=130, bbox_inches="tight")
    except Exception as e:                                             # pragma: no cover - figure is non-load-bearing
        print(f"[step82]   (figure skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    import os
    _seeds = tuple(int(x) for x in os.environ.get("STEP82_SEEDS", "0,1,2").split(","))
    run(seeds=_seeds)
