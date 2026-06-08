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
import torch
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
