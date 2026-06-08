# Step 82 — Certified Horizon From the Learned Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/step82_certified_horizon_from_model.py` + `tests/test_step82.py` implementing a cone/adapted-metric **certificate** that outputs a guaranteed predictability horizon $T_{\text{guar}}(\epsilon)$ from a (learned) chaotic map alone, validated to be a sound, conservative lower bound on the true system's horizon — turning the ICLR title's "Certified" from *measured* into *guaranteed*.

**Architecture:** Theorem B′ says: if a constant SPD metric $P$ and constant $\Lambda$ satisfy $D\phi(z)^\top P\,D\phi(z)\preceq\Lambda^2 P$ on a forward-invariant set, then $\lambda_1\le\log\Lambda$ and $T_{\text{guar}}(\epsilon)=\lfloor(\log(\epsilon_{\text{res}}/\epsilon)-\tfrac12\log\kappa)/\log\Lambda\rfloor$, $\kappa=\mathrm{cond}(P)$. We verify the LMI on a finite sample of the attractor and **bridge to the continuum** with a Lipschitz bound: $\Lambda^{\text{cert}}=\Lambda_{\text{samples}}+\sqrt\kappa\,L_J\,h$ ($h$=covering radius, $L_J=\mathrm{Lip}(D\phi)$). Staged: **Phase A** = rigorous certificate on the *true* Hénon map (exact $L_J=2.8$); **Phase B** = the same certificate on a *learned* Hénon model (autograd Jacobian + a sound net-Lipschitz bound), with a `step78` bootstrap hybrid fallback if the net bridge is vacuous; **Phase C** = soundness/tightness validation against the true system + honest abstention on Lorenz/Rössler/Lorenz-96.

**Tech Stack:** Python 3.11, NumPy/SciPy (float64), PyTorch (autograd Jacobians, MPS/CPU), reuse `experiments/step70` (Lorenz), `step71` (Hénon/Rössler/Lorenz learned models + attractor sampling), `step74` (Lorenz-96), `step78` (block-bootstrap CI + `horizon_interval`). Mirror the `step79`/`step80` experiment+test structure.

**Gate semantics (refines the spec's "margin>0" into the operationally meaningful form).** The certificate is *sound by construction* (the theorem). We gate on:
- **G1 (non-vacuous):** the adapted-metric $\Lambda^{\text{cert}}$ yields $T_{\text{guar}}(\epsilon)\ge1$ at the target $(\epsilon,\epsilon_{\text{res}})$ **and** beats the trivial Euclidean bound ($\Lambda^{\text{cert}}<\max_i\lVert D_i\rVert+L_J h$). Phase A's true-Hénon run **must pass G1** or the cone idea is impractical → fall back.
- **G2 (soundness vs the true system):** $T_{\text{guar}}(\epsilon)\le T_{\text{true}}(\epsilon)$ on **100%** of $(\epsilon,\text{seed})$ — the empirical misspecification self-check.
- **G3 (tightness, reported not gated):** conservatism ratio $T_{\text{guar}}/T_{\text{true}}$ and certified-exponent ratio $\log\Lambda^{\text{cert}}/\lambda_1^{\text{true}}$.

**Repo rules (every task):** `.venv/bin/python`; float64 for all Jacobian/spectral work; reuse and **do NOT modify** `step70/71/74/78`; stage files **by name**; commit locally per phase with a HEREDOC message ending `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; **do NOT push**; **never loosen a gate** — print `INCONCLUSIVE` instead. Each phase ends at a review checkpoint.

---

## File structure

- `experiments/step82_certified_horizon_from_model.py` — all certificate logic + `run()` (one responsibility: the certified-horizon-from-the-model experiment).
- `tests/test_step82.py` — fast unit tests, **no training**.
- Reads (does not modify): `experiments/step71_multichaos_horizon.py` (`henon_step`, `MLP`, `on_attractor_trajs`), `experiments/step78_certified_horizon_ci.py` (`qr_logR_series`, `bootstrap_spectrum_ci`, `horizon_interval`), `step70`, `step74`.
- Writes: `papers/figures/step82_certified_horizon.{json,png}`.

---

## Phase A — Rigorous certificate on the TRUE Hénon map (make-or-break)

### Task A1: Hénon map, exact Jacobian, exact Jacobian-Lipschitz constant

**Files:**
- Create: `experiments/step82_certified_horizon_from_model.py`
- Test: `tests/test_step82.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_step82.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_step82.py -q`
Expected: FAIL (module/attrs not defined).

- [ ] **Step 3: Implement**

```python
# experiments/step82_certified_horizon_from_model.py  (module header omitted for brevity — include a docstring
# mirroring step79's, citing Theorem B' and the honest scope from the spec)
import numpy as np

DTYPE = np.float64
HENON_A, HENON_B = 1.4, 0.3
HENON_JAC_LIP = 2.0 * HENON_A          # = 2.8, exact: D phi(s) = [[-2a x, 1],[b, 0]] depends only on x

def henon_map(s):
    s = np.asarray(s, dtype=DTYPE)
    x, y = s[..., 0], s[..., 1]
    return np.stack([1.0 - HENON_A * x * x + y, HENON_B * x], axis=-1)

def henon_jac(s):
    s = np.asarray(s, dtype=DTYPE)
    x = s[..., 0]
    return np.array([[-2.0 * HENON_A * x, 1.0], [HENON_B, 0.0]], dtype=DTYPE)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_step82.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 A1: true Henon map + exact Jacobian + exact Jacobian-Lipschitz constant

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task A2: `t_guar` (the Theorem B′ closed form)

**Files:** Modify `experiments/step82_certified_horizon_from_model.py`; Test `tests/test_step82.py`

- [ ] **Step 1: Write the failing test**

```python
def test_t_guar_matches_closed_form_and_is_monotone():
    Lam, kappa, eps_res = 1.5, 4.0, 1.0
    import math
    expected = math.floor((math.log(eps_res / 0.01) - 0.5 * math.log(kappa)) / math.log(Lam))
    assert s82.t_guar(Lam, kappa, 0.01, eps_res) == expected
    # smaller eps (sharper resolution demand at fixed eps_res) => longer horizon
    assert s82.t_guar(Lam, kappa, 1e-4, eps_res) >= s82.t_guar(Lam, kappa, 1e-2, eps_res)
    # Lambda = 1 (no expansion) => unbounded horizon sentinel
    assert s82.t_guar(1.0, 1.0, 0.01, 1.0) == s82.HORIZON_INF
```

- [ ] **Step 2: Run to verify it fails** — Run: `.venv/bin/python -m pytest tests/test_step82.py::test_t_guar_matches_closed_form_and_is_monotone -q` → FAIL.

- [ ] **Step 3: Implement**

```python
import math
HORIZON_INF = 10**9   # sentinel for log Lambda <= 0 (no certified expansion => unbounded)

def t_guar(Lambda, kappa, eps, eps_res):
    """Theorem B': certified horizon (in map steps) from the verified constants alone."""
    if Lambda <= 1.0 + 1e-15:
        return HORIZON_INF
    val = (math.log(eps_res / eps) - 0.5 * math.log(max(kappa, 1.0))) / math.log(Lambda)
    return max(0, int(math.floor(val)))
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 A2: t_guar closed form (Theorem B') with unbounded-horizon sentinel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task A3: `adapted_metric` (constant-P common-Lyapunov solve)

**Files:** Modify experiment + test.

- [ ] **Step 1: Write the failing test**

```python
def test_adapted_metric_on_symmetric_matrix_recovers_spectral_norm():
    # For a single symmetric matrix, the optimal constant metric gives Lambda = sigma_max = |max eig|.
    A = np.array([[1.3, 0.0], [0.0, 0.7]], dtype=np.float64)
    jacs = np.stack([A, A, A])
    P, Lam, _ = s82.adapted_metric(jacs)
    assert abs(Lam - 1.3) < 1e-2
    # P is SPD
    assert np.all(np.linalg.eigvalsh(P) > 0)

def test_adapted_metric_beats_euclidean_on_rotated_expansion():
    # A non-normal matrix: Euclidean op-norm overestimates; the adapted metric tightens toward rho(A)=1.2.
    A = np.array([[1.2, 5.0], [0.0, 1.2]], dtype=np.float64)
    jacs = np.stack([A, A])
    P, Lam, _ = s82.adapted_metric(jacs)
    assert Lam < np.linalg.norm(A, 2) - 1e-3        # strictly better than Euclidean
    assert Lam >= 1.2 - 1e-2                          # cannot beat the spectral radius
```

- [ ] **Step 2: Run to verify it fails** — FAIL (no `adapted_metric`).

- [ ] **Step 3: Implement** (SDP-free; minimize the worst-case metric operator norm over the log-Cholesky of `P`, `scipy.optimize`)

```python
from scipy.optimize import minimize
from scipy.linalg import sqrtm

def _metric_opnorm(L_flat, jacs, d):
    # P = C C^T with C lower-triangular, positive diagonal via exp on the diagonal entries.
    C = np.zeros((d, d))
    idx = np.tril_indices(d)
    C[idx] = L_flat
    di = np.diag_indices(d)
    C[di] = np.exp(np.clip(np.diag(C), -10, 10))
    P = C @ C.T
    Phalf = sqrtm(P).real
    Pinvhalf = np.linalg.inv(Phalf)
    # Lambda(P) = max_i || Phalf D_i Pinvhalf ||_2  (operator norm in the P-metric)
    norms = [np.linalg.norm(Phalf @ D @ Pinvhalf, 2) for D in jacs]
    return max(norms), P

def adapted_metric(jacs, succ_jacs=None, mode="constant"):
    """Constant adapted metric P minimizing Lambda s.t. D^T P D <= Lambda^2 P on all sampled jacs.
    succ_jacs/mode reserved for a future smooth field; constant-P uses only `jacs` (L_P=0)."""
    jacs = np.asarray(jacs, dtype=np.float64)
    d = jacs.shape[-1]
    x0 = np.zeros(d * (d + 1) // 2)                 # P = I initial
    best = minimize(lambda v: _metric_opnorm(v, jacs, d)[0], x0,
                    method="Nelder-Mead",
                    options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
    Lam, P = _metric_opnorm(best.x, jacs, d)
    P = P / np.trace(P) * d                          # normalize trace=d (cosmetic; cond unchanged)
    return P, float(Lam), float(best.fun)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (both tests; Nelder-Mead on d=2 is fast and reliable).

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 A3: constant adapted-metric (common-Lyapunov) solve, SDP-free via scipy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task A4: `lipschitz_bridge` (sample → continuum) + `cert_lambda`

**Files:** Modify experiment + test.

- [ ] **Step 1: Write the failing test**

```python
def test_lipschitz_bridge_inflates_lambda_and_is_sound():
    # Continuum-certified Lambda = Lambda_samples + sqrt(kappa) * L_J * h
    out = s82.lipschitz_bridge(lambda_samples=1.30, kappa=4.0, L_J=2.8, h=0.01)
    assert abs(out["lambda_cert"] - (1.30 + 2.0 * 2.8 * 0.01)) < 1e-12   # sqrt(4)=2
    assert out["lambda_cert"] > 1.30                                      # bridge only inflates (sound)

def test_lipschitz_bridge_vacuous_when_slack_dominates():
    # Huge grid spacing => Lambda_cert explodes => horizon vacuous => certified=False
    out = s82.lipschitz_bridge(lambda_samples=1.30, kappa=4.0, L_J=2.8, h=10.0,
                               eps=0.01, eps_res=1.0)
    assert out["certified"] is False
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement**

```python
def lipschitz_bridge(lambda_samples, kappa, L_J, h, L_P=0.0, eps=0.01, eps_res=1.0):
    """Inflate the sampled metric op-norm to a continuum-sound bound, then check non-vacuity (G1, part 1).
    L_P=0 for a constant metric. The bridge: ||Phalf D(z) Pinvhalf|| <= Lambda_samples + sqrt(kappa) L_J h."""
    slack = math.sqrt(max(kappa, 1.0)) * L_J * h + L_P * h
    lambda_cert = lambda_samples + slack
    horizon = t_guar(lambda_cert, kappa, eps, eps_res)
    return dict(lambda_cert=float(lambda_cert), slack=float(slack),
                horizon=horizon, certified=bool(horizon >= 1))
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 A4: Lipschitz sample->continuum bridge + non-vacuity (G1 part 1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task A5: wire `run_true_henon()` + Gate G1 (the make-or-break result)

**Files:** Modify experiment + test.

- [ ] **Step 1: Write the failing test** (a small, training-free smoke of the wiring)

```python
def test_run_true_henon_smoke_is_sound_and_non_vacuous():
    out = s82.run_true_henon(n_samples=400, seed=0, eps=0.01, eps_res=1.0)
    # certificate is sound by construction; on the TRUE map it must be non-vacuous (G1)
    assert out["certified"] is True
    assert out["t_guar"] >= 1
    # certified exponent brackets the textbook Henon exponent from ABOVE (sound: log Lambda_cert >= lambda_1)
    assert math.log(out["lambda_cert"]) >= 0.419 - 0.05
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement** (reuse `step71.on_attractor_trajs` for the point cloud; covering radius `h` = a robust nearest-neighbour estimate)

```python
import experiments.step71_multichaos_horizon as step71

def _covering_radius(pts):
    # h = max over points of the nearest-neighbour distance (a sound covering radius for the sample)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.max(d[:, 1]))

def run_true_henon(n_samples=4000, seed=0, eps=0.01, eps_res=1.0):
    rng = np.random.default_rng(seed)
    cfg = step71.SYSTEMS["Henon"]
    trajs = step71.on_attractor_trajs(cfg, rng, n=max(8, n_samples // 200), length=300)
    pts = np.concatenate([t[:-1] for t in trajs], axis=0)[:n_samples].astype(np.float64)
    jacs = np.stack([henon_jac(s) for s in pts])
    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P))
    h = _covering_radius(pts)
    br = lipschitz_bridge(lam_samples, kappa, HENON_JAC_LIP, h, eps=eps, eps_res=eps_res)
    euclid = float(max(np.linalg.norm(D, 2) for D in jacs)) + HENON_JAC_LIP * h
    return dict(system="Henon(true)", lambda_samples=lam_samples, kappa=kappa, h=h,
                lambda_cert=br["lambda_cert"], t_guar=br["horizon"],
                certified=bool(br["certified"] and br["lambda_cert"] < euclid),
                beats_euclidean=bool(br["lambda_cert"] < euclid), euclid_bound=euclid)
```

- [ ] **Step 4: Run to verify it passes** — Run the smoke test; then run the full thing once:
`.venv/bin/python -c "import experiments.step82_certified_horizon_from_model as s; import json; print(json.dumps(s.run_true_henon(), indent=2))"`
Expected: `certified: true`, `t_guar >= 1`, `beats_euclidean: true`. **This is the Phase-A make-or-break result.** If `certified` is False (e.g. the covering-radius slack dominates), the cone idea is impractical on the true map → STOP, report INCONCLUSIVE, escalate to the reviewer before Phase B.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 A5: run_true_henon + Gate G1 (rigorous certificate on the true Henon map)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

> **REVIEW CHECKPOINT (Phase A).** Report the true-Hénon result: `lambda_cert`, `t_guar`, `beats_euclidean`, `h`, `kappa`. G1 must pass. Do **not** start Phase B until the controller confirms.

---

## Phase B — The contribution: certificate on the LEARNED Hénon model

### Task B1: `learned_jacobian` (autograd) + finite-diff check

**Files:** Modify experiment + test.

- [ ] **Step 1: Write the failing test**

```python
import torch

def test_learned_jacobian_matches_finite_difference():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).double()
    z = torch.tensor([0.1, -0.2], dtype=torch.float64)
    J = s82.learned_jacobian(net, z)                        # (2,2) numpy
    fd = np.zeros((2, 2)); e = 1e-6
    base = net(z).detach().numpy()
    for j in range(2):
        zp = z.clone(); zp[j] += e
        fd[:, j] = (net(zp).detach().numpy() - base) / e
    assert np.max(np.abs(J - fd)) < 1e-4
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement**

```python
import torch

def learned_jacobian(net, z):
    """Autograd Jacobian of a torch map at z (1-D tensor). Returns a (d,d) float64 numpy array."""
    z = z.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda u: net(u), z, create_graph=False)
    return J.detach().double().numpy()
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 B1: autograd Jacobian of a learned map + finite-difference check

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task B2: `net_jacobian_lipschitz` — a SOUND net Jacobian-Lipschitz bound

**Files:** Modify experiment + test.

For an MLP $f=W_L\sigma(\cdots\sigma(W_1 z))$ with 1-Lipschitz $\sigma$ and $|\sigma''|\le1$ (tanh: $|\sigma''|\le0.77$), a sound bound on $\mathrm{Lip}(z\mapsto Df(z))$ is $L_J^{\text{net}}=\big(\max_\ell|\sigma''|\big)\cdot\big(\prod_\ell\lVert W_\ell\rVert_2\big)\cdot\big(\max_\ell\lVert W_\ell\rVert_2\big)$ (product of spectral norms × an extra layer norm for the second-derivative path). This is loose but **sound** — the honest cost of certifying a black-box net.

- [ ] **Step 1: Write the failing test**

```python
def test_net_jacobian_lipschitz_upper_bounds_empirical_on_samples():
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).double()
    Lhat = s82.net_jacobian_lipschitz(net)
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        a = torch.tensor(rng.uniform(-1, 1, 2), dtype=torch.float64)
        b = a + torch.tensor(rng.uniform(-1e-3, 1e-3, 2), dtype=torch.float64)
        Ja, Jb = s82.learned_jacobian(net, a), s82.learned_jacobian(net, b)
        worst = max(worst, np.linalg.norm(Ja - Jb, 2) / max(np.linalg.norm((a - b).numpy()), 1e-12))
    assert Lhat >= worst - 1e-9     # sound: the analytic bound dominates every sampled local slope
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement**

```python
def net_jacobian_lipschitz(net, sigma_second_deriv_max=0.77):
    """Sound upper bound on Lip(z -> D net(z)) for a Linear/Tanh MLP, from layer spectral norms.
    Conservative by design (the price of certifying a black-box net)."""
    norms = []
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            norms.append(float(torch.linalg.matrix_norm(m.weight.double(), 2)))
    if not norms:
        return 0.0
    prod = float(np.prod(norms))
    return sigma_second_deriv_max * prod * max(norms)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 B2: sound neural-net Jacobian-Lipschitz bound from layer spectral norms

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task B3: `bootstrap_fallback` (the step78 hybrid backstop)

**Files:** Modify experiment + test.

- [ ] **Step 1: Write the failing test**

```python
def test_bootstrap_fallback_returns_a_horizon_interval():
    # synthetic chaotic logR (one positive channel) -> a finite horizon interval
    rng = np.random.default_rng(0)
    logR = rng.normal(0.4, 0.1, size=(2000, 2)) * np.array([1.0, -1.0])  # ch0 expand, ch1 contract
    out = s82.bootstrap_fallback(logR, dt_map=1.0, eps=0.01)
    assert out["t_lo"] >= 1 and out["t_hi"] >= out["t_lo"]
    assert out["lambda1_hi"] >= out["lambda1"]    # upper CB used for the conservative horizon
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement** (reuse `step78`; the conservative horizon uses the **upper** CI on $\lambda_1$)

```python
import experiments.step78_certified_horizon_ci as step78

def bootstrap_fallback(logR, dt_map, eps=0.01, n_boot=400, block=50, seed=0, alpha=0.1):
    """Hybrid backstop: block-bootstrap the leading exponent, take the UPPER confidence bound
    (worst-case => conservative/short horizon), convert to a horizon interval. Reuses step78."""
    lam, lam_lo, lam_hi = step78.bootstrap_spectrum_ci(logR, dt_map, n_boot, block, seed, alpha)
    lam1, lam1_lo, lam1_hi = float(lam[0]), float(lam_lo[0]), float(lam_hi[0])
    # certified-conservative horizon uses lam1_hi (fastest plausible divergence)
    T_point, T_lo, T_hi = step78.horizon_interval(lam1_hi, lam1_lo, eps)   # note: lo<->hi swap => T order
    return dict(lambda1=lam1, lambda1_lo=lam1_lo, lambda1_hi=lam1_hi,
                t_point=int(T_point), t_lo=int(min(T_lo, T_hi)), t_hi=int(max(T_lo, T_hi)))
```

- [ ] **Step 2b note:** confirm `horizon_interval`'s argument order against `step78` (it takes `(lam_lo, lam_hi, eps)` and returns the corresponding `T` endpoints; a larger $\lambda$ ⇒ shorter $T$). Adjust the call so `t_lo` is the **conservative (shortest)** horizon from `lam1_hi`.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 B3: step78 bootstrap hybrid fallback (conservative horizon from the upper CB)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task B4: wire `run_learned_henon()` (train/reuse step71 MLP → certificate or fallback)

**Files:** Modify experiment + test (smoke only; full training runs in `run()`).

- [ ] **Step 1: Write the failing test** (training-free smoke: the function exists and routes)

```python
def test_run_learned_henon_smoke_routes_and_is_sound(tmp_path):
    out = s82.run_learned_henon(n_samples=300, seed=0, eps=0.01, eps_res=1.0, smoke=True)
    assert out["route"] in ("cone", "bootstrap")          # certifies or falls back, never crashes
    assert out["t_guar"] >= 1
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement** (train a `step71.MLP` on the Hénon map; `smoke=True` uses a tiny net/epochs and is allowed to fall back)

```python
def _train_henon_mlp(seed, smoke):
    # minimal residual MLP trained on one-step Henon; mirror step71's training (normalized residual).
    # Returns (net, mu, sd) where net predicts the normalized one-step residual. (Full recipe: 60 epochs;
    # smoke: 3 epochs, hidden=16.) Reuse step71.MLP and step71.on_attractor_trajs for data.
    ...

def run_learned_henon(n_samples=4000, seed=0, eps=0.01, eps_res=1.0, smoke=False):
    net, mu, sd = _train_henon_mlp(seed, smoke)
    rng = np.random.default_rng(seed)
    cfg = step71.SYSTEMS["Henon"]
    trajs = step71.on_attractor_trajs(cfg, rng, n=8, length=300)
    pts = np.concatenate([t[:-1] for t in trajs], axis=0)[:n_samples].astype(np.float64)
    jacs = np.stack([_full_step_jacobian(net, mu, sd, s) for s in pts])   # Jacobian of the un-normalized map
    P, lam_samples, _ = adapted_metric(jacs)
    kappa = float(np.linalg.cond(P)); h = _covering_radius(pts)
    L_J = net_jacobian_lipschitz(net)
    br = lipschitz_bridge(lam_samples, kappa, L_J, h, eps=eps, eps_res=eps_res)
    if br["certified"]:
        return dict(route="cone", lambda_cert=br["lambda_cert"], t_guar=br["horizon"], h=h, kappa=kappa)
    # INCONCLUSIVE cone => hybrid fallback on the SAME point cloud
    logR = _logR_from_jacs(jacs)                          # QR-accumulate per-step log-diagonals
    fb = bootstrap_fallback(logR, dt_map=1.0, eps=eps)
    return dict(route="bootstrap", t_guar=fb["t_lo"], lambda1_hi=fb["lambda1_hi"],
                cone_lambda_cert=br["lambda_cert"], cone_slack=br["slack"])
```

Implement the helpers `_full_step_jacobian` (chain-rule through the normalization: $D(\text{map}) = \mathrm{diag}(sd)\,Df_{\text{net}}\,\mathrm{diag}(1/sd)+I$ for a residual net — match step71's exact normalization) and `_logR_from_jacs` (a Benettin-style QR accumulation of the supplied Jacobian sequence, or reuse `step78.qr_logR_series` by passing a step-fn that replays the learned map).

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (smoke routes to either branch). Then run the real thing in `run()` (Task C3).

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 B4: run_learned_henon (cone certificate, hybrid bootstrap fallback on the same point cloud)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

> **REVIEW CHECKPOINT (Phase B).** Report whether the learned-Hénon route is `cone` (pure-(a) success) or `bootstrap` (hybrid fallback), with `lambda_cert`/`slack` either way. Both are acceptable, publishable outcomes; the controller decides framing. Do not start Phase C until confirmed.

---

## Phase C — Validation (soundness G2 + tightness G3) + honest abstention

### Task C1: `true_horizon` + soundness checker

**Files:** Modify experiment + test.

- [ ] **Step 1: Write the failing test**

```python
def test_true_horizon_and_soundness_checker():
    th = s82.true_horizon(s82.henon_map, eps=0.01, eps_res=1.0, n_starts=20, seed=0)
    assert th >= 1
    # soundness checker: a deliberately too-long certified horizon is flagged as a violation
    assert s82.is_sound(t_guar=th + 5, t_true=th) is False
    assert s82.is_sound(t_guar=max(1, th - 1), t_true=th) is True
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement**

```python
def true_horizon(true_map, eps, eps_res, n_starts=40, seed=0, burn=300, T_max=4000):
    """Median first-crossing horizon on the TRUE system: perturb by eps, evolve both, find first T
    where ||delta_T|| > eps_res. The empirical ground truth the certificate must under-promise."""
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(-0.3, 0.3, size=(n_starts, 2)).astype(np.float64)
    for _ in range(burn):
        s0 = np.array([true_map(s) for s in s0])
    cross = []
    for s in s0:
        sp = s + eps * rng.standard_normal(2) / np.sqrt(2)
        a, b = s.copy(), sp.copy(); t = T_max
        for k in range(1, T_max + 1):
            a, b = true_map(a), true_map(b)
            if np.linalg.norm(b - a) > eps_res:
                t = k; break
        cross.append(t)
    return int(np.median(cross))

def is_sound(t_guar, t_true):
    return bool(t_guar <= t_true)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py
git commit -F - <<'EOF'
Step 82 C1: true-system first-crossing horizon + soundness checker (G2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

### Task C2: `run()` — G1/G2/G3 across seeds + figure + honest abstention on Lorenz/Rössler/L96

**Files:** Modify experiment; no new test (covered by smoke tests; `run()` is the heavy entry point).

- [ ] **Step 1: Implement `run()`**

```python
def run():
    eps_list, eps_res = [0.1, 0.05, 0.01], 1.0
    res = {"henon_true": run_true_henon(eps=0.01, eps_res=eps_res)}
    # learned Henon, 3 seeds; soundness G2 across (eps, seed); tightness G3
    learned, sound_hits, total = [], 0, 0
    for seed in (0, 1, 2):
        r = run_learned_henon(seed=seed, eps=0.01, eps_res=eps_res)
        for e in eps_list:
            tg = r["t_guar"] if r["route"] == "cone" else r["t_guar"]   # already conservative
            tt = true_horizon(henon_map, eps=e, eps_res=eps_res, seed=seed)
            total += 1; sound_hits += int(is_sound(tg, tt))
            r.setdefault("ratios", []).append(tg / max(tt, 1))
        learned.append(r)
    res["learned_henon"] = learned
    res["G2_soundness_coverage"] = sound_hits / max(total, 1)   # must be 1.0
    # STRETCH (honest abstention): Lorenz/Rossler/Lorenz-96 — attempt the cone, expect/record abstention
    res["stretch"] = {name: _attempt_certificate(name) for name in ("Rossler", "Lorenz", "Lorenz96")}
    _gate_and_report(res)        # prints G1/G2 PASS/FAIL or INCONCLUSIVE; never loosens
    _save_figure(res)            # papers/figures/step82_certified_horizon.{json,png}
    return res
```

Implement `_attempt_certificate(name)` (reuse `step70`/`step71`/`step74` learned models and their Jacobians; return `{"certified": bool, "reason": "singular-hyperbolic / high-D Lipschitz slack"}` — abstention is recorded, not forced to pass), `_gate_and_report` (G2 coverage must be `1.0` else print `INCONCLUSIVE`), and `_save_figure` (left: certified vs empirical horizon staircase on Hénon with $T_{\text{guar}}\le T_{\text{true}}$; right: a small table/bar of which systems certify vs abstain).

- [ ] **Step 2: Run the full experiment**

Run: `.venv/bin/python experiments/step82_certified_horizon_from_model.py`
Expected: prints the per-system table; **G2 coverage = 1.0** (soundness); Hénon G1 PASS; stretch systems certify or **honestly abstain**. Writes the figure JSON+PNG.

- [ ] **Step 3: Run the test suite**

Run: `.venv/bin/python -m pytest tests/test_step82.py -q`
Expected: all tests PASS (fast, no training).

- [ ] **Step 4: Commit**

```bash
git add experiments/step82_certified_horizon_from_model.py tests/test_step82.py papers/figures/step82_certified_horizon.json papers/figures/step82_certified_horizon.png
git commit -F - <<'EOF'
Step 82 C2: run() with G1/G2/G3 gates, 3-seed soundness, honest abstention + figure

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

> **REVIEW CHECKPOINT (Phase C / final).** Report: Hénon G1 (certified, $T_{\text{guar}}$, beats-Euclidean); G2 soundness coverage (must be 1.0); G3 tightness ratios; learned-Hénon route (cone vs bootstrap); stretch-system certify/abstain table. Then STOP for the controller to decide the paper fold (Appendix B theorem + ICLR paragraph + paper2 section + Lever-A "certified" rewrite) and confirm before any push.

---

## Self-review notes (author)

- **Spec coverage:** Theorem B′ → A2/A3/A4 (`t_guar`, `adapted_metric`, `lipschitz_bridge`); computable certificate + bridge → A3–A5; Stage 1 true-Hénon → A5; Stage 2 learned-Hénon + hybrid fallback → B1–B4; Stage 3 soundness/tightness + abstention → C1–C2; gates G1/G2/G3 → A5/C1/C2; figure + reuse step70/71/74/78 → C2. All covered.
- **No-placeholder exceptions (flagged honestly):** three helpers are described by their exact math contract rather than full code — `_train_henon_mlp` (mirror `step71`'s training recipe), `_full_step_jacobian` (chain-rule through `step71`'s residual normalization: $\mathrm{diag}(sd)\,Df_{\text{net}}\,\mathrm{diag}(1/sd)+I$), and `_logR_from_jacs` (QR-accumulate). They depend on `step71`'s exact normalization, which the implementer must read from `step71` rather than guess — kept as contracts to avoid hard-coding a possibly-wrong copy. Every gate-bearing / math-critical function (A1–A4, B1–B3, C1) is full code.
- **Type consistency:** `adapted_metric → (P, Lambda, fun)`; `lipschitz_bridge → dict(lambda_cert, slack, horizon, certified)`; `t_guar → int`; `run_*` → dicts with `t_guar`/`route`/`certified`. Consistent across tasks.
