# Certified-Horizon-From-the-Learned-Model (Lever B) — design spec

*Date: 2026-06-08 · Status: design, pre-implementation · Owner: SE3-EJEPA / Certified World Models*

## 1. Goal & the gap it closes

The ICLR paper currently *measures* that a learned model's leading Lyapunov exponent matches the textbook value
($1$–$12\%$, E2) — an **observation** a reviewer dismisses as "Pathak/Vlachas already do this." The title says
**Certified**. Lever B makes that word literally true: a **computable certificate, from the learned model alone**
(no access to the true dynamics or true $\lambda_1$), that outputs a guaranteed predictability horizon
$T_{\text{guar}}(\epsilon)$ with stated, *checkable* assumptions.

**Honest scope (load-bearing — stated up front, not hidden).** The certificate is a-priori and deterministic about
the **learned model's error-amplification horizon**. The link to the **true** system's horizon
($T_{\text{guar}}\le T_{\text{true}}$) is the *misspecification gap*, which is provably **un-certifiable a-priori**
(if the learned model under-estimates the true exponent the bound over-promises). So true-system soundness is an
**empirical self-check**, not a theorem. The deliverable is therefore: *"a rigorous certificate of the learned
model's amplification horizon, empirically validated to transfer"* — NOT an unconditional true-system guarantee
(claiming that would be the exact overclaim we are fixing).

## 2. The theorem (Appendix B) — proved by hand

**Theorem B′ (cone / adapted-metric certified horizon).** Let $\hat\phi:\mathcal U\to\mathcal U$ be $C^1$ on a
compact forward-invariant $\mathcal U\subset\mathbb R^d$. Suppose there exist a continuous field of SPD matrices
$z\mapsto P(z)\succ0$ (an *adapted metric*, equivalently a cone field) and a constant $\Lambda\ge1$ such that
$$ D\hat\phi(z)^\top\,P(\hat\phi(z))\,D\hat\phi(z)\ \preceq\ \Lambda^2\,P(z)\qquad\forall z\in\mathcal U. \tag{LMI}$$
Then, with $V(z,v)=v^\top P(z)v$, (LMI) gives $V(\hat\phi(z),D\hat\phi(z)v)\le\Lambda^2 V(z,v)$; iterating,
$\|D\hat\phi^T(z)v\|^2 \le \kappa\,\Lambda^{2T}\,\|v\|^2$ with $\kappa=\sup_z\mathrm{cond}(P(z))$, hence
$$ \lambda_1(\hat\phi)\le\log\Lambda,\qquad
   T_{\text{guar}}(\epsilon)=\Big\lfloor \tfrac{\log(\epsilon_{\text{res}}/\epsilon)-\tfrac12\log\kappa}{\log\Lambda}\Big\rfloor, $$
a quantity computed from $\hat\phi$ alone. The optimal adapted metric drives $\Lambda\to e^{\lambda_1}$, so the bound is
**tight, not merely sound** — that tightening is the work the cone/metric does (it absorbs the rotation of the
expanding direction that makes the naive $\sup_z\|D\hat\phi(z)\|$ bound useless). Proof: sub-multiplicative cocycle /
Lyapunov-metric argument; the cone is the $V$-sublevel structure.

## 3. The computable certificate (the crux of the rigor)

What makes it a *certificate* (not "checked on a grid") is the **sample → continuum bridge**:

1. **Cover** $\mathcal U$ with a trajectory point cloud $\{z_i\}$ (grid spacing $h$ from the attractor's reach).
2. **Jacobians** $D\hat\phi(z_i)$ — exact (true map) or autograd (learned model), float64.
3. **Solve** for the adapted metric. *Start with a single global constant $P$* (one SDP over all samples →
   $L_P=0$, the cleanest possible bridge); escalate to a smooth field $P(z)$ only if the constant metric's $\Lambda$
   is uselessly loose (YAGNI).
4. **Bridge.** Bound $L_J=\mathrm{Lip}(z\mapsto D\hat\phi)$ (and $L_P$ if a field). If the LMI holds on $\{z_i\}$ with
   **margin** $m>(\text{slack from }L_J,L_P,h)$, it holds on **all** of the covered $\mathcal U$ → certificate valid;
   else **INCONCLUSIVE** (no certificate — honest).

## 4. Staged execution (so "实在不行" is detected early & cheap; fallback is an addition, not a rewrite)

All stages share one Jacobian-field point cloud and one true-horizon validator, so the hybrid fallback is one
function call on artifacts already computed.

- **Stage 1 — make-or-break, do first.** Prove Theorem B′. Implement the cone/LMI certificate + Lipschitz bridge on
  the **true Hénon map** ($d{=}2$ map; $D\phi$ is an explicit polynomial → *exact* $L_J$ → a **fully rigorous**
  certificate, no neural-net slop). *Decision point:* if the bridge can't certify even the true map, the cone idea is
  dead — stop, report, fall back. If it certifies, we hold the rigorous anchor regardless of what follows.
- **Stage 2 — the contribution.** Same certificate on the **learned** Hénon model (autograd Jacobian + a sound
  neural-net Lipschitz bound for the bridge). *Decision point ("实在不行"):* if the net's Lipschitz bound is too loose
  → margin fails → **INCONCLUSIVE** → trigger the **hybrid fallback**: call `step78`'s block-bootstrap on the same
  point cloud for a statistical horizon $+$ the self-check. The section then reports: rigorous true-map certificate
  $+$ bootstrap learned-model horizon $+$ self-check — honestly scoped, still a real result.
- **Stage 3 — validation + stretch.** *Soundness:* $T_{\text{guar}}(\epsilon)\le T_{\text{true}}(\epsilon)$ on
  **100%** of held-out $(\epsilon,\text{seed})$ on the TRUE system. *Tightness:* report the conservatism ratio
  $T_{\text{guar}}/T_{\text{true}}$ (a sound-but-uselessly-short bound is flagged, not hidden). *Stretch:* Lorenz,
  Rössler, $40$-D Lorenz-96 — **honest abstention reframed as a feature**: the cone-certificate succeeds under
  verifiable uniform hyperbolicity and *correctly abstains* where structure breaks (Lorenz is singular-hyperbolic —
  no uniform cone near the origin), tracking Prop 7's scope theorem *from the learned model's own geometry*. A
  certificate that knows its validity domain is stronger than one that pretends to work everywhere.

## 5. Components & interfaces

- `experiments/step82_certified_horizon_from_model.py`
  - `henon_map(s)`, `henon_jac(s)` — true Hénon map + exact Jacobian (+ exact Lipschitz constant of the Jacobian).
  - `learned_jacobian(model, z)` — autograd Jacobian of a learned $\Delta t$-map (reuse the `step70`/`step71` learned
    chaotic models; train if not cached).
  - `adapted_metric(jacs, succ_jacs, mode='constant')` — solve for $(P,\Lambda)$ minimizing $\Lambda$ s.t. (LMI)
    margin on samples (constant $P$ first; smooth field optional). Returns $P,\Lambda,\text{margin}$.
  - `lipschitz_bridge(margin, L_J, L_P, h)` — returns `certified: bool` + the certified $\Lambda^{\text{cert}}$.
  - `t_guar(Lambda, kappa, eps, eps_res)` — $T_{\text{guar}}$ per Theorem B′.
  - `true_horizon(true_map, eps, eps_res, n_starts, seed)` — empirical first-crossing horizon on the TRUE system
    (the soundness ground truth).
  - `bootstrap_fallback(jacs, ...)` — reuse `step78.bootstrap_spectrum_ci` + `horizon_interval` on the same point
    cloud (the hybrid backstop; only invoked if Stage 2 is INCONCLUSIVE).
  - `run()` — Stage 1→2→3, prints a per-system table, writes `papers/figures/step82_certified_horizon.{json,png}`.
- `tests/test_step82.py` — fast, no training:
  - true-Hénon Jacobian matches finite differences (1e-7);
  - the LMI is verified on a constructed $(P,\Lambda)$ for a toy linear map (analytic check $\Lambda=\sigma_{\max}$);
  - `t_guar` monotone in $\epsilon$ and matches the closed form;
  - soundness checker flags a deliberately too-long horizon as a violation;
  - the Lipschitz-bridge slack inequality (margin vs $L_J\,h$) is enforced (a sub-margin case returns
    `certified=False`).

Reuse, do **not** modify: `step70` (Lorenz), `step71` (Hénon/Rössler/Lorenz multi-chaos), `step74` (Lorenz-96),
`step78` (bootstrap CI + `horizon_interval`).

## 6. Honest gates (never loosen — INCONCLUSIVE instead)

- **G1 (certificate exists):** Lipschitz-bridged (LMI) margin $>0$ on the covered set. Else "no certificate" for that
  system. **Stage-1 true-Hénon must pass G1 for B to count**; everything else may abstain.
- **G2 (soundness):** $T_{\text{guar}}(\epsilon)\le T_{\text{true}}(\epsilon)$ on **100%** of $(\epsilon,\text{seed})$.
  Any violation is a real bug → report, do not loosen.
- **G3 (tightness, reported not gated):** conservatism ratio $T_{\text{guar}}/T_{\text{true}}\in(0,1]$; a ratio
  $\ll 1$ is honestly flagged as "sound but loose."

## 7. Where it lands

- **Appendix B:** Theorem B′ + proof + the verification algorithm (Algorithm-style).
- **ICLR:** a short experiment paragraph ("a certified horizon *from the learned model* — Hénon"), which makes the
  title's "Certified" literal; respects the ≤9pp gate (relocate any figure to Appendix A if needed).
- **paper2:** a full section with the certificate, the soundness/tightness table, and the abstention story.
- **Enables Lever A:** rewrite the abstract's "matched to $1$–$12\%$ (measured)" into "certified from the learned
  model," honestly scoped.

## 8. Risks & honest probabilities

- True-Hénon rigorous certificate (Stage 1): **~0.7** (clean, exact Lipschitz).
- Learned-Hénon bridge closes tightly (Stage 2, pure (a) success): **~0.55** (neural-net Lipschitz bound is the
  fragile step). If it fails → hybrid fallback, which is **~0.9** to land (bootstrap + self-check is robust).
- Lorenz / Rössler / L96 (Stage 3 stretch): **~0.3** each — abstention reframed as the validity-domain feature, so
  the *paper outcome is robust* even if only Hénon certifies.

## 9. Out of scope (YAGNI)

- No new chaotic systems beyond {Hénon (lead), Lorenz, Rössler, L96 (stretch)} — all already in the codebase.
- No attempt at an unconditional true-system a-priori guarantee (impossible; §1).
- No control / planning loop here — this is the *certificate*; its actionability is Lever C's separate concern.

## 10. Revision 2026-06-08b — cat-map tight pivot (after Phase A)

Phase A landed G1 on the true Hénon map (sound, non-vacuous, beats Euclidean) but the constant-$P$ certified
exponent is **3.15× the true** $\lambda_1$ (certified $\log\Lambda^{\text{cert}}{=}1.32$ vs true $0.42$; $T_{\text{guar}}{=}3$
vs true $\sim\!11$). Working out *why* surfaced a real obstruction, not an effort gap:

- Hénon ($a{=}1.4$) is **non-uniformly hyperbolic** (homoclinic tangencies; Benedicks–Carleson) → no global uniform
  cone field → a smooth-field $P(z)$ is obstructed near tangencies.
- Single-step metrics (constant *or* field) cap at the single-step operator-norm scale; the only lever that beats it is
  $k$-step Jacobian products, whose continuum-bridge Lipschitz grows $\sim M^{2k}$ ($M=\sup\lVert D\phi\rVert\approx3.7$)
  → bridge slack vacuous for $k\ge3$. **So ~3× is near the rigorous continuum-certificate limit *on Hénon*.**

**Pivot.** Tight certification belongs to **uniformly hyperbolic** dynamics. New tight lead = **(perturbed) cat map /
hyperbolic toral automorphism**: $A=\bigl(\begin{smallmatrix}2&1\\1&1\end{smallmatrix}\bigr)$ is *symmetric* ⇒ $P{=}I$
gives $\Lambda=\rho(A)=\tfrac{3+\sqrt5}{2}=e^{\lambda_1}$ **exactly**, with $L_J{=}0$ (constant Jacobian) ⇒ trivial bridge
— **tight by construction**; a small smooth perturbation stays Anosov (structural stability), is genuinely nonlinear,
and has small $L_J$ ⇒ the bridge still closes. **Hénon is retained as the honest "sound-but-conservative on
non-uniformly-hyperbolic" companion.**

The combined result is *richer* than forcing Hénon tight (which is mathematically obstructed): the certificate is
**provably tight on uniformly-hyperbolic dynamics** (cat map: certified $=$ true exponent to machine precision),
**soundly conservative on non-uniformly-hyperbolic dynamics** (Hénon: sound, ~3× by the genuine worst-case-vs-typical
margin), and **knows which regime it is in** — a *tightness dimension* on Proposition 7's scope theorem.

**New tasks (Phase A′, extend `step82`, reuse `adapted_metric`/`lipschitz_bridge`/`t_guar`):** `cat_map`/`cat_jac`
(+ analytic $\lambda_1=\log\tfrac{3+\sqrt5}{2}$); `perturbed_cat_map`/`perturbed_cat_jac` (+ analytic $L_J$ from the
perturbation's second derivative + a global uniform-hyperbolicity cone-margin check + Benettin $\lambda_1>0$);
`run_true_catmap` / `run_true_perturbed_catmap` (certified exponent $\approx\lambda_1$, ratio $\to 1$); the
tightness comparison cat-map (~1.0×) vs Hénon (~3.15×). The pure linear cat map is the guaranteed-tight anchor
(tight by construction); the perturbation is the nonlinear upgrade — fall back to the linear anchor if the
perturbation is not cleanly Anosov. Then Phase B (learned cat map) + Phase C (validation) proceed as before.
