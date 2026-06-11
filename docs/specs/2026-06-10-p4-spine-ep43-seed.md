# E-P4.3 spine — certified subgoal spacing vs measured feasibility boundary — seed spec

*Date: 2026-06-10 · Status: seed (operationalization FROZEN here, before any run — the proposal's
own requirement) · Owner: paper3 · Compute: Mac CPU/MPS + minutes-scale audits*

> The spearhead experiment (claim C3). Post-step2 reality: regimes = **PushT-static
> ($\hat\lambda_1^{\text{env}} \approx 0.006$, neutral) vs PushT-dyn($\kappa{=}0.8$)
> ($\approx 0.088$, expansive)** — the registered two-point contrast. Bases: eq + plain-match
> (anti-explosion; aug stays probes/audit-only). Instruments: gap mode (step3, certified 3/3),
> short-window subgoal-conditioned CEM (to build), $G$ trainer (to build).

## 0. Which loop does the certificate audit? (the design ambiguity, resolved)

The hierarchical planner has **two model loops** with separate spectra: the action-conditioned
low-level $f$ (its error compounds *within* the gap window) and the autonomous subgoal flow $G$
(its error compounds *across* subgoals). The proposal's $\widehat{\mathrm{Err}}(H)$ formula is
hereby pinned to **two curves, each consumed where it binds** (Prop 11 discipline):

- $\widehat{\mathrm{Err}}_f(H)$ — gap-mode audit of $f$ (step3 instrument, as built);
- $\widehat{\mathrm{Err}}_G(m)$ — gap-mode audit of $G$'s autonomous flow at subgoal stride
  granularity ($G$ IS the FF-JEPA latent-flow object; auditing it is the original competitive-intel
  hook, step89/91 machinery).

## 1. $G$ design (registered)

Native stride $h_0 = 5$ env steps (LeWM CHUNK convention); gap $H = 5m$ ⇒ the fixed-$H$ grid
$\{5,10,15,25,40\}$ = rollouts $m \in \{1,2,3,5,8\}$ (FF-JEPA's $H{=}25$ = $m{=}5$). Trained on
oracle-demo latent sequences (200 successful episodes per regime). Per base: plain $G$ = residual
MLP $[2D\to512\to512\to D]$ on $K{=}2$ past subgoal latents; **eq $G$** = the same `GroupConv1x1`
stack on regular fibers **without the action lift** (autonomous) — P4.C's group×invariant
decomposition by construction, with its own exact-equivariance unit test (repo rule).

## 2. $H_{\max}(\epsilon)$ operationalization (FROZEN)

All latent distances in the base's own latent metric; $\epsilon$ grid **dimensionless**:
$\epsilon \in \{2,4,8,16\} \times \hat\delta_f^{\text{mean}}$ per (base, regime) — comparable
across bases without metric games. Held-out demo windows throughout.

- **Model-side boundary** (the certificate's aligned consumer):
  $\mathrm{Err}_f^{\text{meas}}(H) = \lVert f^{(H)}(z_t, a_{t:t+H}) - z_{t+H} \rVert$ over held-out
  windows; $H^{\text{model}}_{\max}(\epsilon) = \max\{H: \text{median } \mathrm{Err}_f^{\text{meas}}(H) \le \epsilon\}$.
- **Planner-side boundary**: subgoal-conditioned CEM (swm solver, latent cost
  $\lVert \hat z_{t+H} - z^{\text{sg}} \rVert$) given **ground-truth** subgoals
  $z^{\text{sg}} = z_{t+H}$; executed in the TRUE env; window success = encoded reached state
  within $\epsilon_{\text{reach}}$ of the subgoal; $H^{\text{plan}}_{\max}(\epsilon_{\text{reach}})
  = \max\{H: \text{reach-rate} \ge 0.5\}$.
- **Crossover (registered descriptive readout):** the region where
  $H^{\text{plan}}_{\max} < H^{\text{model}}_{\max}$ = optimizer-limited; below = model-limited.
- **$\hat\theta^\ast$ elicitation (consumed by C4's certified-$H$ cells, Prop 11(ii)):** at fixed
  $H{=}15$, inject controlled noise $\nu$ into GT subgoals; $\hat\theta^\ast = \max\{\nu:
  \text{reach-rate} \ge 0.5\}$.
- **Stage 2 ($G$ in the loop):** repeat the planner-side boundary with $G$-proposed subgoals; the
  Stage-1$\to$Stage-2 degradation is attributed to $G$ and read against
  $\widehat{\mathrm{Err}}_G(m)$.

## 3. Pre-registered gates (C3; bands $[0.5, 2]$, zero loosening)

- **C3-cal:** $H^*(\epsilon)$ from $\widehat{\mathrm{Err}}_f$ (q90 curve = the certificate-flavoured
  bound; **the q90 curve is the registered consumer**, the mean curve is reported context) vs
  $H^{\text{model}}_{\max}(\epsilon)$: ratio in band across the $\epsilon$ grid, both regimes,
  $\ge 2/3$ seeds, **within the model-limited region** (the crossover locates it, per the
  proposal's scope-law hygiene).
- **C3-wedge:** eq's in-wedge-measured curve transfers out-of-wedge (ratio in band; the
  $m\,\hat\epsilon_{\max}$ term included); plain's does not (out of band). Wedge = collection
  episodes filtered to initial block orientation $\in [0°, 90°)$; out-of-wedge = the remaining
  orbit.
- **C3-regime (the two-point law):** $H^{\text{model}}_{\max}(\epsilon)$ at $\kappa{=}0.8$ is
  predicted by the κ=0.8 certificate, and the static-vs-dyn boundary *ordering* matches the
  certificate ordering at every $\epsilon$ — the surviving form of the signature figure.
- $G$-audit and Stage-2 readouts: registered as **reported** (descriptive), not gated — first
  contact with $G$'s spectrum is exploratory; gates on $G$ would be invented post hoc.

## 4. Build queue (cut order)

1. $f^{(H)}$ rollout evaluator + $\mathrm{Err}_f^{\text{meas}}(H)$ (trivial, uses step1 ckpts).
2. Subgoal-conditioned short-window CEM on swm's solver (latent cost; replan-free single window).
3. $G$ trainer (+ eq-$G$ equivariance test) on κ=0 oracle demos.
4. **κ=0.8 lane:** step1-pipeline rerun at κ=0.8 (WeakPolicy corpus regenerates; checkpoints now
   saved) + **extended-state oracle** (the registered 10-d debt falls due HERE — `set_ext` from
   step2 reused in the oracle's sim clones).
5. Stage-1 boundary measurements both regimes → C3-cal + crossover + $\hat\theta^\ast$.
6. Wedge lane (wedge-filtered collection + eq/plain wedge bases) → C3-wedge.
7. $G$ audit + Stage 2.

## 5. Honest notes

Neutral-regime certificate is linear-budget arithmetic (the exponential content lives in the
κ=0.8 cells — exactly why step2's expansive point is load-bearing). $\epsilon_{\text{reach}}$ and
$\epsilon$ share the latent metric but answer different questions (model trust vs task closure);
they are never conflated in a gate. The known instrument bias ($\approx -0.007$ at gap 0.1,
G-I-measured) is small against the regime contrast ($\Delta\lambda \approx 0.08$) — stated when
the regime gate is read.

## 6. G-pre — linearization-neighborhood diagnostic (registered 2026-06-10, BEFORE any spine run; forced by paper2's E16/step99)

paper2's step99 (same day, V-JEPA 2-AC 1B on real DROID data) just demonstrated the failure mode
this gate guards against: **tangent-space $\lambda_1$ over-promises when trajectories never enter
the linearization neighborhood** — measured deployment error started at the representation-native
step scale with slope $0.033 \ll \lambda_1 = 0.178$; the measured column overturned the
tangent-space numbers. Our exposure is narrower by construction — $\hat\delta_f$ is already a
*measured finite-amplitude* one-step bias (the hybrid step99 recommends on the $\delta$ side) and
the $\epsilon$ grid is already representation-relative (dimensionless $\hat\delta$ multiples) —
but $\hat\lambda_1$ (JVP) remains a tangent quantity evaluated at amplitude $\hat\delta$.

**Diagnostic (runs with the first $\mathrm{Err}_f^{\text{meas}}(H)$ curves, before C3-cal is
read):** fit the finite-amplitude growth rate $\hat\lambda_1^{\text{meas}}$ from the measured
median curve over the model-limited region; report $r = \hat\lambda_1^{\text{meas}} /
\hat\lambda_1^{\text{tangent}}$ per (base, regime).

- $r \in [0.5, 2]$ → the tangent certificate is in jurisdiction; C3-cal proceeds as registered.
- $r$ outside → **FAILED-BY-SCOPE**: C3-cal is reported with the diagnostic attached (the Prop
  7-flavoured scope condition, now amplitude-indexed), and the registered fallback claim is the
  *scoped certificate* (valid where amplitude stays measured-consistent). This outcome is a
  finding, not a burial — paper2's E16 just published the foundation-model side of it.

**Cross-paper synergy (registered observation):** the same instrument now has jurisdiction data at
two scales — paper3's small structured bases vs paper2's 1B foundation models. If small
equivariant models stay in tangent jurisdiction where flagships don't, that contrast is itself a
result the two papers can cite each other for.
