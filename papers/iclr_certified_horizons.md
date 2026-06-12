---
title: "Scale Buys Interpolation, Structure Buys a Horizon: Certified Predictability for Equivariant World Models"
subtitle: "How far a world model can promise — tightly, and only with structure."
author: "Hongbo Wang"
created: 2026-06-05
venue: "ICLR submission draft (focused extraction of the Certified World Models line)"
---

## Abstract

*Scale buys interpolation; structure buys a certified horizon.* A world model's average error says nothing about whether a *particular* prediction can be trusted, or for how long. For **equivariant** latent world models we give a computable, multi-step **certificate of the predictable horizon**: $T$-step rollout error is provably constant over each symmetry orbit (Theorem A) and stratified channel-by-channel by the predictor's Lyapunov spectrum, $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$. The horizon is **two-sided** — a matching lower bound makes approximate equivariance provably horizon-limited (Proposition 6) — and the certificate is **exclusive to structure**: orbit-constant error *characterizes* equivariance (Lemma 2), so no non-equivariant model has it at any scale. Empirically, on $40$-D Lorenz-96 only a $\mathbb{Z}_N$-equivariant network recovers the full Lyapunov spectrum ($R^2{=}0.98$); identically-trained dense and recurrent baselines fail.

Because the spectrum is faithful, the certificate **acts, a priori**: under a fixed sensing budget a $c\times$-inflated certificate provably needs $c\times$ the budget (Proposition 9), and the equivariant certificate meets a budget its inflated dense counterpart cannot — with zero calibration data. The same read-out, unchanged, **audits public pretrained world models training-free**: an **$84$-cell audit of the public TD-MPC2 zoo** lands on the certificate's own scope taxonomy — abstaining on exactly half, calibrated **where the measured horizon is growth-set** (median ratio $0.94$, $8/15$ in $[2/3,3/2]$), optimistic where native one-step bias sets it first — and a deployed sensing-budget monitor replicates the seed map **cell-by-cell, out-of-sample, at zero new estimation**. Across the official $1$M–$317$M multitask ladder, calibration does **not** improve with parameters. And on V-JEPA 2-AC ($1$B, real robot data) the measured cross-check correctly **overrides** an over-promising tangent spectrum — **the cross-validated audit, not the raw number, is the deployable object**. Scale buys interpolation, not a calibrated horizon.

---

## 1. Introduction

A world model is judged by its average prediction error. But an acting agent needs to know, for the situation in
front of it, **whether the model can be trusted, and for how many steps.** Scaling lowers the average; it certifies
nothing about a *particular* configuration and says nothing about the *horizon* — the steps before compounding error
makes the rollout useless.

Equivariance gives *some* per-situation guarantee: a network commuting with a group $G$ behaves identically across
each orbit — used to certify adversarial robustness and tighten conformal sets. But every such result is
**single-shot**: one input, one prediction. A world model predicts a *trajectory*, and the question that matters for
planning — *how far ahead is the guarantee valid?* — has no answer in that literature.

This paper answers it. We show that an equivariant latent world model admits a **certificate of its predictable
horizon**, stratified channel-by-channel by the predictor's spectrum, and we prove the horizon is **tight**. The
contributions are:

1. **A tight certified horizon (§3.2).** For an equivariant model the $T$-step rollout error is orbit-constant
   (Theorem A); under approximate equivariance, channel $j$ is certified to $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$
   (Theorem B). The central new result is a **matching lower bound** (Proposition 6): an
   $\epsilon$-approximately-equivariant model — even with a *perfect* predictor — has orbit-error-variation exactly
   $\epsilon\,e^{\lambda T}$, so the horizon is $\Theta(\log(1/\epsilon)/\lambda)$, **no better**: *approximate
   equivariance is horizon-limited; only exactness or conservation reaches an unbounded horizon.* A scope theorem
   (Proposition 7) says *when* the measured spectrum governs a *learned* model's horizon, a continuity bound
   (Proposition 8) makes the lift rigorous, and the staircase lifts to learned models of genuinely chaotic systems
   (E2: exponents matched to $1$–$12\%$), the near-neutral PushT interior being the predicted degenerate case.
2. **The certificate is exclusive to structure (§3.1).** The converse (Lemma 2): orbit-constant error against every
   equivariant target $\iff$ equivariance — **no non-equivariant model has the certificate at any size**, an
   architectural impossibility, proved, not asserted.
3. **The Noether hinge (§3.3).** Conserved/invariant channels are certified to **all horizons**: Proposition 4
   forces each charge's isotypic placement, and Proposition 5 turns conservation into a long-horizon guarantee
   ($T$-step charge error $\le T\eta$ — linear, never $e^{\lambda T}$). The converse is false and stated (a slow
   channel need not be conserved): **the one direction proved is the one the certificate uses.**
4. **Single-shot certified equivariance is our $T{=}1$ slice (§5)** — the one-step, resolution-free special case;
   the multi-step picture is validated on a contact simulator, non-abelian $\mathrm{SO}(3)$, and raw pixels (frame
   averaging keeping the certificate *accuracy-neutral*).

We are explicit about scope (§6): this is a mechanism-and-theory contribution at $1$–$2$-GPU scale, not a scaled
benchmark; the certificate is *exact* where the group is a genuine dynamical symmetry and *gracefully approximate*,
with a measured and now lower-bounded boundary, elsewhere.

**One certificate, a universal half and an exclusive half.** The spectral audit (Theorem B's law) applies to **any**
$C^1$ latent loop — that is how E13–E16 audit non-equivariant public models; what is **exclusive to structure** is
*a-priori trust in the audited number itself* (Lemma 2; E2: a dense spectrum can be silently wrong at one-step relMSE
$10^{-5}$). A generic model buys that trust with held-out divergence data; an equivariant model has it from the
Jacobian alone — and E16 is what that purchase looks like at $1$B: the cross-check, not the spectrum, carries the
verdict. §4's closing paragraph returns to this division.

**What is new.** A prior-art sweep finds the corner *empty*: no work combines equivariance with a certified, two-sided spectral horizon; the closest guarantee-bearing neighbours each miss the intersection [@conradie2026trustkoopman; @lillemark2026flowm; @mo2026symmetry] (§5). The new results: **(i)** Proposition 6's matching lower bound — growth seeded by the *equivariance residual* rather than an initial-condition error — making approximate equivariance provably horizon-limited; **(ii)** the scope/continuity pair (Props. 7–8) lifting the law to *learned* chaotic models, where shadowing cannot transfer the exponent; **(iii)** the structure-vs-recurrence separation at $40$-D (E2), with the recurrent failure forced by the conditional-Lyapunov condition; **(iv)** Theorem B′ turning the certificate from *measured* into *literal* — sound a priori, tight on uniform hyperbolicity, self-abstaining elsewhere; **(v)** actionability **a priori, with zero calibration data** (E12; Proposition 9); and **(vi)** a training-free trustworthiness audit of public pretrained world models — the TD-MPC2 zoo at **$84$ cells**, LeWM, V-JEPA 2-AC at $1$B (E13–E16) — whose $84$-cell expansion **overturns the seed map's own axis** (calibration tracks growth-set vs.\ bias-set horizons, not $\lambda_1$) and that a deployed monitor replicates **cell-by-cell, out-of-sample, every taxonomy cell type** (E15), its decision boundary formalized as a regret decomposition (Proposition 11). These assemble (Algorithm 1) into one certificate *exclusive to structure* (Lemma 2) whose unbounded-horizon edge is anchored by the conserved/invariant subspace (the Noether hinge). We credit the classical pillars we build on — the
Lyapunov/NWP horizon law, invariant decision theory, and the $e^{\lambda T}$ growth — explicitly in §3 and §5.

![The certificate at a glance. **Left:** an equivariant model certifies the *entire* generated monoid $\langle S\rangle$ from $k$ generator checks (Lemma 1), up to a horizon ceiling set by the predictor spectrum (Theorem B, tight by Proposition 6); a non-equivariant model of any size certifies only an interpolation tube. **Right:** the horizon-resolution law $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ — conserved/invariant channels ($\lambda\le0$) are certified to all horizons, chaotic ones shrink with demanded resolution.](figures/hero_certified_region.png){width=55%}

---


![**The paper in one figure.** **(a) Faithful:** on $40$-D Lorenz-96 the $\mathbb{Z}_N$-equivariant model recovers the full Lyapunov spectrum ($R^2{=}0.98$) where an identically-trained dense model's is garbage ($R^2{<}0$) — E2. **(b) Priced:** under a fixed sensing budget the faithful certificate meets the budget; the inflated one over-observes and starves it (Proposition 9) — E12. **(c) Real:** the same training-free read-out audits the official TD-MPC2 zoo ($84$ cells) — calibrated where the horizon is growth-set, abstaining on half — E13.](figures/hero_certified_world_models.png){width=58%}

## 2. Setup and assumptions

A latent world model is an encoder $E:\mathcal X\to\mathcal Z$ and a predictor $f:\mathcal Z\times\mathcal A\to
\mathcal Z$ with $f(E(x),a)\approx E(\Phi(x,a))$, $\Phi$ the (unknown) environment dynamics; rolling out
$\hat z_T=f(\cdot,a_{T-1})\circ\cdots\circ f(E(x),a_0)$, the $T$-step error is
$\mathrm{Err}_T(x)=\lVert \hat z_T - E(\Phi^T x)\rVert$. A group $G$ acts on $\mathcal X$, $\mathcal Z$
(via a representation $\rho$), and $\mathcal A$ (via $\sigma$). We use:

- **(A1) Encoder equivariance:** $E(g\cdot x)=\rho(g)\,E(x)$.
- **(A2) Predictor equivariance:** $f(\rho(g)z,\sigma(g)a)=\rho(g)\,f(z,a)$.
- **(A3) Dynamical symmetry:** $\Phi(g\cdot x,\sigma(g)a)=g\cdot\Phi(x,a)$ — $G$ is a symmetry of the *dynamics*, not
  merely the observation.
- **(A4) Orthogonality:** $\rho(g)^\top\rho(g)=I$ ($\rho$ preserves the error norm).
- **(A5) Equivariant planner, invariant cost** (closed-loop clause only).

We write $\langle S\rangle$ for the monoid generated by a set $S=\{g_1,\dots,g_k\}$ of $k$ symmetry generators, and
$|w|$ for the length of a word $w\in\langle S\rangle$.

---

## 3. The certificate

### 3.1 The configuration axis, and why only structure has it

**Theorem A (orbit-constant error).** Under (A1)–(A4), for every $w\in\langle S\rangle$, all $x$, and any action
sequence (transported by $w$), $\mathrm{Err}_T(w\cdot x)=\mathrm{Err}_T(x)$ (action-explicit statement and proof in Appendix B). *Proof sketch.* The rolled-out predictor is equivariant
(Lemma 1), so prediction and target transport by the same $\rho(w)$; subtract and use (A4). $\square$

**Lemma 1 (composition closure).** If (A1)–(A3) hold on each generator $g_i\in S$, they hold on every word
$w\in\langle S\rangle$ ((A4) is automatic for all $w$). Thus **$k$ generator checks certify an exponentially large
set** — all of $\langle S\rangle$.

**Lemma 2 (the certificate characterizes equivariance — the converse).** Let $\rho:G\to O(\mathcal Z)$ act *freely* on
an open $U\subseteq\mathcal Z$. If a predictor $f$'s error $\lVert f-\Phi\rVert$ is orbit-constant on $U$ for *every*
equivariant target $\Phi$, then $f$ is equivariant on $U$. *Proof sketch (full proof and the continuity refinement in Appendix B).* Freeness makes $\Phi(\rho(h)z):=\rho(h)c$ a
well-defined equivariant probe target for any $c$; orbit-constancy then gives
$\lVert f(z)-c\rVert=\lVert f(\rho(g)z)-\rho(g)c\rVert=\lVert\rho(g)^{-1}f(\rho(g)z)-c\rVert$ (the last step
by (A4)) for *every* $c$ — two points equidistant from every $c$ coincide, so $f(\rho(g)z)=\rho(g)f(z)$. $\square$ With Theorem A this is a **characterization**: orbit-constant
error $\iff$ equivariance — **no non-equivariant model possesses the certificate at any size**; the impossibility is
a theorem, not an observation.

### 3.2 The horizon axis, and its tightness (the central result)

Real models are only *approximately* equivariant. Let $\epsilon_{\max}=\max_i\sup_x\lVert
E(g_i\cdot x)-\rho(g_i)E(x)\rVert$ be the encoder residual, $\delta$ the per-step predictor error, and let the latent
Jacobian be locally diagonalized on channel $j$ with multiplier $e^{\lambda_j}$.

**Theorem B (spectral degradation — upper bound).** For constants $c_j$ depending on local geometry,
$$
\bigl\lvert \mathrm{Err}_T(w\cdot x)-\mathrm{Err}_T(x)\bigr\rvert \;\le\; \sum_{\text{channels }j} c_j\,(m\,\epsilon_{\max}+T\,\delta)\,e^{\lambda_j T},\qquad m=|w|,
$$
so channel $j$ is certified only to horizon $T_j(\epsilon)\sim\tfrac1{\lambda_j}\log\tfrac1\epsilon$ ($\lambda_j>0$),
or $T_j=\infty$ ($\lambda_j\le0$). As $\epsilon_{\max},\delta\to0$ the configuration term vanishes and Theorem A is
recovered.

**Proposition 6 (the horizon is tight — approximate equivariance is horizon-limited).** Theorem B is an *upper* bound;
here is a matching *lower* bound. Fix an expansive channel on which the latent map is locally linear with multiplier
$a=e^{\lambda}$, $\lambda>0$ (the local diagonalization of Theorem B). There exist an
exactly equivariant target $\Phi$ and a model that is $\epsilon$-approximately equivariant — a *perfect* equivariant
predictor $f=\Phi$ ($\delta=0$) and an encoder that intertwines the dynamics along $x$'s trajectory and differs from
exact equivariance only by a single defect $E(g\cdot x)=\rho(g)E(x)+\epsilon u$ at one orbit point — for which
$$
\bigl\lvert \mathrm{Err}_T(g\cdot x)-\mathrm{Err}_T(x)\bigr\rvert \;=\; \epsilon\,e^{\lambda T}.
$$
*Proof sketch (Appendix B).* Along $x$ the model is exact, $\mathrm{Err}_T(x)=0$; at $g\cdot x$ the single defect
propagates linearly, $f^T(E(g\cdot x))=\rho(g)\Phi^T(Ex)+a^T\epsilon u$, while the target is $\rho(g)\Phi^T(Ex)$;
subtract and use (A4): $\epsilon e^{\lambda T}$. $\square$

Hence the certified horizon $T(\epsilon_{\mathrm{res}})=\tfrac1\lambda\log\tfrac{\epsilon_{\mathrm{res}}}{\epsilon}=
\Theta\!\big(\tfrac1\lambda\log\tfrac1\epsilon\big)$, matching Theorem B: **the horizon's form is two-sided, not merely
an upper estimate.** The conceptual payload is sharp:

> **No certificate derived from an $\epsilon>0$ equivariance residual can promise predictability beyond
> $T\sim\tfrac1\lambda\log\tfrac1\epsilon$ on an expansive channel (worst case over admissible targets). Only
> exactness ($\epsilon=0$) or conservation ($\lambda_j\le0$) yields an unbounded horizon.**

This is the horizon-domain companion of Lemma 2: scale and data buy *approximate* equivariance at best (a fair
augmentation baseline floors at $\epsilon\approx10^{-4}$, never exact), and Proposition 6 amplifies that residual
$e^{\lambda T}$ — a single-step tie between augmentation and equivariance **must break over horizon**.

**The prefactor is the splitting conditioning (Proposition 6′, Appendix B):** $c_j=1/\sin\theta_j$ — exactly $1$ on orthogonal invariant splittings
(isotypic blocks — Schur placement and zero leakage at machine precision), attained to four digits under
controlled shears with the predicted $\log\kappa/\lambda$ haircut; on real loops it is *computable from the
audited Jacobian field* — honestly a distribution (median window-dependent, $\kappa_1\!\sim\!19$–$49$; heavy tail) — while measured
calibration stays $0.83$–$1.02$ (`step65b`, `step95`).

![Proposition 6, numerically (the central claim). **Left:** orbit-error-variation of an $\epsilon{=}10^{-3}$-approximately-equivariant model (markers) equals the analytic $\epsilon\,e^{\lambda T}$ (dashed) to relative error $10^{-14}$–$10^{-13}$, $3$ seeds; an *exactly* equivariant model sits at the machine-precision floor; $\lambda\le0$ channels stay bounded (infinite horizon). **Right:** certified horizon linear in $\log(1/\epsilon)$ with slope exactly $1/\lambda$ ($R^2{=}1.000$) — the $\Theta(\log(1/\epsilon)/\lambda)$ law (`step65`).](figures/step65_horizon_tightness.png){width=65%}

**Proposition 7 (scope — when the local spectrum certifies a *learned* model's horizon).** On a learned model the
answer splits a rigorous **rate** half from an orbit-error **lift** half. Let $\phi$ have an ergodic invariant measure
$\mu$ with $\log^+\lVert D\phi\rVert\in L^1(\mu)$. By **Oseledets** [@oseledets1968],
$\tfrac1t\log\lVert\delta_t\rVert\to\lambda_1$ $\mu$-a.e. for generic perturbations. *(a) Non-degenerate
($\lambda_1>0$):* $T(\epsilon)=\tfrac1{\lambda_1}\log(\epsilon_{\mathrm{res}}/\epsilon)\,(1+o(1))$ — Theorem B's law with
$\lambda_1$ the *measurable* asymptotic rate. For a learned $\hat\phi$, shadowing [@pilyugin1999shadowing] bounds only
the forecast-horizon *floor* $\sim\tfrac1{\lambda_1}\log(1/\delta)$ — trajectory closeness, **not** the asymptotic
exponent — so that $\hat\phi$ *reproduces* $\lambda_1$ is a finite-horizon continuity statement (Proposition 8),
confirmed in E2. *(b) Degenerate ($\lambda_1\approx0$):* the log-law degenerates; the one-step spectrum carries no
horizon rate. This **dichotomy is the certificate's scope** — and it *predicts* where the certificate is vacuous
(E2's PushT-interior probe, $R^2{=}0.02$).

**Proposition 8 (finite-horizon exponent transfer).** The asymptotic exponent is only upper-semicontinuous (why
shadowing cannot transfer it), but the certified horizon is **finite**, and the finite-time exponent
$\lambda_1^{(T)}$ is locally Lipschitz over a finite orbit: a learned $\hat\phi$ with one-step fidelity $\delta$
recovers the staircase slope up to $O(\delta)$ plus finite-$T$ truncation, $T$-uniform under a dominated splitting
[@bochi2005lyapunov]. Falsifiable — the exponent must tighten as one-step error drops — confirmed in E2
(Rössler: $44\%\to8\%$).

### 3.3 The Noether hinge: which channels are unbounded

Theorem B leaves the $\lambda_j\le0$ (unbounded-horizon) channels unidentified; the Noether hinge identifies
which channels are *guaranteed* to sit there: the conserved/invariant ones.

**Proposition 4 (isotypic placement).** A conserved charge of the dynamics must live in a specific isotypic block of
$\rho$: a scalar invariant (energy) in the trivial ($\ell{=}0$) block; a vector charge (angular momentum) in the
$\ell{=}1$ block, recoverable only through the unique degree-2 cross-product equivariant. Placement is *forced* by representation theory [@goodman2009symmetry], not chosen.

**Proposition 5 (conservation $\Rightarrow$ unbounded horizon).** Let $Q:\mathcal Z\to W$ be a charge the model
conserves to one-step defect $\eta$ (i.e. $\lVert Q(f z)-Q(z)\rVert\le\eta$). Then the $T$-step *charge-value*
prediction error satisfies $\lVert Q(\hat z_T)-Q(z_T)\rVert\le T\eta$ — **linear in $T$**, never the chaotic
$e^{\lambda T}$ — and at $\eta{=}0$ the charge value is certified to all horizons. (About the charge value; exact
under an equivariant symplectic discretization; the converse fails — slow need not be conserved.)

Together: **the certified region is the coarse, invariant/conserved, low-composition corner** — reachable only
with structure (Lemma 2).

### 3.4 The certificate is a procedure

The certificate is *computable a priori*. Algorithm 1: (i) check (A1)–(A4) on the $k$ generators to residual
$\epsilon_{\max}$; (ii) estimate the predictor spectrum at the query latent (block-bootstrap CI, Liouville-anchored;
`step78`); (iii) report whether $(w,T,\epsilon)$ lies in the certified region of Theorem B/Proposition 6, escalating
conserved channels via Proposition 5. Proposition 10 gives the finite-sample rate
($n\asymp\log(1/\delta)/\varepsilon^2$; the bootstrap CI brackets $T_1$).

---

## 4. Experiments

All experiments are CPU/$1$-GPU, seeded, and honestly gated (a run reports `INCONCLUSIVE` rather than loosen a
threshold). Proofs: **Appendix B**; reproducibility map (anonymized code, seeds, gates): **Appendix C**; the
supporting suite (E1, E3–E8) is summarized below and written up in **Appendix D**.

**(E2) The horizon staircase — and where structure becomes necessary.** On a controlled latent with a planted spectrum the certified-horizon slope brackets the predicted $1/\lambda$ (chaotic exponent recovered to $0.4\%$). The law lifts to learned models of *real* chaotic dynamics: one-step models of **Lorenz**, **Hénon**, **Rössler** give staircases linear in $\log(1/\epsilon)$ ($R^2{=}0.975$–$0.995$, $3$ seeds) whose slopes recover the textbook exponents to $1$–$12\%$ — Proposition 8's continuity at work, not a shadowing corollary — while the near-neutral **PushT interior** is Proposition 7's predicted *degenerate* branch ($R^2{=}0.02$). At high dimension structure becomes *necessary*: on $40$-D **Lorenz-96** [@lorenz1996predictability] (the Liouville identity $\sum_j\lambda_j{=}-N$ recovered to $0.0\%$ — the estimator is anchored before any learned spectrum is trusted) a $\mathbb{Z}_N$-equivariant cyclic-conv recovers the *full* spectrum ($R^2{=}0.98$–$1.00$, **$10/10$ seeds**; Kaplan–Yorke $\sim\!27$) — hence the per-channel certified horizons — while an identically-trained dense MLP **fails** ($R^2$ median $-1.4$, $10/10$ below zero, $\lambda_1$ inflated $\sim\!3.4\times$) and a GRU-BPTT [@vlachas2020rnn] fails the same way ($10/10$), its joint-state Jacobian breaking the conditional-Lyapunov condition [@hart2024attractor] — **zero seed overlap between the two populations**. Sweeping $N\in\{12,20,28,40\}$ shows a **phase transition**: all three architectures tie through $N{=}28$ (the GRU already intermittent there); only the equivariant one survives $N{=}40$ (`step83`, $n{=}10$).

![Full-spectrum Lyapunov $R^2$ vs. $N$: the $\mathbb{Z}_N$-conv holds while dense MLP and GRU collapse at $N{=}40$ — a phase transition, not a single-$N$ artifact (`step83`).](figures/step83_rsquared_crossover.png){width=56%}

![The certified-horizon law across a class of learned chaotic models (E2). The identical learned-model staircase on a 2D map (Hénon), a small-exponent flow (Rössler), and a large-exponent flow (Lorenz) is linear in $\log(1/\epsilon_0)$ and its slope (blue) recovers the textbook exponent (red dashed). The law is a property of chaotic dynamics, not of Lorenz; the residual bias is decomposed by Proposition 8.](figures/step71_multichaos_horizon.png)

**(E1, E3–E8: the supporting suite — Appendix D.)** $k$ generator checks certify all $2^k$ compositions ($\mathbb{Z}_2^6$: error $\sim10^{-33}$ vs a baseline's $0.59$) (E1). Symmetry breaking degrades the certificate linearly — Proposition 6's predicted crossing — and a fair augmentation baseline is horizon-limited at its $\sim10^{-4}$ floor (E3). An $88\times$-scaled non-equivariant model buys interpolation, never the orbit-floor (E4); on PushT contact dynamics a learned $\mathrm{SO}(2)$-equivariant model is orbit-flat to ratio $1.000$ with no baseline in a $160\times$ sweep reaching its out-of-distribution floor — the impossibility carried by Lemma 2, not E5's magnitude (E5). The certificate lifts to non-abelian $\mathrm{SO}(3)$ and, via frame averaging, accuracy-neutrally to pixels (E6); becomes orbit-invariant closed-loop control (E7); and drives a noise-immune epistemic explorer — $2^7$ compositions certified in $7$ observations where prediction-error curiosity is lured by distractors to $1\%$ (E8).

**(E9) Both axes on one system; the certificate acts.** On controlled Lorenz-96 (exact $\mathbb{Z}_N$, $\lambda_1{\approx}1.8$) an equivariant planner gives orbit-flat control (residual $8\times10^{-16}$) while $T_1(\epsilon)$ drives an active re-observation schedule on the accuracy-observation frontier *untuned* ($2/3$ seeds; tight-$\epsilon$ honestly optimistic per Prop. 8); the pattern lifts to a pendulum ring and a double pendulum (`step79`–`step81`).

**(E10) The certified horizon, made literal — and its exact regime.** E2 *measures* the exponent; Theorem B′
*certifies* it: a computable cone/adapted-metric certificate reads a **sound, a-priori** bound on the model's top
exponent off its Jacobian field alone. On **uniformly-hyperbolic** dynamics it is *tight*: cat-map certified exponent exact on the true map, $1.17\times$ on a *learned* net ($3$ seeds, sound), $1.06\times$ on a nonlinear Anosov perturbation. On **non-uniformly-hyperbolic** Hénon a single metric is provably limited — sound but $\sim\!3\times$ conservative — and the certificate's own cone-margin diagnostic turns negative: it **abstains** rather than over-claim. A tight a-priori horizon is achievable *exactly* in the uniformly-hyperbolic regime, and the certificate **self-diagnoses** its regime (`step82`).

**(E11) On a recognized control benchmark — honest INCONCLUSIVE.** On Acrobot-v1 ($\lambda_1{=}0.094$) the certified horizon *tracks* the measured one (ratio $0.42\to0.93$ as $\epsilon$ coarsens) and *binds*: planning past $\sim\!T_1$ fails outright, and capping depth at $T_1$ solves the task ($67$–$100\%$) — but the return-optimum sits at $\sim T_1/2$, so two pre-registered no-tuning rules are **INCONCLUSIVE** against the best-tuned blind depth (`step84`).

**(E12) The certificate's trustworthiness changes a budgeted decision — a priori.** On $40$-D Lorenz-96 an agent schedules sparse re-observations under a fixed sensing budget; the forecaster is held fixed (both models identically trained, one-step relMSE $\sim10^{-5}$) and *only* the certificate timing re-observation varies: the equivariant certificate yields $3$–$19\%$ aggregate violation at the knee budget (median $10\%$) vs. $53$–$70\%$ (median $63\%$) for the dense certificate, whose $\lambda_1$ is inflated $2.9$–$5.2\times$ and which therefore over-observes and starves the budget (margins $+0.41$–$+0.61$, **$20/20$ seeds**, median $+0.53$, bootstrap $95\%$ CI $[0.48,0.54]$; **Proposition 9** makes the cost a law — $c\times$ inflation needs $c\times$ budget; measured catch-up $2.7$–$3.5\times$). Two controls: a certificate-free adaptive scheduler matches only after $\sim3\times$ the budget (the certificate is an *a-priori* warm-start); a *recalibrated* dense certificate closes the gap ($10/10$ seeds; median margin $+0.52\to+0.06$) but **spends a calibration set** — the equivariant one is correct from the Jacobian with **zero rollout data**. Standard UQ baselines (a $4$-model ensemble, a $10$-rollout conformal quantile) calibrate tighter by spending training or truth access the certificate does not: **no method dominates the accuracy–cost Pareto; the certificate is its only a-priori point** (`step90`). The contrast replicates on a pendulum ring ($N{=}24$, **$15$ seeds**) — and **the condition is now a tested mechanism**: the budget gap appears exactly where the dense $\lambda_1$ inflates ($9/10$ inflated seeds win, $0/5$ faithful seeds win; Fisher one-sided $p{=}0.002$; Spearman $\rho{=}0.96$ between inflation ratio and win margin, flags pre-registered on the first five seeds) — faithfulness left to chance vs. structural. (A budget *allocation* across a chaoticity ensemble did *not* win; reported honestly.) (`step85`/`85b`/`85c`, `step88`, `step90`.)

**(E13) The certificate audits a *public pretrained* world model — training-free.** We rebuild the latent slices of official **TD-MPC2** checkpoints and run the *unchanged* machinery on the policy-prior loop $g(z)=d(z,\tanh\mu_\pi(z))$ — no training, no environment access on the certified side — across **5 tasks $\times$ 3 seeds (15 loops — the seed map)**: a scope map tracking the theory cell-by-cell. Where the loop is **strongly expansive** ($\lambda_1{=}0.25$–$0.30$) the certificate is **calibrated** — measured/certified $0.83$–$1.02$ (walker $0.94/0.95/1.02$; tight-$\epsilon$ optimistic as everywhere, Prop. 8). As expansion **weakens** the certificate turns optimistic (cheetah $0.43/0.50$; hopper-hop $0.13/0.38$ at $\lambda_1{=}0.05$–$0.09$): model bias outpaces Lyapunov amplification — the degeneracy direction Proposition 7 flags. Where the loop **contracts** ($6/15$) the certificate **abstains, correctly in both sub-cases**: finger-spin's stable loops genuinely do not diverge ($15$–$19/20$ starts censored), while acrobot's and hopper-1's residual divergence is bias-driven — outside a Lyapunov certificate's jurisdiction. SimNorm's structural zero-directions appear as a strongly-negative band (reported; the certified scope is the prior loop, not the planner). A Jacobian certificate of a public zoo's *learned latent map*, cross-validated against true-environment divergence and stratified by the certificate's own scope theory, is new (nearest prior: true-environment Lyapunov under RL policies, arXiv:2410.10674). (`step89`.) A second, architecturally disjoint family lands on the same map: the official **LeWM** checkpoint (pixel-input ViT+transformer JEPA, loaded bit-faithfully into the authors' own code) has $\lambda_1{=}0.001$ with CI straddling zero — the certificate **abstains**, and the observed $1$–$2$-step divergence is pure one-step bias, the same sub-case as acrobot. Two families, one read-out, one taxonomy (`step91`).

**The full-zoo expansion ($84$ cells) overturns the seed map's axis (`step89b`).** Re-running the unchanged protocol on **every remaining public single-task dmcontrol checkpoint** (universe fixed by a pre-registered *self-executing* rule — a task enters iff stock dm\_control loads it; $11$ custom variants excluded, listed in the artifact; $69/69$ new cells) gives, as frozen: **$42/84$ abstain** (stable $13$, bias $29$); $42$ expansive, measured/certified median $0.50$ at $\lambda_1\ge0.25$, $0.33$ below. **"Calibrated where strongly expansive" does not survive**: in-band cells span $\lambda_1=0.01$–$0.39$. What predicts calibration is *which quantity sets the measured horizon*: where the native one-step residual already sits at the threshold (median $\le3$ steps — $25/42$: dog, humanoid, quadruped), the certificate prices growth the rollout never exhibits — **E16's mechanism at zoo scale**; where the horizon is growth-set (median $\ge5$: $15$ cells), the certificate is **calibrated** — ratio median $0.94$, $8/15$ in $[2/3,3/2]$, across five domains including two absent from the seed map (cup-catch $1.36$). All statistics at $100$ rollout starts/cell (`step89c`; at the original $20$: $0.95$, $10/15$ — both leavers exit conservatively or to the regime boundary). The re-stratification is descriptive (cuts $\le1/\le2/\le3$: $0.53/0.68/0.90$); the frozen quantities are the protocol and fractions above. One reading from toy to zoo to $1$B: **the spectrum prices error *growth*, the measured column tests error *level*, the audit is their conjunction.**

**(E14) Scale does not rescue trustworthiness (`step92`).** Across the official TD-MPC2 *multitask* ladder (mt30, $1$M–$317$M, same walker-walk task, one official checkpoint per size) the loop's regime flips **non-monotonically** with scale — contracting at $1$M *and* $48$M, expansive at $5$M/$19$M/$317$M — and calibration scatters ($0.37/1.87/1.16$ at $\epsilon{=}0.2$ where expansive; mt80 likewise mixed), **no size matching the single-task $0.94$–$1.02$**. One checkpoint per cell — a descriptive scope-map extension, unambiguous in direction: *trust in a rollout is a property of the loop's dynamics, not of parameter count — scale buys interpolation, not a calibrated horizon.*

![Scale does not buy a calibrated horizon (E14). **(a)** $\lambda_1$ of the walker-walk policy-prior loop across the official multitask ladder: sign-flipping, non-monotone (contracting at $1$M and $48$M). **(b)** Calibration at $\epsilon{=}0.2$: scatter across sizes; no multitask scale reaches the single-task $5$M band ($0.94$–$1.02$, green).](figures/step92_scale_sweep.png)

**(E15) The published certificate prices a deployed monitor, out-of-sample (`step94`).** The scope law's *positive* instance: a **sensor-only monitor** watches cheetah-run under its nominal policy, reads the expensive sensor every $k$ steps, and between reads forecasts with the certified loop $g$ itself (no action telemetry), flagging at a read iff relative error exceeds $\theta{=}0.2$. Certificate numbers are **loaded from the E13 artifact** (issued before this experiment existed); gates frozen before seeds 1–2 ever ran. The in-situ staleness clock replicates the published map **cell-by-cell**: in-situ-vs-bench ratio $0.43$ vs $0.43$ and $0.50$ vs $0.50$ on the out-of-sample cells (optimism *predicted*), $0.67$ vs $0.83$ on the calibrated cell — its $[2/3,3/2]$ check landing $7{\times}10^{-4}$ below the edge — recorded **at-the-edge, not rounded up**. A frozen-actuator fault is then detected at the certificate-derived cadence with recall $1.00$ and median delay $\le k_{\mathrm{op}}$ on $3/3$ seeds; at $n{=}100$ ($1600$ windows/seed) the ratios replicate $n{=}20$ digit-for-digit, at-the-edge cell included. Proposition 11 formalizes both sides: an aligned decision transfers certificate value with **zero new estimation** (clause i); step93's dilution is a resolution mismatch, $H(\theta^{\ast})\approx2$ vs $H(0.2)\approx6$ (clause ii). Honest notes: on walker the deterministic prior is regime-bimodal and clock replication breaks ($0.37$–$0.63$ vs $0.94$–$1.02$, $0/3$ at $n{=}100$; recall $0.9$–$1.0$) — a monitor presumes a nominal regime, **Proposition 7 load-bearing in deployment**.

`step96`–`step97` complete the map: the stable-abstain cells deploy as **free monitoring** ($92$–$94\%$ of windows never cross; recall $1.00$; invalid $4.0$–$5.6\%$ at $n{=}100$, one cell $0.6$pp over the pre-registered $5\%$ line — as-registered, INCONCLUSIVE), the bias-driven cells land inside the $\times1.5$ band of bench (hopper-1: $8.0$ vs $5.5$ — the clock the certificate rightly refused to price) or buy **zero sensing savings** on the architecturally disjoint **pixel family** (LeWM/PushT: usable cadence $1$, alarm channel flooded at every $k\ge2$, fault detection inseparable from drift — all stated before the run), and a further replication cell lands $1.04$ vs $0.95$ (finger-spin-1). **Every taxonomy cell type now has a deployment instance, predicted a-priori — and the taxonomy *orders* deployment value: stable-abstain (free monitoring) $>$ expansive (priced savings) $>$ bias-abstain (do not deploy).**

**(E16) At foundation-model scale the cross-check column is load-bearing (`step98`–`step99`).** **V-JEPA 2-AC** ($1$B encoder, official checkpoint, authors' code; $d{=}360{,}448$) reads **expansive** ($\lambda_1{=}0.178$, two seeds agreeing to $1.8\%$) — yet on $40$ real DROID episodes the deployment error *starts* at the representation's native step motion ($0.63$ one-step vs $0.68$ consecutive-latent distance) and grows at log-slope $0.03\ll\lambda_1$: it never enters the linearization neighborhood, and the pre-registered pricing gate **fails as registered** (measured sub-class: bias — Proposition 7's degeneracy at $1$B scale). A spectrum-only audit would over-promise $T_1{\approx}9$; **the cross-validated audit is the deployable object**. Thresholds are representation-relative; rates price only errors inside the linearization neighborhood (Appendix D).

**What does structure buy, if the read-out audits any smooth model?** The law applies to any $C^1$ latent map — hence E13–E16. What it cannot supply there is *trust in the number*: a dense spectrum can be silently wrong while predictions stay good (E2), so a generic certificate must be cross-validated against held-out divergence — E13's per-model check, **load-bearing by E16** — and by the zoo's bias-dominated majority ($25/42$ expansive cells, E13). Structure removes that requirement where it holds (E2 $\Rightarrow$ E12's zero-calibration action), exclusively so (Lemma 2). **The audit is universal; the a-priori guarantee is structure's.**

---

## 5. Related work

**Single-shot certified equivariance is our $T{=}1$ slice.** Orbit-constant margins [@orbitmargin],
invariance-aware smoothing [@randsmooth], and equivariantized conformal prediction [@ecp] are **single-shot,
forward-only, with no horizon, spectrum, or conservation axis** — the $T{=}1$, $\epsilon$-independent slice of our
certificate; ours adds the multi-step stratification (Theorem B, tight by Proposition 6), the converse (Lemma 2),
and the Noether bridge (Proposition 5).

**Learned conservation laws.** Noether Networks [@noethernet] and Noether's Razor [@noetherrazor] *learn* conserved quantities to improve average prediction; we *certify* — Proposition 5 turns a conserved charge into an a-priori long-horizon guarantee, Proposition 4 forces its isotypic placement. Guarantee versus average accuracy.

**Jacobian-regularized world models.** [@jacobianwm] penalizes the latent-transition Jacobian to damp rollout error — a heuristic. Theorem B is the provable version: read a per-channel certified horizon off the spectrum instead of regularizing toward stability; Proposition 6 prices its decay under approximate symmetry.

**Equivariant predictors and latent-geometry priors.** Cyclic/unitary predictors (BRo-JEPA, arXiv:2606.01372; UWM-JEPA, arXiv:2605.25313) report strong zero-shot transfer; Theorem A explains *why* and Lemma 1 quantifies *how far* (the generated monoid). LeJEPA-style isotropic latent priors target a *distributional* property; ours is a first-order, per-situation guarantee that predicts when a data-discovered anisotropy fails off the orbit.

**Predictability horizons.** The $T(\epsilon)\sim\log(1/\epsilon)/\lambda$ law is classical (Lyapunov; NWP); the
local-to-asymptotic link is Oseledets [@oseledets1968]; shadowing [@pilyugin1999shadowing] bounds a perturbed model's
horizon *floor*, not its exponent. We (i) prove the law *tight* for a learned latent world model and measure it on a
class of genuinely chaotic dynamics ($1$–$12\%$, E2), rigorous via Proposition 8 where shadowing fails; (ii)
characterize *when* it lifts (Proposition 7); (iii) tie the unbounded-horizon subspace to the invariant one (Noether);
(iv) prove the converse (Lemma 2). Spectrum recovery from a learned model is known *conditionally* on contracting
modes [@hart2024attractor] — why the recurrent baseline fails at high $N$; the closest concurrent guarantees
[@conradie2026trustkoopman; @lillemark2026flowm; @mo2026symmetry] each miss the equivariance-certified-horizon
intersection.

---


**Concurrent work (2026, briefly).** The nearest neighbors are symmetry-protected *neutral* Lyapunov modes (Mo, arXiv:2605.03338 — constrains the spectrum's kernel where we certify the horizon; for our discrete $\mathbb{Z}_N$ systems their bound is vacuously zero), conformal rollout bounds (Geng et al., arXiv:2512.08991 — statistical and rollout-hungry where ours is a-priori and training-free), and flow-equivariant world models ([@lillemark2026flowm] — an exactness/closure property, not a quantitative horizon). A Jacobian certificate of a *public* model's latent map, cross-validated against true-environment divergence, is to our knowledge new. **Appendix E** gives the full sweep (through 2026-06) and the per-work distinctions.

## 6. Limitations

- **Exactness needs a genuine dynamical symmetry (A3).** Exact where $G$ is a symmetry of the dynamics, *gracefully
  approximate* elsewhere — the degradation two-sidedly bounded (Theorem B, Proposition 6) and measured (E3).
- **Prefactor: computable and measured — worst-case vs typical disclosed.** $c_j=1/\sin\theta_j$ (Prop. 6′) is
  exactly $1$ on isotypic splittings (measured at machine precision), but on audited chaotic loops the worst-case
  $\kappa_1$ carries a heavy near-tangency tail and a **window-dependent median** (walker $20.9$ at $W{=}120$ vs
  $49.2$ at $W{=}200$, each passing the same stability check — necessary, not sufficient; max $\sim10^2$; at
  $W{=}400$ Lorenz-96 passes its own convergence check, median $11.9$): the measured calibration ($0.83$–$1.02$)
  reflects *typical*, not adversarial, defect alignment — an adversarially-aligned defect could spend the
  $\log\kappa_1/\lambda_1$ haircut.
- **The Noether hinge: forward direction proved, hypotheses measured, emergence open.** Proposition 5 assumes a $G$-invariant
  Hamiltonian latent flow (the defect $\eta$ exact only under an equivariant symplectic discretization); the
  converse fails (slow $\not\Rightarrow$ conserved). It is validated on *constructed* equivariant teachers — E5/E6
  validate *flatness*, not the hinge's *emergence* in a learned model of un-constructed dynamics; that remains open.
- **The horizon law lifts to learned chaotic models and is vacuous on near-neutral dynamics (scope, not a bug).**
  Validated on a synthetic spectrum, a *class* of low-D chaotic systems (exponents to $1$–$12\%$), and a $40$-D
  learned model ($R^2{=}0.98$–$0.99$); informative iff $\lambda_1>0$ — the PushT interior is the predicted degenerate
  branch ($R^2{\approx}0.02$, Prop. 7(b)). Caveats: the lift rests on Proposition 8 (not shadowing), and we verify
  $C^1$-closeness only via one-step $L^2$ error — a real gap in high $N$ (one-step-accurate dense and recurrent
  models mis-estimate the $40$-D spectrum; structure closes it); the real-dynamics evidence is ODEs/maps, not a
  contact simulator.
- **Downstream value is budgeted efficiency, not safety.** The catastrophe-avoidance test was **inconclusive** at
  our scale (escape was re-observation-interval-invariant — control-quality-limited); safety *necessity* is open.

- **Where decision value concentrates — and where it dilutes.** Closing the loop on the real TD-MPC2 agent (faithful
  MPPI replica; cadence-1 anchor mean $986$ over $n{=}10$ episodes, range $977$–$996$ overlapping the official
  $977$–$983$ band), return degrades from replan cadence $k{=}2$ ($90\%$ of anchor, $n{=}10$) — well inside
  $T_1(0.2)\approx5.4$–$6.4$; the control-relevant resolution is finer and sits in the
  certificate's known tight-$\epsilon$ optimistic regime (`step93`). With E11, the honest scope law — value concentrates
  where the decided quantity IS the certified quantity and dilutes behind a task-level map — which **Proposition 11
  makes a theorem**: an aligned decision inherits certificate value at calibration cost alone (zero regret at
  $c{=}1$); a task-mapped one pays an irreducible $|\log(\epsilon/\theta^{\ast})|/\lambda_1$ — and
  $\theta^{\ast}$ is the task's to give, not the certificate's.

- **Scale and modality.** All experiments are $1$–$2$-GPU. Exact flatness transfers across modalities (state,
  $\mathrm{SO}(3)$ point clouds, pixels), but multi-step pixel accuracy is poor for *every* architecture at this
  scale — an architecture-agnostic open problem (the anti-collapse JEPA latent, not equivariance), orthogonal to
  the certificate.

---

## 7. Conclusion

An equivariant world model can certify, a priori, which situations it will handle and **for how many steps** — and the horizon is tight: boundary set by the spectrum, unbounded edge anchored by the conserved subspace, reachable only with structure. Because the horizon is *faithful* it is *actionable* — budgeting sensing (E12), auditing public world models (E13–E16), pricing a deployed monitor (E15). *Scale buys interpolation; structure buys a certified horizon.*

---

## Appendix D — supporting experiments (full write-ups)

**(E1) The configuration axis is exponential.** On a $\mathbb{Z}_2^6$ compositional task (six independent
$180^\circ$ flips), training the equivariance checks on the **6 generators** certifies the model over all **$2^6=64$**
compositions to $\sim10^{-33}$ error, while a non-equivariant baseline degrades from $1.6\!\times\!10^{-5}$ on the
generators to $0.59$ on held-out compositions. Six checks, sixty-four guarantees (Lemma 1).

**(E3) Approximate symmetry validates Proposition 6.** Breaking the world's symmetry by $\epsilon_{\text{world}}$, the
certificate degrades *gracefully and linearly* in $\epsilon_{\text{world}}$ (correlation $0.88$–$0.98$ across seeds)
up to a measured threshold $\epsilon_{\text{world}}\approx0.01$–$0.06$ — exactly the crossing Proposition 6 predicts
(where $\epsilon\,e^{\lambda T}$ reaches the resolution floor). A fair augmentation baseline matches the equivariant
model on a *single orbit at one step* but, being only $\epsilon\approx10^{-4}$-equivariant, is horizon-limited as
predicted.

![Approximate symmetry (E3). **Left:** on a compositional task, exact equivariance certifies machine-exactly while augmentation drives a non-equivariant model only to an $\sim\!10^{-4}$ approximation floor. **Right:** breaking the world's symmetry by $\epsilon_{\rm world}$ degrades the certificate gracefully and *linearly* (the slope Proposition 6 predicts) up to a measured threshold.](figures/step53_approximate_symmetry.png)

**(E4) Structure vs. scale.** The equivariant model is orbit-flat (ratio $1.1$–$1.2$); an $88{\times}$-scaled
non-equivariant model buys in-distribution interpolation ($31$–$166{\times}$ better in-wedge, *beating* the equivariant
model there) but its out-of-wedge error stays $10$–$155{\times}$ above the equivariant floor. Scale buys interpolation;
it never buys the certificate (Lemma 2).

![Structure versus scale (E4). The equivariant model is orbit-flat (a certificate); an $88{\times}$-scaled non-equivariant model buys *in-distribution* interpolation (it beats the equivariant model inside the training wedge) but its error climbs $10$–$155{\times}$ above the equivariant floor *out* of the wedge — scale buys interpolation, not the certificate.](figures/step51_structure_vs_scale.png)

**(E5) The certificate on real contact dynamics.** On PushT — a pymunk physics engine whose square arena makes
$\mathrm{SO}(2)$ scene rotation an exact dynamical symmetry — a *learned* $\mathrm{SO}(2)$-equivariant world model has
$10$-step rollout error **exactly flat over the orbit** (ratio $1.00$, equivariance residual $\sim10^{-7}$) and is
competitive in-distribution ($0.13$–$0.15$ vs. the best baseline $0.14$–$0.19$); **no non-equivariant baseline across a
$160{\times}$ parameter sweep ($1.7\mathrm{k}\to272\mathrm{k}$) reaches the equivariant floor out of distribution.** We
are careful about the *size* of that gap: per baseline it is $\sim3.9{\times}$ at the *smallest* model and
$2.1$–$2.6{\times}$ at the *largest*, i.e. it does **not** grow with scale — the honest control is the strong-baseline
$\sim2\times$. The *force* of E5 is the exactness of the flatness (ratio $1.000$ vs. the baselines' $2$–$3\times$ drift)
and the a-priori certificate, not the OOD-error magnitude (which on one $\mathrm{SO}(2)$ task is modest); the "scale
cannot buy it" claim is carried by Lemma 2 / §3.3, not by E5's gap. The dynamics were not authored by us. (Honest gate
note: the run's load-bearing *flat* and *floor* sub-gates pass on all 3 seeds; the auxiliary "MLP error climbs with
horizon faster than equivariant" sub-gate is met on 1/3 seeds and reported `INCONCLUSIVE` on the others.)

![The certificate on real contact dynamics (E5). On PushT (a physics engine we did not author), a *learned* $\mathrm{SO}(2)$-equivariant world model's multi-step rollout error is flat over the orbit of scene orientations to the floating-point floor, while a $160{\times}$ parameter sweep of non-equivariant baselines never reaches the equivariant floor out of distribution.](figures/step59_pusht_certificate.png)

**(E6) The certificate is not $\mathrm{SO}(2)$-specific, and transfers to pixels at no accuracy cost.** It lifts to
non-abelian $\mathrm{SO}(3)$ on 3D point clouds (learned equivariant rollout exactly flat, ratio $1.000$, with a
$7.4{\times}$-*smaller* model than the baseline). On raw rendered pixels (PushT, exact $C_4$), **frame averaging** [@puny2022frame] —
a plain CNN/MLP made exactly $C_4$-equivariant by a Reynolds average over grid rotations — keeps the exact orbit-flat
certificate (ratio $1.000$) while being **accuracy-neutral**: it matches or beats an unconstrained CNN on a
collapse-robust metric (fraction-of-variance-unexplained ratio $0.68$–$1.07$, mean $0.84$, 3 seeds), with a healthier
latent and a horizon-stable rollout where the steerable baseline diverges. The prior costs nothing *relative to* the
unconstrained CNN — though both remain limited in absolute terms (neither beats predict-the-mean at this scale; §6).

![The certificate on raw pixels at no accuracy cost (E6). On rendered PushT frames (exact $C_4$), frame averaging makes a plain CNN/MLP exactly $C_4$-equivariant: the latent rollout is flat over the orbit (left, the certificate), it matches or beats an unconstrained CNN on a collapse-robust accuracy metric (centre), and its rollout error stays bounded over horizon while the steerable baseline diverges (right, log scale).](figures/step64_frame_averaged_pixel.png)

**(E7) The certificate becomes task competence.** Run through a $G$-equivariant planner (A5), the prediction
certificate becomes an *orbit-invariant control* certificate: closed-loop pose error is flat over the orbit to the
floating-point floor (ratio $1.000$, 3 seeds) while a $4.3{\times}$-larger non-equivariant controller degrades out of
the training wedge. The flatness is exact because (A5) makes the planner $G$-equivariant and the planning cost
$G$-invariant, so the closed-loop trajectory inherits the encoder's orbit-equivariance — Theorem A under the
closed-loop clause.

**(E8) The certificate is an epistemic drive: certificate-driven active inference.** Because the certified region is
*computable* (Algorithm 1), an agent can act to **expand** it — a provable alternative to curiosity's "reduce
prediction error". On a $\mathbb{Z}_2^7$ compositional world whose action space mixes $7$ true generators with noisy
high-error distractors, an explorer that maximizes certified-region growth certifies all $2^7{=}128$ compositions in
exactly $7$ observations (the generator basis — each generator unlocks an exponential swath, Lemma 1), whereas an
error-curiosity explorer is lured by the irreducible-noise distractors and certifies only $1\%$ at the same budget (3
seeds; random $\sim2\%$). So "expand your certified region" is a **noise-immune** epistemic drive that beats
prediction-error curiosity — the classic *noisy-TV* failure mode. *Honest scope:* against *raw-error* curiosity (a
sophisticated information-gain agent would also avoid aleatoric noise); the certificate supplies that immunity **for
free and provably**, no noise model — a toy demonstration of the principle, not a benchmark.

![Certificate-driven active inference (E8). On a $\mathbb{Z}_2^7$ compositional world with noisy distractor actions, an explorer that maximizes the *certified region* certifies all $128$ compositions in exactly $7$ observations (the generator basis), while a prediction-error-curiosity explorer is lured by the high-error distractors and certifies almost nothing at the same budget.](figures/step69_certificate_driven_exploration.png)


**(E2 supplementary figures.)**

![The horizon staircase on a learned model of real chaotic dynamics (E2, Lorenz). **Left:** the learned one-step model's perturbation growth tracks the true Lorenz integrator over $550$ steps. **Right:** the certified horizon $T(\epsilon_0)$ on the *learned* model is linear in $\log(1/\epsilon_0)$ ($R^2{=}0.995$), and the measured slope sits on the prediction $1/(\lambda_1 dt)$ from the textbook Lorenz exponent — Theorem B's law lifted to a learned model of a genuinely chaotic system (Proposition 7(a)).](figures/step70_lorenz_horizon.png)

![High-dimensional spectral horizon (E2, $40$-D Lorenz-96). **Left:** recovered vs. true Lyapunov exponent for all $40$ channels — the $\mathbb{Z}_N$-equivariant cyclic-conv (blue) lies on $y=x$ ($R^2{=}0.98$); the dense MLP (red ×) is scattered far off ($R^2{=}-1.1$). **Right:** per-channel certified horizon $\log(1/\epsilon)/\lambda_j$: the equivariant model tracks the truth across the spectrum. Structure recovers the high-dimensional spectral horizon a dense model of equal data cannot.](figures/step74_lorenz96_spectrum.png)

![The horizon $\times$ resolution staircase on a controlled spectrum (E2). The measured certified horizon per channel as the demanded resolution $\epsilon$ sharpens, recovering $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ (slope $1.3$–$1.6$): chaotic channels are certified to a few steps, slow/invariant channels to dozens.](figures/step52_horizon_resolution.png)


**(E16, full protocol) V-JEPA 2-AC audit + real-robot-data monitoring.** Loading: official `vjepa2_ac_vit_giant`
(ViT-g/16 encoder, $1012$M; AC predictor post-trained on DROID) via torch.hub into the authors' code (an upstream
goof — `VJEPA_BASE_URL` pinned to `localhost` on main — is monkeypatched and disclosed). Audited loop: the authors'
own energy-landscape rollout, $g(z)=\mathrm{LN}(\mathrm{Pred}(z,a^{\ast},s))[:,-256:]$ on per-frame token blocks
($256\times1408$), fixed zero-delta action (the LeWM pre-registered scope); their planning *energy* is the L1
distance between predicted and encoded tokens — the same quantity the monitor thresholds. Certificate: leading-$6$
Benettin via forward-mode JVP, fp32 CUDA (precision disclosed; the authors' explicit-attention branch is used, as
flash/efficient SDPA lack forward-AD), two independent $Q$-seeds agreeing to $1.8\%$:
$\lambda_1=0.176$–$0.180$ across **five** $Q$-seeds ($2.4\%$ spread), envelope CI $[0.133,0.250]$ — **expansive**, nominal $T_1(0.2)=9.1$. Measured side ($40$
real DROID episodes, `lerobot/droid_100`, exterior camera, $4$-frame model step per the official config, telemetry
actions from logged poses via the authors' `poses_to_diff`): one-step relative error $0.629$ (median) vs
consecutive-latent distance $0.680$ and copy-of-last-read baseline $0.742$; staleness error grows
$0.623\to0.774$ over $8$ steps — log-slope $0.030$ (per-window median $0.027$), $5$–$6\times$ below the certified
$\lambda_1$; belief-invalid fraction $1.00$ at every cadence including $k{=}1$; crossing median $1$ ($0\%$
censored). Pre-registered branch G8-E (pricing band $[T_1/1.5, 1.5T_1]$) **fails as registered**; the
pre-registered sub-classification rule reads **bias**. Mechanism: the deployment error *starts* at the
representation's native step-motion scale and never enters the linearization neighborhood — Proposition 7's bias
degeneracy — so the tangent spectrum's jurisdiction and the monitor's operating point do not overlap, and the E13
protocol's measured column correctly overrides the tangent number. The monitor verdict for this cell is the
bias-cell verdict (no Lyapunov pricing); the wording claim is **real-robot data, offline monitoring** — the monitor
is sensor-only/passive, so replaying logged episodes is a faithful instantiation, not a simulation of one.
(`step98`, `step99`; conditional-gate spec frozen before the certificate was read.)

**(E13, ε-axis) The regime split is ε-monotone — in the direction Proposition 8 predicts.** A reviewer may ask
whether the growth-vs-bias re-stratification of the $84$-cell map is an artifact of the $\epsilon{=}0.2$ column.
Recomputing it from the same $100$-start artifact at the other published resolutions: bias-dominated
$25/42\to31/42\to39/42$ and growth-set size $15\to7\to3$ as $\epsilon$ tightens $0.2\to0.1\to0.05$, with the
growth-side ratio median moving $0.94\to0.55\to0.26$. This is not fragility but the mechanism's own dose–response
along the $\epsilon$ axis: the threshold is $\epsilon\cdot\mathrm{scale}$, so as $\epsilon\to0$ the fixed native
one-step residual dominates every horizon — exactly the tight-$\epsilon$ optimistic regime the seed map already
reported (Prop. 8's $\delta$-bias), now measured as a monotone fraction across the zoo. The deployment-relevant
column is the coarse one ($\epsilon{=}0.2$); the certificate's jurisdiction shrinks with $\epsilon$, and the map
says so quantitatively. (`step89c` artifact, descriptive re-analysis; no new runs.)

## Appendix E — concurrent and recent work (full sweep through 2026-06)

Mo (arXiv:2605.03338) proves symmetry-protected *neutral* Lyapunov modes for continuously-equivariant fields ($\ge\dim(G/H)$ zero exponents along the group orbit) — complementary to ours: it constrains the spectrum's *kernel*, while we certify the *horizon* stratified by the whole spectrum (for our discrete $\mathbb{Z}_N$ systems $\dim(G/H){=}0$, so their lower bound is zero there — the two results constrain disjoint parts of the spectrum: they the kernel, we the horizon). Geng et al. (arXiv:2512.08991) bound world-model rollout deviation *conformally* for closed-loop verification — statistical and rollout-hungry where ours is a-priori and training-free; the same trade-off separates us from reachability-based MBRL safety (UPSi, arXiv:2604.26836) and where-to-trust heuristics (arXiv:2606.01363). Pretrained world models have been probed *semantically* (arXiv:2603.21546) and benchmarked by sample rollouts (WorldBench, arXiv:2601.21282; WorldArena, arXiv:2602.08971); a Jacobian certificate of a public model's latent map, cross-validated against true-environment divergence, is to our knowledge new — the nearest priors being latent-space stability analyses of *self-trained* autoencoders (Özalp & Magri, arXiv:2410.00480) and Lyapunov-regularized DreamerV3 *policies* (arXiv:2410.10674). Flow-equivariant world models ([@lillemark2026flowm], now ICML 2026; antecedent FERNN, arXiv:2507.14793) preserve equivariance over arbitrarily long rollouts — an exactness/closure property, not a quantitative horizon. Inductive-bias studies (arXiv:2602.06923) and position papers calling for verified world models (arXiv:2602.23997) support the framing; PDEder (arXiv:2603.22655) *suppresses* latent Lyapunov exponents where we *certify* them; two-sided calibration bounds under equivariance (arXiv:2510.21691) concern calibration error, not horizon; and the data-assimilation literature's observation-frequency-vs-Lyapunov-time rule (e.g. Bocquet et al. 2026) is the classical neighbor of our sensing-budget law (Proposition 9).
