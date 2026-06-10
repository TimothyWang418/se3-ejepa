---
title: "Scale Buys Interpolation, Structure Buys a Horizon: Certified Predictability for Equivariant World Models"
subtitle: "How far a world model can promise — tightly, and only with structure."
author: "Hongbo Wang"
created: 2026-06-05
venue: "ICLR submission draft (focused extraction of the Certified World Models line)"
---

## Abstract

*Scale buys interpolation; structure buys a certified horizon.* A world model's average error says nothing about whether a *particular* prediction can be trusted, or for how long. For **equivariant** latent world models we give a computable, multi-step **certificate of the predictable horizon**: $T$-step rollout error is provably constant over each symmetry orbit (Theorem A), stratified channel-by-channel by the predictor's Lyapunov spectrum, $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$, and **two-sided** — a matching lower bound makes approximate equivariance provably horizon-limited (Proposition 6). The certificate is *equivalent* to equivariance (Lemma 2): no non-equivariant model has it at any scale. Empirically the law lifts to learned models of real chaotic systems — on $40$-D Lorenz-96 only a $\mathbb{Z}_N$-equivariant network recovers the full spectrum ($R^2{=}0.98$; dense and recurrent baselines fail) — and a cone certificate (Theorem B′) makes the horizon a-priori sound, tight exactly on uniformly-hyperbolic dynamics. The certificate is *actionable*: under a fixed sensing budget a $c\times$-inflated certificate provably needs $c\times$ the budget (Proposition 9), and it **audits public pretrained world models training-free** — calibrated on expansive TD-MPC2 latents (ratio $0.94$–$1.02$), correctly abstaining on contracting ones. §6 states the $1$–$2$-GPU scope.

---

## 1. Introduction

A world model is judged by its average prediction error. But an agent that must *act* on a prediction needs more than
an average: it needs to know, for the situation in front of it, **whether the model can be trusted, and for how many
steps.** Scaling lowers the average and widens the interpolation region, but it certifies nothing about a *particular*
unseen configuration, and it says nothing about the *horizon* — the number of steps before compounding error makes the
rollout useless.

Equivariance is known to give *some* per-situation guarantee. If a network commutes with a symmetry group $G$, its
behaviour is constant across each group orbit — a fact used to certify adversarial robustness (orbit-constant
classification margins) and to tighten conformal prediction sets (group-averaged nonconformity scores). But every such
result is **single-shot**: it attaches a guarantee to one input, for one prediction. A world model predicts a
*trajectory*, and the question that matters for planning — *how far ahead is the guarantee valid?* — has no answer in
that literature.

This paper answers it. We show that an equivariant latent world model admits a **certificate of its predictable
horizon**, stratified channel-by-channel by the predictor's spectrum, and we prove the horizon is **tight**. The
contributions are:

1. **A tight certified horizon (§3.2).** For an equivariant model the $T$-step rollout error is orbit-constant
   (Theorem A); relaxing to approximate equivariance, channel $j$ with Lyapunov exponent $\lambda_j$ is certified to
   horizon $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ (Theorem B, the *upper* bound). Our central new result is a
   **matching lower bound** (Proposition 6): an $\epsilon$-approximately-equivariant model — even one with a *perfect*
   predictor — has orbit-error-variation exactly $\epsilon\,e^{\lambda T}$ on an expansive channel, so its certified
   horizon is $\Theta(\log(1/\epsilon)/\lambda)$, **no better**. Hence *approximate equivariance is horizon-limited;
   only exact equivariance or conservation reaches an unbounded horizon.* A **scope theorem** (Proposition 7) then says
   *when* the locally-measured spectrum governs a *learned* model's horizon — Oseledets gives the rate on spectrally
   non-degenerate dynamics, the law degenerates on near-neutral dynamics — and a **finite-horizon continuity bound**
   (Proposition 8) makes the learned-model lift rigorous, not merely empirical. The horizon staircase then **lifts to
   learned models of a class of genuinely chaotic systems** (E2: Lorenz, Hénon, Rössler — the learned model's Lyapunov
   exponent matches the true one to $1$–$12\%$), while the near-neutral PushT interior is the predicted degenerate case.
2. **The certificate is exclusive to structure (§3.1).** The converse (Lemma 2): orbit-constant error against every
   equivariant target $\iff$ the model is equivariant. So **no non-equivariant model has the certificate at any size**
   — an architectural impossibility, proved, not asserted.
3. **The Noether hinge (§3.3).** The unbounded-horizon channels are *exactly* the conserved/invariant ones:
   Proposition 4 places each conserved charge in its forced isotypic block, and Proposition 5 turns conservation into
   a long-horizon guarantee ($T$-step charge error $\le T\eta$, linear, not the chaotic $e^{\lambda T}$). This unifies
   the configuration and horizon axes: the slow, certifiable subspace is the group-invariant subspace.
4. **Single-shot certified equivariance is our $T{=}1$ slice (§5).** We position the robustness/conformal guarantees
   as the one-step, resolution-free special case, and validate the multi-step picture on a real contact simulator,
   on non-abelian $\mathrm{SO}(3)$, and on raw pixels — where frame averaging makes the certificate *accuracy-neutral*.

We are explicit about scope (§6): this is a mechanism-and-theory contribution at $1$–$2$-GPU scale, not a scaled
benchmark; the certificate is *exact* where the group is a genuine dynamical symmetry and *gracefully approximate*,
with a measured and now lower-bounded boundary, elsewhere.

**What is new.** The contribution occupies a corner a prior-art sweep finds *empty* — no work
combines equivariance with a certified, two-sided spectral horizon; the closest guarantee-bearing neighbours each
miss the intersection [@conradie2026trustkoopman; @lillemark2026flowm; @mo2026symmetry] (distinguished in §5). **The new
results:** (i) Proposition 6's *matching lower bound* — growth seeded by the *equivariance residual* $\epsilon$ rather
than an initial-condition error — making the horizon two-sided and approximate equivariance provably horizon-limited;
(ii) the scope and continuity bounds (Propositions 7–8) lifting the law to *learned* chaotic models, where shadowing
cannot transfer the exponent; (iii) the **structure-vs-recurrence separation** (E2) — among Markov models only the equivariant one
recovers a $40$-D spectrum, while an unstructured *recurrent* model provably cannot, by the conditional-Lyapunov
condition; (iv) **Theorem B′** turns the certificate from *measured* into *literal* — a sound, a-priori horizon read
from the learned model, tight on uniformly-hyperbolic dynamics and self-abstaining elsewhere; and (v) the certificate's
faithfulness is **actionable, a priori** (E12; Proposition 9) — holding the forecaster fixed, it meets a fixed sensing
budget a $\sim3\times$-inflated non-equivariant certificate cannot, with **zero calibration data** (a dense certificate
matches only after recalibration on measured rollouts), delivering the opening promise that an acting agent needs to
know not just the average error but *whether, and for how long, to trust the model*. These assemble
(Algorithm 1) into one certificate *exclusive to structure* (Lemma 2) whose unbounded-horizon
subspace is the conserved/invariant one (the Noether hinge). We credit the classical pillars we build on — the
Lyapunov/NWP horizon law, invariant decision theory, and the $e^{\lambda T}$ growth — explicitly in §3 and §5.

![The predictability certificate at a glance. **Left:** in the configuration $\times$ horizon plane, an equivariant model certifies the *entire* generated monoid $\langle S\rangle$ — every composition, from $k$ generator checks (Lemma 1) — up to a horizon ceiling set by the predictor spectrum (Theorem B, tight by Proposition 6); a non-equivariant model of any size certifies only a small interpolation tube. **Right:** the horizon $\times$ resolution trade-off $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ — conserved/invariant ($\lambda\le0$) channels are certified to all horizons (eclipses, millennia), chaotic ($\lambda>0$) channels shrink as the demanded resolution sharpens (weather, $\sim$two weeks).](figures/hero_certified_region.png)

---


![**The paper in one figure** — *scale buys interpolation; structure buys a certified horizon.* **(a) Faithful:** on $40$-D Lorenz-96 the $\mathbb{Z}_N$-equivariant model recovers the full Lyapunov spectrum ($R^2{=}0.98$) where an identically-trained dense model's is garbage ($R^2{<}0$, $\lambda_1$ inflated $\sim3.4\times$) — E2. **(b) Priced:** under a fixed sensing budget, re-observation timed by the faithful certificate meets the budget while the inflated certificate over-observes and starves it — a $c\times$-inflated certificate provably needs $c\times$ the budget (Proposition 9) and a certificate-free adaptive scheduler pays $\sim3\times$ — E12. **(c) Real:** the same training-free read-out audits official TD-MPC2 checkpoints — calibrated (ratio $0.94$–$1.02$) where the latent loop is expansive, correctly abstaining where it contracts (Proposition 7) — E13.](figures/hero_certified_world_models.png)

## 2. Setup and assumptions

A latent world model is an encoder $E:\mathcal X\to\mathcal Z$ and a predictor $f:\mathcal Z\times\mathcal A\to
\mathcal Z$ trained so that $f(E(x),a)\approx E(\Phi(x,a))$, where $\Phi$ is the (unknown) environment dynamics. We
roll out $\hat z_T = f(\cdot,a_{T-1})\circ\dots\circ f(E(x),a_0)$ and compare to the target $E(\Phi^T(x))$; the
$T$-step error is $\mathrm{Err}_T(x)=\lVert \hat z_T - E(\Phi^T x)\rVert$. A group $G$ acts on $\mathcal X$ (situations),
$\mathcal Z$ (latents, via a representation $\rho$), and $\mathcal A$ (actions, via $\sigma$). We use:

- **(A1) Encoder equivariance:** $E(g\cdot x)=\rho(g)\,E(x)$.
- **(A2) Predictor equivariance:** $f(\rho(g)z,\sigma(g)a)=\rho(g)\,f(z,a)$.
- **(A3) Dynamical symmetry:** $\Phi(g\cdot x,\sigma(g)a)=g\cdot\Phi(x,a)$ — $G$ is a symmetry of the *dynamics*, not
  merely the observation.
- **(A4) Orthogonality:** $\rho(g)^\top\rho(g)=I$ (so $\rho$ preserves the latent norm in which error is measured).
- **(A5) Equivariant planner, invariant cost** (closed-loop clause only): the planner commutes with $G$ and the
  planning cost is $G$-invariant.

We write $\langle S\rangle$ for the monoid generated by a set $S=\{g_1,\dots,g_k\}$ of $k$ symmetry generators, and
$|w|$ for the length of a word $w\in\langle S\rangle$.

---

## 3. The certificate

### 3.1 The configuration axis, and why only structure has it

**Theorem A (orbit-constant error).** Under (A1)–(A4), for every $w\in\langle S\rangle$, all $x$, and any action
sequence (transported by $w$), $\mathrm{Err}_T(w\cdot x)=\mathrm{Err}_T(x)$ (Appendix B gives the action-explicit
statement and proof). *Proof sketch.* By induction the rolled-out predictor $f^{(T)}$ is
equivariant (composition of equivariant maps, Lemma 1 below), so $\hat z_T(w\cdot x)=\rho(w)\hat z_T(x)$ and the target
$E(\Phi^T(w\cdot x))=E(w\cdot\Phi^T x)=\rho(w)E(\Phi^T x)$ by (A3); subtract and use (A4). $\square$

**Lemma 1 (composition closure).** If the equivariance assumptions (A1)–(A3) hold on each generator $g_i\in S$, they
hold on every word $w\in\langle S\rangle$ ((A4) is a global property of $\rho$ as a homomorphism into $O(\mathcal Z)$,
automatic for all $w$). Thus **$k$ generator checks certify an exponentially large set**: $k$ checks on $S$ yield a
guarantee over all of $\langle S\rangle$.

**Lemma 2 (the certificate characterizes equivariance — the converse).** Let $\rho:G\to O(\mathcal Z)$ act *freely* on
an open $U\subseteq\mathcal Z$. If a predictor $f$'s error $\lVert f-\Phi\rVert$ is orbit-constant on $U$ for *every*
equivariant target $\Phi$, then $f$ is equivariant on $U$. *Proof.* Fix $z,g$. For any probe $c$, freeness makes
$\Phi(\rho(h)z):=\rho(h)c$ a well-defined equivariant target on the orbit; orbit-constancy gives
$\lVert f(z)-c\rVert=\lVert\rho(g)^{-1}f(\rho(g)z)-c\rVert$ for *all* $c$, and two points equidistant to every $c$
coincide, so $f(\rho(g)z)=\rho(g)f(z)$. $\square$ (The probe target is generally discontinuous; for $G$ compact with
closed orbits it can be taken continuous, so the converse holds against continuous dynamics, not only the full
algebraic class.) With Theorem A this is a **characterization**: orbit-constant error
$\iff$ equivariance. **Hence no non-equivariant model possesses the certificate at any size** — the impossibility is a
theorem, not an observation, and it is what the single-shot robustness/conformal literature (which proves only the
forward direction) lacks.

### 3.2 The horizon axis, and its tightness (the central result)

Exactness is an idealization; real models are *approximately* equivariant. Let $\epsilon_{\max}=\max_i\sup_x\lVert
E(g_i\cdot x)-\rho(g_i)E(x)\rVert$ be the encoder residual, $\delta$ the per-step predictor error, and let the latent
map's Jacobian be locally diagonalized on channel $j$ with multiplier $e^{\lambda_j}$ ($\lambda_j$ a Lyapunov
exponent).

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
*Proof.* $\mathrm{Err}_T(x)=0$ (model exact along $x$). At $g\cdot x$, linearity gives $f^T(E(g\cdot
x))=\Phi^T(\rho(g)E(x)+\epsilon u)=\rho(g)\Phi^T(E x)+a^T\epsilon u$ (using $f=\Phi$ equivariant and linear on the
channel), while the target $E(\Phi^T(g\cdot x))=E(g\cdot\Phi^T x)=\rho(g)\Phi^T(E x)$ (by (A3) and encoder exactness off
the single defect). Subtract and use (A4): $\lVert a^T\epsilon u\rVert=\epsilon e^{\lambda T}$. $\square$

Hence the certified horizon $T(\epsilon_{\mathrm{res}})=\tfrac1\lambda\log\tfrac{\epsilon_{\mathrm{res}}}{\epsilon}=
\Theta\!\big(\tfrac1\lambda\log\tfrac1\epsilon\big)$, matching Theorem B: **the horizon's form is two-sided, not merely
an upper estimate.** The conceptual payload is sharp:

> **The certified horizon guaranteeable from $\epsilon$-approximate equivariance alone is finite on every expansive
> channel: no certificate derived from an $\epsilon>0$ residual can promise predictability beyond
> $T\sim\tfrac1\lambda\log\tfrac1\epsilon$ (worst case over admissible targets). Only exact equivariance
> ($\epsilon=0$) or conservation ($\lambda_j\le0$) yields an unbounded certified horizon.**

This is the horizon-domain companion of Lemma 2, and it sharpens *scale vs. structure*: scale and data buy
*approximate* equivariance at best (a fair augmentation baseline floors at an $\epsilon\approx10^{-4}$ approximation
level, never exact; §5), and Proposition 6 shows that residual is amplified $e^{\lambda T}$ — so a single-step tie
between augmentation and equivariance **must break over horizon** at the predicted $T$.

![Proposition 6, numerically (the central claim). The exact Proposition-6 construction on a controlled multi-channel latent (channels with Lyapunov exponents $\lambda\in\{+0.70,+0.35,0,-0.40\}$). **Left:** the orbit-error-variation of an $\epsilon{=}10^{-3}$-approximately-equivariant model (markers) equals the analytic $\epsilon\,e^{\lambda T}$ (dashed) to a relative error of $10^{-14}$–$10^{-13}$ across 3 seeds — the lower bound is exact, so the horizon is tight; an *exactly* equivariant model has orbit-variation *exactly $0$* (to machine precision) at every horizon (plotted at the $10^{-16}$ log-floor), and conserved/contractive channels ($\lambda\le0$) are bounded for all $T$ (infinite certified horizon). **Right:** the per-channel certified horizon $T(\epsilon_{\rm res})$ is linear in $\log(1/\epsilon)$ with slope exactly $1/\lambda$ ($R^2=1.000$, all seeds) — the $\Theta(\log(1/\epsilon)/\lambda)$ law (`experiments/step65`, seeds $0/1/2$; the trained-model approximate-symmetry degradation is E3, Appendix D).](figures/step65_horizon_tightness.png)

**Proposition 7 (scope — when the local spectrum certifies a *learned* model's horizon).** Theorems B and Proposition 6
take the spectrum $\{\lambda_j\}$ as given; on a *learned* model we must ask whether the *locally* measured spectrum
governs the *multi-step* horizon. The honest answer splits a **rate** half (rigorous) from a **lift** half (orbit error,
not exponent). Let $\phi$ be the true dynamics with an ergodic invariant measure $\mu$, $\log^+\lVert D\phi\rVert\in
L^1(\mu)$. By the **Oseledets multiplicative ergodic theorem** [@oseledets1968], $\tfrac1t\log\lVert\delta_t\rVert\to\lambda_1$ $\mu$-a.e.
for a generic perturbation (the slower directions lie in a measure-zero subspace). *(a) Non-degenerate ($\lambda_1>0$):*
$T(\epsilon)=\tfrac1{\lambda_1}\log(\epsilon_{\mathrm{res}}/\epsilon)+o(t)$ — Theorem B's law with $\lambda_1$ now the
*measurable* asymptotic rate. The lift to a learned $\hat\phi$ is an *orbit-error* statement: on uniformly hyperbolic
$\phi$ the shadowing lemma [@pilyugin1999shadowing] bounds the forecast-horizon *floor* $\sim\tfrac1{\lambda_1}\log(1/\delta)$ for one-step error
$\delta$, but it controls trajectory closeness, **not** the *asymptotic* Lyapunov exponent (only
upper-semicontinuous under $C^1$ perturbation). That $\hat\phi$ *reproduces* $\lambda_1$ is therefore not a shadowing
corollary — but it *is* a finite-horizon continuity statement (Proposition 8 below), confirmed in E2. *(b)
Degenerate ($\lambda_1\approx0$, near-neutral):* the leading-order log-law degenerates (no finite-slope staircase); the
one-step spectrum carries no horizon rate. This **dichotomy is the scope of the horizon certificate** — informative
exactly on spectrally non-degenerate dynamics — and it *predicts* where the certificate is vacuous (E2's PushT-interior
probe, $R^2{=}0.02$), rather than leaving it as a gap.

**Proposition 8 (finite-horizon exponent transfer).** The asymptotic exponent is only upper-semicontinuous — why
*shadowing* cannot transfer it — but the certified horizon is **finite**, and the finite-time exponent
$\lambda_1^{(T)}=\tfrac1T\mathbb E\log\lVert D\phi^T\rVert$ is locally Lipschitz in the dynamics over a finite orbit.
So a learned $\hat\phi$ with one-step fidelity $\delta$ recovers the staircase slope up to an $O(\delta)$ fidelity bias
plus the finite-$T$ truncation $\lvert\lambda_1^{(T)}-\lambda_1\rvert$, $T$-uniform under a dominated splitting
[@bochi2005lyapunov]. The match is thus **falsifiable** — the recovered exponent must tighten as one-step error drops —
which E2 confirms (on Rössler the error falls $44\%\!\to\!8\%$ as training fidelity rises).

### 3.3 The Noether hinge: which channels are unbounded

Theorem B leaves the $\lambda_j\le0$ (unbounded-horizon) channels unidentified. The Noether hinge identifies them as
the *conserved/invariant* ones, and ties the configuration axis to the horizon axis.

**Proposition 4 (isotypic placement).** A conserved charge of the dynamics must live in a specific isotypic block of
$\rho$: a scalar invariant (energy) in the trivial ($\ell{=}0$) block; a vector charge (angular momentum) in the
$\ell{=}1$ block, recoverable only through the unique degree-2 cross-product equivariant. Placement is *forced* by representation theory [@goodman2009symmetry], not chosen.

**Proposition 5 (conservation $\Rightarrow$ unbounded horizon).** Let $Q:\mathcal Z\to W$ be a charge the model
conserves to one-step defect $\eta$ (i.e. $\lVert Q(f z)-Q(z)\rVert\le\eta$). Then the $T$-step *charge-value*
prediction error satisfies $\lVert Q(\hat z_T)-Q(z_T)\rVert\le T\eta$ — **linear in $T$**, never the chaotic
$e^{\lambda T}$. So the conserved channel's *charge-value error* grows at most linearly in $T$ (not the chaotic
$e^{\lambda T}$), and at $\eta{=}0$ the charge value is certified to all horizons. (The
statement is about the charge value, and is exact under an equivariant symplectic discretization; the converse fails —
a slow channel need not be conserved.)

Together: **the certified region is the coarse ($\epsilon$ large), invariant/conserved ($\lambda\le0$), low-composition
($|w|$ small) corner** — and the converse (Lemma 2) says this corner is reachable only with structure.

### 3.4 The certificate is a procedure

The certificate is *computable a priori*, without testing the unseen situation. Algorithm 1: (i) check (A1)–(A4) on
the $k$ generators to a residual $\epsilon_{\max}$; (ii) estimate the predictor Jacobian spectrum $\{\lambda_j\}$ at the
query latent (reported with a block-bootstrap CI, calibrated against the Liouville anchor; `experiments/step78`); (iii) report, for a target situation $w\in\langle S\rangle$, horizon $T$, resolution $\epsilon$, whether
$(w,T,\epsilon)$ lies in the certified region $\{|\Delta\mathrm{Err}|\le\epsilon\}$ implied by Theorem B / Proposition 6,
escalating conserved channels to unbounded horizon via Proposition 5.

---

## 4. Experiments

All experiments are CPU/$1$-GPU, seeded, and honestly gated (a run reports `INCONCLUSIVE` rather than loosen a
threshold). The main text carries the load-bearing results; the supporting suite (E1, E3–E8) is summarized in one
paragraph below and written up in full in **Appendix D**.

**(E2) The horizon staircase — and where structure becomes necessary.** On a controlled latent with a planted spectrum the certified-horizon slope brackets the predicted $1/\lambda$ (Theorem B above, Proposition 6 below; chaotic exponent recovered to $0.4\%$). The law lifts to learned models of *real* chaotic dynamics: plain one-step models of **Lorenz**, **Hénon**, and **Rössler** give staircases linear in $\log(1/\epsilon)$ ($R^2{=}0.975$–$0.995$, $3$ seeds) whose slopes recover the textbook exponents to $1$–$12\%$ across an order of magnitude in $\lambda_1$ — Proposition 8's finite-horizon continuity at work, not a shadowing corollary — while the near-neutral **PushT interior** is Proposition 7's predicted *degenerate* branch ($R^2{=}0.02$: there the spectrum rightly does not predict the rollout). At high dimension, structure becomes *necessary*: on $40$-D **Lorenz-96** [@lorenz1996predictability] ($F{=}8$; Liouville sum $\sum_j\lambda_j{=}-N$ recovered to $0.0\%$) a $\mathbb{Z}_N$-equivariant cyclic-conv recovers the *full* spectrum ($R^2{=}0.98$–$0.99$, $3$ seeds; Kaplan–Yorke $\sim\!27$) — hence the per-channel certified horizons — while an identically-trained dense MLP **fails** ($R^2{<}0$, $\lambda_1$ inflated $\sim\!3.4\times$) and a GRU-BPTT [@vlachas2020rnn] fails the same way at $N{=}40$ ($R^2{\approx}-0.3$), its joint-state Jacobian carrying hidden modes that break the conditional-Lyapunov condition [@hart2024attractor]. Sweeping $N\in\{12,20,28,40\}$ shows a **phase transition**, not a single-$N$ artifact: all three architectures tie through $N{=}28$; only the equivariant one survives $N{=}40$ (`step83`).

![Step 83. Full-spectrum Lyapunov $R^2$ vs. $N$: the $\mathbb{Z}_N$-conv holds across $N$ while the dense MLP and GRU collapse at $N{=}40$ — the single-$N$ separation is a phase transition.](figures/step83_rsquared_crossover.png)

![The certified-horizon law across a class of learned chaotic models (E2). The identical learned-model staircase on a 2D map (Hénon), a small-exponent flow (Rössler), and a large-exponent flow (Lorenz) is linear in $\log(1/\epsilon_0)$ and its slope (blue) recovers the textbook exponent (red dashed). The law is a property of chaotic dynamics, not of Lorenz; the residual bias is decomposed by Proposition 8.](figures/step71_multichaos_horizon.png)

**(E1, E3–E8: the supporting suite — full write-ups in Appendix D.)** The configuration axis is exponential: checking $k$ generators certifies all $2^k$ compositions ($\mathbb{Z}_2^6$: error $\sim10^{-33}$ vs. a non-equivariant baseline's $0.59$) (E1). Breaking the world's symmetry degrades the certificate *gracefully and linearly* — exactly Proposition 6's predicted crossing — and a fair augmentation baseline is horizon-limited at its $\sim10^{-4}$ equivariance floor (E3). An $88\times$-scaled non-equivariant model buys in-distribution interpolation but never the orbit-floor (E4); on **PushT contact dynamics** a learned $\mathrm{SO}(2)$-equivariant model is orbit-flat to ratio $1.000$ while no baseline in a $160\times$ parameter sweep reaches its out-of-distribution floor — with the honest control that the per-baseline gap ($2.1$–$3.9\times$) does not grow with scale; the impossibility claim is carried by Lemma 2, not by E5's magnitude (E5). The certificate lifts to non-abelian $\mathrm{SO}(3)$ and — via frame averaging [@puny2022frame], accuracy-neutrally — to raw pixels (E6); run through an equivariant planner it becomes orbit-invariant closed-loop *control* (E7); and, being computable, it is a **noise-immune epistemic drive**: an explorer that maximizes certified-region growth certifies $2^7$ compositions in exactly $7$ observations where prediction-error curiosity is lured by noisy distractors to $1\%$ (E8).

**(E9) Both axes on one system; the certificate acts.** On **controlled Lorenz-96** (exact $\mathbb{Z}_N$, genuine chaos $\lambda_1{\approx}1.8$) an equivariant planner gives orbit-flat control (residual $8\times10^{-16}$) while the certified $T_1(\epsilon)$ drives an **active re-observation** schedule that sits on the accuracy-vs-observation frontier *untuned* ($2/3$ seeds, asymptotic-$\epsilon$ regime; tight-$\epsilon$ honestly optimistic per Prop. 8). The pattern lifts to a $\mathbb{Z}_N$ pendulum ring and a $\mathbb{Z}_2$ double pendulum — a class property across two groups (`step79`–`step81`; Appendix D).

**(E10) The certified horizon, made literal — and its exact regime.** E2 *measures* that a learned model's Lyapunov
exponent matches the truth; Theorem B′ *certifies* it. A computable cone/adapted-metric certificate reads a **sound,
a-priori** upper bound on the learned model's top exponent off its Jacobian field — a guaranteed horizon
$T_{\mathrm{guar}}(\epsilon)$ from the model alone. On **uniformly-hyperbolic** dynamics it is *tight*: on a hyperbolic
toral automorphism (cat map) the certified exponent equals the true $\lambda_1$ to machine precision (true map) and to
$1.17\times$ on a *learned* net ($3$ seeds, sound, $T_{\mathrm{guar}}\le T_{\mathrm{true}}$), a nonlinear Anosov
perturbation staying tight ($1.06\times$). On **non-uniformly-hyperbolic** dynamics (Hénon, with homoclinic tangencies)
a single metric is provably limited — the certificate is *sound but $\sim\!3\times$ conservative*, its own cone-margin
diagnostic turns negative, and it **abstains** (routing to a statistical bootstrap horizon) rather than over-claim. So a
tight a-priori certified horizon from a learned model is achievable *exactly* in the uniformly-hyperbolic regime, and
the certificate **self-diagnoses** which regime it is in (`step82`).

**(E11) On a recognized control benchmark — honest INCONCLUSIVE.** On Gymnasium Acrobot-v1 ($\lambda_1{=}0.094$) the certified horizon *tracks* the measured one (ratio $0.42\to0.93$ as $\epsilon$ coarsens) and *binds for planning*: return has a sharp interior optimum and planning past $\sim\!T_1$ fails outright. Capping depth at $T_1$ solves the task ($67$–$100\%$) and bounds useful planning, but the return-optimum sits at $\sim T_1/2$, so two pre-registered no-tuning rules are **INCONCLUSIVE** against the best-tuned blind depth: a sound depth within a constant factor of optimal, not the optimum (`step84`).

**(E12) The certificate's trustworthiness changes a budgeted decision — a priori.** On $40$-D Lorenz-96 an agent schedules sparse re-observations of a chaotic forecast under a fixed sensing budget. Holding the forecaster fixed (the $\mathbb{Z}_N$-equivariant model; the two models are identically trained and forecast comparably — one-step relMSE $\sim10^{-5}$, empirical horizons of the same order) and varying *only* the certificate that times re-observation: the equivariant certificate yields $8$–$16\%$ aggregate violation at the knee budget vs. $61$–$65\%$ for the non-equivariant certificate, whose $\lambda_1$ is inflated $\sim3.4\times$ and which therefore over-observes and starves the budget (margins $+0.45/+0.50/+0.57$, $3/3$; **Proposition 9** makes the cost a law — a $c\times$-inflated certificate needs $c\times$ the budget, and the measured catch-up is $2.7$–$3.5\times$). Two controls scope the claim: a certificate-free adaptive scheduler matches only at $\sim3\times$ the budget (the certificate is an *a-priori* warm-start), and a *recalibrated* dense certificate closes the gap ($3/3$) but only by **spending a calibration set** — the equivariant certificate is correct from the Jacobian with **zero rollout data**. Standard UQ baselines complete the picture: a $4$-model ensemble and a $10$-rollout conformal quantile calibrate tighter ($|\log$ ratio$|$ $0.17/0.11$ vs $0.27$) by spending training or truth access the certificate does not — no method dominates the accuracy–cost Pareto; the certificate is its only a-priori point (`step90`). The contrast **replicates on the driven–damped pendulum ring** ($\mathbb{Z}_N$, $N{=}24$: dense $\lambda_1$ inflated $\sim2\times$; equivariant certificate wins under budget on $2/3$ seeds) — two symmetry groups, two physical systems, one mechanism; the within-method framing cancels E11's $\sim2\times$ conservatism. (A budget *allocation* across a chaoticity ensemble did *not* win — the learned spectrum is too inaccurate at low forcing to beat a flat allocation; reported honestly.) (`step85`, `step85b`, `step85c`, `step88`.)

**(E13) The certificate audits a *public pretrained* world model — training-free, and maps onto the paper's own scope theory.** We rebuild the latent-dynamics slices of the official **TD-MPC2** checkpoints (5M single-task DMC models; MIT) and run the *unchanged* certificate machinery on the policy-prior closed loop $g(z)=d(z,\tanh\mu_\pi(z))$ — no training, no environment access on the certified side — across **five tasks $\times$ 3 official seeds (15 latent loops)**. The result is a scope map, not a uniform win, and it tracks the theory cell-by-cell. Where the loop is **strongly expansive** ($\lambda_1{=}0.25$–$0.30$: walker-walk $3/3$, cheetah-run seed 3) the coarse-resolution certificate is **calibrated** — ratio measured/certified $0.83$–$1.02$ (walker $0.94/0.95/1.02$), with the same two-regime $\epsilon$ pattern as every system we trained ourselves (tight-$\epsilon$ optimistic, the Proposition 8 $\delta$-bias). As expansion **weakens** the certificate turns optimistic (cheetah $0.43/0.50$; hopper-hop $0.13/0.38$ at $\lambda_1{=}0.05$–$0.09$): model bias outpaces Lyapunov amplification — the degeneracy direction Proposition 7 flags. Where the loop **contracts** ($6/15$: acrobot $3/3$, finger-spin $2/3$, hopper seed 1) the certificate **abstains, correctly in both sub-cases**: finger-spin's stable loops genuinely do not diverge ($15$–$19/20$ starts censored at $300$ steps — nothing to certify), while acrobot's and hopper-1's residual divergence (median $6$–$45$ steps) is bias-driven, outside a Lyapunov certificate's jurisdiction. The SimNorm structural zero-directions appear as a strongly-negative spectral band ($128$–$270$ directions), reported, not hidden; the certified scope is the prior loop, not the MPPI planner. Closest prior work computes Lyapunov exponents of the *true environment* under RL policies (arXiv:2410.10674); a Jacobian certificate of the *learned latent map* of a public world-model zoo, cross-validated against true-environment divergence and stratified by the certificate's own scope theory, is new. (`step89`.) A second, architecturally disjoint family lands on the same map: the official **LeWM** checkpoint (ViT + transformer JEPA from pixels, PushT; loaded bit-faithfully into the authors' own code) has a free-running fixed-action loop with $\lambda_1{=}0.001$, CI straddling zero (leading band $\{0,0,-0.01..{-}0.08\}$) — the certificate **abstains**, and the observed $1$–$2$-step divergence is pure one-step bias ($0.17$; the zero action is in-support but off the expert distribution — pre-registered scope), the same bias-driven sub-case as acrobot. Two families, one read-out, one taxonomy (`step91`).



**What does structure buy, if the read-out audits any smooth model?** Theorem B's spectral law applies to any $C^1$ latent map — that is what E13 exercises on (non-equivariant) TD-MPC2 and LeWM. What the law cannot supply for a generic model is *trust in the number itself*: a dense model's spectrum can be silently wrong while its predictions stay good (E2: $\lambda_1$ inflated $\sim3.4\times$ at one-step relMSE $10^{-5}$), so a generic certificate must be cross-validated against held-out divergence — exactly the per-model empirical check the E13 scope map performs. Equivariance is what removes that requirement where it holds: structure makes the spectrum *faithful* (E2), hence the certificate *a-priori trustworthy with zero calibration data* (E12 and its recalibration control) — and exclusively so (Lemma 2). The audit is universal; the **a-priori** guarantee is structure's.

---

## 5. Related work

**Single-shot certified equivariance is our $T{=}1$ slice.** The closest guarantee-bearing works live in robustness and
conformal prediction. Equivariant classifiers have *orbit-constant margins* (the decision-boundary gradient norm is
preserved across the orbit), giving uniform adversarial certificates [@orbitmargin]; invariance-aware randomized
smoothing builds *orbit-based* certificates [@randsmooth]; and Equivariantized Conformal Prediction (eCP) [@ecp] group-averages the nonconformity score — frame averaging for distribution-free coverage. **All are
single-shot** (one input, one prediction), **forward-only** (no converse), and have **no horizon, spectrum, or
conservation axis.** They are exactly the $T{=}1$, $\epsilon$-independent slice of our certificate: our contribution is
the multi-step stratification (Theorem B, tight by Proposition 6), the converse (Lemma 2), and the Noether bridge to
unbounded horizon (Proposition 5).

**Learned conservation laws.** Noether Networks [@noethernet] meta-learn conserved quantities
inside the prediction loop, and Noether's Razor [@noetherrazor] learns symmetries-as-conserved-quantities by
Bayesian model selection. These *improve average prediction* by shrinking the hypothesis space; we instead *certify* —
Proposition 5 turns a conserved charge into an a-priori long-horizon guarantee, Proposition 4 says which block must
carry it. Guarantee versus average accuracy.

**Jacobian-regularized world models.** "Towards Unraveling and Improving Generalization in World Models"
[@jacobianwm] penalizes the latent-transition Jacobian to damp rollout error propagation — a heuristic for
robustness. Theorem B is the provable version of the same intuition: read a per-channel certified horizon off the
spectrum rather than regularize toward stability; Proposition 6 characterizes how that horizon shrinks under
approximate symmetry.

**Equivariant predictors and latent-geometry priors.** Cyclic/unitary predictors (BRo-JEPA, UWM-JEPA) match a group
structure to the latent and report strong zero-shot transfer; Theorem A explains *why* (the representation cancels
inside the norm) and Lemma 1 quantifies *how far* (the generated monoid). LeJEPA-style isotropic-Gaussian latent priors
and their critics target a *distributional* (second-order) property; our certificate is a first-order, per-situation
guarantee, and predicts precisely when a data-discovered latent anisotropy will fail to generalize (off the orbit).

**Predictability horizons.** The $T(\epsilon)\sim\log(1/\epsilon)/\lambda$ law is classical for dynamical systems
(Lyapunov; numerical weather prediction); that the *local* spectrum governs the *asymptotic* rate is the Oseledets multiplicative ergodic theorem [@oseledets1968], and on uniformly hyperbolic systems the shadowing lemma [@pilyugin1999shadowing] bounds a perturbed model's
forecast-horizon floor (it controls orbit error, not the exponent). Our contribution is to (i) prove the law *tight* for
a *learned* latent world model and **measure it on learned models of a class of genuinely chaotic dynamics** (Lorenz,
Hénon, Rössler, E2 — matched to $1$–$12\%$), made rigorous by a finite-horizon continuity bound (Proposition 8) where
shadowing fails; (ii) **characterize when it lifts** (Proposition 7: the non-degenerate/degenerate dichotomy,
synthesizing Oseledets for the rate with shadowing for the floor); (iii) tie its unbounded-horizon subspace to the
group-invariant subspace (Noether hinge); and (iv) make the forward direction's converse a theorem (Lemma 2). Recovering a chaotic spectrum from a *learned* model is itself known, *conditionally* on the model's contracting modes
[@hart2024attractor] — the reason the recurrent baseline (E2) fails at high $N$; the closest concurrent guarantees are
non-equivariant certified Koopman forecasts [@conradie2026trustkoopman] and equivariant-but-uncertified world models
[@lillemark2026flowm; @mo2026symmetry], none combining equivariance with a certified two-sided spectral horizon.

---


**Concurrent and recent work (swept through 2026-06).** Mo (arXiv:2605.03338) proves symmetry-protected *neutral* Lyapunov modes for continuously-equivariant fields ($\ge\dim(G/H)$ zero exponents along the group orbit) — complementary to ours: it constrains the spectrum's *kernel*, while we certify the *horizon* stratified by the whole spectrum (for our discrete $\mathbb{Z}_N$ systems $\dim(G/H){=}0$ and their statement is vacuous). Geng et al. (arXiv:2512.08991) bound world-model rollout deviation *conformally* for closed-loop verification — statistical and rollout-hungry where ours is a-priori and training-free; the same trade-off separates us from reachability-based MBRL safety (UPSi, arXiv:2604.26836) and where-to-trust heuristics (arXiv:2606.01363). Pretrained world models have been probed *semantically* (arXiv:2603.21546) and benchmarked by sample rollouts (WorldBench, arXiv:2601.21282; WorldArena, arXiv:2602.08971); a Jacobian certificate of a public model's latent map, cross-validated against true-environment divergence, is to our knowledge new — the nearest priors being latent-space stability analyses of *self-trained* autoencoders (Özalp & Magri, arXiv:2410.00480) and Lyapunov-regularized DreamerV3 *policies* (arXiv:2410.10674). Flow-equivariant world models ([@lillemark2026flowm], now ICML 2026; antecedent FERNN, arXiv:2507.14793) preserve equivariance over arbitrarily long rollouts — an exactness/closure property, not a quantitative horizon. Inductive-bias studies (arXiv:2602.06923) and position papers calling for verified world models (arXiv:2602.23997) support the framing; PDEder (arXiv:2603.22655) *suppresses* latent Lyapunov exponents where we *certify* them; two-sided calibration bounds under equivariance (arXiv:2510.21691) concern calibration error, not horizon; and the data-assimilation literature's observation-frequency-vs-Lyapunov-time rule (e.g. Bocquet et al. 2026) is the classical neighbor of our sensing-budget law (Proposition 9).

## 6. Limitations

- **Exactness needs a genuine dynamical symmetry (A3).** The certificate is *exact* where $G$ is a symmetry of the
  dynamics (orbital/conservative systems, free space, idealized manipulation) and *gracefully approximate* elsewhere,
  with a degradation that is now bounded on both sides (Theorem B and Proposition 6) and measured (E3).
- **Tight in form, not in prefactor.** Proposition 6 makes the horizon's $\log(1/\epsilon)/\lambda$ *form* tight, but
  the constants $c_j$ are not estimated and the isotypic refinement is asserted, not separately measured.
- **The Noether hinge's forward direction is proved; its hypotheses are measured.** Proposition 5 assumes a
  Hamiltonian latent flow with a $G$-invariant Hamiltonian; the defect $\eta$ is exact only under an equivariant
  symplectic discretization; the non-converse (slow $\not\Rightarrow$ conserved) is genuine.
- **The hinge is validated on constructed teachers, not shown to *emerge*.** The Noether experiments (E2 and the
  containment results) use *constructed* equivariant teachers, so the "slow $=$ invariant" coincidence is partly built
  in. The real-ish experiments (E5 PushT, E6 $\mathrm{SO}(3)$) validate *flatness*, not the hinge's emergence in a
  learned model of un-constructed dynamics — that remains open.
- **The horizon axis lifts to learned models of real *chaotic* dynamics, with a continuity bound, and is vacuous on
  near-neutral dynamics (scope, not a bug).** E2 validates the law on a synthetic spectrum, a *class* of low-D chaotic
  systems (Lorenz/Hénon/Rössler, exponent to $1$–$12\%$), and a **$40$-D** learned model ($R^2{=}0.98$–$0.99$);
  Proposition 7 makes it informative iff $\lambda_1>0$ — the **PushT interior is the degenerate branch**
  ($R^2{\approx}0.02$, predicted by 7(b)). Caveats: the lift rests on Proposition 8 (finite-time continuity), not
  shadowing, yet we verify $C^1$-closeness only via one-step $L^2$ error — in high $N$ a real gap (a dense MLP, and a
  recurrent GRU, is one-step-accurate yet mis-estimates the $40$-D spectrum; $\mathbb{Z}_N$-structure closes it); and
  the real-dynamics evidence is chaotic ODEs/maps, not a contact simulator.
- **Demonstrated downstream value is budgeted efficiency, not safety.** E12's win is a within-method, fixed-budget
  re-observation contrast isolating the certificate's *faithfulness*. A separate test of the certified horizon as a
  *catastrophe-avoidance* cadence was **inconclusive** at our scale (escape rate was re-observation-interval-invariant —
  control-quality-limited, not estimate-staleness-limited); the certificate's *necessity* for safety is open.
- **Scale and modality.** All experiments are $1$–$2$-GPU. The certificate's exact flatness transfers across modalities
  (state, $\mathrm{SO}(3)$ point clouds, pixels), but absolute multi-step accuracy on raw pixels is poor for *every*
  architecture at this scale (the anti-collapse JEPA latent, not equivariance); a strong few-step pixel predictor is an
  architecture-agnostic open problem orthogonal to the certificate.

---

## 7. Conclusion

An equivariant world model can certify, a priori and without retraining, which situations it will handle and **for how
many steps** — and the horizon is tight. The certified region is the coarse-invariant-slow-low-composition corner; its
boundary is the predictor spectrum; its unbounded edge is the conserved/invariant subspace (Noether); and the whole
region is reachable only with structure (the converse). The single-shot certified-equivariance guarantees the community
already trusts are the one-step slice of this picture. And because the horizon is *faithful* it is *actionable* — it
lets an agent budget its sensing on a high-dimensional chaotic forecast where an inflated certificate cannot (E12).
*Scale buys interpolation; structure buys a certified horizon.*

---

*Reproducibility and scope.* Full proofs are in **Appendix B**, a reproducibility appendix (anonymized code, experiment-to-code map, seed/gate discipline) in **Appendix C**. This is a focused extraction of a broader program; the full suite is deferred to an extended version.


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
