---
title: "Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group"
subtitle: "Compact main-text draft (additive) — full papers are the comprehensive supplement"
status: DRAFT (review block resolved 2026-06-01) — NOT wired into the arXiv build
target_venue: ICLR (primary) / NeurIPS (equal alternative) — 9-page main-text norm
created: 2026-06-01
---

> ## ⚠️ STRUCTURAL CHOICES — FOR MORNING REVIEW (delete this block before any submission)
>
> This file is a **new, additive** compact main-text draft. I wrote it overnight under the
> "do Tier 1/2/3" instruction; it is the highest-risk (most structural) of the three tiers, so every
> judgment call is flagged here for you to accept or override. **Nothing was deleted or moved** — the
> three full papers are untouched and become the comprehensive supplement.
>
> **1. What this is / is not.**
> - It is a tight, conference-main-length distillation (~1 abstract + 8 short sections) of
>   [[equivariance_generalization_core.md]], reorganised around the spine you named:
>   **the [A]/[B]/[C] device + a recovery-then-saturation *design rule* + encoder localisation.**
> - It is **not** wired into `papers/arxiv/build.py` and does **not** touch
>   `papers/arxiv/_combined.md` or the existing `arxiv_upload.tar.gz`. The mid-submission bundle is
>   undisturbed. If you like this draft, wiring it into the build is a separate, explicit step.
>
> **2. The one genuinely new structural decision — ✅ CONFIRMED (2026-06-01).** The core paper buries the
> Step 27→32→42→43 triangulation inside its §5 *limitations* bullets. I **promoted it to a first-class
> main section (§4, "A design rule")**, because it is the freshest contribution and reads as a *method*
> ("enrich the class, don't drop the prior; localise the cap by recovery-then-saturation"), not a caveat.
> **Reviewer approved keeping §4 promoted**, so the structure below stands.
>
> **3. What I relegated to the supplement (one-line map in the final section).** Active inference
> (§3.5/3.5.1, Steps 34/37), symmetry discovery (Steps 33/36), the sample-efficiency frontier (§3.6),
> the $(g,N)$ phase plane (§3.7), large-$N$ no-widening (Step 23), the full three-optimiser probe
> (Step 26 — headline kept, full table to supp), scene composition (§3.4), many-body transfer (Step 35),
> decoder-free goal-reaching (§3.3.2), and the LeJEPA line ([[equivariant_lejepa.md]]). Each is a real
> result; none is load-bearing for the [A]/[B]/[C]+design-rule spine, so each gets a one-line pointer.
>
> **4. How I read "preserve every number and hedge verbatim."** A compact main physically cannot carry
> all ~100 numbers from [[geometric_payoff.md]]. My rule: the compact main carries the **load-bearing
> headline numbers with their hedges**, the **complete** numeric record is preserved **unchanged** in the
> supplement, and **nothing here contradicts or softens any number or hedge**. The **[A]/[B]/[C] claim
> box** and **Proposition 1 + proof + Corollary 1** are reproduced **verbatim** from the core paper.
>
> **5. Numbers are the FINAL 5-seed values.** Step 42/43 reflect the 2026-06-01 hardening:
> the best encoder rung is now **E1-mul16 @ 0.207 (29% of the gap)** — it *flipped* from E2-mul32 @ 0.227
> at three seeds — the message lever is the null **×1.02 (~3%)**, and the lossless oracle closes **156%**.
> See the morning summary for the seed-hardening deltas.
>
> **6. Decisions resolved (2026-06-01 review).**
> - **§4 promotion:** ✅ keep §4 ("A design rule") as a first-class section (see item 2).
> - **Target venue:** **ICLR** (primary), **NeurIPS** equal alternative — JEPA is representation
>   learning, ICLR's home turf, and the geometric-DL / world-model communities are strong there. Both use
>   a **9-page main-text** norm (refs + appendix separate), which this draft is sized to.
> - **Length:** approved "about right." Final fit happens at LaTeX-conversion time; if it runs >9pp, the
>   first things to push to appendix are the Prop. 1 / Cor. 1 *proofs* and the §4.2 per-axis detail
>   (keep the design-rule box and the three headline numbers in the body).
>
> **7. Genuinely still open** (lower-stakes, can wait): whether to keep the core title or sharpen it
> toward the design rule; whether the supplement is cited as one "appendix" document or three companion
> notes; and the actual ICLR/NeurIPS `\documentclass` conversion (a separate build step — this markdown
> is the *content* draft, deliberately not yet in submission format).

---

# Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group

## Abstract

A latent world model built from an equivariant encoder $E$ and an equivariant predictor $f$ inherits a
*provable* symmetry of its training loss: when the world's dynamics genuinely carries a group $G$ acting
on latents by an **orthogonal** representation $\rho(g)$, the one-step prediction relMSE is **exactly
invariant** across the whole group, so fitting on a restricted slice of orientations *mathematically
determines* the model on the entire orbit (举一反三). We verify this end-to-end at laptop scale
(CPU/MPS, fully seeded), and make three claims. **[A]** the symmetry **survives a real Muon/AdamW $+$ EMA
$+$ VICReg run** — composed residual $\sim\!10^{-6}$ *after* optimisation, under *any* optimiser, because
the Vector-Neuron/`e3nn` weights parametrise the intertwiner space **intrinsically**. **[B]** one-step
error is **flat to five digits across the group** while a same-class non-equivariant baseline fits the
slice but breaks out-of-distribution (VN $\times1.00$ vs baseline $\times13.8$ in 2D latent, $\times17.2$
in 3D, $\times157$ over the full $\mathrm{SE}(3)$ ladder), the equivariant model **$4.5$–$7.4\times$
smaller** and frequently *better* in-distribution. **[C]** under a *matching* equivariant planner the
realised control trajectory at orientation $g$ is exactly $\rho(g)$ applied to the seen trajectory —
**float-floor-exact in 2D/$\mathrm{SO}(2)$** on real PushT and **statistically flat in
3D/$\mathrm{SE}(3)$**. We then extract a **design rule** for the regime where the equivariant class
*under-fits in-distribution*: do not relax the prior (that would forfeit [B], which is free), but
**localise which lossy component caps the fit and widen it**, read off a *recovery-then-saturation*
fingerprint. Across three sweeps the predictor degree recovers-then-saturates (a missing cross-product
primitive), the message is null ($\times1.02$), and a *lossless* point-cloud oracle through the *same*
predictor closes $156\%$ of the gap while a fixed-budget encoder ladder closes only $29\%$ — pinning the
residual cap on the **encoder's lossy translation-invariant latent**. We are explicit about scope (§7):
no binary task-success claim, [C] needs a matching planner, and against Sutton's Bitter Lesson
augmentation/scale/soft-equivariance each close at most the across-group *task* metric, never the
architecture's float-floor *exactness*.

---

## 1. Introduction

The contrarian bet of this project is that a *geometric, equivariant, latent-space* world model can earn
human-like generalisation-from-few-examples (举一反三) without simulating pixels and without brute-force
scale — that hard-wiring a symmetry the world genuinely has is a *cheaper* route to sample-efficient
generalisation than learning it from data. This note states the most robust form of that bet we can stand
behind today, at the **prediction/representation level**, in both $\mathrm{SO}(2)$ (real PushT) and
$\mathrm{SO}(3)$ (3D point clouds), and a closed-loop corollary. All experiments are laptop-scale
(CPU/MPS), fully seeded and deterministic. We state the claim precisely, then earn it.

> **If a latent world model is built from an equivariant encoder $E$ and an equivariant
> predictor $f$, and the world's dynamics genuinely carries the symmetry group $G$, then:**
>
> **[A] the *learned* model stays equivariant to the floating-point floor *after* gradient
> training** — the symmetry is not destroyed by optimisation; and
>
> **[B] one-step prediction error is *exactly flat* across the whole group** — fitting the
> dynamics on a restricted slice of orientations *determines* it on the entire orbit
> (举一反三), whereas a non-equivariant baseline of the same hypothesis class fits the slice
> but breaks out-of-distribution; and
>
> **[C] under a *matching* equivariant planner the result extends to closed loop** — the
> realised control trajectory at orientation $g$ is *exactly* $\rho(g)$ applied to the seen
> trajectory, so closed-loop control error is invariant across the group to the float floor (a
> paired test over $K{=}48$ real-PushT pose tasks: VN seen-vs-OOD block-angle change $=0$ to the
> env float floor; the non-equivariant baseline degrades with a CI excluding $0$). The corollary
> **lifts to the full 3D SE(3) group** (§3.3): on 3D point clouds with an
> SE(3)-equivariant planner the VN's OOD/seen orientation-error ratio is statistically flat
> ($[0.977,0.999]$) and disjoint from the baseline's ($[1.049,1.234]$) — there "exact" means to
> the network's $\sim\!10^{-6}$ equivariance floor (a CEM tie-flip floor), not the literal float
> zero 2D reaches; the single-plan identity $\mathrm{plan}(g\cdot x)=g\cdot\mathrm{plan}(x)$ still
> holds to $1.2\times10^{-7}$.
>
> We show [A]/[B] for $G=\mathrm{SO}(2)$ on a **real** contact-rich simulator (PushT) and for
> $G=\mathrm{SO}(3)$ on 3D **point clouds**, with the equivariant model **$4.5$–$7.4\times$
> smaller** and frequently *better* in-distribution; [C] on real-PushT pose control (2D/SO(2))
> and, lifted, on 3D point clouds under the full SE(3) group (§3.3). We make
> **no** claim here about *binary* task-success sweeps or scaling (§7), and [C] requires the
> **planner** to share the symmetry (§3.3).

[B] is not a lucky empirical trend — it is a **theorem** about the loss (§2.2), realised numerically to
five digits; [C] is that *same* theorem applied to the realised closed-loop trajectory.

![The central result in one figure](figures/killer_figure.png)

> **Figure 1.** The claim, as the three error bars a sceptic asks for. **(a)** OOD/seen
> prediction-error factor: the equivariant model is flat ($\approx\!\times1$) across every setting
> while the same-class baseline blows up $\times13$–$\times157$ (§3.2). **(b)** Five *independently
> trained* models, real-PushT closed-loop pose control: the VN's seen-vs-unseen block-angle sits on
> $y=x$ ($\Delta=-1.0°$) while the baseline sits above it ($\Delta=+9.6°$) — the contrast is the
> *architecture*, not the seed. **(c)** Deliberately breaking the SO(3) symmetry: the prior's OOD error
> rises but stays below the unconstrained baseline past 50% symmetry-breaking — an honest bracket on
> Sutton's Bitter-Lesson crossover.

**Contributions.**

- **A theorem, not a trend (§2.2).** Because the latent action $\rho(g)$ is *orthogonal*, the one-step
  relMSE is **exactly invariant** across the group — fitting a wedge *determines* the orbit (举一反三).
- **The symmetry survives training, under *any* optimiser (§2.3, §3.1)** — Result **[A]**: the
  Vector-Neuron/`e3nn` weights parametrise the intertwiner space **intrinsically**, so the
  Symmetry-Compatible-Optimizer warning (Lau & Su) leaves [A] untouched.
- **Zero-shot generalisation across the group (§3.2)** — Result **[B]**: VN flat to five digits, baseline
  $\times13.8$/$\times17.2$/$\times157$, at $4.5$–$7.4\times$ fewer parameters.
- **A closed-loop corollary (§3.3)** — Result **[C]**: float-floor-exact in 2D, statistically flat in 3D.
- **A *design rule* for the in-distribution cap (§4 — new).** When the equivariant class under-fits
  in-distribution, *enrich the class, don't drop the prior*: a recovery-then-saturation triangulation
  (predictor degree / message / encoder) **localises the residual cap to the encoder's lossy latent**,
  and the across-group [B] exactness is unaffected throughout.
- **An honest Bitter-Lesson bracket (§5, §7).** Augmentation given the whole group, scale at partial
  coverage, and a soft-equivariant interpolation each close at most the across-group *task* metric, never
  the architecture's float-floor *exactness*.

---

## 2. The exact-flatness guarantee

### 2.1 The two measurements

$G$ acts on observations by $x\mapsto g\cdot x$, on actions by $a\mapsto g\cdot a$, and on latents by
$z\mapsto\rho(g)z$ with $\rho(g)\in\mathrm{O}(d)$ **orthogonal**. The model is **$G$-equivariant** iff
$$ E(g\cdot x)=\rho(g)\,E(x), \qquad f\!\big(\rho(g)z,\;g\cdot a\big)=\rho(g)\,f(z,a). $$

**[A] — Equivariance residual.** For the composed predictor $F(x,a):=f(E(x),a)$,
$$ \Delta_{\mathrm{eq}} \;=\; \max_i \big\lVert \rho(g)\,F(x_i,a_i) - F(g\cdot x_i,\,g\cdot a_i)\big\rVert_\infty, $$
measured at a generic $g$ **both at initialisation and after training**.

**[B] — Orientation-binned relMSE (举一反三).** Train on a restricted **wedge** (e.g. $\varphi\in[0,90°)$);
rotate one held-out test set into each orientation bin (legitimate exactly when the world is
$G$-equivariant). Report pooled one-step error
$$ \mathrm{relMSE} \;=\; \frac{\sum_i \lVert F(s_i,a_i)-E(s_i')\rVert^2}{\sum_i \lVert E(s_i')-E(s_i)\rVert^2}, \qquad (<1\text{ usable, }>1\text{ worse than no change}), $$
and the **OOD factor** = (worst unseen bin) / (seen bin) — the scale-free, within-model headline.

### 2.2 The isometry theorem (why [B] is a theorem)

**Proposition 1 (exact $G$-invariance of the one-step relMSE).** *Suppose* **(H1)** *the encoder is
exactly $G$-equivariant, $E(g\cdot x)=\rho(g)E(x)$;* **(H2)** *the latent action is **orthogonal**,
$\rho(g)^\top\rho(g)=I$ (hence an isometry, $\lVert\rho(g)v\rVert=\lVert v\rVert$);* and **(H3)** *the
predictor is a $G$-intertwiner, $f(\rho(g)z,\,g\cdot a)=\rho(g)f(z,a)$, so the composed predictor
$F(x,a):=f(E(x),a)$ satisfies $F(g\cdot x,\,g\cdot a)=\rho(g)F(x,a)$. Then the relMSE of §2.1 is
$G$-invariant: for any transition set $\mathcal D=\{(s_i,a_i,s_i')\}$ and every $g\in G$,
$\mathrm{relMSE}(g\cdot\mathcal D)=\mathrm{relMSE}(\mathcal D)$. In particular the OOD factor (worst
unseen bin $/$ seen bin) of an equivariant model is exactly $1$, at any weights — at initialisation
and after any amount of training.*

*Proof.* Apply $g$ to every transition. By (H1) and (H3), each numerator term of the relMSE
transforms as
$$
\big\lVert F(g s_i,\,g a_i)-E(g s_i')\big\rVert^2
=\big\lVert \rho(g)F(s_i,a_i)-\rho(g)E(s_i')\big\rVert^2
=\big\lVert \rho(g)\big(F(s_i,a_i)-E(s_i')\big)\big\rVert^2
=\big\lVert F(s_i,a_i)-E(s_i')\big\rVert^2,
$$
the final equality by the isometry (H2); the denominator term $\lVert E(g s_i')-E(g s_i)\rVert^2$ is
invariant by the identical step. Summing numerator and denominator separately leaves the ratio
unchanged:
$$ \boxed{\;\mathrm{relMSE}(g\cdot\mathcal{D}) = \mathrm{relMSE}(\mathcal{D})\quad\text{exactly, for every }g\in G.\;} $$
No step refers to the weights, so the identity holds at every point of training. $\qquad\blacksquare$

The OOD curve is **mathematically forced** to be flat ($\times1.00$); the only deviations we observe
($\le0.2\%$) are the floating-point floor.

**Corollary 1 (closed-loop orientation-invariance, why [C] is the *same* theorem).** *Add* **(H4)** *the
planner is $G$-equivariant — its sampling distribution and constraint set commute with the group (an
isotropic search with a $g$-covariant noise model and a $G$-invariant action constraint). Then the entire
receding-horizon trajectory at orientation $g$ is the $\rho(g)$-image of the trajectory at the identity,
and any $G$-invariant control error (e.g. block-angle error) is exactly invariant across $G$.*

*Proof.* At each replan step the planner ranks candidates only through the planning cost $\mathcal C$,
which the isometry step of Proposition 1 leaves $G$-invariant, while (H4) maps the candidate set
$g$-covariantly; so the action sequence selected at orientation $g$ is exactly the $g$-image of the one
selected at the identity. Because the *world* is $G$-equivariant, the executed next state is the
$g$-image of the unrotated one; induction over the loop propagates $\rho(g)$ to the **entire** trajectory,
and a $G$-invariant error read off it is identical across $G$. $\qquad\blacksquare$

This holds to the float floor only when **both** model and planner carry the symmetry; a symmetry-broken
planner softens it to a statistical, *unbiased* tie even with an exact model (the [S] diagnostic, §3.3).

**Expressivity caveat (Schur), up front.** Scalar-weight Vector-Neuron layers are a *complete* equivariant
basis for $\mathrm{SO}(3)$ ($\mathrm{End}_{\mathrm{SO}(3)}(\mathbb{R}^3)=\mathbb{R}$), so the 3D demos'
dynamics live **inside** the model class. For $\mathrm{SO}(2)$ the standard rep has $\mathrm{End}=\mathbb C$
and scalar-weight VN omits the $90°$ generator $J$, so the 2D demos use dynamics that do not require $J$.
This is *why* [B] is a fair "equivariance generalises" test, not a "the baseline can't fit" artefact.

### 2.3 Why the symmetry survives *any* optimiser — intrinsic vs extrinsic equivariance

A sharp recent result (Lau & Su, *A Symmetry-Compatible Principle for Optimizer Design*, arXiv:2605.18106)
shows **Adam/AdamW/RMSProp are geometry-blind** — their per-coordinate $1/\sqrt{v_t}$ rescaling does not
commute with a group action on weight space, so they could *silently* break an equivariance constraint.
This does **not** touch our models. Equivariance of $x\mapsto Wx$ means $W$ lies in the **commutant**
$\mathcal C=\{W:W\rho(g)=\rho'(g)W\}$, a linear subspace. Our layers are **intrinsic**: `VNLinear`/`e3nn`
store a channel-mixing $M$ and realise $W=M\otimes I_d\in\mathcal C$ for *every* $M$ — the parametrisation's
whole image *is* the commutant, so the residual is identically zero for any weights and **any** optimiser
keeps it exact. §3.1 confirms it across three optimisers; the contrast is *extrinsic* equivariance, a free
dense $W$ merely *initialised* in $\mathcal C$, which drifts off.

---

## 3. The symmetry survives, and generalises across the group

### 3.1 [A] — the learned symmetry survives optimisation

Composed encode→predict residual after training, across all four end-to-end demos:

| demo | group / world | $\Delta_{\mathrm{eq}}$ post-train | baseline | params (VN vs base) |
|---|---|---:|---:|---:|
| explicit FM, real PushT | $\mathrm{SO}(2)$, real | $5.4\times10^{-7}$ | $0.25$ | $3360$ vs $18952$ (**5.6×**) |
| **latent JEPA**, real PushT | $\mathrm{SO}(2)$, real | $2.9\times10^{-6}$ | $3.6$ | $37$k vs $167$k (**4.5×**) |
| **latent JEPA**, 3D clouds | $\mathrm{SO}(3)$, synthetic | $3.0\times10^{-5}$ | $4.30$ | $16{,}856$ vs $124{,}512$ (**7.4×**) |

Every equivariant model keeps the symmetry to the float floor **after** training (equivariance at init is
trivial; surviving optimisation is the claim), at $4.5$–$7.4\times$ fewer parameters. Training the 3D-cloud
VN under three optimisers confirms §2.3 (composed SE(3) residual, float64, init $=$ post-train): Muon/AdamW
$3.2\times10^{-6}$, Adam $1.6\times10^{-6}$, SGD $8.9\times10^{-7}$, against a non-equivariant MLP control at
$\mathbf{0.665}$. Parametrisation dominates; the optimiser is a second-order correction. Confidence ≈
**0.95** (the row result is a theorem). *(Full three-optimiser + extrinsic-commutant tables: supplement.)*

### 3.2 [B] — zero-shot generalisation across the group (举一反三)

Train on one orientation wedge; rotate the held-out set across the group. VN flat to the float floor
everywhere; the same-class baseline fits the wedge and degrades OOD:

| demo | group / world | VN relMSE (every bin) | baseline seen → worst-OOD | OOD factor (VN \| base) |
|---|---|---:|---:|---:|
| synthetic teacher, 1-step | $\mathrm{SO}(2)$, synth | $1.4$–$1.7\times10^{-3}$ | $0.032 \to 3.41$ | **×1.17** \| ×107 |
| real PushT, full state | $\mathrm{SO}(2)$, real | $1.05\times10^{-2}$ | $1.66\!\times\!10^{-2} \to 2.69\!\times\!10^{-1}$ | **×1.00** \| ×16.2 |
| real PushT, **latent** | $\mathrm{SO}(2)$, real | $0.2559$ | $1.14 \to 15.70$ | **×1.00** \| ×13.8 |
| 3D clouds, **latent** | $\mathrm{SO}(3)$, synth | $0.228$ | $0.307 \to 5.28$ | **×1.00** \| ×17.2 |
| 3D clouds, **$+$ translation** | $\mathrm{SE}(3)$, synth | $0.228$ | $0.120 \to 18.85$ | **×1.00** \| ×157 |

The VN is flat to five digits — §2.2's theorem realised — including the **new-axis** rotations the
$z$-wedge never showed and the large translations its raw-coordinate baseline never covered ($\times157$).
Translation-invariance is **exact by construction** (the encoder centres the cloud, $E(x+t)=E(x)$), while
rotation-equivariance is **learned** and survives training (composed residual $3\times10^{-5}$).
**Mechanism, decomposed:** off-distribution the baseline keeps its near-linear *self-motion* channel
(`agent_pos` $0.089\ll1$) but **loses its model of the object's rotation** (`block_dir` $0.77\to2.33$,
worse than no-change) — the one channel a pose task depends on — while the VN is flat on every channel.

### 3.3 [C] — the theorem realised in closed loop

A **paired** design turns the exact symmetry into an experimental control: because real interior PushT is
$\mathrm{SO}(2)$-equivariant to $1.8\times10^{-5}$ px, rotating an entire reorientation task by $\Delta$
yields another valid task of identical difficulty, so the same base task runs seen ($\Delta=0$) and at OOD
rotations with env- and CEM-seed fixed. Under a **matching equivariant planner** (isotropic-$\sigma$ CEM,
$R(\Delta)$-rotated noise, a disk action constraint — identical for both models):

| paired OOD$-$seen block-angle error, 95% bootstrap CI over $K{=}48$ | mean | 95% CI |
|---|---:|---:|
| **VN (equivariant)** | $-0.000°$ | $[-0.000,+0.000]$, $\max_i\lvert d_i\rvert=4.9\times10^{-5}$ |
| **MLP (baseline)** | $+3.68°$ | $[+1.49,+6.02]$ (excludes 0) |

The VN's seen-vs-OOD change is **zero to the env float floor** — Corollary 1 realised — while the
same-class baseline on the *same* planner degrades with a CI excluding $0$. **[S] diagnostic:** re-run with
a generic-angle-broken planner and the VN's paired difference is no longer exactly zero (mean $-0.71°$, CI
$[-2.76,+1.01]$, still bracketing $0$) — *the model is exact, the planner broke the symmetry*: closed-loop
invariance needs model **and** planner.

**The SE(3) lift (3D).** The same paired design on 3D point clouds under the full SE(3) group (a closed-form
centroid channel restores translation), $K{=}24$ tasks, $1$ seen $+4$ OOD $(R,t)$:

| OOD/seen orientation-error ratio, 95% CI over $K{=}24$ | ratio | 95% CI |
|---|---:|---:|
| **VN (equivariant)** | $0.989$ | $[0.977,\ 0.999]$ |
| **MLP (baseline)** | $1.134$ | $[1.049,\ 1.234]$ |

The CIs are **disjoint** ($0.999<1.049$). Stated honestly: the 3D VN is equivariant only to `e3nn`'s
$\sim\!1.2\times10^{-6}$ architectural floor (not a float32 issue — float64 barely moves it), so the
defensible 3D headline is the **ratio separation**, not a literal zero; the single plan still commutes to
$1.2\times10^{-7}$, and the VN's $\max_i\lvert d_i\rvert=3.5°$ is a CEM **tie-flip floor, not a symmetry
break**. Confidence ≈ **0.9** on 2D [E], ≈ **0.85** on the 3D lift. *(Decoder-free goal-reaching — the
project's lone open-loop negative, resolved to $+0.527$ of the gap with across-orbit ratio $1.000$ — is
§3.3.2 in the supplement.)*

---

## 4. A design rule: enrich the equivariant class, don't drop the prior

§3 is the headline. §4 is the regime a practitioner actually hits next: an exactly-equivariant model that
is *correct across the group* (Prop. 1) but **under-fits in-distribution** because a vanilla equivariant
predictor cannot represent the dynamics' nonlinearity. The naive reaction — relax the symmetry to buy
capacity — is exactly wrong, and we make the alternative a *rule*.

### 4.1 The interaction rung exposes the cap — and the interpolation/extrapolation flip

Couple objects with an equivariant **torque**: object $i$'s points are reoriented by
$\omega_i=\kappa\,\hat r_{ij}\times a_i$ ($\kappa=0.8$), a degree-3 trilinear teacher that stays
$\mathrm{SO}(3)$-equivariant (so [B] still applies) but is **bilinear** in a way a degree-1 VN cannot form.
Three models — **VN-MP** (equiv $+$ relative-pose message), **MLP-MP** (same message, *no* equivariance):

| | in-distribution relMSE | global-orientation OOD/seen |
|---|---:|---:|
| **VN-MP** (equiv + msg) | $0.331$ | $\times1.00$ |
| **MLP-MP** (msg, no equiv) | $\mathbf{0.067}$ | $\times17.0$ |

Read across the columns and the whole bet is one experiment: **in-distribution the non-equivariant MLP
fits $\sim5\times$ better** (it can form the bilinear cross-product a vanilla VN cannot), **but across the
collapsed group that same MLP degrades $\times17$** — worse than predicting no latent change — while the
equivariant model stays $\times1.00$ (a §2.2 theorem, guarded post-training: VN-MP global SO(3) residual
$3.5\times10^{-5}$). *The better interpolator is the catastrophically worse extrapolator: capacity wins
inside the wedge, the prior wins across the group.* The cap does **not** touch [B] — equivariance is about
how error transforms *across* the group, not in-distribution capacity — so the $\times1.00$-vs-$\times17$
flip stands independent of the cap. Supplying the missing irrep (a tensor-product
$\mathbf1\otimes\mathbf1\to\mathbf1$ message) lets an *exactly* equivariant VN-TP recover **$42\%$** of the
cap ($0.331\to0.229$, $\times1.45$ better) while staying $\times1.00$ across the group; a residual
$\times2.59$ to the MLP shows the degree-1 cap was the *dominant, not the sole* bottleneck — leaving two
candidates open.

### 4.2 Localising the residual cap by *recovery-then-saturation*

Three sweeps, each holding everything fixed but one lossy component, read off a diagnostic fingerprint: a
**degree/primitive** bottleneck recovers at the first rung that supplies the missing irrep and then
**saturates**; a **capacity** bottleneck keeps closing with each doubling; a **lossless bypass** that
closes the gap pins the cap on the bypassed component.

- **Axis 1 — predictor degree (Step 32).** A ladder front-loads $L$ cross-product blocks at fixed depth /
  width / params ($d_{\max}=2^L$, $L\!\in\!\{0,1,2,3\}$). In-distribution relMSE recovers in **one step**
  ($L0\,0.263\to L1\,0.194$, $\times1.36$, $38\%$ of the cap$\to$MLP gap) and then **saturates dead-flat**
  ($L1\!\approx\!L2\!\approx\!L3\!\approx\!0.20$, top rung $+1\%$). A capacity ramp would keep closing
  toward the MLP's $0.080$; instead it stops — a **degree signature**, a missing *primitive*, not a
  capacity climb. Every rung is across-group $\times1.00$ vs the $2.4\times$-larger MLP's $\times10.5$.
  *(Three seeds — kept at three by design; it is a robust qualitative signature.)*

- **Axis 2 — the message (Step 42).** A homogeneous predictor cannot form $1/\lVert r\rVert$ at *any*
  degree, so we hand it the **unit** edge feature $\hat r$ directly (a standard TFN/NequIP/MACE ingredient,
  not the pre-formed answer), five seeds, **paired init**: M0 $[a,r]$, M1 $[a,\hat r]$ (byte-identical
  $65{,}304$ params — a pure content swap), M2 $[a,r,\hat r]$. **Null recovery:** M0 $0.259\to$ M1 $0.253$
  is only $\times1.02$ ($\sim3\%$ of the gap, within seed noise — per-seed $\mathrm{M1}-\mathrm{M0}=
  -0.012,+0.005,+0.000,-0.023,-0.000$, one seed regressing); M2 buys nothing. The message **saturates at
  the unit vector**. Every variant stays exactly equivariant (SE(3) $\le1.1\times10^{-4}$, perm $0$) and
  across-group $\times1.00$ vs the MLP's $\times10.2$ — enriching the message is zero-cost in 举一反三 even
  though it did not help fit. Reported as an honest **INCONCLUSIVE** on recovery (no guard loosened).

- **Axis 3 — the encoder, with a lossless bypass (Step 43).** Two sub-sweeps, five seeds, $80$ epochs.
  *(A) Capacity ladder:* scale the encoder's **internal** width / angular resolution at a **fixed
  $16$-vector output budget** ($\mathrm{mul}\!\in\!\{8,16,32\}$, $\ell_{\max}\!\in\!\{2,3\}$) — the cap
  moves $0.255\to$ at best $0.207$ (E1-mul16), closing only **$29\%$** of the E0$\to$MLP gap: *internal*
  encoder capacity saturates like the predictor and the message. *(B) Lossless oracle:* **bypass** the
  encoder, feeding the true per-object centred cloud $\tilde x_k=x_k-\bar x$ ($72$ numbers, vs the
  encoder's lossy $48$) into the **same degree-3 predictor** at matched $\sim\!65$k params — relMSE
  collapses to $0.00281\pm0.0004$, closing **$156\%$** of the gap, *past* even the non-equivariant MLP,
  while staying exactly equivariant (post-train SE(3) $1.8\times10^{-6}$, perm $0$, across-group
  $\times1.00$). *Lossless, equivariant, and flat coexist.*

**The triangulation is complete.** The residual interaction cap is the **encoder's lossy
translation-invariant output latent** — the $16$-vector pooling that discards the point detail the
trilinear $(\hat r_{ij}\times a_i)\times\tilde x_k$ coupling needs — **not** its internal capacity, **not**
the predictor degree (Axis 1), **not** the message (Axis 2). *Honest caveats:* the oracle relMSE is in
**point space** (read it as *solved* vs E0's *still $\sim\!0.25$*, not subtracted against E0), and the
convergence witness flags non-convergence driven **entirely by the MLP** ($27.8\%$) while the decisive
oracle plateaued at $8.2\%$ (under the $10\%$ bar) — verdict **CONFIRMED**, four guards green. Confidence
≈ **0.85** (up from $0.6$ when the cap was only inferred by elimination).

> **Design rule.** When an *exactly-equivariant* world model under-fits in-distribution, **do not relax
> the prior** — the across-group [B] exactness is free and independent of in-distribution capacity
> (Prop. 1), so dropping equivariance forfeits the whole point to buy capacity that can be had another way.
> Instead, **localise *which* lossy component caps the fit and widen *that* component inside the
> equivariant class**, read off the recovery-then-saturation fingerprint. Here that prescription is
> concrete: **widen the encoder's latent budget, keep the prior.** The lesson is constructive, and the
> safety half — *enriching the class never costs 举一反三* — holds unconditionally (every rung above stays
> $\times1.00$ across the group).

---

## 5. The Bitter-Lesson bracket

The sharpest objection is Sutton's: scale and data beat inductive bias. We stress-test the prior directly.

- **Augmentation, given the whole group.** Sweeping rotation-augmentation **coverage** on the exact teacher:
  with **full** coverage the MLP's OOD/seen *task* ratio collapses to $\times1.06$ (2D) / $\times1.46$ (3D)
  — so *on the task metric*, with the group known, augmentation is a viable substitute and the across-group
  win is **not** architecture-exclusive. **But it never reaches *exactness*:** the residual equivariance
  $\Delta_{\mathrm{eq}}$ plateaus at $7.8\times10^{-2}$ (2D) / $5.1\times10^{-2}$ (3D) even at full coverage
  — $\sim\!3\times10^{5}\times$ the VN's *weight-independent* floor ($\sim\!10^{-7}$). Augmentation
  **approximates** the symmetry, at the price of the same prior *plus* a wider training orbit; the
  architecture **is** the symmetry, for free — and only the architecture delivers the float-floor-exact
  invariance [C] is built on. Five seeds per arm.
- **Scale at partial coverage.** With coverage held *partial* (so uncovered orientations are pure
  extrapolation), sweep MLP width $\in\{64,256,1024\}$ ($\approx1.7$–$313\times$ the VN's params) $\times$
  data $N\in\{256,1024,4096\}$ ($16\times$). Scaling does **not** close the across-group gap — in 2D it
  *widens* it ($\times29.8\to\times48.9$, because more data drops the covered error faster), in 3D the
  ratio stays $\times41$–$\times106$; and a **scale-independent** $\Delta_{\mathrm{eq}}$ plateau
  ($\approx0.34$/$0.36$, $\sim\!10^6\times$ the VN floor) means $313\times$ params and $16\times$ data buy
  no exactness. **Scale substitutes for neither the coverage nor the architecture.** Five seeds per cell.
- **The soft-equivariant dial is not a free lunch.** A Residual Pathway Prior $f_\beta=f_{\mathrm{VN}}+
  f_{\mathrm{free}}$ interpolating hard$\to$free spends across-group reach **and** float-floor exactness to
  buy capacity for a broken symmetry: even at $g{=}0$ the softest model is already $\sim\!10^{5}\times$ the
  floor — **there is no "slightly soft" exactness.** The exact-and-flat-for-free corner belongs to the
  architecture alone.
- **One-step equivariance composes into a rollout guarantee.** Equivariance is **closed under
  composition**: if $\Phi_\theta$ is equivariant so is its $H$-fold iterate (one-line induction). The VN's
  across-group rollout ratio is $\times1.00$ at **every** horizon and its composed residual stays
  $\le\!2\times10^{-7}$, while the MLP's compounds monotonically ($2.3\times10^{-2}\to3.7\times10^{-1}$ over
  $H{=}1\to16$ in 2D). The prior pays a **multi-step** guarantee at the horizon a world model is actually
  planned with. Five seeds.

---

## 6. Related work

This is a *recombination*, not a new layer. **Geometric deep learning** supplies the equivariant
primitives (Cohen & Welling 2016; Weiler & Cesa 2019 / `e2cnn`; Thomas et al. 2018 / `e3nn`, Geiger &
Smidt 2022; Satorras et al. 2021; Deng et al. 2021, Vector Neurons; Bronstein et al. 2021) — we take these
as given and ask what they buy a *predictive world model*. **Equivariant RL** (van der Pol et al. 2020;
Wang, Walters & Platt 2022) hard-wires symmetry into policy/value nets and reports faster learning; we
differ in object (a **JEPA world model**) and claim (an **exact zero-shot across-the-group** statement,
not a learning-curve improvement). **JEPA / latent world models** (LeCun 2022; I-JEPA, Assran et al. 2023;
V-JEPA, Bardes et al. 2024; DreamerV3, Hafner et al. 2023) predict in latent space but obtain invariance
from **augmentation**, not architecture — our training machinery is in this family (EMA à la BYOL + a
VICReg variance hinge) but the encoder/predictor are **exactly equivariant by construction**, which §5
makes quantitative (augmentation plateaus $\sim\!10^5\times$ above the exact floor). Concurrent at-scale
work makes the same bet in a *different* group: $\gamma$-World (Liu et al. 2026) renders a multi-agent
video model permutation-equivariant over agents via a simplex rotary encoding and gets zero-shot $2\to4$
player transfer — independent, at-scale corroboration of *"an exact symmetry prior determines the model
off the training slice"*, but in **pixel space at full generative cost** and a discrete $S_P$; ours is a
**decoder-free abstract latent** carrying the continuous $\mathrm{SE}(3)$ exactly. The two are
complementary (our object-centric variant already factors $\mathrm{SE}(3)^O\rtimes S_O$).

---

## 7. Limitations & honest scope

- **No *binary task-success* claim, and [C] needs a matching planner.** The defensible [C] headline is the
  *angle-error invariance* under an equivariant planner, not a success-rate win; a generic-angle-broken
  planner softens VN exactness to an unbiased statistical tie (the [S] panel) — [C] is a property of model
  **and** planner together.
- **3D [C] is *statistical*, not float-floor-literal.** Exact only to `e3nn`'s $\sim\!10^{-6}$ floor; the
  defensible 3D headline is the ratio separation ($[0.977,0.999]$ vs $[1.049,1.234]$, disjoint).
- **Exactness requires the world to actually carry the symmetry.** Real PushT's *interior* is
  $\mathrm{SO}(2)$-equivariant to $10^{-5}$ px; block↔wall contact breaks it to $C_4$. The guarantee is
  exact only where the symmetry is real.
- **No in-distribution edge.** Inside the training orbit the higher-capacity baseline fits at least as well
  (often better, §3.2/§4.1); the across-group payoff is the whole edge. The §4 cap is precisely this
  in-distribution gap, localised — not closed by dropping the prior.
- **Everything is laptop-scale.** Nothing here speaks to scale; the Bitter-Lesson boundary (§5) is the
  standing caveat, drawn empirically rather than asserted. The defensible statement is narrow: *when the
  dynamics genuinely has a symmetry, hard-wiring it lets a latent world model reach competence across the
  whole group from far fewer interactions and generalise zero-shot at the prediction level — in 2D and 3D,
  at a fraction of the parameters.*

---

## 8. Conclusion

A geometric, equivariant, latent-space world model earns *exact* generalisation across a symmetry group —
举一反三 — without simulating pixels and without brute-force scale, in a precise and falsifiable form. The
isometry theorem that makes the encoder and one-step predictor flat by construction (§2) propagates
**unbroken** through the layers we add (§3), and is recovered to the `e3nn` floor both at initialisation
and after a real training run. Where the equivariant class *under-fits in-distribution*, the fix is a
**design rule**, not a retreat from the prior: localise the lossy component by recovery-then-saturation and
widen it — here, the encoder's latent (§4). We are equally explicit about what the bet does **not** buy: no
in-distribution edge, and an across-group payoff real only to the extent the world respects the symmetry;
augmentation and scale *approximate* equivariance but never reach the exact floor, and soft $\neq$ hard
(§5, §7). The mathematics the result rests on — Lie-group representations, intertwiners, the geometry of
the latent — is permanent capital regardless of how the empirical question resolves at scale; the wager is
that, across the group, it is also the *cheapest* route to sample-efficient generalisation, and the
evidence here is that it is.

---

## Supplement map (where every relegated result lives, unchanged)

The full numeric record and all results not load-bearing for the [A]/[B]/[C]+design-rule spine are
preserved **unchanged** in the comprehensive supplement. One-line pointers:

| Topic | Where (supplement) |
|---|---|
| Full three-optimiser + extrinsic-commutant probe (§3.1 headline here) | [[equivariance_generalization_core.md]] §3.1; Step 26 |
| Decoder-free goal-reaching (the resolved open-loop negative, $+0.527$, ratio $1.000$) | core §3.3.2; `step38_latent_goal_reaching` |
| Scene composition $\mathrm{SE}(3)^O\rtimes S_O$ (the $2\times2$ prior-attribution) | core §3.4; `step19_object_centric` |
| Active inference: curiosity invariance + $\beta$-knob; the POMDP task win | core §3.5/§3.5.1; `step20/step25` |
| Active inference de-constructed: noisy channel; generic $K$-ary search | [[geometric_payoff.md]]; `step34/step37` |
| Symmetry **discovery** ($\mathfrak{so}(3)$ read off data) and **distillation** | payoff; `step33/step36` |
| Few-body $\to$ many-body combinatorial transfer | payoff; `step35` |
| Sample-efficiency frontier (§3.6) and the $(g,N)$ phase plane (§3.7), large-$N$ | core §3.6/§3.7; `step21/step22/step23` |
| Per-experiment narratives, all tables, binary-task-success caveats | [[geometric_payoff.md]] |
| The equivariant-LeJEPA line (variance-floor regulariser on the equivariant latent) | [[equivariant_lejepa.md]] |
| Full reproducibility / provenance / guard-test map | core Appendix A |

*Spine experiments cited in the main text:* `step10/11/12/14` ([A]/[B]/[C], 2D), `step13/15/18` (3D
SE(3)), `step24` (interaction rung), `step27` (TP fix), `step32` (degree ladder), `step42` (message
ladder), `step43` (encoder ladder + lossless oracle), `step28/29/30/31` (Bitter-Lesson bracket). Each
structural claim has a matching guard test (`tests/test_step*.py`) checking equivariance/invariance at
initialisation **and** after training, with a non-equivariant control that fails.
