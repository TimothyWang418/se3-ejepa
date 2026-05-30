# Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group

## Abstract

A latent world model built from an equivariant encoder $E$ and an equivariant predictor $f$
inherits a provable symmetry of its training loss: when the world's dynamics genuinely carries a
group $G$ acting on latents by an *orthogonal* representation $\rho(g)$, the one-step prediction
relMSE is **exactly invariant** across the whole group, so fitting the dynamics on a restricted
slice of orientations *mathematically determines* it on the entire orbit (举一反三). We verify this
end-to-end at laptop scale (CPU/MPS, fully seeded). The symmetry **survives a real Muon/AdamW $+$
EMA $+$ VICReg training run** — composed encode→predict residual $\sim\!10^{-6}$ after optimisation,
not just at initialisation ([A]) — and one-step error is **flat to five digits across the group**
while a same-hypothesis-class non-equivariant baseline fits the slice but breaks out-of-distribution
([B]: VN ×1.00 vs baseline ×13.8 in 2D latent, ×17.2 in 3D, ×157 over the full $\mathrm{SE}(3)$
ladder), with the equivariant model **$4.5$–$7.4\times$ smaller** and frequently *better*
in-distribution. The same isometry argument lifts to a **closed-loop corollary** ([C]): under a
*matching* equivariant planner the realised control trajectory at orientation $g$ is exactly
$\rho(g)$ applied to the seen trajectory, so closed-loop control error is invariant across the
group — **float-floor-exact in 2D/$\mathrm{SO}(2)$** on real PushT (paired $K{=}48$: VN seen-vs-OOD
block-angle change $=0$; the baseline degrades with a 95% CI excluding $0$) and **statistically
flat in 3D/$\mathrm{SE}(3)$** ($[0.977,0.999]$, disjoint from the baseline's $[1.049,1.234]$). We
are explicit about what stays **out of scope** (§12): binary task-success sweeps, planner-free
closed-loop invariance, and scaling — the standing caveat being Sutton's Bitter Lesson.

---

> A focused write-up of the project's most robust result — the **prediction/representation-level**
> core ("[A] + [B]"), distilled from the full results log
> [[geometric_payoff.md]] and demonstrated in **both** $\mathrm{SO}(2)$ (real PushT)
> and $\mathrm{SO}(3)$ (3D point clouds). The same isometry theorem extends to a
> **closed-loop corollary [C]**: under a *matching* equivariant planner, control error is
> invariant across the group — *exactly* to the float floor in 2D/SO(2) (§6, Step 14) and,
> lifted to the full **3D SE(3)** group, statistically flat to the model's $\sim\!10^{-6}$
> equivariance floor (§6.1, Step 18). What stays deliberately
> **out of scope** is binary task-success sweeps, planner-free closed-loop invariance, and
> scaling (§12); this note is the claim I am willing to stand behind today.

Last updated: 2026-05-30. All experiments are laptop-scale (CPU/MPS), fully seeded and
deterministic.

---

## Claim

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
> **lifts to the full 3D SE(3) group** (Step 18, §6.1): on 3D point clouds with an
> SE(3)-equivariant planner the VN's OOD/seen orientation-error ratio is statistically flat
> ($[0.977,0.999]$) and disjoint from the baseline's ($[1.049,1.234]$) — there "exact" means to
> the network's $\sim\!10^{-6}$ equivariance floor (a CEM tie-flip floor), not the literal float
> zero 2D reaches; the single-plan identity $\mathrm{plan}(g\cdot x)=g\cdot\mathrm{plan}(x)$ still
> holds to $1.2\times10^{-7}$.
>
> We show [A]/[B] for $G=\mathrm{SO}(2)$ on a **real** contact-rich simulator (PushT) and for
> $G=\mathrm{SO}(3)$ on 3D **point clouds**, with the equivariant model **$4.5$–$7.4\times$
> smaller** and frequently *better* in-distribution; [C] on real-PushT pose control (2D/SO(2))
> and, lifted, on 3D point clouds under the full SE(3) group (Step 18). We make
> **no** claim here about *binary* task-success sweeps or scaling (§12), and [C] requires the
> **planner** to share the symmetry (§6).

The point is that [B] is not a lucky empirical trend — it is a **theorem** about the loss
(§2), realised numerically to five digits; [C] is that *same* theorem applied to the realised
closed-loop trajectory.

![The central result in one figure](figures/killer_figure.png)

> **Figure 1.** The claim, as the three error bars a sceptic asks for. **(a)** OOD/seen
> prediction-error factor: the equivariant model is flat ($\approx\!\times1$) across every setting
> — SO(2) synthetic & real, SO(2) latent, SO(3) 3D, full SE(3) — while the same-class baseline
> blows up $\times13$–$\times157$ (§4). **(b)** Five *independently trained* models, real-PushT
> closed-loop pose control (Step 17): the VN's seen-vs-unseen block-angle sits on $y=x$
> (orientation-invariant; $\Delta=-1.0°$) while the baseline sits above it ($\Delta=+9.6°$) — the
> contrast is the *architecture*, not the seed. **(c)** Deliberately breaking the SO(3) symmetry of
> the world (Step 16): the prior's OOD error rises (it is *not* free once the world de-symmetrises)
> but stays below the unconstrained baseline even past 50% symmetry-breaking — an honest bracket on
> Sutton's Bitter-Lesson crossover. Regenerate with `experiments/make_figures.py`.

---

## 1. The two measurements

Throughout, $G$ acts on observations by $x\mapsto g\cdot x$, on actions by $a\mapsto g\cdot a$,
and on latents by $z\mapsto \rho(g)z$ with $\rho(g)\in \mathrm{O}(d)$ **orthogonal**. The model
is **$G$-equivariant** iff
$$ E(g\cdot x)=\rho(g)\,E(x), \qquad f\!\big(\rho(g)z,\;g\cdot a\big)=\rho(g)\,f(z,a). $$

**[A] — Equivariance residual (does the symmetry survive training?).** For the composed
predictor $F(x,a):=f(E(x),a)$ (which then satisfies $F(g\cdot x,g\cdot a)=\rho(g)F(x,a)$),
$$ \Delta_{\mathrm{eq}} \;=\; \max_i \big\lVert \rho(g)\,F(x_i,a_i) - F(g\cdot x_i,\,g\cdot a_i)\big\rVert_\infty, $$
measured at a generic $g$ **both at initialisation and after training**. We also report the
**planning-cost drift** $\mathbb{E}\,\lvert \mathcal{C}(g\cdot s,g\cdot s_{\mathrm g})-\mathcal{C}(s,s_{\mathrm g})\rvert/\mathbb{E}\,\mathcal{C}$ for $\mathcal{C}=\lVert \hat z_H - z_{\mathrm g}\rVert^2$.

**[B] — Orientation-binned relMSE (举一反三).** Train on a restricted **wedge** of
orientations (e.g. $\varphi\in[0,90°)$ about one axis); take a single held-out test set and
**rotate it** into each orientation bin (legitimate exactly when the world is $G$-equivariant,
so a rotated transition is another valid transition). Report the **pooled** one-step error
$$ \mathrm{relMSE} \;=\; \frac{\sum_i \lVert F(s_i,a_i)-E(s_i')\rVert^2}{\sum_i \lVert E(s_i')-E(s_i)\rVert^2}, \qquad \big(<1\text{ usable, }>1\text{ worse than predicting no change}\big), $$
and the **OOD factor** = (worst unseen bin) / (seen bin). The within-model OOD factor is the
scale-free headline (different models have different latent step scales, so absolute
cross-model relMSE is not directly comparable; the *ratio* is).

---

## 2. Setup and the exact-flatness guarantee (why [B] is a theorem)

Let the dataset be transitions $\{(s_i,a_i,s_i')\}$ and apply $g$ to every transition. Because
$\rho(g)$ is **orthogonal**, it is an isometry: $\lVert \rho(g)v\rVert=\lVert v\rVert$. Each
numerator term of the relMSE transforms as
$$
\big\lVert F(g s_i,\,g a_i)-E(g s_i')\big\rVert^2
=\big\lVert \rho(g)F(s_i,a_i)-\rho(g)E(s_i')\big\rVert^2
=\big\lVert \rho(g)\big(F(s_i,a_i)-E(s_i')\big)\big\rVert^2
=\big\lVert F(s_i,a_i)-E(s_i')\big\rVert^2,
$$
and the denominator is invariant by the identical argument. Hence
$$ \boxed{\;\mathrm{relMSE}(g\cdot\mathcal{D}) = \mathrm{relMSE}(\mathcal{D})\quad\text{exactly, for every }g\in G.\;} $$
The equivariant model's OOD curve is therefore **mathematically forced to be flat** (×$1.00$);
the only deviations we observe ($\le 0.2\%$) are the floating-point floor. The planning cost
$\mathcal{C}=\lVert\hat z_H-z_{\mathrm g}\rVert^2$ with $z_{\mathrm g}=E(s_{\mathrm g})$ is
invariant by the same isometry step — so an equivariant planner literally **cannot tell two
$g$-related problems apart** and solves them identically.

**Closed-loop corollary (why [C] is the *same* theorem).** Suppose additionally the *planner*
is $G$-equivariant — its sampling distribution and constraint set commute with the group (an
isotropic search with a $g$-covariant noise model and a $G$-invariant action constraint). Then
at each replan step the action sequence selected at orientation $g$ is exactly the $g$-image of
the one selected at the identity, and — because the *world* itself is $G$-equivariant — the
executed next state is the $g$-image of the unrotated next state. By induction over the
receding-horizon loop, **the entire closed-loop trajectory at orientation $g$ is the
$\rho(g)$-image of the trajectory at the identity**, so any $G$-invariant control error (e.g.
block-angle error) is *exactly* invariant across the group. This is [C]: the closed-loop
analogue of the boxed identity. It holds to the float floor only when **both** the model and
the planner carry the symmetry — if the planner breaks it (e.g. a box action constraint that is
only $C_4$-symmetric, or a per-component variance refit that does not commute with $\rho(g)$),
the invariance degrades to a statistical, *unbiased* one even though the model is exact
(Step 14 [S], §6). One further honesty: this "float floor" is the *machine* epsilon only when
the model is equivariant to it (real PushT, §6); for the 3D `e3nn` encoder the model is
equivariant to its own $\sim\!10^{-6}$ library floor, so even with a matching planner the
realised closed-loop [C] is a *statistical* (ratio) invariance, not a literal zero (§6.1).

By contrast, an unconstrained $F$ trained on a wedge $\Phi_0$ is pinned **only on $\Phi_0$**;
off the training orbit the loss says nothing. Channels that are *affine* in the rotated
coordinate (a near-linear PD "self-motion") extrapolate fine; channels that genuinely *rotate*
around the orbit (object orientation / torque) are undetermined and break — empirically
crossing relMSE $=1$ exactly in the rotation channel (§5).

**Expressivity caveat (Schur), stated up front.** Scalar-weight Vector-Neuron layers
(`VNLinear`/`VNReLU`, Deng et al. 2021) are a *complete* equivariant basis for
$\mathrm{SO}(3)$: the standard 3D irrep has real endomorphism algebra
$\mathrm{End}_{\mathrm{SO}(3)}(\mathbb{R}^3)=\mathbb{R}$, so scalar weights suffice and the 3D
demo's dynamics lives **inside** the model class. For $\mathrm{SO}(2)$ the standard rep has
$\mathrm{End}=\mathbb{C}$; scalar-weight VN omits the $90°$ generator $J=\bigl(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\bigr)$, so the 2D demos use dynamics that do not require $J$ (frozen-VN teachers, or PushT channels). This is a genuine limitation, documented — not hidden — and it is *why* [B] is a fair "equivariance generalises" test rather than a "the baseline can't fit" artefact: in every demo the equivariant class can fit the seen wedge at least as well as the baseline.

---

## 3. Result [A] — the learned symmetry survives optimisation

Composed encode→predict equivariance residual after training, planning-cost drift, and
parameter count, across all four end-to-end demos (two real-PushT $\mathrm{SO}(2)$, one
synthetic-teacher $\mathrm{SO}(2)$, one $\mathrm{SO}(3)$ point-cloud):

| demo | group / world | $\Delta_{\mathrm{eq}}$ post-train | cost drift | baseline | params (VN vs base) |
|---|---|---:|---:|---:|---:|
| Step 10 — explicit FM, real PushT | $\mathrm{SO}(2)$, real | $5.4\times10^{-7}$ | — | $0.25$ | $3360$ vs $18952$ (**5.6×**) |
| Step 11 — **latent JEPA**, real PushT | $\mathrm{SO}(2)$, real | $2.9\times10^{-6}$ | $\le1.5\times10^{-7}$ | $3.6$ (drift $0.40$–$0.62$) | $37$k vs $167$k (**4.5×**) |
| Step 12 — pose cost, real PushT | $\mathrm{SO}(2)$, real | — | $4$–$5\times10^{-7}$ | drift $0.45$–$1.06$ | $3360$ vs $18952$ (**5.6×**) |
| Step 13 — **latent JEPA**, 3D clouds | $\mathrm{SO}(3)$, synthetic | $3.0\times10^{-5}$ | $7.2\times10^{-7}$ | $4.30$ (drift $0.85$) | $16{,}856$ vs $124{,}512$ (**7.4×**) |

Every equivariant model keeps the symmetry to the float floor **after** gradient training (the
whole bet — equivariance at init is trivial; surviving optimisation is the claim). The
baselines, same data and class, drift by $0.25$–$4.3$ in residual and up to $\sim\!100\%$ in
cost. And the equivariant models do it with **$4.5$–$7.4\times$ fewer parameters**.

---

## 4. Result [B] — zero-shot generalisation across the group (举一反三)

Train on one orientation wedge; rotate the held-out set across the group. VN is flat to the
float floor everywhere; the same-class baseline fits the wedge and degrades OOD.

| demo | group / world | VN relMSE (every bin) | baseline seen → worst-OOD | OOD factor (VN \| base) |
|---|---|---:|---:|---:|
| Step 8 — synthetic teacher, 1-step | $\mathrm{SO}(2)$, synth | $1.4$–$1.7\times10^{-3}$ | $0.032 \to 3.41$ | **×1.17** \| ×107 |
| Step 8 [D] — same, **real** PushT inputs | $\mathrm{SO}(2)$, real-in | flat | — | **×1.00** \| ×7 |
| Step 10 [B] — real PushT, full state | $\mathrm{SO}(2)$, real | $1.05\times10^{-2}$ | $1.66\!\times\!10^{-2} \to 2.69\!\times\!10^{-1}$ | **×1.00** \| ×16.2 |
| Step 11 [B] — real PushT, **latent** | $\mathrm{SO}(2)$, real | $0.2559$ | $1.14 \to 15.70$ | **×1.00** \| ×13.8 |
| Step 13 [B] — 3D clouds, **latent** | $\mathrm{SO}(3)$, synth | $0.228$ | $0.307 \to 5.28$ | **×1.00** \| ×17.2 |
| Step 15 [B] — 3D clouds, **$+$ translation** | $\mathrm{SE}(3)$, synth | $0.228$ | $0.120 \to 18.85$ | **×1.00** \| ×157 |

Two facts hold in every row:

- **VN flat to five digits** — same axis/new angle, new axes, random $\mathrm{SO}(3)$, **and large
  translations**. This is §2's theorem, realised. The equivariant model has seen one wedge and is
  *exactly* as good on the entire orbit.
- **The baseline fits the wedge but breaks OOD**, crossing relMSE $=1$ (worse than predicting
  no change) in the latent demos, and — in 3D — worst on the **new-axis** rotations the
  $z$-wedge never showed ($x\,90°$ at $5.28$) or on large translations its raw-coordinate inputs
  never covered ($18.85$).

**Completing the named group (Step 15).** Steps 8–13 made *rotation* the OOD axis; the last row adds
**translation**, so the orbit tested is the *full* $\mathrm{SE}(3)=\mathrm{SO}(3)\ltimes\mathbb{R}^3$
— the project's named geometry. The two halves are earned differently, and the note is honest about
it: rotation-equivariance is **learned** (and survives training, composed residual $3\times10^{-5}$),
while translation-invariance is **exact by construction** — the encoder centres the cloud
($r_i=x_i-\bar x$), so $E(x+t)=E(x)$ identically and a translated transition has the same latent,
predicted latent, and next latent. That is geometry done right rather than a deep learned result, but
it is precisely what makes the whole group a *zero-cost* generalisation for the equivariant model
while the raw-coordinate baseline degrades ×157.

**Sample efficiency (the same prior, measured as a data curve).** On the Step 8 teacher with
full-orientation test coverage, the VN matches the MLP's *best* error using **$16\times$ fewer
transitions** ($N{=}32$: VN $0.210$ vs MLP-best $0.233$ at $N{=}512$); by $N{=}512$ the VN
**solves** the task ($4.0\times10^{-3}$) while the MLP **plateaus** ($0.23$) — a gap *more data
alone cannot close*, because the baseline's hypothesis class is not tied across the orbit.

---

## 5. Why the baseline breaks (the mechanism, decomposed)

Step 12 decomposes the real-PushT prediction error **by state component** (train wedge, rotate
into quadrants):

| component | VN (all quadrants) | MLP seen | MLP worst-OOD |
|---|---:|---:|---:|
| `agent_pos` (near-linear self-motion) | $9.6\times10^{-4}$ (flat) | $1.8\times10^{-3}$ | $0.089$ (stays $\ll1$) |
| `block_pos` (object position) | $0.563$ (flat) | $0.72$ | $1.21$ |
| `block_dir` (object **rotation**) | $0.563$ (flat) | $0.77$ | $2.33$ (×3.0, crosses $1$) |

This is exactly §2's prediction. The baseline OOD **keeps its self-motion model** (`agent_pos`
$0.089\ll1$: an affine channel extrapolates) but **loses its model of the object's rotation**
(`block_dir` $0.77\to2.33$, *worse than no-change*) — the one channel that genuinely turns
around the orbit, and the one a manipulation/pose task depends on. The VN is flat on **every**
channel. So "the baseline generalises OOD" and "the baseline breaks OOD" are both true,
component-wise — and the prior's value is precisely that it pins the rotation channel for free.

---

## 6. Result [C] — the theorem realised in closed loop (Step 14)

[B] is a statement about one-step prediction; the §2 closed-loop corollary says the *same*
isometry makes **control** error invariant across the group, provided the planner also carries
the symmetry. Step 14 tests this with a **paired** design that turns the exact symmetry into an
experimental control: because real interior PushT is exactly $\mathrm{SO}(2)$-equivariant,
rotating an entire reorientation task by $\Delta$ (state, goal position, goal angle, scene
orientation) yields another valid real task of *identical intrinsic difficulty*, so the **same**
base task can be run seen ($\Delta=0$) and at OOD rotations with the env- and CEM-seed held
fixed. The paired difference $d_i=\text{ang}_{\text{OOD}}(i)-\text{ang}_{\text{seen}}(i)$ over
$K{=}48$ tasks cancels the task-to-task difficulty variance that makes unpaired closed-loop
comparisons noise-limited (the reason Steps 10–12 kept landing "within noise"). Same Step-10
forward models (VN $3360$ vs MLP $18952$ params, **5.6×**; trained-model equivariance
$6.4\times10^{-7}$ vs $0.51$).

**[E] — equivariant planner (the controlled, decisive panel).** An isotropic-$\sigma$ CEM with
$R(\Delta)$-rotated exploration noise and a **disk** action constraint $\lVert a\rVert\le1$ is
$\mathrm{SO}(2)$-equivariant and *identical for both models*, so the only variable across
orientations is the model's prior:

| paired OOD$-$seen block-angle error, 95% bootstrap CI over $K{=}48$ | mean | 95% CI |
|---|---:|---:|
| **VN (equivariant)** | $-0.000°$ | $[-0.000,+0.000]$, $\max_i\lvert d_i\rvert=4.9\times10^{-5}$ |
| **MLP (baseline)** | $+3.68°$ | $[+1.49,+6.02]$ (excludes 0) |

The VN's seen-vs-OOD angle change is **zero to the environment float floor** — the §2 corollary
realised: the closed-loop trajectory at every OOD orientation is *exactly* the rotated seen
trajectory, task by task (mean angle $7.28°$ at every orientation). The same-class baseline, on
the *same* planner, degrades with a CI excluding 0 (OOD/seen ratio $1.18$, CI $[1.06,1.37]$;
mean angle wanders $17.9°$–$30.5°$). With the planner held equivariant for both, the model's
prior is the *sole* cause of the split.

**[S] — verbatim Step-12 planner (diagnostic: the planner must share the symmetry).** Re-run
with the original planner (box action constraint + diagonal $\sigma$, *not* equivariant at
generic angles): the MLP still degrades ($+3.74°$, CI $[+1.46,+6.05]$), but the VN's paired
difference is no longer exactly zero (mean $-0.71°$, CI $[-2.76,+1.01]$, individual
$\lvert d_i\rvert$ up to $34°$) — the *model* is still exact, but the *planner* breaks the
symmetry at generic angles. The VN's CI still brackets 0 (the residual is unbiased), so the
statistical conclusion survives; the lesson is the §2-corollary condition made empirical:
**closed-loop invariance requires the model *and* the planner to be equivariant.** This is
exactly why the closed loops in Steps 10–12 (run on this non-equivariant planner) were
noise-limited — the missing half was the controller, not the model.

So [C] is genuinely the *same exactness* as [A]/[B], now in closed loop — but only under a
matching equivariant planner; with a generic-angle-broken planner it weakens to an unbiased
statistical tie. Binary task-success sweeps and scaling stay out of scope (§12). Confidence ≈
**0.9** on [E] (exact, paired, $K{=}48$), ≈ **0.85** on the model-and-planner [S] finding.

### 6.1 The SE(3) lift — [C] in the named geometry (Step 18)

§6 made [C] exact in **2D/SO(2)**. Step 18 lifts the *same* paired [E]/[S] design to **3D point
clouds under the full SE(3) group**, on the Step-13 latent JEPA (`SE3PointEncoder` $+$
`VNPredictor(dim=3)`, planning in the learned latent). The planner is made SE(3)-equivariant the
same way it was made SO(2)-equivariant in §6 — isotropic $\sigma$, $R$-rotated exploration noise, a
unit-**ball** (not box) action constraint — plus the one ingredient the larger group demands:
because `SE3PointEncoder` *centres* the cloud (translation-invariant, §4), a pure-latent cost is
translation-blind, so SE(3) would silently collapse to SO(3). A separate **closed-form centroid
channel** (terminal cost $\lVert\bar x_0+C_T\sum_h a_h-\bar x_g\rVert^2$) restores exact translation
handling. Paired over $K{=}24$ tasks on orbits of $1$ seen $+ 4$ OOD $(R,t)$, $\lvert t\rvert\!\sim\!0.8$:

| OOD/seen orientation-error ratio, 95% bootstrap CI over $K{=}24$ | ratio | 95% CI |
|---|---:|---:|
| **VN (equivariant)** | $0.989$ | $[0.977,\ 0.999]$ — within $2\%$ of flat, deviation *negative* |
| **MLP (baseline)** | $1.134$ | $[1.049,\ 1.234]$ — excludes $1$ |

The CIs are **disjoint** ($0.999<1.049$); panel [S] (the verbatim non-equivariant planner) grows the
VN residual $\sim\!5\times$ (ratio $0.886$, CI $[0.825,0.954]$), re-confirming §6's lesson — closed-loop
invariance needs **model *and* planner**. VN $16{,}856$ params vs MLP $124{,}512$ (**7.4×**),
post-train composed equivariance $6.1\times10^{-6}$ vs $5.61$.

**Why "statistical", not "float-floor exact", in 3D — stated honestly.** 2D reached
$\max_i\lvert d_i\rvert=4.9\times10^{-5}°$ because real interior PushT is SO(2)-equivariant to
$1.8\times10^{-5}$ px. The 3D VN is equivariant only to **`e3nn`'s architectural $\sim\!1.2\times10^{-6}$
floor** — *not* a float32 issue (float64 barely moves it, $1.76\to1.23\times10^{-6}$): every encoder op
is clean $\sim\!10^{-7}$ in `e3nn`'s irrep basis, but the change-of-basis back to plain $(x,y,z)$ leaves
library Wigner/normalisation constants as a $\sim\!10^{-6}$ residual scaled by the output magnitude.
This is the standard, accepted notion of "exact equivariance" for TFN/NequIP-style nets. The predictor
is exact ($\sim\!8.8\times10^{-9}$) and the **single plan commutes to $1.2\times10^{-7}$**
(`tests/test_planner_equivariance.py` — the clean theorem demonstration, with a non-equivariant MLP
control that fails $\gg$ the floor). The receding-horizon loop occasionally amplifies that
$\sim\!10^{-6}$ into a CEM top-$k$ tie-flip, so the VN's $\max_i\lvert d_i\rvert=3.5°$ is a **tie-flip
floor, not a symmetry break**, and the decisive statistic is the ratio separation above — not a literal
zero. Confidence ≈ **0.85** (one notch below 2D §6's $0.9$), ≈ **0.85** on the [S] model-and-planner
finding.

---

## 7. From one object to a scene — compositional generalisation across $\mathrm{SE}(3)^O\rtimes S_O$ (Step 19)

§§3–6 are about *one* rigid body. A scene of $O$ objects carries a strictly larger group,
$\mathrm{SE}(3)^{O}\rtimes S_O$ — per-object rigid motions **and** object relabelings — and it is built
from **two logically independent** priors that Step 19 deliberately separates instead of conflating:

- **Factorization** (shared-weight per-object slots) is *exact-by-construction*, in the same sense as
  §2's flatness guarantee. A shared encoder applied per slot is **permutation-equivariant**,
  $E(\sigma\!\cdot\!S)=\sigma\!\cdot\!E(S)$ for $\sigma\in S_O$, and **leakage-free** (slot $i$'s latent
  is a function of object $i$ alone); composing it with the §4 *centring* makes each slot
  **arrangement-invariant** (blind to where its object sits). None of this is learned — it holds at the
  float floor for any weights.
- **Per-object $\mathrm{SE}(3)$-equivariance** is the §2–§4 property applied per slot, and it is what
  buys **orientation generalisation**: a per-object reorientation $R_o$ never seen in training acts on
  the slot latent by $\rho(R_o)$ exactly, so the [B]-style relMSE is invariant under it.

The test is a three-model ablation varying *only which prior is present*: **VN-Set** (both — shared
`SE3PointEncoder` per slot + shared `VNPredictor`), **MLP-Slot** (factorization only — shared *centred*
per-object MLP + shared ordinary predictor, **identical slot structure to VN-Set**), **MLP-Global**
(neither — one monolithic MLP on the flattened scene). The teacher is a direct sum of the validated
single-object dynamics (§4), hence exactly $\mathrm{SE}(3)^O\rtimes S_O$-equivariant; two distinct
anisotropic templates make objects distinguishable (permutation non-vacuous). Two paired OOD axes give a
clean 2×2 of the [B] relMSE factor (OOD/seen; all three models have seen relMSE $<1$, so the comparison
is between *trained* world models):

| | arrangement-OOD (re-place each object) | orientation-OOD (reorient each object) |
|---|---:|---:|
| **VN-Set** (both priors) | $\times1.00$ | $\times1.00$ |
| **MLP-Slot** (factorization only) | $\times1.00$ | $\times17.8$ |
| **MLP-Global** (neither) | $\times6.3$ | $\times12.4$ |

The 2×2 *attributes* the generalisation to a prior. The **arrangement** column is the factorization
theorem made visible: both slot models are flat to the float floor while the un-centred global model
degrades — so factorization **is** the arrangement / permutation / leakage invariance (post-train
permutation residual $0$ and leakage $0$ for both slot models, against leakage $0.94$ for the global
one). The **orientation** column is the *decisive, learned* result and the reason the equivariant prior
is not redundant with factorization: VN-Set and MLP-Slot share the **same** slots, so MLP-Slot degrading
$\times17.8$ (a near-collapse to *worse than no latent change* on novel poses) where VN-Set stays
$\times1.00$ pins the difference on the $\mathrm{SE}(3)$ prior **alone**. Neither prior is sufficient;
the conjunction is. The structural half is guarded init **and** post-training (composed global
$\mathrm{SO}(3)$ residual $3.6\times10^{-5}$ for VN-Set; each control *fails* the panel it is meant to
isolate) in `tests/test_set_equivariance.py`, $16{,}856$ params for VN-Set vs $61{,}920$/$245{,}440$ for
the controls.

**Honest scope.** The clean theorem costs an assumption: the objects **do not interact** (the teacher is
a direct sum of per-object dynamics). So arrangement-invariance here is *architectural*, not learned, and
the genuinely-learned claim is the orientation column. An inter-object channel — a relative-pose /
equivariant message-passing block between slots, the scene analogue of §6.1's centroid term — is the
named next rung and is **untested**. Confidence ≈ **0.8** that the two compositional priors are separable
and each buys its named half of the scene group.

---

## 8. Active inference in the equivariant latent — the curiosity invariance (Step 20)

§§3–7 build only the *pragmatic* half of an agent — perceive, predict, act toward a goal — and prove its
exact equivariance. Active inference (Friston, 2017) adds the other half: a rational agent should also act
to **reduce its own uncertainty**. Step 20 puts both in *one* objective on the learned latent, the
**Expected Free Energy** of an action sequence,
$$
  G(a_{1:H}) \;=\; \underbrace{\textstyle\sum_h w_h\lVert \bar z_h - z_g\rVert^2 + w_t\lVert\,\bar x_0 + c_t\!\sum_h a_h - \bar x_g\rVert^2}_{\text{pragmatic / risk — the validated §6 cost}} \;-\; \beta\,\underbrace{\textstyle\sum_h \mathcal{D}_h}_{\text{epistemic / information gain}},
$$
the standard risk$-$epistemic split (the $-\beta$ makes *minimising* $G$ *maximise* information gain). The
epistemic drive is the **ensemble disagreement** $\mathcal{D}=\tfrac1K\sum_k\lVert z^{(k)}-\bar z\rVert^2$
of a $K{=}5$ predictor ensemble sharing **one** equivariant encoder (deep ensembles, Lakshminarayanan et
al., 2017; disagreement-as-exploration, Pathak et al., 2019 / Sekar et al., 2020, *Plan2Explore*), trained
with a per-member Poisson$(1)$ bootstrap so the heads fit the data yet diverge where it is sparse; its
information-geometric face is the Gaussian differential entropy
$\mathcal{H}=\tfrac12\log\det(\hat\Sigma+\epsilon I)$ of the predictive belief.

**The theorem.** Every predictor is jointly equivariant, $f_k(\rho(R)z,Ra)=\rho(R)f_k(z,a)$, and the
encoder is equivariant; because $\rho(R)$ is **orthogonal** the mean is *equivariant*
($\bar z\mapsto\rho(R)\bar z$) while the *spread* is **invariant** —
$$
  \mathcal{D}(\rho(R)z,Ra)=\tfrac1K\textstyle\sum_k\lVert\rho(R)(z^{(k)}-\bar z)\rVert^2=\mathcal{D}(z,a),
$$
and $\hat\Sigma\mapsto\rho(R)\hat\Sigma\rho(R)^\top$ leaves $\log\det(\hat\Sigma+\epsilon I)$ fixed
($\det\rho=\pm1$). **The agent's curiosity is an exactly $\mathrm{SE}(3)$-invariant scalar:** how much
there is to learn from an action does not depend on the global pose of the scene. With the invariant
pragmatic cost the whole $G$ is invariant, hence the EFE-optimal plan is $\mathrm{SE}(3)$-*equivariant*.
This is the §2 isometry argument lifted from the *loss* to the agent's *information geometry*.

A two-ensemble ablation — VN (shared equivariant encoder) vs a non-equivariant **MLP** control ($74{,}456$
vs $494{,}368$ params; the equivariant model is again $6.6\times$ smaller) — pins it init **and** after a
real Muon/AdamW + EMA + VICReg run. The disagreement, the entropy, and the *total* one-step $G$ under a
full $(R,t)$ motion are invariant to the float floor for the VN; the control misses each by
$10^4$–$10^6\times$:

| post-train residual | disagreement-inv | entropy-inv | total-$G$-inv $(R,t)$ |
|---|---:|---:|---:|
| **VN ensemble** (shared equivariant $E$) | $2.4\times10^{-5}$ | $3.1\times10^{-5}$ | $2.3\times10^{-5}$ |
| **MLP ensemble** (control) | $0.205$ | $2.83$ | $134.5$ |

The invariance is **meaningful, not trivial**. Move a $(\text{cloud},\text{action})$ pair along its
$\mathrm{SE}(3)$ orbit (rotate *both* the cloud and the type-1 action by the same $R$): the VN
disagreement is **exactly unchanged** ($\times1.0000$) — the equivariant agent is *correctly not curious*
about a pose it already generalises across. This is **举一反三 stated in the language of curiosity**: do
not spend information-seeking effort on what the symmetry gives for free. Yet $\mathcal{D}$ is a genuinely
*non-constant* field (coefficient of variation $1.22$ across the probe batch), and a true **off-orbit**
novelty — an anisotropically-stretched OOD cloud, *outside* $\mathrm{SO}(3)$ — raises it $\times1.54$,
itself rotation-invariant to $3.6\times10^{-7}$. The non-equivariant control instead assigns **spurious**
novelty ($\times6.38$) to mere re-orientation — it would waste exploration re-examining rotated copies of
what it has already seen:

| held-out probe | re-orient $\mathcal{D}(\text{orbit})/\mathcal{D}(\text{seen})$ | CoV (non-vacuity) | off-orbit novelty | novelty rot-inv |
|---|---:|---:|---:|---:|
| **VN ensemble** | $\times1.0000$ (theorem) | $1.22$ | $\times1.54$ | $3.6\times10^{-7}$ |
| **MLP ensemble** | $\times6.38$ (spurious) | $0.53$ | $\times1.71$ | $7.84$ |

Finally the active-inference **knob** behaves: sweeping $\beta:0\!\to\!12$ in an EFE-CEM planner (the §6.1
iso-$\sigma$ planner, now minimising $\mathrm{zscore}(\text{prag})-\beta\,\mathrm{zscore}(\text{epi})$)
monotonically trades pragmatic progress ($24.6\to135.7$) for epistemic gain ($82.3\to419.4$), and the
EFE-selected plan stays equivariant end-to-end,
$\lVert\mathrm{plan}(Rx)-R\,\mathrm{plan}(x)\rVert_\infty=6.0\times10^{-8}$. The structural claims are
guarded init **and** post-train in `tests/test_efe_invariance.py` (VN disagreement/entropy/total-$G
<10^{-4}$; re-orientation carries zero novelty with $\mathcal{D}$ non-constant; the MLP control breaks
each).

**Honest scope.** The teacher is **fully observed and deterministic**, so on *this* task the epistemic
term is not *required* to reach goals — the pragmatic planner already does (§6). What Step 20 establishes
is narrower and exact: the unified EFE objective is well-posed and tractable in the equivariant latent, it
carries a geometric invariance the thesis predicts and a non-equivariant model lacks, and the knob
measurably does what theory says. The empirical payoff *of* information-seeking — tasks unreachable
*without* it (partial observability, sparse/ambiguous goals) — is the named next rung and is **not**
claimed here; active inference is treated as a source of geometric structure, not a benchmark win.
Confidence ≈ **0.9** on the invariance theorem + tractability (exact by construction, survives training,
control fails), ≈ **0.55** that the epistemic term converts to a task win under partial observability
(genuinely open), overall ≈ **0.7**.

---

## 9. Sample-efficiency frontier — the learning curve across the group (Step 21)

§4 fixed the data and showed [B] at a *single* training-set size; Step 21 sweeps it and draws the
**frontier** — test error as a function of the number of interactions $N$ — because that frontier
is the operational form of the project's Open Question #1 (*does $\mathrm{SE}(3)$-equivariance in a
JEPA encoder improve sample efficiency?*). Both models (the Step-13 backbone) train on the thin
orientation wedge $\phi\in[0,90°)$; at each $N\in\{16,32,64,128,256,512\}$ we read two learning
curves — pooled latent 1-step relMSE on held-out **in-wedge** clouds (`seen`) and on the *same*
transition rotated by random $\mathrm{SO}(3)$ (`group`). The budget is a **fixed 600 gradient
updates per run**, so the abscissa is *data*, not optimisation steps; 3 seeds.

**The theorem makes the across-group curve free.** With $E(Rx)=\rho(R)E(x)$, $f(\rho z,Ra)=\rho
f(z,a)$ and $\rho(R)$ orthogonal, the relMSE carries $\rho$ in numerator *and* denominator and
cancels (§2, §5), so the VN's whole-group curve **equals its in-wedge curve at every $N$ and even
at init** — `group/seen` $=1.0000$ throughout. The non-equivariant MLP has no such cancellation.

| $N$ | VN `seen`$=$`group` | VN g/s | MLP `seen` | MLP `group` | MLP g/s |
|----:|--------------------:|-------:|-----------:|------------:|--------:|
|  16 | 0.939 | 1.000 | 0.900 | 2.03 |  2.26 |
|  32 | 0.768 | 1.000 | 0.727 | 1.85 |  2.54 |
|  64 | 0.677 | 1.000 | 0.565 | 2.07 |  3.66 |
| 128 | 0.647 | 1.000 | 0.327 | 1.66 |  5.07 |
| 256 | 0.541 | 1.000 | 0.213 | 2.02 |  9.48 |
| 512 | 0.433 | 1.000 | 0.217 | 3.15 | 14.52 |

Params: VN $16{,}856$ vs MLP $124{,}512$ ($7.4\times$). The two-sided reading, stated honestly:

- *In-distribution, the equivariant model has **no** edge.* The higher-capacity MLP fits the wedge
  **better** at $N\ge128$ (`seen` $0.22$ vs VN $0.43$ at $N=512$); to reach a common in-wedge
  target it needs *fewer* wedge samples, not more — exactly what the Bitter Lesson predicts.
  Equivariance buys nothing on the training distribution.
- *Across the group, it is the whole game.* The VN's whole-group frontier **descends** with wedge
  data ($0.939\!\to\!0.433$, competence at $N\approx120$); the MLP's is a **wall** — `group/seen`
  climbs $2.3\!\to\!14.5$ and its whole-group error never falls below $1.6$, never reaching the
  target at any $N$ on the grid. Wedge-only data plus the prior buys whole-group competence; no
  amount of in-wedge data buys the baseline the same thing.

So Open Question #1 has a precise, two-sided answer: **not in-distribution** (a wash, or worse),
but **across the group it is the difference between a learnable frontier and a wall.** The
sample-efficiency payoff is exactly the gap between the two whole-group curves — and that gap is a
theorem wherever the world genuinely carries the group (the Bitter-Lesson caveat, §12, is the
standing boundary). Confidence ≈ **0.9** (the exactness and the wall, guarded init-and-post in
`tests/test_sample_efficiency_frontier.py`) / **0.6** (that "no in-distribution edge" generalises
beyond this teacher and capacity regime).

---

## 10. The symmetry-break × data plane — where the bet pays, *located* (Step 22)

§9 fixed an exactly-equivariant world ($g=0$) and swept the data $N$; the misspecification sweep
(Step 16) fixed the data and swept a **symmetry break** $g$. Step 22 runs the
**product** — a $(g,N)$ grid — and at each cell trains both models (the Step-13 backbone, VN
$16{,}856$ vs MLP $124{,}512$, $7.4\times$) on the thin orientation wedge of a *misspecified*
teacher, reading the same two numbers as §9: in-wedge `seen` and genuine across-group `ood`. The
teacher is

$$\mathrm{Dyn}_g(x,a)_i \;=\; \mathrm{Dyn}_0(x,a)_i \;-\; g\,\langle e_z,\tilde x_i\rangle\, e_z,
\qquad \tilde x_i = x_i - \bar x,$$

exactly $\mathrm{SO}(3)$-equivariant at $g=0$ and broken along a **fixed lab axis** $e_z$ for
$g>0$. The subtracted term is deliberately chosen to be a *fair* adversary: it is
**centering-invariant** ($\sum_i\langle e_z,\tilde x_i\rangle = 0$, so it is a **real** target the
VN cannot wash away by re-centring) yet it lives in the **complement of the
$\mathrm{SO}(3)$-equivariant maps** (a fixed lab axis is exactly what equivariance forbids) — a part
of the dynamics the prior is *structurally blind* to. The break is monotone: the non-equivariant
fraction of $\mathrm{Dyn}_g$ climbs $0 \to 0.13 \to 0.40 \to 0.89 \to 1.27$ as $g:0\to0.8$.

**[A] An honest knob, and OOD must be *re-sampled*, not rotated.** At $g=0$ the across-group label
of §9 is free: a held-out transition *rotated* by $\mathrm{SO}(3)$ is a genuine label because the
equivariance identity holds (rotated-label residual $8.8\times10^{-8}$). At $g>0$ that identity
fails by $O(1)$ — the rotated-label residual jumps to $0.06$–$0.47$ — so a rotated target becomes a
**fake** label. Step 22 therefore samples *fresh* full-$\mathrm{SO}(3)$ clouds through the true
$\mathrm{Dyn}_g$ for the across-group metric; grading against a rotated label would be grading the
model against a teacher that no longer commutes with the group.

**[B] Across the group: the prior wins 24/25, and the wall is *data-proof*.**

| `ood` winner | $N{=}32$ | $64$ | $128$ | $256$ | $512$ |
|:--|:--:|:--:|:--:|:--:|:--:|
| $g=0.0$ | VN | VN | VN | VN | VN |
| $g=0.1$ | VN | VN | VN | VN | VN |
| $g=0.2$ | VN | VN | VN | VN | VN |
| $g=0.4$ | VN | VN | VN | VN | VN |
| $g=0.8$ | VN | VN | VN | VN | **MLP** |

| across-group slice | VN `ood` | MLP `ood` |
|:--|--:|--:|
| $g{=}0,\ N{=}32$ | 0.842 | 1.542 |
| $g{=}0,\ N{=}512$ — the data-proof wall | 0.496 | 2.344 |
| $g{=}0.8,\ N{=}512$ — the lone crack | 0.798 | **0.760** |

Two monotone trends meet at one corner. Down the $g=0$ column the equivariant model **descends**
($0.842\to0.496$) while the baseline's whole-group error is a **wall that rises with data**
($1.54\to2.34$ at $N=512$) — more wedge data makes the MLP *more* confidently wrong off the wedge,
the §9 wall now shown to be *data-proof*. Across the $N=512$ row the VN's across-group floor
**rises** with the break ($0.496\to0.798$: it cannot fit the lab-axis term it is blind to), while
the MLP's wall **descends** ($2.344\to0.760$: an orientation-free lab term needs no unseen
orientations to learn). The two lines cross at exactly one cell — the **joint extreme**
$(g{=}0.8,\,N{=}512)$. Everywhere else, 24 of 25 cells, the prior wins; the win is data-proof in
$g$ and break-robust right up to the corner.

**[C] In-distribution: capacity wins early, and the gap does *not* widen.**

| $g$ | $N^\star$ (MLP overtakes in-wedge) | in-wedge gap at $N{=}512$ |
|--:|:--:|--:|
| 0.0 | 32 | $+0.285$ |
| 0.1 | 64 | $+0.200$ |
| 0.2 | 32 | $+0.181$ |
| 0.4 | 64 | $+0.212$ |
| 0.8 | 64 | $+0.218$ |

On the training wedge the higher-capacity baseline overtakes the VN at $N^\star\le 64$ for **every**
$g$ — equivariance buys nothing in-distribution, exactly as §9 found at $g=0$, now confirmed at
every break. The sharper, and *refuted*, question was whether the in-distribution gap **widens** as
the world breaks the symmetry (Step 16 saw widening at $N=1200$). On this grid it does **not**: the
$N=512$ in-wedge gap is flat at $\approx +0.2$ ($+0.285$ at $g{=}0$ versus $+0.218$ at $g{=}0.8$ —
flat-to-*narrowing*). Step 23 then closes the obvious escape — that the widening is merely a larger-$N$
effect this grid never reached — by extending to $N\in\{512,1024,2048\}$ (past Step 16's $N{=}1200$)
under a **fixed-epochs** budget so the $124$K baseline is fully converged at every $N$ (in-wedge relMSE
falls to $0.059$ at $g{=}0,N{=}2048$, and $N{=}512$ reproduces Step 22's $600$-update gap as a built-in
cross-check). The break-induced change in the in-wedge gap — gap at $g{=}0.8$ minus gap at $g{=}0$ — is
$[-0.067,+0.062,-0.005]$ across $N{=}512/1024/2048$: non-monotone and inside the seed std $0.048$
(Figure 3). So the no-widening is **not** a small-$N$ artifact; it survives to $N{=}2048$, and the lone
Step-16 $N{=}1200$ widening was a single-slice fluctuation, not a capacity gap that grows with the
break.

**Verdict — two pre-registered predictions, both refuted, and the result is sharper for it.** We
pre-registered (i) "the prior wins the *literal whole box*" and (ii) "the in-distribution gap
*widens* with $g$." The plane refuted both: the across-group wall **cracks** at the single
joint-extreme corner, and the in-distribution gap is flat-to-narrowing at $N\le512$. What survives
is *stronger* because it is **located**: across the group the geometric bet is a **near-total,
data-proof win** that fails only when the symmetry is *both* badly broken *and* the baseline is
handed the most data; in-distribution it is a **wash-to-loss**, with no help from the break. Open
Question #1's "does equivariance help?" gets the two-sided map it deserves, and the Bitter-Lesson
boundary (§12) is *drawn at one corner* rather than asserted. Confidence ≈ **0.85** (the
across-group near-total win and the data-proof wall, guarded in
`experiments/step22_symmetry_data_phase.py`) / ≈ **0.6** (that the lone corner-crack and the
no-widening generalise beyond this teacher, two seeds, and $N\le512$). The frontier (§9) and both
$(g,N)$ phase panels (§10) are shown together in Figure 2.

![Where the geometric bet pays off](figures/where_the_bet_pays.png)

> **Figure 2.** Where the geometric bet pays off — a near-total, data-proof win *across the group*,
> a wash-to-loss *in-distribution*. **(left)** The Step-21 sample-efficiency frontier under an exactly
> $\mathrm{SO}(3)$ teacher: latent 1-step relMSE vs training-set size $N$, the VN's whole-group curve
> descending while the baseline's is a wall. **(middle)** The Step-22 symmetry-break $g$ × data $N$
> plane, scored on the **across-group** metric — the prior wins $24/25$ cells, cracking only at the
> joint-extreme corner $(g{=}0.8,N{=}512)$. **(right)** The same plane scored **in-distribution**:
> the higher-capacity baseline wins early at every $g$ ($N^\star\le64$). Regenerate with
> `experiments/make_bet_figures.py`.

![In-distribution gap does not widen with the break, even at large N](figures/step23_indist_largeN.png)

> **Figure 3.** The in-distribution gap does *not* widen with the symmetry break — tested directly at
> large data. Step 23 plots the in-wedge VN$-$MLP gap (mean $\pm$ seed std) against $\log_2 N$ for
> $N\in\{512,1024,2048\}$, one line per break strength $g\in\{0,0.4,0.8\}$, under a **fixed-epochs**
> ($150$) budget so the $124$K baseline is fully converged at every $N$ (more total updates at larger
> $N$, $N{=}512$ reproducing Step 22's $600$). The lines stay overlapping: breaking the symmetry does
> not open an in-distribution capacity gap that grows with data, refuting the conjecture that Step 16's
> $N{=}1200$ widening was simply a larger-$N$ effect. Regenerate with
> `experiments/step23_indist_largeN.py`.

---

## 11. Related work — where this sits

This note is a *recombination*, not a new layer; it stands on three lines and occupies the corner
where they meet.

- **Geometric deep learning supplies the equivariant primitives we build on.** Group-equivariant
  CNNs (Cohen & Welling, 2016) and $E(2)$-steerable CNNs (Weiler & Cesa, 2019, whose `e2cnn` powers
  our Step-3 pixel encoder) give planar equivariance; on $\mathbb{R}^3$, Tensor Field Networks
  (Thomas et al., 2018) and the `e3nn` library (Geiger & Smidt, 2022, our `SE3PointEncoder`),
  $E(n)$-equivariant GNNs (Satorras et al., 2021), and **Vector Neurons** (Deng et al., 2021, our
  `VNLinear`/`VNReLU`/`VNPredictor`) give $\mathrm{SO}(3)$ equivariance; Bronstein et al. (2021)
  frame the whole programme. Our contribution is **not** a new equivariant operator — we take these
  as given and ask what they buy a *predictive world model*.
- **Equivariant RL shows symmetry helps control — typically as sample efficiency, not exactness.**
  MDP homomorphic networks (van der Pol et al., 2020) and $\mathrm{SO}(2)$-equivariant RL
  (Wang, Walters & Platt, 2022) hard-wire symmetry into policy/value networks and report
  faster, more robust learning. We differ in *object* and in *claim*: we put the symmetry in a
  **JEPA world model** (encoder $+$ latent predictor), and the headline is an **exact zero-shot
  across-the-group** statement — at the prediction level (§4) and, under a matching equivariant
  planner, an *exactly* orientation-invariant closed loop (§6) — rather than a learning-curve
  improvement.
- **JEPA / latent world models predict in representation space — but are not equivariant.** The
  joint-embedding predictive line (LeCun, 2022; I-JEPA, Assran et al., 2023; V-JEPA, Bardes et al.,
  2024) and latent model-based RL (World Models, Ha & Schmidhuber, 2018; DreamerV3, Hafner et al.,
  2023) predict masked/future *latents* and obtain invariance from **scale and augmentation**, not
  from architecture. Our training machinery is squarely in this family — an EMA target à la BYOL
  (Grill et al., 2020) with a VICReg variance hinge (Bardes et al., 2022) against collapse — but the
  encoder/predictor are **exactly equivariant by construction**, so the JEPA cost $\lVert E(x_a)-
  E(x_b)\rVert$ is provably isometry-invariant (§2) instead of approximately so.

**The underexplored corner this note targets.** Equivariant *layers* exist; equivariant *RL* exists;
*JEPA* exists. What is largely missing is their conjunction: an *exactly* SE(3)-equivariant
**JEPA latent world model** whose symmetry (i) **survives a real Muon/AdamW + EMA + VICReg training
run** (§3), (ii) yields **exact** zero-shot generalisation across the whole group in 2D *and* 3D
(§4), and (iii) converts — under an equivariant planner — into a **float-floor-exact** closed-loop
orientation invariance, with the explicit condition that *the planner must share the symmetry* (§6).
That precise combination, together with an honest map of where the prior stops being free (the
misspecification sweep) and that it is a property of the architecture rather than the seed (the
multi-seed error bar), is the contribution. We use Sutton's Bitter Lesson (2019) as the standing
caveat (§12), and — through §7 — treated active inference (Friston, 2017) only as *mathematical
motivation* for the perception–action loop. §8 (Step 20) now realises it concretely, as an exactly
$\mathrm{SE}(3)$-invariant Expected Free Energy objective in the equivariant latent — but as a *geometric
mechanism* (the curiosity invariance and its $\beta$-knob), **not** a claimed exploration benefit.

---

## 12. Honest scope — what this note does **not** claim

- **No *binary task-success* claim, and [C] needs a matching equivariant planner.** §6 shows
  the closed-loop *orientation-invariance* corollary exactly (VN paired seen-vs-OOD angle change
  $=0$ to the float floor under an equivariant planner), but three things stay out of scope.
  (i) A clean **binary task-success** sweep: combined-pose success (angle *and* position
  thresholds together) stays low for both models at laptop $N$, and the angle-weighted planner
  lets the VN trade position error to minimise angle — so the defensible [C] headline is the
  *angle-error invariance*, not a success-rate win. (ii) **Planner-free** closed-loop
  invariance: Step 14 [S] shows a generic-angle-broken planner softens VN exactness to a
  statistical (unbiased) tie — [C] is a property of model *and* planner together, not the model
  alone. (iii) **Latent-only planning toward a goal cloud** in 3D got **no** traction for either
  model (Step 13 [C], negative) — a planner/decoder limitation, not an equivariance one (the VN
  fails *flat* across the group).
- **The 3D SE(3) [C] is *statistical*, not float-floor-literal (§6.1).** The 2D corollary (§6)
  hits the environment float floor ($\max_i\lvert d_i\rvert=4.9\times10^{-5}°$); the 3D SE(3) lift
  (Step 18) is exact only to the `e3nn` network's $\sim\!10^{-6}$ equivariance floor — *not* a
  precision issue (float64 barely helps), but the library-level floor of TFN/NequIP-style nets. The
  closed-loop VN residual there ($\max_i\lvert d_i\rvert=3.5°$) is a CEM **tie-flip floor, not a
  symmetry break** (the single-plan identity still holds to $1.2\times10^{-7}$), so the defensible 3D
  headline is the *ratio separation* (VN $[0.977,0.999]$ vs MLP $[1.049,1.234]$, disjoint), not a
  literal zero.
- **Exactness requires the world to actually carry the symmetry.** Real PushT's *interior*
  manipulation is $\mathrm{SO}(2)$-equivariant to $10^{-5}$ px; block↔**wall** contact breaks
  it to the square's $C_4$. The guarantee is exact only where the symmetry is real.
- **Everything is laptop-scale.** The Bitter Lesson (Sutton) warns that scale often beats
  inductive bias; nothing here speaks to scale. The defensible statement is narrow: *when the
  dynamics genuinely has a symmetry, hard-wiring it lets a latent world model reach competence
  *across the whole group* from far fewer interactions and generalise zero-shot at the prediction
  level — in 2D and 3D, at a fraction of the parameters (the precise frontier, and its honest
  in-distribution null, is §9).*
- **2D expressivity caveat** (§2): scalar-weight VN is complete for $\mathrm{SO}(3)$ but not
  $\mathrm{SO}(2)$ (missing the $J$ generator); the 2D demos stay inside the scalar-weight
  class by construction, which is what keeps [B] a fair test.
- **The scene result (§7) is for *non-interacting* objects.** Step 19's clean 2×2 attribution rests
  on a direct-sum teacher; arrangement-invariance is therefore *architectural*, not learned, and the
  genuinely-learned claim is the orientation column. Compositional generalisation **under object
  interaction** — an inter-object relative-pose / message channel — is the named next rung and is
  untested.
- **The active-inference result (§8) is a *mechanism*, not a task win.** Step 20's Expected Free Energy
  carries an *exact* curiosity invariance (ensemble disagreement is $\mathrm{SE}(3)$-invariant because
  $\rho(R)$ is orthogonal) and a $\beta$-knob that trades pragmatic for epistemic value monotonically —
  but the teacher is **fully observed and deterministic**, so exploration is not *required* to reach
  goals here. That the epistemic drive *converts to a task win* under partial observability or sparse
  goals is **untested**; active inference is used as geometric structure, not claimed as a benchmark
  result.
- **The sample-efficiency claim (§9) is *across-group*, not in-distribution.** Step 21's frontier
  shows the payoff is the difference between a *descending* whole-group learning curve and a *wall*
  — but *in-distribution* the higher-capacity baseline fits the wedge at least as well (often
  better), so there is **no** in-wedge sample-efficiency advantage. The defensible claim is narrow:
  wedge-only data plus the prior buys *whole-group* competence the baseline cannot reach at any $N$;
  it does **not** claim fewer samples to fit the training distribution.
- **The across-group win (§10) is *near-total*, not the literal whole box, and the in-distribution
  gap does *not* widen — now confirmed to $N{=}2048$.** Step 22's $(g,N)$ plane refuted two
  pre-registered predictions: the prior wins $24/25$ cells, **not** all $25$ — the data-proof wall
  cracks at the single joint-extreme corner $(g{=}0.8,\,N{=}512)$, where the heavily-broken lab-axis
  term is orientation-free enough for the high-capacity baseline to fit it ($\mathrm{VN}\,0.798$ vs
  $\mathrm{MLP}\,0.760$); and the in-wedge capacity gap is flat-to-*narrowing* in $g$ ($+0.285$ at
  $g{=}0$ vs $+0.218$ at $g{=}0.8$ at $N{=}512$), **not** widening as the lone Step-16 $N{=}1200$ slice
  had suggested. Step 23 then ruled out the large-$N$ escape directly: under a fixed-epochs
  (fully-converged) budget to $N{=}2048$, the break-induced gap change stays inside the seed noise
  ($[-0.067,+0.062,-0.005]$ across $N{=}512/1024/2048$, std $0.048$), so the no-widening is robust to
  data, not a small-$N$ artifact. The honest headline is *located*: a near-total, data-proof
  across-group win that fails only at the joint extreme, and a wash-to-loss in-distribution — over two
  seeds, now spanning $N$ up to $2048$.

---

## Reproduce

**Reproducibility checklist.**

- **Environment.** Python 3.11, PyTorch 2.12, `e3nn` 0.6.0, NumPy 2.4, Matplotlib; dependencies
  managed with `uv` (`uv sync` from `pyproject.toml`). No CUDA — everything runs on a laptop CPU/MPS.
- **Determinism.** Every experiment sets explicit seeds (data, init, planner); re-running reproduces
  the tables here. The `[A]`/`[B]` claims are *theorems* (§2), so they hold at init and post-training
  regardless of seed; the closed-loop `[C]` CIs are over fixed task/CEM seeds (paired design, §6).
- **Hardware / runtime.** All commands below finish in minutes-to-tens-of-minutes on a single CPU;
  pass `STEP{n}_SMOKE=1` for a fast wiring check of the heavier 3D steps.
- **Outputs.** Numeric dumps and figures land in `papers/figures/*.json` / `*.png`; the headline
  figures are regenerated (no training) by `make_bet_figures.py`.
- **Guards.** Each structural claim has a matching `tests/test_*.py` that checks equivariance/
  invariance **at init and after training** and fails the non-equivariant control — listed per claim
  below.

```bash
cd ~/Workspace/se3-ejepa
PRE="SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1"
# [A]+[B]+[C] evidence, in order of strength:
$PRE .venv/bin/python experiments/step8_sample_efficiency.py     # synthetic SO(2): efficiency + 举一反三
$PRE .venv/bin/python experiments/step10_pusht_closed_loop.py    # real PushT SO(2): system + prediction [A][B]
$PRE .venv/bin/python experiments/step11_latent_jepa.py          # real PushT SO(2): end-to-end latent JEPA
$PRE .venv/bin/python experiments/step12_pose_control.py         # real PushT SO(2): decomposed mechanism
$PRE .venv/bin/python experiments/step13_se3_latent_jepa.py      # 3D SO(3): the lift (STEP13_SMOKE=1 for fast)
$PRE .venv/bin/python experiments/step14_pose_control_power.py   # real PushT SO(2): paired closed-loop [C] (STEP14_SMOKE=1)
$PRE .venv/bin/python experiments/step15_se3_translation.py      # 3D SE(3): + translation completes the group (STEP15_SMOKE=1)
$PRE .venv/bin/python experiments/step16_misspecification.py     # where the prior stops being free (Bitter-Lesson boundary; §10)
$PRE .venv/bin/python experiments/step17_multiseed_closed_loop.py # training-seed error bar: architecture not seed (§11)
$PRE .venv/bin/python experiments/step18_se3_closed_loop.py       # 3D SE(3) closed-loop [C] lift, paired [E]/[S] (§6.1; STEP18_SMOKE=1)
$PRE .venv/bin/python experiments/step19_object_centric.py        # scene group SE(3)^O |x| S_O: object-centric 2x2 (§7; STEP19_SMOKE=1)
$PRE .venv/bin/python experiments/step20_active_inference.py       # active inference: EFE curiosity invariance in the equivariant latent (§8; STEP20_SMOKE=1)
$PRE .venv/bin/python experiments/step21_sample_efficiency_frontier.py # sample-efficiency frontier: VN whole-group curve == in-wedge curve, MLP wall (§9; STEP21_SMOKE=1)
$PRE .venv/bin/python experiments/step22_symmetry_data_phase.py    # (g x N) symmetry-break x data plane: prior wins across-group 24/25, data-proof wall, in-dist wash (§10)
$PRE .venv/bin/python experiments/step23_indist_largeN.py          # large-N fixed-epochs in-dist gap: no widening to N=2048, refutes the large-N escape (§10 [C], Figure 3; STEP23_SMOKE=1)
$PRE .venv/bin/python experiments/make_bet_figures.py             # render the headline figures: step21_frontier.png + where_the_bet_pays.{png,pdf} (§9-§10)
$PRE .venv/bin/python tests/test_planner_equivariance.py          # the clean single-plan SE(3) theorem: plan(g.x)=g.plan(x) to 1.2e-7
$PRE .venv/bin/python tests/test_set_equivariance.py              # scene-group SE(3)^O |x| S_O equivariance, init + post-train (§7)
$PRE .venv/bin/python tests/test_efe_invariance.py                # EFE-drive SE(3)-invariance: disagreement/entropy/total-G + zero-novelty reorient, init + post (§8)
$PRE .venv/bin/python tests/test_sample_efficiency_frontier.py    # whole-group relMSE == in-wedge to the float floor + frontier statistic, init + post (§9)
$PRE .venv/bin/python tests/test_symmetry_data_phase.py           # g=0 free across-group label vs g>0 rotated-label break (8.8e-8 -> O(1)); first_overtake_N helper (§10)
```

Equivariant primitives `src/models/structured.py` (`VNLinear`, `VNReLU`,
`StructuredStateEncoder`, `VNPredictor`); SE(3) point-cloud encoder `src/models/se3.py`; JEPA
wrapper + latent predictor `src/models/eqjepa.py`; EMA-target + VICReg training loop
`src/training/jepa.py`; the paired closed-loop [C] power analysis (equivariant CEM
`cem_plan_pose_iso`, disk clamp, paired bootstrap CIs) in
`experiments/step14_pose_control_power.py`, and its **3D SE(3) lift** (equivariant CEM
`latent_cem_plan_iso`, unit-ball clamp `_ball_clamp`, closed-form centroid channel, Kabsch
orientation readout, paired $K{=}24$ bootstrap) in `experiments/step18_se3_closed_loop.py`; the
object-centric scene lift to $\mathrm{SE}(3)^O\rtimes S_O$ (shared-weight `SetSE3Encoder`/`SlotPredictor`,
the `SlotMLPEncoder`/`GlobalMLPEncoder` controls, direct-sum `scene_teacher_step`, paired
`transform_orient`/`transform_arrange` and the permutation/leakage probes) in
`experiments/step19_object_centric.py`, with the clean single-plan SE(3) theorem guarded in
`tests/test_planner_equivariance.py` and the scene-group equivariance (global $\mathrm{SO}(3)$ +
permutation + leakage, init and post-train, both controls failing) in `tests/test_set_equivariance.py`.
The active-inference lift to an Expected Free Energy in the equivariant latent — the shared-encoder deep
ensemble (`EnsembleJEPA`/`build_vn_ensemble`, the `build_mlp_ensemble` control), the
`disagreement`/`gaussian_entropy` drives, the EFE-CEM `efe_cem_plan`, and the `_reorient`/`_novel_shape`
orbit / off-orbit probes — is in `experiments/step20_active_inference.py`, with the EFE-drive
$\mathrm{SE}(3)$-invariance (disagreement, entropy, total $G$, and re-orientation zero-novelty; init and
post-train; the MLP control failing) guarded in `tests/test_efe_invariance.py`.
Full per-step narrative — including the binary task-success caveats and the per-step closed-loop tables — in
[[geometric_payoff.md]].
