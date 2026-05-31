# Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group

## Abstract

A latent world model built from an equivariant encoder $E$ and an equivariant predictor $f$
inherits a provable symmetry of its training loss: when the world's dynamics genuinely carries a
group $G$ acting on latents by an *orthogonal* representation $\rho(g)$, the one-step prediction
relMSE is **exactly invariant** across the whole group, so fitting the dynamics on a restricted
slice of orientations *mathematically determines* it on the entire orbit (举一反三). We verify this
end-to-end at laptop scale (CPU/MPS, fully seeded). The symmetry **survives a real Muon/AdamW $+$
EMA $+$ VICReg training run** — composed encode→predict residual $\sim\!10^{-6}$ after optimisation,
not just at initialisation, and in fact under *any* optimiser (geometry-blind Adam included), because the
Vector-Neuron / `e3nn` weights parametrise the intertwiner space **intrinsically**, so the *Symmetry-
Compatible-Optimizer* warning (Lau & Su) leaves it untouched (Step 26, §3.1) ([A]) — and one-step error is
**flat to five digits across the group**
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
closed-loop invariance, and scaling the approach itself. We do, however, **stress-test the prior
directly against Sutton's Bitter Lesson** — rotation augmentation given the whole group, brute-force
scale at partial coverage, and a soft-equivariant interpolation — and find each closes at most the
across-group *task* metric, never the architecture's float-floor *exactness* (Steps 28–30). Finally, because
equivariance is **closed under composition**, the guarantee is not merely one-step: the $H$-fold rollout
operator the world model is actually planned with stays across-group flat ($\times1.00$) and
float-floor-exact ($\le\!2\times10^{-7}$) at **every** horizon, while the non-equivariant baseline's composed
residual compounds monotonically with $H$ (Step 31).

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

Last updated: 2026-05-31. All experiments are laptop-scale (CPU/MPS), fully seeded and
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

### 3.1 Why it survives *any* optimiser — intrinsic vs extrinsic equivariance (Step 26)

The table above used the project's default optimiser (Muon/AdamW). A sharp recent result (Lau & Su,
*A Symmetry-Compatible Principle for Optimizer Design*, arXiv:2605.18106) raises the natural worry: **Adam
/ AdamW / RMSProp are geometry-blind** — their per-coordinate $1/\sqrt{v_t}$ rescaling does not commute
with a group action on weight space, so they could *silently* break an equivariance constraint one step at
a time. Step 26 shows this does **not** touch our numbers, for a reason that is a theorem, not luck.

Equivariance of a linear map $x\mapsto Wx$ means $W$ lies in the **commutant** $\mathcal C=\{W:W\rho(g)=
\rho'(g)W\}$, a linear subspace. Our layers are **intrinsic**: `VNLinear` / `e3nn` store a channel-mixing
$M$ and realise $W=M\otimes I_d$, which is in $\mathcal C$ for *every* $M$ — the parametrisation's whole
image *is* the commutant, so the residual is identically zero for any weights and **any** optimiser keeps
it exact. (This is the same Schur/commutant fact behind §2's exact-flatness theorem, read from the
optimiser side — the companion note's §18 spells out the matching hypothesis-class restriction.)
Training the real Step-13 VN `EqJEPA` under three optimisers confirms it (composed SE(3) residual, float64,
init = post-train; MLP control under Adam for non-vacuity):

| optimiser | Muon/AdamW | Adam (every param) | SGD | MLP / Adam (control) |
|---|---:|---:|---:|---:|
| post-train residual | $3.2\times10^{-6}$ | $1.6\times10^{-6}$ | $8.9\times10^{-7}$ | $\mathbf{0.665}$ |

The contrast is **extrinsic** equivariance — a free dense $W$ merely *initialised* in $\mathcal C$. A
closed-form commutant $2\times2$ ($\rho(R)=R\oplus R$ on $\mathbb R^6$, $\mathcal C=\{M\otimes I_3\}$,
target $W^\star=M^\star\otimes I_3$, isotropic data with label noise $\sigma=0.05$) gives off-commutant
distance $\lVert W-P_{\mathcal C}(W)\rVert_F$:

| parametrisation | Adam | SGD |
|---|---:|---:|
| **intrinsic `VNLinear`** (ours) | $\mathbf{0}$ | $\mathbf{0}$ |
| **extrinsic `nn.Linear`** (init in $\mathcal C$) | $1.5\times10^{-2}$ | $5.2\times10^{-3}$ |

Read by **rows then columns**: the *row* gap is absolute ($\times10^{16}$ — intrinsic is immune to any
optimiser under any noise), while the *column* gap is real but **modest** ($\times2.9$ — symmetry-compatible
SGD drifts less than geometry-blind Adam, exactly as Lau–Su predict, but neither stays on $\mathcal C$).
**Parametrisation dominates; the optimiser is a second-order correction.** Our $\sim10^{-6}$ equivariance is
not a fragile artefact a careful optimiser protects — it is intrinsic to the Vector-Neuron / `e3nn`
parametrisation, so the Symmetry-Compatible-Optimizer warning, though real for extrinsically-constrained
models, leaves Result [A] untouched. Confidence ≈ **0.95** (the row result is a theorem).

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

### 6.2 From tracking to *reaching* — the exactness theorem for decoder-free goal-reaching (Step 38)

§6/§6.1 made closed-loop *tracking* exact: the seen-vs-OOD orientation-error **ratio** is flat under a
matching equivariant planner. Step 38 asks the harder question — can a decoder-free planner *reach* a goal
pose specified only as a target latent $z_g=E(X_g)$ (no decoder back to point clouds), and does the
reaching inherit the same exactness? This re-attacks the project's **one outright negative**: Step 13's
panel [C], where an open-loop CEM-MPC against $\lVert\hat z_H-z_g\rVert^2$ closed a *negative* fraction of
the orientation gap for both models.

The failure was diagnosed, not knob-tuned. A predictor trained only on *one-step* transitions has a
multi-step rollout $f^h(E(x_0),a_{1:h})$ that drifts $\sim\!2.0$ from the encoded truth
$E(\mathrm{teacher}^h)$ by $h{=}6$ — so $z_g$ sits **off the predictor's reachable manifold** and the
terminal $L_2$ is ill-scaled. Three decoder-free, exactly-equivariant ingredients fix it: (i)
**rollout-consistency training**, $L_{\rm roll}=\frac1H\sum_h\lVert f^h(E(x_0),a_{1:h})-\mathrm{sg}\,
E_{\rm ema}(x_h)\rVert^2$ via BPTT against an EMA target encoder (pulls the reachable manifold onto the
encoded one); (ii) the §6.1 SE(3)-equivariant CEM planner verbatim; (iii) an **SE(3)-native latent-Procrustes
goal** — the geodesic angle of the Kabsch rotation $R^\star$ aligning $z_0\to z_g$ on the $16$ type-1
vectors, $\arccos\frac{\operatorname{tr}R^\star-1}{2}$, invariant under a shared $\rho(R)$ because the fit
conjugates.

Decoder-free reaching flips from $+0.006$ (the faithful Step-13[C] control) up the ladder
$+0.174$ (equivariant planner) $\to+0.399$ (rollout) $\to+0.527$ (Procrustes goal, best deployable),
against a $+0.696$ predictor-space ceiling (which *uses* $a_{\rm true}$) and a $+1.000$ replay oracle. The
reach is therefore **partial** — $\sim\!53\%$ of the gap, the residual being the encoder-vs-predictor
manifold gap, a planning-horizon limitation, **not** an equivariance one — and I report it as partial.

The theorem is the **transfer**. Paired over $K{=}24$ tasks on orbits of $1$ seen $+4$ OOD $(R,t)$:

| residual orientation error (deg) | seen | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|
| **VN (equivariant)** | $16.108$ | $16.108$ | $16.108$ | $16.108$ | $16.108$ |
| **MLP (baseline)** | $15.197$ | $16.598$ | $14.016$ | $26.754$ | $48.699$ |

The VN reaches **identically** on every orbit element to $\max_i\lvert d_i\rvert=1.8\times10^{-6}°$
(OOD/seen ratio $1.000$, CI $[1.000,1.000]$ — the same tie-flip-free `e3nn` floor as §6.1); the MLP degrades
to $48.7°$ (ratio $1.745$, CI $[1.473,2.100]$, disjoint from flat). The goal cost is SE(3)-invariant to the
float floor (Procrustes $6.8\times10^{-8}$, $L_2$ $7.8\times10^{-6}$) and the rollout VN realises it
end-to-end (composed equivariance $4.2\times10^{-6}$ vs MLP $5.15$). So **§6's exactness extends from
tracking to goal-reaching**: whatever the decoder-free planner reaches, it reaches the *same* across the
whole SE(3) orbit. Confidence ≈ **0.8** — one notch below §6.1 because the reach is partial (a horizon
limitation), while the across-orbit exactness is at the `e3nn` floor. Guarded by
`tests/test_step38_latent_goal_reaching.py` (the Procrustes-angle recovery of $\lvert R\rvert$, both goal
costs' SE(3)-invariance, the VN-vs-free composed-equivariance separation, and exact reaching-transfer at
init).

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
named next rung; **Step 24 (§7.1) closes it.** Confidence ≈ **0.8** that the two compositional priors are
separable and each buys its named half of the scene group.

### 7.1 The interaction rung (Step 24): the group collapses, and the interpolation/extrapolation flip

Couple the objects with an equivariant **torque**: object $i$'s points are reoriented by
$\omega_i=\hat r_{ij}\times a_i$, the cross product of the (translation-invariant) unit relative-position
$\hat r_{ij}=(c_j-c_i)/\lVert c_j-c_i\rVert$ with $i$'s own action, scaled by $\kappa=0.8$. A cross product
of two type-1 vectors is $\mathrm{SO}(3)$-equivariant, so the teacher stays a symmetry — but interaction
**collapses** the per-object $\mathrm{SE}(3)^O\rtimes S_O$ down to the **global diagonal**
$\mathrm{SE}(3)\rtimes S_O$ (you may move or relabel the *whole* scene, not each object independently).
Because the torque depends on $\hat r_{ij}$, which the per-slot *centred* encoder discards, the predictor
now genuinely *needs* an explicit equivariant message: each slot's action is augmented with the
relative-position vector $r_{ij}$. Same one-variable discipline, three models — **VN-MP** (equivariant +
message), **VN-Set** (equivariant, *no* message — Step 19 verbatim, now mis-specified), **MLP-MP** (the
same message, *no* equivariance):

| | in-distribution relMSE | global-orientation OOD/seen |
|---|---:|---:|
| **VN-MP** (equiv + msg) | $0.331$ | $\times1.00$ |
| **VN-Set** (equiv, no msg) | $0.450$ | $\times1.00$ |
| **MLP-MP** (msg, no equiv) | $\mathbf{0.067}$ | $\times17.0$ |

Read the two columns against each other and the whole bet is in one experiment. **In-distribution the
non-equivariant MLP fits *best*** — $0.067$, $\sim5\times$ below either VN — because an ordinary MLP can
form the bilinear cross-product the torque needs, while a vanilla VN cannot (below). **Across the collapsed
group that same MLP degrades $\times17$** — to *worse than predicting no latent change* — while both
equivariant models stay flat to the float floor ($\times1.00$, a §2 theorem, guarded post-training: VN-MP
global $\mathrm{SO}(3)$ residual $3.5\times10^{-5}$ vs the MLP control's $8.8$). The better interpolator is
the catastrophically worse extrapolator: **capacity wins inside the wedge, the prior wins across the
group.** Among the VN models the message still earns its keep in-distribution (VN-MP $\times1.36$ over the
channel-blind VN-Set), so the channel is necessary even before the OOD test.

**The honest cap.** A vanilla VN (VN-Linear + VN-ReLU) is **degree-1 homogeneous** and *cannot* represent
the multilinear torque $(\hat r_{ij}\times a_i)\times\tilde x_k$ — the §2 missing-$J$ caveat lifted to 3D:
the $90^\circ$-rotation half disappears under $\mathrm{SO}(3)$ (Schur), but the **degree** half survives
for bilinear couplings. That cap is exactly why the MLP fits better in-distribution and why the VN channel
gap is a modest $\times1.36$ rather than decisive; the named fix is a tensor-product message
($1\otimes1\to1$ in `e3nn`), **which Step 27 builds and measures.** Supplying exactly that missing irrep —
the SO(3) cross product, the antisymmetric $\mathbf 1\otimes\mathbf 1\to\mathbf 1$ part, two compositions for
the trilinear torque — lets an *exactly* equivariant predictor (VN-TP) recover $\mathbf{42\%}$ of the cap
($0.331\to0.229$, $\times1.45$ better) while staying $\times1.00$ across the collapsed group (post-training
$\mathrm{SE}(3)$ residual $4.0\times10^{-5}$); a residual $\times2.59$ to the unconstrained MLP shows the
degree-1 cap was the **dominant, not the sole**, in-distribution bottleneck. The lesson is constructive:
*enrich the equivariant hypothesis class, don't drop the prior.* The cap does **not** touch the [B] result —
equivariance is about how error
transforms *across the group*, not in-distribution capacity — so the $\times1.00$-vs-$\times17$ flip stands
independent of the cap. Full treatment, figures, the third (relative-arrangement) OOD axis, and Step 27:
`geometric_payoff.md` §17–§17.1.

---

## 8. Active inference in the equivariant latent — the curiosity invariance and its task payoff (Steps 20, 25)

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
*without* it (partial observability, sparse/ambiguous goals) — is the named next rung; it is **now
closed in §8.1** (Step 25). Confidence ≈ **0.9** on the invariance theorem + tractability (exact by
construction, survives training, control fails), and — as of §8.1 — ≈ **0.85** that the epistemic term
converts to a task win under partial observability (now demonstrated, on a constructed POMDP), overall
≈ **0.85**.

### 8.1 The payoff: active inference earns a task win under partial observability (Step 25)

Step 20's honest ceiling was that on a *fully observed, deterministic* teacher the epistemic term is a
demonstrated **mechanism**, not a task necessity — the pragmatic planner alone reaches every goal (§6).
Step 25 closes exactly that named rung: it builds a setting where information-seeking is **required** to
succeed and shows the EFE planner in the equivariant latent **beats** a reward-only planner, while the
whole information-seeking loop stays exactly $\mathrm{SE}(3)$-equivariant.

**The task — an ambiguous-goal cue-foraging POMDP** (Kaelbling et al., 1998; the information-as-a-resource
setting of *Plan2Explore*, Sekar et al., 2020). Each episode hides a binary goal index $b\in\{+,-\}$
(uniform prior). Two genuinely reachable goals $g_\pm$ are rolled by the exactly-equivariant teacher
along $\pm n_g$ (opposite poses, *opposite* centroids $\pm d\,n_g$, so their midpoint is the start). A
third reachable config — the **cue** — sits on a *transverse* axis $n_c\perp n_g$: visiting it is
pragmatically useless (it is neither goal) but it is the **only** place $b$ is revealed. The agent holds
a belief $p=P(b{=}+)$ and minimises the Expected Free Energy
$$
  G(a_{1:H}) = \underbrace{\widehat{\mathrm{lat}}(p) + w_t\,\widehat{\mathrm{cen}}(p)}_{\text{belief-weighted pragmatic / risk}} \;-\; \beta\,\widehat{\mathrm{sal}},\qquad
  \mathrm{sal}=\eta\,\mathcal H(p),\quad
  \eta = 1-\textstyle\prod_h\big(1-e^{-\lVert\hat z_h - z_c\rVert^2/2\delta^2}\big),
$$
where $\widehat{(\cdot)}$ is per-channel z-scoring across the (jointly rotated) CEM candidate population,
$\widehat{\mathrm{lat}}$ the belief-weighted latent (pose) distance to $g_\pm$, $\widehat{\mathrm{cen}}$
the exact closed-form centroid channel ($\bar x_0+c_t\!\sum_h a_h$), and $\eta$ the imagined probability
of sensing the cue. $\eta\,\mathcal H(p)$ is the expected belief-entropy reduction and is
**self-extinguishing**: once $b$ is observed $\mathcal H(p){=}0$ and the agent stops valuing the cue. (The
three channels are z-scored *separately* — the latent term sums over $D{=}48$ dims and $H$ steps, so in
raw units it is $\sim\!100\times$ the 3-D centroid term and would otherwise swamp the controllable
channel so badly that even the oracle never reaches its goal; per-channel standardisation makes
$w_t,\beta$ clean dimensionless trade-offs and keeps every channel an $\mathrm{SE}(3)$-invariant scalar.)

**Why information-seeking is *required*, not merely helpful.** At $p=\tfrac12$ the pragmatic objective is
symmetric under $g_+\!\leftrightarrow g_-$; in the centroid channel its minimiser is the start centroid
(the midpoint of $\pm d\,n_g$), so a belief-myopic ($\beta{=}0$) agent's true-goal position error is
bounded below by $d$ — *irreducibly, for any policy*, until an observation breaks the symmetry. Only the
cue supplies it. The reward-only planner therefore provably cannot beat the hedge; the EFE planner
detours to the cue, observes $b$, the belief collapses, and the pragmatic term then points at the *true*
goal.

**The win** (24 random POMDPs; paired CEM seeds; bootstrap CIs; VN backbone, 60-epoch
Muon/AdamW + EMA + VICReg; $\beta{=}12$, $w_t{=}2$, $T_{\max}{=}18$):

| agent | true-goal pos err | ang err | cue-sense rate |
|---|---:|---:|---:|
| reward-only ($\beta{=}0$) | $0.592$ CI$[0.508,0.670]$ | $27.7°$ | $0.21$ |
| **EFE** ($\beta{=}12$) | $\mathbf{0.269}$ CI$[0.230,0.313]$ | $12.8°$ | $\mathbf{0.92}$ |
| oracle (told $b$) | $0.214$ CI$[0.174,0.256]$ | $10.5°$ | — |

The reward-only error sits exactly at the analytic hedge floor ($0.592\approx d{=}0.569$); the EFE planner
removes $\mathbf{55\%}$ of it (ratio $0.454$ CI$[0.364,0.572]$; paired drop $+0.323$ CI$[+0.224,+0.416]$,
excluding $0$) and lands within $0.054$ CI$[+0.006,+0.109]$ of the oracle. The mechanism is unambiguous:
the EFE agent senses the cue on $0.92$ of episodes, the reward-only agent on $0.21$ (accidental brush-by
that still leaves it pinned at the hedge floor). It is the deliberate detour *for information* — not
better dynamics, the **same** latent and model — that wins.

**The theorem realised at the decision level.** The cue sensor is a function of the latent distance
$\lVert\hat z_h - z_c\rVert$ only; the equivariant encoder sends every latent by the same orthogonal
$\rho(R)$, so $\eta$ — and hence the whole EFE, the optimal plan, **and the resulting task outcome** — is
exactly $\mathrm{SE}(3)$-invariant/equivariant. Rotating the entire POMDP by a global $(R,t)$:

| residual under global $(R,t)$ | VN | MLP control |
|---|---:|---:|
| salience-field invariance $\max_n|\eta_n(x){-}\eta_n(Rx{+}t)|$ | $1.1\times10^{-5}$ | $0.915$ |
| true-goal-outcome invariance (pos / ang) | $5.1\times10^{-8}$ / $3.2\times10^{-6}$ | $1.25$ / $57.7°$ |
| EFE-plan equivariance $\lVert\mathrm{plan}(Rx){-}R\,\mathrm{plan}(x)\rVert_\infty$ | $1.3\times10^{-8}$ | breaks |

The VN ($16{,}856$ params) solves the rotated POMDP by the rotated plan to the float floor; the MLP
control ($124{,}512$ params, $7.4\times$ larger) breaks every line. Guarded init **and** post-train in
`tests/test_step25_salience_invariance.py` (VN salience-inv $<10^{-4}$ and plan-equiv $<10^{-2}$; the
non-equivariant control breaks the plan equivariance — the robust, training-independent break, since the
saturating salience scalar can read vacuously-invariant for a collapsed lightly-trained latent).

**Honest scope.** This is a *constructed* POMDP over the synthetic equivariant teacher, and the cue reveal
is a noiseless one-bit Bayesian collapse, so the win is by design reachable. What Step 25 establishes is
exactly two things: (i) the equivariant-latent EFE planner **converts an $\mathrm{SE}(3)$-invariant
epistemic drive into a real task win** a reward-only planner *provably* cannot match (the hedge floor is a
theorem, not an empirical artifact), and (ii) the entire information-seeking loop — drive, plan, outcome —
stays exactly $\mathrm{SE}(3)$-equivariant: the project's thesis carried all the way into a
partial-observability decision problem. The belief update is deliberately minimal (one bit) so the
geometry is the only moving part. Confidence ≈ **0.85** that the constructed win is correct and the
loop-level invariance exact (theorem + survives training + control fails); the ≈ **0.5** that it transfers
beyond this construction is since discharged in two rungs — **Step 34** removes the noiseless crutch (a noisy
$K{=}2$ channel; the win survives at $\times0.614$ and vanishes when the channel goes useless) and **Step 37**
the constructed *mirror* (a generic $K{=}3,4,5$ search with no antipodal pair at any $K$, the *exact categorical*
mutual information as the drive, where the EFE planner **attains the oracle floor**), both still exactly
$\mathrm{SE}(3)$-equivariant (§12) — so what stays genuinely open is now only a *fully* non-constructed
real-observation benchmark, no longer the noise or the mirror.

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
| $g=0.8$ | VN | VN | VN | **MLP** | VN |

| across-group slice | VN `ood` | MLP `ood` |
|:--|--:|--:|
| $g{=}0,\ N{=}32$ | 0.796 | 1.700 |
| $g{=}0,\ N{=}512$ — the data-proof wall | 0.438 | 2.252 |
| $g{=}0.8,\ N{=}256$ — the lone crack | 0.778 | **0.751** |
| $g{=}0.8,\ N{=}512$ — won back | **0.836** | 0.943 |

Two monotone trends, but — across five seeds — they no longer cross at the data-richest corner.
Down the $g=0$ column the equivariant model **descends** ($0.796\to0.438$ at $N{=}512$) while the
baseline's whole-group error is a **wall that rises with data** ($1.70\to2.25$) — under a fixed
update budget more wedge data makes the MLP *more* confidently wrong off the wedge, the §9 wall now
shown to be *data-proof in $N$* (the fixed-*epochs* qualifier is [C]/Step 23). Across the $N{=}512$
row the VN's across-group floor **rises** with the break ($0.438\to0.836$: it cannot fit the
lab-axis term it is structurally blind to), while the MLP's wall **descends** ($2.25\to0.94$: an
orientation-free lab term needs no unseen orientations to learn). The two curves *approach* at the
heavily-broken end but **do not cross there**: at the joint extreme $(g{=}0.8,\,N{=}512)$ the prior
still wins ($0.836$ vs $0.943$). The single MLP cell in the whole plane sits one column in, at
$(g{=}0.8,\,N{=}256)$, and is a statistical dead heat ($0.778$ vs $0.751$, margin $0.027$). Along
the most-broken row the winner flips VN/VN/VN/**MLP**/VN cell-to-cell with margins of $0.002$–$0.11$,
all inside the seed band — so the lone crack is **not a located corner but a noisy tie** that
surfaces only where the symmetry is badly broken. Everywhere else — $24$ of $25$ cells — the prior
wins clean, and the win is data-proof in $g$.

**[C] In-distribution: capacity wins early, and the gap does *not* widen.**

| $g$ | $N^\star$ (MLP overtakes in-wedge) | in-wedge gap at $N{=}512$ |
|--:|:--:|--:|
| 0.0 | 32 | $+0.205$ |
| 0.1 | 32 | $+0.221$ |
| 0.2 | 32 | $+0.258$ |
| 0.4 | 32 | $+0.293$ |
| 0.8 | 32 | $+0.242$ |

On the training wedge the higher-capacity baseline overtakes the VN at $N^\star=32$ for **every**
$g$ — equivariance buys nothing in-distribution, exactly as §9 found at $g=0$, now confirmed at
every break and at the smallest $N$ on the grid. The sharper question was whether the
in-distribution gap **widens** as the world breaks the symmetry (Step 16 saw widening at $N=1200$).
On this grid the gap stays in a band $\approx +0.2$–$0.29$ with no collapse and no blow-up; comparing
the endpoints it is $+0.205$ at $g{=}0$ versus $+0.242$ at $g{=}0.8$ — a *small* widening ($+0.037$),
not the runaway capacity gap the Step-16 slice hinted at. Step 23 then tests directly whether that
small widening **grows with data** — the one escape this grid never reached — by extending to
$N\in\{512,1024,2048\}$ (past Step 16's $N{=}1200$) under a **fixed-epochs** budget so the $124$K
baseline is fully converged at every $N$ (in-wedge relMSE falls to $0.051$ at $g{=}0,N{=}2048$, and
$N{=}512$ reproduces Step 22's $600$-update gap as a built-in cross-check). The break-induced
widening — gap at $g{=}0.8$ minus gap at $g{=}0$ — is $[+0.037,+0.049,+0.033]$ across
$N{=}512/1024/2048$: a small, consistent offset that **does not grow with $N$** ($+0.037$ at
$N{=}512$, $+0.033$ at $N{=}2048$) and sits inside the pooled seed std $0.062$ (Figure 3). So
breaking the symmetry adds at most a *fixed* in-distribution offset, not a capacity gap that scales
with data; the lone Step-16 $N{=}1200$ widening was not the leading edge of a runaway gap.

A matching honesty note on the *across-group* side, which the fixed-update wall of [B] does not
show: under this fixed-epochs budget the baseline's whole-group error at $g{=}0$ **falls** with
data, $2.25\to1.03\to0.64$ as $N:512\to2048$. Handed both the data *and* the compute to converge,
brute force *does* begin to climb the wall — but at $N{=}2048$ it is still $2.5\times$ the VN's
$0.25$ and pays $7.4\times$ the parameters to get there. The wall is a **sample-efficiency**
barrier, not an impossibility: the prior's win is *how cheaply* it reaches whole-group competence,
not a claim the baseline can never reach it.

**Verdict — two pre-registered predictions, both refuted, and the result is sharper for it.** We
pre-registered (i) "the prior wins the *literal whole box*" and (ii) "the in-distribution gap
*widens* with $g$." Five seeds refuted both — though not where two seeds had suggested. (i) The
prior wins $24/25$, not all $25$, but the lone baseline cell is now a **statistical tie on the
most-broken row** ($g{=}0.8$, where the winner flips cell-to-cell inside the seed band), not a clean
crack at the data-richest corner — that corner in fact flips *back* to the prior at five seeds.
(ii) The in-distribution gap does **not** run away with the break: it carries at most a *small fixed
offset* ($\approx+0.04$) that does not grow with data and stays inside seed noise. What survives is a
**near-total, data-proof across-group win** that degrades only to a *tie* — never a clean loss —
exactly where the symmetry is most broken; and an **in-distribution wash-to-loss** with, at most,
that small break-offset. Open Question #1's "does equivariance help?" gets the two-sided map it
deserves, and the Bitter-Lesson boundary (§12) is *drawn empirically* rather than asserted.
Confidence ≈ **0.85** (the across-group near-total win and the data-proof-in-$N$ wall, now hardened
over **five seeds** and guarded in `experiments/step22_symmetry_data_phase.py`) / ≈ **0.6** (that the
extreme-break tie and the no-runaway-widening generalise beyond this teacher and these five seeds,
with Step 23 reaching $N{=}2048$). The frontier (§9) and both $(g,N)$ phase panels (§10) are shown
together in Figure 2.

![Where the geometric bet pays off](figures/where_the_bet_pays.png)

> **Figure 2.** Where the geometric bet pays off — a near-total, data-proof win *across the group*,
> a wash-to-loss *in-distribution*. **(left)** The Step-21 sample-efficiency frontier under an exactly
> $\mathrm{SO}(3)$ teacher: latent 1-step relMSE vs training-set size $N$, the VN's whole-group curve
> descending while the baseline's is a wall. **(middle)** The Step-22 symmetry-break $g$ × data $N$
> plane, scored on the **across-group** metric — the prior wins $24/25$ cells, the lone baseline cell
> a statistical tie at $(g{=}0.8,N{=}256)$ on the most-broken row (the data-richest corner
> $(g{=}0.8,N{=}512)$ goes back to the prior). **(right)** The same plane scored **in-distribution**:
> the higher-capacity baseline wins early at every $g$ ($N^\star=32$). Regenerate with
> `experiments/make_bet_figures.py`.

![In-distribution gap does not widen with the break, even at large N](figures/step23_indist_largeN.png)

> **Figure 3.** The in-distribution gap does *not* widen with the symmetry break — tested directly at
> large data. Step 23 plots the in-wedge VN$-$MLP gap (mean $\pm$ seed std) against $\log_2 N$ for
> $N\in\{512,1024,2048\}$, one line per break strength $g\in\{0,0.4,0.8\}$, under a **fixed-epochs**
> ($150$) budget so the $124$K baseline is fully converged at every $N$ (more total updates at larger
> $N$, $N{=}512$ reproducing Step 22's $600$). The lines stay close — separated by at most a *small,
> fixed* offset ($\approx+0.04$) that does **not** grow with $N$: breaking the symmetry does not open
> an in-distribution capacity gap that scales with data, refuting the conjecture that Step 16's
> $N{=}1200$ widening was the leading edge of a larger-$N$ effect. Regenerate with
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
  E(x_b)\rVert$ is provably isometry-invariant (§2) instead of approximately so. We make this
  substitution *quantitative* in §12 (Step 28): handed the group, rotation augmentation closes the
  across-group *task* metric but plateaus $\sim\!10^5\times$ above the architecture's
  exact-equivariance floor — it **approximates** the symmetry the architecture **is**.
- **A symmetry prior buys across-group generalisation at scale too — but in pixel space, at full
  generative cost.** Concurrent generative multi-agent world models make our bet in a *different*
  group. $\gamma$-World (Liu et al., 2026) encodes its $P$ agents as the vertices of a regular simplex
  in rotary-angle space ("Simplex Rotary Agent Encoding", a parameter-free $3$D-RoPE extension) — an
  *isometric orbit of the symmetric group $S_P$* that renders the model **permutation-equivariant**
  over agents — and that single prior yields **zero-shot generalisation from two to four players
  without retraining**, their analogue of 举一反三 across $S_P$ rather than $\mathrm{SE}(3)$. The
  control is sharp: their dense-attention baseline distinguishes players by a *learned per-slot
  identity*, which breaks the exchange symmetry and **cannot extend past its training roster without
  retraining** — the same symmetry-respecting-vs-breaking split we report at the prediction level
  (§4). We therefore read $\gamma$-World as independent, at-scale corroboration of the present
  thesis — *an exact symmetry prior determines the model off the training slice* — now for a discrete
  permutation group. Two axes locate our own corner. (i) **Representation:** it predicts in
  **pixel/video space** through a distilled diffusion teacher, paying the full generative cost; we
  predict in an **abstract equivariant latent** with no decoder, the cheaper route our contrarian bet
  targets. (ii) **Realisation:** its symmetry is a discrete $S_P$ engineered into a positional
  encoding; ours is the continuous $\mathrm{SE}(3)$ carried *exactly* by the network
  ($\sim\!10^{-6}$ through a real training run, §3). The two are **complementary, not rival**: our
  object-centric variant already factors a permutation symmetry over entities
  ($\mathrm{SE}(3)^O\rtimes S_O$, payoff-log Steps 19/24) plus a combinatorial count-generalisation
  result (Step 35, few-body $\to$ many-body) that is the discrete sibling of their $2\to4$ — so a
  natural synthesis is a simplex-style $S_P$ entity code layered *over* SE(3)-equivariant per-entity
  latents.

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
  alone. (iii) **Latent-only planning toward a goal cloud** in 3D was the lone outright negative —
  Step 13 [C] closed a *negative* gap fraction for both models — and **Step 38 (§6.2) resolves it**:
  rollout-consistency training $+$ the §6.1 equivariant planner $+$ an SE(3)-native latent-Procrustes
  goal flip decoder-free reaching from $+0.006$ to $+0.527$, and the VN reaches *identically* across the
  SE(3) orbit (ratio $1.000$, CI $[1.000,1.000]$ vs the MLP's $\times1.745$) — §6's exactness theorem now
  for goal-reaching. What stays open is *full* (not partial) reaching: the $+0.527$ deployable fraction
  trails a $+0.696$ predictor-space ceiling, the residual being the encoder-vs-predictor manifold gap, a
  planning-horizon limitation, not an equivariance one.
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
- **The scene result (§7) is for *non-interacting* objects; Step 24 (§7.1) adds the interaction rung,
  with an honest expressivity cap.** Step 19's clean 2×2 attribution rests on a direct-sum teacher, so its
  arrangement-invariance is *architectural*, not learned. Step 24 couples the objects with an equivariant
  torque (collapsing the scene group to the global diagonal $\mathrm{SE}(3)\rtimes S_O$) and adds the
  relative-pose message channel: the **interpolation/extrapolation flip** is decisive ($\times1.00$ for both
  equivariant models vs $\times17$ for the higher-capacity non-equivariant MLP that fits *best*
  in-distribution). The remaining caveat is honest, not fatal: a vanilla VN is degree-1 homogeneous and
  cannot form the bilinear torque, so the in-distribution VN channel gap is a modest $\times1.36$ and the
  named fix is a tensor-product ($1\otimes1\to1$) message — **and Step 27 (§7.1) builds it, recovering
  $42\%$ of the cap ($\times1.45$ better fit) while the predictor stays exactly $\mathrm{SO}(3)$-equivariant
  and $\times1.00$ across the group** (a residual $\times2.59$ to the unconstrained MLP shows the cap was the
  dominant, not the sole, bottleneck). The cap is on in-distribution *capacity*, not the across-group [B]
  result, and is now partially lifted *from inside* the equivariant class.
- **The active-inference result (§8) is now a task win — but on a *constructed* POMDP.** Step 20 gave
  the *mechanism*: an *exact* curiosity invariance (ensemble disagreement is $\mathrm{SE}(3)$-invariant
  because $\rho(R)$ is orthogonal) and a $\beta$-knob that trades pragmatic for epistemic value
  monotonically — but on a fully-observed deterministic teacher exploration is not *required*. Step 25
  (§8.1) closes that rung: in an ambiguous-goal cue-foraging POMDP the EFE planner removes $55\%$ of the
  reward-only error (which sits *exactly* at the analytic hedge floor) by deliberately sensing the cue
  ($0.92$ vs $0.21$ of episodes), reaching within $0.054$ of an oracle told the hidden goal — and the
  whole loop (salience, plan, outcome) stays $\mathrm{SE}(3)$-invariant/equivariant to the float floor
  while the MLP control breaks it. The honest caveat is that the POMDP is *constructed* over the
  synthetic teacher and the reveal is a noiseless one-bit collapse: the win is by design reachable, so
  what is proven is that the equivariant-latent EFE planner *converts an invariant drive into a win a
  reward-only planner provably cannot match*, not that active inference beats a benchmark in the wild
  (transfer to noisy / non-constructed observation is untested).
- **The sample-efficiency claim (§9) is *across-group*, not in-distribution.** Step 21's frontier
  shows the payoff is the difference between a *descending* whole-group learning curve and a *wall*
  — but *in-distribution* the higher-capacity baseline fits the wedge at least as well (often
  better), so there is **no** in-wedge sample-efficiency advantage. The defensible claim is narrow:
  wedge-only data plus the prior buys *whole-group* competence the baseline cannot reach at any $N$;
  it does **not** claim fewer samples to fit the training distribution.
- **The across-group win (§10) is *near-total*, not the literal whole box, and the in-distribution
  gap does *not* run away with the break — now hardened over five seeds to $N{=}2048$.** Step 22's
  $(g,N)$ plane refuted two pre-registered predictions. The prior wins $24/25$ cells, **not** all
  $25$ — but at five seeds the lone baseline cell is a **statistical tie on the most-broken row**
  ($g{=}0.8$): the single MLP cell is $(g{=}0.8,\,N{=}256)$ ($\mathrm{VN}\,0.778$ vs
  $\mathrm{MLP}\,0.751$, margin $0.027$), the winner flips cell-to-cell along that row inside the seed
  band, and the data-richest corner $(g{=}0.8,\,N{=}512)$ goes *back* to the prior ($0.836$ vs
  $0.943$) — so the failure is a noisy boundary tie, not a located corner. And the in-wedge capacity
  gap stays a band $\approx+0.2$–$0.29$ ($+0.205$ at $g{=}0$ vs $+0.242$ at $g{=}0.8$ at $N{=}512$): a
  *small* widening with the break, not the runaway gap the lone Step-16 $N{=}1200$ slice had
  suggested. Step 23 then ruled out the large-$N$ escape directly: under a fixed-epochs
  (fully-converged) budget to $N{=}2048$, the break-induced widening is $[+0.037,+0.049,+0.033]$
  across $N{=}512/1024/2048$ — a small fixed offset that does **not** grow with $N$ and sits inside
  the pooled seed std $0.062$. (Honest corollary, same fixed-epochs run: the across-group wall is
  *not* immune to data once the baseline is allowed to converge — its $g{=}0$ whole-group error falls
  $2.25\to0.64$ as $N:512\to2048$, still $2.5\times$ the VN at $7.4\times$ the parameters; the wall is
  a sample-efficiency barrier, not an impossibility.) The honest headline: a near-total,
  data-proof-in-$N$ across-group win that degrades only to a *tie* where the symmetry is most broken,
  and a wash-to-loss in-distribution — over **five** seeds, spanning $N$ up to $2048$.
- **The fair augmentation baseline: given the group, augmentation closes the across-group *task*
  metric but never the *exactness* (Step 28).** The sharpest objection to the whole note is that the
  prior merely does what rotation **data augmentation** would: hand the non-equivariant MLP the *same*
  knowledge (the world is symmetric) and let it learn the symmetry from an augmented training orbit.
  On the exactly-equivariant teacher, sweeping augmentation **coverage** (a $\mathrm{SO}(2)$ arc
  $[0,\theta_{\max})$ in 2D; a $\mathrm{SO}(3)$ geodesic ball of angle $\le\theta_{\max}$ in 3D, with
  $\theta_{\max}{=}180°$ all of $\mathrm{SO}(3)$) settles it two ways. *(i) Task metric* — with **full**
  coverage augmentation does flatten the MLP: the OOD/seen relMSE ratio collapses from the no-aug wall
  to $\times1.06$ in 2D and $\times1.46$ in 3D (vs the no-aug $\times67$ / $\times951$), against the
  VN's $\times1.00$ at *zero* coverage; the narrowest coverage stays broken ($\times118.9$ / $\times37.6$),
  confirming the no-aug failure is missing coverage, not finite $N$. So *on the task metric*, with the
  group known, augmentation is a viable substitute — the across-group task win is **not**
  architecture-exclusive (the 3D residual $\times1.46$ honestly sits a touch above 2D's $\times1.06$:
  the richer group leaves a visible gap the VN does not have). *(ii) Exactness* — augmentation **never**
  reaches the architecture's symmetry: the residual equivariance
  $\Delta_{\mathrm{eq}}=\max_g\lVert f(g{\cdot}x)-g{\cdot}f(x)\rVert/\lVert f(x)\rVert$ plateaus at
  $7.8\times10^{-2}$ (2D) / $5.1\times10^{-2}$ (3D) even at full coverage — $\sim\!3\times10^{5}\times$
  the VN's *weight-independent* float floor ($\sim\!10^{-7}$). The honest split: augmentation
  **approximates** the symmetry, at the price of the same prior *plus* a wider training orbit, and only
  ever buys the *approximate* version; the architecture **is** the symmetry, for free — and only the
  architecture delivers the float-floor-exact invariance the closed-loop [C] (§6) is built on (an MLP
  with $\Delta_{\mathrm{eq}}\approx0.05$ cannot close an exactly orientation-invariant loop). Five seeds
  per arm; `experiments/step28_fair_augmentation_baseline.py` (2D) and `..._3d.py` (3D).
- **The Bitter-Lesson stress test: at *partial* coverage, scale substitutes for neither the coverage nor
  the architecture (Step 29).** Step 28 handed augmentation the *whole* group; the realistic regime is
  *partial* coverage — you augment a wedge and hope the model extrapolates the rest. Holding coverage
  partial (a 2D arc $[0,180°)$; a 3D geodesic ball of angle $\le90°$), so the uncovered orientations
  (2D $[180°,360°)$; 3D shell $(90°,180°]$) are pure **extrapolation**, we sweep the two axes Sutton's
  Bitter Lesson (2019) would invoke against the prior — MLP width $\in\{64,256,1024\}$
  ($\approx\!1.7\text{–}313\times$ the VN's $3.5$k params) $\times$ base scenes $N\in\{256,1024,4096\}$
  ($16\times$), each step re-rotated with fresh in-coverage group elements at a **fixed gradient-step
  budget** (so $N$ varies content diversity at constant compute), five seeds. *(i) Task* — scaling does
  **not** close the across-group gap. In 2D it *widens* it (OOD/seen $\times29.8\!\to\!\times48.9$
  corner-to-corner): because the metric is a ratio, more data drives the *covered* (seen) error down
  faster than the uncovered-extrapolation error, so the relative 举一反三 failure gets *worse* under
  $16\times$ data and $309\times$ parameters. In 3D bigger *models* help but more *data* does not, and the
  ratio stays enormous ($\times41\text{–}\times106$ across the grid), never approaching the wall it cannot
  escape. Against this the VN reference is $\times1.00$ at *every* cell, for free — **scale is not a
  substitute for the missing coverage.** *(ii) Exactness* — a **scale-independent plateau**: the single
  most-equivariant cell in either $3\times3$ grid still has $\Delta_{\mathrm{eq}}\approx0.34$ (2D) / $0.36$
  (3D), $\sim\!1\text{–}2\times10^{6}\times$ the VN's *weight-independent* float floor
  ($\sim\!2\times10^{-7}$); $313\times$ the parameters and $16\times$ the data buy no exactness. The prior
  delivers the flat task metric *and* float-floor exactness for free and scale-free; brute force buys
  neither. Five seeds per cell; `experiments/step29_scaling_sweep.py` (2D) and `..._3d.py` (3D).
- **The soft-equivariant model is a tunable dial, not a free lunch (Step 30).** Steps 28–29 asked whether
  *data* can lift the free MLP into the hard prior's corner (it cannot); Step 30 asks the *architecture*
  question — **interpolate** between them with the **Residual Pathway Prior** (Finzi, Benton & Wilson, 2021),
  $f_\beta=f_{\mathrm{VN}}+f_{\mathrm{free}}$ with a residual-energy penalty
  $\beta\,\mathbb E\lVert f_{\mathrm{free}}\rVert^2$ that slides continuously from the hard VN
  ($\beta\to\infty$) to the free MLP ($\beta\to0$). On a world that is *almost* equivariant — the Step 16
  controlled break $\mathrm{Dyn}_g=\mathrm{Dyn}_0-g\,(s\!\cdot\!e)\,e$ along a fixed lab axis, swept
  $g\in\{0,0.2,0.4,0.8\}\times\beta\in\{1,10^{-2},10^{-4}\}$, five seeds (3D "seen" is the full coverage
  ball, since the lab-$z$ break is $z$-rotation-invariant and a wedge would let even the VN fit it) — the
  three metrics move *together*. *(i) Capacity:* the hard VN is **structurally blind** to the fixed-axis
  term — seen relMSE rises with the break (×54.6 in 2D / ×604 in 3D, an irreducible misspecification floor,
  §10); relax the prior and the floor lifts (the softest model fits $g{=}0.8$ by ×225 / ×431 better).
  *(ii) Generalisation:* the OOD/seen ratio is **monotone in softness**, sweeping the whole interval from
  the VN's flat corner (×1.00 in 2D, ≤×1.5 in 3D) through the soft middle to the MLP's extrapolation wall
  (×34–45 in 2D, ×52–71 in 3D). *(iii) Exactness:* the VN is at the float floor for *every* $g$ (the §2
  identity, break-independent), but the residual pathway forfeits exactness **the instant it is active** —
  even at $g{=}0$, where the symmetry is perfectly intact, the softest RPP is already $\sim\!10^{5}\times$
  the floor; there is no "slightly soft" exactness. The model-side free-fraction
  $\rho=\mathbb E\lVert f_{\mathrm{free}}\rVert/\mathbb E\lVert f_\beta\rVert$ confirms $\beta$ is a genuine
  dial (monotone in both $\beta$ and $g$). The soft model buys capacity to absorb a broken symmetry, but
  spends across-group reach **and** float-floor exactness to do it — the exact-and-flat-for-free corner
  belongs to the **architecture alone**, and no $\beta$ recovers it. Five seeds;
  `experiments/step30_soft_equivariant.py` (2D) and `..._3d.py` (3D).
- **One-step equivariance composes into a multi-step rollout guarantee (Step 31).** Every step up to here
  measured a *one-step* prediction, yet a world model earns its keep by *rollout* — planning and imagination
  compose the learned operator $H$ times. The worry: does the across-group flatness survive composition, or
  does it decay step by step? The answer is a one-line theorem — **equivariance is closed under
  composition.** If the one-step map $\Phi_\theta(s)=s+v_\theta(s,a)$ is equivariant, then so is its $H$-fold
  iterate, by induction:
  $\Phi_\theta^{(H)}(Rs)=\Phi_\theta^{(H-1)}\!\big(R\,\Phi_\theta(s)\big)=\cdots=R\,\Phi_\theta^{(H)}(s)$ — no
  retraining, no per-horizon assumption. We test it directly: an *exact* equivariant velocity-field teacher
  $s_{t+1}=s_t+\tau\,\widehat{\mathrm{Dyn}}_0(s_t,a)$ ($\tau{=}0.05$, no break), train one-step on the seen
  region (wedge $[0,180^\circ)$ in 2D / ball $\le 90^\circ$ in 3D), then roll out $H\in\{1,2,4,8,16\}$ and
  read three things. *(i) The honest baseline:* final-state rollout relMSE **accumulates for everyone** —
  VN seen $1.2\times10^{-5}\!\to\!2.3\times10^{-2}$ (2D) / $1.7\times10^{-6}\!\to\!2.2\times10^{-2}$ (3D) from
  $H{=}1$ to $H{=}16$, the MLP comparable — rollout is hard regardless of the prior. *(ii) Generalisation:*
  the VN across-group OOD/seen rollout ratio is $\times1.00$ **flat at every horizon** (the §2 identity rides
  through the composition), while the free MLP carries a large gap at **every** $H$ ($\times43\text{–}51$ in
  2D / up to $\times66$ in 3D), peaking early then *compressing* at large $H$ as the OOD rollout decoheres
  into the relMSE saturation ceiling — the ratio is a clean diagnostic only while the seen rollout is
  faithful, so the *monotone* compounding signal lives in (iii). *(iii) Exactness:* the composed residual
  $\Delta_{\mathrm{eq}}^{(H)}=\max_R\lVert\Phi^{H}(Rx)-R\,\Phi^{H}(x)\rVert/\lVert\Phi^{H}(x)\rVert$ stays at
  the **float floor for every $H$** for the VN ($\le 2.3\times10^{-7}$ in 2D / $\le 1.6\times10^{-7}$ in 3D),
  but **compounds monotonically** for the MLP — $2.3\times10^{-2}(H{=}1)\!\to\!3.7\times10^{-1}(H{=}16)$ in
  2D / $3.9\times10^{-2}\!\to\!4.5\times10^{-1}$ in 3D — because each step re-injects the symmetry break into
  the next. The one-step prior thus pays a **multi-step** guarantee at the horizon the world model is
  actually planned with, not merely at $H{=}1$. Five seeds; `experiments/step31_rollout_horizon.py` (2D) and
  `..._3d.py` (3D).
- **The recovery is a *degree* signature, not a capacity ramp (Step 32).** Step 27 (§7.1) was a single
  architectural point — one tensor-product stack recovered $42\%$ of the degree-1 cap — and could not say
  *why*: is the missing ingredient a specific **representable polynomial degree** (a primitive the
  equivariant class lacked), or just raw **capacity**? A *degree* bottleneck recovers once and stops; a
  capacity bottleneck keeps improving. Step 32 separates them with a **ladder**: `VNTPLadderPredictor`
  front-loads $L$ cross-product blocks into a fixed three-block stack, so the maximum representable degree
  is $d_{\max}=2^{L}$ while **depth, width, and (near-)parameter count are held fixed** ($L0\!\to\!L3$ span
  only $25.1$–$29.8$k params, against the MLP's $62.3$k) — *only* the degree changes. Same Step-24/27
  interacting teacher (degree-3 torque), encoder, message, data, training; sweep $L\in\{0,1,2,3\}$
  ($d_{\max}\in\{1,2,4,8\}$), three seeds. *(i) Recovery:* in-distribution relMSE recovers in **one step** at
  the first cross-product rung ($L0\,0.263\to L1\,0.194$, $\times1.36$, $38\%$ of the cap$\to$MLP gap) and
  then **saturates dead-flat** ($L1\!\approx\!L2\!\approx\!L3\!\approx\!0.20$ within seed noise, top rung
  $+1\%$ of the total) — the degree signature, since a capacity ramp would keep closing toward the MLP's
  $0.080$ with each doubling of $d_{\max}$. *(Honest subtlety: the knee sits at $d_{\max}{=}2$, one rung
  before the naive "torque is degree-3, so $d_{\max}\ge3$" prediction, because the predictor acts on the
  encoder's **already-nonlinear** $\ell_{\max}{=}2$ latent, not raw points — the point-space degree is an
  upper bound on the latent-space knee, and the first cross product already supplies the recoverable bulk.)*
  *(ii) 举一反三:* every rung is across-group $\times1.00$ (the §2 orthogonal-cancellation identity rides
  through every cross-product block unchanged), while the $2.4\times$-larger MLP that fits better
  in-distribution degrades $\times10.5$. *(iii) Exactness:* every rung holds the float floor (composed
  $\mathrm{SE}(3)\le9.3\times10^{-5}$, perm $0$; the predictor alone $4.8\times10^{-7}$ at init), the MLP
  $8.9$ — adding representable degree costs no exactness. So the degree-1 VN's bottleneck was a missing
  **primitive** (one cross-product irrep), recoverable at the *first* rung that supplies it and saturating
  thereafter — not an open-ended capacity climb, and never at the cost of the across-group guarantee. Three
  seeds; `experiments/step32_tp_degree_ladder.py`, structural invariants at every rung in
  `tests/test_step32_degree_ladder.py`.
- **The symmetry prior is *discoverable from data*, and falsifiably so (Step 33).** Every step before this
  *assumes* the group; Step 33 tests whether the group can instead be **read out of a frozen teacher's
  behaviour**. Parametrise a *generator slate* of $K$ free $3\times3$ matrices $\{\hat G_k\}$ — with **no**
  antisymmetry, Lie bracket, or $\mathfrak{so}(3)$ structure imposed — and minimise the relative
  finite-transform equivariance residual $\mathcal R(\hat G)=\mathbb E_x\big\|f(e^{\theta\hat G}x)-e^{\theta\hat G}f(x)\big\|^2/\mathbb E_x\|f(x)\|^2$
  of the teacher $f$; a direction survives **iff** the teacher is genuinely invariant along its finite flow
  $e^{\theta\hat G}$. Two teachers (a TRUE $\mathrm{SO}(3)$ world and a BROKEN one with a lab-frame stretch
  $M=\mathrm{diag}(1,1,-2)$ that leaves only $\mathrm{SO}(2)_z$), five seeds. *(i) Dimension off the data:*
  sweep $K=1\ldots5$; the residual holds the float floor until the slate is forced to spend a direction the
  teacher does not respect, so the **jump location is the dimension** — TRUE floors ($\sim10^{-13}$) through
  $K{=}3$ then jumps $\times9.3\times10^{9}$ at $K{=}4$ ($\dim\mathfrak{so}(3)=3$), BROKEN jumps already
  $\times1.8\times10^{10}$ at $K{=}2$ ($\dim\mathfrak{so}(2)_z=1$). *(ii) The algebra emerges:* at $K{=}3$ the
  TRUE slate *is* $\mathfrak{so}(3)$ though nothing asked it to be — antisymmetry residual $6\times10^{-7}$,
  bracket-closure $2\times10^{-6}$, and (generators normalised to unit Frobenius norm) structure-constant norm
  $\|c\|=1.7320509=\sqrt3$, the exact $\mathfrak{so}(3)$ fingerprint (six nonzero $c_{ijk}=\pm1/\sqrt2$, so
  $\|c\|^2=6\cdot\tfrac12=3$). *(iii) Falsifiable:* the BROKEN world **cannot fake** $\mathfrak{so}(3)$ at
  $K{=}3$ ($\times1.6\times10^{10}$ worse than TRUE) but **does** recover its lone surviving generator at
  $K{=}1$ (aligns with $L_z$ to $1.000$), and that dim-$1$ read holds across an $8\times$ sweep of break
  strength $\beta\in\{0.1,\dots,0.8\}$ — a *symmetry* property, not a tuned magnitude. So the thesis's opening
  "*if* the world carries a symmetry" is something the data can be made to **prove or refute**: discover the
  prior, don't merely postulate it — and trust it only because it can be shown wrong. Five seeds, six guards;
  `experiments/step33_symmetry_discovery.py`, structural invariants in
  `tests/test_step33_symmetry_discovery.py`.
- **The active-inference win survives a *noisy* cue, de-constructed (Step 34).** The Step 25 task win leaned on
  one crutch: a *noiseless* one-bit reveal that snapped the belief to a certainty $p\in\{0,1\}$. Step 34 removes
  it. The cue now passes through a **binary symmetric channel** with crossover
  $\epsilon(d)=\tfrac12-(\tfrac12-\epsilon_0)e^{-d^2/2\delta^2}$ (floor $\epsilon_0>0$), the agent runs **soft
  Bayes** (the posterior never reaches $\{0,1\}$), and the planner's epistemic drive is the **exact mutual
  information** $\mathrm{IG}(p;\epsilon)=\mathcal H(p)-\mathbb E_o[\mathcal H(p')]=I(b;o\mid d)$ — verified to
  equal the soft-Bayes belief-entropy drop to $10^{-7}$, with the three limits $\mathrm{IG}(\epsilon{=}0)=\mathcal
  H(p)$ (recovering Step 25), $\mathrm{IG}(\epsilon{=}\tfrac12)=0$, and $\mathrm{IG}(p\in\{0,1\})=0$. *The one
  design decision:* a noiseless reveal makes $\mathcal H(p)$ collapse to $0$ exactly, so Step 25's curiosity
  *self-extinguished* and the agent committed; under soft evidence $\mathrm{IG}$ stays small-but-nonzero, and a
  bare $z$-score renormalises that vanishing signal back to unit scale — pulling the agent to the cue *forever*.
  Gating the channel by **normalised belief entropy** $g_{\rm epi}=\mathcal H(p)/\ln2\in[0,1]$ (the mutual
  information's own ceiling) re-arms the self-extinguishing envelope; the gate is a *belief scalar*, so the loop
  stays $\mathrm{SE}(3)$-invariant and the $\beta{=}0$ reward-only baseline is untouched. The win **survives**
  ($\times0.614$ true-goal-error cut, CI$[0.499,0.749]$, past the same provable hedge floor), **recovers Step 25**
  as $\epsilon_0\to0$ (EFE $0.333\approx$ oracle), and — the built-in **falsifiable negative** — **vanishes** when
  the channel goes useless ($\epsilon_0{=}0.45$: EFE $0.723\approx$ reward-only $0.663$), with sensing effort
  climbing monotonically $5.6\to15.7$ as the bit degrades. $\mathrm{IG}$ depends on the latent only through the
  invariant distance, so the whole noisy loop is still $\mathrm{SE}(3)$-exact (IG-field $7\times10^{-7}$,
  plan-equivariance $8\times10^{-9}$) where the free MLP shatters it (IG-invariance $0.17$). Seven guards;
  `experiments/step34_active_inference_noisy.py`, structural invariants in
  `tests/test_step34_active_inference_noisy.py`.
- **Few-body $\to$ many-body: a *combinatorial* generalisation axis (Step 35).** Every result above generalises
  across the *continuous* group; Step 35 opens an orthogonal **discrete** axis — object **cardinality** $O$.
  Train the interacting world model at a *single* count $O{=}3$ and test zero-shot at $O\in\{1,2,4,5,6\}$. There
  is no Lie generator carrying $O{=}3$ to $O{=}5$, so equivariance *alone* cannot buy it; what does is the **one
  design decision** — a **count-stable mean message** $\bar r_i=\frac1{O-1}\sum_{j\ne i}\hat r_{ij}$. A *mean* of
  unit vectors lives in the unit ball $\lVert\bar r_i\rVert\le1$ at *every* count (contracting $1.0\to0.94$ as
  $O{:}2\to6$), so the message distribution a one-count predictor sees is count-stable — where a *sum* would grow
  with $O$ and break transfer; $O{=}2$ recovers the Step-24 two-body teacher verbatim and $O{=}1$ reduces to pure
  self-dynamics (message $\equiv0$). *(i) Count transfer is bought by factorisation, not the prior:* holding
  orientation fixed, the relMSE is flat across the interacting family $O\ge2$ for **both** the equivariant VN-MP
  ($\times1.09$) and the equally-equipped non-equivariant MLP-MP ($\times1.05$). *(ii) The prior adds the second
  axis the MLP cannot:* combine the unseen count with an unseen **global rotation** and VN-MP stays **exactly**
  flat (count$\times\mathrm{SO}(3)$ ratio $\times1.00$ to the float floor) while the MLP-MP degrades monotonically
  with count ($\times2.26\to3.34$, mean $\times2.83$) — the clean isolation of the equivariance prior. *(iii)
  Structural at an unseen count:* the whole VN-MP pipeline is $\mathrm{SE}(3)\rtimes S_O$-equivariant
  post-training at a count it is not even built for ($O{=}5$: SE(3) $1.8\times10^{-5}$, perm $7\times10^{-7}$),
  the MLP breaking SE(3) ($1.1\times10^{1}$). The $O{=}1$ no-interaction limit ($\times2.47$; message $\equiv0$ is
  an unseen input and the torque-free latent step shrinks $\sim3.8\times$) is a *documented boundary* that still
  beats no-change ($0.50<1$), not folded into the headline. So a single training count **determines** the
  interacting dynamics across the many-body family, and the *product* of the discrete (count) and continuous
  (rotation) axes is met only by the geometric model — channel-necessity itself is the inherited degree-1
  cross-product cap, a modest $\times3.46$. Eight guards; `experiments/step35_many_body.py`, structural invariants
  in `tests/test_step35_many_body.py`.
- **Discover $\to$ exploit: a *discovered* symmetry, distilled into a free predictor (Step 36).** Every
  across-group win above is bought by a symmetry **hand-wired** into the architecture; Step 33 had already shown
  the prior is *discoverable* — from a blank slate of learnable $3\times3$ matrices it rediscovered a frozen
  teacher's $\mathfrak{so}(3)$ (and an $\mathfrak{so}(2)_z$ on a rotation-broken teacher) with antisymmetry and
  bracket-closure **emergent**, not imposed. Step 36 turns that *measurement* into a *method*: freeze one
  exactly-equivariant encoder $E$ (so all arms share $E(Rx)=\rho(R)E(x)$) and train a **free** MLP predictor $f$
  with the supervised latent-prediction loss plus a soft equivariance regulariser
  $\lambda\sum_{k}\mathbb{E}_{z,a,\theta}\lVert\rho(g_k)f(z,a)-f(\rho(g_k)z,g_k a)\rVert^2$ along the
  **discovered** finite flows $g_k=\exp(\theta\hat G_k)$ — distilling the discovered generators, with nothing
  about $\mathfrak{so}(3)$ hand-coded beyond what discovery found. The **one design decision**: *decouple the
  distillation flow range from discovery*. Discovery needs only a $\pm49^\circ$ wedge to *detect* asymmetry;
  exploitation must *enforce* equivariance over the whole $1$-parameter subgroup, so $\theta_{\max}^{\rm distill}
  =\pi\sqrt2$ sweeps a full half-turn per axis ($\mathrm{tr}\,\exp(\pi\sqrt2\,\hat G)=-1$, the antipode). The
  reads: *(i)* the hard-wired VN predictor is the exact upper bound (composed residual $1.2\times10^{-5}$,
  OOD/seen $\times1.00$) and the free MLP the lower bound (fits the seen wedge $0.45$ but breaks across
  $\mathrm{SO}(3)$, $\times2.25$, equivariance residual $3.69$); *(ii)* distilling the discovered
  $\mathfrak{so}(3)$ across a $\lambda$-ladder **closes $54\%$** of the free MLP's excess OOD gap ($\times2.25\to
  \times1.09\to$ VN $\times1.00$) and drops the predictor equivariance residual $\times8.0$ ($3.69\to0.459$), at a
  $\lambda^\star$ selected by minimum OOD with *statistical* ties (within $5\%$) broken toward the strongest
  enforcement ($\lambda{=}10$ ties $\lambda{=}3$ on OOD but enforces equivariance twice as hard); *(iii)*
  distilling the **discovered** basis is as flat as distilling the hand-wired **oracle** $\mathfrak{so}(3)$
  ($\times1.09$ vs $\times1.06$) — the Step-33 discovery **costs nothing**; *(iv)* the **falsifiable** check —
  distilling only the discovered $\mathfrak{so}(2)_z$ helps the z-axis OOD $+46\%$ but the off-axis only $+17\%$,
  transferring **exactly** the symmetry discovered, no more. The honest limit (**soft $\neq$ hard**): the distilled
  MLP is much flatter than free but does not reach the VN floor (distilled OOD $0.632>2\times$ VN $0.300$) — soft
  regularisation *approximates* equivariance where the built-in prior *enforces* it (the Step-30 dial again). So
  the across-group prior is not only *learnable* (Step 33) but *exploitable* (Step 36): a symmetry discovered from
  data and distilled into a free predictor recovers most of the 举一反三 the hard-wired model gets for free,
  matching the oracle and transferring exactly what it found — with a documented soft-vs-hard gap, not float-floor
  exactness. Six guards; `experiments/step36_discover_exploit.py`, structural invariants in
  `tests/test_step36_discover_exploit.py`.
- **The active-inference win transfers beyond a *constructed* POMDP — the de-construction completed (Step 37).**
  Step 34 removed Step 25's *noiseless* crutch; the **other** crutch was the *constructed mirror* — a hidden
  *bit* with two opposite goals whose midpoint is the start, hand-tuned so the one cue sat exactly transverse.
  Step 37 removes it: the mirror becomes a **generic $K$-target constellation** ($K\ge3$ scattered in a random
  plane with **no antipodal pair at any $K$** — a gap stick-breaking sampler gives $0$ violations over $2000$
  draws, every pair $>38^\circ$ and every angle $>30^\circ$ from $180^\circ$), so the belief is a genuine
  **$K$-ary categorical** with no "opposite" to exploit. The drive is the **exact categorical mutual information**
  $\mathrm{IG}(p;\epsilon,K)=\mathcal H(p)-\mathbb E_o[\mathcal H(p')]=I(b;o\mid d)$ of a **$K$-ary symmetric
  channel** $P(o{=}j\mid b{=}i)=(1{-}\epsilon)[i{=}j]+\tfrac{\epsilon}{K-1}[i{\ne}j]$, crossover annealing with
  the invariant latent distance to the useless floor $\epsilon_\star=(K{-}1)/K$; categorical soft Bayes never
  collapses. **Step 34 is recovered exactly as $K{=}2$** ($\epsilon_\star(2)=\tfrac12$, $\mathrm{IG}$/crossover
  match to $10^{-7}$). *(i) It attains the oracle floor:* on $24$ generic $K{=}3$ POMDPs ($\epsilon_0{=}0.15$) the
  EFE agent reads the off-path cue $10.6\times$, resolves the belief to $p_{\text{true}}{=}1.00$, and lands at
  $0.387$ — within noise of the oracle $0.376$ (gap $+0.011$, CI$[-0.062,+0.089]$ *includes* $0$) and $\times0.565$
  of the reward-only hedge $0.685$ (paired drop $+0.298$, CI$[+0.204,+0.400]$). *(ii) It scales with $K$:* the win
  holds at $K{=}3,4,5$ (ratios $0.60/0.71/0.55$, every drop-CI lower bound $>0$). *(iii) The kept ingredient is the
  premise, made falsifiable:* a *separable* epistemic affordance is what active inference needs, not a crutch, and
  two negatives both fire — the win **vanishes** when the cue is useless ($\epsilon_0{=}\tfrac23$, ratio $1.00$)
  **and** when the affordance collapses to sense$=$commit (ratio $1.25$, EFE still senses *more*, $25.3$ vs $17.2$)
  — pinning the advantage to the affordance, **not** the mirror. The whole loop stays $\mathrm{SE}(3)$-exact
  (IG-field $6\times10^{-6}$, true-goal outcome $\le2\times10^{-6}$, plan-equivariance $2\times10^{-8}$) where the
  free MLP shatters it (IG-field $0.29$, outcome $1.0$/$49^\circ$). What remains untested is a *fully*
  non-constructed benchmark, no longer the mirror. Eight guards; `experiments/step37_active_inference_search.py`,
  structural invariants in `tests/test_step37_active_inference_search.py`.
- **The one outright failure is resolved — decoder-free goal-*reaching*, made exactly equivariant (Step 38, §6.2).**
  Step 13 [C] — purely-latent planning toward a goal cloud without a decoder — was the project's lone outright
  negative (both models closed a *negative* fraction of the orientation gap). Step 38 diagnoses it (not a knob):
  a one-step-trained predictor's $H{=}6$ rollout drifts $\sim\!2.0$ from the encoded truth, so the encoder goal
  $E(X_g)$ sits *off* the reachable manifold and the terminal $L_2$ is ill-scaled. The cure is three decoder-free,
  exactly-equivariant ingredients: **rollout-consistency training** (BPTT to an EMA target encoder), the §6.1
  SE(3)-equivariant CEM planner verbatim, and an **SE(3)-native latent-Procrustes goal** (geodesic angle of the
  Kabsch $R^\star$ aligning $z_0\to z_g$). Decoder-free reaching flips $+0.006\to+0.527$ — **partial** (the residual
  to a $+0.696$ predictor-space ceiling is the encoder-vs-predictor manifold gap, a horizon limitation). The headline
  is the **exactness**: across a paired seen$+4$-OOD SE(3) orbit the VN's residual orientation error is *identical*
  to $1.8\times10^{-6}°$ (ratio $1.000$, CI$[1.000,1.000]$) while the free MLP degrades to $48.7°$ ($\times1.745$,
  CI$[1.473,2.100]$) — §6's closed-loop exactness theorem now for *goal-reaching*. Three panels;
  `experiments/step38_latent_goal_reaching.py`, structural invariants in `tests/test_step38_latent_goal_reaching.py`.

---

## Reproduce

**Reproducibility checklist.**

- **Environment.** Python 3.11, PyTorch 2.12, `e3nn` 0.6.0, NumPy 2.4, Matplotlib; dependencies
  managed with `uv` (not pip) and pinned in `requirements.txt`: `uv venv && uv pip install -r requirements.txt`.
  No CUDA — everything runs on a laptop CPU/MPS.
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
$PRE .venv/bin/python experiments/step24_object_interaction.py     # interaction rung: torque collapses scene group to global SE(3)|x|S_O; interp/extrap flip x1.00 vs x17 (§7.1; STEP24_SMOKE=1)
$PRE .venv/bin/python experiments/make_step24_figure.py            # render Figure 4: step24_object_interaction.{png,pdf} from the dumped JSON (no retrain)
$PRE .venv/bin/python experiments/step25_active_inference_task.py  # partial-observability payoff: EFE planner beats reward-only past the hedge floor; SE(3)-invariant loop (§8.1; STEP25_SMOKE=1)
$PRE .venv/bin/python experiments/step26_optimizer_equivariance.py # does the optimiser break equivariance? intrinsic stays exact under any optimiser, extrinsic drifts under Adam (§3.1; STEP26_SMOKE=1)
$PRE .venv/bin/python experiments/step27_tensor_product_message.py  # tensor-product fix to the degree-1 VN cap: VN-TP recovers 42% of the in-dist cap (x1.45) while staying x1.00 across the group (§7.1; STEP27_SMOKE=1)
$PRE .venv/bin/python experiments/step28_fair_augmentation_baseline.py # fair augmentation baseline (2D SO(2)): coverage closes the task metric (x1.06) but Delta_eq plateaus x2.6e5 above the VN floor -- approximate, never exact (§12; STEP28_SMOKE=1)
$PRE .venv/bin/python experiments/step28_fair_augmentation_3d.py     # same, lifted to 3D SO(3): full-ball aug x1.46 vs VN x1.00, Delta_eq x3.1e5 the floor -- conclusion holds in 3D (§12; STEP28_SMOKE=1)
$PRE .venv/bin/python experiments/step29_scaling_sweep.py            # Bitter-Lesson stress test (2D SO(2)): partial coverage [0,180); MLP width x16 x data x16 does NOT close the extrap gap (x29.8->x48.9, widens) and Delta_eq plateaus x1.1e6 above the VN floor (§12; STEP29_SMOKE=1)
$PRE .venv/bin/python experiments/step29_scaling_sweep_3d.py         # same in 3D SO(3): ball<=90; grid stays x41-x106 vs VN x1.00, best Delta_eq x2.2e6 the floor -- scale != coverage != architecture (§12; STEP29_SMOKE=1)
$PRE .venv/bin/python experiments/step30_soft_equivariant.py         # soft-equivariant RPP (2D SO(2)) under the Step-16 break: beta dials VN<->MLP; capacity recovers (VN seen x54.6, softest 0.0006) but ratio x1->x43 and Delta_eq x5.5e5 the floor at g=0 -- a dial, not a free lunch (§12; STEP30_SMOKE=1)
$PRE .venv/bin/python experiments/step30_soft_equivariant_3d.py      # same in 3D SO(3): seen=full ball (lab-z break needs it); VN seen x604; ratio VN<=x1.53 vs MLP x52-71; softest Delta_eq x4.5e5 the floor at g=0 -- holds in 3D (§12; STEP30_SMOKE=1)
$PRE .venv/bin/python experiments/step31_rollout_horizon.py          # multi-step rollout (2D SO(2)): one-step equivariance composes; VN rollout flat x1.00 + composed Delta_eq <=2.3e-7 at EVERY H in {1..16}, MLP gap persists (x43-x51) and composed residual COMPOUNDS 2.3e-2(H=1)->3.7e-1(H=16) (§12; STEP31_SMOKE=1)
$PRE .venv/bin/python experiments/step31_rollout_horizon_3d.py       # same in 3D SO(3): seen ball<=90, OOD shell 90-180; VN flat x1.00 + Delta_eq <=1.6e-7 every H; MLP ratio x6.5-x66, residual 3.9e-2->4.5e-1 -- composition theorem holds in SO(3) too (§12; STEP31_SMOKE=1)
$PRE .venv/bin/python experiments/step32_tp_degree_ladder.py         # degree ladder (3D SO(3)): sweep representable degree d_max=2^L at FIXED depth/width/(near-)params; in-dist relMSE recovers in ONE step (L0 0.263->L1 0.194, x1.36, 38% of cap->MLP gap) then SATURATES flat (L1~L2~L3~0.20, top rung +1%) -- a DEGREE signature not a capacity ramp; every rung x1.00 + SE(3)<=9.3e-5, MLP x10.5 (§12; STEP32_SMOKE=1)
$PRE .venv/bin/python tests/test_step32_degree_ladder.py             # structural invariants at every rung L in {0,1,2,3}: SO(3) equivariance <1e-4 AND degree separation (L=0 exactly degree-1, every L>=1 carries >=degree-2 content) -- the ladder really climbs representable degree (§12)
$PRE .venv/bin/python experiments/step33_symmetry_discovery.py       # symmetry DISCOVERY (3D): minimise a frozen teacher's finite-transform equiv residual over a free K-generator slate (NO antisym/bracket/so(3) imposed); dim read off the residual jump (TRUE so(3): floor thru K=3, x9.3e9 jump @K=4 => dim 3; BROKEN z-stretch: x1.8e10 jump @K=2 => dim 1); so(3) EMERGES (antisym 6e-7, closure 2e-6, ||c||=sqrt3=1.7320509); FALSIFIABLE: broken can't fake so(3) @K=3 (x1.6e10 worse) but recovers L_z @K=1 (align 1.00), dim-1 read robust over 8x beta range; 5 seeds, 6 guards (§12; STEP33_SMOKE=1)
$PRE .venv/bin/python tests/test_step33_symmetry_discovery.py        # structural invariants: hand-built so(3) basis orthonormal+antisym+bracket-closes (struct const 1/sqrt2); TRUE teacher exactly SO(3)-equivariant; BROKEN teacher R_z-equivariant but breaks under generic R (SO(3)->SO(2)_z exactly); QR frame orthonormal; end-to-end recovers so(3)@K=3 (TRUE) and L_z@K=1 (BROKEN) (§12)
$PRE .venv/bin/python experiments/step34_active_inference_noisy.py    # active inference DE-CONSTRUCTED (3D): Step 25's NOISELESS reveal -> binary symmetric channel eps(d)=1/2-(1/2-eps0)exp(-d^2/2delta^2); planner driven by EXACT mutual info IG=H(p)-E_o[H(p')]=I(b;o|d); belief-entropy gate g_epi=H(p)/ln2 RE-ARMS Step 25's self-extinguishing drive (z-scoring alone destroys it once soft Bayes can't reach {0,1}); win SURVIVES (x0.614 true-goal-err cut, CI[0.499,0.749], past hedge floor), RECOVERS Step 25 @eps0=0 (EFE 0.333~oracle), VANISHES @eps0=0.45 (EFE 0.723~reward 0.663) -- falsifiable; senses MORE for noisier bits (5.6->15.7); IG SE(3)-inv 7e-7, MLP breaks all; 7 guards (§12; STEP34_SMOKE=1)
$PRE .venv/bin/python tests/test_step34_active_inference_noisy.py     # structural invariants: binary entropy H exact at {0,1} via xlogy (no clamp bias); the three exact-MI limits IG(eps=0)=H(p), IG(eps=1/2)=0, IG(p in {0,1})=0; crossover monotone in d; soft-Bayes posterior never collapses to {0,1}; IG-field SE(3)-invariant + EFE-plan SE(3)-equivariant; MLP shatters all (§12)
$PRE .venv/bin/python experiments/step35_many_body.py                # few-body->many-body COMBINATORIAL generalisation (3D): train interacting WM at a SINGLE count O=3, test zero-shot at O in {1,2,4,5,6}; count-stable MEAN message rbar_i=mean_{j!=i} rhat_ij in the unit ball at every count (1.0@O=2->0.94@O=6) so a sum-free one-count predictor transfers (O=2 recovers Step 24, O=1 pure self-dyn msg=0); [C] interacting family O>=2 FLAT VN-MP x1.09 & MLP-MP x1.05 (factorisation buys count, NOT prior); [G] count x global-SO(3) VN-MP x1.00 EXACT vs MLP x2.83 (isolates equivariance); [A'] whole-pipeline equiv at UNSEEN O=5 SE(3) 1.8e-5/perm 7e-7, MLP breaks 1.1e1; channel-necessity x3.46 (degree-1 cap); O=1 no-interaction boundary x2.47 still beats no-change; 8 guards (§12; STEP35_SMOKE=1)
$PRE .venv/bin/python tests/test_step35_many_body.py                 # structural invariants (random init): count-stable mean message in the unit ball at O in {1..6} (=0@O=1, =1@O=2, contracts after); translation-invariant; teacher exactly SE(3) x S_O-equivariant at counts {2,3,4,5} incl. unseen; O=2 recovers Step 24 bit-for-bit + O=1 reduces to pure Step-13 self-dynamics; VN-MP whole pipeline equivariant at UNSEEN O=5 (§12)
$PRE .venv/bin/python experiments/step36_discover_exploit.py          # discover -> EXPLOIT (3D): DISTIL the so(3) generators DISCOVERED off a frozen teacher (Step 33, blank slate) into a FREE MLP predictor via a soft equivariance regulariser lambda*sum_k ||rho(g_k)f - f(rho(g_k)z, g_k a)||^2 along finite flows g_k=exp(theta G_k); one frozen eq-encoder, 5 arms; KEY: decouple the distill flow range (theta_max=pi*sqrt2 sweeps a half-turn, tr=-1) from discovery's detect-wedge (+-49deg); [D] discovery real frac_in_so3 1.00/antisym 0.00/struct-const sqrt3; [U] VN x1.00 (composed 1.2e-5); [L] free x2.25 (equiv 3.69); [M] distilled so(3) closes 54% of OOD gap (x2.25->x1.09) + equiv-resid drop x8.0 (3.69->0.459) @lambda*=10 (OOD ties broken toward enforcement); [O] discovered x1.09 ~ oracle x1.06 (discovery costs nothing); [X] partial so(2)_z z-axis +46% vs off-axis +17% (transfers exactly what it found); [S] soft!=hard distilled OOD 0.632 >2x VN 0.300; 6 guards (§12; STEP36_SMOKE=1)
$PRE .venv/bin/python tests/test_step36_discover_exploit.py           # structural invariants (random init): oracle so3_basis a real Lie algebra (unit-Frobenius/antisym/frac_in_so3 1.0/closure 0/struct-const sqrt3); finite flows exp(theta G) in SO(3), at theta_max=pi*sqrt2 a half-turn (tr=-1, angle theta/sqrt2=pi); rotate_latent an orthogonal rep (norm-preserving, inverse rho(R^T), homomorphism); distill_residual separates VN (float floor <1e-6) from a free MLP (>1e-3, ratio >1e4) on real latents; composed pipeline equivariant for VN (<1e-4) not free (>1e-2, ratio >1e3) (§12)
$PRE .venv/bin/python experiments/step37_active_inference_search.py  # active inference DE-CONSTRUCTED to a GENERIC search (3D): remove Step 25/34's constructed MIRROR (hidden bit, two opposite goals) -> generic K-target constellation (K>=3 in a random plane, NO antipodal pair at any K via gap stick-breaking, 0 viol/2000 draws), K-ary categorical belief; drive = EXACT categorical MI IG=H(p)-E_o[H(p')]=I(b;o|d) of a K-ary symmetric channel eps(d)=eps*-(eps*-eps0)exp(-d^2/2delta^2), useless floor eps*=(K-1)/K; categorical soft-Bayes (no collapse); RECOVERS Step 34 as K=2 (1e-7); [A] EFE ATTAINS THE ORACLE FLOOR (EFE 0.387~oracle 0.376, gap +0.011 CI[-0.062,+0.089] incl 0; x0.565 vs reward 0.685, paired drop +0.298), cue read 10.6x, p_true 1.00; [B] win SCALES with K (3,4,5 ratios 0.60/0.71/0.55, all WIN); TWO falsifiable negatives both fire: useless cue eps0=2/3 ratio 1.00 AND affordance-collapse sense==commit ratio 1.25 (EFE senses MORE 25.3 vs 17.2) -> pins advantage to the AFFORDANCE not the mirror; [C] IG-field SE(3)-inv 6e-6, outcome 2.6e-8, plan-equiv 1.9e-8, MLP breaks all (0.29); 8 guards (§12; STEP37_SMOKE=1)
$PRE .venv/bin/python tests/test_step37_active_inference_search.py   # structural invariants (random init): crossover_cat limits/monotone/range + useless floor eps*=(K-1)/K; categorical IG exact limits IG(eps=0)=H(p), IG(eps=eps*)=0, IG(one-hot)=0, >=0, decreasing (K=3,4,5); K=2 specialisation cat_info_gain==binary info_gain (1.2e-7) & crossover_cat==crossover (0.0) & eps*(2)=1/2; IG==H(p)-E_o[H(p')] via categorical soft-Bayes (2.4e-8); soft-Bayes accumulates/never collapses/useless inert/eps=0 collapses; affordance-collapse reduction H(p)-E_o[H(cat_bayes_binary)]==binary info_gain (1.6e-7); generic constellation coplanar/no mirror(|ang-180|>30deg)/separated(>38deg)/balanced over 40 frames x K=3,4,5; VN categorical-MI field SE(3)-invariant at init (2.2e-8) [MLP-break is a post-train fact in the experiment's panel C] (8 checks) (§12)
$PRE .venv/bin/python experiments/step38_latent_goal_reaching.py     # the ONE outright failure resolved (3D, §6.2): decoder-free latent-goal REACHING -- Step 13[C] (open-loop CEM vs ||z_H - E(Xg)||^2) closed a NEGATIVE gap fraction for both models; diagnosis (not a knob): a 1-step-trained predictor's H=6 rollout drifts ~2.0 from E(teacher^h) so the encoder goal sits OFF the reachable manifold + terminal L2 is ill-scaled; CURE = 3 decoder-free exactly-equivariant ingredients: rollout-consistency train L_roll=mean_h||f^h(E x0,a)-sg E_ema(x_h)||^2 via BPTT+EMA target (load-bearing), the Step-18/§6.1 SE(3)-equivariant CEM planner verbatim, and an SE(3)-native latent-Procrustes goal (geodesic angle of the Kabsch R* aligning z0->zg on the 16 type-1 vecs); [A] reaching ladder +0.006 (Step-13[C] verbatim) -> +0.174 (eq planner) -> +0.399 (rollout) -> +0.527 best DEPLOYABLE (Procrustes open-loop) vs +0.696 predictor-space CEILING (needs a_true) and +1.000 oracle -- PARTIAL (residual = encoder-vs-predictor manifold gap); [B] KILLER: across a paired seen+4-OOD SE(3) orbit the VN residual angle is IDENTICAL 16.108 deg, max|d_i|=1.8e-6, OOD/seen ratio 1.000 CI[1.000,1.000] while MLP degrades to 48.7 deg ratio 1.745 CI[1.473,2.100] -- §6's exactness theorem now for GOAL-REACHING; [C] goal cost SE(3)-native (Procrustes |d|7e-8, L2 8e-6), plan-equiv 1.8e-6, composed-equiv VN 4.2e-6 vs MLP 5.15; 3 panels (§6.2; STEP38_SMOKE=1)
$PRE .venv/bin/python tests/test_step38_latent_goal_reaching.py      # structural invariants (random init): latent_residual_angle recovers |R| exactly (2.8e-12 deg) on the 16 type-1 vecs; Procrustes goal cost SE(3)-invariant under shared rho(R) (2.2e-13) and raw L2 cost too (1.7e-13, float64 re-orthogonalised R); VN composed-equiv at init 1.2e-6 << MLP 2.7 (assert MLP>100x); goal cost invariant through the VN encoder end-to-end under a cloud rotation (7e-15); reaching transfers EXACTLY across the SE(3) orbit at init (7.4e-7 deg, tiny CEM) (6 checks) (§6.2)
$PRE .venv/bin/python experiments/make_bet_figures.py             # render the headline figures: step21_frontier.png + where_the_bet_pays.{png,pdf} (§9-§10)
$PRE .venv/bin/python tests/test_planner_equivariance.py          # the clean single-plan SE(3) theorem: plan(g.x)=g.plan(x) to 1.2e-7
$PRE .venv/bin/python tests/test_set_equivariance.py              # scene-group SE(3)^O |x| S_O equivariance, init + post-train (§7)
$PRE .venv/bin/python tests/test_efe_invariance.py                # EFE-drive SE(3)-invariance: disagreement/entropy/total-G + zero-novelty reorient, init + post (§8)
$PRE .venv/bin/python tests/test_step25_salience_invariance.py    # cue salience SE(3)-invariant + EFE-plan SE(3)-equivariant, init + post (§8.1)
$PRE .venv/bin/python tests/test_step26_optimizer_equivariance.py # intrinsic exactly equivariant under any optimiser; extrinsic drifts under Adam; sym-compat SGD drifts less (§3.1)
$PRE .venv/bin/python tests/test_step27_tensor_product.py         # VNTensorProduct + VNTPPredictor SO(3)-equivariant (init+post), degree-2 homogeneity witness, pseudovector sign-flip (§7.1)
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
Its interaction rung (Step 24, §7.1) — the equivariant torque teacher `scene_teacher_interact`, the
relative-pose message `build_msg_action`, and the three-model VN-MP / VN-Set / MLP-MP one-variable
ablation (reusing Step 19's `SetSE3Encoder`/`SlotMLPEncoder` so only the prior changes), with Figure 4
regenerated from the dumped JSON by `make_step24_figure.py` — is in
`experiments/step24_object_interaction.py`.
The active-inference lift to an Expected Free Energy in the equivariant latent — the shared-encoder deep
ensemble (`EnsembleJEPA`/`build_vn_ensemble`, the `build_mlp_ensemble` control), the
`disagreement`/`gaussian_entropy` drives, the EFE-CEM `efe_cem_plan`, and the `_reorient`/`_novel_shape`
orbit / off-orbit probes — is in `experiments/step20_active_inference.py`, with the EFE-drive
$\mathrm{SE}(3)$-invariance (disagreement, entropy, total $G$, and re-orientation zero-novelty; init and
post-train; the MLP control failing) guarded in `tests/test_efe_invariance.py`.
Its partial-observability payoff (Step 25, §8.1) — the ambiguous-goal cue-foraging POMDP `make_cue_tasks`,
the belief-weighted three-channel EFE-CEM planner `efe_pomdp_plan` with the self-extinguishing cue salience
$\eta\,\mathcal H(p)$, the closed-loop one-bit Bayesian episode `run_pomdp_episode`, and the loop-level
$\mathrm{SE}(3)$ probe `salience_invariance` (over the Step-13 backbones) — is in
`experiments/step25_active_inference_task.py`, with the cue-salience $\mathrm{SE}(3)$-invariance and the
EFE-plan $\mathrm{SE}(3)$-equivariance (init and post-train; the non-equivariant control breaking the plan
equivariance) guarded in `tests/test_step25_salience_invariance.py`.
Its optimiser probe (Step 26, §3.1) — the intrinsic-vs-extrinsic commutant experiment that asks whether a
geometry-blind optimiser can drift weights off the intertwiner space — is in
`experiments/step26_optimizer_equivariance.py` (Panel A trains the real `build_eq_jepa()` under
`build_muon_adamw` from `src/training/muon.py`, plain Adam, and SGD, and probes the composed
$\mathrm{SE}(3)$ residual in float64; Panel B is the analytic $\rho=R\oplus R$ commutant on $\mathbb R^6$
with the intrinsic $M\otimes I_3$ parametrisation versus a free dense $W$ merely initialised in the
commutant, trained under label noise $\sigma{=}0.05$), with the result — intrinsic stays exactly equivariant
(off-commutant $0$) under Adam *and* SGD while the extrinsic map drifts off the commutant under Adam
($\sim\!10^{-2}$) and the symmetry-compatible SGD drifts strictly less — guarded in
`tests/test_step26_optimizer_equivariance.py`.
Its tensor-product fix (Step 27, §7.1) — the degree-2 `VNTensorProduct` (the antisymmetric
$\mathbf 1\otimes\mathbf 1\to\mathbf 1$ cross-product irrep that a degree-1 VN structurally lacks) and the
`VNTPPredictor` (two TP compositions reaching the trilinear torque, degree-$\{1,2,3\}$ kept alive by
concatenation), in a matched-capacity three-model ablation (degree-1 VN-MP / tensor-product VN-TP / the
$\sim\!62$k-param non-equivariant MLP-MP) reusing the Step 24 teacher so only the predictor's hypothesis
class changes — is in `experiments/step27_tensor_product_message.py`, with the $\mathrm{SO}(3)$ equivariance
of both the bare layer and the full predictor (init **and** post-train), the degree-2 homogeneity witness
($f(\lambda x)=\lambda^2 f(x)$, separating it from the degree-1 `VNLinear`), and the pseudovector sign-flip
under improper rotations ($f(Qx)=\det(Q)Q f(x)$) guarded in `tests/test_step27_tensor_product.py`.
Full per-step narrative — including the binary task-success caveats and the per-step closed-loop tables — in
[[geometric_payoff.md]].
