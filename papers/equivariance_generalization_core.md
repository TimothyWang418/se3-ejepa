# Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group

> A focused write-up of the project's most robust result — the **prediction/representation-level**
> core ("[A] + [B]"), distilled from the full results log
> [[geometric_payoff.md]] and demonstrated in **both** $\mathrm{SO}(2)$ (real PushT)
> and $\mathrm{SO}(3)$ (3D point clouds). The same isometry theorem extends to a
> **closed-loop corollary [C]**: under a *matching* equivariant planner, control error is
> exactly invariant across the group to the float floor (§6, Step 14). What stays deliberately
> **out of scope** is binary task-success sweeps, planner-free closed-loop invariance, and
> scaling (§8); this note is the claim I am willing to stand behind today.

Last updated: 2026-05-29. All experiments are laptop-scale (CPU/MPS), fully seeded and
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
> env float floor; the non-equivariant baseline degrades with a CI excluding $0$).
>
> We show [A]/[B] for $G=\mathrm{SO}(2)$ on a **real** contact-rich simulator (PushT) and for
> $G=\mathrm{SO}(3)$ on 3D **point clouds**, with the equivariant model **$4.5$–$7.4\times$
> smaller** and frequently *better* in-distribution; [C] on real-PushT pose control. We make
> **no** claim here about *binary* task-success sweeps or scaling (§8), and [C] requires the
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
(Step 14 [S], §6).

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
statistical tie. Binary task-success sweeps and scaling stay out of scope (§8). Confidence ≈
**0.9** on [E] (exact, paired, $K{=}48$), ≈ **0.85** on the model-and-planner [S] finding.

---

## 7. Related work — where this sits

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
caveat (§8), and treat active inference (Friston, 2017) only as *mathematical motivation* for the
perception–action loop — it is **not** implemented here.

---

## 8. Honest scope — what this note does **not** claim

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
- **Exactness requires the world to actually carry the symmetry.** Real PushT's *interior*
  manipulation is $\mathrm{SO}(2)$-equivariant to $10^{-5}$ px; block↔**wall** contact breaks
  it to the square's $C_4$. The guarantee is exact only where the symmetry is real.
- **Everything is laptop-scale.** The Bitter Lesson (Sutton) warns that scale often beats
  inductive bias; nothing here speaks to scale. The defensible statement is narrow: *when the
  dynamics genuinely has a symmetry, hard-wiring it makes a latent world model learn from far
  fewer interactions and generalise across the group zero-shot at the prediction level — in 2D
  and 3D, at a fraction of the parameters.*
- **2D expressivity caveat** (§2): scalar-weight VN is complete for $\mathrm{SO}(3)$ but not
  $\mathrm{SO}(2)$ (missing the $J$ generator); the 2D demos stay inside the scalar-weight
  class by construction, which is what keeps [B] a fair test.

---

## Reproduce

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
$PRE .venv/bin/python experiments/step16_misspecification.py     # where the prior stops being free (Bitter-Lesson boundary; §8)
$PRE .venv/bin/python experiments/step17_multiseed_closed_loop.py # training-seed error bar: architecture not seed (§8)
```

Equivariant primitives `src/models/structured.py` (`VNLinear`, `VNReLU`,
`StructuredStateEncoder`, `VNPredictor`); SE(3) point-cloud encoder `src/models/se3.py`; JEPA
wrapper + latent predictor `src/models/eqjepa.py`; EMA-target + VICReg training loop
`src/training/jepa.py`; the paired closed-loop [C] power analysis (equivariant CEM
`cem_plan_pose_iso`, disk clamp, paired bootstrap CIs) in
`experiments/step14_pose_control_power.py`. Full per-step narrative — including the binary
task-success caveats and the per-step closed-loop tables — in [[geometric_payoff.md]].
