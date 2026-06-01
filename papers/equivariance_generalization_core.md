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
Compatible-Optimizer* warning (Lau & Su) leaves it untouched (§2.3) ([A]) — and one-step error is
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
are explicit about what stays **out of scope** (§5): binary task-success sweeps, planner-free
closed-loop invariance, and scaling the approach itself. We do, however, **stress-test the prior
directly against Sutton's Bitter Lesson** — rotation augmentation given the whole group, brute-force
scale at partial coverage, and a soft-equivariant interpolation — and find each closes at most the
across-group *task* metric, never the architecture's float-floor *exactness* (§5). Finally, because
equivariance is **closed under composition**, the guarantee is not merely one-step: the $H$-fold rollout
operator the world model is actually planned with stays across-group flat ($\times1.00$) and
float-floor-exact ($\le\!2\times10^{-7}$) at **every** horizon, while the non-equivariant baseline's composed
residual compounds monotonically with $H$ (§5).

---

## 1. Introduction

> A focused write-up of the project's most robust result — the **prediction/representation-level**
> core ("[A] + [B]"), distilled from the full results log
> [[geometric_payoff.md]] and demonstrated in **both** $\mathrm{SO}(2)$ (real PushT)
> and $\mathrm{SO}(3)$ (3D point clouds). The same isometry theorem extends to a
> **closed-loop corollary [C]**: under a *matching* equivariant planner, control error is
> invariant across the group — *exactly* to the float floor in 2D/SO(2) (§3.3) and,
> lifted to the full **3D SE(3)** group, statistically flat to the model's $\sim\!10^{-6}$
> equivariance floor (§3.3.1). What stays deliberately
> **out of scope** is binary task-success sweeps, planner-free closed-loop invariance, and
> scaling (§5); this note is the claim I am willing to stand behind today.

All experiments are laptop-scale (CPU/MPS), fully seeded and deterministic (last updated
2026-05-31). We state the claim precisely, then spend the rest of the note earning it.

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
> **lifts to the full 3D SE(3) group** (§3.3.1): on 3D point clouds with an
> SE(3)-equivariant planner the VN's OOD/seen orientation-error ratio is statistically flat
> ($[0.977,0.999]$) and disjoint from the baseline's ($[1.049,1.234]$) — there "exact" means to
> the network's $\sim\!10^{-6}$ equivariance floor (a CEM tie-flip floor), not the literal float
> zero 2D reaches; the single-plan identity $\mathrm{plan}(g\cdot x)=g\cdot\mathrm{plan}(x)$ still
> holds to $1.2\times10^{-7}$.
>
> We show [A]/[B] for $G=\mathrm{SO}(2)$ on a **real** contact-rich simulator (PushT) and for
> $G=\mathrm{SO}(3)$ on 3D **point clouds**, with the equivariant model **$4.5$–$7.4\times$
> smaller** and frequently *better* in-distribution; [C] on real-PushT pose control (2D/SO(2))
> and, lifted, on 3D point clouds under the full SE(3) group (§3.3.1). We make
> **no** claim here about *binary* task-success sweeps or scaling (§5), and [C] requires the
> **planner** to share the symmetry (§3.3).

The point is that [B] is not a lucky empirical trend — it is a **theorem** about the loss
(§2.2), realised numerically to five digits; [C] is that *same* theorem applied to the realised
closed-loop trajectory.

![The central result in one figure](figures/killer_figure.png)

> **Figure 1.** The claim, as the three error bars a sceptic asks for. **(a)** OOD/seen
> prediction-error factor: the equivariant model is flat ($\approx\!\times1$) across every setting
> — SO(2) synthetic & real, SO(2) latent, SO(3) 3D, full SE(3) — while the same-class baseline
> blows up $\times13$–$\times157$ (§3.2). **(b)** Five *independently trained* models, real-PushT
> closed-loop pose control: the VN's seen-vs-unseen block-angle sits on $y=x$
> (orientation-invariant; $\Delta=-1.0°$) while the baseline sits above it ($\Delta=+9.6°$) — the
> contrast is the *architecture*, not the seed. **(c)** Deliberately breaking the SO(3) symmetry of
> the world: the prior's OOD error rises (it is *not* free once the world de-symmetrises)
> but stays below the unconstrained baseline even past 50% symmetry-breaking — an honest bracket on
> Sutton's Bitter-Lesson crossover.

**Contributions.**

- **A theorem, not a trend (§2.2).** Because the latent group action $\rho(g)$ is *orthogonal*,
  the one-step prediction relMSE is **exactly invariant** across the whole group, so fitting the
  dynamics on a restricted orientation wedge *mathematically determines* it on the entire orbit
  (举一反三).
- **The symmetry survives training — under *any* optimiser (§2.3, §3.1).** The learned model stays
  equivariant to the float floor *after* a real Muon/AdamW $+$ EMA $+$ VICReg run, and provably
  under Adam/SGD too, because the Vector-Neuron / `e3nn` weights parametrise the intertwiner space
  **intrinsically** — Result **[A]**.
- **Zero-shot generalisation across the group (§3.2).** VN one-step error is flat to five digits
  while a same-hypothesis-class non-equivariant baseline fits the wedge and breaks
  out-of-distribution ($\times13.8$ in 2D latent, $\times17.2$ in 3D, $\times157$ over the full
  $\mathrm{SE}(3)$ ladder), with the equivariant model **$4.5$–$7.4\times$ smaller** — Result **[B]**.
- **A closed-loop corollary (§3.3).** Under a *matching* equivariant planner the realised control
  trajectory at orientation $g$ is *exactly* $\rho(g)$ applied to the seen trajectory —
  **float-floor-exact** in 2D/$\mathrm{SO}(2)$ and **statistically flat** in 3D/$\mathrm{SE}(3)$ —
  Result **[C]**.
- **The same prior, extended.** A compositional scene group $\mathrm{SE}(3)^O\rtimes S_O$ (§3.4),
  an $\mathrm{SE}(3)$-invariant active-inference drive that earns a task payoff under partial
  observability (§3.5), and a sample-efficiency frontier (§3.6).
- **An honest Bitter-Lesson bracket (§3.7, §5).** Rotation augmentation given the whole group,
  brute-force scale at partial coverage, and a soft-equivariant interpolation each close at most
  the across-group *task* metric, never the architecture's float-floor *exactness*.

---

## 2. Setup and the exact-flatness guarantee

We state the two measurements the whole note turns on (§2.1), prove the exact-flatness theorem
(Proposition 1) that makes [B] a mathematical guarantee rather than an empirical trend (§2.2), and
show the guarantee is *intrinsic to the parametrisation* — hence preserved by any optimiser (§2.3).

### 2.1 The two measurements

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

### 2.2 The isometry theorem (why [B] is a theorem)

We state the central guarantee as a proposition: the one-step error of an equivariant model is the
*same number* on every orientation, so the across-orbit OOD curve is flat **by necessity, not by
luck**. The three hypotheses are exactly the structural facts §2.1 builds in.

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

The equivariant model's OOD curve is therefore **mathematically forced to be flat** (×$1.00$);
the only deviations we observe ($\le 0.2\%$) are the floating-point floor. The planning cost
$\mathcal{C}=\lVert\hat z_H-z_{\mathrm g}\rVert^2$ with $z_{\mathrm g}=E(s_{\mathrm g})$ is
invariant by the same isometry step — so an equivariant planner literally **cannot tell two
$g$-related problems apart** and solves them identically.

**Corollary 1 (closed-loop orientation-invariance, why [C] is the *same* theorem).** *Add* **(H4)**
*the planner is $G$-equivariant — its sampling distribution and constraint set commute with the group
(an isotropic search with a $g$-covariant noise model and a $G$-invariant action constraint). Then the
entire receding-horizon trajectory at orientation $g$ is the $\rho(g)$-image of the trajectory at the
identity, and any $G$-invariant control error (e.g. block-angle error) is exactly invariant across $G$.*

*Proof.* At each replan step the planner ranks candidates only through the planning cost $\mathcal C$,
which the isometry step of Proposition 1 leaves $G$-invariant, while (H4) maps the candidate set
$g$-covariantly; so
the action sequence selected at orientation $g$ is exactly the $g$-image of the one selected at the
identity. Because the *world* itself is $G$-equivariant, the executed next state is then the $g$-image
of the unrotated next state. Induction over the loop propagates the $\rho(g)$-image to **the entire
closed-loop trajectory**, and a $G$-invariant error read off it is identical across $G$. $\qquad\blacksquare$

This is [C]: the closed-loop analogue of the boxed identity. It holds to the float floor only when **both** the model and
the planner carry the symmetry — if the planner breaks it (e.g. a box action constraint that is
only $C_4$-symmetric, or a per-component variance refit that does not commute with $\rho(g)$),
the invariance degrades to a statistical, *unbiased* one even though the model is exact
(the [S] diagnostic in §3.3). One further honesty: this "float floor" is the *machine* epsilon only when
the model is equivariant to it (real PushT, §3.3); for the 3D `e3nn` encoder the model is
equivariant to its own $\sim\!10^{-6}$ library floor, so even with a matching planner the
realised closed-loop [C] is a *statistical* (ratio) invariance, not a literal zero (§3.3.1).

By contrast, an unconstrained $F$ trained on a wedge $\Phi_0$ is pinned **only on $\Phi_0$**;
off the training orbit the loss says nothing. Channels that are *affine* in the rotated
coordinate (a near-linear PD "self-motion") extrapolate fine; channels that genuinely *rotate*
around the orbit (object orientation / torque) are undetermined and break — empirically
crossing relMSE $=1$ exactly in the rotation channel (§3.2).

**Proposition 2 (discover then exploit — the *discovered*-symmetry analogue of Proposition 1, in the soft
limit).** *Proposition 1 takes $G$ as **given** and hard-wires $\rho(g)$ into the architecture. Suppose instead we are handed only a
**queryable** teacher $f$ — we may evaluate it at transformed inputs but are told nothing of its symmetry.
Parametrise a slate of $K$ generators $\{\hat G_k\}\subset\mathfrak{gl}(3)$ with **no** antisymmetry or Lie
structure imposed, and form the **relative** finite-flow residual
$\mathcal R(\hat G)=\mathbb E_{x,\theta}\big\lVert f(e^{\theta\hat G}x)-e^{\theta\hat G}f(x)\big\rVert^2/\mathbb E_x\lVert f(x)\rVert^2$.*

***(i) Discovery.*** *In exact arithmetic $\mathcal R(\hat G)=0$ **iff** $f$ commutes with the flow
$e^{\theta\hat G}$ — **iff** $\hat G$ generates a symmetry of $f$ — so the residual-nulling directions are
exactly the symmetry algebra $\mathfrak g(f)$, and the least $K$ at which the floor breaks is
$\dim\mathfrak g(f)$.* ***(ii) Exploit.*** *Freeze one exactly-equivariant encoder $E$ (so **(H1)–(H2)** hold
for every arm), let $H=\langle e^{\theta\hat G_k}\rangle\subseteq G$ be the **discovered** subgroup, and train a
**free** predictor $f_\phi$ with the supervised loss plus
$\lambda\,\mathcal R_{\mathrm{distill}}=\lambda\sum_k\mathbb E_{z,a,\theta}\lVert\rho(g_k)f_\phi(z,a)-f_\phi(\rho(g_k)z,\,g_k\!\cdot a)\rVert^2$,
$g_k=e^{\theta\hat G_k}$. Then $\mathcal R_{\mathrm{distill}}(f_\phi)=0$ **iff** $f_\phi$ is a $G$-intertwiner
restricted to $H$ — **exactly (H3) for $H$** — whereupon Proposition 1 applies verbatim and the composed relMSE
is exactly $H$-invariant (across-$H$ OOD factor $1$).* **Honest limit (soft $\neq$ hard):** *a finite $\lambda$
drives $\mathcal R_{\mathrm{distill}}$ toward but not to $0$, so the across-$H$ factor relaxes toward $1$ as
$\lambda$ enforces equivariance more strongly, without reaching the built-in float floor; the guarantee is
"**[B] across the discovered subgroup, in the limit $\mathcal R_{\mathrm{distill}}\to0$**".*

*Proof.* Each residual is a sum of nonnegative terms. For (i), $\mathcal R(\hat G)=0$ iff
$f(e^{\theta\hat G}x)=e^{\theta\hat G}f(x)$ for $\mathbb E_x$-a.e. $x$ and the sampled $\theta$ (a neighbourhood of
$0$ suffices, since the one-parameter-subgroup property then propagates it to all $\theta$); this is the
definition of $\hat G\in\mathfrak g(f)$, and the forward direction is immediate (a symmetry makes the numerator
vanish identically). The nulling set is the subspace $\mathfrak g(f)$, so a slate of size
$K\le\dim\mathfrak g(f)$ fits inside it (floor) while $K>\dim\mathfrak g(f)$ must spend a non-symmetry direction
(jump) — locating the dimension. For (ii), $\mathcal R_{\mathrm{distill}}(f_\phi)=0$ iff
$\rho(g_k)f_\phi(z,a)=f_\phi(\rho(g_k)z,g_k\!\cdot a)$ for every $k$ and $\theta$, i.e.
$f_\phi(\rho(h)z,h\!\cdot a)=\rho(h)f_\phi(z,a)$ for all $h\in H$ (the intertwiner condition is closed under
composition and the $g_k$ generate $H$) — hypothesis (H3) over $H$. With (H1)–(H2) supplied by the frozen
encoder, Proposition 1's boxed identity holds for every $h\in H$. For finite $\lambda$ the penalty has a strictly
positive minimiser, so the implication is exact only in the limit. $\qquad\blacksquare$

*This closes the loop the thesis opens with: the symmetry need not be **postulated** — it can be **read out of
the world's behaviour** and **distilled** into a free predictor to buy the across-group payoff (§5 measures it:
$54\%$ of the free predictor's excess OOD gap recovered, matching the hand-wired oracle, transferring **exactly**
the discovered subgroup), short of the float-floor exactness only a built-in $\rho$ attains. The prior is
**learnable, falsifiable, and cheap to learn** — yet enforcing it exactly still pays, the boundary against which
the Bitter-Lesson caveat (§5) should be read.*

**Expressivity caveat (Schur), stated up front.** Scalar-weight Vector-Neuron layers
(`VNLinear`/`VNReLU`, Deng et al. 2021) are a *complete* equivariant basis for
$\mathrm{SO}(3)$: the standard 3D irrep has real endomorphism algebra
$\mathrm{End}_{\mathrm{SO}(3)}(\mathbb{R}^3)=\mathbb{R}$, so scalar weights suffice and the 3D
demo's dynamics lives **inside** the model class. For $\mathrm{SO}(2)$ the standard rep has
$\mathrm{End}=\mathbb{C}$; scalar-weight VN omits the $90°$ generator $J=\bigl(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\bigr)$, so the 2D demos use dynamics that do not require $J$ (frozen-VN teachers, or PushT channels). This is a genuine limitation, documented — not hidden — and it is *why* [B] is a fair "equivariance generalises" test rather than a "the baseline can't fit" artefact: in every demo the equivariant class can fit the seen wedge at least as well as the baseline.

### 2.3 Why the symmetry survives *any* optimiser — intrinsic vs extrinsic equivariance

One worry about [A] is the optimiser. A sharp recent result (Lau & Su, *A Symmetry-Compatible
Principle for Optimizer Design*, arXiv:2605.18106) shows that **Adam / AdamW / RMSProp are
geometry-blind** — their per-coordinate $1/\sqrt{v_t}$ rescaling does not commute with a group
action on weight space, so they could *silently* break an equivariance constraint one step at a
time. This worry does **not** touch our models, for a reason that is a theorem, not luck.

Equivariance of a linear map $x\mapsto Wx$ means $W$ lies in the **commutant** $\mathcal C=\{W:W\rho(g)=
\rho'(g)W\}$, a linear subspace. Our layers are **intrinsic**: `VNLinear` / `e3nn` store a channel-mixing
$M$ and realise $W=M\otimes I_d$, which is in $\mathcal C$ for *every* $M$ — the parametrisation's whole
image *is* the commutant, so the residual is identically zero for any weights and **any** optimiser keeps
it exact. (This is the same Schur/commutant fact behind §2.2's exact-flatness theorem, read from the
optimiser side — the appendix spells out the matching hypothesis-class restriction.) §3.1 confirms it
empirically across three optimisers.

---

## 3. Experiments

### 3.1 [A] — the learned symmetry survives optimisation

Composed encode→predict equivariance residual after training, planning-cost drift, and
parameter count, across all four end-to-end demos (two real-PushT $\mathrm{SO}(2)$, one
synthetic-teacher $\mathrm{SO}(2)$, one $\mathrm{SO}(3)$ point-cloud):

| demo | group / world | $\Delta_{\mathrm{eq}}$ post-train | cost drift | baseline | params (VN vs base) |
|---|---|---:|---:|---:|---:|
| explicit FM, real PushT | $\mathrm{SO}(2)$, real | $5.4\times10^{-7}$ | — | $0.25$ | $3360$ vs $18952$ (**5.6×**) |
| **latent JEPA**, real PushT | $\mathrm{SO}(2)$, real | $2.9\times10^{-6}$ | $\le1.5\times10^{-7}$ | $3.6$ (drift $0.40$–$0.62$) | $37$k vs $167$k (**4.5×**) |
| pose cost, real PushT | $\mathrm{SO}(2)$, real | — | $4$–$5\times10^{-7}$ | drift $0.45$–$1.06$ | $3360$ vs $18952$ (**5.6×**) |
| **latent JEPA**, 3D clouds | $\mathrm{SO}(3)$, synthetic | $3.0\times10^{-5}$ | $7.2\times10^{-7}$ | $4.30$ (drift $0.85$) | $16{,}856$ vs $124{,}512$ (**7.4×**) |

Every equivariant model keeps the symmetry to the float floor **after** gradient training (the
whole bet — equivariance at init is trivial; surviving optimisation is the claim). The
baselines, same data and class, drift by $0.25$–$4.3$ in residual and up to $\sim\!100\%$ in
cost. And the equivariant models do it with **$4.5$–$7.4\times$ fewer parameters**.

**Any optimiser, not just ours.** The table above used the project's default optimiser (Muon/AdamW);
§2.3 argues the symmetry is *intrinsic to the parametrisation*, so any optimiser preserves it. Training
the real 3D-cloud VN `EqJEPA` under three optimisers confirms it (composed SE(3) residual, float64,
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

### 3.2 [B] — zero-shot generalisation across the group (举一反三)

Train on one orientation wedge; rotate the held-out set across the group. VN is flat to the
float floor everywhere; the same-class baseline fits the wedge and degrades OOD.

| demo | group / world | VN relMSE (every bin) | baseline seen → worst-OOD | OOD factor (VN \| base) |
|---|---|---:|---:|---:|
| synthetic teacher, 1-step | $\mathrm{SO}(2)$, synth | $1.4$–$1.7\times10^{-3}$ | $0.032 \to 3.41$ | **×1.17** \| ×107 |
| [D] — same, **real** PushT inputs | $\mathrm{SO}(2)$, real-in | flat | — | **×1.00** \| ×7 |
| [B] — real PushT, full state | $\mathrm{SO}(2)$, real | $1.05\times10^{-2}$ | $1.66\!\times\!10^{-2} \to 2.69\!\times\!10^{-1}$ | **×1.00** \| ×16.2 |
| [B] — real PushT, **latent** | $\mathrm{SO}(2)$, real | $0.2559$ | $1.14 \to 15.70$ | **×1.00** \| ×13.8 |
| [B] — 3D clouds, **latent** | $\mathrm{SO}(3)$, synth | $0.228$ | $0.307 \to 5.28$ | **×1.00** \| ×17.2 |
| [B] — 3D clouds, **$+$ translation** | $\mathrm{SE}(3)$, synth | $0.228$ | $0.120 \to 18.85$ | **×1.00** \| ×157 |

Two facts hold in every row:

- **VN flat to five digits** — same axis/new angle, new axes, random $\mathrm{SO}(3)$, **and large
  translations**. This is §2.2's theorem, realised. The equivariant model has seen one wedge and is
  *exactly* as good on the entire orbit.
- **The baseline fits the wedge but breaks OOD**, crossing relMSE $=1$ (worse than predicting
  no change) in the latent demos, and — in 3D — worst on the **new-axis** rotations the
  $z$-wedge never showed ($x\,90°$ at $5.28$) or on large translations its raw-coordinate inputs
  never covered ($18.85$).

**Completing the named group.** The rows above made *rotation* the OOD axis; the last row adds
**translation**, so the orbit tested is the *full* $\mathrm{SE}(3)=\mathrm{SO}(3)\ltimes\mathbb{R}^3$
— the project's named geometry. The two halves are earned differently, and the note is honest about
it: rotation-equivariance is **learned** (and survives training, composed residual $3\times10^{-5}$),
while translation-invariance is **exact by construction** — the encoder centres the cloud
($r_i=x_i-\bar x$), so $E(x+t)=E(x)$ identically and a translated transition has the same latent,
predicted latent, and next latent. That is geometry done right rather than a deep learned result, but
it is precisely what makes the whole group a *zero-cost* generalisation for the equivariant model
while the raw-coordinate baseline degrades ×157.

**Sample efficiency (the same prior, measured as a data curve).** On the synthetic teacher with
full-orientation test coverage, the VN matches the MLP's *best* error using **$16\times$ fewer
transitions** ($N{=}32$: VN $0.210$ vs MLP-best $0.233$ at $N{=}512$); by $N{=}512$ the VN
**solves** the task ($4.0\times10^{-3}$) while the MLP **plateaus** ($0.23$) — a gap *more data
alone cannot close*, because the baseline's hypothesis class is not tied across the orbit.

**The mechanism, decomposed — why the baseline breaks.** Decomposing the real-PushT prediction
error **by state component** (train wedge, rotate into quadrants):

| component | VN (all quadrants) | MLP seen | MLP worst-OOD |
|---|---:|---:|---:|
| `agent_pos` (near-linear self-motion) | $9.6\times10^{-4}$ (flat) | $1.8\times10^{-3}$ | $0.089$ (stays $\ll1$) |
| `block_pos` (object position) | $0.563$ (flat) | $0.72$ | $1.21$ |
| `block_dir` (object **rotation**) | $0.563$ (flat) | $0.77$ | $2.33$ (×3.0, crosses $1$) |

This is exactly §2.2's prediction. The baseline OOD **keeps its self-motion model** (`agent_pos`
$0.089\ll1$: an affine channel extrapolates) but **loses its model of the object's rotation**
(`block_dir` $0.77\to2.33$, *worse than no-change*) — the one channel that genuinely turns
around the orbit, and the one a manipulation/pose task depends on. The VN is flat on **every**
channel. So "the baseline generalises OOD" and "the baseline breaks OOD" are both true,
component-wise — and the prior's value is precisely that it pins the rotation channel for free.

### 3.3 [C] — the theorem realised in closed loop

[B] is a statement about one-step prediction; the §2.2 closed-loop corollary says the *same*
isometry makes **control** error invariant across the group, provided the planner also carries
the symmetry. A **paired** design tests this, turning the exact symmetry into an
experimental control: because real interior PushT is exactly $\mathrm{SO}(2)$-equivariant,
rotating an entire reorientation task by $\Delta$ (state, goal position, goal angle, scene
orientation) yields another valid real task of *identical intrinsic difficulty*, so the **same**
base task can be run seen ($\Delta=0$) and at OOD rotations with the env- and CEM-seed held
fixed. The paired difference $d_i=\text{ang}_{\text{OOD}}(i)-\text{ang}_{\text{seen}}(i)$ over
$K{=}48$ tasks cancels the task-to-task difficulty variance that makes unpaired closed-loop
comparisons noise-limited (the reason the earlier closed loops kept landing "within noise"). The same
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

The VN's seen-vs-OOD angle change is **zero to the environment float floor** — the §2.2 corollary
realised: the closed-loop trajectory at every OOD orientation is *exactly* the rotated seen
trajectory, task by task (mean angle $7.28°$ at every orientation). The same-class baseline, on
the *same* planner, degrades with a CI excluding 0 (OOD/seen ratio $1.18$, CI $[1.06,1.37]$;
mean angle wanders $17.9°$–$30.5°$). With the planner held equivariant for both, the model's
prior is the *sole* cause of the split.

**[S] — the original non-equivariant planner (diagnostic: the planner must share the symmetry).** Re-run
with the original planner (box action constraint + diagonal $\sigma$, *not* equivariant at
generic angles): the MLP still degrades ($+3.74°$, CI $[+1.46,+6.05]$), but the VN's paired
difference is no longer exactly zero (mean $-0.71°$, CI $[-2.76,+1.01]$, individual
$\lvert d_i\rvert$ up to $34°$) — the *model* is still exact, but the *planner* breaks the
symmetry at generic angles. The VN's CI still brackets 0 (the residual is unbiased), so the
statistical conclusion survives; the lesson is the §2.2-corollary condition made empirical:
**closed-loop invariance requires the model *and* the planner to be equivariant.** This is
exactly why the earlier closed loops (run on this non-equivariant planner) were
noise-limited — the missing half was the controller, not the model.

So [C] is genuinely the *same exactness* as [A]/[B], now in closed loop — but only under a
matching equivariant planner; with a generic-angle-broken planner it weakens to an unbiased
statistical tie. Binary task-success sweeps and scaling stay out of scope (§5). Confidence ≈
**0.9** on [E] (exact, paired, $K{=}48$), ≈ **0.85** on the model-and-planner [S] finding.

#### 3.3.1 The SE(3) lift — [C] in the named geometry

§3.3 made [C] exact in **2D/SO(2)**. This lifts the *same* paired [E]/[S] design to **3D point
clouds under the full SE(3) group**, on the 3D latent JEPA (`SE3PointEncoder` $+$
`VNPredictor(dim=3)`, planning in the learned latent). The planner is made SE(3)-equivariant the
same way it was made SO(2)-equivariant in §3.3 — isotropic $\sigma$, $R$-rotated exploration noise, a
unit-**ball** (not box) action constraint — plus the one ingredient the larger group demands:
because `SE3PointEncoder` *centres* the cloud (translation-invariant, §3.2), a pure-latent cost is
translation-blind, so SE(3) would silently collapse to SO(3). A separate **closed-form centroid
channel** (terminal cost $\lVert\bar x_0+C_T\sum_h a_h-\bar x_g\rVert^2$) restores exact translation
handling. Paired over $K{=}24$ tasks on orbits of $1$ seen $+ 4$ OOD $(R,t)$, $\lvert t\rvert\!\sim\!0.8$:

| OOD/seen orientation-error ratio, 95% bootstrap CI over $K{=}24$ | ratio | 95% CI |
|---|---:|---:|
| **VN (equivariant)** | $0.989$ | $[0.977,\ 0.999]$ — within $2\%$ of flat, deviation *negative* |
| **MLP (baseline)** | $1.134$ | $[1.049,\ 1.234]$ — excludes $1$ |

The CIs are **disjoint** ($0.999<1.049$); panel [S] (the verbatim non-equivariant planner) grows the
VN residual $\sim\!5\times$ (ratio $0.886$, CI $[0.825,0.954]$), re-confirming §3.3's lesson — closed-loop
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
(the clean theorem demonstration, with a non-equivariant MLP
control that fails $\gg$ the floor). The receding-horizon loop occasionally amplifies that
$\sim\!10^{-6}$ into a CEM top-$k$ tie-flip, so the VN's $\max_i\lvert d_i\rvert=3.5°$ is a **tie-flip
floor, not a symmetry break**, and the decisive statistic is the ratio separation above — not a literal
zero. Confidence ≈ **0.85** (one notch below 2D §3.3's $0.9$), ≈ **0.85** on the [S] model-and-planner
finding.

#### 3.3.2 From tracking to *reaching* — the exactness theorem for decoder-free goal-reaching

§3.3/§3.3.1 made closed-loop *tracking* exact: the seen-vs-OOD orientation-error **ratio** is flat under a
matching equivariant planner. We ask the harder question — can a decoder-free planner *reach* a goal
pose specified only as a target latent $z_g=E(X_g)$ (no decoder back to point clouds), and does the
reaching inherit the same exactness? This re-attacks the project's **one outright negative**: the 3D
panel [C], where an open-loop CEM-MPC against $\lVert\hat z_H-z_g\rVert^2$ closed a *negative* fraction of
the orientation gap for both models.

The failure was diagnosed, not knob-tuned. A predictor trained only on *one-step* transitions has a
multi-step rollout $f^h(E(x_0),a_{1:h})$ that drifts $\sim\!2.0$ from the encoded truth
$E(\mathrm{teacher}^h)$ by $h{=}6$ — so $z_g$ sits **off the predictor's reachable manifold** and the
terminal $L_2$ is ill-scaled. Three decoder-free, exactly-equivariant ingredients fix it: (i)
**rollout-consistency training**, $L_{\rm roll}=\frac1H\sum_h\lVert f^h(E(x_0),a_{1:h})-\mathrm{sg}\,
E_{\rm ema}(x_h)\rVert^2$ via BPTT against an EMA target encoder (pulls the reachable manifold onto the
encoded one); (ii) the §3.3.1 SE(3)-equivariant CEM planner verbatim; (iii) an **SE(3)-native latent-Procrustes
goal** — the geodesic angle of the Kabsch rotation $R^\star$ aligning $z_0\to z_g$ on the $16$ type-1
vectors, $\arccos\frac{\operatorname{tr}R^\star-1}{2}$, invariant under a shared $\rho(R)$ because the fit
conjugates.

Decoder-free reaching flips from $+0.006$ (the faithful open-loop [C] control) up the ladder
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
(OOD/seen ratio $1.000$, CI $[1.000,1.000]$ — the same tie-flip-free `e3nn` floor as §3.3.1); the MLP degrades
to $48.7°$ (ratio $1.745$, CI $[1.473,2.100]$, disjoint from flat). The goal cost is SE(3)-invariant to the
float floor (Procrustes $6.8\times10^{-8}$, $L_2$ $7.8\times10^{-6}$) and the rollout VN realises it
end-to-end (composed equivariance $4.2\times10^{-6}$ vs MLP $5.15$). So **§3.3's exactness extends from
tracking to goal-reaching**: whatever the decoder-free planner reaches, it reaches the *same* across the
whole SE(3) orbit. Confidence ≈ **0.8** — one notch below §3.3.1 because the reach is partial (a horizon
limitation), while the across-orbit exactness is at the `e3nn` floor. Guarded by structural
invariants (the Procrustes-angle recovery of $\lvert R\rvert$, both goal
costs' SE(3)-invariance, the VN-vs-free composed-equivariance separation, and exact reaching-transfer at
init).

### 3.4 From one object to a scene — compositional generalisation across $\mathrm{SE}(3)^O\rtimes S_O$

§§3.1–3.3 are about *one* rigid body. A scene of $O$ objects carries a strictly larger group,
$\mathrm{SE}(3)^{O}\rtimes S_O$ — per-object rigid motions **and** object relabelings — and it is built
from **two logically independent** priors that we deliberately separate instead of conflating:

- **Factorization** (shared-weight per-object slots) is *exact-by-construction*, in the same sense as
  §2.2's flatness guarantee. A shared encoder applied per slot is **permutation-equivariant**,
  $E(\sigma\!\cdot\!S)=\sigma\!\cdot\!E(S)$ for $\sigma\in S_O$, and **leakage-free** (slot $i$'s latent
  is a function of object $i$ alone); composing it with the §3.2 *centring* makes each slot
  **arrangement-invariant** (blind to where its object sits). None of this is learned — it holds at the
  float floor for any weights.
- **Per-object $\mathrm{SE}(3)$-equivariance** is the §2.2–3.2 property applied per slot, and it is what
  buys **orientation generalisation**: a per-object reorientation $R_o$ never seen in training acts on
  the slot latent by $\rho(R_o)$ exactly, so the [B]-style relMSE is invariant under it.

The test is a three-model ablation varying *only which prior is present*: **VN-Set** (both — shared
`SE3PointEncoder` per slot + shared `VNPredictor`), **MLP-Slot** (factorization only — shared *centred*
per-object MLP + shared ordinary predictor, **identical slot structure to VN-Set**), **MLP-Global**
(neither — one monolithic MLP on the flattened scene). The teacher is a direct sum of the validated
single-object dynamics (§3.2), hence exactly $\mathrm{SE}(3)^O\rtimes S_O$-equivariant; two distinct
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
isolate), $16{,}856$ params for VN-Set vs $61{,}920$/$245{,}440$ for
the controls.

**Honest scope.** The clean theorem costs an assumption: the objects **do not interact** (the teacher is
a direct sum of per-object dynamics). So arrangement-invariance here is *architectural*, not learned, and
the genuinely-learned claim is the orientation column. An inter-object channel — a relative-pose /
equivariant message-passing block between slots, the scene analogue of §3.3.1's centroid term — is the
named next rung; **§3.4.1 closes it.** Confidence ≈ **0.8** that the two compositional priors are
separable and each buys its named half of the scene group.

#### 3.4.1 The interaction rung: the group collapses, and the interpolation/extrapolation flip

Couple the objects with an equivariant **torque**: object $i$'s points are reoriented by
$\omega_i=\hat r_{ij}\times a_i$, the cross product of the (translation-invariant) unit relative-position
$\hat r_{ij}=(c_j-c_i)/\lVert c_j-c_i\rVert$ with $i$'s own action, scaled by $\kappa=0.8$. A cross product
of two type-1 vectors is $\mathrm{SO}(3)$-equivariant, so the teacher stays a symmetry — but interaction
**collapses** the per-object $\mathrm{SE}(3)^O\rtimes S_O$ down to the **global diagonal**
$\mathrm{SE}(3)\rtimes S_O$ (you may move or relabel the *whole* scene, not each object independently).
Because the torque depends on $\hat r_{ij}$, which the per-slot *centred* encoder discards, the predictor
now genuinely *needs* an explicit equivariant message: each slot's action is augmented with the
relative-position vector $r_{ij}$. Same one-variable discipline, three models — **VN-MP** (equivariant +
message), **VN-Set** (equivariant, *no* message — the §3.4 model verbatim, now mis-specified), **MLP-MP** (the
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
equivariant models stay flat to the float floor ($\times1.00$, a §2.2 theorem, guarded post-training: VN-MP
global $\mathrm{SO}(3)$ residual $3.5\times10^{-5}$ vs the MLP control's $8.8$). The better interpolator is
the catastrophically worse extrapolator: **capacity wins inside the wedge, the prior wins across the
group.** Among the VN models the message still earns its keep in-distribution (VN-MP $\times1.36$ over the
channel-blind VN-Set), so the channel is necessary even before the OOD test.

**The honest cap.** A vanilla VN (VN-Linear + VN-ReLU) is **degree-1 homogeneous** and *cannot* represent
the multilinear torque $(\hat r_{ij}\times a_i)\times\tilde x_k$ — the §2.2 missing-$J$ caveat lifted to 3D:
the $90^\circ$-rotation half disappears under $\mathrm{SO}(3)$ (Schur), but the **degree** half survives
for bilinear couplings. That cap is exactly why the MLP fits better in-distribution and why the VN channel
gap is a modest $\times1.36$ rather than decisive; the named fix is a tensor-product message
($1\otimes1\to1$ in `e3nn`), built and measured next. Supplying exactly that missing irrep —
the SO(3) cross product, the antisymmetric $\mathbf 1\otimes\mathbf 1\to\mathbf 1$ part, two compositions for
the trilinear torque — lets an *exactly* equivariant predictor (VN-TP) recover $\mathbf{42\%}$ of the cap
($0.331\to0.229$, $\times1.45$ better) while staying $\times1.00$ across the collapsed group (post-training
$\mathrm{SE}(3)$ residual $4.0\times10^{-5}$); a residual $\times2.59$ to the unconstrained MLP shows the
degree-1 cap was the **dominant, not the sole**, in-distribution bottleneck. The lesson is constructive:
*enrich the equivariant hypothesis class, don't drop the prior.* The cap does **not** touch the [B] result —
equivariance is about how error
transforms *across the group*, not in-distribution capacity — so the $\times1.00$-vs-$\times17$ flip stands
independent of the cap. Full treatment, figures, the third (relative-arrangement) OOD axis, and the
tensor-product fix are in the appendix.

### 3.5 Active inference in the equivariant latent — the curiosity invariance and its task payoff

§§3.1–3.4 build only the *pragmatic* half of an agent — perceive, predict, act toward a goal — and prove its
exact equivariance. Active inference (Friston, 2017) adds the other half: a rational agent should also act
to **reduce its own uncertainty**. We put both in *one* objective on the learned latent, the
**Expected Free Energy** of an action sequence,
$$
  G(a_{1:H}) \;=\; \underbrace{\textstyle\sum_h w_h\lVert \bar z_h - z_g\rVert^2 + w_t\lVert\,\bar x_0 + c_t\!\sum_h a_h - \bar x_g\rVert^2}_{\text{pragmatic / risk — the validated §3.3 cost}} \;-\; \beta\,\underbrace{\textstyle\sum_h \mathcal{D}_h}_{\text{epistemic / information gain}},
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
This is the §2.2 isometry argument lifted from the *loss* to the agent's *information geometry*.

**Proposition 3 (exact $G$-invariance of the Expected Free Energy).** *The computation above used nothing
about $\mathcal{D}$ beyond its being a function of $\rho(G)$-invariant latent quantities; stated generally,
this is a property of the EFE itself, not of one drive. Assume* **(H1)–(H3)** *of Proposition 1 (the encoder
is $G$-equivariant, $\rho(g)$ is orthogonal, and the predictor — here every ensemble member $f_k$ — is a
$G$-intertwiner), and write the EFE of an action sequence $a_{1:H}$ in a scene $x$ (the start, goal, and any
cue clouds) as $G_x(a_{1:H})=\mathcal{C}_{\mathrm{prag}}-\beta\,\mathcal{C}_{\mathrm{epi}}$
with* **(E1)** *the pragmatic term $G$-invariant (it is the §3.3 latent/centroid cost, invariant by the
isometry step of Proposition 1) and* **(E2)** *the epistemic term a function of $\rho(G)$-invariant latent
quantities only — any of the ensemble spread $\lVert z^{(k)}-\bar z\rVert$, the log-determinant
$\log\det(\hat\Sigma+\epsilon I)$, or a mutual information whose channel likelihood depends on the latent
only through invariant distances. Then, rotating the whole instance (scene and actions together by $g$),*
$$
  G_{g\cdot x}(g\cdot a_{1:H}) \;=\; G_{x}(a_{1:H}) \qquad\text{for every } g\in G,\ \text{at any weights,}
$$
*so the EFE-optimal plan is $G$-equivariant ($\arg\min_a G_{g\cdot x}$ is the $\rho(g)$-image of the plan
at $x$) and the resulting closed-loop outcome is $G$-invariant.*

*Proof.* Under $x\mapsto g\cdot x$ the encoder sends each latent $z\mapsto\rho(g)z$ and each type-1 action
$a\mapsto g\cdot a$ (H1), composed through the intertwining predictor (H3) — exactly the substitution of
Proposition 1. The pragmatic term is invariant by (E1). For the epistemic term, orthogonality
$\rho(g)^\top\rho(g)=I$ (H2) gives $\lVert\rho(g)(z^{(k)}-\bar z)\rVert=\lVert z^{(k)}-\bar z\rVert$;
$\hat\Sigma\mapsto\rho(g)\hat\Sigma\rho(g)^\top$ leaves $\log\det(\hat\Sigma+\epsilon I)$ fixed
($\det\rho=\pm1$); and any latent distance feeding a channel likelihood is preserved — so every argument of
$\mathcal{C}_{\mathrm{epi}}$ is unchanged and $\mathcal{C}_{\mathrm{epi}}(g\cdot a)=\mathcal{C}_{\mathrm{epi}}(a)$
by (E2). Hence $G(g\cdot a)=G(a)$. No step refers to the weights (H1/H3 are intrinsic; §3.1, Step 26), so the
identity holds at initialisation and after any amount of training; invariance of the scalar field $G$ over
the equivariant candidate population makes its minimiser equivariant and the executed trajectory's terminal
state the $\rho(g)$-image. $\qquad\blacksquare$

**Three verified instances.** The three epistemic drives we test are each a function of invariant latent
quantities, hence each an instance of (E2): §3.5's **ensemble disagreement** $\mathcal{D}$ (and its
$\log\det$ entropy face) just above; §3.5.1's **cue salience** $\eta$ under partial observability; and §5's
**exact categorical mutual information** of the $K$-ary cue channel. Each is guarded init **and** post-train,
with a non-equivariant control that breaks every line. The operational reading is the curiosity analogue of
[B]: *an exploration policy fit on one orientation slice transfers exactly across the whole orbit* — the
agent is **correctly indifferent to global pose** (the $\times1.0000$ re-orientation row below), spending
information-seeking effort only on what the symmetry does not already hand it for free (举一反三 in the
language of curiosity).

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

Finally the active-inference **knob** behaves: sweeping $\beta:0\!\to\!12$ in an EFE-CEM planner (the §3.3.1
iso-$\sigma$ planner, now minimising $\mathrm{zscore}(\text{prag})-\beta\,\mathrm{zscore}(\text{epi})$)
monotonically trades pragmatic progress ($24.6\to135.7$) for epistemic gain ($82.3\to419.4$), and the
EFE-selected plan stays equivariant end-to-end,
$\lVert\mathrm{plan}(Rx)-R\,\mathrm{plan}(x)\rVert_\infty=6.0\times10^{-8}$. The structural claims are
guarded init **and** post-train (VN disagreement/entropy/total-$G
<10^{-4}$; re-orientation carries zero novelty with $\mathcal{D}$ non-constant; the MLP control breaks
each).

**Honest scope.** The teacher is **fully observed and deterministic**, so on *this* task the epistemic
term is not *required* to reach goals — the pragmatic planner already does (§3.3). What this establishes
is narrower and exact: the unified EFE objective is well-posed and tractable in the equivariant latent, it
carries a geometric invariance the thesis predicts and a non-equivariant model lacks, and the knob
measurably does what theory says. The empirical payoff *of* information-seeking — tasks unreachable
*without* it (partial observability, sparse/ambiguous goals) — is the named next rung; it is **now
closed in §3.5.1**. Confidence ≈ **0.9** on the invariance theorem + tractability (exact by
construction, survives training, control fails), and — as of §3.5.1 — ≈ **0.85** that the epistemic term
converts to a task win under partial observability (now demonstrated, on a constructed POMDP), overall
≈ **0.85**.

#### 3.5.1 The payoff: active inference earns a task win under partial observability

§3.5's honest ceiling was that on a *fully observed, deterministic* teacher the epistemic term is a
demonstrated **mechanism**, not a task necessity — the pragmatic planner alone reaches every goal (§3.3).
We close exactly that named rung by building a setting where information-seeking is **required** to
succeed and showing the EFE planner in the equivariant latent **beats** a reward-only planner, while the
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
$\lVert\hat z_h - z_c\rVert$ only — so the salience $\eta$ satisfies hypothesis (E2) and this is an instance
of Proposition 3: the equivariant encoder sends every latent by the same orthogonal
$\rho(R)$, so $\eta$ — and hence the whole EFE, the optimal plan, **and the resulting task outcome** — is
exactly $\mathrm{SE}(3)$-invariant/equivariant. Rotating the entire POMDP by a global $(R,t)$:

| residual under global $(R,t)$ | VN | MLP control |
|---|---:|---:|
| salience-field invariance $\max_n|\eta_n(x){-}\eta_n(Rx{+}t)|$ | $1.1\times10^{-5}$ | $0.915$ |
| true-goal-outcome invariance (pos / ang) | $5.1\times10^{-8}$ / $3.2\times10^{-6}$ | $1.25$ / $57.7°$ |
| EFE-plan equivariance $\lVert\mathrm{plan}(Rx){-}R\,\mathrm{plan}(x)\rVert_\infty$ | $1.3\times10^{-8}$ | breaks |

The VN ($16{,}856$ params) solves the rotated POMDP by the rotated plan to the float floor; the MLP
control ($124{,}512$ params, $7.4\times$ larger) breaks every line. Guarded init **and** post-train
(VN salience-inv $<10^{-4}$ and plan-equiv $<10^{-2}$; the
non-equivariant control breaks the plan equivariance — the robust, training-independent break, since the
saturating salience scalar can read vacuously-invariant for a collapsed lightly-trained latent).

**Honest scope.** This is a *constructed* POMDP over the synthetic equivariant teacher, and the cue reveal
is a noiseless one-bit Bayesian collapse, so the win is by design reachable. What this establishes is
exactly two things: (i) the equivariant-latent EFE planner **converts an $\mathrm{SE}(3)$-invariant
epistemic drive into a real task win** a reward-only planner *provably* cannot match (the hedge floor is a
theorem, not an empirical artifact), and (ii) the entire information-seeking loop — drive, plan, outcome —
stays exactly $\mathrm{SE}(3)$-equivariant: the project's thesis carried all the way into a
partial-observability decision problem. The belief update is deliberately minimal (one bit) so the
geometry is the only moving part. Confidence ≈ **0.85** that the constructed win is correct and the
loop-level invariance exact (theorem + survives training + control fails); the ≈ **0.5** that it transfers
beyond this construction is since discharged in two rungs — a noisy-channel rung removes the noiseless crutch (a noisy
$K{=}2$ channel; the win survives at $\times0.614$ and vanishes when the channel goes useless) and a generic-constellation rung
the constructed *mirror* (a generic $K{=}3,4,5$ search with no antipodal pair at any $K$, the *exact categorical*
mutual information as the drive, where the EFE planner **attains the oracle floor**), both still exactly
$\mathrm{SE}(3)$-equivariant (§5) — so what stays genuinely open is now only a *fully* non-constructed
real-observation benchmark, no longer the noise or the mirror.

### 3.6 Sample-efficiency frontier — the learning curve across the group

§3.2 fixed the data and showed [B] at a *single* training-set size; we sweep it and draw the
**frontier** — test error as a function of the number of interactions $N$ — because that frontier
is the operational form of the project's Open Question #1 (*does $\mathrm{SE}(3)$-equivariance in a
JEPA encoder improve sample efficiency?*). Both models (the 3D backbone) train on the thin
orientation wedge $\phi\in[0,90°)$; at each $N\in\{16,32,64,128,256,512\}$ we read two learning
curves — pooled latent 1-step relMSE on held-out **in-wedge** clouds (`seen`) and on the *same*
transition rotated by random $\mathrm{SO}(3)$ (`group`). The budget is a **fixed 600 gradient
updates per run**, so the abscissa is *data*, not optimisation steps; 3 seeds.

**The theorem makes the across-group curve free.** With $E(Rx)=\rho(R)E(x)$, $f(\rho z,Ra)=\rho
f(z,a)$ and $\rho(R)$ orthogonal, the relMSE carries $\rho$ in numerator *and* denominator and
cancels (§2.2, §3.2), so the VN's whole-group curve **equals its in-wedge curve at every $N$ and even
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
theorem wherever the world genuinely carries the group (the Bitter-Lesson caveat, §5, is the
standing boundary). Confidence ≈ **0.9** (the exactness and the wall, guarded init-and-post) /
**0.6** (that "no in-distribution edge" generalises
beyond this teacher and capacity regime).

### 3.7 The symmetry-break × data plane — where the bet pays, *located*

§3.6 fixed an exactly-equivariant world ($g=0$) and swept the data $N$; the misspecification sweep
fixed the data and swept a **symmetry break** $g$. We run the
**product** — a $(g,N)$ grid — and at each cell train both models (the 3D backbone, VN
$16{,}856$ vs MLP $124{,}512$, $7.4\times$) on the thin orientation wedge of a *misspecified*
teacher, reading the same two numbers as §3.6: in-wedge `seen` and genuine across-group `ood`. The
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
of §3.6 is free: a held-out transition *rotated* by $\mathrm{SO}(3)$ is a genuine label because the
equivariance identity holds (rotated-label residual $8.8\times10^{-8}$). At $g>0$ that identity
fails by $O(1)$ — the rotated-label residual jumps to $0.06$–$0.47$ — so a rotated target becomes a
**fake** label. We therefore sample *fresh* full-$\mathrm{SO}(3)$ clouds through the true
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
update budget more wedge data makes the MLP *more* confidently wrong off the wedge, the §3.6 wall now
shown to be *data-proof in $N$* (the fixed-*epochs* qualifier is panel [C] below). Across the $N{=}512$
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
$g$ — equivariance buys nothing in-distribution, exactly as §3.6 found at $g=0$, now confirmed at
every break and at the smallest $N$ on the grid. The sharper question was whether the
in-distribution gap **widens** as the world breaks the symmetry (the misspecification sweep saw widening at $N=1200$).
On this grid the gap stays in a band $\approx +0.2$–$0.29$ with no collapse and no blow-up; comparing
the endpoints it is $+0.205$ at $g{=}0$ versus $+0.242$ at $g{=}0.8$ — a *small* widening ($+0.037$),
not the runaway capacity gap the earlier slice hinted at. A fixed-epochs experiment then tests directly whether that
small widening **grows with data** — the one escape this grid never reached — by extending to
$N\in\{512,1024,2048\}$ (past the earlier $N{=}1200$) under a **fixed-epochs** budget so the $124$K
baseline is fully converged at every $N$ (in-wedge relMSE falls to $0.051$ at $g{=}0,N{=}2048$, and
$N{=}512$ reproduces the $600$-update gap as a built-in cross-check). The break-induced
widening — gap at $g{=}0.8$ minus gap at $g{=}0$ — is $[+0.037,+0.049,+0.033]$ across
$N{=}512/1024/2048$: a small, consistent offset that **does not grow with $N$** ($+0.037$ at
$N{=}512$, $+0.033$ at $N{=}2048$) and sits inside the pooled seed std $0.062$ (Figure 3). So
breaking the symmetry adds at most a *fixed* in-distribution offset, not a capacity gap that scales
with data; the lone earlier $N{=}1200$ widening was not the leading edge of a runaway gap.

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
deserves, and the Bitter-Lesson boundary (§5) is *drawn empirically* rather than asserted.
Confidence ≈ **0.85** (the across-group near-total win and the data-proof-in-$N$ wall, now hardened
over **five seeds**) / ≈ **0.6** (that the
extreme-break tie and the no-runaway-widening generalise beyond this teacher and these five seeds,
even reaching $N{=}2048$). The frontier (§3.6) and both $(g,N)$ phase panels (§3.7) are shown
together in Figure 2.

![Where the geometric bet pays off](figures/where_the_bet_pays.png)

> **Figure 2.** Where the geometric bet pays off — a near-total, data-proof win *across the group*,
> a wash-to-loss *in-distribution*. **(left)** The sample-efficiency frontier under an exactly
> $\mathrm{SO}(3)$ teacher: latent 1-step relMSE vs training-set size $N$, the VN's whole-group curve
> descending while the baseline's is a wall. **(middle)** The symmetry-break $g$ × data $N$
> plane, scored on the **across-group** metric — the prior wins $24/25$ cells, the lone baseline cell
> a statistical tie at $(g{=}0.8,N{=}256)$ on the most-broken row (the data-richest corner
> $(g{=}0.8,N{=}512)$ goes back to the prior). **(right)** The same plane scored **in-distribution**:
> the higher-capacity baseline wins early at every $g$ ($N^\star=32$).

![In-distribution gap does not widen with the break, even at large N](figures/step23_indist_largeN.png)

> **Figure 3.** The in-distribution gap does *not* widen with the symmetry break — tested directly at
> large data. We plot the in-wedge VN$-$MLP gap (mean $\pm$ seed std) against $\log_2 N$ for
> $N\in\{512,1024,2048\}$, one line per break strength $g\in\{0,0.4,0.8\}$, under a **fixed-epochs**
> ($150$) budget so the $124$K baseline is fully converged at every $N$ (more total updates at larger
> $N$, $N{=}512$ reproducing the phase-plane $600$). The lines stay close — separated by at most a *small,
> fixed* offset ($\approx+0.04$) that does **not** grow with $N$: breaking the symmetry does not open
> an in-distribution capacity gap that scales with data, refuting the conjecture that the earlier slice's
> $N{=}1200$ widening was the leading edge of a larger-$N$ effect.

---

## 4. Related work — where this sits

This note is a *recombination*, not a new layer; it stands on three lines and occupies the corner
where they meet.

- **Geometric deep learning supplies the equivariant primitives we build on.** Group-equivariant
  CNNs (Cohen & Welling, 2016) and $E(2)$-steerable CNNs (Weiler & Cesa, 2019, whose `e2cnn` powers
  our SO(2)-steerable pixel encoder) give planar equivariance; on $\mathbb{R}^3$, Tensor Field Networks
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
  across-the-group** statement — at the prediction level (§3.2) and, under a matching equivariant
  planner, an *exactly* orientation-invariant closed loop (§3.3) — rather than a learning-curve
  improvement.
- **JEPA / latent world models predict in representation space — but are not equivariant.** The
  joint-embedding predictive line (LeCun, 2022; I-JEPA, Assran et al., 2023; V-JEPA, Bardes et al.,
  2024) and latent model-based RL (World Models, Ha & Schmidhuber, 2018; DreamerV3, Hafner et al.,
  2023) predict masked/future *latents* and obtain invariance from **scale and augmentation**, not
  from architecture. Our training machinery is squarely in this family — an EMA target à la BYOL
  (Grill et al., 2020) with a VICReg variance hinge (Bardes et al., 2022) against collapse — but the
  encoder/predictor are **exactly equivariant by construction**, so the JEPA cost $\lVert E(x_a)-
  E(x_b)\rVert$ is provably isometry-invariant (§2.2) instead of approximately so. We make this
  substitution *quantitative* in §5: handed the group, rotation augmentation closes the
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
  (§3.2). We therefore read $\gamma$-World as independent, at-scale corroboration of the present
  thesis — *an exact symmetry prior determines the model off the training slice* — now for a discrete
  permutation group. Two axes locate our own corner. (i) **Representation:** it predicts in
  **pixel/video space** through a distilled diffusion teacher, paying the full generative cost; we
  predict in an **abstract equivariant latent** with no decoder, the cheaper route our contrarian bet
  targets. (ii) **Realisation:** its symmetry is a discrete $S_P$ engineered into a positional
  encoding; ours is the continuous $\mathrm{SE}(3)$ carried *exactly* by the network
  ($\sim\!10^{-6}$ through a real training run, §3.1). The two are **complementary, not rival**: our
  object-centric variant already factors a permutation symmetry over entities
  ($\mathrm{SE}(3)^O\rtimes S_O$, §3.4) plus a combinatorial count-generalisation
  result (few-body $\to$ many-body, §5) that is the discrete sibling of their $2\to4$ — so a
  natural synthesis is a simplex-style $S_P$ entity code layered *over* SE(3)-equivariant per-entity
  latents.

**The underexplored corner this note targets.** Equivariant *layers* exist; equivariant *RL* exists;
*JEPA* exists. What is largely missing is their conjunction: an *exactly* SE(3)-equivariant
**JEPA latent world model** whose symmetry (i) **survives a real Muon/AdamW + EMA + VICReg training
run** (§3.1), (ii) yields **exact** zero-shot generalisation across the whole group in 2D *and* 3D
(§3.2), and (iii) converts — under an equivariant planner — into a **float-floor-exact** closed-loop
orientation invariance, with the explicit condition that *the planner must share the symmetry* (§3.3).
That precise combination, together with an honest map of where the prior stops being free (the
misspecification sweep) and that it is a property of the architecture rather than the seed (the
multi-seed error bar), is the contribution. We use Sutton's Bitter Lesson (2019) as the standing
caveat (§5), and — through §3.4 — treated active inference (Friston, 2017) only as *mathematical
motivation* for the perception–action loop. §3.5 now realises it concretely, as an exactly
$\mathrm{SE}(3)$-invariant Expected Free Energy objective in the equivariant latent — but as a *geometric
mechanism* (the curiosity invariance and its $\beta$-knob), **not** a claimed exploration benefit.

---

## 5. Limitations & honest scope — what this note does **not** claim

- **No *binary task-success* claim, and [C] needs a matching equivariant planner.** §3.3 shows
  the closed-loop *orientation-invariance* corollary exactly (VN paired seen-vs-OOD angle change
  $=0$ to the float floor under an equivariant planner), but three things stay out of scope.
  (i) A clean **binary task-success** sweep: combined-pose success (angle *and* position
  thresholds together) stays low for both models at laptop $N$, and the angle-weighted planner
  lets the VN trade position error to minimise angle — so the defensible [C] headline is the
  *angle-error invariance*, not a success-rate win. (ii) **Planner-free** closed-loop
  invariance: the [S] panel shows a generic-angle-broken planner softens VN exactness to a
  statistical (unbiased) tie — [C] is a property of model *and* planner together, not the model
  alone. (iii) **Latent-only planning toward a goal cloud** in 3D was the lone outright negative —
  the open-loop 3D [C] closed a *negative* gap fraction for both models — and **§3.3.2 resolves it**:
  rollout-consistency training $+$ the §3.3.1 equivariant planner $+$ an SE(3)-native latent-Procrustes
  goal flip decoder-free reaching from $+0.006$ to $+0.527$, and the VN reaches *identically* across the
  SE(3) orbit (ratio $1.000$, CI $[1.000,1.000]$ vs the MLP's $\times1.745$) — §3.3's exactness theorem now
  for goal-reaching. What stays open is *full* (not partial) reaching: the $+0.527$ deployable fraction
  trails a $+0.696$ predictor-space ceiling, the residual being the encoder-vs-predictor manifold gap, a
  planning-horizon limitation, not an equivariance one.
- **The 3D SE(3) [C] is *statistical*, not float-floor-literal (§3.3.1).** The 2D corollary (§3.3)
  hits the environment float floor ($\max_i\lvert d_i\rvert=4.9\times10^{-5}°$); the 3D SE(3) lift
  (§3.3.1) is exact only to the `e3nn` network's $\sim\!10^{-6}$ equivariance floor — *not* a
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
  in-distribution null, is §3.6).*
- **2D expressivity caveat** (§2.2): scalar-weight VN is complete for $\mathrm{SO}(3)$ but not
  $\mathrm{SO}(2)$ (missing the $J$ generator); the 2D demos stay inside the scalar-weight
  class by construction, which is what keeps [B] a fair test.
- **The scene result (§3.4) is for *non-interacting* objects; §3.4.1 adds the interaction rung,
  with an honest expressivity cap.** The clean scene 2×2 attribution rests on a direct-sum teacher, so its
  arrangement-invariance is *architectural*, not learned. The interaction rung couples the objects with an equivariant
  torque (collapsing the scene group to the global diagonal $\mathrm{SE}(3)\rtimes S_O$) and adds the
  relative-pose message channel: the **interpolation/extrapolation flip** is decisive ($\times1.00$ for both
  equivariant models vs $\times17$ for the higher-capacity non-equivariant MLP that fits *best*
  in-distribution). The remaining caveat is honest, not fatal: a vanilla VN is degree-1 homogeneous and
  cannot form the bilinear torque, so the in-distribution VN channel gap is a modest $\times1.36$ and the
  named fix is a tensor-product ($1\otimes1\to1$) message — **and §3.4.1 builds it, recovering
  $42\%$ of the cap ($\times1.45$ better fit) while the predictor stays exactly $\mathrm{SO}(3)$-equivariant
  and $\times1.00$ across the group** (a residual $\times2.59$ to the unconstrained MLP shows the cap was the
  dominant, not the sole, bottleneck). The cap is on in-distribution *capacity*, not the across-group [B]
  result, and is now partially lifted *from inside* the equivariant class.
- **The active-inference result (§3.5) is now a task win — but on a *constructed* POMDP.** §3.5 gave
  the *mechanism*: an *exact* curiosity invariance (ensemble disagreement is $\mathrm{SE}(3)$-invariant
  because $\rho(R)$ is orthogonal) and a $\beta$-knob that trades pragmatic for epistemic value
  monotonically — but on a fully-observed deterministic teacher exploration is not *required*. §3.5.1
  closes that rung: in an ambiguous-goal cue-foraging POMDP the EFE planner removes $55\%$ of the
  reward-only error (which sits *exactly* at the analytic hedge floor) by deliberately sensing the cue
  ($0.92$ vs $0.21$ of episodes), reaching within $0.054$ of an oracle told the hidden goal — and the
  whole loop (salience, plan, outcome) stays $\mathrm{SE}(3)$-invariant/equivariant to the float floor
  while the MLP control breaks it. The honest caveat is that the POMDP is *constructed* over the
  synthetic teacher and the reveal is a noiseless one-bit collapse: the win is by design reachable, so
  what is proven is that the equivariant-latent EFE planner *converts an invariant drive into a win a
  reward-only planner provably cannot match*, not that active inference beats a benchmark in the wild
  (transfer to noisy / non-constructed observation is untested).
- **The sample-efficiency claim (§3.6) is *across-group*, not in-distribution.** The §3.6 frontier
  shows the payoff is the difference between a *descending* whole-group learning curve and a *wall*
  — but *in-distribution* the higher-capacity baseline fits the wedge at least as well (often
  better), so there is **no** in-wedge sample-efficiency advantage. The defensible claim is narrow:
  wedge-only data plus the prior buys *whole-group* competence the baseline cannot reach at any $N$;
  it does **not** claim fewer samples to fit the training distribution.
- **The across-group win (§3.7) is *near-total*, not the literal whole box, and the in-distribution
  gap does *not* run away with the break — now hardened over five seeds to $N{=}2048$.** The §3.7
  $(g,N)$ plane refuted two pre-registered predictions. The prior wins $24/25$ cells, **not** all
  $25$ — but at five seeds the lone baseline cell is a **statistical tie on the most-broken row**
  ($g{=}0.8$): the single MLP cell is $(g{=}0.8,\,N{=}256)$ ($\mathrm{VN}\,0.778$ vs
  $\mathrm{MLP}\,0.751$, margin $0.027$), the winner flips cell-to-cell along that row inside the seed
  band, and the data-richest corner $(g{=}0.8,\,N{=}512)$ goes *back* to the prior ($0.836$ vs
  $0.943$) — so the failure is a noisy boundary tie, not a located corner. And the in-wedge capacity
  gap stays a band $\approx+0.2$–$0.29$ ($+0.205$ at $g{=}0$ vs $+0.242$ at $g{=}0.8$ at $N{=}512$): a
  *small* widening with the break, not the runaway gap the lone $N{=}1200$ misspecification slice had
  suggested. A fixed-epochs sweep then ruled out the large-$N$ escape directly: under a fixed-epochs
  (fully-converged) budget to $N{=}2048$, the break-induced widening is $[+0.037,+0.049,+0.033]$
  across $N{=}512/1024/2048$ — a small fixed offset that does **not** grow with $N$ and sits inside
  the pooled seed std $0.062$. (Honest corollary, same fixed-epochs run: the across-group wall is
  *not* immune to data once the baseline is allowed to converge — its $g{=}0$ whole-group error falls
  $2.25\to0.64$ as $N:512\to2048$, still $2.5\times$ the VN at $7.4\times$ the parameters; the wall is
  a sample-efficiency barrier, not an impossibility.) The honest headline: a near-total,
  data-proof-in-$N$ across-group win that degrades only to a *tie* where the symmetry is most broken,
  and a wash-to-loss in-distribution — over **five** seeds, spanning $N$ up to $2048$.
- **The fair augmentation baseline: given the group, augmentation closes the across-group *task*
  metric but never the *exactness*.** The sharpest objection to the whole note is that the
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
  architecture delivers the float-floor-exact invariance the closed-loop [C] (§3.3) is built on (an MLP
  with $\Delta_{\mathrm{eq}}\approx0.05$ cannot close an exactly orientation-invariant loop). Five seeds
  per arm.
- **The Bitter-Lesson stress test: at *partial* coverage, scale substitutes for neither the coverage nor
  the architecture.** The full-coverage experiment handed augmentation the *whole* group; the realistic regime is
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
  neither. Five seeds per cell.
- **The soft-equivariant model is a tunable dial, not a free lunch.** The augmentation experiments asked whether
  *data* can lift the free MLP into the hard prior's corner (it cannot); the soft-equivariant model asks the *architecture*
  question — **interpolate** between them with the **Residual Pathway Prior** (Finzi, Benton & Wilson, 2021),
  $f_\beta=f_{\mathrm{VN}}+f_{\mathrm{free}}$ with a residual-energy penalty
  $\beta\,\mathbb E\lVert f_{\mathrm{free}}\rVert^2$ that slides continuously from the hard VN
  ($\beta\to\infty$) to the free MLP ($\beta\to0$). On a world that is *almost* equivariant — the
  controlled break $\mathrm{Dyn}_g=\mathrm{Dyn}_0-g\,(s\!\cdot\!e)\,e$ along a fixed lab axis, swept
  $g\in\{0,0.2,0.4,0.8\}\times\beta\in\{1,10^{-2},10^{-4}\}$, five seeds (3D "seen" is the full coverage
  ball, since the lab-$z$ break is $z$-rotation-invariant and a wedge would let even the VN fit it) — the
  three metrics move *together*. *(i) Capacity:* the hard VN is **structurally blind** to the fixed-axis
  term — seen relMSE rises with the break (×54.6 in 2D / ×604 in 3D, an irreducible misspecification floor,
  §3.7); relax the prior and the floor lifts (the softest model fits $g{=}0.8$ by ×225 / ×431 better).
  *(ii) Generalisation:* the OOD/seen ratio is **monotone in softness**, sweeping the whole interval from
  the VN's flat corner (×1.00 in 2D, ≤×1.5 in 3D) through the soft middle to the MLP's extrapolation wall
  (×34–45 in 2D, ×52–71 in 3D). *(iii) Exactness:* the VN is at the float floor for *every* $g$ (the §2.2
  identity, break-independent), but the residual pathway forfeits exactness **the instant it is active** —
  even at $g{=}0$, where the symmetry is perfectly intact, the softest RPP is already $\sim\!10^{5}\times$
  the floor; there is no "slightly soft" exactness. The model-side free-fraction
  $\rho=\mathbb E\lVert f_{\mathrm{free}}\rVert/\mathbb E\lVert f_\beta\rVert$ confirms $\beta$ is a genuine
  dial (monotone in both $\beta$ and $g$). The soft model buys capacity to absorb a broken symmetry, but
  spends across-group reach **and** float-floor exactness to do it — the exact-and-flat-for-free corner
  belongs to the **architecture alone**, and no $\beta$ recovers it. Five seeds.
- **One-step equivariance composes into a multi-step rollout guarantee.** Every result up to here
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
  the VN across-group OOD/seen rollout ratio is $\times1.00$ **flat at every horizon** (the §2.2 identity rides
  through the composition), while the free MLP carries a large gap at **every** $H$ ($\times43\text{–}51$ in
  2D / up to $\times66$ in 3D), peaking early then *compressing* at large $H$ as the OOD rollout decoheres
  into the relMSE saturation ceiling — the ratio is a clean diagnostic only while the seen rollout is
  faithful, so the *monotone* compounding signal lives in (iii). *(iii) Exactness:* the composed residual
  $\Delta_{\mathrm{eq}}^{(H)}=\max_R\lVert\Phi^{H}(Rx)-R\,\Phi^{H}(x)\rVert/\lVert\Phi^{H}(x)\rVert$ stays at
  the **float floor for every $H$** for the VN ($\le 2.3\times10^{-7}$ in 2D / $\le 1.6\times10^{-7}$ in 3D),
  but **compounds monotonically** for the MLP — $2.3\times10^{-2}(H{=}1)\!\to\!3.7\times10^{-1}(H{=}16)$ in
  2D / $3.9\times10^{-2}\!\to\!4.5\times10^{-1}$ in 3D — because each step re-injects the symmetry break into
  the next. The one-step prior thus pays a **multi-step** guarantee at the horizon the world model is
  actually planned with, not merely at $H{=}1$. Five seeds.
- **The recovery is a *degree* signature, not a capacity ramp.** The §3.4.1 tensor-product fix was a single
  architectural point — one tensor-product stack recovered $42\%$ of the degree-1 cap — and could not say
  *why*: is the missing ingredient a specific **representable polynomial degree** (a primitive the
  equivariant class lacked), or just raw **capacity**? A *degree* bottleneck recovers once and stops; a
  capacity bottleneck keeps improving. We separate them with a **ladder**: `VNTPLadderPredictor`
  front-loads $L$ cross-product blocks into a fixed three-block stack, so the maximum representable degree
  is $d_{\max}=2^{L}$ while **depth, width, and (near-)parameter count are held fixed** ($L0\!\to\!L3$ span
  only $25.1$–$29.8$k params, against the MLP's $62.3$k) — *only* the degree changes. Same interacting
  teacher (§3.4.1, degree-3 torque), encoder, message, data, training; sweep $L\in\{0,1,2,3\}$
  ($d_{\max}\in\{1,2,4,8\}$), three seeds. *(i) Recovery:* in-distribution relMSE recovers in **one step** at
  the first cross-product rung ($L0\,0.263\to L1\,0.194$, $\times1.36$, $38\%$ of the cap$\to$MLP gap) and
  then **saturates dead-flat** ($L1\!\approx\!L2\!\approx\!L3\!\approx\!0.20$ within seed noise, top rung
  $+1\%$ of the total) — the degree signature, since a capacity ramp would keep closing toward the MLP's
  $0.080$ with each doubling of $d_{\max}$. *(Honest subtlety: the knee sits at $d_{\max}{=}2$, one rung
  before the naive "torque is degree-3, so $d_{\max}\ge3$" prediction, because the predictor acts on the
  encoder's **already-nonlinear** $\ell_{\max}{=}2$ latent, not raw points — the point-space degree is an
  upper bound on the latent-space knee, and the first cross product already supplies the recoverable bulk.)*
  *(ii) 举一反三:* every rung is across-group $\times1.00$ (the §2.2 orthogonal-cancellation identity rides
  through every cross-product block unchanged), while the $2.4\times$-larger MLP that fits better
  in-distribution degrades $\times10.5$. *(iii) Exactness:* every rung holds the float floor (composed
  $\mathrm{SE}(3)\le9.3\times10^{-5}$, perm $0$; the predictor alone $4.8\times10^{-7}$ at init), the MLP
  $8.9$ — adding representable degree costs no exactness. So the degree-1 VN's bottleneck was a missing
  **primitive** (one cross-product irrep), recoverable at the *first* rung that supplies it and saturating
  thereafter — not an open-ended capacity climb, and never at the cost of the across-group guarantee. Three
  seeds, with structural invariants checked at every rung.
- **The symmetry prior is *discoverable from data*, and falsifiably so.** Every result before this
  *assumes* the group; we test whether the group can instead be **read out of a frozen teacher's
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
  prior, don't merely postulate it — and trust it only because it can be shown wrong. Five seeds, six guards.
- **The active-inference win survives a *noisy* cue, de-constructed.** The §3.5.1 task win leaned on
  one crutch: a *noiseless* one-bit reveal that snapped the belief to a certainty $p\in\{0,1\}$. This rung removes
  it. The cue now passes through a **binary symmetric channel** with crossover
  $\epsilon(d)=\tfrac12-(\tfrac12-\epsilon_0)e^{-d^2/2\delta^2}$ (floor $\epsilon_0>0$), the agent runs **soft
  Bayes** (the posterior never reaches $\{0,1\}$), and the planner's epistemic drive is the **exact mutual
  information** $\mathrm{IG}(p;\epsilon)=\mathcal H(p)-\mathbb E_o[\mathcal H(p')]=I(b;o\mid d)$ — verified to
  equal the soft-Bayes belief-entropy drop to $10^{-7}$, with the three limits $\mathrm{IG}(\epsilon{=}0)=\mathcal
  H(p)$ (recovering the §3.5.1 win), $\mathrm{IG}(\epsilon{=}\tfrac12)=0$, and $\mathrm{IG}(p\in\{0,1\})=0$. *The one
  design decision:* a noiseless reveal makes $\mathcal H(p)$ collapse to $0$ exactly, so the §3.5.1 curiosity
  *self-extinguished* and the agent committed; under soft evidence $\mathrm{IG}$ stays small-but-nonzero, and a
  bare $z$-score renormalises that vanishing signal back to unit scale — pulling the agent to the cue *forever*.
  Gating the channel by **normalised belief entropy** $g_{\rm epi}=\mathcal H(p)/\ln2\in[0,1]$ (the mutual
  information's own ceiling) re-arms the self-extinguishing envelope; the gate is a *belief scalar*, so the loop
  stays $\mathrm{SE}(3)$-invariant and the $\beta{=}0$ reward-only baseline is untouched. The win **survives**
  ($\times0.614$ true-goal-error cut, CI$[0.499,0.749]$, past the same provable hedge floor), **recovers the §3.5.1 win**
  as $\epsilon_0\to0$ (EFE $0.333\approx$ oracle), and — the built-in **falsifiable negative** — **vanishes** when
  the channel goes useless ($\epsilon_0{=}0.45$: EFE $0.723\approx$ reward-only $0.663$), with sensing effort
  climbing monotonically $5.6\to15.7$ as the bit degrades. $\mathrm{IG}$ depends on the latent only through the
  invariant distance, so the whole noisy loop is still $\mathrm{SE}(3)$-exact (IG-field $7\times10^{-7}$,
  plan-equivariance $8\times10^{-9}$) where the free MLP shatters it (IG-invariance $0.17$). Seven guards.
- **Few-body $\to$ many-body: a *combinatorial* generalisation axis.** Every result above generalises
  across the *continuous* group; we open an orthogonal **discrete** axis — object **cardinality** $O$.
  Train the interacting world model at a *single* count $O{=}3$ and test zero-shot at $O\in\{1,2,4,5,6\}$. There
  is no Lie generator carrying $O{=}3$ to $O{=}5$, so equivariance *alone* cannot buy it; what does is the **one
  design decision** — a **count-stable mean message** $\bar r_i=\frac1{O-1}\sum_{j\ne i}\hat r_{ij}$. A *mean* of
  unit vectors lives in the unit ball $\lVert\bar r_i\rVert\le1$ at *every* count (contracting $1.0\to0.94$ as
  $O{:}2\to6$), so the message distribution a one-count predictor sees is count-stable — where a *sum* would grow
  with $O$ and break transfer; $O{=}2$ recovers the §3.4.1 two-body teacher verbatim and $O{=}1$ reduces to pure
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
  cross-product cap, a modest $\times3.46$. Eight guards.
- **Discover $\to$ exploit: a *discovered* symmetry, distilled into a free predictor.** Every
  across-group win above is bought by a symmetry **hand-wired** into the architecture; the symmetry-discovery rung had already shown
  the prior is *discoverable* — from a blank slate of learnable $3\times3$ matrices it rediscovered a frozen
  teacher's $\mathfrak{so}(3)$ (and an $\mathfrak{so}(2)_z$ on a rotation-broken teacher) with antisymmetry and
  bracket-closure **emergent**, not imposed. This rung turns that *measurement* into a *method*: freeze one
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
  ($\times1.09$ vs $\times1.06$) — the discovery **costs nothing**; *(iv)* the **falsifiable** check —
  distilling only the discovered $\mathfrak{so}(2)_z$ helps the z-axis OOD $+46\%$ but the off-axis only $+17\%$,
  transferring **exactly** the symmetry discovered, no more. The honest limit (**soft $\neq$ hard**): the distilled
  MLP is much flatter than free but does not reach the VN floor (distilled OOD $0.632>2\times$ VN $0.300$) — soft
  regularisation *approximates* equivariance where the built-in prior *enforces* it (the soft-equivariant dial again). So
  the across-group prior is not only *learnable* but *exploitable*: a symmetry discovered from
  data and distilled into a free predictor recovers most of the 举一反三 the hard-wired model gets for free
  (this is **Proposition 2**'s exploit half made concrete — at the penalty's zero the free predictor meets
  Proposition 1's (H3) over the *discovered* subgroup, so [B] rides across it, in the soft limit),
  matching the oracle and transferring exactly what it found — with a documented soft-vs-hard gap, not float-floor
  exactness. Six guards.
- **The active-inference win transfers beyond a *constructed* POMDP — the de-construction completed.**
  The noisy-cue rung removed §3.5.1's *noiseless* crutch; the **other** crutch was the *constructed mirror* — a hidden
  *bit* with two opposite goals whose midpoint is the start, hand-tuned so the one cue sat exactly transverse.
  This rung removes it: the mirror becomes a **generic $K$-target constellation** ($K\ge3$ scattered in a random
  plane with **no antipodal pair at any $K$** — a gap stick-breaking sampler gives $0$ violations over $2000$
  draws, every pair $>38^\circ$ and every angle $>30^\circ$ from $180^\circ$), so the belief is a genuine
  **$K$-ary categorical** with no "opposite" to exploit. The drive is the **exact categorical mutual information**
  $\mathrm{IG}(p;\epsilon,K)=\mathcal H(p)-\mathbb E_o[\mathcal H(p')]=I(b;o\mid d)$ of a **$K$-ary symmetric
  channel** $P(o{=}j\mid b{=}i)=(1{-}\epsilon)[i{=}j]+\tfrac{\epsilon}{K-1}[i{\ne}j]$, crossover annealing with
  the invariant latent distance to the useless floor $\epsilon_\star=(K{-}1)/K$; categorical soft Bayes never
  collapses. **The noisy-cue rung is recovered exactly as $K{=}2$** ($\epsilon_\star(2)=\tfrac12$, $\mathrm{IG}$/crossover
  match to $10^{-7}$). *(i) It attains the oracle floor:* on $24$ generic $K{=}3$ POMDPs ($\epsilon_0{=}0.15$) the
  EFE agent reads the off-path cue $10.6\times$, resolves the belief to $p_{\text{true}}{=}1.00$, and lands at
  $0.387$ — within noise of the oracle $0.376$ (gap $+0.011$, CI$[-0.062,+0.089]$ *includes* $0$) and $\times0.565$
  of the reward-only hedge $0.685$ (paired drop $+0.298$, CI$[+0.204,+0.400]$). *(ii) It scales with $K$:* the win
  holds at $K{=}3,4,5$ (ratios $0.60/0.71/0.55$, every drop-CI lower bound $>0$). *(iii) The kept ingredient is the
  premise, made falsifiable:* a *separable* epistemic affordance is what active inference needs, not a crutch, and
  two negatives both fire — the win **vanishes** when the cue is useless ($\epsilon_0{=}\tfrac23$, ratio $1.00$)
  **and** when the affordance collapses to sense$=$commit (ratio $1.25$, EFE still senses *more*, $25.3$ vs $17.2$)
  — pinning the advantage to the affordance, **not** the mirror. The whole loop stays $\mathrm{SE}(3)$-exact
  (the categorical MI is a function of the invariant latent cue-distance, so this is the §5 instance of
  Proposition 3: IG-field $6\times10^{-6}$, true-goal outcome $\le2\times10^{-6}$, plan-equivariance
  $2\times10^{-8}$) where the free MLP shatters it (IG-field $0.29$, outcome $1.0$/$49^\circ$). What remains untested is a *fully*
  non-constructed benchmark, no longer the mirror. Eight guards.
- **The one outright failure is resolved — decoder-free goal-*reaching*, made exactly equivariant (§3.3.2).**
  The open-loop 3D [C] — purely-latent planning toward a goal cloud without a decoder — was the project's lone outright
  negative (both models closed a *negative* fraction of the orientation gap). §3.3.2 diagnoses it (not a knob):
  a one-step-trained predictor's $H{=}6$ rollout drifts $\sim\!2.0$ from the encoded truth, so the encoder goal
  $E(X_g)$ sits *off* the reachable manifold and the terminal $L_2$ is ill-scaled. The cure is three decoder-free,
  exactly-equivariant ingredients: **rollout-consistency training** (BPTT to an EMA target encoder), the §3.3.1
  SE(3)-equivariant CEM planner verbatim, and an **SE(3)-native latent-Procrustes goal** (geodesic angle of the
  Kabsch $R^\star$ aligning $z_0\to z_g$). Decoder-free reaching flips $+0.006\to+0.527$ — **partial** (the residual
  to a $+0.696$ predictor-space ceiling is the encoder-vs-predictor manifold gap, a horizon limitation). The headline
  is the **exactness**: across a paired seen$+4$-OOD SE(3) orbit the VN's residual orientation error is *identical*
  to $1.8\times10^{-6}°$ (ratio $1.000$, CI$[1.000,1.000]$) while the free MLP degrades to $48.7°$ ($\times1.745$,
  CI$[1.473,2.100]$) — §3.3's closed-loop exactness theorem now for *goal-reaching*. Three panels.

---

## 6. Conclusion

We set out to test one contrarian claim: that a geometric, equivariant, latent-space world model can
earn *exact* generalisation across a symmetry group — 举一反三 — without simulating pixels and without
brute-force scale. Across §3 the claim holds in a precise, falsifiable form. The equivariance theorem
that makes the encoder and the one-step predictor flat by construction (§2) propagates **unbroken**
through every layer we add on top of it: to closed-loop planning in 2D and its full $\mathrm{SE}(3)$
lift (§3.3, §3.3.1), to decoder-free goal-*reaching* (§3.3.2), to a scene group
$\mathrm{SE}(3)^O\rtimes S_O$ with object interaction (§3.4), to an active-inference planner whose
epistemic drive stays $\mathrm{SE}(3)$-invariant under partial observability (§3.5, §3.5.1), and out to
a combinatorial few-body$\to$many-body transfer (§5). Where the world genuinely carries the group, the
across-group gap between the geometric model and a strong non-equivariant baseline is not a tuned margin
but the difference between a *learnable frontier and a wall* (§3.6, §3.7) — and it is a theorem,
recovered to the `e3nn` floor both at initialisation and after training.

We are equally explicit about what the bet does **not** buy. The prior confers **no in-distribution
edge**: inside the training orbit the two models wash out (§3.7), and the across-group payoff is real
only to the extent that the world actually respects the symmetry. Under a controlled break the advantage
degrades gracefully but does not survive (§5); augmentation and scale *approximate* equivariance but
never reach the exact floor, and soft $\neq$ hard. These are the standing boundaries of the
Bitter-Lesson caveat, and we report them as limits, not footnotes.

What remains is to carry the same exactness from these controlled teachers and constructed decision
problems to fully non-constructed embodied benchmarks at scale — the direction the per-experiment
appendix maps out. The mathematics the result rests on — Lie-group representations, intertwiners, and the
geometry of the latent — is permanent capital regardless of how that empirical question resolves; the
wager of this paper is that, across the group, it is also the *cheapest* route to sample-efficient
generalisation, and the evidence here is that it is.

---

## Appendix A. Reproducibility & experiment provenance

### A.1 Environment and determinism

Python 3.11, PyTorch 2.12, `e3nn` 0.6.0, NumPy 2.4, Matplotlib. Dependencies are managed with `uv`
(not pip) and pinned in `requirements.txt`; there is no CUDA — everything runs on a laptop CPU/MPS:

```bash
cd ~/Workspace/se3-ejepa
uv venv && uv pip install -r requirements.txt
```

Every experiment sets explicit seeds (data, initialisation, planner), so re-running reproduces the
tables in the body. The `[A]`/`[B]` claims are *theorems* (§2): they hold at initialisation and after
training, independent of seed. The closed-loop `[C]` confidence intervals are over fixed task/CEM seeds
under a paired design (§3.3). Each structural claim has a matching guard test (the **Guard** column
below) that checks equivariance or invariance **at initialisation and after training** and confirms the
non-equivariant control fails.

### A.2 Running the experiments

With the environment active, every row of the provenance table runs as

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
  .venv/bin/python experiments/<script>.py
```

and each guard test as `.venv/bin/python tests/<test>.py`. The heavier 3D experiments accept a
per-experiment `SMOKE=1` flag (for example `STEP18_SMOKE=1`) for a fast wiring check; numeric dumps and
figures are written to `papers/figures/`. The headline figures (Figure 2, Figure 3) are regenerated
**without retraining** by `make_bet_figures.py`, and Figure 4 by `make_step24_figure.py`.

### A.3 Experiment provenance

Each row maps a result in the body to the experiment that produces it (a script under `experiments/`)
and, where it asserts a symmetry, the guard test that checks it (under `tests/`).

| § | Result | Experiment | Guard |
|---|---|---|---|
| 3.1 | optimiser preserves intrinsic equivariance | `step26_optimizer_equivariance` | `test_step26_optimizer_equivariance` |
| 3.2 | sample efficiency $+$ 举一反三 (synthetic $\mathrm{SO}(2)$) | `step8_sample_efficiency` | — |
| 3.2 | real PushT system $+$ prediction, panels [A][B] | `step10_pusht_closed_loop` | — |
| 3.2 | end-to-end latent JEPA | `step11_latent_jepa` | — |
| 3.2 | decomposed pose-control mechanism | `step12_pose_control` | — |
| 3.3 | paired closed-loop [C], 2D $\mathrm{SO}(2)$ | `step14_pose_control_power` | `test_planner_equivariance` |
| 3.3.1 | 3D $\mathrm{SO}(3)$ latent-JEPA backbone | `step13_se3_latent_jepa` | — |
| 3.3.1 | 3D $\mathrm{SE}(3)$ closed-loop [C] lift | `step18_se3_closed_loop` | — |
| 3.3.1 | translation completes the group | `step15_se3_translation` | — |
| 3.3.2 | decoder-free goal-reaching | `step38_latent_goal_reaching` | `test_step38_latent_goal_reaching` |
| 3.4 | object-centric scene $2\times2$ | `step19_object_centric` | `test_set_equivariance` |
| 3.4.1 | interaction rung (Figure 4) | `step24_object_interaction` | — |
| 3.4.1 | tensor-product degree-2 fix | `step27_tensor_product_message` | `test_step27_tensor_product` |
| 3.5 | EFE curiosity invariance | `step20_active_inference` | `test_efe_invariance` |
| 3.5.1 | partial-observability task win | `step25_active_inference_task` | `test_step25_salience_invariance` |
| 3.6 | sample-efficiency frontier (Figure 2) | `step21_sample_efficiency_frontier` | `test_sample_efficiency_frontier` |
| 3.7 | $(g\times N)$ symmetry-break $\times$ data plane | `step22_symmetry_data_phase` | `test_symmetry_data_phase` |
| 3.7 | large-$N$ no-widening (Figure 3) | `step23_indist_largeN` | — |
| 4 | training-seed error bar | `step17_multiseed_closed_loop` | — |
| 5 | misspecification boundary | `step16_misspecification` | — |
| 5 | fair augmentation baseline (2D, 3D) | `step28_fair_augmentation_baseline`, `step28_fair_augmentation_3d` | — |
| 5 | controlled scaling sweep (2D, 3D) | `step29_scaling_sweep`, `step29_scaling_sweep_3d` | — |
| 5 | soft-equivariant dial (2D, 3D) | `step30_soft_equivariant`, `step30_soft_equivariant_3d` | — |
| 5 | multi-step rollout horizon (2D, 3D) | `step31_rollout_horizon`, `step31_rollout_horizon_3d` | — |
| 5 | tensor-product degree ladder | `step32_tp_degree_ladder` | `test_step32_degree_ladder` |
| 5 | emergent symmetry discovery | `step33_symmetry_discovery` | `test_step33_symmetry_discovery` |
| 5 | active inference, noisy-channel curiosity | `step34_active_inference_noisy` | `test_step34_active_inference_noisy` |
| 5 | few-body $\to$ many-body transfer | `step35_many_body` | `test_step35_many_body` |
| 5 | discover $\to$ exploit distillation | `step36_discover_exploit` | `test_step36_discover_exploit` |
| 5 | active inference, generic search | `step37_active_inference_search` | `test_step37_active_inference_search` |

### A.4 Core modules

The shared equivariant code lives under `src/`:

- `models/structured.py` — the Vector-Neuron primitives `VNLinear`, `VNReLU`, `StructuredStateEncoder`, `VNPredictor`;
- `models/se3.py` — the $\mathrm{SE}(3)$ point-cloud encoder;
- `models/eqjepa.py` — the JEPA wrapper and latent predictor;
- `training/jepa.py` — the EMA-target $+$ VICReg training loop;
- `training/muon.py` — the symmetry-compatible optimiser probed in §3.1.

A full per-experiment narrative — including the binary task-success caveats and the per-experiment
closed-loop tables — is in [[geometric_payoff.md]].
