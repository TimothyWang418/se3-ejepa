# Equivariant LeJEPA: symmetry-structured identifiability for latent world models

> **Status:** research note / Direction-1 plan (2026-05-31). Two results are now **proved + instantiated**
> with seeded, falsifiable experiments: Prop. 1 (C1, block-isotropy; Step 39, extended to the product
> group $S_O\times SO(3)$ in Step 40) and Prop. 2 (C2, equivariant latent dynamics; Step 41 — the world
> model resolves the gauge pure SSL leaves free). One remains a **proposition-to-finish** (C3, planning
> under $G$-invariant cost) with a proof sketch and an existing experiment (Step 38) that instantiates it.
>
> **The one-paragraph pitch.** LeCun, Balestriero & Klindt now have a *theory* of when a JEPA recovers
> the world's latent variables: LeJEPA's embeddings are **linearly identifiable up to a global rotation
> $Q\in O(n)$**, and that rotation is treated as an unavoidable *nuisance* "inherent to the isotropic
> Gaussian." Their latent-planning guarantee (Thm 5.4) then has to *assume* the cost is invariant under
> the **entire** $O(n)$. That assumption is physically far too strong, and the $O(n)$ indeterminacy is
> exactly the slot a **world symmetry group** $G\hookrightarrow O(n)$ lives in. An *equivariant* JEPA
> replaces the unstructured $O(n)$ nuisance with a known orthogonal representation $\rho(G)$: it (C1)
> changes the optimal SIGReg target from full isotropy to **block-isotropy** (proved below via Schur),
> (C2) makes their stationarity condition transportable across group orbits, and (C3) weakens the
> planning theorem's hypothesis from "$O(n)$-invariant cost" to the realistic "**$G$-invariant cost**"
> — a regime our Step 38 already verifies. The differentiator is not the plumbing (SIGReg on an
> equivariant net — anyone can do that) but the **symmetry-structured identifiability theory**, which
> is absent from their paper and is precisely a representation-theory contribution.

---

## 1. The two papers we stand on

### 1.1 LeJEPA (Balestriero & LeCun, arXiv:2511.08544)

LeJEPA replaces the heuristic anti-collapse machinery of SSL (stop-grad, EMA targets, whitening,
teacher schedules) with a single principled regulariser.

- **Optimal-embedding theorem (Thm 1).** Among embedding distributions with a fixed scalar covariance
  budget, the **isotropic Gaussian $\mathcal N(\mathbf 0,\mathbf I_d)$ uniquely minimises the
  integrated squared bias** of downstream linear/kernel/$k$-NN probes. Lemma 1: anisotropy amplifies
  bias; Lemma 2: anisotropy amplifies variance. So $\mathcal N(\mathbf 0,\mathbf I_d)$ is the
  task-agnostic optimum.
- **SIGReg (Def. 2).** Drive the embedding to that optimum by *sketching* a 1-D normality test along
  random directions $\mathbf a$:
  $$\mathrm{SIGReg}_T\big(\mathbb A,\{f_\theta(x_n)\}\big)=\frac1{|\mathbb A|}\sum_{\mathbf a\in\mathbb A}T\big(\{\mathbf a^\top f_\theta(x_n)\}_{n=1}^N\big),$$
  with the recommended test $T$ = **Epps–Pulley** (weighted $L^2$ distance between the empirical
  characteristic function $\hat\phi_X(t)=\tfrac1n\sum_j e^{itX_j}$ and the standard-Gaussian CF
  $\phi(t)=e^{-t^2/2}$, weight $w(t)=e^{-t^2/\sigma^2}$).
- **Full objective (Eq. 9).** $\displaystyle \mathcal L_{\mathrm{LeJEPA}}=\frac{\lambda}{V}\sum_v
  \mathrm{SIGReg}(\{z_{n,v}\})+\frac{1-\lambda}{B}\sum_n\|\mu_n-z_{n,v'}\|_2^2$, prediction loss
  pulling each view to the mean global-view embedding $\mu_n$; single hyperparameter $\lambda\!\approx\!0.05$.
- **No symmetry.** The paper has *no* group action, equivariance, or orbit; invariance is only the
  multi-view augmentation prior.

### 1.2 When Does LeJEPA Learn a World Model? (Klindt, LeCun & Balestriero, arXiv:2605.26379, 2026-05-25)

This is the **identifiability** theory — when does the LeJEPA latent recover the *true* generative
factors. (Their AR/OU coefficient is written $r\in(0,1)$ here to avoid clashing with our
representation $\rho$.)

- **World assumptions (3.1).** (i) factorised latents/transitions across coordinates; (ii)
  **stationarity** $p(z)=p(z')$; (iii) **additive noise** $z_i'=m_i(z_i)+\eta_i$, $\eta_i\perp z_i$.
  For Gaussian latents these force the **Ornstein–Uhlenbeck** transition
  $z'=r\,z+\sqrt{1-r^2}\,\eta,\ \eta\sim\mathcal N(\mathbf 0,\mathbf I_n)$.
- **Linear / orthogonal identifiability.** The composed map $h=f\circ g$ is *linearly identifiable*
  when $h(z)=Qz$ for some **orthogonal $Q\in O(n)$** — recovery up to a global rotation/reflection,
  "inherent to the isotropic Gaussian."
- **Thm 5.1 (forward).** For any measurable $h$ with $h(z)\sim\mathcal N(\mathbf 0,\mathbf I_n)$,
  $\mathcal L(h)\ge 2(1-r)n$, with **equality iff $h(z)=Qz$, $Q\in O(n)$.** Mechanism: a Hermite-
  polynomial **spectral decomposition** in which *every degree of nonlinearity strictly reduces the
  positive-pair correlation*, so the linear map is the unique optimum.
- **Thm 5.2 (converse).** If every whitened minimiser is linear, then $z$ is **Gaussian** — Gaussian
  is the *unique* latent law admitting the guarantee.
- **Thm 5.3 (approximate).** Graceful degradation: $\mathbb E\|h(z)-Qz\|^2\le D+(\varepsilon+D)^2$
  with $D=\delta/(2r(1-r))$; the alignment gap $\delta$ dominates, whitening error $\varepsilon$ is
  "essentially free."
- **Thm 5.4 (planning).** Under $h(z)=Qz$ **and a cost invariant under the whole $O(n)$**,
  $\ell(Rz,a)=\ell(z,a)\ \forall R\in O(n)$, latent-space planning is exact:
  $\hat V^\*(h(z_0))=V^\*(z_0)$ and $\hat a^\*_{1:t}=a^\*_{1:t}$.
- **Still no group.** Orthogonality is treated as a *nuisance* to quotient; the only "symmetry" is the
  rotation-invariance of the isotropic Gaussian and the (strong) $O(n)$-invariant-cost hypothesis.

---

## 2. The gap

Their entire theory is phrased **"up to a global $O(n)$"** and then has to **assume $O(n)$-invariant
costs** to plan. But:

1. $O(n)$ is the *largest possible* indeterminacy; for a world with a real symmetry it is far too
   coarse. The physically meaningful object is a **subgroup** $G\hookrightarrow O(n)$ (e.g.
   $\rho(\mathrm{SO}(3))$ acting on type-1 latents), not all of $O(n)$.
2. Almost no real cost is invariant under *arbitrary* latent rotations — the $O(n)$-invariant-cost
   hypothesis of Thm 5.4 is close to vacuous in practice. Real costs are invariant under the *world's*
   symmetry $G$ (a reaching cost is SE(3)-invariant, not invariant under scrambling unrelated latent
   axes).
3. Their model is **passive**: identifiability is something the data-generating process either grants
   or doesn't. Equivariance lets us *install* the symmetry in the architecture and ask a sharper
   question — when the world has symmetry $G$, what does an encoder that **carries $\rho(G)$ exactly**
   add to the identifiability picture?

This is the white space. Below, $G$ is a compact group, $\rho:G\to O(n)$ an orthogonal representation
on the latent $\mathbb R^n$, and "equivariant encoder" means the latent law is $G$-invariant,
$\rho(g)Z\overset{d}{=}Z\ \forall g\in G$ (which holds when the data law is $G$-invariant and
$f_\theta$ is equivariant, and can always be enforced by symmetrisation).

---

## 3. C1 — Block-isotropy is the equivariant SIGReg target (proved)

**Proposition 1 (Schur block-isotropy).** Let $Z\in\mathbb R^n$ be mean-zero with $G$-invariant law
under $\rho:G\to O(n)$, and $\Sigma=\mathbb E[ZZ^\top]$. Decompose into real isotypic components
$\mathbb R^n=\bigoplus_i V_i^{\oplus m_i}$ ($V_i$ the distinct real irreducibles, $d_i=\dim V_i$,
multiplicity $m_i$). Then
$$\rho(g)\,\Sigma=\Sigma\,\rho(g)\ \ \forall g\quad\Longrightarrow\quad \Sigma=\bigoplus_i\big(\mathbf I_{d_i}\otimes B_i\big),\qquad B_i\succeq 0\ \text{symmetric }m_i\times m_i .$$
That is, $\Sigma$ is **block-isotropic** (in any $G$-adapted orthonormal basis — one block-diagonalising
$\rho$ into irreps; our type-0/type-1 latent is already such a basis): a scalar multiple of the identity
*inside* each irreducible copy, with mixing allowed only across the $m_i$ multiplicity slots of the
*same* irrep, and **zero** coupling between inequivalent irreps. (For complex/quaternionic-type $V_i$,
$B_i$ is taken over $\mathbb C$/$\mathbb H$; the "scalar on each irrep copy" conclusion is unchanged. The
case we use — $\mathrm{SO}(3)$ on integer-$\ell$ features — is **entirely real type**, so the clean form
above holds verbatim.)

*Proof.* $G$-invariance of the law gives $\rho(g)\Sigma\rho(g)^\top=\mathbb E[\rho(g)ZZ^\top\rho(g)^\top]
=\mathbb E[ZZ^\top]=\Sigma$, and since $\rho(g)\in O(n)$ this is $\rho(g)\Sigma=\Sigma\rho(g)$: $\Sigma$
is a $G$-equivariant endomorphism of $\mathbb R^n$. By Schur's lemma an equivariant map carries the
$V_i$-isotypic component into itself and **annihilates** cross-terms between inequivalent irreps
(a nonzero equivariant map between non-isomorphic irreducibles would be an isomorphism). Restricted to
$V_i^{\oplus m_i}\cong V_i\otimes\mathbb R^{m_i}$, the commutant of $\rho|_{V_i}$ is (real type)
$\mathbf I_{d_i}\otimes\mathrm{Mat}_{m_i}(\mathbb R)$, so $\Sigma|_i=\mathbf I_{d_i}\otimes B_i$;
symmetry and PSD-ness of $\Sigma$ pass to $B_i$. $\qquad\blacksquare$

**Why it matters.**

- LeJEPA's target $\Sigma=\sigma^2\mathbf I_n$ is the special case $B_i=\sigma^2\mathbf I_{m_i}$ — i.e.
  **forcing the same scale on every irrep.** It is *attainable* inside the equivariant class but is a
  measure-zero slice of it, and there is no reason a type-0 scalar feature and a type-1 vector feature
  should carry equal variance. Vanilla isotropic SIGReg therefore **fights** equivariance whenever
  $\rho$ mixes inequivalent irreps of different natural scale.
- The correct equivariant objective is **block-SIGReg**: sketch normality *within* each isotypic
  block and test each toward an isotropic Gaussian of its **own** scale $\sigma_i^2$, leaving the
  cross-block scales free. This is the maximum-entropy $G$-invariant Gaussian at given per-irrep
  variances — the equivariant analogue of "isotropic Gaussian."
- **Identifiability refinement (the prize) — stated precisely.** Three groups must be kept apart, and
  an earlier draft conflated them. Let $\Sigma$ be block-isotropic with **distinct** per-irrep scales
  $\sigma_i^2$. The residual gauge is the set of orthogonal $Q$ relating two equally-valid solutions:
  1. **Law-matching only.** $Q$ must preserve the target law, $Q\Sigma Q^\top=\Sigma$. With $\sigma_i^2$
     distinct, $\Sigma$'s eigenspaces are exactly the isotypic components, so $Q$ must preserve each:
     $$Q\in\mathrm{Stab}_{O(n)}(\Sigma)=\textstyle\prod_i O(d_i m_i).$$
     This already drops the gauge from $O(n)$ (LeJEPA's degenerate $\Sigma=\sigma^2 I$, eigenspace all
     of $\mathbb R^n$) to the within-block product — a strict, spectrum-driven reduction.
  2. **Equivariant recovery.** If we additionally demand the recovery map $h=f\circ g$ be
     $G$-equivariant (true for a matched equivariant encoder on equivariant data), then $Qz=h(z)$
     forces $Q\in\mathrm{Comm}(\rho):=\{Q\in O(n):Q\rho(g)=\rho(g)Q\}$. Intersecting with (1), in the
     real type $Q=\bigoplus_i \mathbf I_{d_i}\otimes Q_i$ with $Q_i\in O(m_i)$, i.e. the residual gauge
     is the **orthogonal commutant**
     $$\boxed{\ \prod_i O(m_i)\ }\quad(\text{mixing only within each multiplicity space}).$$
     **Multiplicity-free** ($m_i\le1$): this is $\prod_i\{\pm1\}$ — a **finite** group of per-irrep sign
     flips. So the latent is identified **up to a finite group**, and the full $\rho(G)$-module
     structure (which axes carry which irrep, and the within-irrep frames) is pinned.
  3. **$\rho(G)$ itself** is a *third* group (the image of the representation); the gauge is **not**
     $\rho(G)$ — it is $\rho(G)$'s commutant. The honest one-liner is therefore: *equivariance +
     block-isotropy + distinct scales reduces the gauge from $O(n)$ to the (finite, when
     multiplicity-free) commutant $\prod_i O(m_i)$, and in doing so **identifies the $\rho(G)$-module
     structure** — i.e. recovers the true degrees of freedom together with their symmetry labels.* That
     last clause is exactly the "recover the true DOF *with their symmetry structure*" desideratum their
     abstract opens with, now with the gauge named correctly.

Confidence: Prop. 1 itself **0.9** (textbook Schur; for SO(3) integer-$\ell$ all irreps are real type, so
the clean form $\Sigma=\bigoplus_i\mathbf I_{d_i}\otimes B_i$ holds with full rigor — no complex/
quaternionic case to handle). "Block-SIGReg is the right target" 0.8. The gauge refinement **0.8** now
that it is stated as the commutant rather than $\rho(G)$ (the one dependency is matched representations,
so that $h$ is genuinely $\rho$-equivariant; the distinct-scale hypothesis is what makes the spectrum
expose the blocks — equal scales degenerate the spectrum and re-inflate the gauge to $\prod_i O(d_i m_i)$).

---

## 4. C2 — Equivariant latent dynamics: the world model resolves the gauge SSL leaves free (Step 41)

Their guarantee requires the world to lie in the stationary additive-noise (OU) class, and identifies
the latent only up to the *static* nuisance $Q\in O(n)$. §3 sharpened the static picture but found the
per-irrep scales **underdetermined in pure SSL** (§7): with equal scales $\Sigma_\infty=\sigma^2\mathbf
I$, the static spectrum is degenerate and the gauge stays stuck at $O(n)$. C2 puts a *world* on top — a
$G$-equivariant transition — and shows the **dynamics** carry the identifiability the static covariance
cannot. (As elsewhere, the AR/OU coefficient is written $r$ to avoid clashing with $\rho$.)

**Proposition 2 (equivariant OU: Schur dynamics, gauge resolution, orbit transport).** Let
$\rho:G\to O(n)$ with real isotypic decomposition $\mathbb R^n=\bigoplus_i V_i^{\oplus m_i}$ ($d_i=\dim
V_i$). Let the latent evolve by a linear-Gaussian OU $z_{t+1}=Az_t+\varepsilon_t$,
$\varepsilon_t\sim\mathcal N(\mathbf 0,Q)$, whose kernel is **$G$-equivariant**,
$T(\rho(g)z'\mid\rho(g)z)=T(z'\mid z)$ — equivalently $A\rho(g)=\rho(g)A$ and $\rho(g)Q\rho(g)^\top=Q$ for
all $g$. Then:

(a) **Schur block dynamics.** $A=\bigoplus_i\mathbf I_{d_i}\otimes A_i$,
$Q=\bigoplus_i\mathbf I_{d_i}\otimes Q_i$, and the stationary covariance (the unique PSD solution of the
discrete Lyapunov equation $\Sigma_\infty=A\Sigma_\infty A^\top+Q$) is
$\Sigma_\infty=\bigoplus_i\mathbf I_{d_i}\otimes S_i$ with $S_i=A_iS_iA_i^\top+Q_i$ — the dynamical
analogue of Prop. 1.

(b) **The dynamics resolves the static gauge.** Take $A_i=r_i\mathbf I_{m_i}$ and choose
$Q_i=\sigma^2(1-r_i^2)\mathbf I_{m_i}$, so $S_i=\sigma^2\mathbf I_{m_i}$: **distinct dynamics, equal
stationary scale.** Then the *static* spectrum is degenerate — $\Sigma_\infty=\sigma^2\mathbf I_n$, gauge
$\mathrm{Stab}_{O(n)}(\Sigma_\infty)=O(n)$ — exactly §7's underdetermined regime; yet the *dynamical*
(drift) operator has spectrum $\operatorname{spec}A=\{r_i\}$ with each $r_i$ of multiplicity $d_im_i$, and when the $r_i$ are **distinct**
its eigenspaces are precisely the isotypic blocks, so the gauge that commutes with $A$ drops to
$\prod_i O(d_im_i)\subsetneq O(n)$. The world model's transition therefore identifies strictly more than
its stationary law.

(c) **Orbit transport (C2 proper) + refined forward bound.** The one-step Bayes-optimal predictor is the
conditional mean $z\mapsto Az$ (linear, equivariant), and its Bayes error is **orbit-constant**: for every $g$,
$$\mathbb E\big\|\rho(g)z'-A\,\rho(g)z\big\|^2=\mathbb E\big\|\rho(g)(z'-Az)\big\|^2=\mathbb E\|z'-Az\|^2=\operatorname{tr}Q=\sum_i d_im_i\,\sigma^2(1-r_i^2),$$
using $A\rho(g)=\rho(g)A$, kernel-equivariance of the target, and orthogonality of $\rho(g)$. Thus
stationarity $+$ additive noise, verified on a fundamental domain $\mathcal F$, transport to all of
$\mathbb R^n$; the last equality is the **per-irrep refinement** of KLB's forward bound (Thm 5.1's scalar
$2(1-r)n$ resolves into $\sum_i d_im_i\sigma^2(1-r_i^2)$, each irrep contributing its own $1-r_i^2$).

*Proof.* (a) $G$-equivariance of $A,Q$ is Schur exactly as Prop. 1; the commutant
$\{\bigoplus_i\mathbf I_{d_i}\otimes M_i\}$ is a subalgebra closed under $M\mapsto AMA^\top+Q$, and the
Lyapunov solution is the limit of its iterates from $0$, hence block-diagonal with $S_i$ solving the
per-block equation. (b) For $A_i=r_i\mathbf I$, $S_i=Q_i/(1-r_i^2)$; the choice
$Q_i=\sigma^2(1-r_i^2)\mathbf I$ gives $S_i=\sigma^2\mathbf I$, so $\Sigma_\infty=\sigma^2\mathbf I_n$
(eigenvalue $\sigma^2$, multiplicity $n$ — full $O(n)$), while $A=\bigoplus_i r_i\mathbf I_{d_im_i}$ has
eigenvalue $r_i$ on $V_i^{\oplus m_i}$; distinct $r_i$ make these the eigenspaces, and an orthogonal
commuting with $A$ must preserve each, giving $\prod_iO(d_im_i)$. (c) Immediate from the three stated
facts. $\qquad\blacksquare$

The honesty clause from the earlier sketch survives intact: equivariance does **not** force an arbitrary
world into the OU class — it reduces *verification* of an already-$G$-symmetric world from $\mathbb R^n$
to $\mathcal F$, and the flatness identity (c) certifies the transport is exact. What is new beyond the
sketch is (b): the dynamics supply, *for free*, the per-irrep scale separation pure SSL leaves
underdetermined (§7) — the predictor **is** the "scale-sensitive task" Step 40 [E2] had to install by
hand, here handed over by the world itself.

### 4.1 Minimal experiment — built and run (Step 41, laptop CPU, seeded)

`experiments/step41_equivariant_dynamics.py` (+ `tests/test_step41_equivariant_dynamics.py`, 9 gates)
instantiates Prop. 2 on the same mixed-type latent as §7: $n=22$,
$\rho(R)=\mathbf I_4\oplus(\mathbf I_6\otimes R)$, with the headline OU $A=0.2\,\mathbf I_4\oplus0.9\,\mathbf
I_{18}$, $Q=\operatorname{diag}\big(\sigma^2(1-r_i^2)\big)$, $\Sigma_\infty=\mathbf I_{22}$ — distinct
dynamics $r_0=0.2\neq r_1=0.9$ but equal stationary scale. Two halves with separate guards; **all pass**
(full run, seeded; smoke via `STEP41_SMOKE=1`).

**[A] Objective level (deterministic — the rigorous core).** [A1] the commutant-projected drift commutes
with $\rho(R)$ to the float floor ($0.0$) while a generic dense drift does not ($2.94$), and the
projection is faithful on the headline $A$ ($\|P_C(A)-A\|_\infty=6\times10^{-8}$). [A2] **the headline
gauge ladder:** the static spectrum of $\Sigma_\infty=\mathbf I$ is one $22$-fold cluster → $\dim
O(22)=231$, while the *drift* spectrum splits into the $[18,4]$ isotypic eigenspaces → $\dim O(18)\times
O(4)=159$ — robust across `gap_factor`$\in\{1.5,2,3,4\}$ (analytic ladder $231\xrightarrow{\text{distinct
}r}159\xrightarrow{\text{known }\rho}21$). [A3] **C2 flatness, made discriminating:** on the anisotropic
$z_t$ law the commuting drift's one-step Bayes error is orbit-constant ($7.378$ vs predicted
$\operatorname{tr}Q=7.260$; spread $4.5\times10^{-7}$) while a spatially-anisotropic, non-commuting drift
varies along the orbit (spread $0.448$). (On the *isotropic* law $\mathbb E\|\cdot\|^2$ collapses to a
rotation-invariant Frobenius norm, so even a wrong drift looks flat in expectation; the test transports
on the anisotropic law, where a non-equivariant world genuinely varies — a principled fix, not a loosened
threshold.)

**[B/A′] Predictor equivariance, init and post-training.** A mixed-type equivariant predictor (a
Vector-Neuron channel-mix gated by invariant features, with cross-type *capacity*) is exactly equivariant
at init ($3.6\times10^{-7}$) and **stays so after 30 epochs** of one-step-MSE training
($7.2\times10^{-7}$); the MLP control misses by $\sim0.63$ at init and $1.15$ after training.

**[C] Prop. 2 on the *learned* transition.** On the $G$-invariant law the equivariant predictor's
cross-time second moment $C_1=\mathbb E[f(z)z^\top]$ is Schur block-diagonal (cross $=0.070$, each $1o$
block $3\times3$-isotropic at $1.06$) and recovers the true per-irrep AR coefficients $\hat
r=(0.208,0.902)$ vs truth $(0.2,0.9)$. **Honest nuance:** the MLP *also* fits a near-block-diagonal $C_1$
and recovers $\hat r=(0.204,0.885)$ — the linear OU drift is an easy target, so [C] on the invariant law
is **not** where eq and MLP part ways (the gate is on the eq predictor's exact recovery, not on an MLP
failure here). The **negative control** is the falsifier: the *same* equivariant map on a
non-$G$-invariant *anisotropic* law breaks $3\times3$-isotropy (iso $4.83$) — so [C] *can* fail, and fails
exactly when Prop. 2's premise (invariant law) is removed.

**[D] The payoff.** The *static* covariance of $z_t$ is degenerate (gauge $231$), but the *learned*
equivariant drift's dynamical spectrum lands on the $[18,4]$ rung (gauge $159$) — on a learned net, the
world model resolves the gauge pure SSL leaves free (§7's underdetermined split), realising (b)
empirically. The MLP's learned drift also reaches $159$ *in-distribution* (the OU's $r_1/r_0=4.5$ spectral
gap is easy to inherit), but its drift is **not** equivariant, so that rung does not transport off the
orbit — which is exactly what [E] exposes.

**[E] 举一反三.** A predictor fit on a thin $z$-rotation wedge transfers across all of SO(3) for the
equivariant model — OOD/seen relMSE $\times1.02$ (flat, the orbit-transport of [A3] realised on a learned
net) — while the MLP degrades $\times2.41$ off the wedge. The eq model's $159$ rung is the *same* rung on
every orbit; the MLP's is valid only where it was trained.

**Controls & falsifiability.** Seeds fixed (full run reproducible); smoke vs full sizes; a dedicated
$N=8192$ covariance sample for [C]/[D]; equivariance asserted init $+$ post-training. The suite gates the
deterministic core ([A1]/[A2]/[A3]), the structural Prop.-2 claim and its negative control ([C]), and the
learned payoff ([D]/[E]); the nine mechanism guards in `tests/test_step41_equivariant_dynamics.py` mirror
them (Schur drift $+$ commutant projection, stationary degenerate static spectrum, dynamical ladder $+$
`gap_factor` robustness, orbit-flatness, Haar law, and Prop. 2 failing exactly when its premise is
removed). A run that fails to separate reports `INCONCLUSIVE` rather than relaxing a threshold.

Confidence: Prop. 2(a) **0.9** (Schur $+$ Lyapunov, same rigour as Prop. 1); the gauge-resolution (b)
**0.85** as a target-class statement (distinct $r_i$ is the live hypothesis — the exact mirror of §7's
distinct-scale condition) and **0.7** realised on a learned net (Step 41 [D] reaches $159$, but so does
the MLP in-distribution; equivariance is what makes the rung *transport*, [E]); the orbit-transport
flatness (c) **0.85** (a clean identity, certified to $10^{-6}$). C2 overall **0.8** — upgraded from the
0.65 sketch now that it is a theorem with a falsifiable experiment.

---

## 5. C3 — Planning under $G$-invariant (not $O(n)$-invariant) costs; Step 38 is the instance

Thm 5.4 needs the cost invariant under **all** of $O(n)$. Under an equivariant encoder whose
identifiability is pinned to $\rho(G)$ (C1, distinct-scale case), the same argument goes through under
the strictly weaker, physically natural hypothesis:

**Proposition 3 (equivariant planning, sketch).** If $h(z)=\rho(g_0)\,z$ for some fixed $g_0$ (recovery
up to the world symmetry, not arbitrary $O(n)$) and the cost is **$G$-invariant**,
$\ell(\rho(g)z,a\cdot g)=\ell(z,a)\ \forall g\in G$ (with the induced action on actions), then latent
planning is exact: $\hat V^\*(h(z_0))=V^\*(z_0)$ and the optimal action sequences coincide up to the
group action. The proof mirrors 5.4 but only invokes invariance under $\rho(G)\subset O(n)$.

**This is already verified.** Step 38 (latent-goal reaching *without a decoder*): an SE(3)-invariant
reaching cost, planned by an equivariant CEM planner directly in the latent, reaches **identically
across the SE(3) orbit** — OOD/seen ratio $1.000$, CI $[1.000,1.000]$ — versus a non-equivariant MLP
planner at $\times1.745$. That is exactly Thm 5.4's conclusion ($\hat V^\*=V^\*$, matched optimal
actions) holding in the regime their theorem does **not** cover (a $G$-invariant, *not* $O(n)$-
invariant, cost). Step 38 is the experiment a referee would demand for Prop. 3, and we ran it before we
knew it was the experiment. Confidence 0.75 (sound given C1's gauge-pinning; the one dependency is that
the encoder's residual gauge really is $\rho(G)$, i.e. C1's distinct-scale hypothesis).

---

## 6. A bridge already built: Step 32 ↔ their Hermite spectral penalty

Thm 5.1's forward direction is a **Hermite-degree** spectral decomposition: each degree of nonlinearity
strictly reduces positive-pair correlation, so the linear map wins. Our **Step 32** built a predictor
with a *tunable* maximum polynomial degree, $d_{\max}(L)=2^L$ (the `VNTPLadderPredictor`), and showed a
degree-3 interaction target is first representable at rung $L=2$. So our degree ladder is a
**constructive, equivariant** realisation of their spectral-degree analysis: their scalar-Hermite basis
is the $G$-trivial case, and the equivariant generalisation replaces Hermite polynomials by the
Clebsch–Gordan / spherical-harmonic decomposition of tensor powers of $\rho$. Conjecture: "alignment
penalises Hermite degree" becomes "alignment penalises the higher-$\ell$ irreps in $\rho^{\otimes k}$,"
and the degree ladder measures the penalty rung by rung. Confidence 0.4 (suggestive; a genuine
opportunity, not yet a result).

---

## 7. Minimal experiment — built and run (Step 39, laptop CPU, seeded)

`experiments/step39_block_sigreg.py` (+ `tests/test_step39_block_sigreg.py`) realises C1 on a mixed-type
SO(3) point-cloud latent: $n_0=4$ invariant scalars (`0e`) and $n_1=6$ vectors (`1o`), so
$\rho(R)=\mathbf I_4\oplus(\mathbf I_6\otimes R)$ on $\mathbb R^{22}$ — **two inequivalent irreps**, the
minimal setting where vanilla and block-SIGReg genuinely differ (with one irrep they coincide). The
analytic gauge ladder is $O(22)\,[\dim 231]\xrightarrow{\text{block}}O(4)\times O(18)\,[159]
\xrightarrow{\text{known }\rho\text{'s }\otimes\mathbf I_3}$ commutant $O(4)\times O(6)\,[21]$. The
encoder (`src/models/se3.py`) gained an `n_out_scalar` head and an `irrep_blocks()` layout descriptor
(`src/geometry/irreps.py`); the two SIGReg variants live in `src/training/sigreg.py`.

The script has two halves with separate guards. **All pass** (full run, seeded; smoke via `STEP39_SMOKE=1`).

**[A] Objective-level, deterministic (the rigorous core).** On *synthetic* block-isotropic Gaussians with
a controlled scale split $\sigma_1/\sigma_0$ at fixed total budget $\tfrac1n\operatorname{tr}\Sigma=1$:
vanilla SIGReg's statistic **grows $\times44$** from ratio 1 → 4 — it *penalises valid, Prop.-1-optimal
laws* — while block-SIGReg stays **flat ($\times0.99$)**. Reading the spectral gauge off the same controlled
laws: the equal-scale law (vanilla's target) is one eigenvalue cluster of 22 → $\dim\mathrm{Stab}_{O(22)}=231$;
a distinct-scale law splits into the $[18,4]$ eigenspaces → $\dim O(18)\times O(4)=159$. This pins the gauge
claim at the level of the **objective's target class**, with no optimisation noise.

*Two falsifiability guards on [A] itself.* (i) **Anti-vacuity positive control:** "block-SIGReg flat on the
valid laws" would be empty if it were flat on *everything*. So we feed it a *spatially-anisotropic* vector
block (each channel $\sim\mathcal N(0,\operatorname{diag}(g))$, $g\not\propto\mathbf 1$, same total budget) —
a law **outside** Prop. 1's class, breaking the $\propto\mathbf I_3$ structure both Prop. 1 *and* block-SIGReg
require. block-SIGReg **spikes $\times205$** (full; $\times22$ at smoke's higher floor) on it: the flatness is
discriminating. (ii) **Gauge-ladder robustness:** the $231/159$ split is a $\sim16\times$ eigenvalue gap, so it
must survive any clustering threshold — confirmed identical across `gap_factor` $\in\{1.5,2,3,4\}$ (and to 8),
so the ladder is not an artefact of one tuned cut-off.

**[B/A'] Equivariance, init and post-training.** The mixed-type encoder is exactly equivariant (scalar-inv
$2.4\times10^{-7}$, vector-equiv $2.3\times10^{-6}$) and **stays so after 40 epochs** of the faithful
LeJEPA loss (jitter-augmented views pulled to their grad-carrying mean — *no* EMA, *no* stop-grad, *no*
teacher — plus the SIGReg variant); the non-equivariant MLP control misses by $\sim5$–$7$.

**[C] Block-isotropy of the *learned* latent (Prop. 1).** On a Haar (hence $G$-invariant) cloud law, at
$N=8192$ the equivariant latent has cross-irrep coupling $0.015$ and per-channel vector isotropy ratio
$1.07$ — right at the finite-sample floor $1.080$ — i.e. $\Sigma\to\bigoplus_i\mathbf I_{d_i}\otimes B_i$
to noise. The MLP fails both ($0.40$, $2.14$). **Negative control (the falsifier):** Prop. 1 needs *both*
equivariance *and* a $G$-invariant law, so feeding the **same** equivariant encoder a non-$G$-invariant
*wedge* law ($z$-rotations in $[0,90°)$) must *break* block-isotropy — and it does, hard (cross $0.59$, vec-iso
$72.7$). So [C] *can* fail and fails **exactly** when the premise is removed; it is not a metric that passes
regardless. (This is structural — it already holds at init — so the test needs no training.)

**[E] 举一反三 (the payoff).** A *type-respecting* linear probe $\hat y=\sum_a w_a v_a$ fitted on a thin
$z$-rotation wedge transfers across **all** of SO(3): OOD/seen relMSE $\times0.98$ (flat). The MLP's affine
probe degrades $\times8455$ off the wedge. This is the equivariance-flatness theorem (§4) made concrete on
the LeJEPA-regularised latent.

**Honest negative finding — and why it doesn't dent the claim.** [D] reports the *learned* per-irrep scale
split, and in **pure SSL it is underdetermined**: block-SIGReg standardises each block by a *detached* RMS
(so it constrains shape, not scale), the budget penalty pins only the *total*, and the pull-to-mean term is
block-symmetric — so nothing drives the $\sigma_1/\sigma_0$ ratio to a particular value (the run even drove
$\mathrm{var}(0e)\to0$). The scale separation is therefore a property of the **target class** (proved &
demonstrated deterministically in [A]), not something pure SSL converges to; we gate on [A]'s controlled
ladder and report [D] as an un-gated diagnostic rather than weaken a threshold. *Driving* the split to a
chosen value needs a task signal on each irrep — the natural Direction-3 follow-up, **now built and run (§8)**.

**A correctness lesson worth keeping.** Prop. 1 needs the cloud law to be $G$-**invariant**, which requires
the random orientations to be the **Haar** measure (left-invariance). An initial axis-uniform + angle-uniform
$[0,2\pi)$ sampler is *not* Haar ($\mathbb E[R]=\tfrac13\mathbf I\neq\mathbf0$), and it left a systematic
cross-irrep coupling ($\|C_{01}\|_F\approx0.017$) that did **not** shrink with $N$ — masking block-isotropy.
Switching to a uniform-unit-quaternion Haar sampler made cross $\to0$ and vec-iso $\to1$ as $1/\sqrt N$, as
the theorem predicts. A `test_rand_so3_is_haar` regression guard now pins this.

**Controls & falsifiability.** Seeds fixed throughout (full run reproducible byte-for-byte); smoke vs full
sizes; a dedicated large covariance sample ($N=8192$) for [C]/[D] so the isotropy estimate clears its noise
floor; equivariance asserted init + post-training. Beyond positive results, the suite now carries explicit
*falsifiers*, each gated and mirrored in `tests/test_step39_block_sigreg.py` (8 gates): an **anti-vacuity
positive control** (block-SIGReg must spike on a non-Prop.-1 law), a **gap_factor robustness sweep** (the
gauge ladder must not depend on a tuned threshold), and a **Prop.-1 negative control** (block-isotropy must
break on a non-$G$-invariant law). A run that fails to *separate* on any of these reports `INCONCLUSIVE`
rather than relaxing a threshold — so every headline number has a way to be wrong.

---

## 8. Direction 3 — compositional bi-block-SIGReg on a product symmetry $S_O\times SO(3)$ (Step 40)

§7 proved block-SIGReg on a **single** object's SE(3)-type structure. The open question it leaves: does a
*product* symmetry buy a strictly finer identifiability rung that single-object block-SIGReg cannot reach? A
scene of several interchangeable, individually-rotating objects is the natural test — its symmetry group is
$S_O\times SO(3)$ (relabel the objects $\times$ rotate them as one rigid frame), and that product is exactly
what an object-centric world model must respect.

**Prop. 1′ (product-group block-isotropy).** Take a scene of $O$ distinguishable objects, each carrying
$n_0$ scalar features ($0e$) and $n_1$ vector features ($1o$). The scene latent lives in
$\mathbb R^{O}\otimes\mathbb R^{D_{\mathrm{obj}}}$ and carries the **outer-tensor** representation
$P\boxtimes\rho_{SE3}$ of $S_O\times SO(3)$, where $P$ is the $O$-dimensional permutation rep and
$\rho_{SE3}=n_0\,\mathbf 0e\oplus n_1\,\mathbf 1o$. Because the permutation rep splits
$\mathbb R^{O}=\mathbb 1\oplus\mathbf{std}$ (trivial $\oplus$ standard, $\dim\mathbf{std}=O-1$), the latent
decomposes into **four** bi-isotypic blocks
$$(\mathbb 1,0e),\qquad(\mathbb 1,1o),\qquad(\mathbf{std},0e),\qquad(\mathbf{std},1o).$$
Under an $S_O\times SO(3)$-invariant data law, real-type Schur forces the covariance block-diagonal across
these four, each block isotropic in its irrep: $\Sigma=\bigoplus_i \mathbf I_{d_i}\otimes B_i$ — the
product-group analogue of Prop. 1. The $(\mathbb 1,\cdot)$ blocks are the **aggregate**
(permutation-invariant) content; the $(\mathbf{std},\cdot)$ blocks are the **relational** content.
Decoupling $\mathbb 1$ from $\mathbf{std}$ is precisely what $S_O$ — not $SO(3)$ — buys.

**The compositional gauge rung.** With $O=4$, $n_0=n_1=2$ ($D_{\mathrm{obj}}=8$, latent $n=32$), the bi-block
widths $(d\!\cdot\!m)$ are $(2,6,6,18)$ and the residual-gauge ladder is
$$\underbrace{O(32)}_{496}\ \xrightarrow{\ SE(3)\ }\ \underbrace{O(8)\times O(24)}_{304}\ \xrightarrow{\ +\,S_O\ }\ \underbrace{O(2)\,O(6)\,O(6)\,O(18)}_{184}\ \xrightarrow{\ \text{known }\rho\ }\ \underbrace{O(2)^4}_{4}.$$
The middle rung $304\to184$ is the **payoff**: SE(3)-block-SIGReg (§7, which sees only the scalar/vector
split) stops at $304=\binom 82+\binom{24}2$; resolving the $\mathbb 1\oplus\mathbf{std}$ split *inside* each
SE(3) block reaches $184=\binom22+2\binom62+\binom{18}2$. This rung **does not exist for a single object**: at
$O=1$, $\mathbb R^1=\mathbb 1$ and $\mathbf{std}=0$, so there is nothing for $S_O$ to refine — it is a
genuinely *compositional* identifiability gain. An **orthogonal** Helmert change of basis $U\otimes\mathbf
I_{D_{\mathrm{obj}}}$ on the object axis (row $0=$ the mean $=\mathbb 1$; rows $1..O\!-\!1=$ an orthonormal
basis of $\mathbb 1^\perp=\mathbf{std}$) makes the four blocks contiguous without touching the spectrum, so
the ladder is a property of the law, not of the chart.

**[A] Objective level (deterministic, gated).** Bi-block-SIGReg is flat ($\approx3\times10^{-5}$) on every
block-isotropic law, while vanilla isotropic-SIGReg grows $\times86$ on a distinct-scale bi-type law. The
compositional separation is the headline: on a *within-type* $S_O$ split (trivial vs. standard scaled
differently at a fixed SO(3)-type budget) the Step-39 SE(3)-block objective **grows $\times247$** — it
literally cannot represent the split — while bi-block stays flat ($\times1.00$). Anti-vacuity holds:
bi-block **spikes $\times100$** on a spatially-anisotropic $(\mathbf{std},1o)$ block
($\mathrm{cov}\not\propto\mathbf I_3$, outside Prop. 1′). And the deterministic spectral gauge lands exactly
on the ladder — se3-type law $\to304$ (clusters $[24,8]$), bi-type law $\to184$ (clusters $[18,6,6,2]$) —
stable for every clustering `gap_factor` in $\{1.5,2,3,4\}$ inside the separating window $(1,9)$.

**[B / A′] Exact equivariance, init and post-training.** The scene encoder is per-object SE(3)-equivariant,
$S_O$-permutation-equivariant, and translation-invariant to the float floor at init (scalar-inv
$1.8\times10^{-7}$, vector-equiv $4.3\times10^{-6}$, perm $0$, trans-inv $5.7\times10^{-6}$), and faithful
LeJEPA training does **not** damage it (post-train $1.2\times10^{-7}$ / $1.7\times10^{-6}$ / perm $0$). The
MLP control is perm-equivariant by construction but has no rotation prior ($0.32/5.50$ after training).

**[C] Prop. 1′ on the learned latent + negative control.** On the $S_O\times SO(3)$-invariant (Haar $+$
permute) law the trained equivariant latent is bi-block-isotropic: the six cross-block couplings collapse
(cross $=0.030$) and each $1o$ block is $3\times3$-isotropic (iso_rel $1.06$); the MLP fails (cross $0.80$).
The **negative control** is the sharp one: the *same* equivariant encoder on a **fixed-slot** law (still
rotation-invariant, but $S_O$-*broken*) fails decoupling (cross $0.67$) — so [C] *can* fail, and fails
exactly when the $S_O$ premise is removed. Block-isotropy is a consequence of the product symmetry, not a
metric that passes regardless.

**[E1] 举一反三 across *both* groups.** A type-respecting relational probe fitted on one seen slice transfers
flat across all of SO(3) **and** all of $S_O$: rot-OOD/seen $\times1.01$, perm-OOD/seen $\times0.99$. The MLP
degrades $\times789$ under rotation and $\times2079$ under relabeling. The equivariance-flatness theorem (§4)
now holds on the *product* group — the relational content is genuinely permutation- and rotation-robust.

**[D / E2] The honest boundary — and a sharper lesson.** As in §7, the *learned* per-block scales are
**underdetermined in pure SSL** (bi-block-SIGReg is scale-detached, the budget pins only the total, the
pull-to-mean is block-symmetric), so the gauge claim is gated *deterministically* in [A], with [D] reported
as an un-gated diagnostic. [E2] then asks whether a **relational task** — natural to a compositional scene —
can *realise* the $(\mathbf{std},1o)$ split on the learned net. The lesson is worth keeping: a task scored by
the relMSE of a **free linear fit** is *scale-invariant* in the latent (the fitted weight absorbs any block
rescaling), so it exerts **zero** scale pressure and cannot realise a scale-based refinement. The principled
fix is a **parameter-free, scale-sensitive** equivariant readout (no free multiplicative weight). With it the
relational task drives the $(\mathbf{std},1o)$ block from collapsed to $\mathrm{rel\text{-}scale}\;0.63$ and
pulls the residual gauge $288\to240$, *toward* the $184$ rung [A] proved reachable — honestly, it moves in
the right direction at this 1-GPU scale, it does not snap to $184$. The design principle — *the task that
realises a scale-based gauge reduction must itself be scale-sensitive on the target irrep* — is itself a
transferable finding.

**Controls & falsifiability.** Seeds fixed (full run reproducible); smoke vs. full sizes; a dedicated
covariance sample ($N=6144$) for [C]/[D]; equivariance asserted init $+$ post-training. The full run gates
**nine** deterministic/structural claims (the compositional separation, anti-vacuity, the gauge ladder and
its robustness sweep, exact equivariance, Prop. 1′ and its negative control, dual-group 举一反三), mirrored by
**seven** mechanism guards in `tests/test_step40_compositional_sigreg.py`. [E2] is explicitly an **un-gated
diagnostic**, not a pass/fail gate — per the standing rule, a run that fails to separate reports
`INCONCLUSIVE` rather than relaxing a threshold.

---

## 9. Honest scope, risks, confidence

- **The user's own worry is correct and worth stating in the application:** the *plumbing* (SIGReg on
  an equivariant net) is easy and AMI could do it in an afternoon. The contribution is **not** the
  plumbing — it is the **symmetry-structured identifiability theory** (C1's block-isotropy + gauge
  refinement, C3's weakening of the planning hypothesis), which is *absent* from 2605.26379 and is a
  representation-theory result. Lead with the theorem, not the code.
- **Novelty risk.** They may already be moving this way internally; symmetry is the obvious next axis.
  Mitigant: the specific refinement (turn $O(n)$-up-to into $\rho(G)$-up-to; block-isotropy as the
  SIGReg target; $G$-invariant-cost planning) is concrete and provable now, and we have two experiments
  (Steps 38, 32) already on the board. Speed matters — this is a 6-day-old paper.
- **Degenerate cases — now demonstrated, not just feared.** Equal per-irrep scales collapse the gauge
  refinement back to $O(n)$ (Step 39 [A]: gauge $231$); the clean $\rho(G)$-commutant result needs
  *distinct* scales / multiplicity-freeness. Crucially, Step 39 showed pure SSL does **not** by itself
  produce distinct scales (the split is underdetermined), so the sharp gauge claim is a statement about
  the objective's *target class* (proved + shown deterministically), and *realising* it on a trained
  encoder needs a per-irrep task signal — stated plainly as the honest boundary of C1. **Direction 3
  (Step 40, §8) extends this to the product group $S_O\times SO(3)$:** the compositional rung
  $304\to184$ is reachable as a target-class statement (deterministic [A]), and a **scale-sensitive**
  relational task partially realises it on the learned net (gauge $288\to240$, toward $184$) where a
  *scale-invariant* free-fit task — zero scale pressure — provably cannot. **Direction 2 (Step 41, §4)
  closes the loop the other way:** instead of a hand-built task, a $G$-equivariant *world* (an OU
  transition commuting with $\rho$) supplies the per-irrep signal for free — distinct dynamics $r_i$ at
  equal stationary scale make the *dynamical* gauge $231\to159$ where the *static* covariance is stuck at
  $O(22)$, and the learned equivariant predictor realises that rung and transports it across the orbit
  ([E], $\times1.02$). The scale-sensitive signal §8 installs by hand is, in a world model, just the
  dynamics.
- **Honest confidences:** Prop. 1 0.95 (proof verified + empirically at the noise floor); Prop. 1′
  (product-group block-isotropy) 0.9 (same Schur argument; [C] at the floor + a passing negative control);
  block-SIGReg-as-target 0.8; gauge refinement *as a target-class statement* 0.85, *as something SSL
  reaches unaided* 0.35 (Step 39 negative finding); the compositional rung $304\to184$ *as a target-class
  statement* 0.85, *as something a scale-sensitive task realises on the learned net* 0.4 (Step 40 [E2]:
  moves $288\to240$, not to $184$); **C2 (Prop. 2, equivariant dynamics) 0.8** — upgraded from a 0.65
  sketch to a theorem $+$ falsifiable experiment (Step 41): the dynamical gauge ladder $231\to159$ is
  deterministic (2a/2b), orbit-transport flatness is certified to $10^{-6}$ (2c), and the learned net
  realises the $159$ rung — *realised-on-a-learned-net* 0.7 (the MLP reaches it in-distribution too;
  equivariance is what makes it transport off-orbit, [E]); C3 0.75; Step 32↔Hermite 0.4; "this becomes a
  paper AMI cares about" 0.6.

## 10. Why this strengthens an AMI application

It demonstrates, on their *newest* theory paper, the exact profile a world-model lab is short on: a
mathematician who (1) reads the identifiability theory closely enough to find that $O(n)$ is doing too
much work, (2) **proves** the symmetry-structured refinement (Schur), (3) already has the experiments
(Step 38 = $G$-invariant-cost planning; Step 32 = constructive degree spectrum) that instantiate the
refined theorems, and (4) frames it as "here is the next theorem in your program," not "here is a model
I trained." The narrative writes itself: *LeJEPA recovers the world up to a rotation; equivariance
recovers it up to the world's symmetry — and that is what a world model is supposed to do.*

---

### Sources
- LeJEPA — Balestriero & LeCun, arXiv:2511.08544.
- When Does LeJEPA Learn a World Model? — Klindt, LeCun & Balestriero, arXiv:2605.26379 (2026-05-25).
- This project: `equivariance_generalization_core.md` (flatness theorem, §4), `geometric_payoff.md`
  (Step 32 degree ladder §…, Step 38 latent-goal reaching §30).
