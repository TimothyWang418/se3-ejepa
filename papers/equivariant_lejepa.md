# Equivariant LeJEPA: symmetry-structured identifiability for latent world models

> **Status:** research note / Direction-1 plan (2026-05-31). One result is **proved** here (Prop. 1,
> block-isotropy). Two are stated as **propositions-to-finish** (C2, C3) with proof sketches and an
> existing experiment that already instantiates each. The empirical validation (block-SIGReg on our
> VN/e3nn latent) is scoped in §7 but **not yet run**.
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

## 4. C2 — Equivariance transports their world-condition across orbits

Their guarantee requires the world to lie in the stationary additive-noise (OU) class. Our **flatness
theorem** (core paper §4) says: for a $G$-equivariant encoder *and* $G$-equivariant predictor, the
one-step prediction error is **exactly orbit-constant**, $\mathcal E(\rho(g)\!\cdot\! x)=\mathcal E(x)\
\forall g\in G$, to $\sim\!10^{-6}$ through a real training run.

**Proposition 2 (orbit transport, sketch).** Suppose the true transition kernel commutes with $G$,
$T(\rho(g)z'\mid\rho(g)z)=T(z'\mid z)$ (the world dynamics are $G$-equivariant — the physical case).
Then stationarity and the additive-noise form of §3.1 need only be **verified on a fundamental domain**
$\mathcal F$ (one representative per $G$-orbit); equivariance transports them to all of $\mathbb R^n$.
Consequently the LeJEPA identifiability hypotheses, *checked on a slice*, hold on the whole space, and
the flatness theorem is the empirical certificate that the transport is exact.

*Why this is honest, not magic:* equivariance does **not** make an arbitrary world satisfy the OU
condition — it reduces the *verification* of an already-$G$-symmetric world from $\mathbb R^n$ to
$\mathcal F$, and certifies (via flatness) that predictability is genuinely orbit-invariant rather than
approximately so. Confidence 0.65 (a constructive corollary + a clean statement of when their theory's
hypotheses are cheap to certify).

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
chosen value needs a task signal on each irrep — the natural Direction-3 follow-up.

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

## 8. Honest scope, risks, confidence

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
  encoder needs a per-irrep task signal — stated plainly as the honest boundary of C1.
- **Honest confidences:** Prop. 1 0.95 (proof verified + empirically at the noise floor); block-SIGReg-as-
  target 0.8; gauge refinement *as a target-class statement* 0.85, *as something SSL reaches unaided* 0.35
  (Step 39 negative finding); C2 0.65; C3 0.75; Step 32↔Hermite 0.4; "this becomes a paper AMI cares
  about" 0.6.

## 9. Why this strengthens an AMI application

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
