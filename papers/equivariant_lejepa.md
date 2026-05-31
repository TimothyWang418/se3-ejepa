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
That is, $\Sigma$ is **block-isotropic**: a scalar multiple of the identity *inside* each irreducible
copy, with mixing allowed only across the $m_i$ multiplicity slots of the *same* irrep, and **zero**
coupling between inequivalent irreps. (For complex/quaternionic-type $V_i$, $B_i$ is taken over
$\mathbb C$/$\mathbb H$; the "scalar on each irrep copy" conclusion is unchanged.)

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
- **Identifiability refinement (the prize).** A block-isotropic $\Sigma$ with **distinct** scales
  $\sigma_i^2$ exposes the isotypic decomposition in its own spectrum, so the residual gauge of the
  latent law collapses from all of $O(n)$ to the centraliser of $\rho(G)$ compatible with the blocks —
  generically $\rho(G)$ times multiplicity-space mixing $\prod_i O(m_i)$, and exactly $\rho(G)$ in the
  multiplicity-free, distinct-scale case. In words: **equivariance + block-isotropy upgrades
  "recover up to an arbitrary rotation $Q\in O(n)$" to "recover up to the world's actual symmetry
  $\rho(G)$."** That is precisely the "recover the true degrees of freedom *with their symmetry
  structure*" desideratum their abstract opens with.

Confidence: Prop. 1 itself 0.9 (textbook Schur). "Block-SIGReg is the right target" 0.8. The residual-
gauge refinement 0.7 (the distinct-scale / multiplicity-free hypotheses must be stated carefully; equal
scales degenerate back toward a larger symmetry).

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

## 7. Minimal experiment (laptop, CPU/MPS, seeded)

Validate C1 end-to-end on the existing 3D point-cloud VN/e3nn latent:

1. **Add SIGReg** to the equivariant JEPA in two variants: (a) **vanilla isotropic** (all projections
   $\to\mathcal N(0,1)$); (b) **block-SIGReg** (project within each isotypic block; test each toward an
   own-scale isotropic Gaussian). ~50 lines, linear cost, $\lambda\!\approx\!0.05$.
2. **Measure.** (i) Does the latent covariance go block-isotropic (Prop. 1), and does (a) inflate the
   equivariance residual / loss while (b) sits comfortably? (ii) **Cross-orbit linear identifiability:**
   best-orthogonal-$Q$ regression of learned onto true latent, and whether $Q\in\rho(G)$ (block
   structure preserved) versus arbitrary $O(n)$. (iii) Does block-SIGReg preserve/improve across-group
   generalisation vs the no-SIGReg equivariant model and vs a no-symmetry LeJEPA?
3. **Equivariance unit test** (init + post-training) that block-SIGReg keeps the encoder equivariant.
4. **Controls:** seeds logged, git hash logged, Hydra/JSON config saved; compare against the existing
   no-SIGReg checkpoints so the *only* change is the regulariser.

Predicted outcome (falsifiable): block-SIGReg matches or beats vanilla isotropic on probe quality while
**not** degrading equivariance, and the recovered $Q$ stays in $\rho(G)$ — empirically pinning the
gauge to the world symmetry. If vanilla isotropic SIGReg *also* leaves the gauge at $\rho(G)$ with no
penalty, C1's "fights equivariance" claim weakens and we report that honestly.

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
- **Degenerate cases.** Equal per-irrep scales collapse the gauge refinement back toward a larger
  symmetry; the clean $\rho(G)$ result needs distinct scales / multiplicity-freeness, stated carefully.
- **Honest confidences:** Prop. 1 0.9; block-SIGReg-as-target 0.8; gauge refinement 0.7; C2 0.65;
  C3 0.75; Step 32↔Hermite 0.4; "this becomes a paper AMI cares about" 0.6.

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
