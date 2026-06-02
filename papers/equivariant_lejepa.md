# Equivariant LeJEPA: symmetry-structured identifiability for latent world models

**Abstract.** LeCun, Balestriero & Klindt now have a *theory* of when a JEPA recovers
the world's latent variables: LeJEPA's embeddings are **linearly identifiable up to a global rotation
$Q\in O(n)$**, and that rotation is treated as an unavoidable *nuisance* "inherent to the isotropic
Gaussian." Their latent-planning guarantee (Thm 5.4) then has to *assume* the cost is invariant under
the **entire** $O(n)$. That assumption is physically far too strong, and the $O(n)$ indeterminacy is
exactly the slot a **world symmetry group** $G\hookrightarrow O(n)$ lives in. An *equivariant* JEPA
replaces the unstructured $O(n)$ nuisance with a known orthogonal representation $\rho(G)$: it (C1)
changes the optimal SIGReg target from full isotropy to **block-isotropy** (proved below via Schur),
(C2) makes their stationarity condition transportable across group orbits, and (C3) weakens the
planning theorem's hypothesis from "$O(n)$-invariant cost" to the realistic "**$G$-invariant cost**"
— a regime our decoder-free latent-goal–reaching experiment already verifies (§5). The differentiator
is not the plumbing (SIGReg on an equivariant net — anyone can do that) but the **symmetry-structured
identifiability theory**, which is absent from their paper and is precisely a representation-theory
contribution.

**Contributions and status.** All three contributions are **proved as target-class statements** —
claims about the optimal embedding / dynamics / planner the objective *defines* — with seeded,
falsifiable experiments. Their **realisation on a *trained* encoder is partial**, and each section
carries its own honest confidence; the headline gap is that the gauge refinement is a theorem about the
target class, while pure SSL realises only *part* of it on the learned net (§7 [D], §8 [E2]; conf.
$\approx0.4$, the main open empirical claim). C3 was upgraded from a proof sketch to two full
propositions $+$ an init/post-training guard (§5).

- **C1** — block-isotropy is the equivariant SIGReg target (Prop. 1): **proved as a target-class
  statement**, instantiated on a mixed-type SO(3) latent (§7) and extended to the product group
  $S_O\times SO(3)$ (Prop. 1$'$, §8). The *gauge-refinement payoff* is realised only **partially** on
  the trained net (§7 [D], §8 [E2]).
- **C2** — equivariant latent dynamics (Prop. 2): **proved**, instantiated by an equivariant OU world
  model whose distinct per-irrep dynamics resolve, *for free*, the gauge pure SSL leaves underdetermined (§4).
- **C3** — planning under a $G$-invariant (not $O(n)$-invariant) cost (Prop. 3 $+$ Prop. 3$'$): **proved**
  — the dynamic-programming optimum *and* the realised iso-CEM estimator are both $G$-equivariant —
  instantiated by the decoder-free latent-goal–reaching experiment and an init/post-training
  planner-equivariance guard (§5).

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
  $\hat V^*(h(z_0))=V^*(z_0)$ and $\hat a^*_{1:t}=a^*_{1:t}$.
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
   hypothesis of Thm 5.4 is unrealistically strong in practice. Real costs are invariant under the *world's*
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
- **What carries block-SIGReg's correctness — the exogenous partition, not within-block isotropy.**
  A SIGReg-style test is a battery of *one-dimensional* projection Gaussianity checks, and a 1-D
  projection is **blind to within-block manifold structure**: by the Diaconis–Freedman projection
  phenomenon, almost every low-dimensional projection of a high-dimensional law looks near-Gaussian even
  when the joint law concentrates on a curved low-dimensional set (concurrent UR-JEPA (Le, 2026) uses
  exactly this to argue SIGReg's 1-D test cannot *see* the manifold structure it leaves intact). The
  same blindness applies *inside* each block: block-SIGReg's 1-D tests cannot certify that the law is
  genuinely isotropic within a block. Its correctness therefore rests **not** on any claim that
  within-block isotropy is itself optimal, but on the **group-prescribed, exogenous block partition by
  isotypic components** — the blocks come from $\rho$'s representation theory (Prop. 1), are fixed before
  any data is seen, and are what make block-isotropy the right *target*; the within-block normality test
  only standardises each block's marginal, it does not justify the partition. This is the precise sense
  in which our anisotropy is prescriptive rather than discovered.
  - **The prescriptive signature is directly measurable (Step 54, `experiments/step54_latent_spectrum_v2.py`).**
    On a genuinely rotation-symmetric distribution (the $\mathrm{SO}(2)$ central-force data of Step 50, sampled
    isotropically) an equivariant encoder's latent law is exactly $\rho$-invariant, so its covariance must
    **commute with the group representation**, $\rho(R)\,\Sigma\,\rho(R)^\top=\Sigma$ (Schur). We verify this:
    the equivariant latent's $\rho$-invariance residual $\lVert\rho\Sigma\rho^\top-\Sigma\rVert_F/\lVert\Sigma\rVert_F$
    is $\approx3\times10^{-4}$ — **$\sim\!3000\times$ below** an identically-trained non-equivariant MLP's
    ($1.04$) — and each $\ell{=}1$ block's $2\times2$ self-covariance is isotropic (anisotropy $0.005$). The
    covariance is *pinned to $\rho$* by the architecture, a group-given second-moment signature the MLP has no
    reason to exhibit. (An earlier attempt, Step 48, was honestly inconclusive — $\times2$ only — because the
    interacting-teacher data there was **not** rotation-symmetric; the fix was a symmetric distribution, not a
    new metric.)
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
     structure (which axes carry which irrep, and the within-irrep frames) is pinned. **Caveat — the
     boxed finite case is the multiplicity-free idealisation, not our experiment.** The §7/§8 latent is
     $m_0=4$ scalars and $m_1=6$ vectors, so the realised commutant is the *continuous* group
     $O(4)\times O(6)$ (dimension $6+15=21$), **not** a finite sign group; the within-multiplicity frames
     are not pinned there. "Up to $\prod_i\{\pm1\}$" is the clean special case one gets only when every
     irrep appears at most once; with multiplicities $>1$ the prize degrades from *finite identifiability*
     to *block-diagonal identifiability* — the symmetry labels are still recovered, the within-block frame
     is not.
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

## 4. C2 — Equivariant latent dynamics: the world model resolves the gauge SSL leaves free

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
$\mathbb R^n$; the last equality is the **per-irrep analogue** of KLB's forward bound — *not* literally
equal to it. Our $\operatorname{tr}Q=\sum_i d_im_i\sigma^2(1-r_i^2)$ is the *optimal-predictor
innovation* $\mathbb E\lVert z'-Az\rVert^2$, whereas KLB's scalar $2(1-r)n$ is the *positive-pair
distance* $\mathbb E\lVert z'-z\rVert^2$ of a whitened ($\sigma^2{=}1$, single $r$) embedding; the two
differ by the drift term $\mathbb E\lVert(A-I)z\rVert^2$ (so $\mathbb E\lVert z'-z\rVert^2=
\operatorname{tr}Q+\mathbb E\lVert(A-I)z\rVert^2$, and they coincide only as $r\to1$). The refinement
that *does* carry over is structural: each irrep contributes its **own** $1-r_i^2$, splitting KLB's
single scalar across the isotypic blocks.

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
underdetermined (§7) — the predictor **is** the "scale-sensitive task" §8 [E2] had to install by
hand, here handed over by the world itself.

### 4.1 Minimal experiment — built and run (laptop CPU, seeded)

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

**[B/A$'$] Predictor equivariance, init and post-training.** A mixed-type equivariant predictor (a
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
distinct-scale condition) and **0.7** realised on a learned net (§4.1 [D] reaches $159$, but so does
the MLP in-distribution; equivariance is what makes the rung *transport*, [E]); the orbit-transport
flatness (c) **0.85** (a clean identity, certified to $10^{-6}$). C2 overall **0.8** — upgraded from the
0.65 sketch now that it is a theorem with a falsifiable experiment.

---

## 5. C3 — Planning under $G$-invariant (not $O(n)$-invariant) costs

Thm 5.4 needs the cost invariant under **all** of $O(n)$ — a hypothesis unrealistically strong in practice,
since real planning costs are invariant under the *world's* symmetry $G$, not an arbitrary latent
rotation. Under an equivariant encoder whose residual identifiability is pinned to $\rho(G)$ (C1,
distinct-scale case), the guarantee goes through under the strictly weaker, physically natural
$G$-invariant hypothesis. We give it in two halves: the **idealised optimum** (Prop. 3 — the
dynamic-programming claim 5.4 makes for $O(n)$, now for $\rho(G)\subset O(n)$), and — the part 5.4
never addresses — the **realised finite-horizon CEM estimator** we actually deploy (Prop. 3$'$), whose
equivariance is what makes the closed loop [C] a theorem rather than an observation.

**Setup.** Latent dynamics $z_{t+1}=f(z_t,a_t)$ that are **$G$-equivariant**,
$f(\rho(g)z,\,g\cdot a)=\rho(g)f(z,a)$; a stage cost $\ell$ and terminal cost $c_T$ that are
**$G$-invariant**, $\ell(\rho(g)z,\,g\cdot a)=\ell(z,a)$ and $c_T(\rho(g)z,\rho(g)z_g)=c_T(z,z_g)$
($g\cdot a$ is the induced action on actions — for a velocity action, $g\cdot a=Ra$); and an action set
$\mathcal A$ that is **$g$-stable**, $g\cdot\mathcal A=\mathcal A$ for all $g\in G$ (the unit ball
qualifies, as $\lVert g\cdot a\rVert=\lVert a\rVert$). Write the finite-horizon value with goal $z_g$ as a
parameter, $V^*_t(z;z_g)=\min_{a\in\mathcal A}\{\ell(z,a)+V^*_{t-1}(f(z,a);z_g)\}$, $V^*_0(z;z_g)=c_T(z,z_g)$.

**Proposition 3 (exact $G$-invariant latent planning).** *Under the Setup:* **(a)** *the optimal value is
$G$-invariant, $V^*_t(\rho(g)z;\rho(g)z_g)=V^*_t(z;z_g)$ for all $g\in G$, $t\le T$, and the optimal
control transforms covariantly — if $a^*_{1:T}$ is optimal at $(z,z_g)$ then $g\cdot a^*_{1:T}$ is optimal
at $(\rho(g)z,\rho(g)z_g)$;* **(b)** *consequently, if the encoder is recovered only up to a fixed
world-symmetry gauge $h(z)=\rho(g_0)z$, $g_0\in G$ (not an arbitrary $Q\in O(n)$), latent planning is*
**exact**: *$\hat V^*(h(z_0))=V^*(z_0)$ and $\hat a^*_{1:T}=g_0\cdot a^*_{1:T}$.*

*Proof.* (a) Backward induction on $t$. Base $t=0$: $V^*_0(\rho(g)z;\rho(g)z_g)=c_T(\rho(g)z,\rho(g)z_g)
=c_T(z,z_g)$ by $G$-invariance of $c_T$. Step: assume the claim at $t-1$. Substituting $a=g\cdot a'$ —
a bijection of $\mathcal A$ onto itself by $g$-stability — and using $\ell(\rho(g)z,g\cdot a')=\ell(z,a')$
and $f(\rho(g)z,g\cdot a')=\rho(g)f(z,a')$,
$$ V^*_t(\rho(g)z;\rho(g)z_g)=\min_{a'\in\mathcal A}\Big\{\ell(z,a')+V^*_{t-1}\big(\rho(g)f(z,a');\rho(g)z_g\big)\Big\}
=\min_{a'\in\mathcal A}\Big\{\ell(z,a')+V^*_{t-1}\big(f(z,a');z_g\big)\Big\}=V^*_t(z;z_g), $$
the middle equality by the inductive hypothesis. The minimiser at $\rho(g)z$ is thus $g\cdot a'^\star$ with
$a'^\star$ the minimiser at $z$; composing this per-step covariance along the equivariant rollout gives the
sequence claim. (b) Apply (a) with $g=g_0$: planning in $h$-coordinates is the $g_0$-image of the true
problem, so the optimum value coincides and the optimal actions are its $g_0$-image. Only invariance under
$\rho(G)$ — never under a general $O(n)$ element — is used, which is exactly what C1's $\rho(G)$
gauge-pinning supplies. $\qquad\blacksquare$

**The realised estimator is equivariant too — not just the optimum.** Prop. 3 is the dynamic-programming
statement; the planner we deploy is a finite-sample iso-CEM-MPC (`experiments/step18.latent_cem_plan_iso`),
a *stochastic* map. 5.4 stops at the optimum, but closed-loop [C] needs the **realised** plan to commute
with $G$. It does, sample-path-wise.

**Proposition 3$'$ (the iso-CEM-MPC estimator is $G$-equivariant).** *Let the CEM planner use* **(P1)** *a
$\rho$-invariant cost — terminal $\lVert\hat z_H-z_g\rVert^2$ under orthogonal $\rho$ plus the closed-form
centroid drift cost $w_t\lVert\bar x_0+c_t\sum_h a_h-\bar x_g\rVert^2$, both invariant under a joint
$(R,t)$ (the $t$'s cancel in the centroid term);* **(P2)** *a $g$-stable constraint — the unit-ball clamp,
$\mathrm{clip}_{\rm ball}(g\cdot a)=g\cdot\mathrm{clip}_{\rm ball}(a)$;* **(P3)** *isotropic Gaussian
sampling whose noise is pre-rotated by $R$, $\varepsilon\mapsto R\varepsilon$, under a shared random seed;
and* **(P4)** *elite selection and an isotropic per-step $\sigma$-refit that depend on the candidates only
through the $\rho$-invariant cost and an isotropy-pooled variance. Then for every $g=(R,t)\in\mathrm{SE}(3)$,
$\mathrm{plan}(\rho(g)z_0,\rho(g)z_g)=g\cdot\mathrm{plan}(z_0,z_g)$ exactly (to the float floor), at any
weights* ($g\cdot$ *rotates each action by $R$; the translation part acts trivially on velocity actions*).

*Proof.* Induction on the CEM iteration under the seed coupling of (P3), with invariant
$(\mathrm{mean}_k,\sigma_k)\mapsto(R\cdot\mathrm{mean}_k,\sigma_k)$ — the mean is the $R$-image, the
isotropic $\sigma$ is *identical*. Base: $\mathrm{mean}_0=\mathbf 0=R\mathbf 0$, $\sigma_0=\sigma_0\mathbf 1$.
Step: the shared generator draws $\varepsilon$ for the base run and the rotated run multiplies it by $R$
(P3); as $\sigma_k$ is isotropic, $\mathrm{cand}^g=\mathrm{clip}_{\rm ball}(R\,\mathrm{mean}_k+\sigma_k\,
R\varepsilon)=R\cdot\mathrm{cand}$ by (P2). The rolled latents are $\rho(g)$-images (encoder-equivariance
gives $z_0^g=\rho(g)z_0$, then the equivariant predictor), so by (P1) each candidate's cost is *identical*
across the two runs; `topk` over identical costs returns the *same indices* (P4), hence
$\mathrm{mean}_{k+1}^g=R\cdot\mathrm{mean}_{k+1}$ and the isotropy-pooled variance is rotation-invariant,
$\sigma_{k+1}^g=\sigma_{k+1}$. The returned mean is the $R$-image. Swap any one hypothesis back — a *box*
clamp (only $B_n$-stable, breaking P2) or a *diagonal* $\sigma$-refit (breaking isotropy in P4) — and the
candidate sets cease to be $R$-related at generic $R$; this is exactly the [S] panel's controlled drift. $\qquad\blacksquare$

**This is verified — init *and* post-training, with a non-vacuous control.** Two seeded CPU guards:

- `tests/test_planner_equivariance.py` certifies Prop. 3$'$ directly:
  $\max\lvert R\cdot\mathrm{plan}(x_0,x_g)-\mathrm{plan}(Rx_0+t,Rx_g+t)\rvert=1.19\times10^{-7}$ (the CEM
  float floor) for the VN at **both** random init and after a real training run — *identical*, because the
  property is architectural — over pure rotations *and* large translations $t$; the *same* equivariant
  planner on a non-equivariant MLP world misses by $1.35$ ($\sim\!10^{7}\times$ the VN floor), so the test
  is not vacuous. The Kabsch orientation readout the loop reads off is itself $\mathrm{SE}(3)$-invariant
  ($\le1.1\times10^{-3}$ deg).
- `experiments/step38_latent_goal_reaching.py` (+ a 6-gate test) instantiates the payoff: a decoder-free
  goal cost (an $L_2$ latent cost *and* a Procrustes geodesic-angle signal, both $\rho$-invariant to
  $\sim\!10^{-13}$), planned by the equivariant CEM directly in the latent, **reaches identically across the
  $\mathrm{SE}(3)$ orbit** — OOD/seen fraction-of-gap-closed ratio $1.000$ (per-task seen-vs-OOD spread
  $7.4\times10^{-7}$ deg at init) — versus a non-equivariant MLP planner at $\times1.745$. This is 5.4's
  conclusion ($\hat V^*=V^*$, matched optimal actions) holding where 5.4 does **not** apply (a $G$-invariant,
  *not* $O(n)$-invariant, cost), and — beyond 5.4 — for the *realised* estimator, certified to survive
  optimisation.

**Confidence.** Prop. 3(a) **0.9** (elementary backward induction $+$ orthogonality of $\rho$, the standing
of Prop. 1/2's algebra); Prop. 3$'$ **0.9** (a clean sample-path coupling, verified to the float floor at
init *and* post-training with a non-vacuous MLP control); the *end-to-end deployment* claim **0.75** (sound
given C1's gauge-pinning — the one live dependency is that the encoder's residual gauge really is $\rho(G)$,
i.e. C1's distinct-scale hypothesis, the exact mirror of C2(b)'s distinct-$r_i$ caveat). **C3 overall
0.85** — upgraded from the 0.65/0.75 sketch now that it is two theorems with full proofs and a falsifiable
experiment (Step 38 $+$ the init/post-training planner guard), the same upgrade C2 received.

---

## 6. A bridge already built: the degree ladder ↔ their Hermite spectral penalty

Thm 5.1's forward direction is a **Hermite-degree** spectral decomposition: each degree of nonlinearity
strictly reduces positive-pair correlation, so the linear map wins. We built a predictor
with a *tunable* maximum polynomial degree, $d_{\max}(L)=2^L$ (the degree-ladder predictor), and showed a
degree-3 interaction target is first representable at rung $L=2$. So our degree ladder is a
**constructive, equivariant** realisation of their spectral-degree analysis: their scalar-Hermite basis
is the $G$-trivial case, and the equivariant generalisation replaces Hermite polynomials by the
Clebsch–Gordan / spherical-harmonic decomposition of tensor powers of $\rho$. Conjecture: "alignment
penalises Hermite degree" becomes "alignment penalises the higher-$\ell$ irreps in $\rho^{\otimes k}$,"
and the degree ladder measures the penalty rung by rung. Confidence 0.4 (suggestive; a genuine
opportunity, not yet a result).

---

## 7. Minimal experiment for C1 — built and run (laptop CPU, seeded)

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
probe degrades $\times8455$ off the wedge. This is the equivariance-flatness theorem (core paper §4) made concrete on
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

## 8. Direction 3 — compositional bi-block-SIGReg on a product symmetry $S_O\times SO(3)$

§7 proved block-SIGReg on a **single** object's SE(3)-type structure. The open question it leaves: does a
*product* symmetry buy a strictly finer identifiability rung that single-object block-SIGReg cannot reach? A
scene of several interchangeable, individually-rotating objects is the natural test — its symmetry group is
$S_O\times SO(3)$ (relabel the objects $\times$ rotate them as one rigid frame), and that product is exactly
what an object-centric world model must respect.

**Prop. 1$'$ (product-group block-isotropy).** Take a scene of $O$ distinguishable objects, each carrying
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
differently at a fixed SO(3)-type budget) the §7 SE(3)-block objective **grows $\times247$** — it
literally cannot represent the split — while bi-block stays flat ($\times1.00$). Anti-vacuity holds:
bi-block **spikes $\times100$** on a spatially-anisotropic $(\mathbf{std},1o)$ block
($\mathrm{cov}\not\propto\mathbf I_3$, outside Prop. 1$'$). And the deterministic spectral gauge lands exactly
on the ladder — se3-type law $\to304$ (clusters $[24,8]$), bi-type law $\to184$ (clusters $[18,6,6,2]$) —
stable for every clustering `gap_factor` in $\{1.5,2,3,4\}$ inside the separating window $(1,9)$.

**[B / A$'$] Exact equivariance, init and post-training.** The scene encoder is per-object SE(3)-equivariant,
$S_O$-permutation-equivariant, and translation-invariant to the float floor at init (scalar-inv
$1.8\times10^{-7}$, vector-equiv $4.3\times10^{-6}$, perm $0$, trans-inv $5.7\times10^{-6}$), and faithful
LeJEPA training does **not** damage it (post-train $1.2\times10^{-7}$ / $1.7\times10^{-6}$ / perm $0$). The
MLP control is perm-equivariant by construction but has no rotation prior ($0.32/5.50$ after training).

**[C] Prop. 1$'$ on the learned latent + negative control.** On the $S_O\times SO(3)$-invariant (Haar $+$
permute) law the trained equivariant latent is bi-block-isotropic: the six cross-block couplings collapse
(cross $=0.030$) and each $1o$ block is $3\times3$-isotropic (iso_rel $1.06$); the MLP fails (cross $0.80$).
The **negative control** is the sharp one: the *same* equivariant encoder on a **fixed-slot** law (still
rotation-invariant, but $S_O$-*broken*) fails decoupling (cross $0.67$) — so [C] *can* fail, and fails
exactly when the $S_O$ premise is removed. Block-isotropy is a consequence of the product symmetry, not a
metric that passes regardless.

**[E1] 举一反三 across *both* groups.** A type-respecting relational probe fitted on one seen slice transfers
flat across all of SO(3) **and** all of $S_O$: rot-OOD/seen $\times1.01$, perm-OOD/seen $\times0.99$. The MLP
degrades $\times789$ under rotation and $\times2079$ under relabeling. The equivariance-flatness theorem (core paper §4)
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
pulls the residual gauge $288\to240$, *toward* the $184$ rung [A] proved reachable — quantitatively, that
closes $288-240=48$ of the $288-184=104$ reachable gauge dimensions (**$\approx46\%$**) on a single 1-GPU
run: honestly the right direction, not a snap to $184$. The design principle — *the task that
realises a scale-based gauge reduction must itself be scale-sensitive on the target irrep* — is itself a
transferable finding.

**Controls & falsifiability.** Seeds fixed (full run reproducible); smoke vs. full sizes; a dedicated
covariance sample ($N=6144$) for [C]/[D]; equivariance asserted init $+$ post-training. The full run gates
**nine** deterministic/structural claims (the compositional separation, anti-vacuity, the gauge ladder and
its robustness sweep, exact equivariance, Prop. 1$'$ and its negative control, dual-group 举一反三), mirrored by
**seven** mechanism guards in `tests/test_step40_compositional_sigreg.py`. [E2] is explicitly an **un-gated
diagnostic**, not a pass/fail gate — per the standing rule, a run that fails to separate reports
`INCONCLUSIVE` rather than relaxing a threshold.

---

## 9. Honest scope, risks, confidence

- **The contribution is the theory, not the plumbing — and the *proven* part is a target-class
  statement.** Implementing SIGReg on an equivariant network is routine. What is new is **not** that
  engineering but the **symmetry-structured identifiability theory** (C1's block-isotropy SIGReg
  *target* $+$ the gauge accounting that reduces $O(n)$ to the commutant $\prod_i O(m_i)$, C3's weakening
  of the planning hypothesis), which is *absent* from arXiv:2605.26379 and is a representation-theory
  result. Be precise about what "proved" covers: the **proven** novelty is the *target-class* statement
  — the optimal embedding the objective defines is block-isotropic, and on it the residual gauge is the
  named commutant. The **realised identifiability gain on a *trained* encoder is partial** (conf. $0.4$;
  §8 [E2] closes $\approx46\%$ of the reachable gauge dims, not all), and that gap — does pure or
  task-shaped SSL actually *reach* the target-class identifiability on a learned net? — is **the main
  open empirical claim** of this note, not a settled result. The theorem, not the code, is what is new;
  the *empirics* of realisation are deliberately reported as unfinished.
- **Novelty risk.** Symmetry is an obvious next axis, so concurrent work is plausible. What is concrete
  here: the specific refinement (turn $O(n)$-up-to into $\rho(G)$-up-to; block-isotropy as the SIGReg
  target; $G$-invariant-cost planning) is provable now, with two experiments already instantiating it —
  the $G$-invariant-cost planner (§5) and the degree ladder (§6). The identifiability paper it builds on
  is recent (arXiv:2605.26379, 2026-05-25). A concurrent data-driven alternative, UR-JEPA (Le, 2026),
  shapes the latent toward a low-dimensional manifold of *intrinsic dimension* $n$ — but that $n$ is a
  **load-bearing hyperparameter** (it reports a catastrophic collapse at $n{=}4$). Our anisotropy carries
  no such knob: the block structure is fixed by the representation $\rho$ (the irrep dimensions $d_i$ and
  multiplicities $m_i$ are read off $G$, not tuned), so **there is no $n$ to choose** — the
  dimensionality of each block is dictated by representation theory rather than selected against a
  validation set.
- **Degenerate cases — now demonstrated, not just feared.** Equal per-irrep scales collapse the gauge
  refinement back to $O(n)$ (§7 [A]: gauge $231$); the clean $\rho(G)$-commutant result needs
  *distinct* scales / multiplicity-freeness. Crucially, §7 showed pure SSL does **not** by itself
  produce distinct scales (the split is underdetermined), so the sharp gauge claim is a statement about
  the objective's *target class* (proved + shown deterministically), and *realising* it on a trained
  encoder needs a per-irrep task signal — stated plainly as the honest boundary of C1. **Direction 3
  (§8) extends this to the product group $S_O\times SO(3)$:** the compositional rung
  $304\to184$ is reachable as a target-class statement (deterministic [A]), and a **scale-sensitive**
  relational task partially realises it on the learned net (gauge $288\to240$, toward $184$) where a
  *scale-invariant* free-fit task — zero scale pressure — provably cannot. **Direction 2 (§4)
  closes the loop the other way:** instead of a hand-built task, a $G$-equivariant *world* (an OU
  transition commuting with $\rho$) supplies the per-irrep signal for free — distinct dynamics $r_i$ at
  equal stationary scale make the *dynamical* gauge $231\to159$ where the *static* covariance is stuck at
  $O(22)$, and the learned equivariant predictor realises that rung and transports it across the orbit
  ([E], $\times1.02$). The scale-sensitive signal §8 installs by hand is, in a world model, just the
  dynamics.
- **Honest confidences:** Prop. 1 0.95 (proof verified + empirically at the noise floor); Prop. 1$'$
  (product-group block-isotropy) 0.9 (same Schur argument; [C] at the floor + a passing negative control);
  block-SIGReg-as-target 0.8; gauge refinement *as a target-class statement* 0.85, *as something SSL
  reaches unaided* 0.35 (§7 negative finding); the compositional rung $304\to184$ *as a target-class
  statement* 0.85, *as something a scale-sensitive task realises on the learned net* 0.4 (§8 [E2]:
  moves $288\to240$, not to $184$); **C2 (Prop. 2, equivariant dynamics) 0.8** — upgraded from a 0.65
  sketch to a theorem $+$ falsifiable experiment (§4.1): the dynamical gauge ladder $231\to159$ is
  deterministic (2a/2b), orbit-transport flatness is certified to $10^{-6}$ (2c), and the learned net
  realises the $159$ rung — *realised-on-a-learned-net* 0.7 (the MLP reaches it in-distribution too;
  equivariance is what makes it transport off-orbit, [E]); **C3 (Prop. 3 $+$ 3$'$, $G$-invariant-cost planning) 0.85** — upgraded from a 0.65/0.75
  sketch to two full-proof theorems $+$ a falsifiable experiment (§5): the DP optimum is $G$-invariant (3a)
  and the *realised* iso-CEM estimator commutes with $\mathrm{SE}(3)$ to the float floor at init *and*
  post-training (3$'$), end-to-end deployment held at 0.75 (lone dependency: C1's $\rho(G)$ gauge-pinning);
  the degree-ladder↔Hermite bridge (§6) 0.4; "this becomes a
  publishable contribution" 0.6.

## 10. Discussion: what is new, and where it sits in the program

This work builds on the identifiability program of arXiv:2605.26379 and advances it on one axis. It
(1) locates where the $O(n)$ indeterminacy is doing too much work, (2) **proves** the symmetry-structured
refinement (Schur), (3) instantiates the refined theorems with experiments already on the board
($G$-invariant-cost planning, §5; a constructive degree spectrum, §6), and (4) is framed as the *next
theorem* in that program rather than as a trained model. The one-line summary: *LeJEPA recovers the
world up to a rotation; equivariance recovers it up to the world's symmetry — which is what a world
model is supposed to do.*

---

### Sources
- LeJEPA — Balestriero & LeCun, arXiv:2511.08544.
- When Does LeJEPA Learn a World Model? — Klindt, LeCun & Balestriero, arXiv:2605.26379 (2026-05-25).
- This project: `equivariance_generalization_core.md` (flatness theorem, §4), `geometric_payoff.md`
  (the degree ladder, §24; latent-goal reaching, §30).
