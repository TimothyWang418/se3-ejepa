# Proposal — Certified Compositional 举一反三: Structure as an Alternative to Scale

> **Status**: research proposal (north-star for the next paper). Draft 2026-06-02.
> **One-line thesis**: *Structure buys something scale provably cannot — a certificate.* A (possibly **discovered**) symmetry group lets an equivariant latent world model **certify** zero-shot generalization over an **exponentially large compositional** set of situations — multi-step and closed-loop — with a graceful, measured degradation under approximate symmetry. No amount of scaling provides such a certificate.

---

## 1. The breakthrough claim (and why it is impact, not a benchmark number)

Scaling buys **interpolation without guarantees**: more data/params → lower test error on the data manifold, but no statement about a *specific* unseen situation. We claim a *different kind* of result:

> Given $k$ primitive transformations $S=\{g_1,\dots,g_k\}$ whose equivariance the model satisfies (verified to the float floor on the $k$ generators), the model is **certified** — provably, without further training or data — to generalize over the **entire generated monoid** $\langle S\rangle$ (size exponential in composition length), including multi-step rollout and closed-loop task success.

This reframes the Bitter-Lesson debate from a *degree* contest (who needs less data — scale wins eventually) to a *kind* contest (who can issue a **certificate** — only structure can). The impact is a new tool for the field — *when* a symmetry exists (or is discovered), you get certified extrapolation over an exponential set from $k$ checks — not a new leaderboard row. (Model of impact: a position/mechanism paper à la JEPA, demonstrated as a clean proof-of-principle at 1–2 GPU scale, **not** a SOTA-beating benchmark — see §7 honesty.)

---

## 2. The certificate — precise definition and theorem

**Setup.** State space $\mathcal X$, latent $\mathcal Z$, encoder $E:\mathcal X\to\mathcal Z$, action-conditioned predictor $f:\mathcal Z\times\mathcal A\to\mathcal Z$. A group/monoid element $g$ acts on states by $g\cdot x$, on the latent by an **orthogonal** representation $\rho(g)$, on actions by $\sigma(g)$. **Exact equivariance** means
$$
E(g\cdot x)=\rho(g)\,E(x),\qquad f\!\big(\rho(g)z,\ \sigma(g)a\big)=\rho(g)\,f(z,a).
$$
Let $f^{(T)}$ be the $T$-step rollout under an action sequence $\vec a$, and define a trajectory error
$\mathrm{Err}_T(x;\vec a):=\big\lVert f^{(T)}\!\big(E(x),\vec a\big)-E(x^{\text{true}}_T)\big\rVert$ (or the closed-loop task cost $J(x;\text{goal})$ for the behavioral version).

**Lemma (composition closure).** If $E,f$ are equivariant under each generator $g_i\in S$, they are equivariant under every word $w=g_{i_1}\cdots g_{i_m}\in\langle S\rangle$, because $\rho$ is a (monoid) homomorphism: $\rho(g_{i_1}\cdots g_{i_m})=\rho(g_{i_1})\cdots\rho(g_{i_m})$. *Thus $k$ equivariance checks certify an exponential set.*

**Theorem (exact certificate).** Under exact equivariance with orthogonal $\rho$, for **every** $w\in\langle S\rangle$, every horizon $T$, every action sequence:
$$
\mathrm{Err}_T(w\cdot x;\ \sigma(w)\vec a)=\mathrm{Err}_T(x;\vec a),\qquad
J(w\cdot x;\ w\cdot\text{goal})=J(x;\text{goal}).
$$
*Proof sketch.* Equivariance of $f^{(T)}$ follows by induction; orthogonality of $\rho$ kills the transformation inside the norm: $\lVert\rho(w)[\cdot]\rVert=\lVert[\cdot]\rVert$. (This is the multi-step, full-monoid lift of the one-step [B] cancellation.) $\square$

**Theorem (approximate-symmetry degradation — the honest, load-bearing result).** Let the measured residuals be $\epsilon_i=\sup_x\lVert E(g_i x)-\rho(g_i)E(x)\rVert$ (encoder), $\delta=\sup\lVert f(\rho z,\sigma a)-\rho f(z,a)\rVert$ (predictor), and $L=$ Lipschitz constant of $f$ in $z$. Then for a word of length $|w|=m$ and horizon $T$,
$$
\big\lvert \mathrm{Err}_T(w\cdot x)-\mathrm{Err}_T(x)\big\rvert \ \le\ C\,\big(m\,\epsilon_{\max}+T\,\delta\big)\cdot \Phi(L,T),\quad
\Phi(L,T)=\begin{cases}\Theta(1)&L\le 1\ (\text{isometric/contractive})\\ \Theta(L^{T})&L>1\ (\text{expansive})\end{cases}.
$$
With **exact** equivariance ($\epsilon,\delta\approx10^{-6}$ by [A]) the bound is $\approx0$ → the certificate is tight over the exponential set. With **approximate** symmetry the certificate **degrades linearly in composition length and horizon** and is meaningful iff $\epsilon_{\max},\delta$ are small and $L\!\lesssim\!1$. *This bound is the paper's spine: it states exactly when "structure beats scale" holds and when it dissolves.*

---

## 3. Why this is a NEW theorem (precise delta vs [A]/[B]/[C])

| | quantifier | horizon | level | what's certified |
|---|---|---|---|---|
| **[A]** | one $g$ | — | encoder | equivariance residual $\le10^{-6}$ |
| **[B]** | one $g$ | $T{=}1$ | prediction | $\mathrm{relMSE}(g x)=\mathrm{relMSE}(x)$ |
| **[C]** | one $g$ | closed-loop | behavior | $J(g x)=J(x)$ |
| **Certificate (new)** | **all $w\in\langle S\rangle$ (exp.)** | **any $T$** | **prediction + behavior** | $\lvert\mathrm{Err}_T(w x)-\mathrm{Err}_T(x)\rvert\le C(m\epsilon+T\delta)\Phi(L,T)$ |

[A]/[B]/[C] are the $|w|{=}1$ corners. The genuinely new content: **(i) composition closure** ($k$ generators ⟹ exponential certified set), **(ii) multi-step error propagation**, **(iii) the approximate-symmetry degradation bound**. None of these are stated by [B]/[C].

---

## 4. Five building blocks (each maps to machinery you already have)

1. **Definition + theorem** (§2) — *new*. The certificate + degradation bound.
2. **Discovery** — the generating set $S$ is **discovered from interaction** (active inference / free energy), not hand-wired ⇒ the certificate is *earned*. Reuses **Step 33/36** (discover generators from behaviour) + **Step 34/37** (invariant curiosity). This is what defuses "you hand-designed the symmetry."
3. **Generation = certified orbit traversal** — "generating" a novel valid world-state/skill = applying $w\in\langle S\rangle$ in latent; the certificate guarantees validity. Unifies generation ≡ generalization. Reuses **Step 38** (decoder-free latent-goal reaching).
4. **Compositional reach** — the certified set is exponential; demonstrate on **Step 35** (few→many-body) + **Step 19/24** (object-centric + interaction: per-object SE(3) × object permutation generate a huge monoid).
5. **Honest degradation** — measure $\epsilon,\delta,L$ and the bound's shape under **approximate** symmetry (**Step 30**) and **multi-step** (**Step 31**), and locate the certified region on the **Step 22** symmetry×data phase diagram.

> The new paper is mostly **assembly + one new theorem + one new experiment**, not from-scratch — most empirical organs exist. That is precisely why it is feasible at your scale.

---

## 5. The killer experiment (compositional × closed-loop × certified-vs-scaled)

**World.** Object-centric 3D scene with $k$ primitive symmetries whose generated monoid is exponential: per-object $\mathrm{SE}(3)$ + object permutation $S_O$ (so $\langle S\rangle\supseteq \mathrm{SE}(3)^O\rtimes S_O$), reusing the Step 19/24 generator.

**Certified model.** The exact-equivariant JEPA + equivariant planner (Steps 13/18/41). Verify equivariance on the $k$ generators ([A]); then **without further training**, evaluate prediction error and closed-loop task success at compositions $w\cdot x$ for $|w|=1,2,\dots,M$ and horizons $T=1,\dots,H$.

**Two headline plots.**
- **Plot 1 (the certificate).** $\mathrm{Err}_T(w\cdot x)$ vs composition length $|w|$ and horizon $T$ — flat (certified) for the exact model; overlay the *predicted* degradation bound $C(m\epsilon+T\delta)\Phi(L,T)$ and show it tracks the measured curve. *Verifying the theorem empirically is itself a result.*
- **Plot 2 (structure vs scale).** Certified region size (exponential in $|w|$, from $k$ generators) vs the region a **scaled non-equivariant baseline** (MLP/transformer + composition augmentation) actually covers (polynomial in data) — and task success vs $|w|$: certified stays flat, scaled decays and, crucially, **carries no certificate** (its error on an unseen $w$ is unpredictable).

**Discovery arm.** Repeat with $S$ **discovered** by an active-inference agent (not given) → the certificate is over a discovered group.

---

## 6. Phased plan

- **P0 — Theorem.** Write §2 rigorously (composition closure + multi-step + degradation bound), with the proof. *Output: the math section + an equivariance/closure unit test.*
- **P1 — Compositional certification experiment.** Plot 1 on the Step 19/24 world: error flat over $\langle S\rangle$, multi-step; fit the degradation bound. (Reuses step19/24/31/35.)
- **P2 — Structure-vs-scale baseline.** Plot 2: scaled non-equivariant + augmentation; show data-cost gap + no-certificate.
- **P3 — Discovery arm.** Discover $S$ via active inference (step33/36/34) → earned certificate.
- **P4 — Approximate-symmetry degradation.** Sweep $\epsilon$ (step30) × composition × horizon; map the certified region onto the step22 phase diagram. *This is the拷打-proof core: measure when the certificate dies.*
- **P5 — Generation demo + writeup.** Generate (= certified orbit-traverse) the solution to an unseen composed task and execute it closed-loop (step38/18).

---

## 7. Honesty (the part审稿人 and we both care about)

- **What is achievable (1–2 GPU):** the theorem + a clean, convincing proof-of-principle (Plots 1–2 at toy/小-scale). Impact via the *idea + certificate + clean demo*.
- **What is NOT claimed:** beating a scaled SOTA on a real large benchmark — that is multi-person/multi-year and we will **not** pretend otherwise.
- **The genuine risk (and our response):** real-world symmetry is approximate/weak → certificate degrades. The degradation theorem **quantifies** this; P4 **measures** it; if $\epsilon$ is large the honest conclusion becomes "the certificate is meaningful only above a symmetry-content threshold" (the Step 22 boundary) — *still a useful, publishable law*, not a failure.

### Attack points → contingencies
1. *"Just [B] restated."* → composition closure + multi-step + degradation bound are new; [B] is the $|w|{=}1,T{=}1$ corner; the **exponential certified set from $k$ checks** is the headline.
2. *"Approximate symmetry kills it."* → that's the **measured** degradation bound, not an unmodeled hole (P4).
3. *"Toy / hand-wired group."* → discovery arm (P3) earns the group from interaction; the world carries real (even approximate) symmetry.
4. *"Compounding rollout error makes the bound vacuous."* → measure $L$; with exact equivariance ($\epsilon\!\sim\!10^{-6}$) and $L\!\lesssim\!1$ the bound is tight; report the horizon where it stays meaningful.

---

## 8. Working title candidates

- *Certified 举一反三: structure as an alternative to scale in equivariant world models*
- *A generalization certificate for compositional symmetry in latent world models*
- *Scale buys interpolation; structure buys a certificate*
