## B. Proofs

We restate each formal claim and give a complete proof. Notation is as in §2: an encoder $E:\mathcal X\to\mathcal Z$, a
predictor $f:\mathcal Z\times\mathcal A\to\mathcal Z$, true dynamics $\Phi$, a group $G$ acting on situations
($x\mapsto g\cdot x$), latents (via $\rho$), and actions (via $\sigma$). The $T$-step rollout under an action sequence
$\bar a=(a_0,\dots,a_{T-1})$ is $\hat z_T(x;\bar a)=f(\cdot,a_{T-1})\circ\cdots\circ f(E(x),a_0)$, the target is
$E(\Phi^T(x;\bar a))$, and $\mathrm{Err}_T(x;\bar a)=\lVert \hat z_T(x;\bar a)-E(\Phi^T(x;\bar a))\rVert$. For a word
$w\in\langle S\rangle$ we write $\bar a^{\,w}=(\sigma(w)a_0,\dots,\sigma(w)a_{T-1})$ for the action sequence transported
by $w$; the configuration-axis statements compare the rollout at $x$ under $\bar a$ with the rollout at $w\cdot x$ under
$\bar a^{\,w}$ (the natural pairing, since $G$ acts on actions as well).

### Lemma 1 (composition closure)

*Statement.* If (A1)–(A3) hold on every generator $g_i\in S$, they hold on every word $w\in\langle S\rangle$.

*Proof.* Induction on word length $|w|$. For $|w|=0$, $w=e$ and $\rho(e)=I$, $\sigma(e)=I$, $e\cdot x=x$, so all three
are trivial. For $|w|=1$, $w=g_i$ is a generator, true by hypothesis. Inductive step: write $w=g_i\,w'$ with $|w'|=m$
and assume the claim for $w'$. Using that $\rho,\sigma$ are homomorphisms and the action is a left action:
$$
E\big((g_i w')\cdot x\big)=E\big(g_i\cdot(w'\cdot x)\big)\overset{\text{(A1)}}{=}\rho(g_i)E(w'\cdot x)
\overset{\text{IH}}{=}\rho(g_i)\rho(w')E(x)=\rho(g_iw')E(x),
$$
which is (A1) for $w$. For (A2), for all $z,a$,
$$
f\big(\rho(w)z,\sigma(w)a\big)=f\big(\rho(g_i)\rho(w')z,\sigma(g_i)\sigma(w')a\big)
\overset{\text{(A2)}}{=}\rho(g_i)f\big(\rho(w')z,\sigma(w')a\big)\overset{\text{IH}}{=}\rho(g_i)\rho(w')f(z,a)=\rho(w)f(z,a).
$$
For (A3), $\Phi\big(w\cdot x,\sigma(w)a\big)=\Phi\big(g_i\cdot(w'\cdot x),\sigma(g_i)\sigma(w')a\big)
\overset{\text{(A3)}}{=}g_i\cdot\Phi(w'\cdot x,\sigma(w')a)\overset{\text{IH}}{=}g_i\cdot w'\cdot\Phi(x,a)=w\cdot\Phi(x,a)$.
$\square$

A consequence we use repeatedly: the rolled-out predictor is equivariant, $\hat z_T(\rho(w)z;\bar a^{\,w})=\rho(w)\hat
z_T(z;\bar a)$, by applying (A2)-for-$w$ at each of the $T$ steps.

### Theorem A (orbit-constant error)

*Statement.* Under (A1)–(A4), for every $w\in\langle S\rangle$, every $x$, and every action sequence $\bar a$,
$\mathrm{Err}_T(w\cdot x;\bar a^{\,w})=\mathrm{Err}_T(x;\bar a)$.

*Proof.* By (A1) and Lemma 1, $\hat z_T(w\cdot x;\bar a^{\,w})=\hat z_T(\rho(w)E(x)\,;\bar a^{\,w})=\rho(w)\hat
z_T(E(x);\bar a)$, i.e. the model rollout transforms by $\rho(w)$. For the target, Lemma 1 applied to (A3) gives
$\Phi^T(w\cdot x;\bar a^{\,w})=w\cdot\Phi^T(x;\bar a)$, hence by (A1) $E\big(\Phi^T(w\cdot x;\bar a^{\,w})\big)
=\rho(w)E\big(\Phi^T(x;\bar a)\big)$. Subtracting,
$$
\mathrm{Err}_T(w\cdot x;\bar a^{\,w})=\big\lVert \rho(w)\big(\hat z_T(x;\bar a)-E(\Phi^T(x;\bar a))\big)\big\rVert
\overset{\text{(A4)}}{=}\big\lVert \hat z_T(x;\bar a)-E(\Phi^T(x;\bar a))\big\rVert=\mathrm{Err}_T(x;\bar a),
$$
where (A4) ($\rho(w)$ orthogonal, as $\rho$ is a homomorphism into $O(\mathcal Z)$) preserves the norm. $\square$

### Lemma 2 (the certificate characterizes equivariance — the converse)

*Statement.* Let $\rho:G\to O(\mathcal Z)$ act **freely** on an open set $U\subseteq\mathcal Z$. If a predictor $f$ has
orbit-constant error $\lVert f-\Phi\rVert$ on $U$ for **every** equivariant target $\Phi$, then $f$ is equivariant on
$U$ (i.e. $f(\rho(g)z)=\rho(g)f(z)$ for all $z\in U$, $g\in G$ with $\rho(g)z\in U$).

*Proof.* Fix $z\in U$ and $g\in G$. For an arbitrary $c\in\mathcal Z$ define a target on the orbit $G\cdot z$ by
$\Phi_c(\rho(h)z):=\rho(h)\,c$. This is well defined: freeness means the map $h\mapsto\rho(h)z$ is injective, so each
orbit point determines a unique $h$; and $\Phi_c$ is equivariant by construction, $\Phi_c(\rho(h')\rho(h)z)=\rho(h'h)c
=\rho(h')\Phi_c(\rho(h)z)$. Orbit-constancy of $\lVert f-\Phi_c\rVert$ at the two orbit points $z$ and $\rho(g)z$ reads
$$
\lVert f(z)-\Phi_c(z)\rVert=\lVert f(\rho(g)z)-\Phi_c(\rho(g)z)\rVert
\;\Longleftrightarrow\;\lVert f(z)-c\rVert=\lVert f(\rho(g)z)-\rho(g)c\rVert .
$$
Since $\rho(g)$ is orthogonal, $\lVert f(\rho(g)z)-\rho(g)c\rVert=\lVert \rho(g)^{-1}f(\rho(g)z)-c\rVert$. Thus
$\lVert f(z)-c\rVert=\lVert \rho(g)^{-1}f(\rho(g)z)-c\rVert$ for **all** $c\in\mathcal Z$. Two points $p,q$ with
$\lVert p-c\rVert=\lVert q-c\rVert$ for all $c$ are equal (set $c=p$ to get $\lVert q-p\rVert=0$). Therefore
$f(z)=\rho(g)^{-1}f(\rho(g)z)$, i.e. $f(\rho(g)z)=\rho(g)f(z)$. $\square$

*Remark (continuity).* $\Phi_c$ is in general only defined on the orbit and need not extend to a continuous global
target. When $G$ is compact with closed orbits, a continuous $G$-equivariant extension of $\Phi_c$ to a neighbourhood
exists (e.g. by an equivariant tubular-neighbourhood/averaging construction), so the converse holds even when the class
of admissible targets is restricted to continuous dynamics, not only the full algebraic class. Together with Theorem A
this gives the characterization *orbit-constant error $\iff$ equivariance*, and hence: **no non-equivariant predictor
has the certificate, at any parameter count.**

*Remark (the quantifier is necessary — and tight).* The "every equivariant target" quantifier cannot be weakened to a single target: take $f=\Phi+v$ for a fixed vector $v$ with $\rho(g)v\ne v$. Against that one $\Phi$ the error $\lVert f-\Phi\rVert\equiv\lVert v\rVert$ is exactly orbit-constant, yet $f(\rho(g)z)=\rho(g)\Phi(z)+v\ne\rho(g)f(z)$ — not equivariant. Lemma 2's hypothesis is thus the minimal one under which the converse holds; operationally this is why the certificate checks the model-side conditions (A1)–(A4) on generators rather than error-flatness against a single environment.

### Theorem B (spectral degradation — upper bound)

*Statement.* Let $\epsilon_{\max}=\max_i\sup_x\lVert E(g_i\cdot x)-\rho(g_i)E(x)\rVert$ be the encoder residual,
$\delta$ a uniform per-step predictor error $\sup_{z,a}\lVert f(z,a)-E(\Phi(E^{-1}z,a))\rVert$, and suppose the rollout's
latent Jacobian is, in a local frame, block-diagonal with channel-$j$ multiplier $e^{\lambda_j}$ and basis condition
number $\kappa_j$. Then for a word $w$ of length $m=|w|$,
$$
\big\lvert \mathrm{Err}_T(w\cdot x;\bar a^{\,w})-\mathrm{Err}_T(x;\bar a)\big\rvert\;\le\;\sum_j c_j\,(m\,\epsilon_{\max}+T\,\delta)\,e^{\lambda_j T},\qquad c_j=O(\kappa_j).
$$

*Proof.* Write the orbit-error variation as the norm of the difference between the two rollouts' residual vectors.
Relative to the exactly-equivariant idealization (Theorem A), two perturbation sources break orbit-constancy. (i) The
**encoder defect**: along the word $w=g_{i_1}\cdots g_{i_m}$, each generator contributes a latent perturbation of size
$\le\epsilon_{\max}$ to $E(w\cdot x)-\rho(w)E(x)$; by the triangle inequality and orthogonality of each $\rho(g_{i_\ell})$
these accumulate to $\lVert E(w\cdot x)-\rho(w)E(x)\rVert\le m\,\epsilon_{\max}$, a perturbation injected at $t=0$. (ii)
The **predictor defect**: at each of the $T$ steps the rollout deviates from the equivariant push-forward by $\le\delta$.
Let $u_t$ denote the injected perturbation at step $t$ ($\lVert u_0\rVert\le m\epsilon_{\max}$, $\lVert u_t\rVert\le\delta$
for $t\ge1$). Linearizing the rollout about the unperturbed trajectory, the contribution of $u_t$ to the terminal latent
is $D f^{(T-t)}\,u_t$, where $Df^{(T-t)}$ is the product of $T-t$ one-step Jacobians. In the local frame, the
channel-$j$ component is amplified by $\prod e^{\lambda_j}=e^{\lambda_j(T-t)}\le e^{\lambda_j T}$ for $\lambda_j>0$ (and
by $\le1$, hence $\le e^{\lambda_j T}$ trivially, for $\lambda_j\le0$), up to the basis condition number $\kappa_j$.
Summing over the injection times and channels,
$$
\big\lVert \textstyle\sum_t Df^{(T-t)}u_t\big\rVert\le\sum_j\kappa_j\Big(\lVert u_0\rVert+\sum_{t\ge1}\lVert u_t\rVert\Big)e^{\lambda_j T}
\le\sum_j\kappa_j\,(m\,\epsilon_{\max}+T\,\delta)\,e^{\lambda_j T},
$$
and the orbit-error variation is bounded by this quantity (the residual difference is exactly the propagated
perturbation, up to second order in $\epsilon_{\max},\delta$, which the local-linear regime discards). Setting
$c_j=O(\kappa_j)$ gives the bound. As $\epsilon_{\max},\delta\to0$ the right-hand side vanishes and Theorem A is
recovered. The certified horizon for channel $j$ — the largest $T$ with the bound $\le\epsilon_{\mathrm{res}}$ — is
$T_j(\epsilon)\sim\frac1{\lambda_j}\log\frac1\epsilon$ for $\lambda_j>0$, and $T_j=\infty$ for $\lambda_j\le0$. $\square$

*Scope.* This is a first-order propagation bound: the constants $c_j$ absorb the local non-normality (the change of
basis that diagonalizes the Jacobian), and the linearization is valid while the propagated perturbation stays within the
local-linear neighbourhood — exactly the regime in which a finite-resolution certificate is meaningful. Proposition 6
shows the *form* $\Theta(\log(1/\epsilon)/\lambda)$ is not improvable.

### Theorem B′ (cone / adapted-metric certified horizon)

**Theorem B′ (cone / adapted-metric certified horizon).** Let $\hat\phi:\mathcal U\to\mathcal U$ be $C^1$ on a compact forward-invariant $\mathcal U\subset\mathbb R^d$. Suppose there exist a continuous field of symmetric positive-definite matrices $z\mapsto P(z)$ and a constant $\Lambda\ge 1$ with $D\hat\phi(z)^\top P(\hat\phi(z))\,D\hat\phi(z)\preceq \Lambda^2 P(z)$ for all $z\in\mathcal U$. Let $\kappa=\big(\sup_z\lambda_{\max}P(z)\big)/\big(\inf_z\lambda_{\min}P(z)\big)$. Then $\lambda_1(\hat\phi)\le\log\Lambda$, and the linearized rollout error from an $\epsilon$-perturbation stays $\le\epsilon_{\mathrm{res}}$ for all $T\le T_{\mathrm{guar}}(\epsilon)=\big\lfloor(\log(\epsilon_{\mathrm{res}}/\epsilon)-\tfrac12\log\kappa)/\log\Lambda\big\rfloor$ — computed from $\hat\phi$ alone.

*Proof.* Put $V(z,v)=v^\top P(z)v$. The hypothesis gives $V(\hat\phi(z),D\hat\phi(z)v)\le\Lambda^2 V(z,v)$; iterating along an orbit $z_{t+1}=\hat\phi(z_t)$, $v_{t+1}=D\hat\phi(z_t)v_t$, yields $V(z_T,v_T)\le\Lambda^{2T}V(z_0,v_0)$ with $v_T=D\hat\phi^T(z_0)v_0$. Since $\lambda_{\min}(P(z))\|v\|^2\le V(z,v)\le\lambda_{\max}(P(z))\|v\|^2$, $\|D\hat\phi^T(z_0)\|\le\sqrt\kappa\,\Lambda^T$, so $\lambda_1\le\log\Lambda$ ($\sqrt\kappa$ is sub-exponential). The horizon bound follows from $\sqrt\kappa\,\Lambda^T\epsilon\le\epsilon_{\mathrm{res}}$. $\square$

*Remarks.* (i) First-order statement; Proposition 8's $C^1$-vs-$L^2$ caveat applies. (ii) Sound for any feasible $(P,\Lambda)$, tight when $P$ is the adapted (Oseledets) metric ($\Lambda\to e^{\lambda_1}$). (iii) **Continuum certificate:** verified on an $h$-cover with $D\hat\phi,P$ Lipschitz ($L_J,L_P$), it holds on all of $\mathcal U$ with $\Lambda^{\mathrm{cert}}=\Lambda_{\mathrm{samples}}+\sqrt\kappa\,L_J h+O(L_P h)$. (iv) Uniform hyperbolicity is precisely the regime where a $(P,\Lambda)$ exists with $\Lambda\to e^{\lambda_1}$; the cone-margin diagnostic detects it.

### Proposition 6 (the horizon is tight — matching lower bound)

*Statement.* Fix an expansive channel on which the latent map is locally linear with multiplier $a=e^\lambda$,
$\lambda>0$. There exist an exactly equivariant target $\Phi$ and an $\epsilon$-approximately-equivariant model — a
*perfect* equivariant predictor $f=\Phi$ ($\delta=0$) and an encoder equal to the exact-equivariant one except for a
single defect $E(g\cdot x)=\rho(g)E(x)+\epsilon u$ at one orbit point ($\lVert u\rVert=1$, $u$ in the channel) — such
that $\big\lvert \mathrm{Err}_T(g\cdot x)-\mathrm{Err}_T(x)\big\rvert=\epsilon\,e^{\lambda T}$.

*Proof.* Along $x$'s trajectory the model is exact, so $\mathrm{Err}_T(x)=0$. At $g\cdot x$, the encoder produces
$E(g\cdot x)=\rho(g)E(x)+\epsilon u$. Because $f=\Phi$ is equivariant and acts linearly with multiplier $a$ on the
channel, the $T$-step rollout is
$$
\hat z_T(g\cdot x)=f^{T}\big(\rho(g)E(x)+\epsilon u\big)=\rho(g)\,f^{T}(E x)+a^{T}\epsilon u,
$$
using linearity to split the defect and equivariance of $f^T$ (Lemma 1) on the first term. The target is, by (A3) and
encoder-exactness off the single defect, $E(\Phi^T(g\cdot x))=E(g\cdot\Phi^T x)=\rho(g)E(\Phi^T x)=\rho(g)f^T(Ex)$.
Subtracting and using (A4),
$$
\mathrm{Err}_T(g\cdot x)=\lVert a^T\epsilon u\rVert=\epsilon\,e^{\lambda T},\qquad\text{so}\quad\big\lvert\mathrm{Err}_T(g\cdot x)-\mathrm{Err}_T(x)\big\rvert=\epsilon\,e^{\lambda T}. \square
$$
Thus the largest horizon with orbit-variation $\le\epsilon_{\mathrm{res}}$ is
$T=\frac1\lambda\log\frac{\epsilon_{\mathrm{res}}}{\epsilon}=\Theta\!\big(\frac1\lambda\log\frac1\epsilon\big)$, matching
the upper bound of Theorem B. Consequently a certificate derived from an $\epsilon>0$ equivariance residual cannot
promise predictability beyond $T\sim\frac1\lambda\log\frac1\epsilon$ on any expansive channel (worst case over admissible
targets); only $\epsilon=0$ (exact equivariance) or $\lambda\le0$ (conservation/contraction) gives an unbounded horizon.

### Proposition 7 (scope — when the local spectrum certifies a learned model's horizon)

*Statement.* Let $\phi$ have an ergodic invariant measure $\mu$ with $\log^+\lVert D\phi\rVert\in L^1(\mu)$. **(a)
Non-degenerate ($\lambda_1>0$):** the certified horizon obeys $T(\epsilon)=\frac1{\lambda_1}\log(\epsilon_{\mathrm{res}}/\epsilon)+o(t)$, with $\lambda_1$ the measurable
asymptotic rate. **(b) Degenerate ($\lambda_1\approx0$):** the leading-order log-law degenerates and the one-step
spectrum carries no finite-slope horizon rate.

*Proof.* By the Oseledets multiplicative ergodic theorem, for $\mu$-a.e. $x$ and Lebesgue-a.e. perturbation direction
$v$ (those not lying in the measure-zero slower Oseledets subspaces), $\frac1t\log\lVert D\phi^t_x v\rVert\to\lambda_1$.
Hence the perturbation magnitude satisfies $\lVert\delta_t\rVert=\lVert D\phi^t_x v\rVert=e^{(\lambda_1+o(1))t}\lVert\delta_0\rVert$. **(a)** The certified horizon is the largest $T$ with
$\lVert\delta_T\rVert\le\epsilon_{\mathrm{res}}$ starting from resolution $\epsilon$; solving
$e^{(\lambda_1+o(1))T}\epsilon=\epsilon_{\mathrm{res}}$ gives
$T(\epsilon)=\frac1{\lambda_1}\log(\epsilon_{\mathrm{res}}/\epsilon)+o(T)$, which is Theorem B's law with $\lambda_1$ the
Oseledets rate. **(b)** If $\lambda_1=0$ (or $\to0$), the exponent in $\lVert\delta_t\rVert=e^{(\lambda_1+o(1))t}$
vanishes to leading order, so $\log\lVert\delta_t\rVert$ is $o(t)$: there is no finite, $\epsilon$-independent slope
$\mathrm dT/\mathrm d\log(1/\epsilon)$, the leading-order law is empty, and the one-step Jacobian spectrum does not
determine a multi-step horizon. $\square$

The dichotomy is the *scope* of the horizon certificate: informative exactly on spectrally non-degenerate dynamics. It
predicts, rather than patches, the failure on near-neutral dynamics (the learned-PushT-interior probe, where $\lambda_1\approx0$ and the local spectrum does not predict the rollout, $R^2\approx0.02$). The *lift* to a learned
$\hat\phi$ — that $\hat\phi$ reproduces $\lambda_1$ — is not a corollary of shadowing (which controls trajectory
closeness, not the asymptotic exponent, the latter being only upper-semicontinuous under $C^1$ perturbation); it is the
finite-horizon continuity statement of Proposition 8.

### Proposition 8 (finite-horizon exponent transfer)

*Statement.* The finite-time exponent $\lambda_1^{(T)}(\phi)=\frac1T\,\mathbb E_x\log\lVert D\phi^T_x\rVert$ is locally
Lipschitz in $\phi$ in the $C^1$ topology. Consequently a learned $\hat\phi$ with $\lVert\hat\phi-\phi\rVert_{C^1}\le\delta$ recovers the certified-horizon staircase slope up to an $O(\delta)$ model-fidelity bias plus the
finite-$T$ truncation $\lvert\lambda_1^{(T)}-\lambda_1\rvert$; under a dominated splitting the bias is $T$-uniform.

*Proof.* Fix $T$ and a base point $x$ and let $D\phi^T_x=\prod_{t=0}^{T-1}D\phi_{\phi^t x}$ be the $T$-step cocycle. For
two dynamics $\phi,\hat\phi$ with $\lVert\hat\phi-\phi\rVert_{C^1}\le\delta$, each one-step Jacobian satisfies $\lVert
D\hat\phi-D\phi\rVert\le\delta$ (definition of the $C^1$ norm), and the orbits stay $O(\delta)$-close over the finite
horizon $T$ (Grönwall over $T$ bounded steps). A finite product of matrices is a Lipschitz function of its factors on
any bounded set: if $\lVert A_t\rVert,\lVert \hat A_t\rVert\le M$ and $\lVert\hat A_t-A_t\rVert\le\delta'$, then
$\lVert\prod\hat A_t-\prod A_t\rVert\le T M^{T-1}\delta'$, and $\log\lVert\cdot\rVert$ is Lipschitz away from $0$ (the
norm is bounded below by hyperbolicity along the expanding direction). Averaging over $x$ and dividing by $T$,
$$
\big\lvert \lambda_1^{(T)}(\hat\phi)-\lambda_1^{(T)}(\phi)\big\rvert\;\le\;L_T\,\delta,
$$
a (horizon-dependent) Lipschitz bound — the $O(\delta)$ fidelity bias. The staircase slope estimates
$\lambda_1^{(T)}(\hat\phi)$; the remaining gap to the *asymptotic* $\lambda_1$ is the finite-$T$ truncation
$\lvert\lambda_1^{(T)}(\phi)-\lambda_1\rvert\to0$. Under a dominated splitting the asymptotic exponent is continuous in
the dynamics (Bochi–Viana), so the constant $L_T$ can be taken uniform in $T$ and the truncation decays uniformly,
making the recovered exponent $T$-uniformly close to $\lambda_1$ up to $O(\delta)$. The bound is **falsifiable**: the
recovered exponent must tighten as $\delta$ (one-step error) drops — confirmed empirically (Rössler: relative error
$44\%\to8\%$ as training fidelity improves). $\square$

*Honest caveat.* We verify $C^1$-closeness only through the one-step error (an $L^2$ proxy for the Jacobian distance),
and the high-dimensional experiment shows this proxy degrades with dimension: a one-step-accurate dense model can have
an inaccurate Jacobian and hence a wrong spectrum; structure (a $\mathbb{Z}_N$-equivariant, banded-Jacobian model)
restores it. The dominated-splitting hypothesis is assumed, not certified, for the learned models.

### Proposition 9 (budgeted re-observation — a mis-estimated horizon costs a proportional budget)

*Statement.* Consider an agent that forecasts the map open-loop from a re-observation, the leading-mode error growing
as $\delta_t\approx\delta_0 e^{\lambda_1\Delta t\,t}$ over map steps $t$ (step $\Delta t$, $\lambda_1>0$), so the
trustworthy horizon at resolution $\epsilon$ is $H(\epsilon)=\lfloor\log(\epsilon/\delta_0)/(\lambda_1\Delta t)\rfloor$
(the certified horizon of §3.2, in map steps). The agent re-observes (resets the error to $\delta_0$) at a fixed
cadence and may re-observe at most $B$ times over an episode of $L$ map steps (a **sensing budget**); a step is a
*violation* if its forecast error exceeds $\epsilon$. A certificate reporting $\hat\lambda_1=c\,\lambda_1$ ($c>0$)
prescribes cadence $\hat H=H/c$. Then **(i)** for $c\ge1$ in the budget-binding regime $BH/c<L$, the aggregate
violation rate is
$$V(c)=\max\!\big(0,\;L-BH/c-H\big)/L,$$
which is **non-decreasing in $c$** (strictly increasing while $BH/c+H<L$); and **(ii)** the budget needed to drive $V$
to zero is $B^\star(c)=\lceil c\,(L-H)/H\rceil$ — **linear in $c$**: a certificate inflated $c\times$ demands $c\times$
the observations to certify the same episode.

*Proof.* With cadence $\hat H=H/c\le H$, every window before the last re-observation (at step $B\hat H=BH/c$) stays
under $\epsilon$ — the error reaches $\epsilon$ only after $H\ge\hat H$ steps — so the covered $BH/c$ steps are
violation-free; after the last re-observation the open-loop forecast exceeds $\epsilon$ only after a further $H$ steps,
leaving $\max(0,\,L-BH/c-H)$ violating steps, which is $V(c)$. As $BH/c$ is non-increasing in $c$, $V$ is
non-decreasing, strictly so while the numerator is positive; $V=0$ requires $BH/c\ge L-H$, i.e. $B\ge c(L-H)/H$.
$\square$

*Remark.* $V(c)$ is computable a priori from the inflation $c$ and the budget $B$, so it *predicts* the violation gap;
the calibrated certificate ($c{=}1$) is the budget-minimal violation-free cadence, and only structure delivers $c{=}1$
a priori (E2 / Proposition 8). E12 instantiates the law: a non-equivariant certificate with $c\approx3.4$ on Lorenz-96
(resp. $c\approx2$ on the pendulum ring) needs $\approx3\times$ (resp. $\approx2\times$) the budget to match the
equivariant certificate, and the gap closes exactly when recalibration restores $c\to1$. (Integer-rounding of the cadence $\hat H=H/c$ costs $O(1)$ per window and is absorbed in the measured catch-up factor, $2.7$–$3.5\times$ against the predicted $c\approx3.4$.)


### Proposition 10 (finite-sample certified-horizon interval)

*Statement.* Let $g$ be $C^1$ on a compact forward-invariant $\mathcal U$ (e.g. the SimNorm simplex product of E13),
and let $\ell_t$ be the per-step leading log-stretches produced by the Benettin recursion along an orbit of $g$, so
$\hat\lambda_1^{(n)}=\frac1n\sum_{t=1}^n\ell_t$ and $|\ell_t|\le B:=\sup_{z\in\mathcal U}\log\lVert Dg(z)\rVert<\infty$
(compactness). Assume $(\ell_t)$ is stationary and strongly mixing with summable autocovariances, and let
$\sigma_\infty^2=\sum_{h\in\mathbb Z}\mathrm{cov}(\ell_0,\ell_h)<\infty$ be the long-run variance. Then for every
$\delta\in(0,1)$ there is $\varepsilon_n(\delta)=\sigma_\infty\sqrt{2\log(2/\delta)/n}\,(1+o(1))$ such that with
probability $\ge1-\delta$, $\lambda_1\in[\hat\lambda_1^{(n)}-\varepsilon_n,\hat\lambda_1^{(n)}+\varepsilon_n]$, and
whenever $\hat\lambda_1^{(n)}-\varepsilon_n>0$ the certified horizon is bracketed,
$$ T_1(\epsilon)\in\Big[\tfrac{\log(1/\epsilon)}{\hat\lambda_1^{(n)}+\varepsilon_n},\;
\tfrac{\log(1/\epsilon)}{\hat\lambda_1^{(n)}-\varepsilon_n}\Big]; $$
otherwise the certificate **abstains**. In particular the certificate's sample complexity is
$n\asymp\sigma_\infty^2\log(1/\delta)/\varepsilon^2$ — logarithmic in confidence, quadratic in precision.

*Proof sketch.* Boundedness gives $\ell_t\in[-B,B]$; stationarity + strong mixing with summable covariances give a
Bernstein/CLT-type concentration for the empirical mean of a bounded mixing sequence with variance proxy
$\sigma_\infty^2$ (e.g. Merlevède–Peligrad–Rio); the horizon bracket follows since $\lambda\mapsto\log(1/\epsilon)/\lambda$
is monotone on $(0,\infty)$. $\square$

*Remarks.* (i) The moving-block bootstrap used throughout the experiments is precisely a consistent estimator of
$\sigma_\infty^2$ under the same mixing assumptions — Proposition 10 is the rate statement behind those CIs, not a new
procedure. (ii) The dynamical assumptions (stationarity/mixing along the audited orbit) are inherited from
Proposition 7's scope and are *assumed, not certified*, for learned models — stated honestly, as everywhere else.
(iii) $B$ is finite and computable on compact latents (E13's SimNorm product), so the bound is fully effective there.

### Proposition 11 (decision scope — where a horizon certificate carries decision value)

*Statement.* Adopt Proposition 9's forecast-error model: between re-observations the leading-mode error grows as
$\delta_t\approx\delta_0e^{\lambda_1\Delta t\,t}$, the certified horizon at resolution $\epsilon$ is
$H(\epsilon)=\lfloor\log(\epsilon/\delta_0)/(\lambda_1\Delta t)\rfloor$, the agent re-observes at most $B$ times over
$L$ map steps, and the certificate reports $\hat\lambda_1=c\,\lambda_1$ ($c\ge1$).

**(i) (scope-aligned decision: decided quantity $=$ certified quantity.)** If the decision rule is a function of the
certified predicate alone — re-observe/flag so that forecast error stays $\le\epsilon$ — then the certificate-prescribed
cadence $\hat H=H(\epsilon)/c$ incurs violation-rate regret against the omnisciently calibrated policy of
$$R_{\mathrm{align}}(c)\;=\;V(c)-V(1)\;\le\;\frac{B\,H(\epsilon)}{L}\Big(1-\frac1c\Big),$$
which **vanishes at $c=1$**: for an aligned decision, a calibrated certificate is decision-optimal a priori, and the
entire regret is the calibration factor.

**(ii) (task-mapped decision: decided quantity $=h(\text{certified quantity})$.)** If instead decision quality is the
episode average of a task loss $\ell(\delta_t)$ with an **implicit task tolerance** $\theta^{\ast}$ — $\ell\equiv0$ on
$[0,\theta^{\ast}]$ and $\ell\le\ell_{\max}$ beyond — then the $\epsilon$-certificate's prescription, even perfectly
calibrated ($c=1$), incurs
$$R_{\mathrm{task}}(\epsilon)\;\le\;\ell_{\max}\,
\frac{\max\!\big(0,\,H(\epsilon)-H(\theta^{\ast})\big)}{H(\epsilon)}
\qquad\text{(task violations when }\epsilon>\theta^{\ast}\text{)},$$
and wastes $\Delta B=B\big(H(\theta^{\ast})/H(\epsilon)-1\big)_+$ re-observations when $\epsilon<\theta^{\ast}$. Both
terms vanish **iff** $H(\epsilon)=H(\theta^{\ast})$ — iff the certificate is issued at the task's own resolution — and
the mis-resolution penalty in horizon units is $|H(\epsilon)-H(\theta^{\ast})|=\big|\log(\epsilon/\theta^{\ast})\big|/(\lambda_1\Delta t)$.
Since $\theta^{\ast}$ is a property of the task map $h$, not of the dynamics, no dynamics-only certificate can supply
it.

*Proof.* (i) From Proposition 9, $V(c)=\max(0,L-BH/c-H)/L$ with $H=H(\epsilon)$. For $c\ge1$,
$L-BH/c-H\;\ge\;L-BH-H$, and $\max(0,x)-\max(0,y)\le x-y$ for $x\ge y$, so
$V(c)-V(1)\le\big[(L-BH/c-H)-(L-BH-H)\big]/L=(BH/L)(1-1/c)$, zero at $c=1$.
(ii) Within one re-observation window of length $H(\epsilon)$ the error is monotone in staleness, so the task-violating
steps ($\delta_t>\theta^{\ast}$) are exactly the last $H(\epsilon)-H(\theta^{\ast})$ when $\epsilon>\theta^{\ast}$
(none are certificate-violating: $\delta_t\le\epsilon$ throughout — the dilution is **invisible to the certificate**);
each contributes at most $\ell_{\max}$, giving the per-window (hence episode-average) bound. When
$\epsilon<\theta^{\ast}$ every step is task-valid already at cadence $H(\theta^{\ast})$, so running at $H(\epsilon)$
spends $B\,(H(\theta^{\ast})/H(\epsilon)-1)$ avoidable reads. Both quantities are zero iff
$H(\epsilon)=H(\theta^{\ast})$; the horizon-unit gap is the difference of the two logarithms. $\square$

*Remark (the scope law, now a theorem — and its experimental instances).* Clause (i) is E12 and the deployment monitor
E15/step94: the decided quantity *is* latent $\theta$-validity, and the published certificate priced the in-situ
staleness clock with no new estimation (in-situ ratio $0.75$ vs bench $0.83$ on the calibrated cell). Clause (ii) is
E11 and step93: a planner (MPPI) absorbs latent staleness up to an implicit tolerance — empirically
$H(\theta^{\ast})\approx2$ agent-steps against $H(0.2)\approx6$ — so a $0.2$-resolution certificate over-prescribes
the replan cadence by the predicted mis-resolution factor $\approx3$, and the return gap dilutes exactly as the bound
allows. The honest limit is built in: re-issuing the certificate at $\theta^{\ast}$ would close the gap, but
$\theta^{\ast}$ must be *elicited from the task* (e.g. a one-knob cadence sweep) — the theorem makes the boundary of
a-priori decision value precise instead of promising past it.

### Proposition 4 (isotypic placement)

*Statement.* A conserved charge $Q:\mathcal Z\to W$ that is $G$-equivariant (intertwines $\rho$ with a representation
$\tau$ on $W$, $Q\circ\rho(g)=\tau(g)\circ Q$) has its non-trivial action confined to the $\tau$-isotypic component of
$\rho$: a scalar invariant (trivial $\tau$, $\ell{=}0$) reads off the invariant block; a vector charge ($\tau$ the
standard $\ell{=}1$ representation, e.g. angular momentum) reads off the $\ell{=}1$ block and, when built bilinearly
from $\ell{=}1$ latent features, is realized by the **unique** equivariant degree-2 map — the cross product.

*Proof.* Decompose $\rho\cong\bigoplus_\mu \rho_\mu^{\oplus n_\mu}$ into isotypic components ($\rho_\mu$ the distinct
irreducibles). Restricted to the source, the equivariant linear part of $Q$ is an intertwiner $\rho\to\tau$. By Schur's
lemma, $\mathrm{Hom}_G(\rho_\mu,\tau)=0$ unless $\rho_\mu\cong\tau$, so any equivariant (linear) read-out of a
$\tau$-type charge must vanish on every isotypic block $\mu\not\cong\tau$ and is supported on the $\tau$-isotypic block —
"placement is forced, not chosen." For a **bilinear** charge built from two $\ell{=}1$ (vector) latent features in
$\mathrm{SO}(3)$, the Clebsch–Gordan decomposition $\mathbf 1\otimes\mathbf 1=\mathbf 0\oplus\mathbf 1\oplus\mathbf 2$
contains the target $\ell{=}1$ ($\mathbf 1$) exactly once; the corresponding equivariant projection is the antisymmetric
part, i.e. the cross product (the scalar $\mathbf 0$ is the dot product, the $\mathbf 2$ the symmetric-traceless part).
Uniqueness up to scale is again Schur ($\dim\mathrm{Hom}_G(\mathbf 1\otimes\mathbf 1,\mathbf 1)=1$). $\square$

### Proposition 5 (conservation $\Rightarrow$ unbounded horizon)

*Statement.* Let $Q:\mathcal Z\to W$ be a charge the model conserves to one-step defect $\eta$, i.e. $\lVert
Q(f(z,a))-Q(z)\rVert\le\eta$ for all $(z,a)$ along the rollout, and let the true dynamics conserve $Q$ exactly
($Q(z_{t+1})=Q(z_t)$), with matched initial charge $Q(\hat z_0)=Q(z_0)$. Then the $T$-step charge-value error obeys
$\lVert Q(\hat z_T)-Q(z_T)\rVert\le T\eta$ — linear in $T$, never $e^{\lambda T}$ — and at $\eta=0$ the charge is
certified to all horizons.

*Proof.* Telescoping the model rollout and using the one-step defect at each step,
$$
\lVert Q(\hat z_T)-Q(\hat z_0)\rVert=\Big\lVert\sum_{t=0}^{T-1}\big(Q(\hat z_{t+1})-Q(\hat z_t)\big)\Big\rVert
\le\sum_{t=0}^{T-1}\lVert Q(\hat z_{t+1})-Q(\hat z_t)\rVert\le T\eta.
$$
The true charge is conserved, $Q(z_T)=Q(z_0)$, and $Q(\hat z_0)=Q(z_0)$ by hypothesis, so
$\lVert Q(\hat z_T)-Q(z_T)\rVert=\lVert Q(\hat z_T)-Q(z_0)\rVert=\lVert Q(\hat z_T)-Q(\hat z_0)\rVert\le T\eta$. The growth
is linear; at $\eta=0$ it is $0$ for all $T$, an unbounded certified horizon for the charge value. $\square$

*Scope.* The statement is about the **charge value** $Q(\hat z_T)$, not the full latent state, and the defect $\eta$ is
exact only under an equivariant symplectic discretization of a $G$-invariant Hamiltonian flow (momenta exactly
conserved, energy to $O(\Delta t^p)$); for a general learned $f$, $\eta$ is measured. The converse fails: a slow
($\lambda\le0$) channel need not carry a conserved charge.

## C. Reproducibility

All experiments are CPU/single-GPU, run with explicit random seeds, and **honestly gated**: a run prints
`INCONCLUSIVE` rather than loosen a threshold. Anonymized code accompanies the submission; every figure and the
per-seed JSONs are regenerated by the scripts below, and load-bearing claims carry equivariance/protocol unit tests.

Experiment-to-code map (claim — code — test — seeds):

- **E1** configuration certificate ($\mathbb{Z}_2^6$: $6\!\to\!64$) — `experiments/step49_iching_certificate.py` — (1 seed).
- **E2** horizon staircase (controlled spectrum; analytic tightness) — `experiments/step52_horizon_resolution.py`, `step65` — `tests/test_step52_horizon_resolution.py` — (3).
- **E2** horizon on a learned model of *real* chaos (Lorenz) — `experiments/step70_lorenz_horizon.py` — `tests/test_step70.py` — (3).
- **E2** horizon across a *class* (Hénon, Rössler, Lorenz) — `experiments/step71_multichaos_horizon.py` — `tests/test_step71.py` — (3).
- **E2** **high-dimensional** spectral horizon (40-D Lorenz-96; structure vs. dense) — `experiments/step74_lorenz96_spectrum.py` — `tests/test_step74.py` (Liouville $\sum_j\lambda_j=-N$) — (3).
- **Proposition 6** tightness (numerical confirmation of the lower bound) — `experiments/step65_horizon_tightness.py` — (3).
- Certificate on real contact dynamics (PushT) — `experiments/step59_pusht_certificate.py` — (3).
- **E3** approximate-symmetry degradation — `experiments/step53_approximate_symmetry.py` — (3).

Multi-seed steps commit per-seed JSONs to `papers/figures/`; every range in the text is the seed min–max from those
files. The estimator behind the horizon experiments is anchored by an exact physical invariant: for Lorenz-96 the
vector-field divergence is $-N$, so $\sum_j\lambda_j=-N$ exactly, which the Benettin-QR estimator reproduces to $0.0\%$
(`tests/test_step74.py`) before any learned-model spectrum is trusted. The full test suite passes together; everything
is CPU/MPS (no CUDA required). A single command rebuilds the paper end-to-end from the committed JSONs.
