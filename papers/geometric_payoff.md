# The geometric payoff: does SO(2)-equivariance buy sample efficiency and 举一反三?

> Results log for the contrarian thesis (CLAUDE.md, open question #1): *if the
> world has a symmetry, does building that symmetry into a latent world model let
> it learn from fewer interactions and generalise to configurations it never saw?*
> Steps 3–7 established that the symmetry holds **exactly**; Steps 8–14 are the
> **payoff** experiments that test whether exactness converts into the two things
> the thesis actually claims — data efficiency and zero-shot generalisation across
> the symmetry group (举一反三) — through the full Phase-4 architecture (Step 11: an
> equivariant JEPA that predicts and plans *in a learned latent space*), a
> contact-dominated *pose*-control closed loop on real PushT (Step 12), the
> **SO(3) lift** to an end-to-end 3D point-cloud latent JEPA (Step 13) — the
> project's actual target geometry — and a **paired closed-loop power analysis**
> (Step 14) that finally converts the prediction gap into an *exact* closed-loop
> orientation-invariance result under an equivariant planner.

Last updated: 2026-05-30. All experiments run CPU/MPS on a laptop, fully seeded
and deterministic (re-running reproduces every number below).

![The geometric payoff in one figure](figures/killer_figure.png)

> **Figure 1.** The payoff as the three error bars a sceptic asks for, read straight
> from the per-step runs logged below. **(a)** OOD/seen prediction-error factor: the
> equivariant model is flat ($\approx\!\times1$) across every setting — SO(2) synth &
> real (Steps 8, 10), SO(2) latent (Step 11), SO(3) 3D (Step 13), full SE(3) (Step 15)
> — while the same-hypothesis-class baseline blows up $\times13$–$\times157$. **(b)**
> Five *independently trained* (VN, MLP) pairs, real-PushT closed-loop pose control
> (Step 17): the VN's seen-vs-unseen block-angle sits on $y=x$ (orientation-invariant,
> $\Delta=-1.0°$) while the baseline sits above ($\Delta=+9.6°$) — the contrast is the
> *architecture*, not the lucky seed. **(c)** Deliberately breaking the SO(3) symmetry
> of the teacher (Step 16): the prior's OOD error rises (it is *not* free once the world
> de-symmetrises) yet stays below the unconstrained baseline even past 50%
> symmetry-breaking — an honest bracket on Sutton's Bitter-Lesson crossover. Regenerate
> with `experiments/make_figures.py`.

---

## 0. Setup and notation

A latent world model is an encoder $E_\theta:\text{obs}\to z$ plus a forward
(predictor) model $f_\phi(z,a)\approx z'$. We say the model is **$G$-equivariant**
if a group element $g\in G$ acting on the input transforms the latent by a known
representation $\rho(g)$:
$$ E_\theta(g\cdot x) = \rho(g)\,E_\theta(x), \qquad f_\phi(\rho(g)z,\,g\cdot a)=\rho(g)\,f_\phi(z,a). $$
When $\rho(g)$ is **orthogonal**, the planning cost $\mathcal{C}=\lVert \hat z_H - z_g\rVert_2^2$
is invariant to a joint action of $g$ on (state, goal, actions) — the planner cannot
tell two $g$-related problems apart, so it solves them identically.

We work with $G=\mathrm{SO}(2)$ (planar rotations) acting on stacks of **type-1
vectors** $v\in\mathbb{R}^2$ by the ordinary rotation $R_\alpha$, so $\rho(\alpha)$
is block-diagonal copies of $R_\alpha$ — orthogonal, as required.

**Vector Neurons (VN)** (Deng et al., 2021) are the equivariant primitive: a
`VNLinear` mixes channels with *scalar* weights, $V'_o=\sum_i W_{oi}V_i$ (rotation
acts on the spatial axis, weights on the channel axis, **no bias**); a `VNReLU`
rectifies each vector against a learned equivariant direction. Composing them gives
a map that is exactly $\mathrm{SO}(d)$-equivariant by construction — and is the
*same* code for $d=2$ and $d=3$.

---

## 1. Foundation (Steps 3–7, one-line recap)

| Step | Claim established | Worst-case equivariance error |
|------|-------------------|-------------------------------|
| 3 | SO(2)-steerable **encoder** (e2cnn) is exactly equivariant | float floor |
| 4 | Steerable encoder beats ordinary CNN under field-of-view shift (A/B, Muon) | — |
| 5 | Structured/point-cloud state path; PushT 6-vector extraction | — |
| 6 | **Whole** world model (encoder + VN predictor) equivariant, 2D *and* SE(3) 3D | $\sim 10^{-6}$ |
| 7 | Robustness ranking across 16 envs | — |

Steps 3–7 are *necessary but not the thesis*: an exactly-symmetric model is only
worth building if exactness pays off in learning. Steps 8–9 are that test.

---

## 2. Step 8 — sample efficiency + 举一反三, one-step prediction

**Design (controlled head-to-head).** Instantiate a *world* whose one-step dynamics
$s'=T(s,a)$ is *exactly* SO(2)-equivariant by making $T$ a **frozen random VN net
with a single nonlinearity** ("the world happens to have this symmetry"). Two
students learn $T$ from $N$ sampled transitions:

- **VN (equivariant)** — a *deeper* (two-nonlinearity) VN net, ~3.5k params. It does
  **not** clone the teacher's architecture; it only shares the *symmetry class*.
- **MLP (baseline)** — a plain MLP on the flattened state+action, **~5.7× more
  parameters**. Not starved — it simply lacks the symmetry prior.

Equivariance of both the world and the student is checked at init (errors $<10^{-4}$).

### [B] Sample efficiency — test relMSE vs. number of training transitions $N$

Isotropic test set, full orientation coverage. relMSE is normalised by target
power (1.0 = predicting zero).

| $N$ | VN relMSE | MLP relMSE |
|----:|----------:|-----------:|
| 16  | `0.241` | `0.521` |
| 32  | `0.210` | `0.332` |
| 64  | `0.194` | `0.263` |
| 128 | `0.085` | `0.268` |
| 256 | `0.015` | `0.257` |
| 512 | `0.0040` | `0.233` |

- The VN at $N=32$ (`0.210`) already beats the MLP's best ($N=512$, `0.233`):
  it **matches the MLP's best error using 16× fewer transitions.**
- At $N=512$ the VN essentially **solves** the task ($4.0\times10^{-3}$) while the
  MLP **plateaus** ($0.23$) — a generalisation gap that *more data alone will not
  close*, because the MLP's hypothesis class is not tied across the orbit.

### [C] 举一反三 — train on a $[0°,90°)$ wedge, test across the whole circle

Crucial subtlety: inputs must be **anisotropic** (a fixed canonical layout + noise),
otherwise the OOD test is *vacuous* — an isotropic Gaussian cloud is
rotation-invariant *as a distribution*, so rotating it lands in the same region.
With anisotropic inputs, a global rotation genuinely moves the cluster into an
unseen region.

| test orientation | VN relMSE | MLP relMSE |
|---|---:|---:|
| $[0°,90°)$ (seen) | `1.43e-3` | `0.032` |
| $[90°,180°)$ | `1.51e-3` | `0.68` |
| $[180°,270°)$ | `1.41e-3` | `3.41` |
| $[270°,360°)$ | `1.67e-3` | `1.24` |
| **degradation (worst/seen)** | **×1.17** | **×107** |

For an equivariant map, fitting it on a wedge *mathematically determines* it on the
whole orbit: the VN cannot tell the orientations apart, so its error is **flat**
(×1.17). The MLP must extrapolate to unseen orientations and **collapses** (×107).

### [D] Reality check — same test, inputs drawn from REAL PushT states

Repeating [C] with the input state distribution taken from real PushT (still with
the synthetic equivariant target) gives **VN flat (×1.00)** vs **MLP ×7** — the
conclusion is not an artefact of synthetic inputs.

**Step 8 verdict.** When the world is equivariant, the geometric prior converts
exactness into (i) ~16× sample efficiency and (ii) zero-shot generalisation across
the rotation group. Confidence ≈ **0.9**.

---

## 3. Step 9 — closed-loop few-shot planning + 举一反三 (CEM-MPC)

Step 8 proved the benefit for *one-step* prediction. A world model earns its name
only if it can **plan**: roll its own dynamics forward over a horizon and act.
Compounding model error over a rollout is exactly what kills naive learned planners,
so "good 1-step error" does *not* automatically give "good closed-loop control".
Step 9 closes that gap.

**Task.** A damped point mass reaching the origin, with dynamics
$$ v' = v + \Delta t\bigl(a - c_1 v + \kappa\,N(v,a)\bigr), \qquad p' = p + \Delta t\,v', $$
where $N$ is a **frozen random VN net** (so the ground truth lies *inside* the
equivariant hypothesis class — see §17 for why this matters), $a-c_1v$ is a
controllable, contractive skeleton, and $\kappa=1.5$ scales the direction-coupled
nonlinearity. The map is *exactly* SO(2)-equivariant (verified to $\sim 10^{-7}$).

**Models.** Equivariant VN forward model (3232 params) vs. plain MLP forward model
(17924 params, **5.5×**), trained on transitions whose $(v,a)$ directions lie in a
$[0°,90°)$ wedge.

**Planner.** Real CEM model-predictive control, run **open-loop**: the model rolls
its *own* dynamics over the whole $H=20$ horizon and we execute the plan **without
per-step correction**, so success depends on the *model's* multi-step accuracy, not
on the true env babysitting it. (With per-step replanning the true env corrects
model error every step and *both* models look fine — a deliberately easy regime we
avoid; this itself is the point that you need a good model when you can't lean on
constant correction.) A **true-dynamics oracle** planner is the ceiling proving the
CEM controller works.

### [B] One-step fit (1500 wedge transitions)

| 1-step relMSE | VN | MLP |
|---|---:|---:|
| in-wedge $[0°,90°)$ | `2.9e-5` | `4.3e-5` |
| full circle | `2.1e-4` | `6.0e-3` |

Both fit the wedge (the MLP **can** fit — fair comparison); off-wedge the MLP is
~28× worse, the VN stays flat (equivariance).

### [C] 举一反三 in CLOSED LOOP — reach in directions never practised

Success rate over 24 reaches per motion-direction quadrant; open-loop plan-and-execute.

| motion dir | oracle | VN (equiv) | MLP |
|---|---:|---:|---:|
| $[0°,90°)$ practised | `1.00` | **`1.00`** | `0.83` |
| $[90°,180°)$ unseen | `1.00` | **`1.00`** | `0.50` |
| $[180°,270°)$ unseen | `1.00` | **`1.00`** | `0.58` |
| $[270°,360°)$ unseen | `1.00` | **`1.00`** | `0.33` |
| **unseen-dir mean** | — | **`1.00`** | `0.47` |

The oracle reaches everywhere (the controller is sound). The **equivariant planner
is flat at 1.00 across all four quadrants** — it plans reaches in directions it
never practised. The MLP works where it practised ($0.83$, so it is genuinely
capable) but **degrades to $0.47$ on unseen directions**. This is closed-loop
举一反三.

### [D] Reality check — real PushT multi-step rollout (approx. equivariant)

Few-shot (256 transitions) multi-step rollout relMSE vs. horizon, on real PushT
6-vector state:

| horizon $h$ | VN relMSE | MLP relMSE |
|---:|---:|---:|
| 1 | `5.8e-4` | `5.2e-4` |
| 3 | `3.3e-3` | `3.3e-3` |
| 6 | `6.9e-3` | `8.2e-3` |
| **mean over horizon** | **`3.68e-3`** | `4.05e-3` |

On the real, only-*approximately*-equivariant system, the equivariant model tracks
the dynamics with **lower compounding error** from few transitions — the property
planning actually needs.

**Step 9 verdict.** The geometric prior turns "practice in one direction" into "act
in all directions" — sample-efficient generalisation demonstrated in **closed loop**,
not just one-step regression. Confidence ≈ **0.9** on the exactly-equivariant task.

---

## 4. Step 10 — external validity: closed-loop control on *real* PushT

Steps 8–9 proved the payoff on dynamics that are *exactly* SO(2)-equivariant **by
construction** (frozen random VN teacher / damped point mass). The honest question is
whether any of it survives a real, contact-rich simulator whose symmetry we do **not**
get to design. Step 10 tests this on PushT (push a T-block to a goal with a circular
pusher).

**A symmetry we did not build.** A probe establishes the key fact: real PushT's
*interior* agent↔block manipulation is *exactly* SO(2)-equivariant — rotating the whole
scene (pusher, T, velocities) and the action sequence by **any** angle about the arena
centre maps one real rollout onto another, to the float floor. Only block↔**wall**
contact breaks it (the square arena reduces SO(2) to $C_4$, and wall contact is
numerically stiff). So as long as the block stays in the interior, PushT is an *exactly*
SO(2)-equivariant system we did not construct — the right place to ask whether the prior
pays off.

### [A] The real system is exactly SO(2)-equivariant; the VN inherits it, the MLP does not

| quantity | value |
|---|---:|
| real env, interior manipulation, generic $37°$ rotation — max position residual | `1.8e-5` px |
| real env, push block into wall, same rotation — max position residual | `11.7` px |
| VN forward model, $\lvert M(Rs,Ra)-R\,M(s,a)\rvert$ (random $0.7$ rad) | `5.4e-7` |
| MLP forward model, same | `2.5e-1` |
| params | VN `3360`, MLP `18952` (**5.6×**) |

Interior manipulation is equivariant to $10^{-5}$ px at a generic angle; the wall breaks
it by $\sim 12$ px. The VN forward model is equivariant by construction ($10^{-7}$); the
param-matched-class MLP is not ($0.25$).

### [B] 举一反三 in PREDICTION — fit one wedge, test all orientations (REAL data)

Fit both models on 1500 real interior transitions from push tasks whose direction lies in
a $[0°,90°)$ wedge. Take **one** held-out test set and **rotate it** into each quadrant —
legitimate precisely because interior dynamics is exactly SO(2)-equivariant, so a rotated
real transition is another real transition. This isolates orientation while holding the
test difficulty *identical*.

| test orientation | VN relMSE | MLP relMSE |
|---|---:|---:|
| $[0°,90°)$ seen | `1.05e-2` | `1.66e-2` |
| $[90°,180°)$ | `1.05e-2` | `7.13e-2` |
| $[180°,270°)$ | `1.05e-2` | `2.69e-1` |
| $[270°,360°)$ | `1.05e-2` | `7.38e-2` |
| **OOD factor** | **`×1.00`** | `×16.2` |

The VN's error is **identical to five digits across all four quadrants** — fitting the
wedge *determines* the whole circle. The MLP fits the wedge (fair: it genuinely *can*
fit) but degrades **×16** out-of-distribution. This is 举一反三 at the prediction level
on a **real**, contact-rich simulator — the strongest external-validity evidence in the
project so far (Steps 8–9 were synthetic).

### [C] Closed-loop task success — an honest tie (noise-limited)

Learn the forward model on the $[0°,90°)$ wedge, then run **open-loop** CEM-MPC (plan
$H{=}20$, execute the whole plan with no per-step correction, so success depends on the
*model's* multi-step accuracy) on push tasks in all four quadrants; success = block within
$24$px of a goal $60$px away. Averaged over 2 seeds × 25 tasks/bin.

| push dir | VN succ | VN dist | MLP succ | MLP dist |
|---|---:|---:|---:|---:|
| $[0°,90°)$ seen | `0.40` | `34.1`px | `0.44` | `30.9`px |
| $[90°,180°)$ | `0.36` | `36.1`px | `0.38` | `34.0`px |
| $[180°,270°)$ | `0.56` | `25.2`px | `0.46` | `29.6`px |
| $[270°,360°)$ | `0.46` | `27.4`px | `0.44` | `29.4`px |
| **unseen mean** | `0.46` | `29.6`px | `0.43` | `31.0`px |

**This is a statistical tie, and I report it as one.** VN's OOD distance ratio comes out
$×0.87$ and MLP's $×1.00$ — but VN is *exactly* equivariant on an *exactly* equivariant
system, so its **true** OOD ratio is $1.00$; the observed $0.87$ (VN apparently doing
*better* OOD, which is impossible in expectation) is finite-sample noise, and across four
runs the ratio wobbled in $[0.87,1.02]$ with no consistent direction. Binary success is a
dead heat (VN $0.46$ vs MLP $0.43$ unseen, well within Bernoulli noise at $N{=}150$).

**Why the ×16 prediction gap does not convert.** The open-loop rollout is dominated by the
**agent's PD motion**, which is near-linear — the MLP extrapolates it fine OOD. The
component where equivariance actually bites, **block-contact dynamics**, is a small
fraction of each trajectory, and a position-only success threshold tolerates residual block
error. To surface the prediction advantage in closed loop one needs a **contact-dominated,
pose-controlled** task (tight orientation tolerance) — the concrete next experiment.

**Step 10 verdict.** The exact interior SO(2)-symmetry of real PushT is a genuine,
non-obvious finding [A]; the equivariant prior delivers **clean prediction-level 举一反三
on this real system** (×16 OOD, VN flat) [B]; but at laptop scale the advantage **does not
yet show up in closed-loop task success** on position-only pushing [C] — an honest null,
with a concrete mechanism and a concrete fix. Confidence ≈ **0.9** on [A]/[B] (prediction);
the closed-loop task-success question is **open**, not refuted.

---

## 5. Step 11 — the end-to-end equivariant *latent* JEPA (举一反三 in latent space)

Steps 8–10 all learned an **explicit-coordinate** forward model $M:(s,a)\mapsto s'$
that predicts the next *physical* state, and planned against a cost on the block's
pixel coordinates. That is a world model, but not the architecture the project is
named for. Step 11 builds the real thing — a **JEPA latent world model** (LeCun
2022; Bardes et al. V-JEPA 2024) — and asks the sharper question: when the encoder
*and* the predictor are equivariant, does the **learned representation** inherit the
symmetry, so that prediction and planning happen 举一反三 **in latent space**?

The model is *composed* from the modules built in Steps 5–6, nothing re-invented:

| part | equivariant (VN) | baseline (MLP) |
|------|------------------|----------------|
| encoder $E_\theta:\text{state}\to z\in\mathbb{R}^{128}$ | `StructuredStateEncoder` (exact continuous SO(2)) | `MLPStateEncoder` (no prior) |
| predictor $f_\phi:(z,a)\to z$ | `VNPredictor` (jointly equivariant) | `LatentPredictor` (residual MLP) |
| training | `train_jepa` — EMA-target + VICReg variance hinge + Muon/AdamW, **unchanged**, fed structured $(N,4,2)$ transitions | same |

State is four type-1 vectors $[\,\text{agent\_pos},\text{agent\_vel},\text{block\_pos},
\text{block\_dir}\,]$; the latent decomposes as $64$ stacked 2-vectors, so $\rho(g)$ is
block-diagonal $R(g)$ — orthogonal, which is exactly what makes the JEPA cost
$\mathcal{C}=\lVert E(s)-E(s_g)\rVert$ rotation-invariant. Both models train cleanly on
the $[0,90°)$ wedge (final latent std $1.31$ / $1.12$, no collapse; comparable
prediction MSE $\sim\!5\times10^{-3}$). The VN model uses **4.5× fewer parameters**
(37k vs 167k).

### [A] The learned latent is *exactly* SO(2)-equivariant — and stays so after training

The whole bet is that the symmetry survives optimisation, not just initialisation.
Equivariance residuals $\max\lVert\rho(g)\,(\cdot)-(\cdot)(g\cdot)\rVert_\infty$ at a
generic $0.7$ rad, and the cost-drift $\mathbb{E}\lvert\mathcal{C}(gs,gs_g)-\mathcal{C}(s,s_g)\rvert/\mathbb{E}\,\mathcal{C}$
at continuous angles:

| quantity | VN (init) | VN (post-train) | MLP (post-train) |
|----------|-----------|-----------------|------------------|
| composed encode→predict residual | $2.3\times10^{-6}$ | $2.9\times10^{-6}$ | $3.6$ |
| cost drift (37°/90°/153°/211°) | — | $\le 1.5\times10^{-7}$ | $0.40$–$0.62$ |

The equivariant JEPA's planning cost is invariant to a joint rotation of (state, goal)
to the float floor, for **every** continuous angle; the ordinary encoder's cost drifts
by 40–62%. This is the continuous-angle, latent-space analog of the pixel/$90°$
`fov_cost_drift` metric from Step 4 — and the structured path achieves at every angle
what the steerable-pixel encoder could only reach at $90°$ multiples.

### [B] 举一反三 in LATENT space — the decisive result

Train the latent dynamics on the $[0,90°)$ wedge; take one held-out interior set and
**rotate it into each quadrant** (legitimate: real interior PushT is exactly
SO(2)-equivariant, Step 10 [A]). Report the **latent** one-step error
$\lVert f_\phi(E(s),a)-E(s')\rVert^2/\lVert E(s')-E(s)\rVert^2$:

| orientation | VN latent relMSE | MLP latent relMSE |
|-------------|------------------|-------------------|
| $[0,90°)$ seen | $0.2559$ | $1.14$ |
| $[90,180°)$ | $0.2559$ | $4.01$ |
| $[180,270°)$ | $0.2559$ | $15.70$ |
| $[270,360°)$ | $0.2559$ | $2.64$ |
| **OOD ratio** | **×1.00** (flat) | **×13.8** (degrades) |

The VN latent error is **identical to five significant figures across all four
quadrants**: the equivariance theorem is realised end-to-end — rotating a transition
rotates numerator and denominator by the same orthogonal $\rho$, so the latent relMSE
*cannot* change. The baseline, fit on the wedge, degrades ×13.8 out of distribution.
(Honest note: compare *within* each model. The cross-model **absolute** relMSE differs
because the two latents have different step scales — the trained prediction MSE was in
fact comparable — so the decisive, scale-free claim is the **within-model OOD ratio**:
×1.00 vs ×13.8.)

### [C] Latent-space closed-loop planning — it works, OOD gap noise-limited

CEM-MPC against a **purely latent** terminal cost $\lVert\hat z_H-z_g\rVert^2$ (no
physical state inside the rollout, $z_g=E(s_g)$ the encoded goal state), open-loop
$H{=}20$, on real PushT, $2$ seeds × $15$ tasks/bin:

| orientation | VN succ / dist | MLP succ / dist |
|-------------|----------------|-----------------|
| $[0,90°)$ seen | $0.27$ / $36.7$px | $0.13$ / $39.7$px |
| $[90,180°)$ | $0.13$ / $42.5$px | $0.13$ / $41.1$px |
| $[180,270°)$ | $0.70$ / $24.3$px | $0.33$ / $31.7$px |
| $[270,360°)$ | $0.40$ / $31.4$px | $0.30$ / $35.0$px |

Two honest readings. (i) **The latent planner closes the loop**: planning entirely
through the learned latent cost drives the block from $60$px toward the goal (VN
averages $\sim\!34$px, one bin reaches $0.70$ success / $24$px) — the Phase-4
deliverable runs end-to-end, and the equivariant model edges the baseline in raw
success in 3 of 4 bins. (ii) **The OOD task gap is noise-limited**, exactly as Step 10
found: distance OOD-ratios VN ×0.89, MLP ×0.91 (the VN's *true* ratio is 1.00; the
deviation is finite-sample noise at $N{=}15{\times}2$). The ×14 *prediction* gap [B]
does not convert to a closed-loop *task* gap on position-only pushing.

**Step 11 verdict.** The project's central architectural claim is now demonstrated
end-to-end on real data: **an equivariant encoder + jointly-equivariant predictor
produce a learned latent world model that is exactly SO(2)-equivariant after
training** ($2.9\times10^{-6}$), so its planning cost is rotation-invariant
($1.5\times10^{-7}$) and **latent-space prediction generalises across the whole circle
from a single $90°$ wedge** (×1.00, vs the baseline's ×13.8). This is the Phase-4
thesis — "predict in an abstract, geometric latent space" — *realised*, not asserted.
Closed-loop *task success* remains the same honest open question as Step 10: the latent
planner works, but the OOD advantage is below the noise floor on position-only pushing.
Confidence ≈ **0.9** on the representation-level result ([A]+[B]); the closed-loop
task-success gap stays **open**.

---

## 6. Step 12 — the contact test: does the prediction gap convert under *pose* control?

Steps 10–11 ended on one honest open question. On real PushT the equivariant model
shows clean prediction-level 举一反三 (×16 OOD), but it did **not** convert to a
closed-loop *task-success* gap — twice, noise-limited. The diagnosis was mechanistic:
a position-only push is dominated by the agent's near-linear PD motion (which even the
non-equivariant MLP extrapolates fine OOD), while the **block-contact dynamics** — the
only place equivariance bites — is a small fraction of the trajectory and is tolerated
by a position-only threshold. Step 12 changes the regime to where the mechanism predicts
the gap should appear: a **contact-dominated reorientation task** — rotate the block to a
target angle $\theta_{\text{goal}}=\varphi+\Delta\theta$ ($|\Delta\theta|=35°$, only a
small translation), so the task metric depends on block-pose dynamics. Same Step-10
forward models (VN `3360` vs MLP `18952` params, **5.6×**), same wedge training; only the
task and the SO(2)-invariant pose cost
$\mathcal{C}=W_{\text{pos}}\lVert b_H-g\rVert^2 + W_{\text{ang}}\bigl(1-\langle d_H,g_{\text{dir}}\rangle\bigr)$
are new.

### [A] The pose cost is SO(2)-invariant; the VN keeps it so, the MLP drifts past 100%

| rotation | VN cost drift | MLP cost drift |
|---|---:|---:|
| $37°$ | $4.8\times10^{-7}$ | $0.45$ |
| $90°$ | $4.3\times10^{-7}$ | $0.97$ |
| $153°$ | $4.0\times10^{-7}$ | $1.05$ |
| $211°$ | $5.4\times10^{-7}$ | $1.06$ |

The equivariant rollout keeps the pose-planning cost invariant to the float floor at
**every** angle; the MLP's drifts by 45–106% (a drift $>100\%$ means the planned cost is
essentially decorrelated from the true rotated cost).

### [B] Decomposed prediction 举一反三 — *where* the OOD gap lives (decisive)

Fit on the $[0,90°)$ wedge, rotate one held-out test set into each quadrant, report
one-step relMSE **by state component** (pooled normalisation; $<1$ = usable, i.e. better
than predicting no-change, $>1$ = broken):

| component | VN (all 4 quadrants) | MLP seen | MLP worst-OOD |
|---|---:|---:|---:|
| `agent_pos` (self) | $9.6\times10^{-4}$ (flat ×1.00) | $1.8\times10^{-3}$ | $0.089$ (stays $\ll 1$) |
| `block_pos` (object) | $0.563$ (flat ×1.00) | $0.72$ | $1.21$ (×1.7) |
| `block_dir` (rotation) | $0.563$ (flat ×1.00) | $0.77$ | $2.33$ (×3.0) |

Two facts, both honest:
- **The VN is identical to five digits across all four quadrants on every channel**
  (×1.00) — exact equivariance realised. It is also *better in-distribution* on the block
  channels than the 5.6×-larger MLP ($0.563$ vs $0.77$): the prior fits contact dynamics
  more sample-efficiently (echoing Step 8).
- **OOD, the MLP keeps its self-motion model usable** (`agent_pos` $0.089\ll1$) **but its
  model of the block breaks** — `block_dir` crosses $1$ (worse than no-change) at $2.33$,
  worst in exactly the channel a pose task depends on. This *quantifies* the Step 10/11
  mechanism: the position-only task was carried by the agent channel the MLP retains; a
  pose task stresses the block-rotation channel it loses.

### [C] Closed-loop pose control — the first non-tie OOD signal

Receding-horizon CEM-MPC (2 seeds × 15 tasks/bin); continuous block **angle error** (deg)
is the headline (binary success is noisy at this $N$):

| orientation | VN angle | MLP angle |
|---|---:|---:|
| $[0,90°)$ seen | $5.2°$ | $11.8°$ |
| $[90,180°)$ | $5.9°$ | $13.4°$ |
| $[180,270°)$ | $4.5°$ | $27.6°$ |
| $[270,360°)$ | $6.7°$ | $24.4°$ |
| **OOD ratio** | **×1.09** (flat) | **×1.85** (degrades) |

For the first time in the project, the closed-loop OOD comparison is **not a noise-limited
tie**. The equivariant planner holds block-orientation error at $\sim 5\text{–}6°$ across
the *entire circle* (flat, ×1.09 ≈ the true $1.00$), while the MLP degrades from $11.8°$
(seen) to $\sim 22°$ (unseen, ×1.85). The contact-dominated task surfaced the gap the
position-only task hid.

Honest caveats. (i) Part of the VN's *seen*-quadrant angle advantage ($5.2°$ vs $11.8°$)
is better in-distribution fit (prior → sample efficiency), so the clean **equivariance**
signal is the OOD *ratio* (×1.09 vs ×1.85), not the absolute level. (ii) Binary
combined-pose success (angle $<18°$ **and** position $<24$px) stays low for both (VN
$\le 0.23$, MLP $\le 0.13$): the task is genuinely hard at laptop scale, and the
angle-weighted planner lets the VN trade position error ($32\text{–}49$px) to minimise
rotation. So this is a **control-relevant angle-error signal**, not a clean task-success
sweep. (iii) $N=15\times2$/bin is small.

**Step 12 verdict.** The Step 10/11 open question is answered at the mechanism level and
*partially* at the control level. [A]+[B] are decisive: the OOD gap lives **specifically
in the block-rotation channel** (`block_dir` relMSE $0.77\to2.33$ for the MLP; $0.56$ flat
for the VN), exactly where equivariance bites and exactly what a pose task needs. [C]
converts this into the **first closed-loop OOD signal that isn't a tie** — equivariant
orientation control flat across the circle (×1.09) vs the baseline degrading (×1.85) —
though not into a clean binary task-success win at laptop scale. Confidence ≈ **0.9** on
[A]+[B]; ≈ **0.6** on the [C] angle-control signal (right direction, modest $N$, an
in-distribution-fit confound on the absolute level).

---

## 7. Step 13 — the SO(3) lift: does end-to-end latent 举一反三 survive one dimension up?

Steps 10–12 all live in 2D / SO(2) on PushT. But the thesis is about *geometry*, and the
architecture the project is actually building (CLAUDE.md Phase 4) is **SE(3)** on 3D point
clouds. Step 6 proved the SE(3) encoder + VN predictor equivariant **at init on random
data** — necessary but not sufficient: it says nothing about whether a *trained* 3D latent
world model keeps the symmetry, nor whether 举一反三 holds across the much larger group
SO(3) (a 2-sphere of axes $\times$ an angle, not a single circle). Step 13 runs the
**Step-11 protocol one dimension up**: train the end-to-end latent JEPA — `SE3PointEncoder`
$E$ composed with `VNPredictor(dim=3)` $f$, planning **in the learned latent** — on 3D
clouds, add a non-equivariant baseline (flatten-MLP encoder + MLP predictor), and test
generalisation from a restricted training wedge to the whole of $\mathrm{SO}(3)$.

**Exactly-SO(3)-equivariant teacher.** No laptop-scale 3D simulator is *provably*
equivariant, so (as in Steps 8–9) the ground-truth dynamics is a synthetic in-class map. For
a centred cloud $\tilde x_i = x_i-\bar x$ with unit directions $\hat u_i=\tilde x_i/\lVert\tilde x_i\rVert$
and a type-1 action $a\in\mathbb{R}^3$,
$$ x_i' = x_i + \underbrace{c_t\,a}_{\text{drift}} + \underbrace{c_r\,(a\times\tilde x_i)}_{\text{torque}} + \underbrace{c_d\,\langle a,\hat u_i\rangle\,\hat u_i}_{\text{stretch}}, \qquad (c_t,c_r,c_d)=(0.15,\,0.15,\,0.08). $$
Each term is SO(3)-equivariant: $Ra\times R\tilde x = R(a\times\tilde x)$ for proper
rotations, and $\langle a,\hat u\rangle$ is invariant. The **torque** term $a\times\tilde x_i$
is the 3D analogue of PushT's block-rotation channel — the place equivariance bites; the
**drift** $c_t a$ is the easy near-linear "self-motion" channel a non-equivariant net
extrapolates fine. (The cross product is only SO(3)- not O(3)-equivariant; the VN is
O(3)-equivariant by construction, and we test only $\mathrm{SO}(3)\subset\mathrm{O}(3)$, so
the model class genuinely contains the teacher.)

**Anisotropy + restricted wedge** (the Step-8 condition, without which OOD is meaningless).
The template is an **anisotropic** 24-point cloud (per-axis scale $[1.0,0.55,0.3]$, no
rotational symmetry), per-sample jittered and axis-scaled, then rotated **only within a
$z$-axis wedge $\varphi\in[0,90°)$** for training. The OOD test rotates held-out transitions
by **full random $R\in\mathrm{SO}(3)$** — new axes *and* angles the wedge never showed.

### [A'] Equivariance survives training; the planning cost is rotation-invariant

| quantity | VN (equivariant) | MLP (baseline) |
|---|---:|---:|
| composed residual, **at init** | $7.3\times10^{-6}$ | $2.95$ |
| composed residual, **after 60 epochs** | $3.0\times10^{-5}$ | $4.30$ |
| JEPA cost drift, random SO(3) (max) | $7.2\times10^{-7}$ | $0.85$ |
| parameters | $16{,}856$ | $124{,}512$ (**7.4×**) |

The learned 3D latent keeps the exact symmetry through optimisation (composed residual at
the float floor, $3.0\times10^{-5}$), so the JEPA planning cost
$\mathcal{C}=\lVert\hat z_H-z_g\rVert^2$ is rotation-invariant to $\sim10^{-7}$ under random
SO(3) while the baseline's cost decorrelates (drift up to $0.85$) — and the equivariant model
does it with **7.4× fewer parameters**.

### [B] Latent prediction 举一反三 across SO(3) (decisive)

One-step **latent** relMSE on the *same* held-out set rotated into each orientation bin
(pooled normalisation; $<1$ usable, i.e. beats predicting no latent change, $>1$ broken):

| orientation bin | VN relMSE | MLP relMSE |
|---|---:|---:|
| $z\,45°$ (seen wedge) | $0.228$ | $0.307$ |
| $z\,180°$ (OOD angle) | $0.228$ | $2.63$ |
| $x\,90°$ (OOD axis) | $0.228$ | $5.28$ |
| $y\,90°$ (OOD axis) | $0.228$ | $1.03$ |
| random SO(3) ×8 | $0.228$ | $1.57$ |
| **OOD / seen** | **×1.00** (flat) | **×17.2** |

The VN is **flat to four digits across the entire group** — same axis/new angle, brand-new
axes, random SO(3) — exact 举一反三; and it is also **better in-distribution** than the
7.4×-larger baseline ($0.228$ vs $0.307$: the prior fits the dynamics more sample-efficiently,
echoing Steps 8/12). The MLP fits the seen wedge ($0.307$) but **breaks OOD**, crossing $1$
(worse than no-change) and peaking at $5.28$ — and its worst bins are the **new-axis**
rotations ($x\,90°$), exactly the directions the $z$-wedge never exercised. This is Step 11
reproduced one dimension up, in a strictly larger group.

### [C] Latent closed-loop planning to a goal cloud (honest negative)

CEM in the learned latent, executed on the (equivariant) teacher as ground-truth env;
fraction of the start→goal gap closed ($1$ = reached, $0$ = no progress):

| model | seen (identity) | OOD random SO(3) | OOD / seen |
|---|---:|---:|---:|
| VN | $-0.61$ | $-0.64$ | ×$-1.04$ (flat) |
| MLP | $-1.23$ | $-2.06$ | ×$-1.68$ |

Honest read: **purely-latent planning gets no cloud-space traction here** — both models post
*negative* frac-closed (they nudge the cloud away from the goal). This is the same limitation
Step 11 flagged for the purely-latent planner, **not** an equivariance failure: tellingly,
even in failure the VN's OOD/seen ratio is essentially flat (×$-1.04$ — it fails *identically*
across the group, as exact invariance demands), while the baseline's degrades (×$-1.68$). A
useful 3D latent-only planner needs a decoder or a cloud-space cost; that is future work, not
a result I will dress up.

**Step 13 verdict.** The end-to-end SO(3) point-cloud latent JEPA **works at the level the
thesis claims**: the *learned* 3D latent inherits exact SO(3) equivariance after training
($3.0\times10^{-5}$), its planning cost is rotation-invariant ($10^{-7}$ vs the baseline's
$0.85$), and latent prediction is 举一反三 across the whole group from a single $z$-wedge
(VN flat ×1.00; MLP ×17.2, worst on new axes) — with **7.4× fewer parameters** *and* a better
in-distribution fit. This is the Steps 10–11 mechanism confirmed **in 3D / SO(3)**, the
project's actual target geometry. The honest negative is [C]: purely-latent planning toward a
goal *cloud* gets no traction for either model — a planner/decoder limitation, not an
equivariance one (the VN still fails flat across the group). Confidence ≈ **0.9** on [A']+[B]
(exact, decisive); the [C] latent planner is an acknowledged gap.

---

## 8. Step 14 — the paired power test: converting the prediction gap into an *exact* closed-loop result

Step 12 [C] gave the first closed-loop OOD signal that wasn't a tie, but it was an
*unpaired* comparison (2 seeds × 15 tasks/bin) with two honest weaknesses: the absolute
angle level carried an in-distribution-fit confound, and task-to-task difficulty variance —
different blocks, goals, contact geometries — is large enough that Steps 10–12 kept landing
"within noise." Step 14 removes both by exploiting the exact symmetry as an *experimental
design*, not just a model property.

**The paired design.** Because real *interior* PushT is **exactly** SO(2)-equivariant
(Step 10 [A]: $1.8\times10^{-5}$ px at a generic angle), rotating an *entire reorientation
task* — state, goal position $g$, goal angle $\theta_{\text{goal}}$, and scene orientation
$\varphi$ — by any $\Delta$ produces **another valid real task at $\varphi+\Delta$ with
identical intrinsic difficulty**. So we sample $K=48$ base tasks in the seen wedge and
evaluate the *same* base task at $\Delta=0$ (seen) and at four OOD rotations
$\Delta\in\{90°,150°,210°,270°\}$, holding the **env seed and the CEM seed fixed across
orientations**. Only the global rotation changes. The paired difference
$$ d_i \;=\; \text{ang}_{\text{OOD}}(i) \;-\; \text{ang}_{\text{seen}}(i) $$
cancels the per-task variance that washed out the unpaired comparisons, and a bootstrap CI
over the $K$ tasks tests whether OOD control degrades. Same Step-10 forward models (VN
`3360` vs MLP `18952` params, **5.6×**); trained-model equivariance VN $6.4\times10^{-7}$
vs MLP $0.51$; success defined as angle $<18°$ **and** position $<24$px.

### [E] EXACT — a rotation-equivariant planner makes the prior the *sole* variable

The Step-12 planner is **not** itself rotation-equivariant at generic angles: the box action
constraint $a\in[-1,1]^2$ is only dihedral- ($C_4$-)symmetric, and a diagonal per-component
$\sigma$ refit does not commute with $R_\alpha$. Panel [E] replaces both with an *equivariant*
CEM: an **isotropic** $\sigma$ (pooling the two spatial components makes the variance
rotation-invariant, $\sum_c (R v)_c^2=\lVert v\rVert^2$), exploration noise **pre-rotated** by
$R(\Delta)$, and a **disk** constraint $\lVert a\rVert\le 1$ (rotation-equivariant). This
planner is *identical for both models*, so the only thing that can differ across orientations
is the **model's symmetry prior**. For the exactly-equivariant VN this forces the closed-loop
trajectory at orientation $\Delta$ to be *exactly* $R(\Delta)$ applied to the seen trajectory —
so the block-angle error must be identical task-by-task, to the float floor.

| orientation | VN angle | MLP angle |
|---|---:|---:|
| seen ($\Delta=0$) | $7.28°$ | $20.41°$ |
| $+90°$ | $7.28°$ | $17.90°$ |
| $+150°$ | $7.28°$ | $24.75°$ |
| $+210°$ | $7.28°$ | $30.49°$ |
| $+270°$ | $7.28°$ | $23.20°$ |

| paired OOD$-$seen (deg), 95% bootstrap CI over $K{=}48$ | mean | 95% CI |
|---|---:|---:|
| **VN** | $-0.000$ | $[-0.000,\,+0.000]$ ($\max_i\lvert d_i\rvert=4.9\times10^{-5}$) |
| **MLP** | $+3.681$ | $[+1.488,\,+6.015]$ (excludes 0) |

The VN's paired difference is **zero to the environment float floor**
($\max_i\lvert d_i\rvert=4.9\times10^{-5}$ deg): every one of the 48 tasks produces the
*identical* angle error seen and OOD — the SO(2) theorem realised end-to-end in closed loop,
not statistically but **exactly**. The OOD/seen ratio is $1.000$, CI $[1.000,1.000]$. The MLP,
on the *same* equivariant planner, degrades by $+3.68°$ with a CI that **excludes zero**
(ratio $1.180$, CI $[1.059,1.367]$). With the planner held equivariant for both, the only
explanation for the split is the model's prior.

### [S] DIAGNOSTIC — the verbatim Step-12 planner (not equivariant at generic angles)

Re-running the paired test with the **unmodified** Step-12 planner (box clamp + diagonal
$\sigma$) is a diagnostic, not the headline:

| paired OOD$-$seen (deg), 95% CI over $K{=}48$ | mean | 95% CI |
|---|---:|---:|
| **VN** | $-0.709$ | $[-2.762,\,+1.007]$ (brackets 0; $\max_i\lvert d_i\rvert=34.3$) |
| **MLP** | $+3.742$ | $[+1.462,\,+6.051]$ (excludes 0) |

Two findings. (i) The MLP **still degrades** (CI excludes 0, $+3.74°$) — the separation is
robust to the planner. (ii) The VN's paired difference is now *small but no longer exactly
zero* (mean $-0.71°$, and individual $\lvert d_i\rvert$ up to $34°$), even though the *model*
is exactly equivariant — because the **planner** breaks the symmetry the model preserves at
generic angles. Its CI still **brackets 0** (the residual is unbiased), so the statistical
conclusion survives, but the contrast with [E] is the real lesson: **closed-loop
orientation-invariance requires both an equivariant model *and* an equivariant planner.** That
is precisely why Steps 10–12, run on a non-equivariant planner, were noise-limited in closed
loop — the missing half was the controller, not the model.

**Step 14 verdict.** The prediction-level OOD gap (VN flat, MLP ×13–17; Steps 10–13) **does**
convert to a closed-loop statement once the planner is also equivariant: on the
exactly-SO(2) PushT interior, an equivariant model + equivariant planner closes the pose loop
with a block-angle error **invariant to global reorientation to the float floor** (VN paired
diff $=4.9\times10^{-5}$ deg over 48 tasks), while the non-equivariant model degrades with a
CI excluding 0 ($+3.68°$, $[+1.49,+6.02]$). The paired design removed the task variance that
left Steps 10–12 within noise. Honest scope: [E] is a **controlled-planner** result (the
decisive one — it isolates the prior); [S] shows that with a generic-angle-broken planner the
VN's exactness degrades to a still-unbiased statistical tie, i.e. closed-loop invariance is a
property of the model **and** planner together. Confidence ≈ **0.9** on [E] (exact, paired,
$K{=}48$), ≈ **0.85** on [S] (the model/planner-jointly-equivariant finding).

---

## 9. Step 15 — completing the group: SE(3) $=$ SO(3) $\ltimes\ \mathbb{R}^3$ (translation 举一反三)

The project is named for **SE(3)**, but every generalisation test so far isolated the
*rotation* subgroup: Steps 10–12 are SO(2), Step 13 is SO(3). Translation — the other half of
$g=(R,t)$ acting by $x\mapsto Rx+t$ — was never the OOD axis. Step 15 closes that gap with the
*same* Step-13 pipeline (same encoders, teacher, recipe, latent relMSE metric), and is honest
that the two halves are earned differently:

* **Rotation is *learned*** equivariance — the e3nn `SE3PointEncoder` maps a global $R$ to the
  block-diagonal $\rho(R)$ on the latent, and that survives training (this is the non-trivial half).
* **Translation is *exact by construction*** — the encoder **centres** the cloud
  ($r_i=x_i-\bar x$), so $E(x+t)=E(x)$ *identically*. The teacher centres internally, so it is
  translation-*equivariant* ($\mathrm{Dyn}(x+t,a)=\mathrm{Dyn}(x,a)+t$). A translated transition
  therefore has the **same** latent, the **same** predicted latent, and the **same** next latent —
  the latent relMSE is unchanged to the float floor. We do not oversell this as a deep result; it
  is geometry done right, and it is exactly what makes the *full* group a no-cost generalisation.

It is a real test, not a vacuous one: training clouds sit near the origin (template $+$ jitter,
rotated only in a $+z$ wedge, **never translated**), while the baseline `MLPPointEncoder` flattens
**raw** coordinates, so a large test-time translation pushes its inputs out of their trained range.

### [A] SE(3) mechanism after training

| residual (after a real Muon/AdamW + EMA training run) | VN | MLP |
|---|---:|---:|
| translation-invariance $\max\lvert E(x+t)-E(x)\rvert$, $\lvert t\rvert$ small | $3.6\times10^{-5}$ | $4.04$ |
| translation-invariance, $\lvert t\rvert$ large | $5.3\times10^{-5}$ | $17.39$ |
| composed rotation $\max\lvert\rho(R)f(E x,a)-f(E(Rx),Ra)\rvert$ | $3.0\times10^{-5}$ | $4.30$ |

(teacher translation-equivariance residual $1.9\times10^{-6}$ — it commutes with translation, so the
target is well-defined.) The VN is translation-invariant *and* rotation-equivariant to the float
floor; the raw-coordinate MLP is sensitive to both.

### [B] Latent 举一反三 across an SE(3) ladder (decisive)

Same held-out set, mapped by each SE(3) element; latent 1-step relMSE:

| SE(3) transform | VN relMSE | MLP relMSE |
|---|---:|---:|
| identity (seen) | $0.228$ | $0.120$ |
| translate small | $0.228$ | $2.40$ |
| translate **large** | $0.228$ | $4.57$ |
| rotate SO(3) only | $0.228$ | $0.144$ |
| translate $+$ SO(3) | $0.228$ | $4.48$ |
| translate $+$ SO(3) (2) | $0.228$ | $18.85$ |

The VN is **flat to four digits** ($0.228$ on every bin including the worst composition, OOD/seen
$=1.00$) while the baseline degrades up to **×157** (seen $0.120$ → worst OOD $18.85$), at **7.4×
fewer parameters** (VN `16856` vs MLP `124512`). Two honest readings: (i) the unconstrained MLP fits
the *seen* set slightly *better* ($0.120$ vs the VN's $0.228$) — the classic inductive-bias trade, a
little in-distribution fit for exact across-group invariance; (ii) the MLP's break here is driven by
**translation and composition** (raw-coordinate range explodes), and it partially tolerates the one
benign rotation `R_a` ($0.144$) — though Step 13's harder *multi*-rotation OOD broke it ×17. The
headline is unchanged: the equivariant latent world model is **flat across the whole of SE(3)**,
closing the gap between the project's named target geometry and what had been tested. Confidence ≈
**0.9** on [B]; the translation half is exact-by-centering (architectural), the rotation half learned.

---

## 10. Step 16 — robustness sweep: how much symmetry-breaking can the prior tolerate? (the Bitter-Lesson boundary)

**Honest scoping of "Task 4."** The original Phase-4 plan named a *real 3D manipulation simulator*
(ManiSkill / RLBench) as the next validation. Those renderers need CUDA / EGL and **do not run on
this CPU-only Mac** — a genuine platform blocker, stated plainly, not worked around. Rather than fake
a 3D-sim result, Step 16 answers the question that actually *load-bears* on the whole thesis and that
the laptop **can** settle decisively: a hard symmetry prior helps when the world *has* the symmetry —
but real worlds only *approximately* do, so **how much symmetry-breaking can the SO(3) prior absorb
before the unconstrained model catches up?** This is Sutton's Bitter-Lesson tension made quantitative.

**Design.** Break the exactly-SO(3) Step-13 teacher $\mathrm{Dyn}_0$ with a fixed lab-axis,
gravity-like term controlled by a knob $g$:
$$ \mathrm{Dyn}_g(x,a)_i \;=\; \mathrm{Dyn}_0(x,a)_i \;-\; g\,\bigl(e_z\!\cdot\!\tilde x_i\bigr)\,e_z,
\qquad \tilde x_i = x_i-\bar x. $$
This term is chosen so that (a) it **survives centering** — $\sum_i \tilde x_i = 0$, so it adds
nothing to the centroid and is a genuinely *visible* target, **not** a disguised translation; and
(b) it is **not** SO(3)-equivariant — the fixed lab axis $e_z$ does not commute with rotation. At
$g=0$ it recovers the exact teacher. We quantify the broken fraction by
$$ \mathrm{noneq}(g) \;=\; \frac{\sum\lVert \mathrm{Dyn}_g(Rx,Ra)-R\,\mathrm{Dyn}_g(x,a)\rVert^2}
        {\sum\lVert \mathrm{Dyn}_g(x,a)-x\rVert^2}, $$
the share of the dynamics that violates the symmetry. **Method point that matters:** OOD is
**re-sampled at full SO(3) and pushed through the true $\mathrm{Dyn}_g$** — *not* the
rotate-a-seen-target trick, which manufactures a fake label the moment the teacher stops being
equivariant.

A **12-point** grid (each point seeded independently of grid position, so the six values present in
the earlier 6-point run reproduce bit-for-bit; the new points only fill in and extend the curve):

| $g$ | noneq frac | VN seen | VN OOD | MLP seen | MLP OOD | winner OOD |
|---:|---:|---:|---:|---:|---:|:--:|
| $0.000$ | $\approx 0$ | $0.268$ | $0.301$ | $0.181$ | $1.893$ | VN |
| $0.025$ | $0.009$ | $0.255$ | $0.334$ | $0.263$ | $1.671$ | VN |
| $0.050$ | $0.034$ | $0.315$ | $0.453$ | $0.279$ | $2.423$ | VN |
| $0.100$ | $0.126$ | $0.306$ | $0.614$ | $0.302$ | $2.430$ | VN |
| $0.150$ | $0.256$ | $0.384$ | $0.769$ | $0.313$ | $1.661$ | VN |
| $0.200$ | $0.402$ | $0.369$ | $0.772$ | $0.261$ | $1.461$ | VN |
| $0.300$ | $0.676$ | $0.386$ | $0.815$ | $0.168$ | $1.535$ | VN |
| $0.400$ | $0.888$ | $0.382$ | $0.836$ | $0.168$ | $1.784$ | VN |
| $0.600$ | $1.143$ | $0.411$ | $0.879$ | $0.276$ | $1.342$ | VN |
| $0.800$ | $1.270$ | $0.350$ | $0.938$ | $0.282$ | $1.612$ | VN |
| $1.200$ | $1.380$ | $0.191$ | $0.864$ | $0.269$ | $1.790$ | VN |
| $1.600$ | $1.422$ | $0.152$ | $0.896$ | $0.335$ | $1.457$ | VN |

Two things happen, and both are honest. (i) **The prior is not free once the world breaks the
symmetry:** the VN's OOD relMSE climbs $\approx\!\times3$ ($0.30\to0.94$) as $g$ grows and then
**saturates** in the $0.85$–$0.94$ band — the equivariant model pays a *bounded* price for the part
of the dynamics it structurally *cannot* represent (the un-representable fixed-axis residual does not
keep growing once it dominates). (ii) **Yet the SO(3) prior still wins OOD at all 12 points:** VN OOD
stays below MLP OOD throughout — even at the largest break $g=1.6$, where $\mathrm{noneq}=1.42$ means
the symmetry-breaking component *exceeds* the equivariant one in norm (the dynamics is well past
"half non-symmetric"), the VN's $0.90$ still beats the MLP's $1.46$. There is **no crossover inside
the tested range**, because the MLP's failure mode (no SO(3) OOD generalisation *at all* — already
×6 broken at $g=0$) is worse than the VN's failure mode (a structured model that is merely
*mis*specified).

**Verdict (deliberately bracketed, not over-claimed).** This *brackets* the Bitter-Lesson boundary
rather than pinpointing it: the hard prior is robust to **substantial** misspecification — still
ahead even when the broken component is $\approx\!1.4\times$ the symmetric one — but its OOD margin
shrinks as the world's symmetry erodes, exactly as theory predicts. We do **not** claim the prior
always wins; we show it tolerates more misspecification than one might fear, and we push the bracket
out to $\mathrm{noneq}\approx1.42$ without finding the crossover at this scale. Confidence ≈ **0.85**;
the real-3D-sim validation that "Task 4" named remains genuine future work, gated on GPU hardware.

---

## 11. Step 17 — the training-seed error bar (multi-seed closed-loop OOD degradation)

Step 12 [C] and Step 14 both reported the closed-loop OOD contrast from **one** trained VN and
**one** trained MLP (Step 14 added a paired bootstrap over $K{=}48$ *tasks*, but still on a single
model per architecture). The remaining publishability gap is **training-seed variance**: is "VN flat,
MLP degrades" a property of the *architecture*, or an artefact of the lucky seed-0 weights? Step 17
trains $K=5$ **independent** $(\text{VN},\text{MLP})$ pairs — each with its own data seed *and*
optimisation seed — runs the **verbatim** Step-12 receding-horizon CEM closed loop on real PushT for
every one, and reports the *distribution* across seeds. Nothing about the planner changes; only the
trained weights vary.

**Metric (honest).** The headline is the degree degradation
$\Delta = \overline{\text{ang}}_{\text{unseen}}-\overline{\text{ang}}_{\text{seen}}$ (Step-14-aligned,
robust when the seen error is small). The ratio unseen/seen is kept only as a *noisy secondary*
readout — it inflates when the seen denominator is tiny (one seed had VN seen $=2.0°$ → ratio $5.45$
noise, while the absolute angles told the true story).

| across 5 training seeds | mean $\Delta$ (deg) | 95% CI (normal) | absolute OOD angle |
|---|---:|---:|---:|
| **VN** | $-0.97 \pm 1.64$ | $[-2.41,\ +0.47]$ (**straddles 0**) | $10.0° \pm 2.9°$ |
| **MLP** | $+9.57 \pm 4.01$ | $[+6.05,\ +13.08]$ (**excludes 0**) | $23.2° \pm 1.6°$ |

The two confidence intervals **do not overlap**. Per-seed, *every* VN $\Delta$ is near-zero or
negative $\{-0.3,+0.3,-0.1,-3.8,-1.0\}$ while *every* MLP $\Delta$ is robustly positive
$\{+10.3,+10.7,+7.1,+4.5,+15.2\}$ — the same qualitative split in all five independent draws — and the
VN reaches unseen orientations more than **2× more accurately** in absolute terms ($10.0°$ vs
$23.2°$). So the closed-loop contrast is a property of the **architecture, not the seed**. Honest
scope: this uses the verbatim Step-12 planner (not equivariant at generic angles), so the VN's small
residual is **planner**-induced — Step 14 [E], with an equivariant planner, drives the paired
difference to the float floor; Step 17's distinct contribution is the *training-seed* error bar that
Steps 12/14 did not provide. Pair it with Step 14's $K{=}48$ task-variance bootstrap for the full
statistical picture. Confidence ≈ **0.85**.

---

## 12. Step 18 — the SE(3) closed-loop lift (the [C]/[E] theorem in 3D)

Step 14 made the closed-loop orientation invariance **exact** — but in **2D / SO(2)**, on PushT.
The project's *named* geometry is **SE(3)**. Step 18 lifts the entire [E]/[S] paired design to **3D
point clouds under the full SE(3) group** (rotation **and** translation), on the end-to-end latent
JEPA of Step 13 (`SE3PointEncoder` + `VNPredictor(dim=3)`, planning in the learned latent). The
theorem being tested: under an SE(3)-equivariant world model **and** a matching SE(3)-equivariant
planner,
$$\mathrm{plan}\big(Rx_0+t,\;Rx_g+t\big)=R\cdot\mathrm{plan}(x_0,x_g),$$
so the realised closed-loop trajectory at a globally transformed goal $(R,t)$ is $(R,t)$ applied to
the canonical trajectory, and the Kabsch orientation error $\theta$ (and centroid distance) is
**invariant** across the group. $K{=}24$ paired base tasks (goals reorient $30.7°$ on average), each
run on an orbit of $1$ seen $+ 4$ OOD $(R,t)$ with $\lvert t\rvert\!\sim\!0.8$ and the **same** per-task
CEM seed; closed loop $T_{\max}{=}18$, replan every $6$, centroid weight $w_t{=}0.5$. VN $16{,}856$
params vs MLP $124{,}512$ (**7.4×**); post-train composed equivariance VN $6.1\times10^{-6}$ vs MLP
$5.61$; latent std VN $0.569$ / MLP $1.205$ (no collapse).

**Honest scope of "exact" in 3D — read this before the tables.** Unlike 2D (where real interior
PushT is SO(2)-equivariant to $1.8\times10^{-5}$ px and Step 14 hit $\max_i\lvert d_i\rvert
=4.9\times10^{-5}°$ — *literally* the float floor), the 3D VN is equivariant only to **e3nn's
architectural floor $\sim\!1.2\times10^{-6}$**. This is **not** a float32 precision issue: it barely
improves under float64 ($1.755\times10^{-6}\to1.233\times10^{-6}$). It is the standard, accepted notion
of "exact equivariance" for TFN/NequIP-style nets — every encoder op is clean $\sim\!10^{-7}$ in
e3nn's own irrep basis, but the change-of-basis back to plain $(x,y,z)$ leaves e3nn's internal
Wigner/normalisation constants as a $\sim\!10^{-6}$ residual scaled by the output magnitude. The
**predictor is exact** ($\sim\!8.8\times10^{-9}$) and the **single plan commutes to $1.2\times10^{-7}$**
(`tests/test_planner_equivariance.py` — the clean theorem demonstration). What the receding-horizon
loop does is *occasionally amplify* that $\sim\!10^{-6}$ into a CEM top-$k$ tie-flip at the
$n_{\text{elite}}{=}25/n_{\text{samples}}{=}256$ boundary, compounding to a few degrees on a handful of
tasks. So the decisive [E] statistic in 3D is **not** "zero to the float floor" but the **OOD/seen
orientation-error ratio**.

**[E] EXACT — equivariant planner (iso-$\sigma$, unit-**ball** clamp, $R$-rotated noise, latent +
closed-form centroid cost), held *identical* for both models:**

| over $K{=}24$ paired tasks | OOD/seen ratio | 95% CI | paired OOD$-$seen angle (deg) |
|---|---:|---:|---:|
| **VN (equivariant)** | $0.989$ | $[0.977,\ 0.999]$ (within $2\%$ of flat) | $-0.27$, CI $[-0.63,-0.03]$, $\max_i\lvert d_i\rvert=3.54$ |
| **MLP (baseline)** | $1.134$ | $[1.049,\ 1.234]$ (**excludes 1**) | $+9.92$, CI $[+3.80,+15.94]$ |

The two ratio CIs are **disjoint** ($0.999<1.049$). The VN's deviation is *negative* (OOD marginally
*better* than seen) and tiny — a tie-flip floor, not a degradation; the MLP's is $+13\%$ and climbing.
By group element the VN orientation error is essentially flat — $\{25.86,25.86,25.86,25.15,25.49\}°$
across $\{$seen$,g_1,g_2,g_3,g_4\}$ (the first three identical to $10^{-6}$: they share the
pure-rotation orbit; $g_3,g_4$ carry the large translation, and the small wobble there is the
tie-flip floor) — while the MLP swings $\{74,84,59,91,102\}°$. VN centroid position error is flat
$\{0.532,0.532,0.532,0.520,0.524\}$.

**Translation, honestly.** `SE3PointEncoder` is translation-**invariant** (it centres the cloud), so a
pure-latent cost is translation-blind and SE(3) would silently collapse to SO(3). The fix is a separate
**closed-form centroid channel**: a terminal cost $\lVert \bar x_0+C_T\sum_h a_h-\bar x_g\rVert^2$ that
is *exactly* SE(3)-invariant by construction (drift-only — it ignores the stretch's centroid
contribution, an approximation that costs control quality, not the theorem). Same ledger as Step 15:
**SO(3) learned** (latent, survives training to $6.1\times10^{-6}$), **translation exact**
(network-independent centroid arithmetic).

**[S] DIAGNOSTIC — verbatim Step 13 planner (box clamp, diagonal $\sigma$, latent-only cost).** Swap
the equivariant planner back for the generic one and the VN's deviation from flat *grows* — ratio
$0.886$, CI $[0.825,0.954]$, $\max_i\lvert d_i\rvert=16.0°$ (mean $-3.63°$, CI $[-6.07,-1.31]$),
$\sim\!5\times$ the [E] residual — while the MLP degrades further (ratio $1.251$, CI $[1.163,1.361]$).
Exactly as in 2D Step 14 [S]: **closed-loop SE(3)-invariance is a property of the model *and* the
planner together** — the model preserving the symmetry is necessary but not sufficient; a
non-equivariant planner (a box clamp that is only $C_4$/octahedral-symmetric, a per-component $\sigma$
refit that does not commute with $R$) re-injects the asymmetry the model removed.

**Verdict — all four guards green:** model-equiv (VN composed $6.1\times10^{-6}\!<\!10^{-4}$) ✓;
VN-flat (ratio CI upper $0.999<1.05$) ✓; MLP-degrades (ratio CI lower $1.049>1$) ✓; ratio-CIs-disjoint
($0.999<1.049$) ✓. **PASS.** Confidence ≈ **0.85** on the SE(3) closed-loop [E] — one notch below the
2D Step 14 [E]'s $0.9$, precisely because the VN residual is a CEM **tie-flip floor** at the e3nn
$\sim\!10^{-6}$ equivariance, not the literal float zero 2D achieved (the $1.2\times10^{-7}$
single-plan unit test is the clean theorem; the closed loop is the realistic one) — and ≈ **0.85** on
the model-and-planner [S] finding, mirroring Step 14.

---

## 13. Step 19 — object-centric compositionality: which prior buys which generalisation? ($\mathrm{SE}(3)^O\rtimes S_O$)

Steps 13–18 proved the claims for a **single** rigid body under $\mathrm{SE}(3)$. The world has *many*
objects, and CLAUDE.md Open Question #3 asks the next thing directly: *how do compositional /
object-centric abstractions emerge in equivariant latent world models?* A scene of $O$ objects carries
a strictly larger symmetry — $\mathrm{SE}(3)^{O}\rtimes S_O$, per-object rigid motions **and** object
relabelings — assembled from **two logically independent** architectural priors, and the whole point of
Step 19 is to refuse to conflate them:

1. **Factorization** (shared-weight per-object *slots*). Alone this buys three *exact* properties:
   **permutation-equivariance** $E(\sigma\!\cdot\!S)=\sigma\!\cdot\!E(S)$ for $\sigma\in S_O$,
   **leakage-freedom** (object $i$'s latent block is independent of object $j$'s state), and — with a
   centred per-object encoder — **arrangement-invariance** (the per-object latent ignores *where* the
   object sits).
2. **Per-object $\mathrm{SE}(3)$-equivariance**. This buys **orientation 举一反三**: a per-object
   reorientation never seen in training maps the per-object latent block by $\rho(R_o)$, exactly.

Three models differ in *which prior they carry, and nothing else*: **VN-Set** (both: a shared
`SE3PointEncoder` per slot + a shared jointly-equivariant `VNPredictor`), **MLP-Slot** (factorization
only: a shared *centred* per-object MLP + a shared ordinary `LatentPredictor` — identical slot structure
to VN-Set, missing **only** the rotation prior), and **MLP-Global** (neither: one monolithic MLP on the
flattened scene). The teacher is a **direct sum** of the validated Step-13 per-object dynamics — exactly
$\mathrm{SE}(3)^O\rtimes S_O$-equivariant — with two distinct anisotropic templates so the objects are
distinguishable (permutation non-vacuous, orientation observable per object). Metric: the same pooled
1-step latent relMSE as Step 13 ($<1$ beats predicting no change), on the pooled scene latent. FULL run:
$N_{\text{train}}{=}1500$, $60$ epochs, $K{=}6$ OOD draws; params VN-Set $16{,}856$ / MLP-Slot $61{,}920$
/ MLP-Global $245{,}440$ (the equivariant model is **3.7–14.6× smaller**); latent std $0.579/1.157/1.425$
(no collapse); seen relMSE all $<1$ ($0.295/0.097/0.152$ — all three are *genuinely trained* world
models, not degenerate baselines, so the OOD comparison is fair).

**The 2×2 that isolates each prior.** Two OOD axes, each the *transform of the same* held-out
transitions (paired; the transform of a valid teacher transition is a valid teacher transition):
**orientation-OOD** reorients each object independently by a random $\mathrm{SO}(3)$ about its own
centroid; **arrangement-OOD** translates each object independently to a novel placement. The OOD/seen
relMSE factor:

| over held-out scenes | arrangement-OOD | orientation-OOD |
|---|---:|---:|
| **VN-Set** (both priors) | $\times\,1.000$ | $\times\,1.000$ |
| **MLP-Slot** (factorization only) | $\times\,1.000$ | $\times\,17.76$ |
| **MLP-Global** (neither) | $\times\,6.31$ | $\times\,12.43$ |

Read the columns. The **arrangement** column is *exact-by-construction*: a translation-invariant,
shared-weight, per-object encoder simply cannot see a re-placement, so VN-Set and MLP-Slot are flat to
the float floor (ratio $1.0000$) while the un-centred MLP-Global degrades $6.3\times$ — this **isolates
the factorization contribution**. The **orientation** column is the *decisive, learned* result: VN-Set
and MLP-Slot have **identical** slot structure, so the only thing differing between them is the
$\mathrm{SE}(3)$ prior, and VN-Set staying flat ($\times1.000$) where MLP-Slot blows up ($\times17.76$:
seen $0.097\to$ OOD $1.72$, i.e. on novel per-object poses the non-equivariant slot predictor collapses
to *worse than predicting no latent change*) **isolates the equivariance contribution**, cleanly, with
factorization held fixed. MLP-Global, carrying neither prior, degrades on both. **You need both priors
for full compositional 举一反三** — that is the headline, and it is a 2×2, not a single number.

**Structural backbone (init *and* post-train — the unit-test half).** The exact properties survive
optimisation, verified to the float floor in `tests/test_set_equivariance.py`: post-train VN-Set
composed global-$\mathrm{SO}(3)$ residual $3.6\times10^{-5}$ and permutation residual $0$; MLP-Slot
permutation $0$ (factorized) but $\mathrm{SO}(3)$ **broken** at $4.9$ (the control that makes "VN-Set is
equivariant" non-vacuous); MLP-Global permutation **broken** at $6.4$ and leakage $0.935$ (the control
that makes "the slot models are factorized" non-vacuous), against $0.000$ leakage for *both* slotted
models. Every exactness claim thus has a model that demonstrably *fails* it.

**Honest scope — read before believing the headline.** The objects **do not interact**: the scene
teacher is a direct sum of per-object dynamics. That is exactly what makes the factorization theorem
clean, and it is the price of a *provable* compositional symmetry at laptop scale — so
**arrangement-invariance is architectural (centring), not learned**, and the genuinely-learned, decisive
comparison is the orientation column (VN-Set vs MLP-Slot, identical factorization). An *interaction*
channel — a relative-pose / equivariant message-passing block between slots, the multi-object analogue
of Step 18's centroid term — is the obvious next rung and is **explicitly future work**; until it
exists, Step 19 establishes compositional generalisation for *non-interacting* objects only. **Verdict —
all five guards green:** VN-equivariant (composed $<10^{-4}$) ✓; factorization-permutation (slots exact,
global breaks) ✓; leakage (slots $0$, global $0.94$) ✓; orientation (VN flat, both MLPs degrade) ✓;
arrangement (both slots flat, global degrades) ✓. **PASS.** Confidence ≈ **0.8** that the two priors are
separable and each buys its named half of the scene group — one notch below Step 18 because the
*interaction-free* teacher is a real scope limit, not because any panel is weak (the separations are
large and the structural half is exact).

---

## 14. Step 20 — active inference: the Expected Free Energy in the equivariant latent (the curiosity invariance)

Steps 13–19 built the *pragmatic* half of the loop — perceive, predict, and act toward a goal — and
proved its exact $\mathrm{SE}(3)$-equivariance. Friston's active inference adds the *other* half: an
agent should also act to **reduce its own uncertainty**. CLAUDE.md Open Questions #2 and #5 ask for
exactly this — *a tractable, information-geometric formulation of active inference for a deep
equivariant world model, unified with self-supervised latent prediction.* Step 20 answers with a
concrete construction: the agent minimises the **Expected Free Energy** (EFE) of an action sequence,
$$
  G(a_{1:H}) \;=\; \underbrace{\sum_h w_h\lVert \bar z_h - z_g\rVert^2 + w_t\lVert \bar x_0 + c_t\!\textstyle\sum_h a_h - \bar x_g\rVert^2}_{\text{pragmatic / risk — the validated Step-18 cost}}
  \;-\; \beta\,\underbrace{\sum_h \mathcal{D}_h}_{\text{epistemic / information gain}},
$$
the standard risk$-$epistemic decomposition (Friston 2017; the $-\beta$ means *minimising* $G$
*maximises* information gain). Both halves live in the **learned latent** of the equivariant JEPA: the
pragmatic term is the Step-18 cost (latent terminal distance + the exact closed-form centroid channel)
on the ensemble-mean latent $\bar z_h$; the epistemic term is the **ensemble disagreement**
$\mathcal{D}_h=\tfrac1K\sum_k\lVert z_h^{(k)}-\bar z_h\rVert^2$ of a $K{=}5$ predictor ensemble sharing
**one** equivariant encoder (deep ensembles, Lakshminarayanan 2017; disagreement-as-exploration,
Pathak 2019 / Sekar 2020 "Plan2Explore"), trained with a per-member Poisson(1) bootstrap so the heads
fit the data yet diverge where it is sparse. Its information-geometric face is the Gaussian differential
entropy $\mathcal{H}=\tfrac12\log\det(\hat\Sigma+\epsilon I)$ of the predictive belief.

**The theorem (why this belongs in *this* project).** Every predictor is jointly equivariant,
$f_k(\rho(R)z,Ra)=\rho(R)f_k(z,a)$, and the shared encoder is equivariant. Because $\rho(R)$ is
**orthogonal**, the mean is equivariant ($\bar z\mapsto\rho(R)\bar z$) while the disagreement is
**invariant**: $\mathcal{D}(\rho(R)z,Ra)=\tfrac1K\sum_k\lVert\rho(R)(z^{(k)}-\bar z)\rVert^2=\mathcal{D}(z,a)$,
and likewise $\hat\Sigma\mapsto\rho(R)\hat\Sigma\rho(R)^\top$ so $\log\det(\hat\Sigma+\epsilon I)$ is
unchanged ($\det\rho=\pm1$). **The agent's epistemic drive — its curiosity — is an exactly
$\mathrm{SE}(3)$-invariant scalar:** *how much there is to learn from an action does not depend on the
global pose of the scene.* With the invariant pragmatic cost the whole EFE $G$ is invariant, so the
EFE-optimal plan is $\mathrm{SE}(3)$-*equivariant*. A non-equivariant ensemble has none of this — the
control. FULL run ($N_{\text{train}}{=}1500$, $60$ epochs, $K{=}5$): params VN $74{,}456$ / MLP $494{,}368$
(the equivariant model is **6.6× smaller**); final latent std $0.715/1.137$ (no collapse).

**[A] EFE invariance — the theorem, init *and* post-train.** The disagreement, the Gaussian entropy,
and the *total* one-step $G$ (under a full $(R,t)$ motion) are all $\mathrm{SE}(3)$-invariant to the
e3nn floor for the VN ensemble, before and after a real Muon/AdamW + EMA-target + VICReg run; the MLP
ensemble misses each by orders of magnitude (the control that makes the assertion non-vacuous):

| post-train residual | disagreement-inv | entropy-inv | total-$G$-inv $(R,t)$ |
|---|---:|---:|---:|
| **VN ensemble** (shared equivariant $E$) | $2.4\times10^{-5}$ | $3.1\times10^{-5}$ | $2.3\times10^{-5}$ |
| **MLP ensemble** (control) | $0.205$ | $2.83$ | $134.5$ |

(at init the VN residuals are $\sim\!10^{-7}$–$10^{-5}$; the MLP is already broken at $1.03$ / $0.34$).
Pinned to the float floor in `tests/test_efe_invariance.py` (VN disagreement/entropy/total-$G$ $<10^{-4}$
init + post; MLP control breaks each).

**[B] Epistemic geometry — curiosity is blind to re-orientation, but not constant.** Move a
$(\text{cloud},\text{action})$ pair to another point of its $\mathrm{SE}(3)$ orbit (rotate *both* the
cloud and the type-1 action by the same $R$): the VN disagreement is **exactly unchanged** — the
equivariant agent is *correctly not curious* about a pose it already generalises across (举一反三). Yet
the drive is a genuinely *non-constant* field (coefficient of variation $1.22$ across the probe batch,
and off-orbit novelty — an OOD-shape cloud, which scaling/jitter put outside $\mathrm{SO}(3)$ — raises it
$\times1.54$), so the invariance is **non-vacuous**, and that elevated novelty signal is itself
rotation-invariant to $3.6\times10^{-7}$. The non-equivariant control instead assigns **spurious**
novelty to mere re-orientation:

| held-out probe | re-orient ratio $\mathcal{D}(\text{orbit})/\mathcal{D}(\text{seen})$ | CoV (non-vacuity) | off-orbit novelty | novelty rot-inv |
|---|---:|---:|---:|---:|
| **VN ensemble** | $\times\,1.0000$ (theorem) | $1.22$ | $\times\,1.54$ | $3.6\times10^{-7}$ |
| **MLP ensemble** | $\times\,6.38$ (spurious) | $0.53$ | $\times\,1.71$ | $7.84$ |

The VN's $\times1.0000$ is the 举一反三 thesis stated in the language of curiosity: *do not spend
information-seeking effort on what the symmetry already gives you for free.* The MLP conflates pose with
novelty ($\times6.38$) — it would waste exploration re-examining rotated copies of what it has seen.

**[C] The active-inference knob.** Sweeping $\beta$ in an EFE CEM planner (the Step-18 iso-$\sigma$
planner, now minimising $\mathrm{zscore}(\text{prag})-\beta\,\mathrm{zscore}(\text{epi})$) trades
pragmatic progress for epistemic gain **monotonically** — $\beta:0\!\to\!12$ raises the selected plan's
epistemic value $82.3\to419.4$ while its pragmatic cost rises $24.6\to135.7$ (more $\beta\Rightarrow$ seek
information, trade goal distance — exactly what active inference predicts) — and the EFE-selected plan
stays $\mathrm{SE}(3)$-equivariant end-to-end: $\lVert\mathrm{plan}(Rx)-R\,\mathrm{plan}(x)\rVert_\infty=
6.0\times10^{-8}$ (theorem realised through the whole closed loop, perception + prediction + epistemic
*and* pragmatic drives).

**Honest scope — read before believing the headline.** The teacher is **fully observed and
deterministic**, so on *this* task the epistemic term is not *required* to reach goals — the pragmatic
planner already does (Step 18). What Step 20 establishes is that the unified EFE objective is (i)
well-posed and tractable in the equivariant latent, (ii) carries an *exact* geometric invariance the
thesis predicts and a non-equivariant model lacks, and (iii) the active-inference knob measurably does
what theory says. The empirical payoff *of exploration* — tasks that are unreachable *without*
information-seeking (partial observability, sparse/ambiguous goals) — is the named next rung and is
**not** claimed here; active inference is treated as the source of a geometric structure, not as a
benchmark win (per CLAUDE.md's standing caveat that it has been "almost there" for 15 years). **Verdict —
all five guards green:** VN-invariant (disagree/entropy/total-$G<10^{-4}$) ✓; MLP-breaks (control bites)
✓; epistemic geometry (re-orient $\times1.000$, CoV$>0$, novelty rot-inv $<10^{-4}$, MLP spurious) ✓;
$\beta$-knob (epi & prag rise) ✓; plan equivariance ($6\times10^{-8}$) ✓. **PASS.** Confidence in the
*invariance theorem and tractability* ≈ **0.9** (it is exact by construction and survives training, with
a control that fails); confidence that the epistemic term *converts to a task win* on a harder
partially-observed problem ≈ **0.55** (genuinely open); overall ≈ **0.7** — the geometry is certain, the
active-inference payoff is a demonstrated mechanism, not yet a result.

---

## 15. Step 21 — the sample-efficiency frontier: equivariance as a learning curve, not a point (Open Question #1)

Every step so far measured generalisation at a *single* training-set size. Step 21 sweeps it and
draws the **frontier** — test error vs the number of interactions $N$ — because that frontier is the
operational form of CLAUDE.md Open Question #1 (*does $\mathrm{SE}(3)$-equivariance in a JEPA encoder
improve sample efficiency?*) and the sharpest statement of the thesis: the inductive-bias payoff is
exactly the gap between two learning curves.

**Protocol.** Both models — the Step-13 backbone (`SE3PointEncoder`+`VNPredictor` vs a
param-comparable `MLPPointEncoder`+`LatentPredictor`) — train on the thin orientation wedge
$\phi\in[0,90°)$. At each $N\in\{16,32,64,128,256,512\}$ we read two curves: pooled latent 1-step
relMSE on held-out **in-wedge** clouds (`seen`) and on the *same* transition rotated by random
$\mathrm{SO}(3)$ (`group`). The budget is a **fixed 600 gradient updates per run**
($\text{epochs}=\mathrm{round}(600/\lceil N/\text{bs}\rceil)$), so the abscissa is *data size*, not
optimisation steps; 3 seeds; same `train_jepa` (EMA target + VICReg + Muon/AdamW) as every step.

### [A] The theorem: the equivariant whole-group curve *is* its in-wedge curve, at every $N$
With $E(Rx)=\rho(R)E(x)$, $f(\rho z,Ra)=\rho f(z,a)$ and $\rho(R)$ orthogonal, the relMSE numerator
$\lVert\rho(R)(f(E(x),a)-E(x'))\rVert^2$ and denominator $\lVert\rho(R)(E(x')-E(x))\rVert^2$ are both
$\rho$-invariant, so $\mathrm{relMSE}(Rx,Ra,Rx')=\mathrm{relMSE}(x,a,x')$ for **all** $R$, **all
weights, all $N$, even at init**. Measured `group/seen` $=1.0000$ at all six $N$ (the VN `seen` and
`group` columns coincide identically). The non-equivariant MLP has no such cancellation.

### [B] The frontier (the decisive table)

| $N$ | VN `seen`$=$`group` | VN g/s | MLP `seen` | MLP `group` | MLP g/s |
|----:|--------------------:|-------:|-----------:|------------:|--------:|
|  16 | 0.939 | 1.000 | 0.900 | 2.03 |  2.26 |
|  32 | 0.768 | 1.000 | 0.727 | 1.85 |  2.54 |
|  64 | 0.677 | 1.000 | 0.565 | 2.07 |  3.66 |
| 128 | 0.647 | 1.000 | 0.327 | 1.66 |  5.07 |
| 256 | 0.541 | 1.000 | 0.213 | 2.02 |  9.48 |
| 512 | 0.433 | 1.000 | 0.217 | 3.15 | 14.52 |

VN $16{,}856$ params vs MLP $124{,}512$ ($7.4\times$). In-distribution target $\tau{=}0.65$: VN
reaches it at $N\approx120$, the MLP at $N\approx44$ (the baseline needs *fewer* wedge samples).
Whole-group target $\tau{=}0.65$: VN at $N\approx120$, MLP **wall** (never, on the grid).

### [C] The honest reading — across the group, not in-distribution
The two-sided answer to Open Question #1, stated without varnish:

- **In-distribution: no equivariant edge — if anything the opposite.** The MLP, with $7.4\times$ the
  parameters, fits the wedge *better* once $N\ge128$ (`seen` $0.22$ vs VN $0.43$ at $N{=}512$) and
  reaches any common in-wedge target with *fewer* samples. On its own training distribution the
  unconstrained model wins — exactly what Sutton's Bitter Lesson predicts. The equivariant prior is
  **not** a free in-distribution accelerator here, and I will not claim it is.
- **Across the group: the whole game.** The VN's whole-group curve **descends** with wedge data
  ($0.939\!\to\!0.433$) and reaches competence ($\le0.65$) at $N\approx120$ wedge samples it never
  saw rotated; the MLP's whole-group error is a **wall** — flat-high at $1.6$–$3.2$, `group/seen`
  climbing to $14.5$, never reaching the target at any $N$. Wedge-only data $+$ the prior
  $\Rightarrow$ whole-group competence; no amount of in-wedge data buys the baseline the same thing.

So the sample-efficiency payoff is real but *located*: it is the gap between a **learnable
whole-group frontier and a wall**, not a smaller-$N$-to-fit-the-wedge story. This is the most honest
version of the thesis — and it *sharpens* the geometric claim rather than softening it: where the
world genuinely carries the group, equivariance converts a thin slice of data into competence over
the entire orbit (举一反三), which brute capacity cannot do at any $N$.

**Honest scope — read before believing the headline.** (i) The teacher is the synthetic
exactly-$\mathrm{SO}(3)$ Step-13 world — the price of a *provable* 3D symmetry at laptop scale;
nothing here speaks to approximate or absent symmetry (cf. Step 16's misspecification boundary, where
the prior stops being free, and the Bitter Lesson as the standing caveat). (ii) The in-distribution
comparison is deliberately *not* reported as a VN win — it is a wash or a loss, and saying so is the
point. (iii) One task family, laptop compute, latent 1-step relMSE (not binary task success).
**Verdict — all six guards green:** VN-flat (`group/seen`$<1.10$ at every $N$) ✓; MLP-wall
(`group/seen`$=14.5$ at $N{=}512$) ✓; VN-fits (in-wedge relMSE $0.43<0.9$, beating the no-change
predictor's $1.0$) ✓; VN-descends (whole-group $0.939\!\to\!0.433$) ✓; smaller ($7.4\times$) ✓;
group-frontier (VN reaches the group target; MLP never) ✓. **PASS.** Confidence in the *across-group
frontier and the wall* ≈ **0.9** (a quantitative face of the equivariance theorem, init-and-post
guarded in `tests/test_sample_efficiency_frontier.py`); confidence that "no in-distribution edge"
*generalises* beyond this teacher/capacity regime ≈ **0.6** (it is the honest reading here, but
architecture-dependent). The cleanest statement of Open Question #1 the project can make.

---

## 16. Step 22 — the symmetry-break × data phase diagram: locating the Bitter-Lesson boundary

Steps 16 and 21 each swept *one* axis of the geometric bet and pinned the other: Step 16 swept the
**symmetry break** $g$ (a fixed lab-$z$ field added to the exact-SO(3) Step-13 teacher) at a single
large data size $N{=}1200$; Step 21 swept the **data size** $N$ at a single symmetry level $g{=}0$.
Neither answers the question the real world actually poses — symmetry is *approximate* **and** data is
*finite* — so Step 22 fills the whole $g\times N$ plane. At every cell it trains both backbones on the
thin $z$-wedge of the misspecified teacher
$$ \mathrm{Dyn}_g(x,a)_i \;=\; \mathrm{Dyn}_0(x,a)_i \;-\; g\,\langle e_z,\tilde x_i\rangle\,e_z, \qquad \tilde x_i = x_i-\bar x, $$
and reads **two** latent 1-step relMSE metrics: held-out **in-wedge** (`seen`) and **across the whole
group** (`ood` — genuine full-SO(3) transitions of the *true* $\mathrm{Dyn}_g$). The result is a
two-metric map of the project thesis against Sutton's Bitter Lesson (2019): a $5\times5$ grid in
$(g,N)$, 2 seeds, 600 updates/run; VN $16{,}856$ vs MLP $124{,}512$ params ($7.4\times$).

### [A] The knob is honest, and OOD must be re-sampled, not rotated
The added term is **centering-invariant** ($\sum_i\langle e_z,\tilde x_i\rangle = 0$, so it is a *real*
prediction target the VN encoder cannot wash out as a mere translation) yet lies in the **complement of
the SO(3)-equivariant maps** (a *fixed* lab axis), and it breaks the symmetry **monotonically**: the
non-equivariance fraction climbs $0\to0.13\to0.40\to0.89\to1.27$ as $g:0\to0.8$. Crucially, at $g{=}0$
the teacher is equivariant, so a *rotated* held-out transition is a genuine label (Step 21's "rotate
the test set" is valid); once $g{>}0$ that identity fails by $O(1)$ — a rotated target becomes a *fake*
label — so the across-group set must be **re-sampled** at full SO(3) through the true $\mathrm{Dyn}_g$.
The rotated-label residual jumps from the float floor ($9\times10^{-8}$ at $g{=}0$) to $0.06$–$0.47$ the
instant $g{>}0$. Both the honest knob and the re-sample necessity are guarded in
`tests/test_symmetry_data_phase.py`.

### [B] Across the group: the prior wins 24 of 25 cells (decisive)
Winner per cell (lower `ood` relMSE); the single MLP win in bold:

| $g$ (noneq) | $N{=}32$ | $64$ | $128$ | $256$ | $512$ |
|---|:--:|:--:|:--:|:--:|:--:|
| $0.0$ ($0.00$) | VN | VN | VN | VN | VN |
| $0.1$ ($0.13$) | VN | VN | VN | VN | VN |
| $0.2$ ($0.40$) | VN | VN | VN | VN | VN |
| $0.4$ ($0.89$) | VN | VN | VN | VN | VN |
| $0.8$ ($1.27$) | VN | VN | VN | VN | **MLP** |

The geometric prior wins the across-group metric **everywhere except the single joint-extreme corner**
$(g{=}0.8, N{=}512)$, and even there only barely (VN `ood` $0.798$ vs MLP $0.760$). Two structural
facts drive it:
- **The MLP wall is data-proof.** Along the $g{=}0$ column its across-group error is
  $\{1.54,1.76,1.17,1.81,2.34\}$ — flat-high and, if anything, *rising* with $N$: more wedge data never
  lowers it, because whole-group competence needs the off-wedge *orientations* a wedge never shows.
  (The VN column descends $0.84\!\to\!0.50$ — the Step-21 frontier, here as one slice.)
- **The wall only cracks where the break is maximal *and* data maximal.** As $g$ grows the
  fixed-lab-frame component grows with it, and that component is *orientation-free*, so the
  unconstrained MLP can fit it without ever seeing new orientations: its wall **descends** along the
  bottom row ($2.34\!\to\!0.76$ as $g:0\to0.8$ at $N{=}512$). Meanwhile the VN's *own* across-group
  floor **rises** with $g$ ($0.50\!\to\!0.80$) because it structurally cannot represent that lab term.
  The two converge and finally cross only at the corner — exactly the regime (most-asymmetric world ×
  most data) where the Bitter Lesson predicts scale should win.

### [C] In-distribution: capacity wins early everywhere — and the gap does *not* widen with $g$
Winner per cell (lower `seen` relMSE), with the in-wedge crossover $N^\star(g)$:

| $g$ | $N{=}32$ | $64$ | $128$ | $256$ | $512$ | $N^\star$ |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| $0.0$ | MLP | MLP | MLP | MLP | MLP | $32$ |
| $0.1$ | VN | MLP | MLP | MLP | MLP | $64$ |
| $0.2$ | MLP | MLP | MLP | MLP | MLP | $32$ |
| $0.4$ | VN | MLP | MLP | MLP | MLP | $64$ |
| $0.8$ | VN | MLP | MLP | MLP | MLP | $64$ |

On its own training wedge the $7.4\times$-larger MLP takes over by the **second grid point at every
symmetry level** ($N^\star\le64$): the capacity win Sutton predicts, immediate and near-total. But the
Step-16 prediction that the in-distribution gap should **widen** with $g$ (the VN unable to fit the lab
term in-wedge either) **does not hold at these data sizes**: the VN$-$MLP `seen` gap at $N{=}512$ is
$+0.285$ at $g{=}0$ and $+0.218$ at $g{=}0.8$ — roughly flat, slightly *narrowing*. The Step-16
widening was a higher-data ($N{=}1200$) phenomenon; at $N\le512$ both models are still data-limited and
the fixed-axis term, being orientation-free, is easy for **both**, so the VN's in-wedge floor has not
yet blown up. An honest correction to the single-slice story, not a hidden one.

**Step 22 verdict.** The $g\times N$ plane *locates* the geometric payoff rather than asserting it.
Across the group it is a **data-proof, near-total win** (VN 24/25; the lone loss the joint extreme of
maximal break × maximal data); in-distribution, capacity wins early at every $g$ ($N^\star\le64$).
*The metric decides — and that is the result:* where you must generalise across a group the world
(approximately) has, hard-coding it turns a thin data slice into whole-orbit competence that scale
cannot buy until the world is both maximally asymmetric and data-rich; where you only need to fit what
you have already seen, capacity wins. Two *pre-registered* predictions did **not** survive contact with
the plane — "the VN wins the literal whole box" (it wins 24/25; the wall cracks at the corner) and "the
in-distribution gap widens with $g$" (flat-to-narrowing at $N\le512$) — and I report both as refuted:
*locating* the boundary is more informative than a clean sweep would have been. The robust facts that
replaced them (near-total located across-group win, data-proof wall, early in-wedge crossover at every
$g$, monotone honest knob) are guarded in `tests/test_symmetry_data_phase.py`; see
`figures/where_the_bet_pays.png` for the frontier + two-metric phase panels. Confidence ≈ **0.85** on
the located across-group win and the data-proof wall; ≈ **0.6** that the precise corner of the crossover
generalises beyond this teacher/capacity/compute regime.

---

## 17. Key architectural finding: the VN hypothesis class in 2D

While designing Step 9 I first tried an *analytic* nonlinear dynamics with quadratic
drag $c_2\lVert v\rVert v$ and a gyroscopic curl $c_3\lVert v\rVert\,v^\perp$. The VN
model then **failed to fit even the training wedge** (in-wedge relMSE $1.5\times10^{-2}$,
worse than the MLP). The reason is a genuine and important constraint:

1. **Degree-1 homogeneity.** `VNLinear`+`VNReLU` are positively homogeneous of
   degree 1: $f(\lambda x)=\lambda f(x)$ for $\lambda>0$. So a scalar-weight VN net
   **cannot represent** $\lVert v\rVert v$ (degree 2).

2. **No $90°$ rotation in 2D.** A linear map $M$ that is SO(2)-equivariant must
   commute with every rotation, so $M\in\{aI+bJ\}$ where $J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$
   is the $90°$ rotation — the endomorphism algebra of the standard 2D rep is
   $\mathbb{C}$, not $\mathbb{R}$. **Scalar-weight VNLinear realises only the $aI$
   part**, so it *cannot apply $J$* and cannot represent the curl $v^\perp=Jv$.
   (In SO(3) this never bites: by Schur's lemma the endomorphism algebra of the
   irreducible standard 3D rep is $\mathbb{R}$, which is exactly why VNs use scalar
   weights — they were designed for 3D.)

Putting an analytic dynamics with $\lVert v\rVert v$ or $Jv$ *outside* the VN class
is not a fair "equivariance helps generalise" test — it is "the VN can't fit it."
The correct fix (matching Step 8) is to make the ground-truth dynamics itself a
**frozen random VN net**, which is in-class, exactly equivariant, and
direction-coupled via the VNReLU rectification (a non-globally-affine map the MLP
can match in-wedge but cannot extrapolate). This is why Step 9's dynamics is built
the way it is.

> **Implication for the project.** If we ever want a 2D world model whose latent
> dynamics must *rotate* features (curl/Coriolis-like effects), scalar-weight Vector
> Neurons are insufficient — we would need SO(2)-**steerable** layers (complex-linear
> weights, i.e. allowing the $bJ$ term), or to lift to 3D where the issue disappears.
> For purely *scaling/mixing* equivariant dynamics, VNs are exactly right.

---

## 18. Honest scope, confidence, and what's next

- **Mechanism (equivariance ⇒ generalisation across the group):** confidence ≈ **0.9**.
  Clean at the *prediction* level on exactly-equivariant dynamics — now including a
  **real** simulator (Step 10 [B]: ×16 OOD on PushT, VN flat), not only synthetic teachers.
- **A real system with exact *interior* symmetry (Step 10 [A]).** PushT turns out to be
  *more* symmetric than I assumed: interior agent↔block manipulation is SO(2)-equivariant
  to $10^{-5}$ px at any angle; only block↔wall contact breaks SO(2) down to the square's
  $C_4$. So in the interior the equivariant model has **no misspecification floor** — the
  earlier worry about "only approximate symmetry" was simply wrong for that regime.
- **The Phase-4 architecture itself is now realised end-to-end (Step 11).** The earlier
  steps used an explicit-coordinate forward model; Step 11 wires the equivariant *encoder* +
  equivariant *predictor* + planning **in the learned latent** into one JEPA and shows the
  learned representation **inherits the exact symmetry after training** (composed residual
  $2.9\times10^{-6}$, cost drift $1.5\times10^{-7}$ at every continuous angle), so latent-space
  prediction is 举一反三 across the whole circle from one $90°$ wedge (×1.00 vs the baseline's
  ×13.8). The latent planner closes the loop on real PushT. Confidence ≈ **0.9** at the
  representation level.
- **The SO(3) lift to 3D point clouds (Step 13).** The same end-to-end recipe —
  `SE3PointEncoder` + `VNPredictor(dim=3)`, planning in the learned latent — trained on an
  anisotropic cloud rotated only in a $z$-wedge, **keeps exact SO(3) equivariance after
  training** (composed residual $3.0\times10^{-5}$; planning cost drift $7\times10^{-7}$ vs
  the baseline's $0.85$) and is **举一反三 across the whole group** — latent relMSE flat ×1.00
  (VN $0.228$ on every bin: new angles, new axes, random SO(3)) while the baseline breaks OOD
  to $5.28$ (×17.2, worst on new axes) — with **7.4× fewer parameters** and a *better*
  in-distribution fit. So the decisive prediction-level result — exact equivariance after
  training **plus** zero-shot 举一反三 — now holds in **both 2D/SO(2) and 3D/SO(3)**, the
  project's target geometry. The honest negative: Step 13
  [C] purely-latent planning toward a goal *cloud* gets no traction for either model — a
  planner/decoder gap, not an equivariance one (the VN fails *flat* across the group, ×$-1.04$).
  Confidence ≈ **0.9** on [A']+[B].
- **Closed-loop gap: position-only was a tie (Steps 10–11 [C]); the contact-dominated
  *pose* task is the first non-tie (Step 12 [C]).** On position-only pushing the ×16/×14
  *prediction* advantage did **not** convert to a closed-loop task-success gap — an honest
  tie both times, because the rollout is dominated by the near-linear agent-PD subsystem
  (which the MLP extrapolates fine OOD), not the block-contact dynamics where equivariance
  bites. Step 12 fixes the task: a reorientation goal under an **SO(2)-invariant pose cost**
  $\mathcal{C}=W_{\text{pos}}\lVert b_H-g\rVert^2+W_{\text{ang}}\bigl(1-\langle d_H,g_{\text{dir}}\rangle\bigr)$
  makes block-**rotation** the metric. The conversion is now decisive at the mechanism level
  and partial at the control level: the MLP's *block* dynamics breaks OOD (block_dir relMSE
  $0.77\!\to\!2.33$, *worse than predicting no-change*) while it keeps its own near-linear
  motion ($0.089$); the VN stays flat ×1.00 on every channel. Closed-loop **orientation error**
  is the first OOD signal that is not a noise-limited tie — VN $5.2°\!\to\!5.7°$ (×1.09, true
  flat) vs MLP $11.8°\!\to\!21.8°$ (×1.85). It is **not** a clean binary task-success sweep:
  combined-pose success stays low for both at $N=15\times2$/bin, and the VN trades position
  error to minimise angle. Confidence ≈ **0.9** on [A]+[B], ≈ **0.6** on [C].
- **Closed-loop conversion made exact and paired (Step 14).** The remaining weakness of
  Step 12 [C] was that it was *unpaired* — task-to-task difficulty variance is what kept
  Steps 10–12 "within noise." Step 14 uses the exact symmetry as a *design*: rotate an entire
  reorientation task by $\Delta$ (a valid real task at identical difficulty, by Step 10 [A]),
  evaluate the **same** base task seen vs OOD with env- and CEM-seed fixed, and take the
  **paired** difference $d_i$ over $K{=}48$ tasks. With an **equivariant planner** (isotropic
  $\sigma$, $R(\Delta)$-rotated noise, disk action constraint) held identical for both models,
  the VN's paired OOD$-$seen angle change is **zero to the float floor**
  ($\max_i\lvert d_i\rvert=4.9\times10^{-5}$ deg — the trajectory at $\Delta$ is *exactly*
  $R(\Delta)$ times the seen trajectory), while the MLP degrades $+3.68°$, CI $[+1.49,+6.02]$,
  excluding 0. The diagnostic panel [S] (verbatim Step-12 planner, *not* equivariant at generic
  angles) shows the MLP still degrades but the VN's exactness softens to a still-unbiased tie
  (mean $-0.71°$, CI $[-2.76,+1.01]$) — establishing that **closed-loop orientation-invariance
  needs both an equivariant model and an equivariant planner**, which is exactly why the
  earlier closed loops (non-equivariant planner) were noise-limited. Confidence ≈ **0.9** on
  [E], ≈ **0.85** on the model+planner [S] finding.
- **The full SE(3) group, not just rotation (Step 15).** The named target geometry is SE(3); Steps
  10–13 only ever made *rotation* the OOD axis. Step 15 adds **translation** and shows the equivariant
  latent world model is flat across the *whole* group: latent relMSE $0.228$ on every SE(3) bin
  (×1.00) vs the baseline's ×157 worst, at 7.4× fewer params. Honest asymmetry: rotation-equivariance
  is *learned* (and survives training, $3\times10^{-5}$), translation-invariance is *exact by
  centering* — geometry done right, not a deep result. Confidence ≈ **0.9** on the flatness, stated
  with that caveat.
- **The prior is robust to misspecification, but not free (Step 16).** Real worlds only approximately
  have a symmetry, so I broke the SO(3) teacher with a tunable fixed-lab-axis term and swept it. The
  VN's OOD error *does* climb as the world de-symmetrises (≈×3 over the sweep, then saturating — the
  prior costs something once it is partly wrong), yet it **still beats the unconstrained MLP OOD at
  all 12 grid points tested**, even when the broken component is ≈1.4× the symmetric one
  ($\mathrm{noneq}=1.42$). This *brackets* the Bitter-Lesson crossover rather than pinpointing it.
  **Platform-honest note:** this CPU sweep is the
  substitute for the originally-planned real-3D-sim ("Task 4"), which needs GPU/CUDA this Mac lacks;
  that validation remains genuine future work. Confidence ≈ **0.85**.
- **The contrast is the architecture, not the seed (Step 17).** Steps 12/14 reported the closed loop
  from a single trained model per architecture. Training 5 **independent** $(\text{VN},\text{MLP})$
  pairs, the seen→unseen angle degradation is VN $-0.97°\pm1.64°$ (95% CI $[-2.41,+0.47]$, straddles 0)
  vs MLP $+9.57°\pm4.01°$ ($[+6.05,+13.08]$, excludes 0) — **non-overlapping CIs across seeds**, every
  one of the five showing the same split. This is the *training-seed* error bar Steps 12/14 lacked;
  the VN's residual here is planner-induced (Step 14 [E] is the exact version). Confidence ≈ **0.85**.
- **The closed-loop theorem now holds in the named geometry — 3D SE(3) (Step 18).** Step 14 made the
  closed loop *exact* but in 2D/SO(2); Step 18 lifts the same paired [E]/[S] design to 3D point clouds
  under the **full SE(3) group**, on the Step-13 latent JEPA with an SE(3)-equivariant CEM planner
  (iso-$\sigma$, unit-**ball** clamp, $R$-rotated noise, latent + closed-form **centroid** cost — the
  centroid channel makes translation handling *exact by construction*, so SE(3) does not silently
  collapse to SO(3)). Over $K{=}24$ paired tasks on orbits of $1$ seen $+ 4$ OOD $(R,t)$, the VN's
  OOD/seen orientation-error ratio is $0.989$, CI $[0.977,0.999]$ (within $2\%$ of flat, the deviation
  *negative*) while the MLP's is $1.134$, CI $[1.049,1.234]$ (excludes 1) — **disjoint CIs**; [S] (the
  verbatim non-equivariant planner) grows the VN residual $\sim\!5\times$, re-confirming that
  closed-loop invariance needs **model *and* planner** equivariant. The honest difference from 2D: the
  3D VN is equivariant only to e3nn's **architectural $\sim\!10^{-6}$ floor** (not float32 — float64
  barely moves it; predictor exact $\sim\!10^{-8}$, single plan commutes to $1.2\times10^{-7}$ in the
  unit test), and the receding-horizon loop occasionally amplifies that into a CEM tie-flip — so the
  VN's $\max_i\lvert d_i\rvert=3.5°$ is a **tie-flip floor, not a symmetry break**, and the decisive
  statistic is the *ratio separation*, not the literal float zero 2D hit. Confidence ≈ **0.85**.
- **One object becomes a *scene*: the two compositional priors are separable, and each buys a named
  half of $\mathrm{SE}(3)^O\rtimes S_O$ (Step 19).** A three-model ablation differing *only* in prior —
  VN-Set (factorization **+** per-object SE(3)), MLP-Slot (factorization only, identical slots), MLP-Global
  (neither) — runs a 2×2 of OOD axes. **Orientation-OOD** (each object reoriented by a novel $\mathrm{SO}(3)$):
  VN-Set flat ($\times1.00$) vs MLP-Slot $\times17.8$ — *the same factorization*, so this **isolates the
  equivariance prior** as the cause. **Arrangement-OOD** (each object re-placed): VN-Set **and** MLP-Slot
  flat ($\times1.00$, exact, translation-invariant slots) vs MLP-Global $\times6.3$ — this **isolates the
  factorization prior**. Only VN-Set, carrying both, is flat on both axes; the structural half (permutation
  $0$, leakage $0$ for slot models vs $0.94$ for global; VN-Set composed SO(3) $3.6\times10^{-5}$ post-train)
  is exact and guarded in `tests/test_set_equivariance.py`. **Honest scope: the objects do not interact** —
  the teacher is a direct sum, arrangement-invariance is architectural (centring) not learned, so the
  decisive *learned* result is the orientation column; an inter-object relative-pose / message channel is
  the explicit next rung. Confidence ≈ **0.8**.
- **Active inference unifies with the equivariant world model, and the agent's curiosity is an exact
  geometric invariant (Step 20).** An Expected Free Energy objective — pragmatic goal-seeking (the Step-18
  cost) **minus** $\beta\times$ epistemic information gain (the disagreement of a $K{=}5$ predictor ensemble
  sharing one equivariant encoder) — is well-posed and tractable in the learned latent, answering Open
  Questions #2/#5. Its defining property is a *theorem*: because $\rho(R)$ is orthogonal, the disagreement
  and its Gaussian-entropy face are **exactly $\mathrm{SE}(3)$-invariant** (VN post-train residuals
  $\sim\!10^{-5}$ on disagreement, entropy, and the total $G$; the MLP control breaks each by
  $10^{4}$–$10^{6}\times$), so the whole EFE is invariant and the EFE-optimal plan equivariant
  ($6\times10^{-8}$). Operationally this means **re-orientation carries zero epistemic novelty** for the
  equivariant agent ($\mathcal{D}(\text{orbit})/\mathcal{D}(\text{seen})=\times1.0000$, vs the MLP's
  spurious $\times6.4$) while genuine off-orbit novelty still raises it ($\times1.54$, CoV $1.22$:
  non-vacuous) — 举一反三 in the language of curiosity. Guarded init + post in
  `tests/test_efe_invariance.py`. **Honest scope:** the teacher is *fully observed*, so the epistemic term
  is a demonstrated *mechanism with an exact geometric guarantee*, **not** a task-success necessity here;
  the payoff of exploration on a partially-observed problem is the named next rung. Confidence ≈ **0.9** on
  the invariance theorem + tractability, ≈ **0.55** that it converts to a task win, overall ≈ **0.7**.
- **The sample-efficiency *frontier*, and an honest in-distribution null (Step 21).** Sweeping the
  training-set size $N$ on the Step-13 wedge teacher draws two learning curves per model. The VN's
  whole-group curve **equals its in-wedge curve at every $N$ and at init** (`group/seen`$=1.0000$, the
  relMSE's orthogonal $\rho(R)$ cancelling in numerator and denominator) and **descends** with wedge
  data ($0.939\!\to\!0.433$, whole-group competence at $N\approx120$); the MLP fits the wedge but its
  whole-group error is a **wall** (`group/seen` $2.3\!\to\!14.5$, never reaching the target at any $N$).
  The honest half I will not hide: *in-distribution the higher-capacity MLP fits the wedge at least as
  well* ($0.22$ vs VN $0.43$ at $N{=}512$), so equivariance buys **no** in-wedge sample-efficiency edge —
  the payoff is *entirely* across the group. This is the operational form of Open Question #1: the
  payoff is the gap between a learnable frontier and a wall, not a smaller-$N$-to-fit-the-wedge story.
  Guarded init + post in `tests/test_sample_efficiency_frontier.py`. Confidence ≈ **0.9** on the
  across-group frontier/wall, ≈ **0.6** that the in-distribution null generalises beyond this teacher.
- **The payoff *located* on the whole symmetry × data plane (Step 22).** Steps 16 and 21 each moved one
  knob; Step 22 fills the $g\times N$ grid and scores both metrics at every cell. *Across the group* the
  prior wins **24 of 25 cells** — the MLP wall is **data-proof** (g=0 column flat-high $1.2$–$2.3$, not
  falling with $N$) and the only loss is the joint extreme $(g{=}0.8,N{=}512)$, where the now-large
  *orientation-free* lab term lets capacity finally edge across the group (VN $0.798$ vs MLP $0.760$)
  as the VN's own across-group floor rises ($0.50\!\to\!0.80$) and the MLP wall descends
  ($2.34\!\to\!0.76$). *In-distribution* capacity wins early at every $g$ (crossover $N^\star\le64$),
  but — correcting Step 16's single $N{=}1200$ slice — the gap does **not** widen with $g$ at $N\le512$
  ($+0.285\!\to\!+0.218$, flat-to-narrowing): both nets are still data-limited and the fixed-axis term
  is easy for both. Two pre-registered predictions ("VN wins the literal whole box"; "the in-wedge gap
  widens with $g$") were **refuted** and reported as such — locating the Bitter-Lesson boundary beats a
  clean sweep. Robust facts guarded in `tests/test_symmetry_data_phase.py`; figure
  `figures/where_the_bet_pays.png`. Confidence ≈ **0.85** across-group, ≈ **0.6** on the corner's
  generality.

### Caveat against over-claiming

The Bitter Lesson (Sutton) warns that brute-force scaling often beats clever inductive
biases. Everything above is laptop-scale. The result we can stand behind today is narrow and
specific: **when the world's dynamics genuinely has a symmetry, a model that hard-wires it
reaches competence *across the group* from far fewer interactions and generalises zero-shot
(an across-group payoff, not an in-distribution one — Step 21) — in
closed-loop planning on exactly-equivariant *synthetic* dynamics (Step 9), at the
*prediction* level on a *real* simulator (Steps 10–11) and in an end-to-end **3D / SO(3)**
latent JEPA (Step 13: VN flat ×1.00 vs baseline ×17.2, at 7.4× fewer params), and — on a
contact-dominated *pose* task — as a closed-loop *orientation* advantage that is first a
non-tie (Step 12 [C]: VN ×1.09 flat vs MLP ×1.85) and then, under a *paired* design with an
equivariant planner, **exact**: the VN's seen-vs-OOD angle change is zero to the float floor
over 48 tasks while the MLP degrades with a CI excluding 0 (Step 14 [E]) — and that 2D
closed-loop theorem now **lifts to the full 3D SE(3) group** (Step 18 [E]: VN OOD/seen
orientation-error ratio statistically flat at $[0.977,0.999]$, disjoint from the MLP's
$[1.049,1.234]$, with translation handled by an exact closed-form centroid channel), with the
honest caveat that in 3D the VN's residual is a CEM **tie-flip floor** at the model's
$\sim\!10^{-6}$ e3nn equivariance, not the literal float zero 2D reached (the single-plan
unit test still commutes to $1.2\times10^{-7}$).** What is **not** yet shown: that this converts to a clean *binary task-success* sweep on a real contact-rich
system (Step 12's combined-pose success is low for both models at small $N$, and the
equivariant model trades position error to minimise angle); that the exact closed-loop
invariance holds *without* a matching equivariant planner (Step 14 [S] shows a generic-angle
planner softens VN exactness to a still-unbiased statistical tie — closed-loop invariance is a
property of model **and** planner together); that purely-latent planning toward a goal *cloud*
works without a decoder (Step 13 [C] is negative for both models); that compositional generalisation
survives **object interaction** (Step 19's clean object-centric 2×2 is on a *non-interacting* direct-sum
scene — an inter-object relative-pose/message channel is the named next rung, untested); that the
**active-inference epistemic drive earns its keep on a task** (Step 20 proves the EFE is tractable in the
equivariant latent and that curiosity is an *exact* geometric invariant, but the teacher is fully
observed, so information-seeking is a demonstrated *mechanism*, not yet a benchmark win — partial
observability / sparse goals are the named test); nor that any of
it scales. Those are the next tests — not foregone conclusions.

---

## Reproduce

```bash
cd ~/Workspace/se3-ejepa
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step8_sample_efficiency.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step9_closed_loop.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step10_pusht_closed_loop.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step11_latent_jepa.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step12_pose_control.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step13_se3_latent_jepa.py   # SO(3) lift (STEP13_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step14_pose_control_power.py  # paired closed-loop power (STEP14_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step15_se3_translation.py     # SE(3): + translation (STEP15_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step16_misspecification.py    # SO(3) misspecification sweep (STEP16_SMOKE=1)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step17_multiseed_closed_loop.py  # training-seed error bar (STEP17_SMOKE=1)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step18_se3_closed_loop.py     # 3D SE(3) closed-loop [E]/[S] lift (STEP18_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step19_object_centric.py      # SE(3)^O |x| S_O object-centric 2x2 ablation (STEP19_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step20_active_inference.py    # EFE active inference: curiosity invariance + beta-knob (STEP20_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step21_sample_efficiency_frontier.py # sample-efficiency frontier: VN whole-group == in-wedge, MLP wall (STEP21_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step22_symmetry_data_phase.py # g×N phase diagram: prior wins 24/25 across the group, capacity wins in-wedge (STEP22_SMOKE=1 for fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/make_bet_figures.py          # renders step21_frontier.png + where_the_bet_pays.{png,pdf} from the Step-21/22 JSON dumps
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python tests/test_planner_equivariance.py        # the clean single-plan SE(3) theorem (residual 1.2e-7)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python tests/test_set_equivariance.py            # scene-group SE(3)^O |x| S_O equivariance, init + post-train
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python tests/test_efe_invariance.py              # active-inference EFE drives SE(3)-invariant, init + post-train
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python tests/test_sample_efficiency_frontier.py  # whole-group relMSE == in-wedge (float floor) + frontier statistic, init + post
```

Source: `experiments/step8_sample_efficiency.py`, `experiments/step9_closed_loop.py`,
`experiments/step10_pusht_closed_loop.py`, `experiments/step11_latent_jepa.py`,
`experiments/step12_pose_control.py` (reuses Step 10's forward models + env, adds the
reorientation task sampler and SO(2)-invariant pose cost),
`experiments/step13_se3_latent_jepa.py` (the SO(3) lift: end-to-end 3D point-cloud latent
JEPA — `SE3PointEncoder`+`VNPredictor(dim=3)` — with an exactly-SO(3)-equivariant synthetic
teacher, anisotropic $z$-wedge training, and full-SO(3) OOD bins),
`experiments/step14_pose_control_power.py` (the paired seen-vs-OOD closed-loop power analysis:
reuses Step 10's models + Step 12's pose task/cost, adds `rotate_pose_task`, an equivariant CEM
planner `cem_plan_pose_iso` — isotropic $\sigma$ + $R(\Delta)$-rotated noise + disk action clamp
`_disk_clamp` — and paired bootstrap CIs over $K{=}48$ tasks; panel [E] equivariant planner,
panel [S] verbatim Step-12 planner),
`experiments/step15_se3_translation.py` (completes SE(3): reuses the Step-13 encoders/teacher/recipe,
adds `translate_points` + a translation-invariance/teacher-translation-equivariance probe and an
SE(3) ladder of OOD bins — pure translation, pure SO(3), and their composition),
`experiments/step16_misspecification.py` (the robustness sweep: `teacher_step_g` adds a
centering-surviving, non-SO(3)-equivariant fixed-axis term with knob $g$, `noneq_fraction` measures
the broken share, and OOD is re-sampled at full SO(3) through the *true* $\mathrm{Dyn}_g$; dumps
`papers/figures/step16_misspecification.{json,png}`),
`experiments/step17_multiseed_closed_loop.py` (the training-seed error bar: trains $K{=}5$ independent
VN/MLP pairs, runs the verbatim Step-12 closed loop on each, reports the degree-degradation
distribution; dumps `papers/figures/step17_multiseed.json`),
`experiments/step18_se3_closed_loop.py` (the 3D SE(3) closed-loop lift: reuses the Step-13
`SE3PointEncoder`+`VNPredictor(dim=3)` latent JEPA + synthetic teacher, adds an SE(3)-equivariant CEM
planner `latent_cem_plan_iso` — iso-$\sigma$, unit-ball clamp `_ball_clamp`, $R$-rotated noise, latent
+ closed-form **centroid** cost — a Kabsch orientation readout `kabsch_rotation`/`rotation_angle_deg`,
and the paired seen-vs-OOD$(R,t)$ bootstrap over $K{=}24$ tasks; panel [E] equivariant planner, panel
[S] verbatim Step-13 planner; dumps `papers/figures/step18_se3_closed_loop.json`),
`experiments/step19_object_centric.py` (the object-centric lift to a *scene* group
$\mathrm{SE}(3)^O\rtimes S_O$: reuses the Step-13 `SE3PointEncoder`+`VNPredictor` per slot, adds the
shared-weight `SetSE3Encoder`/`SlotPredictor`, the two non-equivariant controls `SlotMLPEncoder` and
`GlobalMLPEncoder`, a direct-sum `scene_teacher_step`, the paired `transform_orient`/`transform_arrange`
OOD transforms, and the permutation/leakage/composed-equivariance probes; dumps the 2×2 to
`papers/figures/step19_object_centric.json`),
`experiments/step20_active_inference.py` (the active-inference EFE objective: a shared-encoder $K{=}5$
deep ensemble `EnsembleJEPA` over the Step-13 `SE3PointEncoder`+`VNPredictor`, the epistemic
`disagreement` + `gaussian_entropy` quantities, a per-member-bootstrap `train_ensemble_jepa`, the
`efe_term_invariance`/`total_efe_invariance` probes, the same-orbit `_reorient` + off-orbit `_novel_shape`
calibration sets, and an SE(3)-equivariant `efe_cem_plan` $\beta$-sweep planner; dumps
`papers/figures/step20_active_inference.json`),
`experiments/step21_sample_efficiency_frontier.py` (the sample-efficiency **frontier**: reuses the
Step-13 `build_eq_jepa`/`build_mlp_jepa` + `collect_cloud_transitions` + `latent_rel_mse`, trains both
on the thin $z$-wedge while sweeping the training-set size $N$ at a *fixed* gradient-update budget via
`train_eval_one`, reads two learning curves per model with `eval_seen_ood` — held-out in-wedge relMSE vs
the same transition rotated by random $\mathrm{SO}(3)$ — and a pure log-$N$ interpolator
`frontier_n_at_target` that turns a curve into the smallest $N$ to reach a target error and reports a
*wall* as `None`; dumps `papers/figures/step21_sample_efficiency_frontier.json`),
`experiments/step22_symmetry_data_phase.py` (the symmetry-break × data phase diagram: reuses the
Step-13 backbones + Step-16 `teacher_step_g`/`collect_transitions_g`/`noneq_fraction`, trains a
`train_eval_cell` over a $g\times N$ grid reading both held-out in-wedge and *re-sampled* full-SO(3)
metrics, scores winner-per-cell on each, and a pure `first_overtake_N` in-wedge crossover extractor;
dumps `papers/figures/step22_symmetry_data_phase.{json,png}`),
`experiments/make_bet_figures.py` (pure read-and-plot, no training: renders `step21_frontier.png` and
the combined `where_the_bet_pays.{png,pdf}` — frontier + the two-metric phase panels — from the
Step-21/22 JSON dumps), with the clean
single-plan SE(3) theorem guarded in `tests/test_planner_equivariance.py`
($\mathrm{plan}(g\cdot x)=g\cdot\mathrm{plan}(x)$ to $1.2\times10^{-7}$, MLP control fails), the
scene-group $\mathrm{SE}(3)^O\rtimes S_O$ equivariance (global $\mathrm{SO}(3)$ + permutation + leakage,
init **and** post-train, with both non-equivariant controls failing) guarded in
`tests/test_set_equivariance.py`, and the active-inference EFE drives (disagreement, Gaussian entropy,
total $G$ all $\mathrm{SE}(3)$-invariant init **and** post-train, MLP ensemble control failing) guarded in
`tests/test_efe_invariance.py`, and the whole-group sample-efficiency frontier (VN whole-group relMSE
$=$ in-wedge relMSE to the e3nn float floor — `ood/seen` $=1$ — init **and** post-train, the
non-equivariant MLP breaking it into a wall, plus the pure `frontier_n_at_target` log-$N$
interpolation / grid-hit / wall / monotonicity) guarded in `tests/test_sample_efficiency_frontier.py`,
and the Step-22 misspecified-teacher knob + phase extractor ($\mathrm{Dyn}_g$ reduces to the exact
SO(3) teacher at $g{=}0$ and breaks it monotonically for $g{>}0$; the break term is centering-invariant,
lab-$z$, a genuine target; the rotated OOD label is exact at $g{=}0$ but *fake* at $g{>}0$ ⇒ re-sample;
and `first_overtake_N` interpolates/walls/orders correctly) guarded in
`tests/test_symmetry_data_phase.py`;
equivariant
primitives in `src/models/structured.py` (`VNLinear`, `VNReLU`, `StructuredStateEncoder`,
`VNPredictor`),
the JEPA wrapper + latent predictor in `src/models/eqjepa.py`, EMA/VICReg training loop in
`src/training/jepa.py`, SE(3) encoder in `src/models/se3.py`.
