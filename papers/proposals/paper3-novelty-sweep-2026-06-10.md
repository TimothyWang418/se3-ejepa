# paper3 novelty pre-sweep — 2026-06-10 (pre-W1, by design before any build)

> 4 parallel verified sweeps (one per claim), ~90 queries total, every cited item's abstract
> WebFetch-verified. Discipline: paper2 lit-sweep format. Verdicts below; raw agent outputs in the
> session transcript of 2026-06-10.

**Overall: 4/4 SAFE-WITH-CAVEATS — GO for W1.** No preemption of any claim. Exact-phrase space
("certified subgoal spacing", "horizon certificate", "equivariant subgoal", "equivariant JEPA") is
**empty on arXiv as of today**. Both flanks are occupied and converging fast (two hierarchical
latent-planning papers Apr–Jun 2026; symmetry×skills heating in 2026) — speed matters.

---

## Claim 1 — certified subgoal spacing: SAFE

Nobody derives subgoal spacing / replan cadence **a priori** from a measured spectral certificate of
a learned latent model. Components exist separately; the composition is open.

| Must-cite | What it is | Differentiation (one sentence) |
|---|---|---|
| **HWM arXiv:2604.03208** (Apr 2026) ⚠️ highest priority | Hierarchical latent WM planning, multi-timescale, 70% real-robot pick-place; **post-hoc error-vs-horizon analysis (their Fig 6)** | Their Fig 6 *validates* a hand-set hierarchy post hoc; our certificate *derives* the spacing a priori — the law they observed is the one we predict |
| FF-JEPA arXiv:2606.09311 | Hierarchical JEPA subgoals on PushT, verbatim "We set the planning horizon H=25", zero justification | The motivating gap itself |
| AdaSubS arXiv:2206.00702 (ICLR'23) | Adaptive subgoal distance via search + learned reachability verifier (symbolic domains) | Runtime search vs a-priori closed form; no latent WM, no spectrum |
| THICK (ICLR'24) | Adaptive temporal abstraction from sparse latent context changes | Structural boundaries, not error-certified spacing |
| HIRO 1805.08296 / Director 2206.04114* | Canonical fixed-c / fixed-K latent subgoals | The convention we replace (*Director abs page still to fetch — flagged) |
| MBPO 1906.08253 | Return bound vs rollout length; authors call it "overly pessimistic"; k is scheduled, **bound never sets the knob** | Our certificate is the knob, two-sided tight (Prop 6/6′) |
| AdaMVE 1912.11206 | TD-learned state-dependent model error adapts value-expansion horizon | Statistical, in-training, value expansion ≠ subgoal spacing |
| STEVE 1807.01675 / M2AC 2010.04893 / MACURA 2405.19014 | Ensemble-uncertainty-weighted/masked rollout lengths | Reactive statistical triggers vs a-priori spectral certificate |
| Adaptive Planning w/ Gen. Models 2408.01510 | Ensemble-ID uncertainty adjusts replanning intervals (~10% of steps) | Nearest learned-model cadence work; reactive trigger, no certificate, no spectrum |
| When-to-Replan 2304.12046 / MPC-update-RL 2011.13365 | Replan timing learned from task reward | The "sweep/learn the knob" pattern we eliminate |
| **Self-triggered / event-triggered control** (family): GP-ETMPC 2110.12214, lifted self-triggered GP 2202.10174, neural ETC 2507.14653, survey 2009.12783 | Classical a-priori update-time computation from Lyapunov conditions | A priori — but from *known/GP dynamics at the control-input level*; no latent WM, no subgoals, no zero-data equivariant calibration |
| Pathak et al. 1710.07313 (Chaos'17) | Learned surrogates measured in Lyapunov times | The classical $1/\lambda_1$ predictability yardstick — we repurpose it as an a-priori *planning* knob |
| Latent-space CBF 2507.13871 / SALSA-RL 2502.15512 | Certificates in latent WM space (safety / post-hoc stability) | Certificate trend exists; none touches cadence/spacing |

Flagged (check before camera-ready): Director abs page; OpenReview VeO03T59Sh (conformal planning,
PDF unparseable); 2605.17058 §timescale-selection; self-triggered MPC w/ adaptive horizon
(nano-satellite, paywalled); Hieros 2310.05167; 2512.20605.

## Claim 2 — equivariant hierarchical latent planning: SAFE

Hard evidence of empty intersection: arXiv full-text "equivariant subgoal" = 0; abs:subgoal AND
abs:equivariant = 0; "equivariant world model" full-text = 3, none hierarchical.

| Must-cite | What it is | Differentiation |
|---|---|---|
| **HEP 2502.05728** (ICML'25, Walters/Platt) ⚠️ reviewer-bait | Hierarchical equivariant *policy* (frame transfer), imitation | Equivariance × hierarchy at policy level; **no world model, no latent subgoals, no certificates** |
| Imagination Policy 2406.11740 (CoRL'24) | SO(3) bi-equivariant point-cloud goal generation | Subgoal-like generation in *observation* space; BC, no WM, no planning loop |
| van der Pol 2002.11963 (AAMAS'20) | Action-equivariant latent WM + VI planning | Origin of equivariant latent WM planning; flat, discrete — nobody went hierarchical |
| EDGI 2303.12410 | SE(3)×Z×Sn equivariant diffusion planner | Flagship equivariant model-based planning; trajectory-level, flat |
| Group-Structured Latent WM 2506.01529 | Group-structured latent space + learned transitions | Stops exactly where our hierarchy + certificates begin |
| seq-JEPA 2505.03176 | Invariant+equivariant split in action-conditioned JEPA | Representation only; no hierarchy/planning |
| Flow Equivariant WM 2601.01075 | Lie-group flow equivariant memory WM | Prediction only (already in paper2 cites as lillemark) |
| Equivariant Action Sampling 2412.12237 | Equivariant MPPI in equivariant TD-MPC | Equivariant *flat* planning baseline framing |
| Class-Pose 2207.03116 | Latent = invariant class × group pose | The static ancestor of P4.C's group×invariant *dynamics* decomposition |
| Equivariant GCRL 2507.16139 | Goal-conditioned group-invariant MDP formalism | Overlapping problem statement, model-free flat |
| Ravindran-Barto SMDP homomorphisms + relativized options; 2605.22711 revival; GISD 2601.14000 | Symmetry × temporal abstraction lineage | Classical/discrete or model-free; no equivariant WM |
| PAC-Bayes equivariant bounds 2210.13150, quotient-space 1910.06552, Markov-data 2503.00292, correct/incorrect equivariance 2303.04745 | Theory context for wedge→orbit | Generalization-bound flavor; cite as foundation, not competition |

Flagged: hierarchical-symmetry MARL (Beihang); 2403.19024; 2206.03674; equivariant-RL-robustness
line (2408.14336, 2411.04225, 2512.00915 — relevant to certificate caveats under symmetry breaking);
OpenReview under-review pool is the residual blind spot (re-sweep at submission).

## Claim 3 — H1 rotation sample-efficiency: SAFE, two mandatory positioning requirements

1. **The rotation-$R^2$ probe metric belongs to the 3DIEBench line** (SIE 2302.10283 → EquiCaps
   2506.09895 → seq-JEPA 2505.03176 → ContextSSL 2405.18193). **We claim the axes, not the metric**:
   data-fraction sweep × architectural equivariance × manipulation/world-model latents. Closest =
   seq-JEPA (JEPA + rotation $R^2$ = 0.71, ResNet-18, **no architectural equivariance, no data
   sweep**); EquiCaps (architecture→0.78, no data axis, no WM).
2. **Brehmer et al. 2410.23179 ("Does equivariance matter at scale?")**: equivariance improves data
   efficiency BUT **augmentation closes the gap given enough epochs**. ⇒ **E-P4.1 must carry an
   augmentation arm (R-aug)**, and the honest claim hierarchy is: H1 is an empirical bet that may be
   closed by augmentation (report either way); **the certificate moat is theorem-protected against
   exactly this** — paper2 Lemma 2: orbit-constant error ⟺ equivariance, so an augmented model has
   no certificate transfer at any parameter count.

Theory anchors: Elesedy & Zaidi 2102.10333 (provably strict generalization benefit), Tahmasebi &
Jegelka 2303.14269 (exact sample-complexity gain) — H1 = their per-dimension empirical instantiation
inside a JEPA. Policy-level landmarks for the established general claim (verified): NDF 2112.05124,
SO(2)-Eq RL 2203.04439, EquivAct 2310.16050, EquiBot 2407.01479, EquiDiffPo 2407.01812. Context: PoR
2606.07687 has **zero citations as of today**; "equivariant JEPA" unclaimed on arXiv — window open,
speed matters. Flagged: Lie-derivative metric 2210.02984 (report as complementary equivariance
measurement); El Banani 2404.08636 §4 before quoting; EqR/Equivariant MuZero quick-check.

## Claim 4 — spacing-vs-spectrum law (signature figure): SAFE

Nobody measures hierarchical-planning feasibility against a controlled $\lambda_1$ knob; nobody
locates a model-error/optimizer crossover. Load-bearing differentiations:

| Must-cite | What it is | Differentiation |
|---|---|---|
| **2506.03889** (temporal horizons in forecasting) | Proves loss-landscape roughness $O(e^{\lambda T})$, $\lambda$ = Lyapunov exponent; U-shaped optimal *training* horizon | Training-loss geometry in forecasting; no planner, no certificate, explicitly no $\log(1/\epsilon)/\lambda$ horizon law |
| Jiang et al. AAMAS'15 | Optimal planning horizon shorter under model error (complexity control) | Model-error-limited regime only; discount horizon ≠ subgoal spacing; no spectrum, no optimizer regime |
| 2512.09929 (train-test gap in gradient planning) | Two failure *hypotheses* (model OOD vs optimization landscape) | The qualitative version; ours is quantitative with a located crossover |
| FTLE of MPC/RL 2304.03326 | FTLE fields of *controlled* agents | Direction reversed (planner shapes FTLE); only prior work with planner-horizon + Lyapunov in one frame |
| iCEM 2008.06389 | CEM degrades on long horizons; colored-noise fix | The optimizer-limb mechanism, engineering treatment |
| Compounding-errors study 2203.09637 | Environment dynamics = strongest factor in long-horizon error | Supports the premise; no $\lambda$ quantification, no planning link |
| Lyapunov-exponent robot optimization 2412.06776 | Measures/optimizes $\lambda$ in contact-rich robotics via diff-sim | $\lambda$ as design objective; we use $\lambda_1$ as *independent variable* |

Follow-up scan of the four paper2-known neighbors (2410.10674, Özalp&Magri line, Mo 2605.03338, Geng
2512.08991): all citations are forecasting/fluids/policy-side — **none moves toward planning. SAFE.**
Flagged: "Dynamic Push-T" variant (check which paper, whether physics-dialing — one targeted query
in W1 lane (b)); 2508.15588 (FTLE policy verification); 2510.16949; 2510.04342 (chaos curriculum —
nearest "stability knob", forecasting-side).

---

## Action items folded into the proposal (same-day amendments)

1. **R-aug base added** (rotation-augmented R-plain, Brehmer control): runs E-P4.1 + E-P4.2 only
   (anti-explosion); C2 gates now compare eq against *both* plain and aug-plain; claim hierarchy
   updated (H1 = empirical bet; certificate transfer = Lemma-2-protected).
2. **HWM enters §7 boundaries** next to FF-JEPA with the post-hoc-vs-a-priori differentiation; its
   existence reconfirms "the direction is heating" (2 hierarchical-latent-planning papers in 8 weeks).
3. **Self-triggered control** named in §7 as the classical family with the ready differentiation.
4. Re-sweep checklist at submission: Director abs page, OpenReview pool, flagged items above.
