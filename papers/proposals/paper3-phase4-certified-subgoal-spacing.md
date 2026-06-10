---
title: "paper3 (Phase 4) — Certified Subgoal Spacing for Equivariant World Models"
status: proposal approved 2026-06-10 (design brainstormed + section-by-section user sign-off); W1 recon next
created: 2026-06-10
venue-target: ICLR 2027 (paper deadline expected mid–late Sept 2026; ICLR'26 pattern was abstract 9/19, paper 9/24; official CFP not yet posted) + arXiv flag-plant as soon as C3 lands
framing: B-skeleton + C-ambition — factorial measurement design as insurance, certified-spacing claim as the spearhead, pre-registered title pivot rule
---

# paper3 / Phase 4 — Certified Subgoal Spacing for Equivariant World Models

> **One-line thesis (the flag-planting sentence).** FF-JEPA's $H{=}25$ is *one number, valid for one
> demonstration distribution*; our $H^*(\epsilon)$ is a **function** — and equivariance makes it a
> **globally valid** function of the orbit, certified from partial data.
>
> **Lineage.** This is the roadmap's Phase 4 ("SE(3)-Equivariant JEPA for manipulation"), upgraded by
> three 2026-06 intelligence events: FF-JEPA (arXiv:2606.09311) proved the *planning-layer* demand and
> left $H$ unprincipled; Prediction-over-Reconstruction (arXiv:2606.07687) measured the *rotation
> bottleneck* that motivates equivariance and supplied the probe protocol; paper2's certificate
> machinery (Thm A/B, Prop 7–11, `wm_audit.py`) supplies the principled answer to $H$. The three-piece
> blueprint: **equivariant encoder + hierarchical subgoals + certificate-chosen spacing** — the third
> piece is the moat nobody else has.

**Governing principles (inherited from paper2, zero loosening):** honest, gated, reproducible; every
claim pre-registered with its kill condition; INCONCLUSIVE reported as such; laptop/1–2-GPU scale;
**3 seeds line-standard** unless a cell states otherwise; calibration bands are multiplicative
$[0.5,2]$ unless a gate states otherwise.

---

## 0. The step93 scope law, internalized as a design axiom

step93's honest FAIL bought the lesson this paper is built on — and the research line has since
upgraded it to a theorem (**Prop 11, the decision-scope theorem**): decisions *aligned* with the
certified quantity incur zero extra regret at $c{=}1$, while *task-mapped* decisions carry an
irreducible misresolution penalty $\sim\lvert\log(\epsilon/\theta^\ast)\rvert/\lambda_1$. Therefore:

- **The certified object = the consumed object.** The hierarchical planner consumes *subgoal latents
  at gap $H$*; we certify *latent prediction error over the gap*, never "task return." Subgoal
  spacing is an **aligned** decision in Prop 11's sense — this paper deliberately stations its
  spearhead where the theorem says decision value concentrates.
- Closed-loop return is a **separate, quarantined readout** (C4): return is a *task-mapped* quantity,
  exactly where Prop 11 predicts the penalty. If certified-$H$ misses the oracle fixed-$H$ on return,
  C3 does not die with it — the miss is Prop 11's predicted penalty, *measured*, and gets reported as
  such.

---

## 1. Setup

Encoder $E:\mathcal X\to\mathcal Z$ (pixels $\to$ latent), action-conditioned predictor
$f:\mathcal Z\times\mathcal A\to\mathcal Z$, and a **subgoal predictor** $G$ operating at the coarse
timescale: $\hat z_{sg,m+1}=G(z_{sg,\le m})$ — a latent autonomous flow (FF-JEPA's object; step89/91's
audit pipeline reads exactly this kind of loop). Group $g$ acts on observations, $\rho(g)$ orthogonal
on latents; equivariance $E(g\cdot x)=\rho(g)E(x)$, $f(\rho(g)z,\sigma(g)a)=\rho(g)f(z,a)$.

**Certified spacing.** With measured one-step bias $\hat\delta$, leading exponent $\hat\lambda_1$,
and equivariance residual $\hat\epsilon_{\max}$ (all three via `wm_audit` gap mode, CIs by Prop 10):

$$
\widehat{\mathrm{Err}}(H)\;=\;\hat\delta\sum_{t=0}^{H-1}e^{\hat\lambda_1 t},
\qquad
H^*(\epsilon)\;=\;\max\{H:\widehat{\mathrm{Err}}(H)\le\epsilon\}.
$$

Two regimes, both in scope *by design*: **neutral** ($\hat\lambda_1\approx 0$, quasi-static PushT —
step91 measured exactly this) where the formula degenerates honestly to the linear budget
$H\hat\delta$ and the certificate's job is calibrating $\hat\delta$; **expansive**
($\hat\lambda_1>0$, the dynamics knob of §3.2) where the paper2 exponential machinery wakes up and
$H^*(\epsilon)\sim\log(1+\epsilon\hat\lambda_1/\hat\delta)/\hat\lambda_1$ bites. For
**group-transported** claims (wedge transfer, P4.B) the certified curve additionally carries Thm B's
$m\,\hat\epsilon_{\max}$ term — the resampling floor enters the certificate exactly here, and is
reported, never absorbed.

**Why equivariance is load-bearing and not decoration:** (i) paper2 E12 — equivariant models recover
the spectrum *zero-data prior-correct*; dense models need a calibration set; (ii) Thm A orbit
constancy — the certified curve $\widehat{\mathrm{Err}}(H)$ measured on a wedge is valid on the whole
orbit (P4.B below). Equivariance is what upgrades $H^*(\epsilon)$ from "a calibrated number" to "a
globally valid function."

---

## 2. Pre-registered claims ladder (each independently alive)

### C1 — Certificate prior-correctness on the pixel substrate (E12 lift). *Confidence 0.8.*

Posed **per regime** — posing it only on PushT-static would risk the claim dying for regime reasons
(step91's LeWM audit correctly ABSTAINED there: $\lambda_1\approx0$ makes the spectrum trivial), not
truth reasons.

- **C1a (expansive regime — the E12 lift proper).** On PushT-dyn($\kappa{>}0$): the equivariant
  JEPA's spectral certificate calibrates **with zero calibration data**; the plain JEPA needs a
  calibration set. **Gate:** eq zero-calibration ratio $\in[0.5,2]$ on $\ge2/3$ seeds; plain outside,
  or inside only after calibration.
- **C1b (neutral regime — orbit transfer of calibration).** On PushT-static the certificate honestly
  reduces to the linear $\hat\delta$ budget; the zero-shot content is **where $\hat\delta$ is valid**:
  eq's in-wedge $\hat\delta$ transfers across the orbit (Thm A), plain's under-covers out-of-wedge.
  **Gate:** eq out-of-wedge/in-wedge $\hat\delta$ ratio $\in[0.5,2]$ (expected $\approx1$ up to the
  resampling floor) on $\ge2/3$ seeds; plain outside.

**Kill condition (either sub-claim):** plain matches eq zero-shot $\Rightarrow$ that cell reported as
no-difference.

### C2 — Rotation sample-efficiency, H1-PushT (PoR protocol, *adapted*). *Confidence 0.6 headline / 0.95 curve.*

PushT's action space is 2-D pusher position — **there is no rotation dimension in the action**; the
rotation lives in the T-block pose $\theta$. The honest PushT instantiation, with **two co-primary
readouts**:

- **state-pose probe:** frozen encoder + fixed MLP probe $[D\to256\to128\to3]$, per-dim
  $(x,y,\theta)$ $R^2$;
- **effect probe:** $(z_t,z_{t+1})\to\Delta\theta$ — the faithful PushT translation of PoR's
  "action-relevant rotation structure" (closes the "static pose probing is too easy" hole).

**Fraction semantics (pinned):** fractions are **episode-level** and apply to the **encoder-training
demo set** $\{1\%,5\%,10\%,30\%,100\%\}\approx\{2,10,20,60,\sim200\}$ episodes — the thesis-relevant
quantity is representation sample-efficiency. The probe's own training budget is **fixed and
identical across all cells** (held-out episodes), so probe sample-efficiency is never the thing
measured.

**Registered readout = the full curves** $R^2$ vs encoder-demo fraction, per dim, both probes.
**Headline gate, two pre-registered tiers (both co-primary probes must pass within a tier):**

- **H1-strong:** $R^2_\theta(\text{eq},1\%)\ \ge\ R^2_\theta(\text{plain},100\%)$ and same for
  $\Delta\theta$;
- **H1-primary:** the same inequalities at $5\%$ ($\approx10$ episodes).

The paper reports whichever tier survives (strong $\Rightarrow$ primary); curves reported regardless.
Rationale: at $1\%$ ($=2$ episodes) *any* pixel model may degenerate — the strong tier must not be
the claim's single point of failure. Any curve shape is a result — including "plain learns it but
$30\times$ slower," which is the sample-efficiency story in raw numbers. PoR-original H1 (7-DoF
action rotation dims) runs natively on the 3-D co-anchor (E-P4.5). *Citation discipline: label as
"adapted: state-pose + effect probe," never claim protocol replication.*

### C3 — Certified subgoal spacing predicts the feasibility boundary (the spearhead). *Confidence 0.75.*

The certificate curve $\widehat{\mathrm{Err}}(H)$ (with Prop 10 CIs) predicts the **measured
feasibility boundary** $H_{\max}(\epsilon)$ of subgoal-conditioned short-window planning, across the
$\epsilon$ grid, across bases, and **across the dynamics knob** (§3.2). Precedent: Exp 22 already
validated this mechanism in a different guise — certificate-determined *re-observation cadence* under
a sensing budget (8–16% violations vs 61–65% for inflated certificates). Certified subgoal spacing is
the same mathematical object with the planner, not the sensor, as consumer.

Two registered sub-readouts:

- **Regime decomposition (new finding shape):** the feasibility boundary is set by two mechanisms
  FF-JEPA conflates — model-error budget (the certificate's jurisdiction) and CEM optimizer
  degradation with window length. We isolate the optimizer limb with ground-truth subgoals and
  **locate the crossover**.
- **Wedge transfer (the equivariance moat made visible):** certified curve measured in-wedge, tested
  out-of-wedge — eq transfers (Thm A / P4.B), plain does not.

**Precision of the claim (scope-law hygiene):** the certificate predicts the *model limb*; therefore
the prediction $H^*(\epsilon)\approx H_{\max}(\epsilon)$ is registered **for the region where the
model limb binds** (below the optimizer crossover), and the regime decomposition *locates* that
region rather than assuming it.

**Gate:** boundary-calibration ratio $H^*(\epsilon)/H_{\max}(\epsilon)\in[0.5,2]$ on $\ge 2/3$ seeds
per knob setting (within the model-limited region); wedge-transfer ratio in $[0.5,2]$ for eq and
outside for plain. Exact operationalization of $H_{\max}(\epsilon)$ in the E-P4.3 spec, frozen before
the run.

### C4 — Closed-loop factorial (quarantined system cells). *Confidence: (i) 0.85, (ii) 0.5, (iii) 0.75.*

Success rate @ $t\in\{25,75,150\}$ on the factorial of §3.1 (FF-JEPA's headline cell for context:
flat $3.5\%\to$ hierarchical $91.8\%$ @ $t{=}75$). Pre-registered predictions:
(i) hierarchical $\gg$ flat on our stack (effect replication);
(ii) certified-$H$ within 5 pp of the *oracle* fixed-$H$ **without running the sweep**. **The
resolution is not a free knob (Prop 11(ii) compliance):** the deployed spacing is
$H^*(\hat\theta^\ast)$ where $\hat\theta^\ast$ is the **planner tolerance elicited by E-P4.3's
optimizer-limb instrument** — Prop 11(ii) proves $\theta^\ast$ must come from the task side, and
E-P4.3 *is* that elicitation; no closed-loop return is consulted in choosing $H$. (Prop 11 frames
the expectation: spacing is aligned, return is task-mapped — a residual gap is the theorem's
penalty made visible, informative either way.) This closes the open end step93 left: certificate
supplies $H(\cdot)$, task supplies $\theta^\ast$, deployment is fully a priori;
(iii) eq $\times$ OOD-orientation $\gg$ plain $\times$ OOD.
All cells reported; no claim dies for another's miss.

### Title pivot rule (pre-registered)

- C3 $\wedge$ C4 alive $\to$ *Certified Subgoal Spacing for Equivariant World Models* (C-face).
- Only C1 $\wedge$ C2 alive $\to$ measurement-paper face ("where does structure pay — a layer-wise
  accounting of geometric priors in latent world models") (B-face).
- All alive $\to$ C-face title, B-face factorial as the §4 master table.

---

## 3. Experimental design

### 3.1 The factorial

| | P-flat (full-horizon CEM) | P-fixH ($G$ + fixed $H$ grid) | P-certH ($G$ + $H^*(\epsilon)$) |
|---|---|---|---|
| **R-eq** — $C_{16}$-steerable pixel JEPA (`eqjepa.py::SteerableEncoder` + `train_jepa`) | ▢ | ▢ | ★ spearhead cell |
| **R-plain** — parameter-matched CNN ladder, 3 sizes (step51 protocol) | ▢ | ▢ | ▢ |
| **R-lewm** — frozen official LeWM (step91 strict-load, zero training) | ▢ (FF-JEPA's failing baseline) | ▢ ($H{=}25$ = FF-JEPA replication cell) | ▢ |

Fixed-$H$ grid $\{5,10,15,25,40\}$ (FF-JEPA's 25 included). **Anti-explosion rule (pre-registered cell
priority):** full $3\times3$ only on PushT-static; PushT-dyn runs the C3 spine + selected closed-loop
cells (eq/plain $\times$ fixH/certH); 3-D co-anchor runs H1-original + C3-3D + a minimal closed-loop
pair.

### 3.2 Environments — including the dynamics knob (the potential signature figure)

1. **PushT-static** ($\lambda_1\approx 0$, step91-measured): the neutral-regime anchor, FF-JEPA
   comparable.
2. **PushT-dyn($\kappa$)**: damping/momentum parameterization of the same task — dial $\kappa$ to
   move $\lambda_1$ from $0$ into the expansive regime *with task semantics held fixed*.
   **Signature-figure candidate:** $x$-axis $\hat\lambda_1(\kappa)$, $y$-axis measured
   $H_{\max}(\epsilon)$ vs certified $H^*(\epsilon)$ — *subgoal spacing is a function of the
   environment's spectrum, and the certificate gives it a priori*. This is the law-grade plot a 9+
   paper hangs on, and it plugs directly into paper2's two-sided horizon theorems.
   **W1 validation gate (before betting the spine on it):** $\hat\lambda_1(\kappa)$ monotone and
   resolvable (CI-separated) across the $\kappa$ grid. **Fallback:** two-point regime contrast
   (static vs one dynamic setting) still stands.
   **Demo regeneration (the knob's hidden chain-cost):** changing the physics invalidates the static
   human demos — each $\kappa$ needs its own demonstration set, generated by a **ground-truth-state
   MPC/scripted controller** (the repo's state-based planner line). W1 recon verifies this per-$\kappa$
   generation works. **Fairness rule (pre-registered):** all hyperparameters tuned at $\kappa{=}0$
   only, frozen across the sweep; $\kappa$ grid kept small (3–4 points + static).
3. **3-D co-anchor** (promoted from stretch; descope rule below): ONE task from
   {ManiSkill, LIBERO-spatial} (W1 recon decides), point-cloud or multi-view observations, small
   SE(3)-equivariant encoder (vector neurons or e3nn-lite; W1 recon decides). Runs PoR-original H1
   (7-DoF action rotation dims), C3-3D, and a minimal closed-loop pair.
   **Descope rule (pre-registered, no agonizing):** if 3-D bases are not training by **2026-08-05**,
   the 3-D track reverts to stretch and the paper ships as "2-D, two spectral regimes."

### 3.3 Data

Standard PushT demonstration set (~200 episodes, lerobot/stable-worldmodel pipeline — W1 recon pins
the exact count/source). Three variants: full; fraction sweep $\{1\%,10\%,100\%\}$ (feeds C2);
**wedge-restricted orientations** (the paper2 Exp 7/9/step51 wedge protocol) for OOD cells.

### 3.4 Experiments

| ID | Feeds | Design (gates in §2) |
|---|---|---|
| **E-P4.1** | C2 | Probe rig: R-eq vs R-plain ladder $\times$ demo fractions; frozen encoders; state-pose + effect probes; per-dim curves; 3 seeds. 3-D version (H1-original) when E-P4.5 bases land. |
| **E-P4.2** | C1 | `wm_audit` gap mode on all three bases along the demo distribution; zero-calibration (eq) vs calibration-needed (plain). **C1b** wedge-transfer cells on PushT-static (W2–4); **C1a** spectral cells on PushT-dyn($\kappa{>}0$), riding the E-P4.3 $\kappa$ lane (W5–7). |
| **E-P4.3** | C3 | **The spine.** Per base: train $G$; measure $\mathrm{Err}(H)$ growth; certificate curve + Prop 10 CIs; $H^*(\epsilon)$ vs $H_{\max}(\epsilon)$ over the $\epsilon$ grid **and over $\kappa$**; optimizer-limb isolation w/ ground-truth subgoals (crossover location) — **doubles as the $\hat\theta^\ast$ elicitation consumed by E-P4.4's certified-$H$ cells (Prop 11(ii))**; wedge transfer eq-vs-plain. |
| **E-P4.4** | C4 | Closed-loop factorial: cells per §3.1 priority $\times$ {in-dist, OOD-orientation} $\times$ demo budget $\{100\%,10\%\}$; success @ $t\in\{25,75,150\}$; 3 seeds $\times$ ~50 episodes/cell. |
| **E-P4.5** | C2/C3 | 3-D co-anchor per §3.2(3): SE(3) bases, H1-original probes, C3-3D, minimal closed-loop pair. Descope checkpoint 08-05. |

### 3.5 Instruments to build (by effort, descending)

1. **`wm_audit` gap mode** (the big one): extend the free-running Benettin audit to
   action-conditioned, demo-distribution-windowed estimation of
   $(\hat\delta,\hat\lambda_1,\hat\epsilon_{\max})$.
2. Pixel-PushT pipeline (render + demo loading; stable-worldmodel base) + PushT-dyn($\kappa$) variant.
3. Plain-CNN parameter-matched ladder (trainer exists); subgoal predictor $G$ trainer (per base).
4. Subgoal-conditioned **short-window CEM** (the E-P4.3 instrument — needed W5, before the full
   planner) on the step10/12 base; then the full hierarchical integration ($G$ + windowing + $H$
   pluggable: fixed or certified, per-state adaptive) for E-P4.4 at W8.
5. Probe rig (PoR-style, fixed architecture, frozen encoders).
6. 3-D track: env + point-cloud SE(3) encoder + JEPA training (the only genuinely GPU-hungry item).
7. **Equivariance unit tests for every new equivariant module** (repo standing rule, no exceptions).

---

## 4. Theory items (all standing on paper2's shoulders; cite, don't re-prove)

| ID | Statement (sketch) | Cost |
|---|---|---|
| **P4.A — selection-rule theorem** (the paper's own main theorem) | Two-sided guarantee for certified spacing: under calibrated $(\hat\delta,\hat\lambda_1)$ with Prop 10 finite-sample CIs, (safety) the gap error at $H^*(\epsilon)$ exceeds $\epsilon$ with probability $\le\delta'$; (efficiency) the implied subgoal density exceeds the oracle's by at most an explicit factor driven by the CI width and the two-sided tightness of the horizon law (Thm B + Prop 6). Prop 9's budget law lifted from the sensing layer to the planning layer (the mapping is verbatim: subgoal $=$ re-observation, spacing $=$ cadence, subgoal count $=$ budget $B$); Prop 11 supplies the alignment premise (spacing is an aligned decision $\Rightarrow$ zero extra regret at $c{=}1$) and the $\theta^\ast$-elicitation requirement that E-P4.3 discharges. Assumptions inherited from Prop 7/8 (stationarity/mixing along audited orbits; learned-model exponent transfer with its honest $C^1$-proxy caveat) — stated, not certified. Neutral-regime cells use the $\hat\delta$ CI variant (same Bernstein-for-mixing machinery as Prop 10; spelled out in P4.A's writeup). | low (inheritance chain: Thm B intermediate bound + Prop 6 + Prop 9/10/11, scope via Prop 7/8) |
| **P4.B — orbit-transfer corollary** | For equivariant $E,f,G$ with orthogonal $\rho$: the gap functional is orbit-constant (Thm A applied to $\mathrm{Err}(H)$), hence $\widehat{\mathrm{Err}}(H)$ estimated on a wedge certifies the entire orbit. *The mathematical body of the flag-planting sentence.* | trivial |
| **P4.C — equivariant $G$ decomposition** (design principle + falsifiable prediction) | An equivariant subgoal flow decomposes into a group component and an invariant-content evolution; prediction: FF-JEPA's failure mode (b) ("subgoal loses the agent") is a symptom of unstructured extrapolation, and the decomposition removes that failure class. | low |

---

## 5. Timeline (14 weeks to ICLR; today = 2026-06-10)

| Weeks | Milestones |
|---|---|
| **W1** (06/10–17) | Proposal freeze. Three-way recon: (a) pixel-PushT pipeline + exact demo set; (b) $\kappa$-knob design + **knob validation gate** + per-$\kappa$ GT-state demo generation; (c) 3-D env + encoder selection. First build spec. |
| **W2–4** (–07/08) | PushT bases trained (eq + plain ladder); probe rig + E-P4.1; `wm_audit` gap mode; E-P4.2. PushT-dyn built. **3-D track starts in parallel** (the GPU lane). |
| **W5–7** (–07/29) | $G$ training; short-window subgoal-conditioned CEM (E-P4.3 instrument); **E-P4.3 spine incl. $\kappa$ sweep**; P4.A/B written up; 3-D bases + H1-original probes. |
| **W8–10** (–08/19) | Hierarchical planner; E-P4.4 factorial; 3-D minimal cells; **08/05 = 3-D descope checkpoint**; multi-seed consolidation. |
| **W11–12** (–09/02) | **arXiv flag-plant as soon as C3 lands.** Stretch-or-consolidate; ablations; writing starts. |
| **W13–14** (–deadline) | Full draft, figures, paper2-style compression discipline, submission. September bandwidth shared with paper2 final packaging. |

---

## 6. Risk register (pre-registered countermeasures)

1. **3-D infrastructure sinkhole** (the biggest): 08-05 descope rule — automatic, no agonizing, no
   gate-loosening.
2. **$\kappa$-knob fails to dial a clean spectrum, or per-$\kappa$ demo generation fails** (contact
   dynamics are noisy; demos must be regenerated per physics setting): W1 validation gate *before*
   the spine bets on it — covering both the spectrum dial and GT-state demo generation; fallback =
   two-point regime contrast.
3. **Resampling floor** ($C_N$ off-grid $\sim10^{-1}$, documented in-repo): large $N$ ($C_{16}$),
   report the floor honestly; certificates land in Thm B's approximate regime with $\epsilon_{\max}$
   measured (on-thesis, not a bug).
4. **C4(ii) repeats step93**: quarantined by design; C3 does not die with it; a miss is scope-law
   evidence and gets reported as such.
5. **September double-paper crunch**: arXiv flag-plant early; the factorial is insurance — partial
   survival still ships (title pivot rule).
6. **FF-JEPA group scales first** (they have robots): our moat is the two layers they lack
   (equivariance + certificates); speed is the hedge — W1 starts now.
7. **Honest negative possibility:** certified-$H$ materially worse than oracle fixed-$H$ in closed
   loop $\to$ reported; the load-bearing claim is the feasibility boundary (C3), the practical-value
   cell is C4(ii) and stands or falls on its own.

---

## 7. Boundaries and citation discipline

- **vs paper2:** paper2 = the certificate itself (mechanism/theory); paper3 = the certificate as a
  *component* of an embodied hierarchical planner. E-P4.1/E-P4.4's pixel + closed-loop cells double as
  evidence for paper2's named frontier (scale-lift S1/S2) — cross-pollination, separate papers.
- **vs FF-JEPA:** no public code $\to$ **no head-to-head claim**; we replicate the *effect* (flat
  collapses, hierarchy rescues) on our stack and cite their numbers as cross-protocol context with
  their own footnote discipline. Their $H{=}25$ enters as a grid point.
- **vs PoR:** protocol *adapted* (state-pose + effect probe on PushT; original action-rotation probe
  on the 3-D co-anchor); always labeled "adapted"; their Dreamer-4 negative cited only with the
  non-official-reproduction caveat.
- **vs LeWM:** frozen official checkpoints only (step91 pipeline); we are a consumer of their base, an
  auditor of their loop, and now a *retrofitter* of their planning layer.

## 8. Bookkeeping

- Experiment ledger: `papers/paper3_record.md` (to be created at W1 first build, paper2_record
  format, honest-verdict discipline).
- Specs: `docs/specs/2026-MM-DD-p4-stepN-*.md`, recon→spec→build cadence unchanged.
- Vault bridge: snapshot page `raw/papers/研究线-certified-world-models-快照.md` updated (Phase 4
  section added; paper-pipeline review hooks extended to hierarchical-planning / subgoal / SE(3)
  manipulation-representation papers).
- This session (vault/paper-pipeline) is territory-upgraded to write this repo for Phase 4 work, per
  user decision 2026-06-10.
