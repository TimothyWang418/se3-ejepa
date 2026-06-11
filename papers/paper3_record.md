# paper3 experiment record — Certified Subgoal Spacing for Equivariant World Models

> Honest-verdict ledger, paper2_record format. Registered claims + gates:
> `papers/proposals/paper3-phase4-certified-subgoal-spacing.md` (protocol v1.1).
> Novelty sweep: `papers/proposals/paper3-novelty-sweep-2026-06-10.md` (4/4 SAFE).
> W1 recon: `docs/specs/2026-06-10-p4-w1-recon.md`. No gate is loosened, ever; INCONCLUSIVE is
> reported as such.

## [2026-06-10] P4-step1 (part 1) — the missing equivariant predictor lands EXACT; G0a confirms the κ lane's load-bearing assumption

**The gap the build surfaced first:** Theorem A needs $E$ *and* $f$ equivariant; the repo had the
$C_N$-steerable encoder but **no predictor for the regular-representation fiber** (the
`LatentPredictor` is a plain MLP; the VN family implements a different group action —
$\mathrm{SO}(3)$ on 3-vector channels). Built `src/models/cn_regular.py`:

- `GroupConv1x1` — the most general $C_N$-equivariant linear map on regular fibers (per-field-pair
  circulant = group convolution; Cohen & Welling 2016), implemented as conv1d with circular padding
  over the group axis; pointwise ReLU between layers (permutation-commuting).
- `CNRegularPredictor` — residual, action-conditioned, `LatentPredictor`-interface-compatible
  (drops into `EqJEPA` unchanged). Action conditioning via the **frequency-1 lift**
  $A_a[n] = a_x\cos(2\pi n/N) + a_y\sin(2\pi n/N)$ (intertwines the planar rotation of the action
  with the regular shift; proof in the module docstring) + an invariant $\lVert a\rVert$ channel.

**Verdicts (tests/test_p4_step1.py, all PASS):**

1. **Fiber equivariance EXACT**: $\max_g |f(\rho(g)z, R(\theta_g)a) - \rho(g)f(z,a)| =
   2.4\times10^{-7}$ over all of $C_{16}$ — float precision, no interpolation floor (unlike the
   encoder, no pixels are involved). The predictor side of Thm A's premise costs *zero* residual.
2. **Full-stack sign calibration** (the convention trap, killed empirically): e2cnn's fiber
   permutation pairs with action rotation **sign $s=+1$** (CCW), max err $3.6\times10^{-7}$ on the
   $C_4$ grid (interpolation-free angles isolate the sign question). The pipeline and the R-aug
   augmentation must use $s=+1$; the test pins it.
3. **G0a env smoke**: `gym.make("swm/PushT-v1", resolution=96, damping=0.95)` →
   `space.damping == 0.95` — **the κ-knob kwarg pass-through, the dynamics lane's load-bearing
   assumption, is confirmed**. Default damping $=0$ (quasi-static anchor) confirmed. Obs contract
   pinned: pixels via `render()` (not in the obs dict);
   `state = [agent_x, agent_y, block_x, block_y, block_\theta, agent_{v_x}, agent_{v_y}]` (7-d;
   probe targets = `state[2:5]`); **`relative=True` default — actions are displacement vectors**
   (rotate about the origin, not the canvas centre; predictor/augmentation configured accordingly,
   `action_center=(0,0)`).

**Honest scope registered in-module:** the PushT *environment*'s $\mathrm{SO}(2)$ symmetry is
broken by the square workspace boundary (only $C_4$ exact); wall-contact transitions contribute to
the *measured* $\hat\epsilon_{\max}$ — on-thesis (Thm B absorbs measured residuals).

## [2026-06-10] P4-step1 part 2 — pipeline + smoke + overnight launch

**Pipeline built, smoke end-to-end (13.9 s, bit-identical across runs), full seed-0 grid LAUNCHED
overnight.** `experiments/p4_step1_pipeline.py` (G0b collection → G0c oracle-CEM gate → param
ladder → grid → probes-on-every-cell → JSON). Three smoke nails, fixed honestly and recorded
in-code: (i) vendored `WeakPolicy` is VecEnv-shaped — verbatim single-env replica of its sampling
(cited to file:lines); (ii) vendored `ConvEncoder` requires `width % 8 == 0` (GroupNorm); (iii)
G0c assert demoted to a **recorded verdict** — an infrastructure-gate FAIL must not burn the
overnight grid it does not feed. Resilience: per-base MPS device-probe with honest CPU fallback
(e2cnn-on-MPS unverified). Artifact lands at `papers/figures/p4_step1.json`; read-out + G1/G3
verdicts next session. Spec: `docs/specs/2026-06-10-p4-step1-pipeline-bases-seed.md`.
*Post-launch review note (same day): the in-flight run does NOT save checkpoints (`del model`) —
acceptable for the sizing pass (full determinism: same seed ⇒ same model), but the audit steps
consume trained bases, so checkpoint saving was added to the script for all future runs; the
in-flight process is unaffected (its code was loaded at commit `b0e58e4`).*

## [2026-06-10] P4-step1 part 3 — seed-0 grid readout: two infrastructure FAILs (as-designed responses queued), one loud structural signal, one textbook augmentation-shortcut collapse

Full grid landed (`papers/figures/p4_step1.json`, 52 min wall, **everything on MPS** — e2cnn
included; 3-seed replication sized at ~2.5 h). Param ladder well-matched: eq 163k vs match 165k
(+1.3%), half 144k, double 368k.

**Gate verdicts (registered, zero loosening):**

- **G0b PASS** (260 episodes, shapes/dtypes as contracted).
- **G0c FAIL — oracle-CEM 0/20** (815 s). The horizon-4 greedy position-distance MPC does not
  solve pose-matching PushT (success needs agent+block pos <20 px AND angle <π/9 vs a random
  goal). The verdict-not-crash design did its job (the grid does not consume oracle demos).
  Response (solver identity is registered-free): longer horizon + iCEM + angle-aware cost shaping;
  queued as the next infrastructure fix.
- **G1 PASS on fractions ≥10** (latent_std 0.76–2.1); the @2-episode column is collapsed
  (std 0.007–0.085) — *exactly the degeneracy the two-tier H1 gate pre-anticipated*; @2 cells are
  flagged degenerate, not deleted.
- **G3 FAIL — θ is not linearly probeable in ANY base** (θ-R² negative everywhere, incl. eq).
  Working hypotheses (ranked): (a) WeakPolicy pokes barely rotate the block within episodes —
  one-step JEPA has no incentive to encode a per-episode-constant; (b) 20 epochs undertrains;
  (c) θ is encoded nonlinearly. The spec's pre-authorized response path activates: **protocol
  v1.2 candidates** — mix oracle-demo frames into the encoder corpus (successful demos rotate the
  block purposefully), and/or train longer; to be registered as v1.2 BEFORE the 3-seed claim runs.

**Shapes (seed-0 sizing pass — registered as observations, NO claim verdicts):**

1. **eq's xy-probe dominance is loud**: $R^2_{xy}$ = 0.80/0.80/0.71/0.55/0.31 across fractions vs
   plain's ≤0.12 (mostly negative). The equivariant encoder linearly encodes block position from
   **2 episodes**. (The @200 drop 0.55→0.31 is noted, unexplained — epochs fixed while data grows
   is the lead suspect; 3-seed will tell.)
2. **eq's latent is far more predictable**: pred_loss 0.008–0.017 vs plain 0.055–0.32 at
   comparable latent_std — consistent with C1's mechanism story (and unlike the aug case below,
   eq is predictable AND content-bearing).
3. **aug collapsed to a textbook augmentation shortcut**: pred_loss → 0.0000, latent_std 1.6–2.1,
   ALL probes ≈ 0 (content-free). Diagnosis (to verify with an angle-probe check): the encoder
   encodes the **augmentation angle itself** — o and o₂ share the same random rotation, making the
   angle the most predictable high-variance signal; JEPA + naive rotation augmentation invites
   exactly this shortcut. If confirmed, this is a *reportable observation*: the Brehmer
   augmentation control transfers from supervised learning to JEPA **only with anti-shortcut
   design** — the R-aug arm needs a v1.2 (e.g., angle-marginalized pairs or augmentation only at
   probe time), registered before any C2 reading.

**Sizing conclusions:** full 5×5 grid = 52 min on MPS ⇒ 3 seeds ≈ 2.5 h; collection 13 min;
oracle gate dominated by sim resets (8 s/episode-attempt) — budget fine once the solver solves.

## [2026-06-10] P4-step3 — gap-mode instrument built and CERTIFIED (3/3 gates) before it certifies anything

`src/audit/gap_mode.py`: `audit_gap(model, frames, actions)` → the certificate's consumed triple
$(\hat\delta, \hat\lambda_1, \hat\epsilon_{\max})$ + the certified curve
$\widehat{\mathrm{Err}}(H)$ with CI band. Design facts locked at build: **the loop Jacobian runs
through the predictor only** (the seed spec's e2cnn forward-AD risk is moot — the encoder never
enters the loop), in float64 on a deep-copied predictor; the encoder stays in NATIVE dtype
(lesson: e2cnn's R2Conv cannot be wholesale `.double()`-ed — expanded filter stays f32; and δ̂ at
$10^{-1}$ scale is indifferent); `eps_max = None` for plain/aug (Lemma 2 — no canonical ρ, no fake
numbers).

**Instrument gates (tests/test_p4_step3.py, all PASS):**

1. **G-I planted-spectrum recovery** — the gate caught a real instrument bias on first run:
   without burn-in, planted $-0.05$ read as $-0.076$ (transient alignment of the Benettin band
   contaminates short windows). Fixed with the standard burn-in (B=10); the residual
   finite-window bias at deployed settings (W=40, gap 0.1) is **uniform $\approx-0.007$**
   (planted $-0.05/0/+0.08$ → $-0.0567/-0.0067/+0.0733$), shrinking to $\sim2\times10^{-4}$ at
   W=220 — the gap-mode analogue of Prop 8's finite-$T$ truncation, now *measured* and registered
   as a known instrument bias (reported, not corrected away).
2. **G-II determinism** — bit-identical artifacts across runs.
3. **G-III orbit-invariance witness** — random-weights exactly-equivariant base:
   $\hat\lambda_1$ on $g$-transformed data $+0.1065$ vs base $+0.1040$ (within tolerance), grid
   angle $\hat\epsilon = 2.5\times10^{-6}$ — instrument and equivariance jointly consistent.

Consumption blocked only on step1 checkpoints (overnight grid in flight; original run
re-materializes by seed-determinism). Spec: `docs/specs/2026-06-10-p4-step3-gap-mode-seed.md`.

## [2026-06-10] P4-step2 — κ validation gate: FALLBACK-2PT, honestly — the dial saturates, but TWO regimes are real and resolved

Run concurrently with step1's grid (true-env measurement, no learned model). **Registered
discovery first**: `_set_state` (env.py:514–527) never sets block velocity/angular velocity ⇒ the
7-d obs state **under-specifies κ>0 dynamics**; step2 manages the 10-d extended state via direct
pymunk handles; **registered debt**: any κ>0 oracle-CEM must adopt the extended-state setter
(step1's κ=0 oracle is safe — damping 0 kills velocities).

**Protocol** (spec, pre-registered): κ grid $\{0,0.5,0.8,0.95,1.0\}$; per κ 10 WeakPolicy
episodes; capture points $\{10,25,40,55\}$ × 60-action windows — **as-run n=30/κ, not the spec's
40**: the $t{=}55$ captures never exist ($55+60>100$ = a capture-window arithmetic slip, caught in
post-run review; the fit filter itself dropped 0 pairs — this entry's first version mis-attributed
the 30, corrected same-day). Twin trajectories, block-angle $\delta_0=10^{-4}$; config-space
metric (pos/512, angle/$\pi$); G0 determinism self-check ($\delta_0{=}0$ twins coincide exactly)
PASS at all κ. n=30 is ample for the endpoint-CI gate; any rerun fixing the capture grid is a
registered protocol v1.1.

**Result** (`papers/figures/p4_step2_kappa_gate.json`, 3.3 s):
$\hat\lambda_1$ per control step — κ=0: $+0.006\,[-0.004,+0.017]$ (**exactly neutral**, median
$0.0002$ — independent confirmation of step91's quasi-static reading); κ=0.5:
$+0.043\,[-0.006,+0.076]$ (transitional, grazes 0); κ=0.8: $+0.088\,[+0.059,+0.118]$; κ=0.95:
$+0.076$; κ=1.0: $+0.075\,[+0.036,+0.110]$. **The dial saturates at κ≈0.8** (plateau dips are
within CI noise).

**Gate verdicts (registered, zero loosening): GATE-A monotone FAIL** (Spearman $\rho=+0.60<0.8$ —
saturation, not noise); **GATE-B endpoint resolution PASS** (κ=0 and κ=1 CIs disjoint).
**VERDICT: FALLBACK-2PT** — the spine (E-P4.3) runs the pre-registered two-point regime contrast
(static $\lambda_1\approx0$ vs **κ=0.8**, the best-separated expansive point) instead of a 5-point
λ-dial. The signature figure's x-axis shrinks from 5 points to 2 regimes; if a finer dial is
wanted, a step2b knob-redesign (mass/friction axes) must be **registered before running** — no
silent shopping. Honest upside: both regimes the theory needs (neutral linear-budget vs expansive
exponential) are now *measured* in the true env, with the neutral anchor independently
cross-validated.

