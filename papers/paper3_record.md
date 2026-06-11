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

## [2026-06-10] P4-step1b — diagnosis pass: both first hypotheses REFUTED; the truth was better

(`experiments/p4_step1b_diagnosis.py`, `papers/figures/p4_step1b_diagnosis.json`, 2×~6 min; bases
re-materialized by seed determinism.)

1. **aug collapse — first hypothesis (encodes the applied angle) REFUTED** (angle-$R^2$ −0.07),
   which forced the real mechanism into the open: `augment_x4` drew rotation angles **per copy,
   not per sample** — the entire augmented dataset contained 4 discrete angles, each with a
   zeros-padding corner-wedge signature ⇒ **4-cluster collapse** (high variance across clusters,
   o/o₂ same cluster ⇒ pred_loss→0, zero content; the continuous-angle probe is OOD for it, mod-4
   readout 0.49). An *implementation artifact*, not the Brehmer control's substance — honest
   distinction recorded. (Side data: ALL bases read the probe's own corner artifact at mod-4,
   0.49–0.84 — the probe's rotation introduces the cue; eliminated in v1.2 by circular masking.)
   **step1c direct verification (CONFIRMED at three levels)**
   (`experiments/p4_step1c_cluster_check.py`): between/within-copy variance ratio **686,393**
   (latents are 4 near-point clusters); copy-identity probe accuracy **0.742 ≈ 3/4 = the
   merged-pair signature**; the merged pair is quantitatively explained by the corner-wedge
   arithmetic — drawn angles mod 90° = $\{0.0, 88.7, 6.6, 31.9\}$°, so copy-1 (178.7° ≈ 180°,
   which needs no padding on a square image) sits 1.3° from copy-0 in wedge space and the two
   clusters coincide. Mechanism, not just structure, verified.
2. **θ unreadability — BOTH prior hypotheses dead**: corpus rotation is ample (median 70.6°
   within-episode, q90 357°); longer training makes it WORSE (θ +0.03→−0.40 at 60 epochs) and
   **decays xy content too** (0.70→0.44 — the @200 column's drop reproduced at fixed data). Live
   hypothesis: 1-step prediction at 10 Hz (~0.7°/step) never needs orientation. **Response =
   protocol v1.2** (registered in the step1 spec before any 3-seed run): stride-5 action-chunked
   transitions (LeWM/FF-JEPA convention), per-sample-angle + circular-mask R-aug, epochs held at
   20 with probe-vs-epoch curves logged (content-decay registered as a monitored axis — relates to
   Step-64's predictability–variance tension; possibly reportable, not yet theorized).
3. eq's xy dominance reproduced under re-materialization (0.703 vs grid's 0.709) — the loud signal
   is stable.

## [2026-06-11] P4-spine Stage-1a first contact — a deployment-pairing catch, then an OPEN instrumentation incident (all pixel-eq audit numbers QUARANTINED)

The first certified-vs-measured run on real bases turned into an instrumentation investigation —
recorded in full because every step is evidence:

1. **The pairing catch (real, stands):** the deployable JEPA world model is **(EMA-target encoder,
   predictor)** — train_jepa regresses the predictor onto the target encoder's latents; the v1.2
   checkpoints saved only the online encoder. Decisive in-process check: δ̂(online pair) = 12.20 vs
   **δ̂(target pair) = 0.82 ≈ the train pred_loss norm 0.80** (15×, exact closure). Checkpoints
   extended to save both halves (`ckpt2_*`).
2. **Then the contradiction:** the SAVED target pair is inconsistent too — δ̂ = 12.15 held-out
   **and 11.95 on the training prefix** (not a memorization basin). State-dicts verified distinct
   (online↔target relative distance 0.35, keys 1:1); load path verified exact (deepcopy ≡ fresh
   ≡ ≠ online).
3. **One-process round trip (40-ep run): the round trip CHANGES the module** —
   ‖tgt_inproc(x) − tgt_reload(x)‖ ≈ 16 at scale ~40; and that run's in-process pair read 84.8
   (vs the 200-ep run's 0.82) — in-process behavior is itself run-dependent.
4. **Leading mechanism (explains 1, 2, and the 0.82↔12 flip; does NOT yet explain 84.8):**
   `train_jepa` puts the target in `.eval()` then EMA-updates its **parameters** — but e2cnn's
   `R2Conv` caches the expanded filter at `.eval()` time and **parameter updates do not refresh
   the cache**. In-process target behavior = stale-cached filters (what the predictor actually
   trained against); its state_dict = EMA weights; reload re-expands from those ⇒ a different
   function. Also explains why the eval-mode deepcopy of an e2cnn module crashes (cached non-leaf
   tensor). **Upstream-relevant:** paper2's S1 (pixel scale-lift) uses the same train_jepa +
   SteerableEncoder pair — flagged to the research line via the snapshot.
5. **Quarantine:** both Stage-1a artifacts (`p4_spine_stage1a*.json` — pairbug and "fixed" rerun)
   are descriptive of MIS-PAIRED or cache-ambiguous models; **no pixel-eq audit number is usable
   until the target-encoder handling is redesigned e2cnn-aware** (refresh cache after EMA / keep
   functional form explicit) and verified by an in-process == reloaded equality test. plain bases
   (no e2cnn) are less exposed but get the same verification before use.
   Kept honest positives: the instrument-bias calibration at spine settings (uniform −0.029,
   W=16/B=4) and the analysis-code note (linear fit needs intercept) survive.

*Same-day review addendum:* the mechanism survives a consistency audit with one refinement — the
eval cache must be a **non-persistent buffer** (the only form consistent with all four
micro-evidences: moves with `.cpu()`/`._apply` so CPU forwards don't crash; excluded from
state_dict so strict loads pass against cache-free modules; non-leaf so eval-mode deepcopy
crashes; stale after parameter-only EMA). The 40-ep run's in-process 84.8 remains the open
residual for the matrix experiment.

## [2026-06-11] Stage-2κ — the κ=0.8 lane: J≡I dies, the moat collapses to parity, and G-pre fires in the OPPOSITE direction from step99

(`experiments/p4_spine_stage2_kappa08.py`, 3.5 min total — collection is actually ~15 s for 200
eps; the "collection 13 min" figure in the part-3 entry was the ORACLE gate's 815 s, corrected
here. Artifact `p4_spine_stage2_kappa08.json`; bases `ckpt3_*_k08.pt`, #9 fix on, κ=0
hyperparameters frozen per the fairness rule.)

**Registered expectations vs reality (declared in the script docstring before the run):**

1. **J≡I is DEAD — both predictors learned z-sensitivity** (zsens: eq ‖∂Δ/∂z v‖ mean 1.38, plain
   0.49). My 0.6-confidence prediction that action-blindness persists was WRONG: the momentum
   regime forced state-dependence. C3-zsens label: *z-sensitive* both bases.
2. **The moat quantity COLLAPSED to parity at κ=0.8**: δ̂_eq = 4.15 vs δ̂_plain = 4.45 (was 1.47
   vs 7.68 = 5.2× at κ=0); pred_loss near-equal (0.177/0.183) and 38× worse than eq's κ=0 value.
   Mechanism (the registered single-frame suspicion): **block velocity is unobservable in one
   frame ⇒ the latent dynamics are non-Markov at κ>0** — unpredictable drift dominates BOTH
   bases equally. Honest finding shape: *the equivariance predictability advantage lives where
   geometry determines dynamics; momentum hides state from single frames and blinds both.*
   Registered response: **v1.4 candidate = frame-pair encoder input** (2-frame channel stack;
   both frames rotate together ⇒ equivariance trivially preserved) — restores Markovianity and
   TESTS whether the eq advantage returns with velocity observable. The highest-value next
   experiment.
3. **G-pre fires, opposite direction from step99**: r = λ̂_meas/λ̂_tangent = **7.2 (eq), 2.5
   (plain)** — both outside [0.5,2] ⇒ FAILED-BY-SCOPE at κ=0.8/single-frame. Where step99's 1B
   model's tangent OVER-promised (0.178 vs measured 0.033), ours UNDER-promises: with correlated,
   state-dependent bias (the model systematically lags moving blocks), measured error compounds
   faster than the Jacobian channel predicts. **The jurisdiction map now has edges on BOTH
   sides** — tangent certificates can over- or under-promise depending on which premise breaks
   (linearization neighborhood vs weakly-correlated residuals). Cross-paper material with E16.
4. **Measured curves SATURATE** (3.59→7.59, decelerating increments; r²_lin 0.90 vs r²_exp 0.83 —
   neither clean shape): at κ=0.8/single-frame the error process is fast correlated early growth
   → ceiling, not the exponential regime the theory wants. The fitted λ̂_meas is window-dependent
   — the G-pre r above is therefore directionally robust but numerically fragile; stated.
5. **C3-regime δ̂-channel ordering holds** (certified horizons shrink static→0.8: eq δ̂ 1.47→4.15
   ✓), shape channel INCONCLUSIVE (saturation). C3-cal (quantile-matched) mixed: in-band at the
   ε extremes, conservative at mid-ε — seed-0 descriptive.

**Net:** the lane did exactly what it was built to do — every prior got updated, two registered
response paths opened (frame-pair v1.4; jurisdiction-both-edges as paper material), zero gates
loosened.

*Same-day review correction (the "parity" claim was metric-naive):* raw cross-base δ̂ live in
different latent metrics — the very tension flagged at Stage-1b, not applied to my own headline.
Under the now-REGISTERED normalizer (per-dim latent_std × √D ≈ typical latent norm):
κ=0 → eq 0.132 vs plain 0.824 (**6.3×, the moat survives normalization**); κ=0.8 → eq 0.653 vs
plain **0.406 — not parity but INVERSION (eq 1.6× worse)**. Two confounds stated: (i) eq's κ=0.8
latent_std = 0.562 — the training recipe itself partially collapsed there, so the inversion mixes
"equivariance × momentum" with recipe instability; (ii) the normalizer choice is itself a
registered, reportable decision. The v1.4 frame-pair experiment now answers a sharper question:
*with velocity observable and the recipe stable, which side of 1× does the normalized ratio land
on, in each regime?*

## [2026-06-11] #10 (half 2) — Stage-1b planner side: feedback planning BEATS the open-loop boundary; θ̂* protocol v1 caught vacuous at κ=0

(`experiments/p4_spine_stage1b.py`, 11 s; artifact `p4_spine_stage1b.json`. Fixed-goal windows
(seed 2, `FIXED_GOAL` shared with G-training) with a same-windows measured curve — the Stage-1a
random-goal sets are not re-entered; apples-to-apples crossover by construction.)

1. **$H^{\text{plan}}_{\max} \ge H^{\text{model}}_{\max}$ at every ε, both bases** (eq: 3/8/8 vs
   2/5/8 at ε ∈ {2,4,8}δ̂) — **feedback planning corrects model bias**; the optimizer limb does
   not bind at κ=0 with chunked actions (the registered crossover sits at/above the grid
   ceiling). C3 implication, strengthening: the certified open-loop curve is a *conservative
   bound* on closed-loop reach — certificate-as-guarantee, not certificate-as-estimate.
2. **Cross-base ε tension made concrete:** plain's per-base "2δ̂" threshold is 6× eq's in
   absolute terms (δ̂ 6.97 vs 1.09 on these windows) — its boundaries saturate trivially. The
   honest cross-base statement stays the δ̂ ratio + per-base calibration (C3 is per-base by
   design; unaffected). Absolute cross-space thresholds are apples-to-oranges (different
   metrics) — not introduced.
3. **θ̂* elicitation v1 is VACUOUS in the quasi-static regime — caught by its own sweep:**
   reach-rate 0.80–0.97 flat across ν ∈ {0..16}δ̂ (both bases) ⇒ θ̂* = sweep ceiling. Cause: at
   H=3 with weak-poke windows, $z(t{+}3) \approx z(t)$ — ANY mild action sequence lands within
   ε_reach=4δ̂ of the true future, independent of the subgoal. The Prop-11 *principle* stands;
   the instantiation needs **v2 (registered before any C4 consumption): motion-selected windows
   ($\lVert z(t{+}H) - z(t)\rVert > 4\hat\delta$) + tighter ε_reach (2δ̂) + report the
   no-plan/random-plan control row** so vacuity is visible by construction next time.
4. Reach medians grow gracefully for eq (0.68δ̂ → 2.5δ̂ over H=1..8); CEM (K=64×3) on the f64
   deployable pair plans 8-chunk windows in ~10 ms each — the planner stack is cheap.

**Stage-1 seed-0 is now complete end-to-end** (1a audits + 1b planner) on certified instruments:
the remaining spine work is 3-seed replication, the κ=0.8 lane (bases + audits at the expansive
point), wedge cells, and the v2 θ* protocol.

## [2026-06-11] #10 (half 1) — $G$ trained on human demos via state-re-rendering; both G's barely beat persistence (honest, registered-descriptive)

(`experiments/p4_step10_g_trainer.py`, 18.5 s end-to-end; artifact `p4_step10_g.json`.)

- **Demo ingestion design**: lerobot/pusht parquet exposes only the 2-d agent state (unusable) ⇒
  source = the canonical diffusion-policy `pusht_cchi_v7_replay.zarr` (206 eps, full 5-d pose;
  new venv-only dev dep `zarr`). **State re-rendering in OUR env** at chunk boundaries
  (`reset(options={'goal_state': FIXED_GOAL, 'state': s5})`, goal pinned for pixel consistency) —
  kills rendering mismatch AND replay drift; $G$ is autonomous, demo actions unneeded. 206
  episodes re-rendered in 7 s.
- $G$'s trained in the **ckpt3 TARGET-encoder space** (deployable pair): plain-G residual MLP;
  eq-G = GroupConv stack on stacked regular fields (P4.C by construction; **exact equivariance
  assert < 1e-10**).
- **Readouts (descriptive by registration):** eq-G δ̂_G = 1.019 vs persistence 1.089 (+6%);
  plain-G 6.70 vs 8.01 (+16%). Both G's are weak against persistence — the human-demo subgoal
  flow is genuinely multimodal at 0.5 s boundaries; single-point prediction is the wrong-shaped
  tool (the same limitation FF-JEPA's $G$ faces; their failure mode (b) lives here). Exactly why
  Stage-2 was registered reported-not-gated. Conflation note stands (base-space × G-architecture).
- Analysis-code fix landed same-commit: `fit_growth` linear fit now carries an intercept (the
  exp-vs-linear comparison is meaningful going forward).
- Stage-1b (planner side: subgoal-conditioned CEM, $H^{\text{plan}}$, crossover, $\hat\theta^\ast$)
  = next session's blade.

## [2026-06-11] #9 RESOLVED — e2cnn-aware pairing fix lands, equality gates 3/3, quarantine lifted; first CLEAN certified-vs-measured numbers

**Mechanism, source-confirmed** (e2cnn `r2convolution.py`): `train(False)` re-expands the filter
cache **only `if self.training`** — an already-eval module's stale cache survives `.eval()` calls
silently; the cache is a re-registered buffer (in state_dict when eval-mode). So: in-process
target = init-frozen filters (predictor's true regression object); reload into a fresh train-mode
module + `.eval()` re-expands from the EMA weights ⇒ a different function. Every observation of
the incident is now derived from source.

**Fix:** `train_jepa(..., refresh_target_cache=True)` — a `train()/eval()` cycle on the target
after each EMA update (re-expansion from current weights; no forward between flips ⇒ BN stats
untouched). **Additive flag, default False — paper2-era behaviour bit-preserved** (their S1 call
sites unchanged; snapshot already carries the warning).

**Equality gates (tests/test_p4_step9_pairfix.py, 3/3 PASS):** E-I fix: function gap **exactly
0.0** across save/load, δ̂ identical to all digits; E-II teeth: legacy path reproduces the bug
(gap 0.374) — permanent forensic control; E-III plain: exact round trip. Bonus: the fix improves
pairing quality itself (legacy in-process δ̂ 13.6 → fixed 3.05 at 20 eps — the predictor finally
regresses a coherent moving target).

**Stage-1a CLEAN rerun (ckpt3 pairs, seed-0 descriptive,
`papers/figures/p4_spine_stage1a.json`; ckpt2-era artifacts archived as `*_quarantined2`):**

- **The moat quantity appears: δ̂_eq = 1.47 vs δ̂_plain = 7.68 (5.2×) at matched params on
  held-out data** — equal-ε certified horizons will be ~5× longer for eq. This is C3's
  load-bearing input, now on certified instruments.
- **Neutral-regime certificate shape CONFIRMED for eq**: measured Err(H) grows LINEARLY
  (r²_lin 0.98 vs r²_exp 0.90; 1.10 → 5.87 over H=1..8), exactly the δ̂·H form the κ=0
  certificate predicts; measured per-chunk rate 0.77 vs certified 1.47 — conservative ~1.9×,
  within the registered band. C3-cal (q90 consumer): ratios 0.33/0.29/0.50/1.00 over the ε grid —
  conservative at fine ε, in-band at coarse (q90-vs-median directionality is by construction).
- eq λ̂₁ = +0.0000 with degenerate CI — neutral as the env (cross-consistent with step2's κ=0)
  but suspiciously exact (identity-dominant residual predictor?); flagged for a precision look.
- plain: bit-identical to its ckpt2 run (no e2cnn ⇒ flag is a no-op; also confirms MPS run
  determinism here); its exp-vs-linear fit comparison still carries the known no-intercept
  analysis bug — fix pending, conclusions not drawn from it.

*Same-day review addenda:*

1. **Closure both ways:** δ̂_eq held-out 1.47 vs train-loss norm 1.62; δ̂_plain 7.68 vs 6.72 —
   both bases generalize cleanly, so the 5.2× gap is a genuine fit-quality gap, not a
   generalization artifact.
2. **The λ̂=0.0000 mystery solved by direct probe:** $\partial\Delta/\partial z = 0$ to machine
   precision (8 random JVP directions all 0.0; $\Delta(z')-\Delta(z) = 2\times10^{-16}$) — the eq
   predictor trained to a **purely action-driven** solution $f(z,a) = z + \Delta(a)$ ($J\equiv I$;
   quasi-static PushT makes this near-optimal, and the frequency-1 action lift makes it cheap to
   express). Registered consequences: new **C3-zsens pre-gate** in the spine spec (λ̂ labelled
   *structural* when $J\equiv I$; expansive-regime under-amplification named in advance); plain's
   λ̂ (−0.006, proper CI) shows it DID learn mild z-dependence — an asymmetry worth tracking.
3. **C3-cal gate amended to like-for-like quantiles** (q90↔q90, median↔median reported) — the
   seed-0 fine-ε "failures" (0.29–0.33) were the q90-vs-median construction, not calibration.
4. **Instrument-bias note:** the planted −0.029 offset is a *transient-alignment* bias — it
   applies to gapped spectra, NOT to $J\equiv I$ bases (no transient exists); bias accounting is
   regime-dependent and stated per base.

## [2026-06-11] G0c oracle v3 — 4/20 IDENTICAL to v2: the bottleneck is structural, not budget; gate re-scoped

Doubling the budget (h 12→16, K 64→96, cap 200→300; 23 min) changed nothing — the same 4
configurations succeed. Reading: greedy shooting MPC cannot cross the "re-position the pusher to
another face" cost plateaus that pose-matching PushT requires; this is exactly why the field uses
demonstrations/diffusion policies here. **Registered re-scope instead of solver iteration #4:**

- **κ=0 $G$-training demos = the human lerobot set (206 eps)** — ecosystem-comparable
  (FF-JEPA-style), no oracle needed. (Spec change registered in the spine seed spec; legitimate —
  no $G$ has been trained.)
- **κ=0.8**: Stage-1 needs NO demos (GT subgoals from WeakPolicy trajectories). Stage-2 at κ=0.8
  (already "reported, not gated") is **deferred behind the incident fix**; if wanted later, the
  honest options are rejection-sampling at the measured 20% acceptance (~19 h overnight) or a
  BC-bootstrap generator (recon first).
- G0c the *gate* is retired in favour of this re-scope; the oracle code remains as a utility.

## [2026-06-10] P4-step1 v1.2 grid readout — not a clean win: a deeper confound surfaced, and one mechanism now explains the board

(`papers/figures/p4_step1_v12.json`, 8 min wall, MPS, **25/25 checkpoints saved** — the supply
line for gap audits and $G$-training is open.)

- **G3 STILL FAIL** — θ unreadable in every base under stride-5 (best −0.01). The "1-step never
  needs orientation" hypothesis was insufficient *alone*.
- **NEW: training-budget confound** — chunking cut transitions/cell 5× while epochs stayed 20 ⇒
  small-fraction cells are step-starved (std 0.002–0.16 at @2–@20); @60/@200 (most steps) have
  healthy std but **xy turned negative** (eq −0.17/−0.16 vs v1.1's +0.55/+0.31). Jointly with the
  monitored content-decay axis, ONE mechanism explains v1.1+v1.2 coherently: **linear content
  peaks early in training and decays; cells differ chiefly in optimizer steps, contaminating the
  fraction axis.** The v1.1 "eq from 2 episodes" reading survives qualitatively (eq leads xy at
  every comparable cell: 0.55/0.57 at @10/@20 vs plain ≤0.06) but quantitative fraction curves
  from BOTH grids are confounded — neither enters any claim.
- **aug**: structural 4-cluster collapse FIXED (healthy std, no clusters) — but pred_loss is still
  suspiciously low (0.0012 vs plain 0.35) with zero content: per-sample drawing left the angle a
  *continuous* shared channel (scene orientation: walls/goal zone survive the mask). Check queued.
- **eq's predictability edge WIDENED under chunking**: pred_loss 0.005–0.010 vs plain 0.24–0.68
  (70–140×) at matched params and healthy std (@60/@200). Flagged, not yet interpreted (the aug
  case proves predictability alone ≠ content; eq has both only where probes are positive).
- **Response = protocol v1.3 (registered in the step1 spec before any 3-seed run):** equal
  optimizer-step budget per cell (3,000 steps; the data axis becomes pure), probe-vs-step curves
  at {300, 1000, 3000} as primary instrumentation (the decay axis measured, not suspected;
  no early-stop-by-probe rule — circularity guarded), aug continuous-angle check.

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

