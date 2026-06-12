# paper3 experiment record — Certified Subgoal Spacing for Equivariant World Models

> Honest-verdict ledger, paper2_record format. Registered claims + gates:
> `papers/proposals/paper3-phase4-certified-subgoal-spacing.md` (protocol v1.1 + amendments).
> Novelty sweep: `papers/proposals/paper3-novelty-sweep-2026-06-10.md` (4/4 SAFE).
> W1 recon: `docs/specs/2026-06-10-p4-w1-recon.md`. No gate is loosened, ever; INCONCLUSIVE is
> reported as such. Entries below are newest-first.

## STATUS DIGEST (updated 2026-06-11 evening, post-E0.3 / mid-champion)

- **Protocol:** v1.2 + v1.6 (two-stage tuning, n=10 runs, stability floor std ≥ 0.7) in force;
  v1.3 registered NOT run; v1.4/v1.5 CLOSED (banked config = single-frame κ=0). **C3-guar
  faithful mechanics amendment** (2026-06-11, between champion r1/r2): dual boundaries, direct
  $H_{\mathrm{cert}} \le H_{\mathrm{meas}}$, 0≤0 passes, cal band on both-positive cells only.
  Caveat registered: cells censored at h_max=8 on both sides ((8,8)) are pass-by-censoring —
  verdict language must say "within the audited horizon".
- **Instruments (all certified):** C_N predictor (exact); κ-gate (two regimes); gap mode
  (G-I/II/III, biases −0.007@W40/−0.029@W16); pairing-equality (#9); eq-G; planner stack;
  **VN-JEPA 3D stack (G0: equivariance at 1e-16, pairing bit-exact)**; `to_transitions_lean`
  (OOM-safe, numerically identical). Taxonomy: stable-but-empty vs collapsed-but-contentful.
- **In flight:** champion confirmation (v0.3+aux0.3 × c2000 × n=10): **MPS arm** (registered
  eval; 4/4 guar ✓ so far, cal trending over-conservative-FAIL) + **CUDA replication arm**
  (3080, fresh n=10, device-robustness). aux-health n=10 read: aux0.3_v0.15 = 3/10 stable but
  stable-xy ≈ 0.84; **aux0.5_v0.3 = 7/10 stable, stable-xy up to 0.84 — next champion candidate
  (needs c2000)**. Trade-off curve is now measured, not anecdotal.
- **Claims:** C3 spear-tip: 3-seed PASS*, E0.3 adds **OOD one-sidedness: 109/109 evaluable
  cells ≤ 1 across 4 unseen shapes** (11 vacuous cells under dual-boundary re-verification,
  E0.3b); narrative settling at "guarantee holds (incl. OOD), point-estimate conservative".
  Moat distributional [3.79, 1.10, 2.73] κ=0, inverts κ=0.8 (open frontier). C2 blocked on
  θ-readability (θ-only anchor backfires — night shift). C1b wedge: **data ready** (corpus +
  in/out held-outs). C4 untouched.
- **Fleet:** Mac M5 (MPS×2 + CPU), 3080 box (shared w/ paper2, lock protocol, `.venv3` full
  deps), backup Intel MacBook (papers 2+3 small CPU jobs). 3D lane: G0 done; render = CPU
  (llvmpipe/WSL boundary), train = CUDA; demos downloaded (1000 eps).
- **Next blades (order):** champion verdicts (both arms) → wedge lane (C1b/C3-wedge; data
  banked, CPU free) → aux0.5_v0.3 × c2000 stage-B → 3D protocol spec v1 → v1.3 grid (C2) → C4.

---

## [2026-06-11] plain_ctrl attribution: COUNT, not restriction — the moat is a **low-N data-efficiency moat**; champion champ-side complete: 18/18 faithful guar across two devices

**plain_ctrl (plain × unfiltered@2606, eq_ctrl's exact data, n=4):** 4/4 stable, in-δ̂ =
[4.23, 7.99, 10.66, 10.89], median 9.33 ⇒ the registered branch **"COUNT (data lever at low
N)"** fires. plain explodes at 2606 transitions with or without the wedge restriction
(filtered 10.54 / unfiltered 9.33); at 4000 it sits at ≈ 3.8. eq is flat across all of it
(2.49–3.29). **Relabel: the 4.2× moat is a low-N data-efficiency moat** — halve the data, eq
doesn't notice, plain breaks. This is the 举一反三 thesis in its purest measured form so far
(structure ⇒ sample efficiency), found by an experiment designed to test something else.
Restriction is innocent; v2's filtering even *helps* eq (2.49 < 3.21 — registered as a quirk,
mechanism unknown). Open probe (EMP queue): locate plain's cliff in (2606, 4000) — cheap n=2
cells per point; backup-MacBook-able.

**Champion confirmation, champ side complete both arms (plainc control rows still running):**
- MPS (registered eval): stable 8/10, **faithful guar 8/8**; stable-xy spread [0.10, 0.63] —
  content variance remains the recipe's weak axis, irrelevant to C3-guar.
- CUDA (replication arm): stable **10/10**, **faithful guar 10/10** — and CUDA training is
  measurably more stable than MPS (10/10 vs 8/10), consistent with the MPS-nondeterminism
  stability tax hypothesis. **Cross-device: 18/18 qualifying runs faithful-one-sided.**
  C3-guar's champ side cannot fail from here; full verdicts await the plainc rows.

## [2026-06-11] Wedge v2 verdicts — G-W1 **PASS 7/7** (first wedge gate pass); G-W2 pass-by-sign honestly deflated; the unexpected headline is a 4.2× restricted-data moat

**G-W1 (C3-wedge guarantee): PASS — 7/7 stable eq runs faithful-one-sided on ho_out** (quorum 4
met at 7; canonical semantics; 0 anticonservative cells). With E0.3's 30/30 shapes, the
certificate's guarantee now holds under BOTH geometry-OOD and group-pose-OOD evaluation.
*"Certificate as guarantee" is the most replicated result of the project.*

**G-W2 (C1b-prelim sign gate): PASS as registered — and we deflate it ourselves.** Medians: eq
1.017 [1.000, 1.023] vs plain 1.025 [1.004, 1.052]. The sign is right but Δ = 0.008 with
overlapping CI90s; the pre-registered prediction said "both ≈ 1 ⇒ C1b-prelim refuted in
substance at this κ" — that reading applies. **We do not claim C1b 经由 out/in ratios.** The
out/in degradation ratio is simply not an observable where equivariance pays in this regime
(eq_ctrl, trained on LEAKY angles, also sits at 1.029 — everything generalizes out-wedge).

**Orbit transport, n=7: all within 4.1%** (ratios 0.959–1.026) — Lemma-2 orbit-constancy
through the pixel pipeline, now at 7-run replication. Banked as a §C1 instrument result.

**The unexpected headline (descriptive, no gate was registered for it):** under strictly
in-wedge training data (2606 transitions), median in-distribution δ̂:

| arm | data | median δ̂ | stability |
|---|---|---|---|
| eq_w2 | filtered@2606 | **2.49** | 7/8 |
| eq_ctrl | unfiltered@2606 | 3.21 | 4/4 |
| plain_w2 | filtered@2606 | **10.54** (5/6 ≈ 10.3–10.9; one run 3.53) | 6/6 |
| (v1 ref) plain | unfiltered@4000 | ≈ 3.8 | 6/6 |

— a **4.2× moat that OPENS under data restriction** (v1 unfiltered: ~1.2×), while filtering
*improves* eq (2.49 < 3.21; stability 7/8 vs v1's 3/6). eq is count-insensitive (ctrl ≈ v1).
This is the 举一反三 thesis's exact prediction surfacing unprompted. plain's explosion is still
confounded (count 4000→2606 vs restriction) ⇒ **plain_ctrl addendum registered + launched**
(plain × unfiltered@2606 n=4, eq_ctrl's exact data): ≈3.8 ⇒ restriction kills plain (the
structural mechanism); ≈10.5 ⇒ count (sample-efficiency moat at low N). Either way the measured
moat stands; only the attribution moves.

## [2026-06-11] Wedge lane v1 — verdicts as registered + the design-leak diagnosis; orbit transport ≤2%; v2 registered

**Verdicts (exactly as registered, no reinterpretation):**
- **G-W1 (C3-wedge guarantee): INCONCLUSIVE-BY-STABILITY** — stable eq 3/6 < quorum 4. The 3
  stable eq runs were 3/3 faithful-guar ✓ on ho_out with 0 violating cells (below quorum, so
  reported descriptively, not as a gate pass). eq std spread [1.20, 0.64, 0.26, 1.39, 0.60,
  0.80] — within the winner-recipe@200eps historical range, no new anomaly. plain: 6/6 stable.
- **G-W2 (C1b-prelim sign gate): FAIL as registered** — median out/in δ̂ ratio eq 1.030 [CI90
  1.028, 1.048] vs plain 1.015 [0.994, 1.025].

**Diagnosis (disclosed in order):** the flat result (both arms ≈ 1.0) triggered a data audit →
**the wedge leaks**: only initial angles were constrained; WeakPolicy rotates the block during
episodes — 31.1% of training timesteps are out-of-wedge (median per-episode drift 41.7°, q90
109.7°; 49.5% of final frames remain in-wedge; ho_in itself only 73.3% in-wedge timesteps).
Both arms therefore TRAINED on substantial out-of-wedge state coverage — **the v1 design has no
power to separate the C1b hypotheses**. G-W2's FAIL stands against v1's design; it is not
evidence against C1b. (ho_out is 92.8% out-of-wedge — the eval side is sound.)

**Uncontaminated positives:**
- **Orbit transport (the day's cleanest result):** eq δ̂(transported)/δ̂(in) = 1.021 / 1.011 /
  0.998 — rotating out-of-wedge episodes back by nearest-$C_{16}$ inverse reproduces in-wedge
  audit statistics to ≤ 2%. **Lemma-2 orbit-constancy validated through the real pixel
  pipeline** (residual ≤ 11.25° quantization documented).
- **First eq/plain separation on guarantees:** plain shows its first faithful-guar violation
  (plain_r3 on ho_out, 1 cell anticonservative); eq 0 violations in 3/3. Descriptive (n tiny).

**Wedge v2 (registered now, before any further run):** training transitions filtered to chunk
windows whose block angle stays in-wedge throughout ($\theta_t \in [0°, 90°)$ for all 6 window
frames; ≈ 60–65% of transitions survive). Eval sets unchanged (leak fractions documented).
Arms: eq-w2 n=8 (quorum head-room at ~50% stability), plain-w2 n=6, **eq count-control n=4**
(unfiltered transitions subsampled to the filtered count — separates "wedge restriction" from
"fewer transitions"; the data lever is a known confound). Gates: G-W1/G-W2 unchanged, canonical
`gates.py`. Prediction registered: if C1b is real, plain-w2's out/in ratio rises materially
above 1 while eq-w2 stays ≈ 1 (transport); if both stay ≈ 1, C1b-prelim is refuted at this κ.

## [2026-06-11] E0.3 shape-generalization audits (CPU batch v2) — certificate one-sidedness holds OOD on all 4 unseen shapes, 30/30

**Setup (as declared):** ckpt7 stable Stage-B pairs (eq r0/r1/r3, plain r0/r1/r2; all T-trained),
audited on held-out episodes of 4 never-seen block shapes {L, Z, square, +} + on-T reference
(30 eps each; shape pinned via the autopsy-fixed `variation_values` API, render-diff guarded).

**Verdicts (CORRECTED by review pass + E0.3b, same day):** the first run's "one-sided ✓" used
the pre-amendment ratio semantics (None auto-passed; 11/120 cells were None). E0.3b recomputed
all 30 audits storing dual boundaries (collection is seed-7 CPU-deterministic — exact replay):
- **Faithful one-sided on OOD shapes: 30/30 rows, 0 violations** — 109/120 cells evaluable and
  all ≤ 1; the 11 vacuous cells are all $(H_{\mathrm{cert}}, H_{\mathrm{meas}}) = (0,0)$
  (vacuous-conservative, NOT hidden violations; same phenomenology as the champion cells).
  *The certificate-as-guarantee property survives object-geometry OOD, now under the faithful
  semantics.* (C1b-adjacent free evidence; the wedge lane tests the group-theoretic version.)
- Artifact: `p4_e03b_boundaries.json`. Lesson doubled: the ratio-semantics gap was planted in
  TWO scripts independently — gate mechanics belong in ONE shared function, not copies.
- **δ̂ is nearly flat across shapes (eq):** r0 3.18(T) → 3.41(L) / 3.46(Z) / 3.17(sq) / 4.14(+)
  — worst-case +30% on the most dissimilar shape ('+'); same pattern r1/r3. The certificate's
  load-bearing input barely notices geometry it has never seen.
- **Moat is distributional, not uniform, OOD too:** plain r0/r2 sit at δ̂ 8.2–9.7 across shapes
  (2.4–2.9× eq), but the known good plain seed (r1) reaches 3.8–4.5 ≈ eq levels — consistent
  with the 5-run moat spread [3.79, 1.10, 2.73]; report as distribution, never as a single ×.

**Parallel read, aux-health n=10 (final for the two registered configs):** the night-shift
"content star" at real n — **aux0.3_v0.15: 3/10 stable** (n=3's 3/3 was small-sample flattery —
the exact failure the v1.6 n-upgrade exists to catch), stable-conditional xy ≈ 0.84 but
unbettable odds. **aux0.5_v0.3: 7/10 stable, stable-xy up to 0.84** — stability AND content
simultaneously; **promoted to next champion candidate (needs c2000 stage-B before any claim
gate)**. The stability–content trade-off is now a measured curve, not an anecdote.

## [2026-06-11] 3D lane G0 — ALL GATES PASS on the 3080; contracts banked, plumbing validated end-to-end

**Model gates (V-I..V-VI, run on Mac CPU AND box/cu124, identical):** VN equivariance at
**machine precision** — VNLinear 4e-16, encoder end-to-end (knn regraph included) 1e-16,
predictor $f(\rho(R)z, R{\cdot}a) = \rho(R)f(z,a)$ 4e-16, translation invariance 4e-15 (all
f64; 8 orders of magnitude better than registered tolerances — VN linear equivariance is
associativity, not approximation). Pairing equality **bit-exact** (V-VI; incident #9's gate
installed before any 3D audit, as registered).

**Contract (PegInsertionSide-v1, banked in `p4_3d_g0_contract.json`):** action = 7-d
`pd_ee_delta_pose`, Box(-1,1); **arm frame = `root_translation:root_aligned_body_rotation`** —
translation deltas are root(world)-frame vectors ⇒ the equivariant action lift is sound for
the translation block; rotation in root-aligned body frame fits the gravity-aligned symmetry
tabletop tasks actually have (full SO(3) claims need a frame argument — registered as a
protocol-design input, not assumed). sim 100 Hz / control 20 Hz; demos: 1000 eps, source
`pd_joint_pos`, mean length 149.

**Replay:** official CLI conversion path failed (WSL render boundary, by design caught) →
**state-replay fallback engaged**: 3 eps → 538 clouds (1024 pts, valid-masked) + 535
tcp-derived approx ee-delta actions (**flagged approx — plumbing only, not for claims**;
the real corpus needs the CLI conversion fixed or a controller-faithful local conversion).
Whole G0: **1.2 min** — llvmpipe state-replay rendering is far cheaper than the 0.22 s/step
worst case; 1000-demo corpus prep is hours-not-days even on CPU render.

**CUDA JEPA smoke:** 200 steps @ 87 ms/step on the 3080; pred_loss 3e-3 → 4e-4; tcp-position
probe $R^2 = 0.748$ (content present); latent std 0.023 — **collapse pressure with zero
variance regularization, exactly as the PushT taxonomy predicts** (smoke has no var_coef by
design; the v1.6 lesson set transfers as the 3D protocol's starting point).

**Next blade:** 3D protocol spec v1 (chunk/stride decision off the 20 Hz timescale, variance
recipe, SO(2)-of-SO(3) symmetry scope, controller-faithful action conversion, corpus budget).

## [2026-06-11] C3-guar mechanics amendment (registered between r1 and r2) + champion r0/r1 read + CPU-batch autopsy

**Champion confirmation, first two cells (pre-amendment observations, flagged):** r0 std 0.951 /
xy 0.116; r1 std 1.164 / xy 0.424. The xy spread retires the "uniform content anemia" abort
hypothesis — this is the registered eq-on-MPS run-to-run variance, not a dead recipe; run
continues. Cell cost ~41 min (c2000), full n=10×2 ≈ 14 h.

**The amendment (the real catch):** both runs returned `ratio=None` at the ε=2δ̂ cell — the
measured q90 curve exceeds 2δ̂ already at H=1 (heavy tail: q90/mean > 2), so the measured
boundary is 0 and the ratio is undefined. Under the ratio-based code a run could NEVER qualify
⇒ **C3-guar was mechanically unpassable regardless of certificate quality** — an unfaithful
approximation of the registered text ("certified ≤ measured boundary"). Faithful mechanics now
in force (registered BEFORE r2 data): store both boundaries; guar = direct $H_{\mathrm{cert}}
\le H_{\mathrm{meas}}$ per cell (**0 ≤ 0 passes** — the certificate refuses a horizon the world
also refuses; $H_{\mathrm{cert}} > H_{\mathrm{meas}} = 0$ fails — that IS anticonservative);
cal band uses ratios only where both > 0. This is code-to-registration alignment, not a
loosening: the comparison set is unchanged, only the undefined-cell semantics are defined.
r0/r1 re-evaluated from saved ckpt8 pairs (audit-refresh, no retraining) under new mechanics.
Honest flag: r0/r1 were seen before the amendment; r2..r9 + all plainc are untouched fresh.

**CPU-batch autopsy (separate kill, same review pass):** first launch died at the shapes block —
`variation_space["block"]["shape"].value = idx` raises (read-only property). Sanctioned API
(source-read + 3-point verification): `reset(options={'variation_values': {'block.shape': idx}})`
(dotted path). Banked side-facts: shape catalog o=0 L=1 **T=2 (default)** Z=3 square=4 I=5
small_tee=6 +=7 (the script's mapping was accidentally correct); **default resets always revert
to T=2** — the "historical corpora are T-only" premise is now directly verified, not assumed;
selection is non-sticky (options must ride every reset). Wedge ×3 + c5000 landed before the
crash (no loss); skip-if-exists guards added; relaunched.

## [2026-06-11] 3D lane bootstrap + shared-box incident — zero damage, protocol installed, WSL render boundary measured

**Incident (user caught it):** the 3D-lane rsync targeted `~/se3-ejepa` on the WSL box **without
checking prior tenancy**. Post-hoc forensics: the box IS the paper2 research line's CUDA lane
(step74/75/84 in `.bash_history`; `.venv` py3.11 + torch cu130 built 06-06). Damage audit —
**zero**: rsync had no `--delete` (nothing removed); research-line stashes `stash@{0}` (step73
WIP) / `stash@{1}` (exp17 WIP) intact; the "Step 77" dangling object is a pre-amend duplicate of
`eb1d7aa` (in main); untracked box artifact `step85_phase1_frontier_seed10.json` untouched;
`.venv` imports torch+CUDA fine. Phantom "ahead of origin by 1" was a stale `origin/main` ref
(cleared by fetch). Lesson registered: **probe target dirs before rsync — same class of error as
the pairing incident (assumed instead of checked).**

**Protocol installed** (spec `2026-06-12-p4-3d-lane-bootstrap.md` §shared-lane): env split
`.venv`(theirs)/`.venv3d`(ours); box = pull-only git leaf (commits happen on the Mac); GPU
scheduling via `nvidia-smi` + `~/GPU_LANE.lock` with research-line priority on contention;
3D-lane writes confined to `data/p4_3d/` + `p4_3d_` prefixes.

**Bootstrap verdicts:** smoke 1 PASS (torch 2.6.0+cu124, RTX 3080 visible). Smoke 2 FAIL→PASS:
SAPIEN `RenderSystem` cannot bind `cuda:0` (WSL Vulkan = dzn wrapper + llvmpipe only, no NVIDIA
CUDA-interop extensions — an environment boundary, not a config bug); **llvmpipe CPU-render
fallback works**: reset 0.5 s, pointcloud (1, 32768, 4), ~0.22 s/step, action dim 8
(pd_joint_delta_pos default — H1-original wants `pd_ee_delta_pose`, set at `gym.make`). Smoke 3
PASS (PegInsertionSide demos, 29.5 MB). Smoke 4 PASS (vnn reference clone). Architecture
consequence: **render on CPU/llvmpipe (or Mac/MoltenVK later), train on the 3080** — training
scripts must not import SAPIEN.

**Next blade (G0, mirrors PushT discipline):** G0a obs/action contract + demo replay → G0b
VN-DGCNN forward + SO(3) equivariance unit test (float-exact for VN linear) → VN pairing-equality
test (incident #9's lesson, installed BEFORE any audit) → JEPA training smoke.

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

## [2026-06-12] Night shift (6.2 h, 4 blocks, autonomous) — three old conclusions overturned; the data lever dominates; a stable+contentful champion candidate emerges

(`p4_nightshift_0611.json`. All health-metric tiers; no claim consulted.)

1. **E0.1 data scaling (the night's biggest find):** winner recipe, xy content vs corpus —
   200: 0.098 → 500: 0.242 → 1000: 0.481 → **2000: 0.613** (monotone, 6×). **θ-darkness and
   content-weakness were substantially small-data artifacts**; we sat at 200 episodes for two
   days because v1.1 mirrored lerobot's 206, while collection costs 15 s/200. Stability stays
   2/3 at all sizes — the collapse lottery is recipe-bound, not data-bound.
2. **E0.5.1 aux family:** content ceiling rebroken (aux0.3+v0.15: **xy 0.81**; aux0.5+v0.3:
   0.79); **champion candidate = v0.3 + aux0.3: 3/3 stable AND xy 0.567** — stability and
   content simultaneously, for the first time. **θ-only anchor BACKFIRES** (xy −0.27…−0.41):
   a 2-d regression target drags the representation into a small subspace — clean negative,
   recorded.
3. **E0.2 long-train (overturns the decay story):** under the v0.2 floor, 200 epochs IMPROVES
   content 3× (0.098 → 0.315, std 0.863 stable). **The v1.1/v1.2-era "content decays with
   training" observation was an artifact of the weak-floor recipe**, not a law. aux1.0 × ep200
   does not stack (0.343) — levers are not additive at high coef; data remains the strongest
   single lever.
4. E0.5.2 neighborhood: three 3/3 configs, all weak content (≤ 0.20) — without the anchor, the
   stability–content trade-off stands.

**Champion confirmation launched** (declared this morning): v0.3+aux0.3 × c2000 × n=10, with
C3-cal re-evaluation AND **C3-guar's first legal evaluation** (fresh runs, as the anti-HARKing
registration requires).

## [2026-06-12] v1.6 Stage-B (n=10, first statistically adequate evaluation) — C3-cal FAILS both bases; the failure is PERFECTLY ONE-SIDED; C3-guar registered for fresh-run evaluation

(`p4_v16_stageB.json`, 52 min. Recipes selected on health only — the gate had no hand in recipe
choice; the system worked as designed.)

| | eq (var0.2 winner) | plain (ep40 winner) |
|---|---|---|
| true stability rate | **7/10** (the sweep's 3/3 was partly lottery) | 10/10 |
| C3-cal (two-sided band) | **FAIL 0/7** | **FAIL 2/10** |
| ratio pattern | [None, 0.50, 0.38, 0.88] — nearly identical across runs | similar, conservative |

1. **Verdict stands: C3-cal as registered FAILS at n=10.** The two-sided calibration claim is
   dead under current recipes at fine-to-mid ε.
2. **The failure structure is the finding**: all 68 qualifying ratios ≤ 1.0 — **the certificate
   never over-promised once** across 17 runs × 4 ε. Conservative by ~2–2.7× at mid-ε, exact at
   coarse. The ratio pattern is nearly run-invariant (a stable property of the recipe-data pair,
   not noise). Consistent with every prior signal (rate conservatism, planner ≥ model): **this
   object is a GUARANTEE, not an estimate.**
3. **C3-guar registered NOW (anti-HARKing discipline): the one-sided claim** — certified
   boundary ≤ measured boundary at every ε, every qualifying run — is motivated by this data and
   therefore must be **evaluated on fresh runs only** (next n=10 batch). Its band: ratio ≤ 1 at
   ALL cells, ≥ 90% of qualifying runs.
4. ε=2δ̂ None cells: grid-anchoring artifact (ε anchored on δ̂_mean, boundaries on q90 curves —
   heavy-tailed one-step errors make the finest cell vacuous). A q90-anchored grid is registered
   as the v2 analysis for FUTURE evaluations; the current FAIL is not retroactively touched.
5. eq's 70% stability + Tier-0.5 refinement loop owns the gap; night shift (neighborhood, data
   scale, aux family, long-train) is running as this is written.

## [2026-06-11] 5-RUN VERDICT — INCONCLUSIVE-BY-STABILITY: the eq recipe collapses 3/5 runs; recipe stabilization IS the critical path

(`experiments/p4_5run_extension.py`, 7.8 min, +4 cells; aggregation rule declared pre-run.
Artifact `p4_5run_spine.json`.)

| | eq | plain |
|---|---|---|
| collapse rate (std < 0.7) | **3/5** (runs 2,3,4: std 0.27/0.60/0.46) | 0/5 |
| C3-cal-static | **INCONCLUSIVE-BY-STABILITY** (2 qualifying < 3) | FAIL (1/5 in band) |
| G-pre shape | INCONCLUSIVE | FAIL (0/5; conservative direction throughout) |
| moat (conditioned) | n=2: mean 2.45, range [1.10, 3.79] — unbankable | — |

**The honest read:** the provisional pass did not survive — not because the calibration is wrong
(the qualifying runs look fine; plain's failures are conservative-direction) but because **the
training recipe cannot produce 3 stable eq runs out of 5**. The collapse lottery (60% on the
banked lane) is now unambiguously the project's critical path: no claim of any kind banks until
the recipe is stabilized. The instrument stack did its job — it refused to certify on a rotten
foundation.

**Registered response path (freeze lifts into it, candidates ordered):**

1. **Hyperparameter honesty first**: the "frozen" hyperparameters were never actually tuned —
   they are v1.1 defaults (ema_decay 0.99, var_coef 0.04, lr defaults). A small stability sweep
   (ema_decay × var_coef × lr, eq base, ~12 cells, success = stable-run fraction over 3 runs/
   config) is legitimate pre-claim infrastructure, not gate-shopping — gates are about claims;
   this is about having a training procedure at all.
2. **TC-WM block + proprio anchor (§7.5 amendment)** — upgraded from "C2 candidate" to
   **co-primary stability candidate**: the collapse attractor is predictability-driven, and an
   InfoNCE anchor to proprioception makes content non-collapsible by construction (free in sim,
   identifiability math on file). Run as a separate arm.
3. CPU-deterministic training as the fallback variance-elimination tool (slow but kills the run
   lottery for final claims).

Everything downstream (wedge, C4, 3-seed re-conversion, 3D) queues behind a stable recipe.

## [2026-06-11] 3-SEED CONVERSION — first gate PASS on record (C3-cal-static eq, 2/3) WITH AN ASTERISK; seed variance becomes the story

(`experiments/p4_3seed_spine.py`, 11.3 min, 6 cells, protocol freeze honored — only registered
gates evaluated. Artifact `p4_3seed_spine.json`; ckpt5 pairs saved.)

| cell | std | δ̂ | δ̂_norm | C3-cal ratios (ε=2/4/8/16) |
|---|---|---|---|---|
| eq s0 | 1.224 ✓ | 3.30 | 0.238 | [—, 0.50, 0.375✗, 0.875] |
| eq s1 | 0.917 ✓ | 2.80 | 0.270 | [1.0, 0.67, 0.50, 1.0] ✓ |
| eq s2 | **0.274 ✗ collapsed** | 1.05 | 0.337 | [1.0, 0.67, 0.50, 1.0] "✓" |
| plain s0 | 0.750 ✓ | 7.68 | 0.904 | [0.2✗, 0.25✗, 0.50, 1.0] |
| plain s1 | 1.106 ✓ | 3.73 | 0.298 | all ✓ |
| plain s2 | 0.898 ✓ | 9.35 | 0.920 | [0.25✗, 0.25✗, 0.625, 1.0] |

**Gate verdicts (as registered):**

1. **C3-cal-static (eq): PASS 2/3 — provisional, with the asterisk stated**: one of the two
   passing seeds (s2) is variance-collapsed (std 0.274); conditioned on the stability floor the
   tally is 1/2 stable seeds = inconclusive at n=3. **Not banked dirty**: status =
   PROVISIONAL-PASS pending a 5-seed extension under the stability-conditioned reading
   (registered below). plain: FAIL 1/3.
2. **G-pre shape: FAIL 0/3 both bases — in the CONSERVATIVE direction**: linear shape confirmed
   everywhere (r²_lin > r²_exp, eq), but the certified per-chunk rate (δ̂) runs 2.3–2.7× the
   measured growth rate (errors accumulate sub-linearly in δ̂ — correlation/cancellation). As
   registered = FAIL; consumer-aligned note: the BOUNDARY gate (C3-cal, the quantity the spacing
   rule consumes) is the one that passes — the rate sub-gate is a shape diagnostic. Certificate
   as guarantee ✓ (never over-promises at κ=0); as point-estimate, ~2.4× conservative.
3. **Stability: eq 2/3, plain 3/3** — **the collapse lottery reaches the banked lane**: 1/3 eq
   seeds collapsed even in the single-frame κ=0 config. The recipe instability is seed-level,
   not architecture-level.
4. **Moat (normalized): per-seed [3.79, 1.10, 2.73], mean 2.54, range wide** — and the wideness
   is two-sided: plain's δ̂ itself swings 2.5× across seeds (7.68/3.73/9.35). The 6.3× headline
   was partly plain's bad seed-0. Stability-conditioned (eq s0,s1 × plain s0,s1): ~2.4× at n=2.
   Direction real; magnitude unbankable at n=3.

**Registered next step (before any further seeds are run):** all gates henceforth condition on
the stability floor (std ≥ 0.7) — the consistent extension of the v1.4 registration; collapsed
seeds are non-qualifying for every gate and are REPORTED (not hidden, not replaced). A 5-seed
extension (+s3, +s4) then resolves C3-cal-static and prices the collapse rate properly.

*Same-day review discovery (changes the statistical language):* **eq training is
run-nondeterministic on MPS** — three nominally identical seed-0 runs give δ̂ = {1.47, 3.30,
2.75} (2.2× spread; third-run verification deliberate), while plain reproduces bit-close
(7.677/7.68). Attribution: e2cnn kernels on MPS (plain's stack is deterministic there).
Consequences, registered: (i) for eq, "seeds" are labels — **n counts RUNS**, sampled from a run
distribution; all eq statistics are stated per-run; (ii) iteration continues on MPS with
run-count language (the extra variance is real and honestly reported); CPU-deterministic training
is reserved as a camera-ready reproducibility option; (iii) **upstream notice to paper2/S1**
(same steerable-on-MPS combination) via the snapshot. The collapse lottery and the run lottery
are now one phenomenon class: the eq recipe's outcome distribution is wide on MPS.

## [2026-06-11] v1.5 stability sweep — H-v1.5a REFUTED (the isotypic floor ACCELERATES collapse); the attractor is predictability-driven; strategic re-scope

(`experiments/p4_v15_stability.py`, 18 min; artifact `p4_v15_stability.json`; per-field floor
roll-invariance asserted 6e-8 before use.)

**Sweep on the worst cell (eq/6-ch@κ0):**

| recipe | floor-stat | pred_loss | xy | gate |
|---|---|---|---|---|
| per_dim (baseline) | 0.307 | 0.0176 | +0.41 | fail |
| per_dim+gated (Step-64) | 0.327 | 0.0130 | +0.34 | fail |
| **per_field (H-v1.5a)** | **0.083** | **0.0015** | **+0.48** | fail |
| per_field+gated | 0.136 | 0.0021 | +0.38 | fail |

1. **H-v1.5a REFUTED, instructively:** the isotypic-aware floor didn't rescue stability — it
   collapsed 4× HARDER (0.083) while pred_loss dropped 12× and xy content went UP (+0.48, the
   best of the board). Reading: the collapse attractor is **predictability-driven** — any
   relaxation of the isotropy constraint is *exploited* for deeper collapse into an
   ultra-predictable shrunken manifold (which still carries content!). The variance term's
   isotropy wasn't the disease; it was a partial brake. The "stable-but-empty" failure mode
   (aug-collapse) and this "collapsed-but-contentful" mode are distinct — collapse ≠ content
   loss here, but a std-0.08 space is numerically unusable downstream (normalized δ̂ explodes).
2. H-v1.5b marginal (+0.02 std). No recipe passes the 0.7 floor on eq/6-ch; the moat re-ask ran
   under the least-bad recipe and **remains unbankable** (3.56× @κ0 / 0.57× @κ0.8, eq floors
   FAIL throughout). Side-signal: plain/6-ch@κ0 has xy = −0.001 — ZERO position content where eq
   holds +0.34 even collapsed; the content asymmetry persists through everything.
3. **Strategic re-scope (the honest read of three experiments):** 6-ch frame-pairs destabilize
   eq AND never delivered velocity (v1.4); no variance-term variant fixes it (v1.5). Meanwhile
   the **single-frame κ=0 configuration was stable all along** (std 0.985/0.823, the clean
   Stage-1 numbers). Decision: **bank the stable lane** — 3-seed the single-frame κ=0 spine
   (where C3-cal shapes, the 6.3× normalized moat, and certificate-as-guarantee live);
   κ=0.8 + frame-pair stability is documented as the **open frontier** (the paper's honest
   Limitations §: "the momentum regime requires temporal observability that this recipe does
   not stably provide; the certificate's G-pre diagnostic correctly refuses jurisdiction
   there"). The signature figure's surviving form: static-regime calibration + the κ-gate's
   environment-side two-regime measurement + G-pre's jurisdiction map.

## [2026-06-11] v1.4 frame-pair — INCONCLUSIVE-BY-RECIPE-INSTABILITY on the moat; velocity-observability hypothesis itself takes damage; first in-jurisdiction certificate cell appears

(`experiments/p4_v14_framepair.py`, 7.9 min, 4 cells; artifact `p4_v14_framepair.json`.)

| cell | pred | latent_std | δ̂_norm | λ̂_t /chunk | G-pre r |
|---|---|---|---|---|---|
| eq@κ0 | 0.018 | **0.307 ⚠** | 0.464 | **+0.404** | **0.64 — IN BAND** |
| plain@κ0 | 0.513 | 0.945 | 0.936 | +0.0002 | (degenerate) |
| eq@κ0.8 | 0.219 | 0.664 ⚠ | 0.667 | +0.015 | 6.6 |
| plain@κ0.8 | 0.271 | 1.041 | 0.435 | +0.0006 | (degenerate) |

Normalized moat: κ=0 → 2.02× (eq better, **down from single-frame 6.3×**); κ=0.8 → 0.65×
(inversion persists). **Verdicts, honestly:**

1. **The moat question is BLOCKED, not answered**: the v1.4 spec's own stability check fires —
   eq's recipe partially collapsed with 6-ch inputs (std 0.307 at κ=0!, 0.664 at κ=0.8) while
   plain stayed healthy. Neither the 2.02× nor the 0.65× is bankable; the registered next lever
   is a **v1.5 stability pass for the eq/6-ch combination** — `predictability_gated_var` (the
   repo's own Step-64 tool, built for exactly this variance↔predictability tension) is the
   first candidate, var_coef the second; gate: std ≥ 0.7 before the moat is re-asked.
2. **The velocity-observability hypothesis itself took damage**: raw δ̂ did NOT drop at κ=0.8
   (eq 4.15→5.01, plain 4.45→5.12 — both *worse* than single-frame). Frame pairs ≠ velocity
   extraction: at 96 px/0.1 s the block displacement is ~1–5 px and nothing in either encoder
   biases toward temporal differencing in 20 epochs. The non-Markov diagnosis stands; the cheap
   fix didn't bite. (Alternatives if pursued: explicit difference channel, longer training, or
   accepting the regime-scoped moat statement.)
3. **First in-jurisdiction certificate cell**: eq@κ0 frame-pair posts λ̂_t = +0.404/chunk with
   G-pre r = 0.64 — the first tangent certificate inside its band. **Flagged, not celebrated**:
   it coincides with the collapsed-variance space (std 0.307), where amplified low-variance
   directions can manufacture expansive-looking, self-consistent dynamics. Re-examine only
   after v1.5 stability.
4. Analysis-guard note: G-pre r for near-zero tangents (plain: λ̂≈2e-4 ⇒ r≈208/152) is junk —
   the |λ|>1e-6 guard is too low; r is meaningful only when λ̂_t's CI excludes 0 (fix queued
   with the next analysis edit).

**Net:** zero gates loosened; the moat narrative is now explicitly hostage to recipe stability —
which is itself the C2 lane's content-decay/collapse axis. The two lanes have CONVERGED on one
underlying phenomenon: **this JEPA recipe's variance–predictability tension is the project's
current binding constraint.** v1.5 (stability) is therefore the next blade for BOTH lanes.

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

