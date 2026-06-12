# step100 — S₂(leg-exchange)-equivariant world model ON the zoo's own task (walker-walk)

Date frozen: 2026-06-12, before any training run. Seeds n=10 from the start (lesson: budget n at design time).

## The gap this closes

The paper's two halves currently live on disjoint substrates: *structure ⇒ a-priori trustworthy certificate*
(Lorenz-96, pendulum ring — constructed systems), and *generic zoo ⇒ certificate must be cross-validated*
(84-cell TD-MPC2 map, LeWM, V-JEPA 2-AC). Missing bridge: **structure evaluated on the zoo's own flagship
task**. Planar walker dynamics carry a genuine finite symmetry — the two legs are mechanically identical, so
leg exchange is an S₂ ≅ Z₂ symmetry of the dynamics (to be verified on the MuJoCo model before training;
if the XML is exactly symmetric the symmetry is exact, else E3's approximate-symmetry law governs and the
measured asymmetry is disclosed as ε_world).

## Design (mirrors step89's audit pipeline exactly)

- **Data**: transitions from the true environment under the official TD-MPC2 walker-walk-1 policy
  (state-action-next-state tuples; the collector is fixed and identical for both arms).
- **Group action**: ρ = permutation of left/right leg coordinate blocks in obs (orientations/velocities of the
  6 leg joints) and action (3+3), identity on torso coordinates. Pure permutation representation — weight
  tying implements equivariance exactly; no e3nn (and no MPS-nondeterminism exposure; train on CPU).
- **Arms** (identical recipe, param-matched, same data, K-step rollout loss; n=10 seeds each):
  1. **eq**: S₂-equivariant encoder/dynamics/decoder (shared per-leg subnets + symmetric torso block).
  2. **dense**: unconstrained MLP twin.
  3. **dense+aug** (the E3-style augmentation control): dense twin trained with mirrored-copy augmentation.
- **Loop closure**: distill a small policy head π̂(z) from the collector's actions (same distillation for all
  arms); audit the autonomous loop g(z) = d(z, π̂(z)) with the unchanged step89 certify()/measure()
  machinery (Benettin + block bootstrap; measured horizons from 100 mid-episode starts on true rollouts).

## Pre-registered gates (no loosening; INCONCLUSIVE over reinterpretation)

- **G-EQ** (instrumental): eq arm's equivariance defect ≤ 1e-5 (machine precision for permutation tying);
  dense arm's defect ≥ 1e-2. Fails ⇒ implementation bug, fix before any science is read.
- **G-SYM** (world): measured leg-exchange defect of the TRUE env transition map on collected states
  (‖T(ρs,ρa) − ρT(s,a)‖/‖T‖). Reported as ε_world; if > 0.05 the experiment is governed by the
  approximate-symmetry clause and says so.
- **G-CAL** (the claim): eq arm's certificate ratio@ε=0.2 ∈ [2/3, 3/2] on ≥ 7/10 seeds **with zero
  calibration data**, while the dense arm's |log ratio| is strictly worse on a majority of seeds.
- **G-TAME** (honest escape): if BOTH arms calibrate (the walker prior loop may be too tame to separate
  architectures at this latent size), the verdict is INCONCLUSIVE-BY-TAMENESS, reported as such — the
  experiment then still yields the first equivariant-WM certificate on a dmcontrol task (descriptive value).
- **G-AUG**: dense+aug arm: defect intermediate; certificate quality between eq and dense (E3's prediction)
  — observational, no gate.

## Estimate & infrastructure

Small models (latent ≤ 128), CPU; data collection ~30 min (box, ckpt + env present); training 10 seeds × 3
arms ≈ hours on the M-chip 4-way worktree pattern. Artifact: papers/figures/step100_walker_s2.json
(incremental per seed). Equivariance unit test ships with the layer (tests/test_step100.py).

## Honest risks (stated up front)

(i) Policy distillation may inject asymmetry (mitigation: distill from symmetrized action targets and
report the distillation defect); (ii) the walker prior loop's λ₁ at this scale may differ from TD-MPC2's
0.25–0.30 — the certificate row regime is whatever it is, reported as-is; (iii) scope: this tests S₂ on one
task — it does not claim SE(3)-scale generality; it claims the *bridge* (structure's a-priori trust on a
zoo task, same audit machinery).

## Paper placement

E17 + appendix write-up; one-sentence pointer in main text only if the 9-page budget allows post-September
polish; otherwise appendix-only with rebuttal entry. No main-text restructure before the ICLR push.

## Addendum (2026-06-12, before any arm-comparison was read)

First 400-epoch pass: ALL arms land measured-median = 1 at every ε (one-step error above the coarsest
threshold; training loss still descending at cutoff). That is a training-adequacy floor, not a
structure-vs-dense comparison — no arm differences were read. Recipe iteration, applied SYMMETRICALLY:
epochs 400 → 5000, everything else unchanged. The 400-epoch artifact is archived as
`step100_walker_s2_results_e400.json` (diagnostic). Gates unchanged.
