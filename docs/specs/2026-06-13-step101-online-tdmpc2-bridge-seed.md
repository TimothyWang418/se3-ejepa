# step101 — the bridge, full online TD-MPC2 with an S₂-equivariant latent (GPU, RunPod)

Frozen 2026-06-13, before any online run. Supersedes step100's offline attempts (all INCONCLUSIVE-BY-
LEVEL-DOMINANCE). This is the GPU-justified path the user committed to after `我想要桥`.

## Why offline failed (the reason this needs online)

step100b (reconstruction) under-fit; step100c (latent-consistency + reward, up to LAT=96/15k ep) drove
λ₁ positive but the one-step latent error floored at ~0.2 (consistency loss plateaus ~0.017 regardless of
capacity) → measured median 1–2, ratio ~0.04, **bias-dominated, no calibration, no eq advantage** (dense
even slightly beat eq on one-step accuracy — equivariance is a capacity constraint, not a one-step-accuracy
lever). The floor is DATA-limited: 300k offline on-policy transitions cannot make the latent accurate
enough that Lyapunov growth dominates before native bias. The official TD-MPC2 walker latent reaches
λ₁=0.25–0.30 with a 5–6 step calibrated horizon (E13: ratio 0.94) ONLY via the full online recipe
(consistency + reward + **value/Q** + policy, millions of online steps, MPPI data collection). So the bridge
requires reproducing that recipe — with an equivariant latent.

## The claim (what the bridge actually tests — NOT "eq calibrates, dense doesn't")

E13 already showed *dense* TD-MPC2 calibrates on walker (0.94). The bridge mirrors the **E12 mechanism** on
a zoo task: across K seeds, the **equivariant** WM's certificate calibrates **a-priori and reliably**
(low cross-seed variance of the measured/certified ratio) while the **dense** twin's calibration is a
**seed lottery** (high variance — you need cross-validation to know which seed you drew). Pre-registered
metric: Var_seed(|log ratio@ε=0.2|) for eq strictly below dense, AND eq median ratio ∈ [2/3,3/2].

## Design

- **Base**: official `nicklashansen/tdmpc2` trainer (cloned at `external/tdmpc2`), single-task
  `walker-walk`, default cfg (latent_dim 512, simnorm_dim 8 → 64 groups, mlp_dim 512, num_q 5, horizon 3).
  Train to ~1M steps (DMC-walker converges well before the 10M default).
- **Dense arm** = stock WorldModel (the E13 architecture, retrained K seeds — this is the control).
- **Eq arm** = `EqWorldModel`: swap `_encoder` and `_dynamics` for S₂-weight-tied versions; make
  `_reward`/`_Qs` S₂-**invariant** and `_pi` S₂-**equivariant**. S₂ latent structure on the 64 SimNorm
  groups: **32 invariant groups (256-d) + 16 leg-pairs (R-groups 128-d ↔ L-groups 128-d under swap)**;
  ρ_z swaps the R/L group blocks, identity on the invariant block. Encoder: tied leg subnet produces
  R-groups from (R-obs, torso) and L-groups from (L-obs, torso); invariant subnet from (torso, pooled
  legs); SimNorm per group (commutes with the permutation since it is group-wise). Dynamics: tied per-leg
  map + symmetric-pooled invariant map (step100b structure, but with NormedLinear+SimNorm). Action S₂
  layout from step100: act swap [0:3]↔[3:6]; obs swap blocks per step100 header.
- **G-EQ** (instrumental, must pass before reading science): eq equivariance defect ≤ 1e-5 on encoder &
  dynamics (weight-tying → machine precision); checked by the step100 ρ-action test.
- **Audit**: identical to step89/step100 — policy-prior loop g(z)=dynamics(z, π(z)); Benettin-QR λ₁ + CI;
  measured crossings from 100 mid-episode true rollouts; ratio@ε=0.2.
- **K = 5 seeds** each arm to start (10 runs); extend to 10 if the variance signal is borderline.

## Gates (pre-registered, no loosening)

- **G-BR-CAL**: eq median ratio@0.2 ∈ [2/3,3/2] over K seeds (eq reaches the calibrated regime).
- **G-BR-VAR**: Var_seed(|log ratio|)_eq < Var_seed(|log ratio|)_dense (the reliability claim) — report
  with a bootstrap CI on the variance ratio; PASS only if the CI excludes 1.
- **G-BR-TAME**: if dense ALSO calibrates with equally low variance → INCONCLUSIVE-NO-SEPARATION (honest;
  the bridge would then be a null — structure doesn't buy reliability here, recorded as such).
- **G-BR-PERF**: both arms must reach comparable task return (≥ a floor fraction of stock TD-MPC2 walker),
  else a mis-trained arm confounds the certificate comparison — disclosed per seed.

## Honest payoff caveat (stated before spending)

Offline evidence shows equivariance does NOT improve one-step accuracy; the bridge bets on the subtler
cross-seed-reliability claim holding online. It may not — eq and dense may both calibrate reliably (null),
or eq may underperform on task return at this capacity. This is a genuine experiment with a real chance of
an honest negative. Budget accordingly: ~1 day fork + ~$20–40 RunPod + K-seed orchestration. It is a
post-submission E17 enhancement to an already-arXiv'd paper, not acceptance-critical.

## Compute plan (RunPod, per the B300 post-mortem: rent for PARALLELISM, not FLOPs)

Single walker run ≈ a few GPU-h (env-stepping-bound, not FLOP-bound — a 4090 ≈ a 3080 per run; the win is
fanning K×2 runs across N instances). Rent ~5 cheap GPUs (4090/3090), one (arm,seed) per instance, two
waves → ~one-run wall-clock for all 10. Transfer via `runpodctl send/receive` (NOT the SSH proxy — 0.4MB/s
trap); work on `/root` local disk (NOT the MooseFS network volume — breaks rsync temp-renames). Deps:
`scripts/runpod_bridge_setup.sh` (torch+tensordict+mujoco+dm_control+hydra; MUJOCO_GL=egl headless).
DEVELOP + SMOKE LOCALLY FIRST (Mac/3080, ~5k steps) — rented GPUs run only validated full-length jobs.
