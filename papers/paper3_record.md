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

## [2026-06-10] P4-step2 — κ validation gate: FALLBACK-2PT, honestly — the dial saturates, but TWO regimes are real and resolved

Run concurrently with step1's grid (true-env measurement, no learned model). **Registered
discovery first**: `_set_state` (env.py:514–527) never sets block velocity/angular velocity ⇒ the
7-d obs state **under-specifies κ>0 dynamics**; step2 manages the 10-d extended state via direct
pymunk handles; **registered debt**: any κ>0 oracle-CEM must adopt the extended-state setter
(step1's κ=0 oracle is safe — damping 0 kills velocities).

**Protocol** (spec, pre-registered): κ grid $\{0,0.5,0.8,0.95,1.0\}$; per κ 10 WeakPolicy
episodes, 40 (extended-state, 60-action-window) pairs (n=30 survive the fit-window filter —
reported); twin trajectories, block-angle $\delta_0=10^{-4}$; config-space metric (pos/512,
angle/$\pi$); G0 determinism self-check ($\delta_0{=}0$ twins coincide exactly) PASS at all κ.

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

## [2026-06-10] P4-step1 part 2 — pipeline + smoke + overnight launch

**Pipeline built, smoke end-to-end (13.9 s, bit-identical across runs), full
seed-0 grid LAUNCHED overnight.** `experiments/p4_step1_pipeline.py` (G0b collection → G0c
oracle-CEM gate → param ladder → grid → probes-on-every-cell → JSON). Three smoke nails, fixed
honestly and recorded in-code: (i) vendored `WeakPolicy` is VecEnv-shaped — verbatim single-env
replica of its sampling (cited to file:lines); (ii) vendored `ConvEncoder` requires
`width % 8 == 0` (GroupNorm); (iii) G0c assert demoted to a **recorded verdict** — an
infrastructure-gate FAIL must not burn the overnight grid it does not feed. Resilience: per-base
MPS device-probe with honest CPU fallback (e2cnn-on-MPS unverified). Artifact lands at
`papers/figures/p4_step1.json`; read-out + G1/G3 verdicts next session.
Spec: `docs/specs/2026-06-10-p4-step1-pipeline-bases-seed.md`.
