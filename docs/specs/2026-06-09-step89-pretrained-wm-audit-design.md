# Step 89 — Certify a PUBLIC PRETRAINED world model (TD-MPC2) — design spec

*Date: 2026-06-09 · Status: design, recon-verified (GO) · Owner: SE3-EJEPA / Certified World Models · Compute: CPU float64 (Mac or box); no training*

> The #1 gap to an exceptional ICLR paper (reviewer synthesis): the certificate has never touched a model the
> community recognizes. Fix: the certificate is a **read-out** — apply it, training-free, to official **TD-MPC2**
> checkpoints (Hansen et al., MIT; deterministic continuous latent dynamics $z_{t+1}=d(z,a)$, exactly our $\hat\phi$).
> Recon (2026-06-09, sub-agent, URLs verified): checkpoints at `huggingface.co/nicklashansen/tdmpc2/resolve/main/
> dmcontrol/{task}-{seed}.pt` (~31 MB, 5M params, 30 DMC tasks × 3 seeds, incl. **acrobot-swingup** — direct
> continuity with step84 — and walker-walk); `mujoco==3.1.2` + `dm-control==1.0.16` wheels clean on macOS-arm64 and
> Linux; **the exact audit does not exist** (closest: arXiv:2410.10674 — Lyapunov of the TRUE env under RL policies,
> not of the learned latent map; Özalp & Magri 2025 — latent spectra for scientific-ML, not RL WMs). Must-cite both.

## 1. Deliverable

**"The certificate reads a real, community-recognized world model."** For official TD-MPC2 single-task checkpoints
(acrobot-swingup = the step84 bridge; walker-walk = the canonical headline; 3 seeds each):

1. **Certified side (model-only, zero env access):** rebuild the `_encoder/_dynamics/_pi` slices as plain MLPs
   (NormedLinear = Linear+LayerNorm+Mish; SimNorm output), `.double()`, wrap the autonomous policy-prior loop
   $g(z)=d\big(z,\tanh(\mu_\pi(z))\big)$, and run the UNCHANGED step78 machinery (`qr_logR_series` →
   `bootstrap_spectrum_ci` → `horizon_interval`) → spectrum, $\lambda_1$ CI, certified $T_1(\epsilon)$ in model steps.
2. **Measured side (the cross-validation):** roll the TRUE dm_control env (their wrapper semantics: flattened obs,
   `action_repeat=2`) under the same policy prior; open-loop the latent from $z_0=\mathrm{enc}(o_0)$; measured horizon
   = first model-step where $\lVert\hat z_t-\mathrm{enc}(o_t)\rVert/\lVert\mathrm{enc}(o_t)\rVert>\epsilon$, over
   $\ge20$ starts.
3. **Calibration:** ratio measured/certified per task/seed — the step85-Phase-0 read, now on a model people use.

## 2. Pre-registered risks & their honest readings (from recon — decided BEFORE running)

- **(a) SimNorm structure:** the 512-d latent lives on $\prod_{64}\Delta^7$; each softmax group has a zero direction
  ⇒ ~64 structural strongly-negative exponents. Report the full spectrum; $\lambda_1$ (the max) is unaffected;
  flag the structural band in the figure, do not hide it.
- **(b) Scope:** the certificate covers the **policy-prior closed loop** $g$; deployed TD-MPC2 acts via MPPI
  (non-smooth in $z$). State this; optionally add the step84-style time-varying product $\prod_t D_z d(z_t,a_t)$
  along recorded planner trajectories as a corroborator. NOT a blocker.
- **(c) $\lambda_1\le0$ is a finding, not a failure:** on a stable gait the latent loop may be contracting ⇒ the
  certificate reports a long/unbounded horizon — then the measured side either confirms (long horizon, certificate
  calibrated in the stable regime) or refutes (early divergence ⇒ the pretrained model **cannot certify its own
  competence** — the paper-thesis reading, now on a SOTA model). Both outcomes are publishable; pre-registering this
  prevents post-hoc spin.

## 3. Gates (never loosen — INCONCLUSIVE instead)

- **G0 (pipeline validity):** rebuilt slices reproduce the checkpoint's forward (encoder/dynamics agree with a
  reference forward pass on random inputs to float tolerance — guarded by unit tests); spectrum finite; structural
  SimNorm band identifiable.
- **G1 (the audit lands):** per task, the certified-vs-measured relation is **coherent** under one of the
  pre-registered readings: (i) calibrated (ratio $\in[1/2,2]$) — "the certificate reads a real model a-priori"; or
  (ii) $\lambda_1\le0$ + long measured horizon — "certified stability, cross-validated"; or (iii) miscalibrated —
  reported as "a SOTA dense model cannot certify its own competence" with the measured side as ground truth.
  INCONCLUSIVE only if the measured side itself is too noisy to read (e.g. encoder mismatch artifacts).

## 4. Components

- `experiments/step89_pretrained_wm_audit.py`: `SimNorm`, `NormedLinear`-faithful MLP rebuilds; `load_tdmpc2_slices
  (ckpt_path)` (key-subset state_dict load, defensive key-print on mismatch); `g(z)`; `certify(...)` (step78 reuse);
  `measure(...)` (dm_control rollout + latent open-loop); `run()` per task/seed → JSON + figure.
- `tests/test_step89.py` (no checkpoint needed): SimNorm math (per-group softmax, simplex output), NormedLinear
  forward (LayerNorm+Mish ordering), loader key-subset logic on a synthetic state_dict, $g$ differentiability +
  Jacobian shape $512\times512$.
- Checkpoints + wheels are NOT committed (`models/tdmpc2/` gitignored; URLs pinned here and in the script header).

## 5. Where it lands

- ICLR: a short **"Certifying a public pretrained world model"** experiment (E13) + one figure (spectrum with
  structural band | certified vs measured horizon per task) — converts "toy program" into "audit of models people
  use"; cites arXiv:2410.10674 and Özalp & Magri 2025 as the novelty delimiters. paper2: full section.
- Risks to the schedule: dm_control/mujoco install friction (recon says wheels are clean); checkpoint key-name drift
  (defensive loader); compute trivial (no training; 512-d Jacobian QR on CPU float64).
