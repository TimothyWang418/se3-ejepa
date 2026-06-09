# Step 87 — certificate-gated MBRL sample efficiency (direction ②) — design spec

*Date: 2026-06-09 · Status: design, build-ready (pre-implementation) · Owner: SE3-EJEPA / Certified World Models · Compute: RTX 3080 (CUDA) required for the full run*

> Direction ② of `docs/specs/2026-06-08-three-directions-to-8-seed.md` — the seed's **heaviest, riskiest (~0.4)**, "do
> last." This spec pins a *correct, minimal* online MBRL loop and the cert-gating contrast so it is build-ready; the
> full sample-efficiency result is the gamble and is reported honestly (positive **or** INCONCLUSIVE — the seed flags it
> is close to §5.19's $\sim2\times$ plan-depth failure mode).

## 1. Thesis & honest risk

**Thesis.** Use the certified per-orbit horizon $T_1(\epsilon)$ to **gate the imagination horizon** in Dreamer-style
policy learning: backprop policy gradients through imagined world-model rollouts only within $T_1$; beyond it the model
is untrustworthy, so its gradients are *noise*. Cert-gated imagination should be more **sample-efficient** (return per
real env-step) than fixed-horizon or ungated imagination.

**Why it can win where §5.19 (return-MPC) did not.** §5.19 is single-shot MPC depth (the cert was $\sim2\times$ the
return-optimal *depth*). ② is different: the metric is **sample efficiency**, not plan depth, and the cert-gate's job is
to **cut noisy gradients** from the model's untrustworthy tail, not to set an optimal depth. A stale rollout's tail
gradients point in wrong directions; removing them should speed policy-gradient convergence.

**Honest risk (load-bearing).** It may still inherit the §5.19 failure mode (the conservative $T_1$ cuts *useful*
gradient too, and a slightly-longer fixed horizon wins). Confidence ~0.4. **INCONCLUSIVE if no sample-efficiency gain.**

## 2. Environment & reuse

**Env: controlled Lorenz-96 (reuse `step79`).** Continuous control $u\in[-u_{\max},u_{\max}]^N$ (direct policy
gradient, unlike Acrobot's discrete actions); exact $\mathbb{Z}_N$-equivariant action-conditioned WM
(`step79.L96ControlledConv` / `make_equivariant_wm`) with a proven certificate; goal = **chaos suppression** toward the
uniform fixed point $x_i{=}F$, reward $r_t=-\lVert x_t-F\mathbf1\rVert^2/N$.

Reuse, do **not** modify: `step79` (`rk4_controlled`, `L96ControlledConv`, `collect_data_mixed`, `train_wm`,
`certificate`, `certified_T1_steps`); `step78` (bootstrap CI). New: actor, critic, imagination loop, cert-gated loss,
the online MBRL outer loop. (Grep confirmed: **no existing actor/critic/Dreamer code** — build from scratch.)

## 3. Components (minimal, 1-GPU)

- `Actor(N) -> Normal(μ,σ)`: MLP (2 hidden, 128) → $\mu=\tanh(\cdot)$ (action in $[-1,1]$, scaled by $u_{\max}$),
  learnable per-dim $\log\sigma$. `rsample()` for reparameterized gradients.
- `Critic(N) -> value`: MLP (2 hidden, 128) → scalar value baseline (variance reduction).
- `imagine(model, actor, z0, H, mu, sd)`: roll the WM under the policy for $H$ steps; return latent traj, rewards,
  log-probs, entropies. (Equivariant WM ⇒ reparameterized gradients preserve $\mathbb{Z}_N$-equivariance.)
- `actor_objective(model, actor, critic, z0, mu, sd, H_g)`: **PATHWISE (Dreamer-style)** — the cert-gating lever is the
  backprop depth $H_g$. Roll the *differentiable* WM under the reparameterized policy (`rsample`) for $H_g$ steps,
  accumulate the discounted reward through the rollout, and **bootstrap the tail with the critic**:
  $J=\sum_{t<H_g}\gamma^t r_t + \gamma^{H_g} V(z_{H_g})$ — the gradient flows *through the model dynamics* for $H_g$
  steps, then the learned value carries the rest (no model gradient past $H_g$). Actor maximizes $J$; critic is a
  $\lambda$-return / TD baseline. **This is the cert-gate: pathwise gradients through a chaotic rollout amplify as
  $e^{\lambda_1 H_g}$ (the very $\lambda_1$ the certificate reads), so past $T_1$ they EXPLODE and are noise; gating at
  $H_g=T_1$ keeps them well-conditioned.** (REINFORCE/score-function gradients, where the advantage is detached, do NOT
  exhibit this — the model-rollout gradient is the whole point, so the loss must be *pathwise*.)
  - **cert-gated:** $H_g=T_1^{\rm steps}$ (per-iteration, from the current WM's certificate).
  - **fixed-H:** $H_g\in\{T_1/2, T_1, 2T_1\}$ (the ablation ladder).
  - **ungated:** $H_g$ large (the gradient should be visibly noisier / exploding).

## 4. Online MBRL loop (the sample-efficiency setting)

Per iteration $k$ (track cumulative **real env-steps**):
1. **Collect** a small batch of real rollouts with the current actor on `rk4_controlled` (env-steps += batch·len); add to a replay buffer.
2. **Train WM** a few epochs on the buffer (`step79.train_wm` pattern), warm-started.
3. **Certificate** → $T_1^{\rm steps}$ (`certificate` + `certified_T1_steps`) from the current WM.
4. **Improve policy** with $M$ imagined-rollout gradient steps under the chosen `policy_loss` gate.
5. **Evaluate** actor return on the true env (held-out starts). Record (env-steps, return).

Plot **return vs env-steps** (sample-efficiency curve) + **final return**, for cert-gated vs fixed-H ladder vs ungated.
Everything else identical across arms (same WM updates, buffer, actor/critic init, seeds) ⇒ a clean within-method contrast.

## 5. Honest gates (never loosen — INCONCLUSIVE instead)

- **G2 (sample-efficiency win):** cert-gated reaches a target return at $\le$ the env-steps of the best fixed-$H$ arm on
  $\ge2/3$ seeds, AND $\ge$ ungated. **INCONCLUSIVE if no gain** (it then mirrors §5.19's plan-depth conservatism — flag,
  do not loosen).
- Equivariance guard (`tests/test_step87.py`): the equivariant actor/WM imagination is $\mathbb{Z}_N$-equivariant (a
  cyclic shift of $z_0$ shifts the imagined action sequence), to float round-off.

## 6. Build staging (so INCONCLUSIVE is cheap)

- **Stage A (CPU smoke, make-or-break wiring):** components + `imagine` + the pathwise `actor_objective`; **mechanism
  test** — the pathwise actor-gradient norm $\lVert\nabla_\theta J\rVert$ grows $\sim e^{\lambda_1 H_g}$ with the backprop
  depth and **explodes past $T_1$**, while gating at $H_g=T_1$ keeps it bounded (this is the cert-gate's reason for
  being); plus an equivariance check (imagination is $\mathbb{Z}_N$-equivariant) and an end-to-end tiny-config run. Cheap.
- **Stage B (3080 full):** the online loop, 3 seeds, cert-gated vs fixed ladder vs ungated; the return-vs-env-steps figure.
- **Decision point:** if Stage A shows cert-gating doesn't change learning at all, or Stage B shows no gain → INCONCLUSIVE,
  reported like step85b's honest C-negative.

## 7. Risks & honest probabilities

- Loop wiring correct + runs (Stage A): ~0.85 (standard components; RL is bug-prone — the equivariance + zero-gradient
  tests de-risk).
- Sample-efficiency win lands (G2, Stage B): ~0.4 (the seed's number; the §5.19 conservatism is the live risk).
- Outcome robustness: even INCONCLUSIVE is a publishable honest negative ("cert-gating the imagination horizon does not
  buy sample efficiency on this task — the conservative horizon cuts useful gradient too"), parallel to step85b's C.

## 8. Out of scope (YAGNI)

- No full DreamerV3 (no discrete latents, no symlog/twohot, no replay prioritization) — a *minimal* actor-critic +
  imagination is enough to test the cert-gating hypothesis on 1 GPU.
- No new env (reuse controlled Lorenz-96); no discrete-action variant (continuous control is the clean PG setting).
- This is the **last** of the three directions; run only after ① (step86) is settled.
