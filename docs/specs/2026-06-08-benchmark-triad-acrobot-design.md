# Benchmark Triad on Acrobot-v1 (step84) — design spec

*Date: 2026-06-08 · Status: design, pre-implementation · Owner: SE3-EJEPA / Certified World Models · Compute: RTX 3080 (CUDA) WSL2, git-mediated*

## 1. Goal

The spotlight experiment: on a **learned model of a recognized chaotic control benchmark**, show the certified predictability horizon is (i) *accurate* (≈ the model's measured rollout-divergence horizon), (ii) *actionable* (a horizon-gated planner beats a horizon-blind one on **task return**, ablated to the certificate), on (iii) an environment where the **horizon is the binding constraint**. Target: move the ICLR paper from ~7 toward 8/oral. **Honest pre-commitment:** if the binding pre-check (§5-iii) fails — return is flat in planning depth, the step79-D1 failure mode — the triad is reported **INCONCLUSIVE** and we fall back to the D2 replan-cadence framing. No binding, no claim.

## 2. Environment — Gymnasium `Acrobot-v1`

Standard underactuated double pendulum, swing-up. Obs $s\in\mathbb R^6=[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\dot\theta_1,\dot\theta_2]$; action $a\in\{-1,0,+1\}$ (torque); reward $-1$/step until the tip clears the bar (episode $\le 500$). **Return $=-$(steps-to-goal)** — faster swing-up ⇒ higher return; well-defined despite sparse reward.

- **Z₂ reflection symmetry** $g$: $\theta_i\mapsto-\theta_i$ (so $\sin\theta_i,\dot\theta_i$ flip, $\cos\theta_i$ fixed) and $a\mapsto-a$. The Acrobot dynamics commute with $g$. The equivariant WM uses **frame averaging over Z₂** (step81's device).
- **Chaotic?** Acrobot swing-up exhibits sensitive dependence; VERIFY $\lambda_1>0$ on the learned WM (Benettin-QR). If the operative regime is *not* chaotic ($\lambda_1\approx0$), that is the PushT degenerate branch and the triad is INCONCLUSIVE (report honestly).
- Dep: `gymnasium[classic_control]` (pure-python classic control, no MuJoCo). The runner verifies/install it.

## 3. World model (GPU, float32) + certificate (CPU, float64)

- **Equivariant WM** $\hat\phi_\theta(s,a)\to s'$: a Z₂-frame-averaged action-conditioned net (discrete-action embedding), trained on collected transitions with a $K$-step rollout loss. **Non-equivariant baseline** of matched capacity. Train on CUDA, **float32**.
- **Certificate (control setting).** The relevant divergence is *along the executed/planned trajectory*: the finite-time leading Lyapunov exponent $\lambda_1^{(T)}$ of the **product of action-conditioned Jacobians** $\prod_t D_s\hat\phi(s_t,a_t)$ along a swing-up rollout (reuse step78 `qr_logR_series`/`bootstrap_spectrum_ci`). Certified horizon $T_1(\epsilon)=\log(1/\epsilon)/\lambda_1^{(T)}$ (+ step82 cone where it certifies; bootstrap CI otherwise). Computed in **float64 on CPU** (small Jacobians; 3080 fp64 is gimped).
- **Measured rollout-divergence horizon**: perturb $s_0$ by $\epsilon$, roll the WM under a fixed action sequence, first-crossing step where $\lVert\Delta_t\rVert>\epsilon_{\rm res}$ — the empirical horizon for triad (i).

## 4. Planner — CEM/MPPI-MPC over discrete actions, via the WM

Sample action sequences $a_{0:H}$, roll the WM, score by predicted return (negative predicted steps-to-goal / a shaped tip-height proxy), commit the first action (receding horizon). Discrete actions via MPPI / random-shooting over $\{-1,0,+1\}^H$. **Plan depth $H$** is the lever.

## 5. The triad protocol

- **(i) certified ≈ measured.** Certified $T_1(\epsilon)$ vs the WM's measured rollout-divergence horizon, across $\epsilon\in\{0.01,0.1,0.3\}$ — the E2 lift onto a *control* env. Report the ratio + the two-regime ε story (Prop-8 δ-bias at small ε, predictive at the asymptotic-Lyapunov ε).
- **(ii) the return win (spotlight).** PRIMARY — **plan-depth gating**: horizon-blind planner uses a long fixed $H_{\text{blind}}\gg T_1$; **cert-aware** caps $H=T_1(\epsilon)$ (*computed, not tuned*). On chaos, $H>T_1$ optimizes against a divergent WM tail ⇒ worse committed action ⇒ lower return. **Ablation:** identical planner, only $H$ differs. Report **net return across $\ge3$ seeds**, cert-aware vs a *sweep* of blind $H$. FALLBACK — **D2 replan-cadence** (the step79-landed result): with a per-replan cost $c$, cert-aware replans every $T_1$ steps; beats the blind fixed-cadence sweep on net return.
- **(iii) horizon-binding pre-check (the make-or-break).** Plot **return vs planning depth $H$**: it must have an **interior optimum near $T_1$** (trusting the WM past $T_1$ costs return). Flat-in-$H$ ⇒ not binding ⇒ INCONCLUSIVE.

## 6. Honest gates (never loosen — INCONCLUSIVE instead)

- **G0 (chaotic):** learned-WM $\lambda_1>0$ in the operative regime; else degenerate-branch INCONCLUSIVE.
- **G-i:** certified vs measured horizon consistent within the two-regime ε story (report ratios; not a hard pass/fail but must not contradict).
- **G-binding (iii):** return-vs-$H$ has an interior optimum (cert-aware $H{=}T_1$ within the good band). **If flat, STOP — report INCONCLUSIVE, no win claimed.**
- **G-ii (the win):** cert-aware net return $\ge$ best *swept* blind $H$ on $\ge2/3$ seeds **and** strictly $>$ both a too-shallow and a too-deep blind planner. Equivariant WM is the substrate (its faithful spectrum is what makes $T_1$ trustworthy); report the non-equivariant-WM variant too.

## 7. GPU / precision / execution

- Device-agnostic: `cuda` if available (the 3080) else cpu/mps. WM train + CEM rollouts **float32/GPU**; spectral readout **float64/CPU**.
- **Deliverable:** `experiments/step84_certified_control_benchmark.py` (CUDA, device-agnostic) + `tests/test_step84.py` (fast, CPU, NO training: Z₂-equivariance of the WM + the Acrobot reflection map, certificate units, planner runs, return monotone-in-success) + `run_step84.sh` (one command: verify `torch.cuda.is_available()`, ensure `gymnasium[classic_control]`, run the full sweep, write `papers/figures/step84_certified_control_benchmark.{png,json}`, then `git add … && git commit && git push`).
- **Git-mediated:** I write+push step84 + a CPU smoke-tested runner; the user runs `bash run_step84.sh` on the 3080 (`~/se3-ejepa`, `.venv`); results commit+push back; I pull + fold into the papers (a new E11 / §5.x, only if the gates pass).
- **CPU smoke (Mac, pre-push):** a tiny config (few episodes, small WM, short sweep) must run end-to-end + tests pass, so the 3080 run is the *scale-up*, not the *debug*.

## 8. Risks (honest)

- **Binding may fail** (step79-D1 precedent): swing-up may stabilize/solve before chaos bites, or CEM may pick OK first actions despite a noisy tail. Pre-check (iii) gates this; fallback to D2. Prob the plan-depth win lands cleanly: **~0.45**; with the D2 fallback included, prob *some* return-win lands: **~0.7**.
- **Sparse reward** makes MBRL harder; mitigate with a shaped tip-height proxy for the planner's internal scoring (the *reported* return stays the true −steps-to-goal).
- **Acrobot chaos may be weak/regime-dependent**; G0 gates it.
- This is the expensive bet; INCONCLUSIVE is an acceptable, honestly-reported outcome.

## 9. Out of scope (YAGNI)

- No new theory (reuse Theorem B/B′, step78/82). No pixels. No multi-environment sweep (one recognized benchmark, done well). No Dreamer-scale agent — CEM-MPC over the learned WM suffices for the plan-depth ablation.
