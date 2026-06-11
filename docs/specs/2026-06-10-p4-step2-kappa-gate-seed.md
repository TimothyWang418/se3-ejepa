# P4-step2 — κ validation gate: does damping dial a measurable spectrum? — seed spec

*Date: 2026-06-10 · Status: seed (build-ready; runs concurrently with step1's training grid) ·
Owner: paper3 · Compute: Mac CPU, minutes*

> The proposal (§3.2(2)) pre-registers this gate BEFORE the spine (E-P4.3's κ sweep) bets on the
> knob: $\hat\lambda_1(\kappa)$ must be monotone and CI-resolvable across the κ grid, else the
> spine falls back to a two-point regime contrast. Plan: W1 recon §lane-b. This step measures the
> TRUE environment's leading exponent — no learned model involved.

## Registered discovery (code-level, found pre-build)

`_set_state` (env.py:514–527) sets agent pos/vel + block pos/angle but **never block velocity or
angular velocity** ⇒ the 7-d obs state **under-specifies the κ>0 dynamics** (block momentum is
hidden state). Consequences, registered:

1. step2's twin protocol manages the **10-d extended state**
   $(\text{agent}_{xy}, \text{block}_{xy}, \theta, \text{agent}_{v}, \text{block}_{v}, \omega)$
   via direct pymunk handles (mirroring `_set_state`'s pattern, plus block momentum).
2. **Oracle-CEM sim cloning at κ>0 inherits the hole** (clones start block-at-rest). step1's
   oracle runs at κ=0 only — safe there (damping 0 kills velocities each step). The κ>0 oracle
   (later step) must adopt the extended-state setter. This line is the registration of that debt.

## Protocol (pre-registered)

- κ grid $\{0, 0.5, 0.8, 0.95, 1.0\}$ (0 = default quasi-static anchor; pymunk damping = velocity
  retention fraction).
- Per κ: 10 WeakPolicy episodes × 100 steps (the κ-native data distribution); capture
  (extended state, next-60-action window) at steps $\{10, 25, 40, 55\}$ → 40 pairs.
- Twin trajectories: identical extended state, **block-angle perturbation $\delta_0 = 10^{-4}$
  rad**; replay the common action window; both twins step the same env deterministically.
  **G0 self-check: $\delta_0 = 0$ twins must give $d(t) \equiv 0$ exactly** (determinism witness).
- $d(t)$ = diagonal-normalized config distance: positions /512, angle /$\pi$ (velocities excluded
  from the metric, registered — config-space divergence is what the spine consumes).
- Per-pair exponent: least-squares slope of $\ln d(t)$ on the window $d \in (10^{-7}, 0.05)$,
  units = per control step (10 Hz). $\hat\lambda_1(\kappa)$ = mean over pairs; CI = 1000×
  bootstrap over pairs; median reported alongside (contact-noise honesty).

## Gates (pre-registered, no loosening)

- **GATE-A (monotone):** $\hat\lambda_1(\kappa)$ means non-decreasing across the grid
  (Spearman $\rho \ge 0.8$).
- **GATE-B (resolution):** bootstrap CIs of $\kappa{=}0$ and $\kappa{=}1.0$ **disjoint**.
- PASS = A∧B → spine's κ sweep is GO as designed. B-only → registered two-point fallback
  (static vs best-separated κ). Neither → knob redesign (mass/friction axes; spine blocked).

## Build

`experiments/p4_step2_kappa_gate.py` (reuses `weak_action` from step1's pipeline; ~150 lines):
collect → twin-divergence → slopes → bootstrap → gate verdicts → JSON
(`papers/figures/p4_step2_kappa_gate.json`). SMOKE via `P4_SMOKE=1`. Runs alongside step1's grid
(separate env instances; pymunk single-threaded; no shared state).
