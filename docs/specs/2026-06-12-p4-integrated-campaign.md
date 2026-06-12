# P4 integrated campaign — planner v2 × C2 θ* × C4 task-fixed ε (one arc, banked pairs)

*Registered 2026-06-12 before any run. Closes the global-review gap: the night of 06-11/12
landed C3 certificate-side (75 qualifying audits); the TITLE's application layer — $H^*(\epsilon)$
actually spacing a planner's subgoals — still rests on Stage-1b seed-0 with the registered
quasi-static-vacuity concern. This campaign runs all three remaining ladder components on the
same infrastructure: the 9 banked candidate pairs (aux0.5_v0.3 × c2000; θ-R² 0.961) + the
degenerate CUDA pair (ckpt8_cuda_champ_r0, δ̂ 14.08) as the C4 detection target.*

## Shared setup

- Pairs: `ckpt8_cand_champ_r{0..9}` stable 9; degenerate `ckpt8_cuda_champ_r0` (C4 only).
- Env/task: PushT κ=0, FIXED_GOAL (G-trainer convention); success tolerance τ = (pos τ_xy,
  orient τ_θ) — bound in D1 from the env's own success thresholds (no invention).
- Planner: Stage-1b CEM stack, budgets FROZEN at Stage-1b defaults. **Contingency (registered
  now): one budget bump allowed only if the no-plan diagnostics show CEM-side failure (reach
  metrics ≈ random-action row on motion-selected windows); flagged as amendment if used.**
- All boundary/gate mechanics via `src/audit/gates.py` (import-only rule).

## Component A — planner v2 (E1.2 as registered in the spine spec)

1. **Motion-selected windows:** evaluation windows with $\lVert z(t{+}H) - z(t) \rVert >
   4\hat\delta$ (kills the quasi-static vacuity).
2. **Control rows:** zero-action and random-action planners on the same windows.
3. **Tightened $\epsilon_{\mathrm{reach}}$ [bound in D1]:** $2\hat\delta$ per-pair (formula
   registered; value bound from each pair's audit).
4. **The application readout:** reach-rate vs subgoal spacing $H \in \{1, 2, 4, 8\}$ (fixed-H
   baselines — the FF-JEPA-style unprincipled choices) vs **$H^*(\epsilon_{\mathrm{reach}})$
   chosen a priori by the certificate**.
   - **G-P2 (sign+band, registered):** certified spacing's reach-rate is (i) strictly above
     both control rows, and (ii) within 10 points of the post-hoc oracle-best fixed H, in
     ≥ 7/9 pairs. *The claim is "the certificate picks a good spacing WITHOUT search," not
     "beats the oracle."*

## Component B — C2 θ* elicitation (Prop 11: θ* must be task-elicited)

1. Probe calibration per pair: ridge probes (xy + θ) on held-out; empirical quantile map from
   latent error to state error.
2. **$\epsilon_{\mathrm{task}}$ elicitation [D1-bound]:** the latent radius whose induced
   state error stays within τ (90th-percentile mapping — formula registered, numbers bound).
3. Readout: $H^*(\epsilon_{\mathrm{task}})$ per pair.
   - **G-C2 (registered):** task-elicited budgets are non-vacuous — $H^*(\epsilon_{\mathrm{task}})
     \ge 1$ in ≥ 80% of healthy pairs; descriptive: alignment of $H^*(\epsilon_{\mathrm{task}})$
     with the planner-optimal fixed H from Component A.

## Component C — C4 task-fixed ε (decision-scope + degeneracy detection)

1. Apply the SAME $\epsilon_{\mathrm{task}}$ (from B, healthy-pair median) to ALL pairs
   including the degenerate one.
   - **G-C4a (degeneracy detection, registered):** degenerate pair $H^*(\epsilon_{\mathrm{task}})
     = 0$ while ≥ 80% of healthy pairs ≥ 1 — turning the self-scaled-ε blind spot (06-12
     methodological note) into a measured detection result.
2. Decision-scope row (Prop 11 alignment check at c=1): aligned-decision regret spot check on
   Component A's planner outputs — descriptive, scope-language per paper2's Prop 11.

## Phases

- **D1 (CPU, ~30 min):** bind τ, ε_reach (2δ̂ per pair), ε_task (quantile map) → freeze in the
  artifact header BEFORE any planner run.
- **D2 (MPS/CPU, ~2-3 h):** Component A planner sweeps (9 pairs × {H*, 1, 2, 4, 8, zero, rand}
  × motion-selected windows).
- **D3 (CPU, ~30 min):** Components B+C readouts off D2 outputs + audits.
- **D4:** verdicts (G-P2 / G-C2 / G-C4a), ledger, paper-figure data.

## What this closes

G-P2 ⇒ the title's application layer. G-C2 ⇒ C2. G-C4a ⇒ C4's most reviewer-legible half +
the degeneracy-detection discussion section. Remaining after this: κ=0.8 (rescue running),
3D G1, v1.3 (C2 diagnostics, may be obsoleted by G-C2's direct path — review then).
