# Three directions to a genuine 8/oral — seed spec

*Date: 2026-06-08 · Status: seed (build-ready, pre-brainstorm) · Owner: SE3-EJEPA / Certified World Models · Compute: RTX 3080 (CUDA) WSL2, git-mediated*

## Why these three

step84 (Acrobot) showed the certificate is *accurate* and *binds for planning* on a recognized benchmark, but the clean "cap-at-$T_1$ beats best-tuned-blind on **return**" win is INCONCLUSIVE because the return-optimal plan depth is $\sim T_1/2$ — the certificate is a **sound, $\sim2\times$-conservative upper bound** on the trustworthy horizon, not the return-optimum. The lesson: **stop asking the certificate to be the *optimum*; ask for a downstream win where its *proven property* (a sound conservative bound, $T_{\text{guar}}\le T_{\text{true}}$, verified everywhere in step82) is exactly what's needed.** Each direction below does that, differently. Execute **one per fresh session**: brainstorm the chosen one → spec → CPU-smoke → 3080. Suggested order by defensibility/confidence: **③ → ① → ②** (the user asked to try all three).

---

## ③ Structure → trustworthy certificate → downstream win (reuses two LANDED results; ~0.5–0.55)

**Thesis (closes the paper's own loop).** "Structure buys a *trustworthy* certificate." Demonstrate end-to-end: the $\mathbb{Z}_N$-equivariant WM's *faithful* high-$N$ spectrum (E2/Step 83: at $N{=}40$ the dense/recurrent spectrum is garbage, $R^2<0$) makes **its** certified horizon *correct*, while the non-equivariant WM's certified horizon is *wrong* (its $\lambda_1$ is inflated $\sim3\times$). So an agent acting on the **equivariant** model's certificate succeeds; one acting on the **non-equivariant** model's certificate **fails** — not sub-optimally, but because it trusts a *wrong* horizon.

- **Task.** The **D2 re-observation decision** (step79, which *landed*) at **high $N$** (Lorenz-96, $N{=}40$): an open-loop forecaster re-observes every $T_1(\epsilon)$ steps. Compare re-observation timed by the **equivariant** certificate vs the **non-equivariant** certificate.
- **Metric.** Forecast-violation rate vs observation-count (the step79 frontier), **equivariant-cert vs non-equivariant-cert**. The non-equiv's inflated $\lambda_1$ → too-short $T_1$ → over-observes (wasteful) OR (depending on sign) its wrong spectrum mistimes → worse frontier.
- **Why it dodges step84's trap.** It is a **within-method contrast** (equiv-cert vs non-equiv-cert), NOT "beat a tuned blind baseline" — so the $\sim2\times$ constant factor cancels (it hits both); the win is that the non-equiv certificate is *wrong*, which is the paper's whole thesis.
- **Gate.** Equivariant-cert agent strictly beats non-equivariant-cert on $\ge2/3$ seeds, AND the gap is attributable to spectrum faithfulness (cross-reference E2 $R^2$). INCONCLUSIVE if the wrong spectrum doesn't actually mistime re-observation.
- **Build.** Reuse step74 (high-$N$ conv vs MLP spectra) + step79 (`reobservation_contrast`, the landed D2). `step85`. Mostly Mac-runnable (reuses cached spectra); 3080 optional.

## ① Certificate as a safety / abstention bound (most novel reframe; ~0.6)

**Thesis.** A *sound conservative* horizon is exactly what **safety** wants. Flip the objective from return-optimality (where the $2\times$ gap hurt) to **constraint-satisfaction / catastrophe-avoidance** (where the conservative bound is the *guarantee*).

- **Task.** A constrained chaotic control task where **over-trusting the model drives a constraint violation**: planning against a *diverged* model rollout pushes the real system into an unsafe region (a velocity/angle bound, a keep-out region). The **cert-aware** agent re-observes / falls back to a safe action every $T_1(\epsilon)$ steps (before the model is stale); the **horizon-blind** agent trusts the model deep and is driven into violations because the planned trajectory and reality diverge past $T_1$.
- **Metric.** **Constraint-violation rate / catastrophe rate** (NOT return), cert-aware vs a sweep of blind cadences/depths, $\ge3$ seeds.
- **Why it dodges step84's trap.** Safety wants the *conservative* side of the horizon; the certificate's soundness ($T_1\le$ true horizon) *is* the safety certificate. The $2\times$ conservatism is now a **feature** (you re-observe early = safe), not a loss.
- **Gate.** Cert-aware violation-rate strictly $<$ best-tuned-blind on $\ge2/3$ seeds AND the blind agent's violations are *caused* by trusting past $T_1$ (ablate). INCONCLUSIVE if the constraint isn't horizon-binding.
- **Build.** Reuse step81/84 dynamics + WM + certificate; add a state constraint + a safe-fallback action + a violation counter. `step86`. 3080 or Mac.

## ② Certificate-gated MBRL sample efficiency (heaviest, riskiest; ~0.4)

**Thesis.** Use the certified per-orbit horizon to **gate imagined rollouts** in model-based policy learning (Dreamer-style): only backprop policy gradients through model rollouts within $T_1(\epsilon)$ — beyond it the model is untrustworthy, so its gradients are noise. Cert-gated imagination should be more **sample-efficient** than fixed-horizon imagination.

- **Task.** A standard MBRL suite task with genuine chaos (a chaotic control task; DMC tasks are mostly near-neutral — pick carefully or use a chaotic pendulum).
- **Metric.** Return vs env-steps (sample efficiency) / final return; cert-gated vs best fixed-imagination-horizon vs ungated, $\ge3$ seeds.
- **Gate.** Cert-gated $\ge$ best fixed-horizon on $\ge2/3$ seeds. INCONCLUSIVE if no gain (it is close to the step84 plan-depth failure mode — flag this risk).
- **Build.** Heaviest — needs a Dreamer-ish actor-critic + per-state certificate. `step87`. 3080 required. **Do this last** (lowest confidence, most compute).

---

## Honest meta

- All three keep the same discipline: **INCONCLUSIVE rather than loosen a gate**; the certificate's *proven soundness* is the load-bearing property, not an assumed optimality.
- Realistic: landing **one** of these cleanly takes the paper to ~8. ③ is the most defensible (reuses two landed results, within-method contrast); ① is the boldest reframe (safety); ② is the stretch.
- Each is its own fresh-session brainstorm → spec → `step8X` → 3080, folded only if its gate passes — exactly the step79–84 cadence.
