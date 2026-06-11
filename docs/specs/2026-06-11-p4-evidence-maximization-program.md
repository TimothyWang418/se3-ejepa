# P4 Evidence-Maximization Program (EMP) — registered 2026-06-11

*User directive: deadline is ~14 weeks out; stop optimizing for speed, optimize for evidence
strength. Tuning-on-failure is endorsed — WITHIN the v1.6 two-stage discipline (tune on health
metrics, claims stay clean; every retry declared).*

Everything to date ran at minimum-viable scale (20 epochs, 200 episodes, 96 px, n ≤ 10, quick
probes). This program upgrades scale and coverage, priority = value ÷ cost.

## Tier 0 — embarrassingly cheap, run immediately

- **E0.1 Data-scaling study**: corpus 200 → {500, 1000, 2000} episodes (collection ≈ 15 s/200 —
  minutes of CPU). Readouts: stability rate, δ̂, xy/θ content vs data size. Directly tests
  whether the collapse lottery and θ-darkness are small-data artifacts. *The single highest
  value-per-minute experiment available.*
- **E0.2 Long-training probe-vs-step curves** (= the registered v1.3 instrument, never run):
  200 epochs with probes at {0.3k, 1k, 3k, 10k, 30k} steps under the new stable recipes — the
  content-decay axis measured properly, at last.
- **E0.3 Shape-generalization arm (free moat experiment)**: the env ships 8 block shapes
  ('o','L','T','Z','square','I','small_tee','+'). Train on T, audit certificate transfer on
  unseen shapes — *certificate generality across tasks* is paper-grade evidence nobody asked us
  for and the env gives it away.

## Tier 1 — the unique-moat experiments (the paper's heart, time-rich versions)

- **E1.1 Wedge lane, full protocol** (C1b + C3-wedge): wedge-restricted collection, bases,
  orbit-transfer of certified curves with the $m\,\hat\epsilon_{\max}$ term; n = 10.
- **E1.2 Planner-side v2** (motion-selected windows + no-plan control rows + tightened ε_reach):
  the registered fix, run at n = 10 with dense H grid.
- **E1.3 θ/C2 full attack**: 224 px arm × big-corpus arm × aux-θ-anchor arm (anchor on θ only) ×
  deep-probe check — four registered levers, factorial-lite.

## Tier 2 — frontier re-entry (time permits what the re-scope deferred)

- **E2.1 κ=0.8 with the new stable recipes** (v0.2 floor may fix the 0.56/0.66 marginal cells);
  intermediate κ {0.3, 0.5} — *where does the moat fade?* (the dose-response version of the
  regime story).
- **E2.2 Temporal observability, done right**: explicit difference channel (o_t − o_{t−1}) as an
  inductive bias (targets velocity directly), frame-stack-3 (LeWM-style) arm, both at 224 px.
- **E2.3 Intermediate-κ certificate dose-response** — revives a continuous x-axis for the
  signature figure if E2.1/E2.2 stabilize.

## Tier 3 — the big poles (start now, long lead times)

- **E3.1 3D co-anchor (ManiSkill3 + VN-DGCNN)**: the descope checkpoint is 08-05; with time
  declared a non-issue, recon+install starts NOW (GPU box; own venv).
- **E3.2 C4 closed-loop factorial** (FF-JEPA replication cells; certified-H vs oracle sweep) —
  after E1.2's planner v2 exists.
- **E3.3 Human-demo robustness row** (lerobot 206, registered long ago).

## Statistical standard (applies to all tiers)

n ≥ 10 runs per claim cell (v1.6); collapse rates with exact binomial CIs; headline ratios with
bootstrap CIs; every fail → diagnose → (optionally) tune on health metrics → declared retry. No
result discarded: fails feed the Limitations/jurisdiction narrative that is now a paper asset.

## Sequencing note

Tier 0 starts immediately (CPU-bound parts run alongside the in-flight Stage-B). Tier 1 next.
Tier 3.1 (3D) starts in parallel on the GPU box as soon as a session can be spared — it is the
longest pole and the biggest 9+ lever.
