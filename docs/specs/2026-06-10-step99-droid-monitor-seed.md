# step99 — V-JEPA 2-AC monitor on REAL ROBOT DATA (droid_100), conditional gates pre-registered

**Written before step98's certificate verdict is known** (smoke launched, no λ₁ read yet). All verdict branches are
specified symmetrically below so no gate can be chosen after the fact. Claim wording红线: "real-robot **data**,
offline monitoring" — never "real-robot deployment/control". The monitor is sensor-only/passive by design, so
offline replay of logged episodes is a faithful instantiation of the deployment (the step97 docstring's argument).

## Data

- `lerobot/droid_100` (HF, 100 real Franka episodes, ~464MB). Camera: the first exterior-view key in alphabetical
  order (disclosed choice rule; wrist cams excluded — V-JEPA 2-AC's energy notebook uses a third-person view).
- 20 episodes with ≥ 60 usable frames; frame subsampling factor set by the official `droid-256px-8f` config's fps
  (read from config; disclosed in the JSON).
- Telemetry: actions derived from logged states via the authors' `poses_to_diff` (their convention, their code);
  pose stream from the same log.

## Monitor (identical structure to step94/97)

State = per-frame token block (256×1408, layer-normed, authors' per-frame encode recipe). Read = encode the camera
frame; between reads forecast with the AC predictor under telemetry actions; staleness = relative L2 on flattened
normed tokens; flag iff > θ=0.2 at a read; resync. k ∈ {1, 2, 4, 8} + a k=24 censoring probe. n_ep = 20.

## Sub-classification rule (pre-registered, applied to the measured side)

- **stable** iff ≥50% of k=24 windows censored (never cross θ) AND median one-step rel err < θ/2;
- **bias** iff median one-step rel err ≥ θ/2 OR median crossing ≤ 3 steps;
- else **mixed** → report descriptively, no gate claimed (honest INCONCLUSIVE).

## Conditional gates (one branch activates, determined by step98's certificate + the rule above)

- **If step98 = EXPANSIVE with $T_1(0.2)$:** G8-E = pooled in-situ median crossing ∈ $[T_1/1.5,\ 1.5\,T_1]$
  (deployment pricing, E15-G1a analog without a bench column).
- **If ABSTAIN + measured-stable:** G8-S = invalid@k24 ≤ 10% AND ≥80% windows censored AND telemetry-fault recall
  ≥ 0.9 at k_op = 8 — free monitoring on real robot data.
- **If ABSTAIN + measured-bias:** G8-B = no usable cadence (invalid > 25% ∀ k≥2) AND flooded channel (per-read flag
  rate ≥ 0.35 ∀ k≥2) — the step97-isomorphic "do not deploy" verdict, now on real robot data.
- **If mixed:** no gate; full curves reported, verdict INCONCLUSIVE.

Fault arm (all branches, descriptive unless G8-S active): telemetry corruption — action stream zeroed from
$t_f$ = mid-episode; report pre/post per-read flag rates (+ recall/delay when the baseline channel is clean).

## Outputs

`papers/figures/step99_droid_monitor.json`: per-episode curves, pooled crossings, branch taken, gate verdicts,
camera key, subsample factor, episode ids — everything needed to re-derive the gates in a test.
