# step89c — 84-cell map measured-column thickening (frozen before any cell runs)

Date: 2026-06-11 (evening, after the step89b axis-correction fold)

## What

Re-run ONLY the measured side of the published 84-cell audit (step89 15 cells + step89b 69 cells)
with `n_starts` 20 → **100** per cell. Nothing else changes: same `measure()` (mid-episode starts on
a TRUE episode, open-loop latent under $g$, relative-error crossing, censor at 300), same eps grid
{0.05, 0.1, 0.2}, same per-cell seed convention (seed = the checkpoint seed), same checkpoints.
The certified side is NOT recomputed — ratios are re-formed against the published `T1_steps` from
the existing artifacts.

## Why

Today's headline (the axis correction: growth-set cells calibrated, ratio median 0.95, 10/15 in-band;
bias-dominated 24/42) rests on 20-start medians. 5× starts hardens the load-bearing column.

## Frozen commitments

1. **No gate, no tuning.** This is a statistics-thickening run. The regime cuts (med≤3 bias-dominated,
   med≥5 growth-set) stay exactly as published in the paper; cell membership is recomputed from the
   n=100 medians and the fractions/medians are folded into the papers HONESTLY, whatever they say —
   if the in-band count or the 0.95 median moves, the paper text moves with it, with the n=20 numbers
   retained in the artifact for comparison.
2. **Resumable, incremental, errors retried** (step89b pattern). One JSON:
   `papers/figures/step89c_measured_n100.json`.
3. n=20 canonical artifacts are NOT overwritten.

## Estimate

measure() is ~5× the n=20 cost per cell; 84 cells sequential on the box ≈ overnight. CPU only.
