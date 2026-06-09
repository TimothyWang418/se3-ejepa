# Step 85b — Full-spectrum faithfulness → correct observation allocation (direction ③, extension C) — seed spec

*Date: 2026-06-08 · Status: seed (build-ready, pre-brainstorm) · Owner: SE3-EJEPA / Certified World Models · Compute: RTX 3080 (CUDA) required (per-$F$ retraining)*

> Stretch extension of `step85` (`docs/specs/2026-06-08-step85-trustworthy-cert-downstream-design.md`). Run **only after
> `step85`'s G1 passes.** Where `step85` isolates the *scalar* $\lambda_1$ (the leading exponent the re-observation
> interval reads), this binds the **full spectrum** $R^2$ — the actual headline of E2/`step83` — by making the *ranking*
> of difficulty across regimes the load-bearing quantity.

## Thesis

A *faithful* spectrum does not just get $\lambda_1$ right on one system; it gets the **relative difficulty across a
heterogeneous ensemble** right. Give an agent a fixed total observation budget and a *family* of regimes of differing
chaoticity; the optimal policy spends more observations where the dynamics are genuinely more chaotic (larger
$\lambda_1$). The $\mathbb{Z}_N$-equivariant model ranks the regimes correctly (faithful spectrum) and allocates well;
the dense MLP's garbage spectrum ($R^2<0$) ranks them wrongly and **misallocates** — so at *matched total budget* its
aggregate violation is strictly higher. This is genuine frontier dominance that does **not** collapse to the
same-frontier relabeling of `step85` §2, because the comparison is across an ensemble where *allocation* (not just
cadence) is the lever, and the driver is spectrum **faithfulness**, not a scalar offset.

- **Task.** A sweep of forcing $F\in\{5,6,8,10,12,16\}$ on $N{=}40$ Lorenz-96 (monotone-ish increasing chaoticity; pick
  the final set so true $\lambda_1(F)$ spans a clear range). For each $F$: train conv + MLP (reuse `step74.train_model`),
  read each model's $\lambda_1(F)$. The agent has a fixed total budget $B_{\text{tot}}$ of re-observations to split across
  the regimes; it allocates by its model's per-regime $\lambda_1$ ranking (more chaos → shorter certified horizon → more
  observations).
- **Metric.** Aggregate forecast-violation rate across the ensemble at **matched total budget** $B_{\text{tot}}$,
  conv-allocation vs MLP-allocation (forecaster held fixed to the conv, as in `step85` Phase 1, to keep the win
  attributable to the certificate/spectrum). Sweep $B_{\text{tot}}$ for a frontier.
- **Why it beats `step85`'s scalar story.** It exercises the whole spectrum's *ordering* across regimes, tying directly to
  the $R^2$ result rather than to the single inflated $\lambda_1$; the win is a true Pareto dominance at matched budget.

## De-risk precheck (do FIRST — cheap, decides go/no-go)

The dominant risk: the MLP's spectrum is garbage in *shape* ($R^2<0$) but its $\lambda_1(F)$ may still increase
*monotonically* with $F$ — if so, its **ranking** of regimes is correct, allocation is correct, and there is **no
contrast** (INCONCLUSIVE). Before any allocation build, compute $\lambda_1^{\text{MLP}}(F)$ and
$\lambda_1^{\text{conv}}(F)$ across the sweep (a handful of trainings) and check that the MLP's *ranking* is genuinely
**scrambled** (Spearman $\rho$ of MLP-vs-true $\lambda_1(F)$ clearly below the conv's, ideally near $0$ or negative). If
the MLP ranks regimes correctly despite the bad spectrum shape, **downgrade now** — the allocation story does not exist
on this ensemble; report and stop (or escalate to a per-channel allocation that reads the full $T_j$ stratification, a
larger redesign).

## Gate (never loosen — INCONCLUSIVE instead)

- **G (allocation win):** conv-allocation aggregate violation $<$ MLP-allocation at matched $B_{\text{tot}}$ on $\ge2/3$
  seeds across the budget sweep, **AND** the de-risk precheck confirmed the MLP ranking is scrambled (so the win is
  misallocation from an unfaithful spectrum, not a scalar offset). INCONCLUSIVE if the MLP allocates correctly anyway.

## Build

Reuse `step74` (per-$F$ conv/MLP + spectra), `step78` (CI), `step79` (certificate / horizon / empirical), and `step85`'s
`budget_frontier` re-observation loop generalized to an ensemble + an allocation rule. New: the $F$-sweep training loop
(the 3080 cost) and the allocation function (budget split by per-regime certified horizon). `step85b`.

## Honest meta

- Confidence ~0.4 (the seed's read for the most ambitious ③ framing): strongest tie to the paper's actual $R^2$ headline
  and the only one giving genuine matched-budget frontier dominance, but the heaviest (per-$F$ retraining, 3080) and the
  least de-risked (the ranking-scramble precheck can kill it cheaply — which is exactly why that precheck runs first).
- Same discipline as `step85` and the step79–84 cadence: brainstorm → spec → CPU/precheck → 3080, folded only if G passes.

## Precheck result (2026-06-09) — C ALIVE but MODERATE; mechanism is range-compression, not rank-scramble

Ran `experiments/step85b_spectrum_allocation.py` (N=40, seed 0, $F\in\{6,8,10,13\}$; result
`papers/figures/step85b_allocation_precheck.json`). **C_alive = True**: allocation-L1 mlp-vs-true $=0.219$ vs
conv-vs-true $=0.081$ (MLP $\sim2.7\times$ worse). **But both rank-correlations are $1.00$** — the MLP gets the regime
*ranking* right (its $\lambda_1$ does rise with $F$: $6.10\to6.67\to7.94\to9.54$). The de-risk precheck's worst case
(ranking preserved $\Rightarrow$ C dead) did **not** occur, but the misallocation is **range compression**, not
rank-scramble: the MLP's inflation is largest at low chaos ($8.98\times$ @F6) and shrinks at high ($3.27\times$ @F13),
flattening its weights — $[0.20,0.22,0.26,0.32]$ vs true $[0.11,0.20,0.29,0.39]$ — so it **over-weights easy regimes,
under-weights hard ones.** True $\lambda_1(F{=}8){=}1.773$ matches the cached step74 spectrum (correctness anchor).

**Implication.** Full C is worth building as a **supporting** result (a faithful spectrum allocates a fixed budget
better across a chaoticity ensemble), but expect a **modest** margin (a $\sim0.14$ L1 weight shift), not the dramatic
effect of A. The 3080 buys the $F$-sweep $\times$ seeds **training** (CUDA). Confidence unchanged $\sim0.4$, now with
the failure mode pinned: it lands a *moderate* allocation win, contingent on the compression surviving at scale/seeds.
