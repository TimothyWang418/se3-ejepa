# paper3 experiment manifest — full run list for the B300 campaign (2026-06-12)

> Compute model: B300 (288GB) makes this small-data regime GPU-unbounded. Bottleneck flips to
> "what's worth running." **Division of labor:** TRAINING fans out to the B300 (hundreds of
> models, batched); AUDITS (f64 gap-mode) and ENV work (corpus/render) stay on the box/Mac CPU
> (GPU doesn't accelerate them). Strategy: **train a fleet cheaply on B300, audit on CPU.**
> Every run keeps the registered discipline (pre-registered gates, faithful gates.py, n≥10 for
> claims, runs-not-seeds, disclosure). Scale targets are MAXIMA — trim by value if time-boxed.

## Estimated totals
- **~520 training runs** (vs ~150 to date). At ~1–3 min/run on B300, **~10–25 GPU-hours**.
- **~200 audit passes** (CPU f64, box 24-core + Mac) — the real wall-clock; parallelizable.
- Corpus production (CPU/env): already in flight (1000 eps); +2D corpora cheap.

---

## TIER 0 — 3D lane (the open frontier; highest marginal value, the paper's growth area)

| ID | Experiment | Scale | GPU | Buys |
|---|---|---|---|---|
| T0.1 | **G2 stage-A** VN-JEPA banked recipe | n=12 (was 6) | B300 | tight stability/content stats, VN floor calibration |
| T0.2 | **3D plain baseline** (PointNet-style, matched-param, no equivariance) | n=12 | B300 | the 3D moat's other side |
| T0.3 | **3D data-scaling**: VN × {200,500,1000} eps | n=6 each | B300 | does the low-N moat replicate in 3D? (thesis core) |
| T0.4 | **3D plain data-scaling** (same grid) | n=6 each | B300 | 3D reliability-cliff comparison |
| T0.5 | **3D recipe neighborhood**: var_coef {0.2,0.3,0.5} × aux {0,0.3,0.5} | n=4 each (24) | B300 | 3D stage-A winner pick |
| T0.6 | **VN architecture sweep**: c_vec {16,32,64} × k {12,16,24} | n=3 each (27) | B300 | capacity/locality frontier; param-match plain |
| T0.7 | 3D audit → **C3-3D certificate curves** (faithful gates.py) | all stable pairs | CPU | the C3-3D co-anchor claim |
| T0.8 | **3D orbit transport** (object-pose rotation through real clouds; A2 object-centric) | all stable | CPU | Lemma-2-in-3D (equivariance pays) |
| T0.9 | **3D object-pose readability** (peg/hole pose probe: VN vs plain) | all stable | CPU | the θ-moat 3D analog (equivariant-only) |
| T0.10 | 3D **stride binding** (G1.3 SNR: strides {1,2,4}) | n=3 each | B300+CPU | protocol close; ε-grid binding |
| T0.11 | **3D GO/NO-GO** (if VN δ̂ small enough: planner row on clouds) | conditional | B300+CPU | the application layer's structural path |

## TIER 1 — 2D claim-armoring (cheap, tightens A-grade claims to publication strength)

| ID | Experiment | Scale | GPU | Buys |
|---|---|---|---|---|
| T1.1 | **C3 n-scaling**: champion+plain × c2000 | n=30 (was 10) | B300 | guarantee CI shrinks; 90→114 qualifying audits |
| T1.2 | **Cliff factor separation**: seed vs subsample (2×2 at N∈{2606,3000,3500}) | n=8 each (48) | B300 | closes the rank-correlation disclosure |
| T1.3 | **Finer reliability cliff**: plain N∈{2400,2600,2800,3000,3200,3400,3600,3800} | n=10 each (80) | B300 | publication-grade P(fail) curve |
| T1.4 | **eq cliff**: N∈{1200,1500,1750,2000,2250} | n=8 each (40) | B300 | pins eq's own threshold (figure completeness) |
| T1.5 | **κ=0.8 full C1a treatment**: winner_v02 + (aux0.5+10d) base × audit | n=10 each | B300+CPU | C1a proper (expansive regime certificate) |
| T1.6 | **More OOD shapes** (all 8 PushT shapes, not 4) + finer wedge (45°/30° grids) | audits | CPU | OOD breadth for the guarantee figure |
| T1.7 | **Cal-conservatism correction confirm** (λ=1.5 on FRESH runs, registered) | n=10 | B300+CPU | upgrades claim 2 from analysis to confirmed |
| T1.8 | **θ-probe robustness**: vary probe capacity, held-out splits | existing pairs | CPU | armors claim 5 against probe-design attacks |

## TIER 2 — robustness, ablations, generality (reviewer pre-emption)

| ID | Experiment | Scale | GPU | Buys |
|---|---|---|---|---|
| T2.1 | **Matched-FLOPS moat row** (not just matched-param/compute) | n=10 | B300 | the last "unfair comparison" attack |
| T2.2 | **Recipe ablation grid** (the v1.6 sweep at n=10, not n=3) | ~20 cfg | B300 | stability-content tradeoff curve, publication n |
| T2.3 | **Second 2D env** (if swm ships another) — moat generality | n=10 | B300+CPU | single-env limitation softened |
| T2.4 | **Predictor depth/width ablation** (C_N regular predictor capacity) | n=6 | B300 | certificate-quality vs predictor capacity |
| T2.5 | **EMA decay sweep** {0.95,0.99,0.995,0.999} × stability | n=6 each | B300 | EMA's role in the stability tax |

## TIER 3 — reach (only if volume is truly unlimited; high-variance payoff)

| ID | Experiment | Scale | Buys |
|---|---|---|---|
| T3.1 | **Cross-architecture**: a second equivariant backbone (steerable CNN variant) vs VN | n=6 | generality of the equivariance economics |
| T3.2 | **Larger 3D corpus** (full 1000+ → 5000 via more demo sources) | corpus + n=10 | 3D data-scaling tail |
| T3.3 | **ATM-style action-decodability panel** (wiki B1.1) across all 3D pairs | CPU | training-loop instrument; stable-but-empty detector |
| T3.4 | **Second 3D task** (LIBERO-spatial or another ManiSkill task) | corpus + n=6 | 3D generality (timeline-gated) |
| T3.5 | **Certified-spacing-as-curriculum** (use H* to schedule training) | exploratory | a forward-looking discussion result |

---

## Execution plan on B300

1. **Phase A (now):** rsync code + corpus (≤1GB) → install torch → smoke (1 VN run). Verify
   determinism note (B300 arch ≠ 3080; runs-not-seeds covers it; record the flag).
2. **Phase B — 3D fleet (Tier 0 train):** ~120 runs, fan out. Pull ckpts back; **audit on box
   CPU** (the f64 gap-mode is the wall-clock, not the training).
3. **Phase C — 2D armoring (Tier 1 train):** ~250 runs. Same audit-on-CPU pattern.
4. **Phase D — Tier 2/3 as time allows.** Each phase ledgered before results, gates pre-set.

## What B300 does NOT help (stays local)
- gap-mode audits (f64, CPU-bound) → box 24-core + Mac
- corpus production / env stepping (Vulkan/physx CPU) → box
- figure generation, ledger, planning → Mac

## Discipline (unchanged, non-negotiable)
Pre-registered gates per experiment; faithful `src/audit/gates.py` only; n≥10 for any claim
number; runs-not-seeds language + device recorded; every tier ledgered in paper3_record.md
BEFORE results; disclosure of every deviation. Volume scales evidence, never lowers the bar.
