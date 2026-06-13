# paper3 skeleton v0 — claims→evidence map, outline, figures (drafted 2026-06-12, mid-ladder)

> Working scaffold, NOT prose. Purpose: expose holes while there is still quarter-left to fill
> them. Every claim row cites ledger entries (paper3_record.md). Title question flagged below —
> resolve against the proposal's pre-registered pivot rule, not by taste.

## 0. Title question (flag, do not decide here)

Registered: *Certified Subgoal Spacing for Equivariant World Models*. Campaign D4: spacing-as-
OPTIMIZER is NO-GO at 96px; spacing-as-REFUSAL (feasibility gate) is validated by agreement.
Ladder may revive the optimizer reading at 192px (in flight). Candidate reframings if the
pivot rule triggers:
- *Certified Horizons for Equivariant World Models: Guarantees, Feasibility Gates, and the
  Price of Pixels*
- *When Not to Plan: Certified Horizon Gates for Latent World Models*
Decision point: after ladder + (if GO) the 192px D2 rerun.

## 1. Claims → evidence map

| # | Claim | Evidence (ledger) | Grade | Hole / risk |
|---|---|---|---|---|
| 1 | **Certificate-as-guarantee**: $H_{\mathrm{cert}} \le H_{\mathrm{meas}}$ under faithful semantics | 38/38 qualifying runs, 2 devices × 2 archs (champion both arms); +30/30 shape-OOD (E0.3b); +7/7 pose-OOD (wedge G-W1) = **75 audits, 0 anticonservative** | **A** | censored-(8,8) language ("within audited horizon"); degenerate self-scaled disclosure |
| 2 | Certificate-as-point-estimate is conservative | C3-cal FAIL both devices, median ratio 0.67, 37% < 0.5; G-pre rate 2.3–2.7× | A- | can conservatism be post-hoc calibrated? (registered idea, unrun) |
| 3 | **Certificate as planning-feasibility gate (GO/NO-GO)** | Campaign D4: $H^*\!\le\!1$ predicted planner infeasibility; empirics agreed (planner ≈ random; truth-replay 10/10 exonerates pipeline; SNR ≈ 0.5) | B+ | **GO side untested** — needs ladder-GO 192px D2 or 3D lane |
| 4 | **Low-N reliability moat**: eq safe ≈ 2000 vs plain ≈ 4000 transitions (≈2×) | P(fail) table n=3–6/point; zone bimodal 2606–3500; cliff entries | B+ | rank-correlation disclosed (caption); factor separation optional |
| 5 | **θ-moat does not invert at high N** (group coordinate is equivariant-only) | plainc θ-R² 0.29 linear / 0.11–0.19 MLP vs eq 0.91–0.96 @40k transitions | A- | probes on 5 pairs/arm; could extend to 10 cheaply |
| 6 | High-N content inversion (xy: plain > eq at 40k) — two-regime honesty | 0.671>0.317 (MPS) and 0.689>0.421 (CUDA), n=10×2 | A- | Brehmer must-cite; framing discipline |
| 7 | Orbit transport (Lemma 2 through pixels) | n=7, all ≤ 4.1% (wedge v2) | A- | C16 quantization residual documented |
| 8 | Instrument exactness | C_N fiber exact; VN 1e-16; pairing bit-exact; canonical gates equivalence-tested | A | — |
| 9 | κ-regime map + aux inversion at κ=0.8 | rescue sweep: winner 3/4 vs aux 1/4; dose ordering; non-Markov anchoring hypothesis | C | n=4; 10-d falsification queued; HYPOTHESIS language only |
| 10 | **Price of pixels: both 2D levers measured dead** — data flat; resolution ratio 0.5→0.75→0.3 (away from GO at top rung; off-tune caveat) | ladder complete, NO-GO 0/12 | **A-** | thesis-aligned; 192px-tuned-recipe rebuttal possible but expensive |
| 11 | 3D feasibility (no pixel floor) | G0 all-pass; G1a open-loop FAIL 0/20 (honest, mechanism known) → closed-loop v2 next; G1.2 timescales banked | slot | **the ONLY remaining GO path** — timeline is the paper's main risk |

## 2. Section outline

1. **Intro.** Pixel world models are expensive at exactly the precision tasks need; we ship
   certificates instead of promises. **Opening exhibit — the field picks $H$ by hand:** FF-JEPA
   fixes subgoal spacing $H{=}25$, EA-WM's "long-horizon" online planning runs $H{=}20$ (wiki
   D2.1) — both unprincipled constants; we DERIVE $H^*(\epsilon)$ and, where it says NO-GO, the
   hand-picked $H$ silently fails (our campaign §5). Contributions: (i) faithful-semantics
   certified horizons with guarantee-grade evidence (75 audits, dual OOD, dual device); (ii) the
   certificate as an a-priori feasibility gate — including an honest NO-GO it called correctly;
   (iii) the equivariance economics: reliability at low N and exclusive access to the group
   coordinate; (iv) the price-of-pixels analysis (resolution, not data, is the binding lever).
   *Thesis backing (latent prediction > pixels), three independent third-party bricks: PoR
   (action $R^2$), YoCausal (causal probes), "Do VFMs understand intuitive physics" — V-JEPA
   tops all (wiki F1).*
2. **Setup.** JEPA deployable pair; audit instruments + known biases; Thm A/B + Lemma 2
   inheritance (cite paper2); $H^*(\epsilon)$; canonical gate semantics ((0,0) passes,
   censoring language, degenerate self-scaled disclosure).
3. **Certified horizons hold.** C3 results: cross-device table, OOD lines, cal conservatism
   with direction. (F1, F2, F5)
4. **The equivariance economics.** Low-N reliability cliff; θ-moat (linear+MLP); high-N
   inversion; two-regime statement (Brehmer-consistent). (F3, F4)
5. **From certificates to planning: the feasibility gate.** Planner v2 protocol; the NO-GO
   verdict and the agreement chain (truth 10/10 / coherent-CEM / SNR); D1 elicitation
   (τ from env, ladder); [SLOT: 192px GO rerun]. (F6, F7)
6. **Beyond pixels.** [SLOT: 3D lane G0/G1 + first VN numbers if ready]. The pixel floor
   argument makes point clouds the natural home for task-precision certificates. **3D
   geometry-injection spectrum (the C2 "learn vs guarantee" comparison matrix, wiki E2/D2.2):**
   OASIS (SE(3) output, non-equivariant) → MRO-GWM (canonical-frame + rigid-transform, learns
   rigidity, non-equivariant) → EquiDexFlow (VN, true-equivariant, generative side) → **ours
   (equivariant JEPA + certificate)** — increasing geometric prescription; the non-equivariant
   points are predicted to degrade on OOD rotation where ours holds by construction. Eval proxy:
   WorldOlympiad's geometry track (GS-reconstruction as a 3D-consistency surrogate, wiki F2).
7. **Related work.** *Subgoal spacing / hierarchical planning (3 tiers of structure):* FF-JEPA
   (no structure, fixed $H{=}25$) / WorldDP (hand-injected object-centric structure) / ours
   (group-prescribed) — and our gate refuses where the fixed-$H$ lines silently fail; HWM
   (post-hoc error-horizon analysis vs our a-priori). *Equivariance economics:* Brehmer 24
   (data closes equivariance gaps — our two-regime refinement: content yes, reliability and
   group coordinate no). *Manifold solvers (orthogonal, composable):* LieIPM optimizes
   trajectories on KNOWN dynamics; we certify the trust window of a LEARNED model — a manifold
   solver can slot in below a certified spacing (note: latent $\mathrm{SE}(3)$ $\exp/\log$ needs
   the Cartan–Schouten connection, non-compact Lie exp ≠ Riemannian exp). *Symmetry handling:*
   EquiDexFlow AVOIDS gravity tension (yaw-only); our $\kappa$-gate MEASURES it (avoidance vs
   measurement). *Audit/instrument neighbors:* ATM action-decodability (orthogonal: direction-
   right vs how-far-trustworthy). V-JEPA/Dreamer lines; certified/safe control. Novelty-sweep
   must-cites.
8. **Honest accounting (limitations).** Cal conservatism; 96px resolution floor; MPS
   stability tax; rank-correlation in cliff cells; single 2D env + early 3D; aux inversion
   at κ=0.8 is hypothesis-grade.

## 3. Figures inventory

| Fig | Content | Data status |
|---|---|---|
| F1 | Certified vs measured curves + dual-boundary cells schematic | HAVE (champion artifacts) |
| F2 | C3 cross-device guar/cal summary | HAVE |
| F3 | **P(fail) vs N reliability cliff** (eq vs plain; zone shading; rank caveat in caption) | HAVE |
| F4 | θ-moat bars: linear & MLP probes × {eq, plain} | HAVE |
| F5 | OOD one-sidedness: 4 shapes × pose-wedge | HAVE |
| F6 | GO/NO-GO mechanism: $H^*$ vs empirical reach (all rows ≈ random at $H^*\!\le\!1$) | HAVE |
| F7 | Resolution ladder: ε_task/δ̂ ratio vs rung | IN FLIGHT |
| F8 | 3D lane: G1 conversion validation + first VN audit | SLOT |

## 4. Holes the map exposes (work queue, ordered)

1. **GO-side validation of claim 3 — 2D branch CLOSED (ladder NO-GO 0/12). 3D is the only
   remaining path**: G1a-v2 closed-loop conversion → corpus → VN training → 3D planner row.
   Fallback ship-shape: claim 3 as "validated in the refusal direction" + 3D early evidence.
2. ~~Matched-COMPUTE moat row~~ DONE (compute doesn't rescue plain, ledgered 06-12).
3. ~~Cal-conservatism correction factor~~ measured (λ=1.5 → 93% in band); confirm on fresh runs.
4. ~~θ-probe all 10 pairs~~ DONE (0.946/0.928 vs 0.285).
5. ~~κ=0.8 10-d falsification~~ DONE at n=8 (10-d 6/8 > winner 4/8; banked κ=0.8 recipe).
6. 3D timeline scoping decision by ~07-15: full section vs "early evidence" paragraph.
7. **ATM action-decodability instrument into the 3D stage-B health panel** (wiki B1.1) —
   two-layer MLP $D_{T,T}$ probe per ckpt; catches stable-but-empty (the campaign's SNR failure
   mode). Ordering-only, absolute values not cited (capacity-confound caveat).

## Reinforcement adoption ledger (wiki list B/D2/E2/F, adjudicated 2026-06-13)

USED: TC-WM proprio anchor (B2.3 → banked aux recipe + C2 unlock + κ=0.8 10-d) · PoR
dimension-wise probes (B3.6 → θ-moat) · EquiDexFlow triple (D2.2 → VN feasibility, eq
verification, avoidance-vs-measurement A2). FOLDED INTO SKELETON (this edit): EA-WM H=20 +
intuitive-physics/YoCausal bricks (§1) · MRO-GWM/OASIS geometry spectrum + WorldOlympiad eval
proxy (§6) · WorldDP/LieIPM/ATM (§7). DEFERRED: PRISM (B1.2, 2D planning closed) · QGF/Echo
(B4, conditions untriggered).
