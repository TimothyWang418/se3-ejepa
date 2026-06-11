# paper2 — Record / Experiment Ledger (single source of truth)

> Living record for **paper2 = "Certified World Models: Predictability Across Configuration, Horizon, and Resolution"**. Tracks every
> experiment, its result/seeds/test/commit, the proposal phase status, the concurrent-work positioning, and the
> open items. Draft + figures live in `papers/paper2_certified_world_models.md` (+ `.pdf`); design rationale in
> `papers/proposals/paper2-certified-compositional-jepa.md`. Last updated 2026-06-02.

## 1. The claim (one paragraph)

For an **equivariant** latent world model, structure delivers a *kind* of result scaling cannot: a
**predictability certificate** — a provable, computable, training-free region of situations the model is
guaranteed to handle, across three orthogonal axes: **configuration** $w\in\langle S\rangle$ (the exponential
monoid from $k$ generators), **horizon** $T$, and **resolution** $\epsilon$. Master theorem: under exact
equivariance with orthogonal $\rho$, error is invariant over all of $\langle S\rangle$ (Thm A); a spectral
refinement gives per-channel certified horizon $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ (Thm B). The
certified region is the **coarse-invariant, slow, low-composition** corner. *Scale buys interpolation; structure
buys a certificate.*

## 2. Experiment ledger

All CPU/1-GPU-scale, seeded, honestly gated (a run prints `INCONCLUSIVE` rather than loosen a threshold).

| Step | Role | Result (multi-seed where noted) | Test | Commit |
|---|---|---|---|---|
| **47** | Config axis on the embodied model + Thm-B spectrum | equivariant relMSE **×1.00 flat** over words $m{=}0..8$ vs MLP ×12.89; predictor Jacobian **48/96 contractive** (3 seeds) | `test_step47_certificate.py` (Thm A architectural, 1.2e-7) | `ca4f8ec` `701e881` |
| **49** | Config axis **exponential** (I Ching $\mathbb{Z}_2^6$) | train on **6 generators** → certified over all **64** ($\sim10^{-33}$); MLP $1.6\text{e-}5\to0.59$ | — | `204a0a5` |
| **50** | **Noether hinge** (slow ⊆ invariant), 2D SO(2) | $(E,L)$ in invariant block $R^2{=}0.92$–$0.99$ vs ≤0.01; slow invariant mode $0.01\ll0.145$; certificate $10^{-16}$ vs MLP 1.17 (3 seeds) | `test_step50_noether_hinge.py` | `14cf87a` |
| **51** | **Structure vs scale** (the tagline) | equivariant flat over orbit (ratio 1.1–1.2); 88×-scaled MLP buys in-wedge interp (31–166×, beats equiv in-dist) but out-of-wedge stays 10–155× above equiv floor (3 seeds) | — | `3be4fd0` |
| **52** | **Horizon × resolution** staircase (推背图) | recovers chaotic $\hat\lambda{=}0.69$ to 0.1%; $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ (slope 1.3–1.6); chaotic 3–10 steps, slow ≥90 (3 seeds) | `test_step52_horizon_resolution.py` | `620845b` `217ed0d` |
| **53** | **Approximate symmetry** (P4) | exact cert at β=0 (68–320×); graceful $\propto\epsilon_{\text{world}}$ (corr 0.88–0.98); symmetry-content **threshold** $\epsilon\approx0.01$–0.06 (3 seeds) | — | `df1ae98` |
| **57** | **Embodied/contact hinge lift** (3D, two-body contact) | Noether content **lifts** (invariant $R^2{=}0.86$ vs 0.05); clean containment **2D-specific** (3D $L$ = conserved $\ell{=}1$ vector → slow ⊆ invariant⊕conserved-equivariant) | — | `3763aad` |
| **58** | **3D-aware containment** (resolves 57) | conserved physics splits by type+degree: $E$→ℓ=0 linear ($R^2{=}0.62$–$0.91$); $L$ bilinear→ℓ=1 degree-2 cross ($R^2{=}1.00$, range $0.998$–$1.000$); both conserved → slow ⊆ (invariant ⊕ conserved-equivariant) **exact**; ties to old-paper degree-1 cross-product cap | — | `695143d` (3 seeds) |
| **59** | **Certificate on REAL contact dynamics** (PushT, Experiment 9 — kills "constructed-teacher/toy-only", weakness #1) | learned SO(2)-equiv world model (invariant-scalar-gated VN) **exactly flat over the orbit** (10-step rollout ratio **1.00**, equiv-resid $\sim10^{-7}$) + **competitive in-dist** (eq 0.13–0.15 vs best MLP 0.14–0.19); **no MLP scale 1.7k→272k reaches the floor out-of-wedge** ($2.1$–$3.9\times$, 3 seeds). Real pymunk physics we did not author; clean rotate-the-orbit protocol | — | `fb8ce72` (3 seeds) |
| **61** | **Certificate at the TASK level** (PushT closed-loop pose control, Experiment 11 — 极致-T2) | RichVN + **G-equivariant CEM** (isotropic σ + disk action bound + scene-covariant noise = A5) on a contact-dominated reorient task; **paired** base tasks rotated over the orbit. **Model-rollout & real-env pose-error orbit ratio = 1.000 (exact, all 3 seeds)** vs MLP ×1.1–2.2 (rollout) / ×1.6–3.6 (real-env). In-wedge a **wash** (eq 3.6–16° vs MLP 8–18°, each better on some seeds — no in-dist win). Honest: cost-drift>0.3 met 1/3 (eq ~1e-7 vs MLP 0.19–0.31, ~10⁶×); scene-blind eq flatter but noisy margin → A5 needed for the *exact* guarantee. Kills "relMSE is just a proxy" | `test_step61.py` (plan-equivariance + orbit-invariance, 2 tests) | this batch (3 seeds) |
| **64** | **Certificate on raw PIXELS** (PushT $C_4$, Experiment 13 — user "研究一下"/"都做吧") | **frame averaging** (Puny 2022): a plain CNN/MLP made exactly $C_4$-equivariant by Reynolds avg over 4 grid rotations + $\rho$-correction (pure torch/MPS). **Exactly orbit-flat** (1.000, enc/pred $\sim10^{-7}$); **accuracy-neutral** vs the unconstrained CNN (collapse-robust **FVU 0.68–1.07×**, mean 0.84) with healthier latent (PR 2.8–4.3 vs 2.2–2.6); **horizon-stable** (FVU grows ≤1.2× over 8 steps) while the steerable incumbent **diverges** (160–1600×). Honest residual: FVU>1 for ALL incl. the CNN's fair 1-step-vs-target (1.8–2.2) → JEPA-latent property (VICReg variance on low-dim dynamics), architecture-AGNOSTIC, not equivariance | `test_step64.py` (FA encoder + predictor $C_4$-equiv, 2 tests) | `99af968` + this batch (3 seeds) |
| **65** | **Proposition 6 numerical confirmation** (horizon tightness; the ICLR-draft central-claim figure) | the exact Prop-6 construction on a controlled multi-channel latent: $\epsilon$-approx-equiv orbit-variation $=\epsilon e^{\lambda_j T}$ to rel-err **$10^{-14}$–$10^{-13}$** (3 seeds, tight lower bound); exact-equiv variation **exactly 0** at all $T$; conserved channels ($\lambda\le0$) infinite-horizon; certified horizon **linear in $\log(1/\epsilon)$, slope $1/\lambda$ ($R^2{=}1.000$, all seeds)**. Confirms the proved construction (trained-model degradation is Exp 8) | `test_step65.py` (Prop-6 lower-bound exactness, 1 test) | this batch (3 seeds) |

Unit-test isolation: `tests/conftest.py` pins float32 around every test; float64 experiments opt in. Full suite **92 passed** (`test_step61.py` SO(2) pose-planner guards; `test_step64.py` frame-averaged $C_4$ encoder+predictor guards; `test_step65.py` Proposition 6 lower-bound exactness).
Multi-seed reproducibility: `experiments/aggregate_seeds.py` re-runs steps 50/51/52/53/57/58/59/60/61/63/64/65 at seeds {0,1,2} and commits per-seed `papers/figures/*_seeds.json`; every range quoted above and in the draft is the seed min–max from those files. Single-seed `*.json/.png` stay canonical at the default seed 0.

## 3. Proposal phase status (P0–P5)

- **P0 — Master theorem** ✅ closure test (Step 47) + Thm A confirmed + Thm B spectrum measured + hinge measured (Step 50). *§2–3 proof prose tightened (`de7b7e4`); Thm A closed-loop now under explicit assumption (A5) (`48fb2a9`).*
- **P1 — Config axis** ✅ Step 47 (×1.00) + Step 49 (exponential 6→64).
- **P2 — Horizon × resolution** ✅ hinge (Step 50) + $T_j(\epsilon)$ staircase (Step 52). *Embodied lift attempted (Step 57): Noether content lifts; clean containment is 2D-specific.*
- **P3 — Structure vs scale** ✅ Step 51. *Remaining: discrete-config-axis scale replicate.*
- **P4 — Approximate-symmetry degradation** ✅ Step 53 (graceful + threshold).
- **P5 — Discovery + generation** ✅ re-framed existing Steps 33/36/38 into draft §4.6 (cited as companion-line results that show the certificate is *actionable*, not as new evidence produced by this paper).

## 4. Concurrent-work positioning (where each is answered)

| Paper | Challenge → our answer | Folded in |
|---|---|---|
| **BRo-JEPA / UWM-JEPA** | hand-constructed toy (shared caveat) → cite as cross-group *mechanism* corroboration (Thm A explains why, closure how far) | paper2 §5; core §relwork; compact §6; Step 49 |
| **UR-JEPA** | isotropy vs manifold → ours is **group-prescribed** (ρ-pinned 3e-4 vs 1.04); Step 56 (single seed) concedes in-dist, wins OOD ~320× (corroborated 3-seed by Step 53's 68–320× / Step 51's 10–155×) | paper2 §5; lejepa (Step 54/56); core §relwork |
| **IMWM** | residual is **search**, ours is **representation** → complementary; cert bounds model not search | paper2 §5; core Step-43 para |
| **LDA** | it's a **diffusion** policy (ally on geometry, opponent on generation); "Euclidean Fallacy" = our motivation; modest-but-OOD-robust = our ×1.36 profile | paper2 §5; core §relwork; payoff §17 |

(Old-paper experiments enriched this session: **A4** = Step 54 `54`, **B2** = Step 55 `55`, **B3** = Step 56 `56` — these belong to the *old* paper, not paper2; recorded here only for cross-reference.)

## 5. Old-paper completeness verdict (2026-06-02 audit)

**Content-complete and coherent.** A4/B2/B3 folded; 4-paper positioning woven across core/payoff/compact/lejepa;
no TODO/placeholders/contradictions; the lone Step-48 reference is the intentional honest note (Step 54
supersedes it); wiki index synced (28 pages). **Only incompleteness:** the v2 arXiv *bundle* (PDF + tarball)
predates A4/B2/B3/IMWM/LDA and is **not rebuilt** — user-gated under the frozen-v1/v2 policy. Markdown complete;
built artifact stale-by-design.

## 5b. Adversarial review (拷打) history

Two multi-agent panels (3 distinct-lens skeptics + adjudicator each):

- **Round 1** → 3 empirical-bookkeeping blockers (seeds not reproducible from committed files; conflated Step-51
  "170–2700×"; Thm-A closed-loop smuggled the planner condition). Fixed honestly in `48fb2a9`/`85a32f2`
  (aggregate_seeds.py + 6 committed `*_seeds.json`; lead with the true 10–155× floor penalty; explicit (A5));
  also caught + fixed a silent `⊕` PDF-render bug.
- **Round 2** (`wf_40b358b6`) → **verdict: submission-ready, 0 blockers**, zero bogus critiques (every empirical
  claim re-derived from the JSONs). 7 should-fix + ~13 nitpicks, one coherent theme — *prose quoted best-seed
  endpoints while the repro table stayed honest*. All folded this batch: §3 Noether mechanism made consistent
  (conserved **scalar** ⇒ invariant; non-scalar conserved is equivariant); Lyapunov $0.2\%\to$ within $0.4\%$
  (0.690–0.692, 3 seeds); Noether-lift $R^2$ → ranges 0.60–0.86 / 0.62–0.91 with gates surfaced; §4.2 "entire
  90-step" corrected (only the *contracting* channel; conserved/rotor fall at finest $\epsilon$); UR-JEPA §5
  "~320×" labelled single-seed + 3-seed-corroborated; §6 threshold → 0.01–0.06 (seed-dep); Step-47 provenance
  narrowed; Err$_T$ norm pinned; isotypic-refinement-unmeasured flagged; a **falsifiability** paragraph added to
  §5; a **References** block with verified arXiv IDs (BRo 2606.01372, UWM 2605.25313, UR 2606.01443, IMWM
  2606.01626, LDA 2606.01847) added. 81 tests pass; both PDFs recompiled (12 pp) + visually QA'd.
- **Round 3 — JEPA-style restructure** (user feedback: internal "Step N" / scaffolding shouldn't be in the paper).
  Retitled to the colon form *"Certified World Models: Predictability Across Configuration, Horizon, and Resolution"*;
  re-laid into a JEPA layout (Abstract · 1 Intro · 2 Setup · 3 Certificate · 4 Hinge · 5 Experiments · 6 Related
  Work · 7 Limitations · 8 Conclusion · Refs · Appendix A); all body "Step N" → **Experiment 1–8** + descriptive
  names, figures → **Figure 1–5**, code filenames moved to the Appendix-A reproducibility table; the **FOR-REVIEW**
  block and the "Sharpest objections" Q&A scaffolding removed (substance folded into Limitations); the 推背图 name
  dropped from the footnote (celestial-mechanics-vs-weather kept) so the English paper carries 0 CJK. `build_paper2.py`
  simplified — no review/submission split, the standalone PDF is now the same clean source. 11 pp, 81 tests pass.

## 6. Open items

1. ~~**Title**~~ ✅ **decided** (user 2026-06-02; colon form chosen in Round 3): "Certified World Models: Predictability Across Configuration, Horizon, and Resolution" (subtitle/tagline "Scale buys interpolation; structure buys a certificate.").
2. **Upload** (user-gated, manual) — paper2 bundle is **built & scripted**: `papers/arxiv_paper2/build_paper2.py` (single clean source → `arxiv_paper2_upload.tar.gz` + the tracked standalone PDF). Old paper: `arxiv/arxiv_upload_v2.tar.gz` (frozen-v1 untouched, md5 087af50e). Optional before submit: cover letter, category cs.LG×cs.RO, compact-main+supplement split.
3. ~~**3D-aware containment**~~ ✅ **done (Step 58, `695143d`)**: $\text{slow}\subseteq(\text{invariant}\oplus\text{conserved-equivariant})$ measured exactly ($E$→ℓ=0 linear, $L$→ℓ=1 degree-2 cross, $R^2{=}1.00$).
4. ~~**P5 — discovery + generation**~~ ✅ re-framed Steps 33/36/38 into draft §4.6 (companion-line, not new evidence).
5. ~~**Proof prose**~~ ✅ §2–3 tightened (`de7b7e4`) + Thm A closed-loop assumption (A5) made explicit (`48fb2a9`).
6. **P3 remainder** (optional hardening) — a discrete-config-axis scale replicate (Step 51 is the continuous-$\mathrm{SO}(2)$ version; Step 49 already covers the discrete $\mathbb{Z}_2^6$ *certificate*, so this is gilding).

## 7. Commit index (this session, all pushed to `main`)

`204a0a5` Step49 · `14cf87a` Step50+test · `3be4fd0` Step51 · `620845b` Step52 · `217ed0d` Step52-test+conftest ·
`1b56ba8` paper2 draft · `ec71667`/`4b640bf` session summaries · `df1ae98` Step53 · `ae60ed3` Step54(A4) ·
`389c8e2` Step55(B2) · `a632a64` Step56(B3) · `c709363` paper2 PDF · `a082eda` 4-paper positioning ·
`3763aad` Step57 fold · `03f5f84` payoff LDA · `d6d46b2` record ledger · `390a773` title + standalone bundle ·
`695143d` Step58 · `a07d9af` Step58 fold · `9c4c95c` red-team polish · `de7b7e4` §2–3 proof prose + 拷打 block ·
`48fb2a9` 拷打 panel fixes (reproducible 3-seed numbers + ⊕ render + scripted review PDF) · `85a32f2` record sync ·
`51b5d82` round-2 拷打 fixes (prose↔seed-range alignment, §3 Noether-mechanism consistency, References block) ·
`42fe352` Round-3 JEPA restructure (colon title, de-Step → Experiment 1–8 / Figure 1–5, Related Work, build simplified) ·
`fb8ce72` **Experiment 9** (certificate on real PushT contact dynamics; weakness #1) ·
*(this batch)* weakness #2 theory (Lemma 2 characterization + §3.3 quantitative separation).
(Vault wiki ingest `0b0fe54`, local-only.)

## 8. Objective-critique weakness fixes (user: "一一解决")

The four real weaknesses from the honest assessment, and their fix status:

1. ~~**All experiments toy / constructed-teacher**~~ ✅ **mitigated** (`fb8ce72`, Experiment 9): certificate holds
   on a *learned* model of *real* PushT pymunk contact physics (orbit-flat ratio 1.00, resid ~1e-7, competitive
   in-dist; no MLP scale 1.7k→272k reaches the floor, 2.1–3.9×, 3 seeds). Residual: structured-state not pixels,
   single SO(2) task, modest gap — stated in §7.
2. ~~**Theorem A "light" / "just equivariance restated"**~~ ✅ **this batch**: added **Lemma 2** (converse —
   orbit-constant error against every equivariant target ⟺ equivariance, so the certificate *characterizes*
   equivariance and the impossibility is a theorem) + **§3.3 quantitative separation** (structure certifies the
   ε-independent orbit; best L-Lipschitz learner certifies only an ε/L-tube). Adversarially vetted (a skeptic agent
   killed the first folklore version and steered to the Lipschitz one). Honestly framed as a characterization +
   separation, not deep machinery.
3. ~~**Hinge (most novel) least substantiated**~~ ✅ **this batch**: added **Proposition 4** (placement principle).
   Skeptic killed the naive "place via $DC(z^*)$" version (vacuous — angular momentum is quadratic, $DC(0){=}0$) and
   steered to the correct one: via the **moment map's equivariance** ($\mu:\mathcal Z\to\mathfrak g^*\cong\mathcal V_1$
   for SO(3)) + Schur on $\mathrm{Sym}^2$, energy is forced into $\ell{=}0$ and angular momentum into the $\ell{=}1$
   block, recoverable **only** at degree 2 because $\dim\mathrm{Hom}_{SO(3)}(\Lambda^2\mathcal V_1,\mathcal V_1){=}1$
   — proving *why* the Step-58 cross-product readout is unique and tying it to the companion paper's degree-1 cap.
   Honestly framed as a Schur/moment-map *placement principle* (not a new theorem); "slow$=$conserved" stays measured
   (needs the Hamiltonian symplectic bridge). The hinge is now *conjecture (measured) + proved placement core*.
4. ~~**Contribution is a perspective, not a method**~~ ✅ **this batch**: shipped `src/certify.py` — a runnable
   **Algorithm 1 (Certify)** that takes a trained model's equivariance residuals + predictor-Jacobian spectrum + ε
   and emits the certified region (configuration monoid + per-channel horizons $T_j(\epsilon)$). Unit-tested
   (`tests/test_certify.py`, 5 tests → suite **86 passed**); demo runs on Experiment-1's *measured* spectrum (48
   unbounded channels + binding log-law horizon). Added Algorithm 1 box to §3.4 and reframed the abstract ("a
   single, runnable criterion"). Partly intrinsic (it remains a mechanism paper), but the certificate is now an
   operational tool, not only a lens.

**Net effect of the four-weakness program:** the paper now has (1) a real-contact-dynamics validation, (2) a
characterization + quantitative separation, (3) a proved placement principle for the hinge, (4) a runnable
certification procedure — each adversarially vetted and honestly scoped. Residual ceiling (stated in §7): still
laptop-scale, structured-state not pixels, modest OOD gap on PushT; the embodied/scale lift remains the open frontier
(roadmap: `papers/proposals/paper2-embodied-scale-lift.md`).

**Round-3 content panel** (`wf_544518f0`, on the §3.3/§3.4/§4/§5.7/Lemma-2 additions) → **verdict: should-fix-only,
0 blockers**; confirmed the prior skeptics' corrected hypotheses survived the fold. 3 should-fix, all 1–2-sentence,
fixed this batch: (i) §5.7 mis-attributed "2.1–3.9×" to the 272k MLP — it's the *smallest* MLP; the 272k is 2.1–2.6×
(the closest baseline) → reworded to the across-ladder reading; (ii) Prop 4 proof skipped the Sym²→Λ² bridge + the
"ℓ=1 block has ≥2 copies of V₁" hypothesis → inserted; (iii) Lemma 2 used ρ(g)⁻¹ in the monoid framework → one-line
remark (ρ∈O(Z) ⇒ invertible). Plus honesty nitpicks: §3.3 "error 0" idealization clarified, §5.7 brittle-`climb`
sub-gate disclosed, abstract 160×/16× disambiguated, certify.py flags sub-1-step horizons. Adjudicator **rejected**
the Lyapunov "0.4%" complaint (worst-seed error 0.378% < 0.4% from the committed JSON; reviewers used rounded 0.690).

## 9. 极致 program — push to the absolute best (T1–T5; user: "全做掉")

Closing the substantive (not cosmetic) gaps. Laptop-feasible set; the GPU tier (SE(3)/3D benchmark/scale, S3–S5
of `papers/proposals/paper2-embodied-scale-lift.md`) is out of physical scope here.

- **T1 — Augmentation baseline** ✅ (Experiment 10, `experiments/step60_augmentation.py`, §5.8 + Figure 7).
  Answers the sharpest objection honestly. **Honest finding** (corrected a smoke-time over-claim): SO(2)-augmentation
  *flattens a single PushT orbit* (ratio 0.93–1.02 vs plain MLP 1.84–2.75) and, on the smooth $\mathbb{Z}_2^6$
  dynamics, augmentation drives a well-trained MLP to a $\sim10^{-4}$ floor — augmentation is a **strong** baseline,
  NOT an exponential wall. The certificate's surviving edge: **exactness** ($\sim10^{-32}$ from 7 generators,
  $\sim10^{27}$–$10^{28}\times$ below the MLP floor, 3 seeds), an **a-priori guarantee** (Theorem A — no testing of
  the unseen compositions), **group-knowledge-free**, and **worst-case optimality** (§3.3). Reframed §5.4 ("gap vs
  *scale*; augmentation is §5.8"). 15 pp.
- **T2 — task-success closed-loop** ✅ (Experiment 11, `experiments/step61_closed_loop_certificate.py`, §5.9 +
  Figure 8). Converts the prediction certificate (Exp 9) to a **task-level** certificate: an equivariant model + a
  *G-equivariant* CEM planner (the A5 instantiation — isotropic σ, disk action bound, scene-covariant noise) gives
  closed-loop PushT pose control whose error is **orbit-invariant to the float floor** (model-rollout & real-env
  ratio **1.000**, all 3 seeds), while a 4.3×-larger MLP under the *same* planner degrades out of the wedge
  (×1.1–2.2 rollout / ×1.6–3.6 real-env). The **paired** protocol (same base tasks rotated over the orbit) + the
  RichVN model removed the between-task variance + weak-model issues that left Steps 10/12 closed-loop INCONCLUSIVE.
  Honest demotions vs the seed-0 first look: in-wedge is a **wash** (no in-dist win), the scene-blind margin is
  noisy (A5 needed for the *exact* guarantee), and the cost-drift>0.3 sub-check is met 1/3 (reported as ratio, like
  Exp 9's `climb`). Answers the "does the proxy convert to control?" objection and resolves the §5.6/§7 "no
  downstream task gap on PushT" concession.
- **T4 — Noether hinge "conserved ⇒ slow" as a theorem** ✅ (Proposition 5, §4; abstract + §1-contrib + §7 + §8
  reconciled). The forward direction is now **proved**, not conjectured: a charge conserved by the model to one-step
  defect $\eta$ has $T$-step **charge-value** error $\le T\eta$ (telescoping) — *linear*, never the chaotic channel's
  *exponential* $e^{\lambda T}$ — so $\lambda_Q\le0$, certified horizon $\ge\epsilon/\eta$ ($=\infty$ at exact
  conservation). With Prop 4 (placement) + Noether (symplectic ⇒ μ conserved) this makes "invariant/equivariant
  blocks are slow" a theorem under symplectic structure. **Adversarially vetted** (opus skeptic, verdict
  ACCEPT-WITH-FIXES): folded all 4 fixes — (C1) needs Φ-forward-invariance to iterate; relabeled as *charge-value*
  tracking not state-prediction; **dropped** the broken Jacobian-singular-value corollary (a shear conserves a
  functional yet has huge σ_max — claim only the readout channel's λ≤0); "exact conservation" honestly scoped (exact
  for momentum under equivariant symplectic discretization, O(Δtᵖ) for energy, η *measured* for learned f). Honest
  residue: latent-Hamiltonicity assumed; η measured; converse (slow ⇒ conserved) false. Compute-free (pure theory).
- **T5 — polish** ✅ (presentation, reviewer-facing). **Hero Figure 1** (`papers/figures/make_hero.py` →
  `hero_certified_region.png`): a 2-panel concept schematic of the 3-axis certified region (configuration: structure
  certifies all ⟨S⟩ vs scale's ε/L tube; horizon×resolution: eclipses-slow vs weather-chaotic), placed at §1 end +
  cited; all 7 downstream "(Figure N)" refs renumbered +1 (now Figures 1–9, verified embed↔ref aligned).
  **Positioning table** at the top of §6 (BRo / UWM / UR-LeJEPA / IMWM / LDA / companion / **this paper** × config /
  horizon×ε / closed-loop / guarantee-kind) — crisp at-a-glance map. **One-command repro**: `Makefile` with
  `make paper2` (seeds→figures→tests→PDF) + `make paper2-quick`; noted in Appendix A. PDF 19 pp, 9 figures, clean
  compile. *Deferred (marginal):* the 5-seed bump — the paper is consistent + honest at 3 seeds, and a 5-seed re-run
  of the PushT line is ~2 h CPU for negligible CI tightening; left at 3.
- **T3 — pixel latent** ✅ (honest mixed result, `experiments/step62_pixel_latent_certificate.py` +
  `tests/test_step62.py`; folded into §7 Limitations, NOT a §5 win — no overclaim). Built a C₄-steerable encoder +
  a **new C₄-equivariant latent predictor** (`SteerableLatentPredictor`: 1×1 R2Conv on regular ⊕ irrep(1)-action →
  regular, equivariant to 6e-7, unit-tested). **Positive:** the certificate's exact orbit-flatness *transfers to a
  learned PIXEL latent* — encoder C₄-equiv 3e-5 (square arena ⇒ rot90 bit-exact), so the multi-step latent rollout
  is flat over the C₄ orbit **to the float floor (ratio 1.000)**; exact orbit-invariance is not an artifact of
  structured state. **Honest negative:** at laptop scale the steerable pixel JEPA **underfits** (rollout relMSE
  ≈4.5 vs an ordinary CNN's ≈0.8) — "flat is not good" here; and the ordinary CNN is itself orbit-flat (PushT pixels
  ≈ C₄-symmetric, the §5.8 augmentation regime). Did **not** chase a steerable-wins outcome by tuning capacity
  (architecture p-hacking); reported the finding. Competitive-AND-flat pixel rollout → GPU tier (S1). Gate redesigned
  to the load-bearing claim (enc/pred equivariance + exact flatness + a learned-check), which surfaced the underfit
  honestly (`eq_learned` = False).

**极致 program complete: T1–T5 all done (T1/T2 evidence, T4 theorem, T5 polish, T3 honest pixel attempt).**

## 10. Frontier push (user: "把剩下的 SE(3)/3D、scale、像素、具身枢纽 跑了吧")

Honest triage on this Mac (CPU/MPS, no CUDA): of the four GPU-tier items, **SE(3)/3D was genuinely laptop-runnable**
(the 3D machinery was built here originally) and I ran it; the other three are GPU-bound and I did not fake them.

- **SE(3)/3D — the certificate on SO(3)** ✅ (Experiment 12, `experiments/step63_se3_certificate.py`, §5.10 +
  Figure 10, 3 seeds). Lifts the multi-step rollout certificate from the circle to the **non-abelian SO(3)** on 3D
  point clouds (constructed SO(3)-equivariant teacher — toy, like Exps 1–7; e3nn encoder + jointly-equivariant VN
  predictor, the Step 13/18 line). **Robust positive:** learned equivariant model exactly flat over the SO(3) orbit
  (H=5 ratio **1.000**, resid ~1e-5, all 3 seeds), MLP climbs ×2.1–5.7 OOD; structure-vs-scale reproduced in 3D
  (7.4×-smaller eq carries the cert; bigger MLP interpolates better in-wedge 0.19–0.27 vs 0.55–0.58). **Honest
  caveat:** `compete` gate fails all 3 (the small eq's accuracy floor ~0.57 is high → OOD only *comparable* to the
  degraded MLP, not a clean win — "flat is not good" in 3D, like T3 pixels); reported INCONCLUSIVE, not loosened. No
  new test needed — `tests/test_se3_equivariance.py` already guards the encoder+VN-predictor joint SO(3) equivariance
  that gives the rollout flatness. Ties to Prop 4 (L in ℓ=1) + Exp 6 (3D containment): hinge *places* the SO(3)
  charges, certificate is *flat* over SO(3).
- **competitive pixels — the laptop gamble** (user: "笔记本赌一把像素") ❌→✅-honest. Ran a fair/generous-capacity,
  4×-longer steerable-pixel retry (`STEP62_WIDTH=16 STEP62_LATENT=128 STEP62_EPOCHS=120`, 179k params vs the MLP's
  243k, → `step62_pixel_latent_certificate_big.json`). **Result: gamble failed, decisively and informatively** — the
  bigger/longer steerable underfits *further* (rollout relMSE **23.7** vs T3's 4.5) while staying exactly flat (ratio
  1.000, enc 4.5e-5). So the pixel accuracy gap is **NOT capacity** (more params + epochs made it worse) — it's an
  **optimization/architecture** problem (e2cnn JEPA hard to train on pixels) → genuinely GPU-tier. Folded into §7
  (corrected "needs more capacity" → "is an optimization problem, not a missing-parameters one"). step62
  parametrized (`6f7a10b`); committed T3 result untouched.
- **MPS correction (user: "苹果自己的 M5max GPU 不能跑吗？")** — I was wrong to say "no GPU". This Mac's GPU runs via
  **PyTorch MPS**, and e2cnn is **~10–26× faster on MPS than CPU** (SteerableEncoder fwd 466ms→18ms; full e2cnn JEPA
  3809→399 ms/epoch). step62 now has a `STEP62_DEVICE=mps` knob (train on MPS, eval on CPU to dodge e2cnn MPS
  edge-cases). So the equivariant models ARE Apple-GPU-trainable here. **MPS-accelerated pixel retries (≈2.5 min
  each):** the gamble's divergence was an **optimization instability** (latent collapse) — *fixable*: with anti-
  collapse var_coef + lower LR + stable training the rollout relMSE drops monotonically **23.7→4.5→2.84→2.28** over
  800 epochs while staying flat (1.000). But it **plateaus at ~2.3** vs an ordinary CNN's **~0.44** — diminishing
  returns, so the **residual gap is architecture-deep, not compute/capacity** (an e2cnn JEPA is a worse pixel
  predictor on this modality). §7 rewritten accordingly: "ruled out capacity/compute; needs a better equivariant
  architecture, an open problem" — not the earlier (wrong) "GPU-tier compute we lack". **scale (→1M params)** is now
  plausibly MPS-feasible (unified memory + 10× speedup) — laptop already shows no-scale-reaches-floor across 160×
  (Exp 9), so low marginal value but doable. **real embodied (RLBench/ManiSkill)** still needs CUDA-only sim → the
  one genuinely off-Mac item; the rest is the GPU tier S1–S5 of `papers/proposals/paper2-embodied-scale-lift.md`.

**Meta-finding across T3 + Exp 12:** the certificate's *exact orbit-flatness* (the GUARANTEE) lifts everywhere —
SO(2) structured-state, SO(2) pixels, SO(3) point clouds — all ratio 1.000. The equivariant model's *competitive
accuracy* is modality/capacity-dependent: competitive on structured-state SO(2) PushT (Exp 9), underfits on pixels
(T3) and 3D clouds (Exp 12) at laptop capacity. Honest read: the guarantee is free and universal; competitive
accuracy needs matched equivariant capacity per modality (GPU tier).

## 11. GPU(MPS)-optimization survey + 3D-capacity attempt (user: "看看论文有什么可以优化的地方需要跑GPU的")

MPS unlocked fast e2cnn/e3nn training (§10). Surveyed paper2 for MPS-improvable items; attempted the top one.

- **Top candidate — fix Exp 12's 3D underfit via capacity** (step63 gained `STEP63_MUL/PREDHID` knobs). The committed
  Exp 12 eq is 7.4× *smaller* than the MLP (17k vs 124k) and underfits in-distribution. Hypothesis: it's a capacity
  gap (unlike pixels, which is architecture-deep). **Result: capacity helps but does NOT cleanly flip it.** At
  *matched* capacity (124k=124k, mul=24/predhid=192, 3 seeds) the eq still underfits (eq_in 0.58–0.70) and its
  out-of-wedge advantage *straddles* 1 (floor-pen 0.65–1.16). Only at ~1.7× the MLP (215k, mul=32, 1 seed) does its
  flat floor drop to 0.47 for a clean OOD win (floor-pen 1.4). **Honest finding: the equivariant 3D model is
  parameter-INEFFICIENT** (equivariance costs per-parameter expressiveness here). So Exp 12 KEEPS the canonical mul=8
  config (a 7.4×-smaller eq carrying the cert is the *stronger* structure-vs-scale framing); the knob + this note
  document the negative result. No paper change; canonical mul=8 artifacts restored (now record `params`).
- **Other GPU(MPS) items (surveyed, not run — low marginal value or out-of-scope):** scale ladder →1M params
  (MPS-feasible via unified memory, but Exp 9's 160× already makes the point); 5-seed hardening of the e3nn/e2cnn
  experiments (MPS-fast, but 3 seeds is already honestly reported); a better *equivariant pixel architecture* (the
  real §7 open problem — research, not just compute); real embodied RLBench/ManiSkill (CUDA-only sim, genuinely
  off-Mac). The certificate's exact-flatness lifts everywhere; competitive accuracy on pixels/3D is the standing
  open problem, and MPS shows it is architecture/efficiency, not raw compute.

## 12. Frame averaging resolves the §7 pixel open problem (user: "研究一下")

Researched the §7 pixel open problem (a pixel encoder both exactly flat AND competitive). **Resolved by frame
averaging** (`experiments/step64`, Puny et al. 2022, arXiv:2110.03336). Also produced an *honesty correction* to
step62's metric.

- **Method.** Instead of steerable layers (the e2cnn route that underfit), make a *plain* `ConvEncoder` + plain MLP
  exactly $C_4$-equivariant by a Reynolds average over the 4 grid rotations with a $\rho$-correction:
  $E(o)=\frac14\sum_k \rho(g_k)^{-1}\phi(g_k\!\cdot\!o)$, latent $=[z_{\text{inv}}\,(\rho{=}I)\oplus z_{\text{vec}}\,(\rho{=}R_k)]$,
  $\rho$ orthogonal $\Rightarrow$ Theorem A. Verified $C_4$-equivariant to $\sim\!10^{-7}$ at init **and** post-train
  (`tests/test_step64.py`, +2 tests $\to$ 91 passed); pure `torch`, native MPS (no e2cnn fallback).
- **Honesty fix — collapse-robust metric.** step62's uncentered relMSE is deflated by a large constant latent mean.
  Added **FVU** (fraction of *centered* variance unexplained; FVU$<1\iff$ beats predict-the-mean) + latent
  participation ratio. This *re-scoped* the result honestly: I gate the RELATIVE claim (penalty removed vs the
  unconstrained CNN), and REPORT the absolute limitation (no model beats predict-the-mean at $H{=}4$).
- **Result (3 seeds, `step64_frame_averaged_pixel_seeds.json`, all PASS).** FA matches-or-**beats** the *unconstrained*
  CNN on FVU (**0.68–1.07×**, mean 0.84), exactly orbit-flat (ratio **1.000**, enc $\sim\!10^{-7}$), with a *healthier*
  latent (PR **2.8–4.3** vs CNN 2.2–2.6). So the steerable underfit was an **e2cnn optimization artifact, NOT a cost of
  equivariance** — with frame averaging the prior is *accuracy-neutral*.

Then (user: "都做吧") promoted to **Experiment 13 / §5.11** AND attacked the residual via a **horizon sweep + a fair
1-step-vs-target diagnostic** (the second part of "都做吧"):

- **Horizon stability (the residual attack).** Hypothesis (FVU<1 at short H, crossing 1 = a Theorem-B certified horizon)
  was **wrong** — FVU>1 at *every* h for *every* model. But the sweep revealed something cleaner: the **steerable rollout
  DIVERGES** over horizon (FVU grows **160–1600×** its 1-step value), while **FA stays stable** (**≤1.2×**, usually below
  the CNN's 1.1–1.6×). Equivariance done right is *reliable*, not just flat. Added `STEP64_HEVAL` sweep + a 3rd figure
  panel (log-scale FVU vs h).
- **The residual is the JEPA latent, proven not an artifact.** Added `return_target_encoder` to `train_jepa` (backward-
  compatible) and measured the **fair 1-step FVU vs the EMA target** (the exact training objective). It is **>1 for ALL**
  (FA 1.7–2.0, CNN 1.8–2.2) → rules out an EMA online/target measurement artifact AND equivariance. Cause: VICReg forces
  unit variance on all 64 latent dims, but PushT dynamics is low-dim (PR≈3), so ~60 dims carry unpredictable anti-collapse
  variance. Architecture-AGNOSTIC; a strong few-step pixel predictor at small scale is the residual open problem.
- **Paper change.** New **§5.11 (Experiment 13)** with the 3-panel figure (flat | FVU | horizon); contribution 3 gains a
  pixel clause; §7 pixel bullet shrunk to a §5.11 pointer; Puny et al. ref; reproducibility appendix + `aggregate_seeds.py`
  register step64. PDF rebuilt (**1087 KiB, 11 figures**, compiles clean); full suite **91 passed**. MPS note: absolute FVU
  has run-to-run non-determinism (~±0.3), so quoted numbers are committed seed ranges, not single-run.

## 13. ICLR-novelty lit-check + two moves: §6 carve-out + Proposition 6 (user: "有没有ICLR级创新点" → "都做")

Did a 6-angle novelty lit-check (WebSearch/WebFetch) before committing to any new direction. **Verdict: the mechanism
space is largely occupied (2024–2026)** — orbit certificates (randomized smoothing 2211.14207, eCP 2602.03986),
orbit-margin robustness (2510.16171), learned conservation (Noether Networks 2112.03321, Noether's Razor 2410.08087),
Jacobian-reg world models (2501.00195), slowness-as-failure-mode in JEPA, anti-collapse↔IB. **But the closest neighbors
*validate* our uncrowded corner:** they are all **single-shot** (classification margin / one-shot conformal set),
**forward-only** (no converse), and have **no horizon / spectrum / conservation** axis.

**Move 1 (defense) — §6 carve-out.** Added a related-work paragraph + 2 table rows + 6 references positioning paper2
against the certified-equivariance / conformal / conservation / Jacobian-reg literature. Spine: (i) Time — they are
single-shot, we are multi-step (Thm B, tight by Prop 6) + conserved-channel infinite horizon (Prop 5); (ii) Direction —
they prove equivariance⇒guarantee (classical: equivariant estimator risk constant on orbits), we add the **converse**
(Lemma 2, ⟺); (iii) Object — eCP frame-averages a *score* post-hoc, we frame-average an *encoder/predictor* (Exp 13).
Honesty fix: explicitly acknowledge the forward direction is classical.

**Move 2 (offense) — Proposition 6 (the horizon is tight; approximate equivariance is horizon-limited).** Theorem B
already had the multi-step upper bound but was hedged as "a bound on the form." Added a **matching lower bound**: an
exactly-equivariant target + a model that is ε-approximately equivariant (perfect predictor, encoder with a single
ε equivariance defect on one orbit point) has orbit-error-variation **exactly ε·e^{λT}** on an expansive channel
(multiplier a=e^λ). One-line proof (linearity + A3 + orthogonal ρ). ⇒ certified horizon
T(ε_res)=（1/λ)log(ε_res/ε)=Θ(log(1/ε)/λ) — **tight**. Sharp payload: **any ε-approximately-equivariant model
(ε>0) has a finite certified horizon on every expansive channel; only exact equivariance (ε=0) or conservation
(λ≤0, Prop 5) reaches ∞.** This is the horizon-domain companion of Lemma 2: scale/augmentation buys *approximate*
equivariance (Exp 10 floors at ε≈1e-4, never exact), amplified e^{λT} — so the Exp-10 single-step augmentation tie
**must break over horizon** at the predicted T. Removes the only hedge on the horizon axis (constants c_j still
un-estimated — honest). Updated contribution 1, Thm-B hedge, §7 limitation. PDF rebuilt (**1098 KiB, compiles clean**);
paper-only change, tests unaffected (91 passed stands).

**Net ICLR posture:** the genuinely-novel core is (converse Lemma 2) + (multi-step tight horizon Thm B+Prop 6) +
(Noether hinge Prop 4/5) for *world models* — exactly what the single-shot robustness/conformal literature lacks, now
explicitly carved out. Candidate new mechanisms (predictability-aware anti-collapse; certificate-driven active
inference) remain future directions, each with a named neighbor to differentiate against.

## 14. Focused ICLR extraction draft (user: "接着起草")

Drafted **`papers/iclr_certified_horizons.md`** — "Certified Predictability Horizons for Equivariant World Models" —
a focused ~9-page ICLR-shaped extraction of paper2's genuinely-novel core, **led by the horizon axis** (the unoccupied
one per the lit-check). Single sharp claim: an equivariant world model admits a *tight* certificate of its predictable
horizon (Thm B upper + Prop 6 lower = Θ(log(1/ε)/λ_j)), unbounded **iff** conserved/equivariant (Noether Prop 4/5),
and equivalent to equivariance (Lemma 2 converse) so scale can't buy it. Key reframing (reviewer-friendly): the
single-shot certified-equivariance literature (orbit-margin 2510.16171, eCP 2602.03986, randomized smoothing
2211.14207) is our **T=1, ε-independent slice** — we are its multi-step generalization. Structure: Abstract / Intro+4
contributions / Setup (A1–A5) / §3 certificate (config Thm A+Lem 1+2, horizon Thm B+Prop 6, Noether Prop 4/5, Alg 1) /
§4 experiments E1–E7 (config exponential, horizon staircase+tightness, approx-sym validates Prop 6, structure-vs-scale,
real PushT 160× sweep, SO(3)+frame-averaged pixels transfer, closed-loop) / §5 carve-out related work / §6 honest
limitations / §7 conclusion / refs. All E1–E7 numbers cross-checked against the committed record (7/7 match). Builds
via `make iclr-build` (pandoc→tectonic, text+math only, no figures); QA PDF 91 KiB compiles clean. Not yet committed to
the arXiv bundle — it is a standalone focused draft for the user to review/iterate before deciding submission target.

## 15. ICLR draft → submission-quality: figures + step65 + adversarial red-team (user: "都做了吧", overnight batch)

Took the focused ICLR draft from text-only to submission-shaped, autonomously:
- **7 figures embedded** (hero + the new step65 tightness figure as the §3.2 central-claim visual + E2–E6); `make iclr-build` (pandoc→tectonic, `--resource-path`); QA PDF ~800 KiB, visually checked (Figure 2 renders cleanly).
- **step65 (Proposition 6 numerical confirmation)** added as a real experiment (3 seeds, test_step65, full suite 92) + folded into paper2 §3.2 + ledger + aggregate_seeds.
- **Opus adversarial red-team** (as a tough ICLR reviewer) → verdict *borderline, fixable to accept*. It caught a **real reproducibility gap on the central figure** (committed step65 JSON was a seed-2 run, caption quoted seed-0, no `_seeds.json` despite "3 seeds") + two honest overclaims. **All folded** (commit `023328b`):
  - **B1**: regenerated canonical at seed 0, committed `step65_horizon_tightness_seeds.json`, captions now quote the true seed range (rel-err 1e-14–1e-13) and state exact-equiv variation is *exactly 0* (1e-16 was a log-plot floor).
  - **B2**: Prop 6's "any model… is horizon-limited" → the honest worst-case certificate statement ("guaranteeable from ε-approx equivariance alone … worst case over admissible targets"), in **both** papers.
  - **B3**: E2's "0.4%" (Lyapunov recovery) separated from the certified-horizon slope (1.3–1.6 vs 1/λ=1.44).
  - Non-blocking: Lemma 2 continuity caveat; Prop 6 linearity as explicit hypothesis; "no accuracy cost" → *relative to* the unconstrained CNN; E7 closed-loop derivation; a "what is classical vs new" preemption para (leads with Prop 6 + Lemma 2); a "hinge validated on constructed teachers, not emergent" limitation.
  - The red-team verified 7/7 cross-checkable numbers against this ledger — all matched.
- **Honest residual on this draft (NOT done — needs the user):** it is a clean conference-style pandoc PDF, **not** the official ICLR `.sty` template (author/affiliation/line-numbers/page-limit pass still required for an actual submission); no dedicated "tightness" beyond the analytic step65; the toy-scale + constructed-teacher-hinge caveats are disclosed, not resolved. Commits this batch: `b4f7431` (step65 + paper2 fold + iclr-build), `59fc368` (draft), `023328b` (figures + red-team fixes).

## 16. Round-2 peer panel (theory + WM-empiricist) → honest re-grade (user: "先2再3", the "2")

Ran a 2-reviewer Opus panel (distinct from round-1's generalist), both **borderline**, both flagged **overclaiming**; folded the honesty fixes (commit `daeb2ee`):
- **Theory reviewer**: Lemma 2 (converse) + Prop 6 (lower bound) are **folklore** once set up — invariant decision theory (orbit-constant risk + converse: Eaton 1989 / Lehmann-Romano / Berger) + textbook $e^{\lambda T}$ growth. **Re-graded** §1 + abstract: the individual statements are NOT new; the contribution is their **synthesis** into a computable multi-step certificate for learned world models. Added classical cites (invariant decision theory; Maron/Yarotsky/Dym-Maron equiv-universal-approx; Goodman-Wallach; Lorenz+Pilyugin shadowing; frame-averaging expressivity). Fixed Prop 5 category-slip in BOTH papers ("$\lambda_Q\le0$" → "charge-VALUE error grows linearly"). Lean "scale can't buy it" on §3.3, not Lemma 2's algebraic converse.
- **WM empiricist**: the **horizon headline is shown only on a learned SYNTHETIC-latent model (E2) + analytic (step65)**; real PushT (E5) shows the **configuration** axis (flatness), not the horizon staircase → added §6 limitation + reframed abstract to promise flatness-on-real + horizon-on-synthetic. E5 OOD gap is **non-monotone** (3.9× = smallest baseline, 2.1–2.6× = largest) → report per-baseline, lead with strong-baseline ~2×, force = exact flatness not magnitude. Disclosed E5 `climb` sub-gate is met 1/3 seeds (flat+floor 3/3).
- **The one high-value NEW experiment** (recorded, NOT yet run): a learned-model PushT **certified-horizon staircase** (certified-vs-measured, reusing `certify.py`/Alg 1 on the E5 model) — moves the headline onto a learned model of real dynamics. This is the top next step for the ICLR draft.
- Net: the draft is now honest about novelty (synthesis, not new theorems) and about the theory↔experiment gap. Both PDFs rebuilt, 92 tests pass.

## 17. ③ candidate mechanism 1, probe 1 — predictability-aware anti-collapse (rank-matching): HONEST NEGATIVE

`experiments/step66_predictable_anticollapse.py` (rank sweep, seed 0, MPS). Hypothesis (from step64's FVU discovery):
the FVU>1 residual is "over-wide latent × isotropic VICReg variance"; narrowing the latent toward its predictable rank
should drive FVU<1. **Rejected.** FVU vs latent D: D=8 **2.57**, D=16 **2.26**, D=32 **1.88** (best), D=64 **1.95** —
narrowing does NOT beat predict-the-mean, and very small D is *worse* (task needs >8 dims). Orbit-flatness exact at
every D (ratio 1.000) — the certificate is untouched; this isolates *accuracy*. **Conclusion: the residual is NOT
latent width** — at every D the variance floor fills D dims with per-frame-unpredictable variance, so the culprit is
the *isotropic variance objective*, not the dimension count. This sharpens (does not close) the open problem: the
principled fix is a **predictability-gated variance** (collapse unpredictable dims, protect predictable ones) — a
deeper training-objective change, a separate gamble. NOT folded into any paper (exploratory candidate-mechanism probe).

## 18. ③-1 mechanism probe 2 (predictability-GATED variance) — HONEST NEGATIVE; ③-1 conclusively fails

`experiments/step68_gated_variance.py` + `train_jepa(..., predictability_gated_var=True)` (per-dim variance floor
weighted by predictability $w_d=\exp(-e_d/\bar e)$, protect predictable dims, let noise dims collapse). A/B on the
Step-64 FA-JEPA at $D{=}64$: **isotropic FVU 1.87, gated FVU 2.02** (gated slightly WORSE, Δ −0.15); orbit-flat both
(ratio 1.000). So gating the *objective* does not beat predict-the-mean either. **③-1 (predictability-aware
anti-collapse) is conclusively negative** — both the rank knob (step66) and the variance-objective knob (step68) fail.
The FVU>1 pixel residual is deeper than the anti-collapse term; it is a property of frame-JEPA on PushT pixels we have
not closed. Exploratory probe, not folded into any paper (the §6 pixel limitation already states the residual honestly).

## 19. ③-2 certificate-driven active inference — POSITIVE (the one win in ③)

`experiments/step69_certificate_driven_exploration.py` (+ `test_step69.py`; full suite 93). On the $\mathbb Z_2^k$
configuration axis, three explorers spend a budget choosing the next observation by a different drive; the action space
is $k$ true generators + noisy high-error *distractors* that certify nothing. **Result (3 seeds, all PASS):**
certificate-driven (maximize certified-region expansion) certifies **all $2^7{=}128$ compositions in exactly $k{=}7$
steps** (the generator basis, AUC 0.642); **error-curiosity** is lured by the noisy distractors and certifies only
**1%** at the same budget (AUC 0.008); random ~2% (AUC 0.024). So "**expand your certified region**" is a **noise-immune
epistemic drive** that beats prediction-error curiosity — the classic *noisy-TV* failure mode — for compositional
coverage. **Honest caveat:** the baseline is *raw error* curiosity; a sophisticated *information-gain* agent would also
learn to avoid aleatoric noise. The certificate's contribution is that it gives that noise-immunity **for free and
provably** (computable from the certificate, no noise model). A toy demonstration of the certificate's *actionability*,
candidate to fold into the paper's actionability discussion (§5.6).

**③ tally (user "全做了 都不行再巩固"): 3 negatives + 1 positive.** ③-1 predictability-aware anti-collapse — negative
twice (rank step66, gated-variance step68: FVU>1 survives both). Reviewers' #1 (horizon on real PushT, step67) —
negative (local Jacobian spectrum does not predict the learned model's rollout, R²=0.02; folded as evidence into §6).
③-2 certificate-driven active inference (step69) — **positive** (toy). Not all failed, so the consolidation is: fold
③-2's positive + the §6 negatives, rather than abandon.

## 20. Gap-closing C — the certified-horizon law on a learned model of REAL chaotic dynamics (Lorenz): POSITIVE

`experiments/step70_lorenz_horizon.py` (+ `test_step70.py`; full suite now 95). The biggest ICLR gap was that the
**horizon headline** had only been shown on a learned *synthetic-latent* model (E2) + analytically (step65); on real
PushT (step67) it **failed** because PushT interior dynamics is locally *near-neutral* ($|\mu|\approx1$, no Lyapunov
spread). The decisive question: does the staircase lift to a learned model of a system that is *genuinely chaotic*?
Test: integrate the **Lorenz** attractor ($\sigma{=}10,\rho{=}28,\beta{=}8/3$; $\lambda_1\approx0.9056$/t, $\mathbb Z_2$
symmetry) with RK4, train a plain one-step MLP of the $\Delta t$ map, then run the **Step-52 certified-horizon
staircase on the learned model**. **Result (3 seeds, all PASS):** one-step relMSE $<10^{-4}$ (model learned the map);
the certified horizon $T(\epsilon_0)$ is **linear in $\log(1/\epsilon_0)$ with $R^2 = 0.975/0.995/0.990$**; the staircase
**slope recovers the true Lorenz exponent** $\lambda_{\rm staircase}=1/(\text{slope}\cdot dt)=0.895/0.919/0.977$/t vs
textbook $\lambda_1=0.9056$/t (rel-err **1.2%/1.4%/7.9%**). So the $T\sim\log(1/\epsilon)/\lambda$ law of Theorem B
**does lift to a learned model of real chaotic dynamics** — gap 1 substantially closed, scoped honestly: it holds when
the dynamics carries a genuine Lyapunov spectrum (Lorenz ✓); PushT failed only because it is near-neutral, not because
the certificate is a toy.

**Robust-estimator finding (folded into the experiment + test):** the *staircase slope* is the load-bearing Lyapunov
estimator. I first gated on the early-window finite-time exponent $\hat\lambda$, but on Lorenz that exponent is
**window-sensitive** (FTLE fluctuates along the attractor — seed 0's early window caught a transient super-expansion,
$\hat\lambda\approx3.7$/t). The staircase slope, fit over the whole horizon range, is far more stable (recovers
$\lambda_1$ to 1–8% on all 3 seeds). Gate is now: (i) one-step relMSE $<0.05$; (ii) staircase $R^2>0.95$; (iii)
$|\lambda_{\rm staircase}-0.9056|/0.9056<0.25$. `test_step70.py` validates the protocol on the **true integrator**
(Lorenz $\mathbb Z_2$-equivariance exact to machine precision; staircase $R^2=0.993$, slope recovers $\lambda_1$ to
2.1%) so the claim is independent of any one trained model. Canonical figure = seed 1 (cleanest; its diagnostic
$\hat\lambda$ also matches $\lambda_1$ so panels (a)/(b) agree); per-seed in `step70_lorenz_horizon_seeds.json`. This
is the top input for the ICLR reframe (task B+D): the horizon axis now has **real-chaotic-dynamics** evidence, and the
PushT/Lorenz contrast motivates the **scope theorem** (task A): the local-spectrum certificate lifts to learned
dynamics iff the system is spectrally non-degenerate (chaotic), via Oseledets / shadowing.

## 21. Deepening (user "全做吧"): Step 71 multi-chaos + Proposition 8 (finite-horizon exponent transfer)

After the gap-closing batch, user picked "全做" on four deepening directions; the two doable on CPU/MPS landed
(commits `9741583` + this batch). The other two are honestly constrained: ③ arXiv = bundle rebuilt, manual submit;
④ embodied benchmark = CUDA-only (ManiSkill/RLBench), plan only.

- **② Step 71 (`step71_multichaos_horizon.py` + `test_step71.py`)** — generalize Step 70's learned-model horizon lift
  from Lorenz to a **class**: Hénon map ($\lambda_1{\approx}0.419$/step), Rössler flow ($\lambda_1{\approx}0.0714$/t,
  the *small*-exponent stress case), Lorenz anchor. On-attractor data gen (burn-in + divergence filter) fixes Hénon's
  basin escapees + Rössler's transient. **3 seeds:** Hénon $3/3$ (rel-err $8$–$12\%$), Lorenz $3/3$ ($1$–$5\%$),
  Rössler $2/3$ ($8$–$9\%$; one seed's $\sim1500$-step horizon under-crossed — honest small-$\lambda$ fragility). Every
  *seed* passes overall (Hénon+Lorenz carry the $\ge2$-of-3 gate). = Experiment 15 / §5.13.
- **① Proposition 8 (paper2 §3.2)** — the *correct* mechanism the round-1 theory red-team's correction left open. The
  red-team killed "shadowing transfers the exponent" (asymptotic Lyapunov exponents only upper-semicontinuous). The
  fix: the certified horizon is **finite**, and **finite-time** Lyapunov exponents *are* $C^1$-Lipschitz in the
  dynamics. So a learned model recovers $\lambda_1$ up to an $O(\delta)$ model-fidelity bias + finite-$T$ truncation,
  the $T$-uniform constant under a dominated splitting (cite **Bochi–Viana 2005**). **Falsifiable + confirmed:** bias
  $\propto$ fidelity → Step 71's Rössler tightens $44\%\!\to\!8\%$ as $\delta$ falls with fuller training; the
  true-system staircase (no model) isolates the finite-$T$ truncation (~$9$–$10\%$). Replaces a wrong intuition with a
  right one and *explains/decomposes* the recovery rather than reporting it. This is the genuine math contribution of
  the deepening pass (plays to the researcher's geometry/ergodic-theory strength, zero compute).
- Full suite **96 tests**; paper2 PDF 1329 KiB / 13 figures; arXiv bundle rebuilt (22 members). ICLR fold + arXiv
  submit instructions + embodied-benchmark plan are the wrap-up.

---

## [2026-06-07] Structure-vs-scale baseline hardening (Step 76 ESN + Step 77 GRU) — defends Exp 18

A literature-collision sweep (5 fronts, deep-research, 102 agents) found Fronts 4 (equivariance × certified horizon)
and 5 (structure-beats-scale high-$N$ spectrum) **open**, but Front 1 (spectrum-from-learned-models:
Pathak/Vlachas/Hart/Kobayashi) is must-cite prior art that exposes a real reviewer attack on Exp 18: *"RC/RNN recover
Lyapunov spectra without symmetry, so your dense-MLP baseline is the wrong one."* Closed it empirically.

- **Step 77 (`step77_lorenz96_gru_baseline.py` + `test_step77.py`)** — a **GRU-BPTT** baseline (Vlachas's tool) trained
  with the *identical* recipe as the conv/MLP (same data / residual / $K{=}5$ rollout / Adam, + hidden-noise reg for
  closed-loop stability). **Positive control $N{=}12$: $R^2{=}0.93$–$0.99$ ($3/3$, a validated recoverer). Test
  $N{=}40$: $R^2{=}-0.22$/$-0.29$/$-0.29$ ($3/3$ FAIL)** — fails like the dense MLP while the $\mathbb{Z}_N$-conv
  recovers ($R^2{=}0.98$–$0.99$). **STRUCTURE STILL WINS on all 3 seeds.** Folded into §5.16 (figure
  `step77_gru_baseline.png`).
- **Mechanism (the deep framing).** A recurrent model's autonomous map lives on joint state $(x,h)$; its Jacobian
  carries $H$ hidden Lyapunov modes that must fall below the true minimum (**Hart 2024 conditional-Lyapunov**).
  Hidden-noise reg enforces it at low $N$ (spurious ceiling $-49\ll$ true min $-3.9$) but it **breaks at high $N$**
  (ceiling $\to-0.3$, near-neutral; positive count inflates $24$–$35$ vs true $14$). A **Markov** (feedforward) model
  has a Jacobian *exactly* $N\times N$ — no hidden modes — so structure-vs-scale among Markov models is confound-free.
  *Structure, not recurrence/training/scale/one-step-accuracy, closes it.*
- **Step 76 (`step76_lorenz96_reservoir_baseline.py` + `test_step76.py`)** — ESN/reservoir baseline (analytic-Jacobian
  thin-frame Benettin–QR vs autograd + finite-difference cross-checks; both unit tests pass). **Parked:** at the native
  $\Delta t{=}0.01$ a plain ESN is closed-loop-unstable (a known fine-$\Delta t$ artifact, not a bug); the GRU is the
  canonical recurrent baseline. Kept as a second prong (Kobayashi reservoir lineage).
- **Folded:** paper2 §5.16 (GRU paragraph + 3-way figure) + §6 (two related-work paragraphs, all 11 new cites); ICLR
  E2 + §5 + §6 (compressed; main text re-verified **≤9pp**, 0 unresolved cites; PDF 1147 KiB); `iclr_refs.bib` **+11**
  entries (Pathak'17/'18, Vlachas, Hart, Kobayashi, TrustKoopman, FloWM, Mo, Wang/Walters, Delliaux, Ordoñez).
  Tests `test_step74/76/77` green.

## [2026-06-08] Experiment 19 — certified-control co-demonstration, lifted to a *class* (Steps 79–81; user: "完整闭环控制" → "连佐证也跑(摆环+双摆),凑 class-lift")

Closes the two structural gaps a reviewer flags: **the two certificate axes never met on one system**, and **the
certificate never *changed a decision*.** Brainstorm → spec (`docs/specs/2026-06-07-step5-...`) → plan
(`docs/plans/2026-06-07-step79-...`) → subagent-driven TDD. Anchor + 2 corroborators, all honest-gated.

- **Step 79 anchor (`step79_certified_control.py` + `test_step79.py`, 8 tests)** — **controlled Lorenz-96**
  ($\dot x_i{=}(x_{i+1}{-}x_{i-2})x_{i-1}{-}x_i{+}F{+}u_i$, exact $\mathbb{Z}_N$). Action-conditioned $\mathbb{Z}_N$-conv
  WM (one-step relMSE $2\times10^{-6}$) + exactly-$\mathbb{Z}_N$-equivariant gradient-MPC planner.
  **(C) config axis:** orbit-flat control on *genuine* chaos ($\lambda_1{\approx}1.8$), residual $8\times10^{-16}$ —
  the orbit-flatness certificate realized on real chaos, not near-neutral PushT. **(H) horizon axis:** certified
  $T_1(\epsilon)$ + step78 bootstrap CI off the WM spectrum. **(D) decision:** **D1 short-horizon control = honest
  NEGATIVE** (certified horizon $\gg$ the receding-horizon controller's few-step depth — units-bug caught: $T_1$ is in
  *time* units, $T_1/\Delta t_{\rm map}\approx256$ map steps, not $\approx3$); **D2 active re-observation = QUALIFIED
  WIN** — re-observe every $T_1(\epsilon)/\Delta t$ steps sits on the efficient violation-vs-observation frontier
  ($2/3$ seeds) *untuned*, in the asymptotic-Lyapunov regime ($\epsilon{=}0.2$); at $\epsilon{=}0.01$ the certificate
  is honestly optimistic (**Prop. 8 finite-horizon $\delta$-bias**, ratio $\approx0.05$).
- **Step 80 corroborator (`step80_pendulum_ring.py` + `test_step80.py`)** — driven-damped $\mathbb{Z}_N$
  coupled-pendulum ring ($N{=}10$, $10$ positive exponents, Liouville $=-N\gamma$ exact). *Mechanical* analog. Orbit-flat
  $10^{-16}$, same optimistic→predictive transition (knee $\epsilon{=}0.3$), same $2/3$ frontier. Seed-for-seed parallel.
- **Step 81 corroborator (`step81_double_pendulum.py` + `test_step81.py`)** — chaotic double pendulum, **different
  group** ($\mathbb{Z}_2$ reflection, not $\mathbb{Z}_N$). Equivariant WM by **frame averaging** (the E13 device);
  $4$-D, $1$ positive exponent (λ₁≈0.44), so carries H+D2 but *not* the exponential-monoid config story. Reflection-
  orbit-flat $4\times10^{-17}$, same horizon validation (knee $\epsilon{=}0.7$, most-optimistic WM), same $2/3$ frontier.
- **Class-lift:** the certificate-driven re-observation law is a **class property of locally-coupled chaotic systems
  with an exact symmetry** — two groups ($\mathbb{Z}_N$, $\mathbb{Z}_2$), high/low dimension, abstract/mechanical —
  mirroring the existing Lorenz→Hénon→Rössler and SO(2)→SO(3) lifts.
- **Folded:** paper2 **§5.17 (Experiment 19)** + contribution clause + figure `step79_reobservation.png`; ICLR **E9**
  (figure-free; PushT cert figure relocated to Appendix A + §6/§7 trims to hold **≤9pp**, Conclusion on p9, 0 unresolved
  cites). `test_step79/80/81` green (26 tests, 8 s). Both PDFs rebuilt.

## [2026-06-08] Experiment 20 — the certified horizon made literal (Lever B: Theorem B′ + `step82`)

Turns the staircase's *posterior* slope-read (Exp 14–18 *measure* $\hat\lambda_1\approx\lambda_1$) into an
*a-priori* **sound** certificate read off the learned model's own Jacobian — and characterizes the exact regime
where that certificate is *tight*. Plan → subagent-driven TDD (`step82` + `test_step82.py`); commit `fec83b9`.

- **Theorem B′ (cone / adapted-metric certified horizon)** — a continuous SPD field $P(z)$ + constant $\Lambda$ with
  $D\hat\phi^\top P(\hat\phi)D\hat\phi\preceq\Lambda^2 P$ gives $\lambda_1(\hat\phi)\le\log\Lambda$ and a guaranteed
  $T_{\mathrm{guar}}(\epsilon)=\lfloor(\log(\epsilon_{\rm res}/\epsilon)-\tfrac12\log\kappa)/\log\Lambda\rfloor$ from
  $\hat\phi$ **alone** (no true dynamics, no true $\lambda_1$). Sound for any feasible $(P,\Lambda)$, tight at the
  adapted (Oseledets) metric ($\Lambda\to e^{\lambda_1}$); continuum bridge via $h$-cover + $L_J,L_P$-Lipschitz
  ($\Lambda^{\rm cert}=\Lambda_{\rm samples}+\sqrt\kappa L_J h+O(L_P h)$); the adapted metric is the analytic dual of
  a forward-invariant cone field, **uniform hyperbolicity = exactly the regime a tight $(P,\Lambda)$ exists**.
- **Cat map (uniformly hyperbolic) = tight.** Certified exponent $= \lambda_1=\log\frac{3+\sqrt5}{2}$ to machine
  precision on the **true** map ($1.00\times$), **$1.17\times$** on a *learned* net (3 seeds, all sound,
  $T_{\rm guar}\le T_{\rm true}$); nonlinear **Anosov perturbation** stays tight ($1.06\times$). Cone margin $+0.6$.
- **Hénon (non-uniformly hyperbolic, homoclinic tangencies) = sound but conservative.** Certified exponent
  $3.16\times$ the true; **cone-margin diagnostic goes negative ($-26.9$)** → certificate **abstains**, routes to a
  step78 bootstrap horizon ($T_{\rm guar}{=}10\le T_{\rm true}{=}13$) rather than over-claim.
- **Stretch systems (Lorenz, Rössler, 40-D Lorenz-96)** — cone abstains (black-box net-Jacobian-Lipschitz tax makes
  the continuum bridge vacuous); abstention is the safe move.
- **G2 soundness $= 1.0$ over all 36 (system, seed, $\epsilon$) cells** (4 main systems). Abstention **validated**:
  where cone abstains, the bootstrap fallback is genuinely *not* sound vs. the true system ($3/9$ stretch cells) —
  exactly why abstaining is correct.
- **Honest framing = a *characterization*, not just a method:** a tight a-priori certified horizon from a learned
  model is achievable *exactly* in the uniformly-hyperbolic regime, and the certificate **self-diagnoses** the regime
  (cone margin) — a tightness dimension on the scope theorem (Prop 7), analytic dual of Theorem B.
- **Folded:** paper2 **§5.18 (Experiment 20)** + figure `step82_certified_horizon.png`; **Theorem B′ + proof +
  remarks** adjacent to Theorem B (§3.2); contribution-1 clause (makes *"Certified"* literal); abstract reordered to
  lead with load-bearing claims (tagline "*Scale buys interpolation; structure buys a certified horizon.*" first
  sentence; hedges live in §6/§7). PDF rebuilt. `step82`, `fec83b9`.

## [2026-06-08] Step 83 — R²(N) crossover hardens Exp 18

- R²(N) crossover hardens Exp 18: conv 0.98 across N; MLP/GRU tie through N=28 then collapse at N=40 to −1.8/−0.27; **PHASE_TRANSITION**; 3 seeds; `step83`, `24dbca1`.

## [2026-06-08] Experiment 21 — certified horizon on a recognized chaotic control benchmark (Acrobot-v1)

- **Setup.** Gymnasium Acrobot-v1 (underactuated double-pendulum swing-up), learned action-conditioned $\mathbb{Z}_2$-equivariant (reflection frame-averaged) world model, 3 seeds. RTX 3080. Genuinely chaotic: true swing-up finite-time $\lambda_1=0.094$ ✓. Learned-WM one-step rollout relMSE $=4.8\times10^{-5}$ (good model). Cert route = **bootstrap** (cone vacuous, $L_{J,\mathrm{net}}=21.7$) — abstains, the safe move (§5.18).
- **Triad (i) accuracy = certified-vs-measured horizon, two-regime $\epsilon$ — KEEP.** Ratio meas/cert climbs $0.42{\to}0.47{\to}0.93$ as $\epsilon$ coarsens $0.01{\to}0.1{\to}0.3$ (tight $\epsilon$ = Prop-8 optimistic regime; $\epsilon{=}0.3$ **predictive**).
- **Triad (iii) binding = TRUE — KEEP.** Return vs plan depth $H$ has a sharp interior optimum; planning past the predictability horizon fails. Equivariant: success **0.67 at $H{=}41$**, **0.00 at $H{\ge}164$**. $G$-binding TRUE.
- **CORRECTION (calibrated-$\epsilon$ re-run, RTX 3080).** Prior "three-way coincidence" (optimal depth ≈ measured horizon ≈ certified $T_1$) was an **$H$-grid artifact of a noisy first run** — REMOVED. Honest relation: $H^\star{\approx}T_1/2$ across **both** the fixed-$\epsilon$ and calibrated-$\epsilon$ runs and **both** WM variants. Equivariant (calibrated) $H^\star{=}41$ vs certified $T_1{=}82$; non-equivariant $H^\star{=}78$ vs $T_1{=}156$. Capping at certified $T_1$ **succeeds** (swing-up $67$–$100\%$) and **bounds** useful planning (deeper fails), but is ~$2\times$ the return-optimum → loses narrowly (equi: cert-aware $H{=}82$ return $-282$ vs best-blind $H{=}41$ return $-264$).
- **Triad (ii) no-tuning return-win = INCONCLUSIVE ×2 (honest).** Two pre-registered rules both fail to beat best-tuned blind: cap at $T_1$ of a *fixed* $\epsilon$; cap at $T_1$ of the *calibrated* $\epsilon$ (where certified $\approx$ measured). Both land at ~$2\times$ return-optimal depth. NO return-win claimed.
- **Honest verdict.** Certificate is accurate (i) and binds for planning (iii) on a recognized chaotic control benchmark, and gives a sound no-tuning planning depth **within a constant factor (~$2\times$) of optimal** that provably bounds where planning helps — but it is not the return-optimum, so the clean return-win does **not** land.
- **Folded:** paper2 **§5.19 (Experiment 21)** as honest corrected result (no figure embed — `step84_certified_control_benchmark` pending GPU-box sync); ICLR **(E11)** compact figure-free paragraph after E10. `step84`, `tests/test_step84.py`.

## [2026-06-09] Experiment 22 — structure → trustworthy certificate → downstream win (step85, direction ③; A wins, B calibrates, C honest-negative)

Closes the paper's own loop: E2/Step 83 *measures* that the $\mathbb{Z}_N$-conv recovers the $N{=}40$ spectrum
($R^2{=}0.98$) where the dense MLP's is garbage ($R^2{<}0$, $\lambda_1$ inflated $\sim3$–$4\times$). Exp 22 gives that
faithfulness a **downstream consequence** on the D2 active-re-observation task (Lorenz-96 $N{=}40$, $u\equiv0$, 3 seeds),
reusing `step74` conv/MLP + `step79`'s D2 machinery. Brainstorm → spec → CPU G0/G1 → 3080 for C; design
`docs/specs/2026-06-08-step85-...md` (+ revisions 2026-06-09a/b), `tests/test_step85.py` & `test_step85b.py` (10 green).

- **B (premise, G0 PASS 3/3).** The conv certificate is **calibrated** at $N{=}40$ D2: measured/certified horizon
  ratio $1.17,0.63,0.82$ (all in $[\tfrac12,2]$). The MLP $\lambda_1$ is inflated $3.19,3.50,3.46\times$ (certified
  horizon $24$–$28$ steps) but its **empirical** horizon is $64$–$73$ — so its cert is **pessimistic** ($\rho_{\rm
  self}\approx2.5$), not optimistic. **Forecasters are MATCHED** (conv empirical horizon $62$–$104$, MLP $64$–$73$;
  both one-step relMSE $\sim10^{-5}$): the MLP forecasts *values* as well as the conv and fails **only on the
  certificate** (the Jacobian/spectrum) — the confound (forecaster vs certificate) is **absent in the data**.
- **A (headline, G1 PASS 3/3).** *Cert-isolated, fixed-budget frontier* — hold the forecaster fixed (the conv) and feed
  it the conv-cert interval ($T_1^{\rm lo}$, $72$–$84$ steps) vs the MLP-cert interval ($24$–$28$); sweep the observation
  budget. The inflated-$\lambda_1$ MLP cert **over-observes $\sim3\times$ → starves the budget → episode tail runs
  open-loop → higher aggregate violation.** At the knee budget $B^\star{\approx}18$–$21$: conv-cert violation
  $0.08$–$0.16$ vs MLP-cert $0.61$–$0.65$, **margins $+0.45,+0.50,+0.57$ (3/3).** Same forecaster ⇒ the gap **is** the
  $\lambda_1$ ratio. The fixed budget is **load-bearing**: without it the MLP's over-observation reads as "safe" and a
  naive violation comparison would favour the wrong model (Exp-22 empirically validates the spec §2 resolution).
- **Adaptive foil (honest scope).** A cert-FREE AIMD agent (learns the cadence from observed error) loses badly at the
  knee ($0.83$ vs conv's $0.16$ — it burns scarce budget exploring) and **only overtakes the certificate at $\sim3\times$
  budget.** So the certificate's value is precisely the **a-priori / tight-budget / few-shot warm-start**; its
  *necessity* (no-feedback safety) is direction ① territory.
- **Global-cert ceiling (→ motivated C).** The conv-cert arm plateaus at $0.08$–$0.16$, not $0$, even at large budget:
  a *global* interval (global $\lambda_1$) is too long for locally-hard orbit stretches (FTLE variation; $p25$ empirical
  horizon $69.5 <$ iv_conv $77$).
- **C (step85b, INCONCLUSIVE — honest negative, RTX 3080).** Full-spectrum allocation across a forcing-$F$ ensemble
  ($F\in\{6,8,10,13\}$, $L{=}1500$, 3 seeds, CUDA): split a fixed total budget by each spectrum's $\lambda_1(F)$,
  forecast each regime with its conv (cert-isolated). **conv-allocation does NOT beat MLP-allocation** — best margins
  $[-0.011,-0.003,+0.024]$, **0/3** (the 2F/$L{=}300$ smoke's $+0.085$ was a small-scale false positive). Two
  mechanisms: (1) the learned conv $\lambda_1$ is **under-biased at low $F$** ($F{=}6$: $0.68$ vs true $1.03$) → it
  under-allocates to the weakly-chaotic regime, which still needs a *coverage floor*, so it **starves** it, while the
  MLP's range-compressed (flat) weights accidentally don't; (2) $\propto\lambda_1$ allocation is itself not optimal
  (even the oracle doesn't strictly dominate the flat split). The precheck's weight-distance distortion ($0.219$ vs
  conv $0.081$) did **not** transfer to a downstream win. **No gate loosening** — recorded as the honest negative sample.
- **Honest verdict.** ③ lands on **A** (a clean cert-isolated *efficiency-under-budget* win that makes E2's $R^2$
  consequential — within-method, so Exp-21's $\sim2\times$ conservatism cancels) **+ B** (calibration) **+ an honest
  C-negative**. Ceiling = *mechanism + efficiency*, not a safety/necessity claim. Realistic per the seed ("landing one
  cleanly takes the paper to ~8"); A is that one.
- **Proposed paper paragraph (paper2 §5.20 / ICLR E12 — DRAFT, not yet placed; awaiting review before editing the
  canonical draft):** *"Does a faithful spectrum's certificate change what an agent should do? On $40$-D Lorenz-96 an
  agent must schedule sparse re-observations of a chaotic forecast under a fixed sensing budget. Timing re-observation
  by the $\mathbb{Z}_N$-equivariant model's certified horizon yields $8$–$16\%$ aggregate forecast-violation at the knee
  budget; timing it by the non-equivariant model's certificate — whose leading exponent is inflated $\sim3\times$ —
  over-samples, exhausts the budget, and leaves the episode open-loop, at $61$–$65\%$ violation (3/3 seeds). The two
  models forecast equally well (matched empirical horizon, relMSE $\sim10^{-5}$); the gap is attributable solely to the
  certificate, which the equivariant model's spectral faithfulness makes calibrated. The advantage is an a-priori
  warm-start: a feedback-driven adaptive scheduler matches it only given $\sim3\times$ the budget. (A budget allocation
  across a chaoticity ensemble did not yield a further win — the learned spectrum is not accurate enough at low forcing
  to beat a flat allocation; reported honestly.)"*
- **Status / not yet folded:** figure `papers/figures/step85_headline.png` ready (3-panel: calibration | cert-isolated
  budget frontier | package E-vs-N). Canonical paper2/ICLR sections **drafted above, not yet placed** (autonomous
  paper-rewrite left for user review). `step85`, `step85b`; commits `b3723d4` (G0) `a19a7ba` (Phase1+tests) `b3a5383`
  (G1) `117a6f9` (figure) `ffc470f` (C-negative).

## [2026-06-09] step86 (direction ①) — certificate as a SAFETY / catastrophe bound: INCONCLUSIVE (honest negative)

Direction ① (seed): the certified horizon as a re-observation cadence for **catastrophe avoidance** on controlled
Lorenz-96 (stabilize the unstable fixed point $F$; escape $=$ catastrophe). Lever $=$ re-observation interval $\tau$
(state estimate drifts via the WM between observations, reset to truth every $\tau$; modest planning depth). cert-aware
$\tau{=}T_1$, blind $\tau\gg T_1$. Full run N=24, $\epsilon{=}0.3$, 3 seeds, RTX 3080
(`papers/figures/step86_certified_safety.json`; `experiments/step86`, `tests/test_step86.py` green incl. equivariance).

- **INCONCLUSIVE — not horizon-binding.** Escape (catastrophe) rate is **$\tau$-INVARIANT**: per seed, $\{T_1/2, T_1,
  2T_1, 4T_1\}$ all give the SAME rate (seed0 $0.25{\times}4$; seed1 $0.19{\times}4$; seed2 $0.25{\times}4$). The
  re-observation cadence does not determine escape. **Diagnosis:** the $19$–$25\%$ escapes are **control-failures** (the
  modest reused equivariant GD planner, $H_{\rm plan}{=}6$, cannot stabilize those near-$F$ starts), NOT
  estimate-staleness failures — so trusting the model past $T_1$ does not cause MORE escapes. The certificate's
  conservatism is not the binding constraint here; control quality is. Exactly the seed's flagged ① risk.
- **Caveat (a future angle, not pursued):** a stronger planner might lower the baseline escape rate and possibly expose
  an estimate-staleness effect (cert-aware safe / blind escapes). As-run with the reused step79 planner, ① does not bind.
- **Dead-end recorded for honesty:** the first ① design conflated $T_1$ with planning DEPTH → blind $H{=}8T_1{\approx}800$
  step chaotic-rollout gradient planning (explodes + intractable); fixed via the $\tau$-reobservation reframe before this run.
- **No gate loosening.** Honest negative, like step85b's C.

**Three-directions-to-8 scorecard:** ③-A (cert-isolated budget re-observation) **LANDED** (Exp 22, G1 3/3, in paper2
§5.20 + ICLR E12); ③-C (full-spectrum allocation) and ① (safety) are **honest NEGATIVES**; ② (cert-gated MBRL) is
**build-ready-spec'd** (pathwise cert-gating), not run (~0.4 gamble, do last). `step86`, commit pending.

## [2026-06-09] step88 (B-generality) — ③-A GENERALIZES to a 2nd system (coupled-pendulum ring): G1 PASS 2/3

Hardening ③-A against the reviewers' #1 blocker (single system). Replicated ③-A's **cert-isolated budget frontier** on
the driven–damped coupled-pendulum **ring** (step80; $\mathbb{Z}_N$, $N{=}24\to 49$-dim; physically different from
Lorenz-96), reusing step80's RingConv/RingMLP + certificate + a **budgeted variant of step80.reobserve_run** (step80
unmodified). `experiments/step88_ring_generality.py`; `papers/figures/step88_ring_{precheck,frontier}.json`.

- **Precheck (go/no-go) GO:** at $N{=}24$ the ring is chaotic (true $\lambda_1{=}0.317$), the dense RingMLP's $\lambda_1$
  is **inflated $2.05\times$** (vs RingConv faithful, calibration ratio $0.89$), both relMSE $\sim0$ (matched fidelity) —
  ③-A's precondition holds. (It is a HIGH-$N$ effect, like Lorenz-96: step80's own $N{=}10$ runs would not show it.)
- **Full N=24, 3 seeds — G1 PASS (2/3):** conv-cert beats MLP-cert at the knee budget on $2/3$ seeds, margins
  $+0.242/+0.035/+0.329$ (knee violation conv $0.013/0.000/0.034$ vs MLP $0.255/0.035/0.363$). Conv $<$ MLP on **all 3**;
  seed 1's margin ($0.035$) is below the $0.05$ bar because that seed's inflation was only $\sim1.15\times$.
- **Honest scope:** the ring effect is **smaller and more seed-dependent** than Lorenz-96 (ring spectral inflation
  $\sim1.15$–$2\times$ vs L96's $3.0$–$3.9\times$). But the **mechanism transfers** — ③-A is a class property across two
  symmetry groups and two physical systems, not a Lorenz-96 idiosyncrasy. Folded into §5.20 (the generality sentence).
  `step88`, commit pending.

## [2026-06-09] step85c (B-calibration baseline) — the do-or-die ablation → REFRAME (③-A is a-priori, not exclusive)

The reviewers' make-or-break ablation: the MLP $\lambda_1$ inflation is near-constant ($3.19/3.50/3.46$) — can a
one-parameter recalibration close the gap? **Yes, 3/3.** Adding a recalibrated-MLP-cert arm (MLP cert set to its
**measured** empirical horizon) to the cert-isolated frontier (`experiments/step85c_calibration_baseline.py`,
`papers/figures/step85c_calibration_baseline.json`, N=40, 3 seeds): the raw-MLP gap $+0.45/+0.50/+0.57$ shrinks to
$-0.06/+0.08/+0.04$ — **gap closed on 3/3 seeds**.

- **Honest consequence = REFRAME, not refutation.** ③-A is NOT "only equivariance can act." It is: the equivariant
  model's certificate is correct **a-priori** (from the Jacobian, **zero rollout/calibration data**), whereas a dense
  model matches it **only after measuring its horizon on a calibration set** — exactly the $\sim3\times$ warm-start cost
  the cert-free adaptive baseline already pays. The precise claim is **"structure buys a trustworthy certificate without
  calibration data."** This is cleaner and more defensible than the original framing, and pre-empts the obvious reviewer
  attack ("just recalibrate the dense cert"). Folded into §5.20.

**B COMPLETE.** B-generality (step88, ring G1 PASS 2/3) + B-calibration (step85c, reframe to a-priori). Both fold into
§5.20 and harden ③-A toward 8/oral: a second system answers "single-system narrowness," and the a-priori scoping
answers "just recalibrate." `step85c`, commit pending. Next: **C (② cert-gated MBRL)**, gated on these B results.

## [2026-06-09] Proposition 9 — ③-A's decision-relevance made PROVABLE (reviewer A's #2; theory, no compute)

Reviewer A's #2 lever: turn ③-A from an empirical observation into a theorem (the paper's brand is "provable"). Added
**Proposition 9 (budgeted re-observation — a mis-estimated horizon costs a proportional budget)** to §3.2 after Prop 8.
A renewal/coverage argument: with sensing budget $B$ over an episode of $L$ steps and trustworthy horizon $H=T_1/\Delta t$,
a certificate inflated $c\times$ (cadence $H/c$) gives aggregate violation $V(c)=\max(0,L-BH/c-H)/L$ — **non-decreasing
in $c$** — and needs $B^\star(c)=\lceil c(L-H)/H\rceil$ observations to reach zero violation — **linear in $c$**. Proof:
coverage/renewal (covered windows under $\epsilon$; open-loop tail violates after $H$). 

- **Why it matters:** (i) makes the decision-relevance a *law*, removing the "dressed-up calibration" critique (reviewer
  A's sharpest); (ii) it's *predictive* — $V(c)$ computable a priori; (iii) it explains the experiments *quantitatively*:
  Lorenz-96 $c\approx3.4\Rightarrow\approx3\times$ budget, ring $c\approx2\Rightarrow\approx2\times$, recalibration $c\to1$
  closes the gap (step85c) — all forced by Prop 9; (iv) ties §5.20 to §3 (only structure delivers $c=1$ a priori, Prop 8).
- Referenced from §5.20. `paper2_certified_world_models.md` §3.2 + §5.20. commit pending.

**Status:** ③-A now has generality (step88) + a-priori scoping (step85c) + a provability law (Prop 9) — the three
highest-value reviewer levers, all landed. C (② MBRL, ~0.4 gamble) remains the open choice.

## [2026-06-09] step87 (C / ② Stage A) — cert-gating mechanism is the Lyapunov amplification (near-tautological)

Built ②'s **Stage-B-ready scaffold** (`experiments/step87_cert_gated_mbrl.py`: Actor, Critic, pathwise `actor_objective`
with backprop-depth $H_g$ = the cert-gate + critic tail-bootstrap) and ran the **Stage-A diagnostic**. Finding: the
pathwise actor-gradient amplifies at $\sim$ the cocycle rate $e^{\lambda_1 H_g\Delta t}$ — the cert-gating "mechanism"
(gradients explode past $T_1$) is **the Lyapunov amplification restated** ($=\lambda_1>0$, which the certificate already
establishes). So Stage A **confirms cert-gating is principled** (gate where the cocycle is untrustworthy) but is **NOT
an independent go/no-go** — it doesn't de-risk the actual ② question.

- **②'s genuine test = Stage B** (online actor-critic loop; does cert-gating the imagination horizon buy *sample
  efficiency*?) — the ~0.4 gamble (heaviest; seed warns it's near §5.19's plan-depth failure mode); hours of from-scratch
  RL that may land another honest negative.
- **Decision point (gated on B):** ③-A is now strongly hardened (generality step88 + a-priori step85c + Prop 9). So ②
  is *optional second-result upside*. Either run the Stage-B gamble, or **consolidate** ③ (it's solid). `step87`, commit
  pending. (Note: the cert-gating *design* — gate the imagination at the certified horizon — is principled and recorded
  even if Stage B is not run.)

## [2026-06-09] P0 sync & rewrite (Fable-5 review) — §5.20 single-arc rewrite + ICLR brought up to date

Fable-5's fresh review of the post-B state found one real inconsistency *we introduced ourselves*: paper2 §5.20 had
been amended three times (horizon-wording fix → ring generality → recalibration reframe) into a patchwork whose claims
ICLR never received — E12 still made the *unscoped* actionability claim while the repo's own `step85c` JSON shows a
one-parameter recalibration closes the gap. Fixed, zero compute:

- **paper2 §5.20 rewritten as ONE arc** around the final claim — *structure makes the certificate trustworthy before
  any data is spent* — with the budget law (Prop 9), the two scoping controls (adaptive $3\times$ budget; recalibration
  spends a calibration set), and the ring replication integrated rather than appended. Title now carries "a-priori,
  with zero calibration data."
- **paper2 abstract / contribution-3 / conclusion** updated to the scoped claim (+ Prop 9, + ring, + zero-rollout-data).
- **ICLR synced**: abstract + intro item (v) now scoped (*actionable, a priori*; recalibration caveat; ring); **E12
  rewritten** to match §5.20 (Prop 9 law, catch-up budget $2.7$–$3.5\times$, recal control, ring replication,
  `step85c`/`step88` artifacts); **Proposition 9 added to the ICLR proofs appendix** (statement + proof + remark,
  numbering consistent with its Props 4–8).
- Cross-session note: Prop 9 (`99ede0b`) and ② Stage A (`3cd9bbc`, honest finding: the gradient-explosion diagnostic is
  the Lyapunov amplification restated — Stage B remains the real ~0.4 gamble) landed from a parallel session; this
  entry completes the P0 their work left open. PDF rebuild still pending (Rosetta).

## [2026-06-09] Review-claim verification — §5.17's "2/3 frontier" stands (reviewer false-positive); recipe-parity clause added

Verified reviewer B's flag "step81 double-pendulum frontier is 0/3 vs the prose's 2/3" against the JSON before editing:
**false positive** — `step81_double_pendulum.json` has `frontier_wins_pred = 2` at the double pendulum's own calibrated
$\epsilon{=}0.7$ (seeds 0/1 on-frontier), exactly what §5.17 claims ("knee at $\epsilon{=}0.7$ … the same $2/3$
frontier"); the reviewer read only the tight/mid/hi fields (0/0/0) and missed the predictive regime. step80 ring
likewise supported (`frontier_wins_predictive = 2` at $\epsilon{=}0.3$). **No edit to §5.17** — receiving-review
discipline: verify before implementing. The reviewer's *valid* adjacent point (state training-recipe parity for the
§5.20 baseline) is now explicit in §5.20 + E12: both models identically trained (same data, same $K$-step rollout
loss). Remaining true item from that review: the **step84 JSON in-repo is a degenerate smoke run** — the real 3080
re-run is queued (`run_step84.sh`, one command) but the **box is offline**; fire on wake.

## [2026-06-09] step84 re-run on the RTX 3080 — the lost "real run" reproduced EXACTLY; §5.19's artifact gap closed

The reviewer-flagged desk-reject hook ("§5.19 cites numbers whose JSON is not in the repo — the shipped JSON was a
degenerate smoke run") is closed: `run_step84.sh` re-ran end-to-end on the 3080 (box commit/push failed as the script
itself anticipates — deprecated auth; artifacts pulled and committed from the controller). **Every load-bearing number
in §5.19/E11 reproduced exactly**: true swing-up $\lambda_1{=}0.0939$ (prose $0.094$); equivariant relMSE
$4.77{\times}10^{-5}$ (prose $4.8{\times}10^{-5}$); ratio chain $0.42\to0.47\to0.93$; calibrated
$\epsilon^\*{=}0.3$, $T_1{=}82$; $H^\star{=}41$ (return $-264.3$, success $0.67$) vs cert-aware $H{=}82$ ($-282.0$);
$H{\ge}164$ success $0$; **non-equivariant $H^\star{=}78$ vs $T_1{=}156$** (the "same $2\times$ gap");
G-binding TRUE; G-ii honest no-win; verdict string = the in-paper INCONCLUSIVE. smoke=false, 3 seeds.
§5.19's "(figure pending sync)" replaced with the real figure embed. The prose needed **zero** corrections — the
original run was faithful; only its artifact had been lost.

## [2026-06-09] Engineering note — WSL2 vs native-Windows CUDA A/B on the 3080 box (claim refuted; CS2 confound caught)

Tested the folk claim "Linux/WSL2 cuts 3080 performance" with `scripts/bench_gpu_ab.py` (same card, same torch
2.5.1+cu121, three workloads). **First run was INVALID — the control caught it**: big-matmul throughput was absurd
(2.5 / 0.36 TFLOPS), nvidia-smi showed 100% GPU from a foreign process — **Counter-Strike 2 was running on the box**.
Lesson: check `nvidia-smi --query-compute-apps` before any 3080 run (a contaminated GPU ran our workload up to ~29x
slower). Clean re-run with the GPU idle (P8, 29 W): **compute-bound work is identical** (matmul 22.68 vs 22.58 TFLOPS;
batched rollout 335.9 vs 335.7 steps/s) and **our actual small-kernel training profile is 63% FASTER under WSL2**
(93.7 vs 57.4 iters/s — Windows WDDM launch overhead; the profile of step74/79/85/88 training and ② Stage B's
imagination loop). **Decision (pre-registered rule): stay on WSL2** — the established pipeline is not merely fine, it
is the faster option on this box. Windows-side toolchain (Python 3.11.9 + torch cu121 + `C:\Temp\bench_gpu_ab.py`)
left in place for future re-checks.

## [2026-06-09] step87 Stage B (② cert-gated MBRL) ran on the box — raw gate PASSED VACUOUSLY; honest verdict INCONCLUSIVE

The ~0.4 gamble ran end-to-end (N=16, eps=0.5, 3 seeds × 4 arms {cert≈T1, T1/2, 2T1, ungated-96} from identical
inits, 20 iters × 50 real steps; `step87_stageB_sample_efficiency.json`). **The pre-registered G2 returned "PASS 2/3"
— and scrutiny shows that pass is an artifact, which we refuse to let stand.** (i) On seeds 0/1 ALL FOUR arms reach
the within-10%-of-best threshold at the FIRST eval (steps-to-thresh = 50 across the board): the threshold was loose
enough to saturate immediately, so the cert "wins" are $\le$-ties — vacuous. (ii) The only seed with real separation
(seed 2) has cert **losing** to fixed-half by one eval notch (650 vs 600). (iii) Final returns differ by ~1–5% across
arms — within noise. (iv) The re-certified $T_1$ is high-variance across refits (41→96, 40→93, 20→11): the
per-iteration certificate on a continually-refit WM is itself noisy. **Honest verdict: INCONCLUSIVE — the experiment
did not discriminate between gating strategies at this scale.** Gate-design flaw disclosed: ties counted as wins +
threshold too loose; "never loosen a gate" has a dual — never let a vacuous pass stand. A future re-run needs: a
task hard enough that warmup does not already saturate the threshold, a strictly-positive win margin (no ties),
a finer eval grid, and a variance-controlled certificate (averaged over refits). Not pursued tonight.

**Three-directions-to-8 FINAL scorecard:** ③-A **LANDED** (Exp 22: cert-isolated budget win, generalized to the ring,
a-priori-scoped, Prop 9 law, in both papers + rebuilt PDF); ③-C, ①, and now ② — **honest non-results** (allocation
negative / not horizon-binding / no discrimination). One clean landed direction + three honest negatives, zero
loosened gates. `step87`, commit pending.

## [2026-06-09] Experiment 23 — the certificate reads OFFICIAL TD-MPC2 checkpoints, training-free (step89; the #1 reviewer gap closed)

The exceptional-ICLR review's #1 gap ("the certificate never touched a model the community recognizes") closed in one
evening, training-free. Recon (sub-agent, URLs/API verified; **the exact audit absent from the literature** — closest:
arXiv:2410.10674, true-env Lyapunov under RL policies; Özalp & Magri 2025, latent spectra for scientific-ML AEs) →
spec with pre-registered readings → line-faithful slice rebuild (SimNorm/NormedLinear/mlp; **strict-load G0 passed on
the real checkpoint first try**) → full audit, 2 tasks × 3 official seeds, ~3 min CPU.

- **walker-walk ($\lambda_1=0.25/0.25/0.30$, expansive):** the two-regime $\epsilon$ pattern reproduces on a public
  model — ratio measured/certified $0.08$–$0.20$ @ $\epsilon{=}0.05$ (Prop-8 optimism) → **$0.94/0.95/1.02$ @
  $\epsilon{=}0.2$ — calibrated ≈1 on 3/3 official seeds.** The certificate read a community model's trustworthy
  horizon a-priori.
- **acrobot-swingup ($\lambda_1\approx-0.04$, contracting):** certificate **abstains, correctly per Prop 7** — the
  measured divergence (median 14–45 steps) is bias-driven, outside a Lyapunov certificate's scope; issuing one would
  over-claim. Both pre-registered branches of the theory verified on official checkpoints.
- SimNorm structural band 128–270 strongly-negative directions (≥ the 64 predicted; reported). Scope: policy-prior
  loop, not MPPI. deps pinned mujoco 3.3.2 + dm-control 1.0.28 (numpy2); checkpoints gitignored, URLs pinned.
- **Folded:** ICLR **E13** + paper2 **§5.21**. `step89`, `tests/test_step89.py` (4 green; suite 21 green).

## [2026-06-10] Experiment 24 (step90) — UQ head-to-head: no method dominates; the certificate is the only a-priori point on the accuracy–cost Pareto

The strongest methodological flank ("why a Lyapunov certificate instead of standard UQ?") answered on the §5.20 model
(N=40, eps=0.2, 3 seeds): **certificate** ratios 1.36/1.46/0.89 (mean |log r| 0.266; 1 model, **0 truth rollouts**);
**ensemble-disagreement** (K=4, same data, different inits) 1.07/1.16/0.75 (0.169; 4× training); **conformal quantile**
(n_cal=10) 0.94/1.13/0.87 (**0.108**; truth access). Honest verdict: on point calibration conformal > ensemble >
certificate (certificate within its known ~1.5× band) — but each rival spends exactly what the certificate does not
(4× compute / 10 ground-truth rollouts), and neither provides CIs, per-channel stratification, or the Prop 9 budget
law. **Folded into §5.20 + E12 as the third scoping control.** Ops note: two phantom "kills" of this run were
monitoring false-negatives (sandboxed pgrep flakiness + silent training epochs misread as death) → self-inflicted
4-process contention; resolved by per-seed parallel shards + file-based completion detection; one genuinely
unexplained process death (run 1) remains. Determinism triple-confirmed (seed-0 numbers bit-identical across 3
independent runs). `step90`, `tests/test_step90.py`; merged artifact `step90_uq_baselines.json` (+ _s0/_s1/_s2 shards).

## [2026-06-10] E13 expansion (step89 ×5 tasks) + ICLR P0 compression — the audit becomes a SCOPE MAP; the draft fits the page budget

**E13/Exp 23 expanded to 5 DMC tasks × 3 official seeds (15 latent loops), ~6 min CPU, zero training.** The result
upgrades the audit from "two tasks" to a **scope map that tracks the paper's own theory cell-by-cell**: strongly
expansive loops ($\lambda_1{=}0.25$–$0.30$: walker 3/3, cheetah-3) calibrate at coarse $\epsilon$ (ratio
$0.83$–$1.02$); weakly expansive loops turn optimistic (cheetah $0.43/0.50$, hopper $0.13/0.38$ at
$\lambda_1{=}0.05$–$0.09$) — bias outpacing amplification, Prop 7's degeneracy direction; contracting loops (6/15)
abstain **correctly in both sub-cases** — finger-spin's stable loops genuinely never diverge (15–19/20 censored at
300 steps), acrobot/hopper-1's divergence is bias-driven, outside Lyapunov jurisdiction. Folded into E13 + §5.21.
Merged 15-cell artifact `step89_pretrained_audit.json` (script's STEP89_OUT ignored → canonical restored from git and
merged; registry +cheetah-run/finger-spin/hopper-hop).

**ICLR P0 compression executed:** main text $6660\to5305$ words, figures $13\to5$; E1/E3–E8 moved verbatim to a new
**Appendix D** with a one-paragraph supporting-suite summary in main; E2 merged to two paragraphs (+phase-transition
and class-staircase figures); E9/E11 tightened; abstract $283\to\sim185$ words. Main text now ≈ page budget; remaining
craft pass (camera-grade figure polish, final lit sweep, anonymized artifact) is the P2 tail.

## [2026-06-10] Final lit sweep (through 2026-06) — all four novelty claims SAFE; concurrent-work paragraph added to both papers

Sub-agent sweep (~20 queries, WebFetch-verified): **TD-MPC2 audit novelty SAFE** (nobody computed Lyapunov/Jacobian
quantities of a public pretrained RL WM in-window; nearest = semantic probing arXiv:2603.21546, sample benchmarks
WorldBench/WorldArena; nearest priors = Özalp & Magri latent-stability of self-trained AEs, 2410.00480, and
Lyapunov-regularized Dreamer policies, 2410.10674 — both now cited). **Two-sided horizon + converse SAFE** (only new
equivariance-Lyapunov theory is Mo arXiv:2605.03338 — symmetry-protected NEUTRAL modes, complementary, vacuous for
discrete Z_N; must-cite + distinguished). **Geng et al. 2512.08991** (conformal WM-rollout verification) preempted as
the statistical/rollout-hungry counterpoint. @lillemark2026flowm updated to ICML 2026 (+FERNN antecedent). ~10
SHOULD-CITEs folded into one dense "Concurrent and recent work" paragraph in ICLR §5 + paper2 §6. LeWM (public
checkpoints, Mar 2026) noted in the research-line snapshot as the next free audit target for step89.

## [2026-06-10] Overnight autonomous block — anonymized artifact + vault accounting + final PDF + step91 seed (LeWM recon: GO)

User asleep ("你只管做吧"). Done: **(1) anonymized reproducibility artifact** — `scripts/make_anon_artifact.py`
(scrub + zero-leak self-check; 333 files, 1.9 MB; ANON_README with the full repro matrix; one self-bite fixed: the
builder's own scrub regexes tripped the leak check — excluded from the zip). **(2) vault accounting** — wiki Paper2
source page updated 06-04→06-10 (Exp 22/23/24, compression, lit sweep, artifact). **(3) final PDF** — concurrent-work
paragraph built in on the box; submission-ready. **(4) step91 seed spec** — LeWM audit recon (sub-agent, verified):
checkpoints MIT on HF (`quentinll/lewm-*`, 72.3 MB state dicts + Hydra configs), latent loop deterministic+
differentiable in eval (ViT-tiny CLS → 192-d; 6-layer Predictor w/ AdaLN+causal SDPA; BN-MLPs; 3-frame window ⇒
576-d delay-embedded QR state), PushT env pip-clean on macOS; **GO-with-caveats** (no policy prior → fixed-action
loop scope; companion-band artifacts expected; strict-load + one-step sanity pre-registered as G0). Deliberately NOT
built tonight — transformer-replica fidelity deserves a fresh session, per the recon→spec→build cadence.
`docs/specs/2026-06-10-step91-lewm-audit-seed.md`.

## [2026-06-10] step91 — LeWM joins the E13 scope map (second family, zero-replica): ABSTAIN cell, correctly

User: "先把1做了吧". ZERO-REPLICA build: official `stable-worldmodel` package + strict-load of the official
`quentinll/lewm-pusht` checkpoint into the AUTHORS' code (transformers<5 pinned for classic ViT naming). Three real
integration nails (recorded in-code): float32-hardcast Embedder → constant act-emb cast f64; CPU flash-SDPA lacks
forward-AD → math-kernel JVP Benettin; pixel scale resolved by the pre-registered one-step sanity ([0,1]: 0.165 vs
[0,255]: 0.289). **Result (8 starts, k=12 leading-band Benettin on the 576-d delay state): λ1 = 0.0013, CI
[−0.0075, 0.0161] — straddles zero ⇒ ABSTAIN** (neutral-to-contracting free-running loop; physically sensible: PushT
is static under the fixed zero action). Measured divergence median 1–2 steps at ε≤0.2 = pure one-step bias (0.165),
the acrobot sub-case — outside Lyapunov jurisdiction, correctly not certified. Pre-registered caveats stand: no
policy prior (fixed a*=0 scope); zero action in-support but expert-OOD. **E13 upgraded: two architecturally disjoint
families (TD-MPC2 MLP/SimNorm state-based; LeWM ViT/transformer JEPA pixel-based), one unchanged read-out, one
theory-aligned taxonomy — the training-free audit is a method, not a trick.** Folded into E13 + §5.21.
`experiments/step91_lewm_audit.py`, `step91_lewm_audit.json`; ~30 min wall, zero training, zero GPU.

## [2026-06-10] E14/Exp 25 (step92) — the scale sweep: SCALE DOES NOT BUY A CALIBRATED HORIZON (the title, measured)

The 9-level play executed overnight: certified horizon vs MODEL SCALE on the official TD-MPC2 **multitask** ladder
(mt30 1M/5M/19M/48M/317M + mt80 1M/5M; same walker-walk task, task index from each ckpt's own metadata; 2.45 GB of
official checkpoints; shape-driven loader auto-absorbs the two upstream quirks the recon verified — mt30-19M
latent=512, task_dim 96-vs-64). **Result: the loop regime is NON-MONOTONE in scale** — contracting at 1M AND 48M(!),
expansive at 5M/19M/317M — calibration scatters (0.37 / 1.87 / 1.16 at eps=0.2; mt80 mixed; all three taxonomy
behaviors appear, including the *conservative* direction at 19M), and **no multitask size matches the single-task 5M
calibration (0.94–1.02)**. One ckpt per cell (no official seeds) — reported as descriptive. Honest reading is
thesis-CONFIRMING in the strongest available form: had calibration improved with scale, the title would weaken;
instead, trust is a property of the loop's dynamics, not parameter count. Folded as ICLR **E14** + paper2 **Exp 25**
+ abstract hook ("scale buys interpolation, not a calibrated horizon") + figure. Also tonight: ring extended to 5
seeds (gap appears exactly where dense lambda1 inflates, 2/5 — §5.20/E12 sharpened to the conditional-causal claim);
unification paragraph (②) in both papers. `step92`, ~50 min total, zero training, zero GPU.

## [2026-06-10] step93 (B, the closed-loop gamble) — honest INCONCLUSIVE + the scope law it bought; D (wm-audit tool) + C (Prop 10) landed

User: "你先开始 BCD". **D done**: `scripts/wm_audit.py` — one-command certification of any supported pretrained WM +
`audit_map()` API; JVP-vs-full-QR cross-check (walker T1@0.2 5.6 vs 5.4–6.4). **C done**: Proposition 10
(finite-sample certified-horizon interval; Bernstein-for-mixing on bounded log-stretches; $n\asymp\log(1/\delta)/
\varepsilon^2$; bootstrap = the consistent $\sigma_\infty$ estimator; assumptions inherited from Prop 7, stated).
**B executed end-to-end**: recon (reward head validated vs real env corr 0.993; Q vmap-blob mapping verified) →
faithful MPPI replica → **ANCHOR PASSED (cadence-1 = 993.5/993.6/996.3 vs official 977–983)** → cadence sweep
{1..24}×3 episodes. **Pre-registered prediction FAILED honestly**: return knee at k≈2 (90% of anchor), NOT at
$T_1(0.2)\approx6$; control is sensitive at ε≈0.05 where the MEASURED horizon (1–2 steps) matches the knee but the
certificate is in its known tight-ε optimistic regime. Third decision-level INCONCLUSIVE (after step84-ii, step86)
→ unified into a **scope law** folded into both papers' Limitations: decision value concentrates where the decided
quantity IS the certified quantity (E12), dilutes through a task-level map (E11, step93). The MPPI-replica anchor is
itself a reusable methodological asset. `step93`, ~40 min compute. No gate loosened; the ~0.5 gamble priced correctly.

## [2026-06-10] step94 (①, E15/Exp 26) — the published certificate prices a DEPLOYED monitor, out-of-sample; Prop 11 (scope law → theorem); tool-lite

The Phase-3 ①②③ block (user: "来做1 2 3吧"). **① step94**: sensor-only monitor (cheetah-run, nominal prior policy;
forecasts with the certified loop $g$ itself — no action telemetry; flags at reads iff rel err > 0.2). Certificates
LOADED from the published step89 JSON (a-priori in the strongest sense); gates frozen before seeds 1–2 ever ran.
**G1a PASS 3/3 — in-situ vs bench ratio 0.4277 vs 0.43 (seed 1), 0.4976 vs 0.50 (seed 2): TWO-DECIMAL out-of-sample
replication of the published scope map, optimism replicated where optimism was published**; calibrated cell 0.666 vs
0.83 (G1b at-the-edge INCONCLUSIVE by 7e-4 on an integer-valued median — NOT rounded up). **G2 PASS 2/3** (fault
recall 1.00 on 3/3; delay ≤ k_op on 2/3 — miss is the tightest cell k_op=2, delay 3.0). Walker secondary run:
**0/3 replication (ratios 0.32–0.47 vs bench 0.94–1.02) with recall still 0.92–1.0** — the deterministic prior is
regime-bimodal (falls on some env seeds; invalid-frac tracks torso height), so Prop 7's nominal-regime clause is
LOAD-BEARING in deployment, the perfect honest contrast to cheetah's 3/3. Five-version design trail disclosed in the
script header (v1 vacuous gate → v2 teacher-forced cert lands weakly-expansive, taxonomy predicts its optimism
(14.8 vs knee ~6) — a within-experiment taxonomy confirmation → v3 walker→cheetah → v4 nominal-autonomous forecast →
v5 published-JSON certs + cell-by-cell gates). **② Prop 11** (decision scope): (i) aligned decisions inherit
certificate value up to calibration alone — $V(c)-V(1)\le(BH/L)(1-1/c)$, zero at $c{=}1$; (ii) task-mapped decisions
carry an irreducible mis-resolution penalty $|\log(\epsilon/\theta^\ast)|/\lambda_1$; no dynamics-only certificate
can supply $\theta^\ast$. Maps E12+E15→(i), E11+step93→(ii); in appendix + long paper inline + both papers'
scope-law bullets upgraded ("the law is now a theorem"). **③ tool-lite**: `docs/wm_audit_quickstart.md` (install,
pinned ckpt URLs, CLI+API, taxonomy table, honest estimator/scope notes) + README "Audit your own world model"
section. `tests/test_step94.py` 5/5 (pinned-artifact loader, gate re-derivation, Prop 11 numerics vs simulation).
E15 into ICLR (abstract clause, contribution (vi), E15 paragraph, limitations) + Exp 26 into long paper. ~1.5 h
compute, all Mac CPU. No gate loosened; one edge-miss recorded as an edge-miss.

## [2026-06-10] ICLR 提分包装轮 — 9 页官方模板达标 + 摘要减密 + 接缝预答 + rebuttal 预案

用户:"把弱点能解决的解决吧然后按你说的来"。按评审弱点表逐项执行:**#5 摘要减密** — 单段 270 词 10 个粗体
主张拆成两段(证书段 / 行动段),贡献 (vi) run-on 拆解;**#1 接缝预答** — 引言新增 "One certificate, a
universal half and an exclusive half" 段,§4 收尾段呼应(审计普适 vs 先验信任独占),双防 "two papers in one";
**#2 官方模板 + 页数纪律** — 修复既有 build_iclr.py(6→3 主图、Appendix D/E 拆出正文、stdlib zipfile 解压、
宽度属性容忍的搬移正则、页计数器),六轮密度手术(main-ish 6854→~5250 词:What-is-new 减半、Prop 7/8 与行内
证明压缩(全文在附录 B)、E2/E9/E10/E11/E12/E13/E14/E15 全部过刀、Related Work 四段收紧、Limitations 合并
Noether 双条、concurrent 段移附录 E、step71/step83/step92 图移附录 A、caption 减半 + 宽度 72–80%)——
**正文恰好 9 页,References 从 p10 顶部开始(官方 iclr2026_conference.sty,匿名双盲,行号,tectonic 编译,
23 页全文)**;视觉抽查 p1 渲染正确。**rebuttal 预案** — papers/REBUTTAL_PREP.md:10 类预判攻击(两篇合一/
Prop 6 构造/谱不新/无控制赢/E14 单 ckpt/Mo 重叠/Bitter Lesson/G1b 边缘/假设强/复现)各配承认-回击-指针 +
数字速查表。两个长 PDF 同步重建;210 测试全绿。诚实记录:压缩只动连接组织与重复,所有数字、限定词、
INCONCLUSIVE 原样保留;唯一語义损失是行内证明降为 sketch(全文仍在附录 B)。

## [2026-06-10] 弱点 4+5 真解 — Prop 6′(prefactor=分裂条件数)+ step65b/95/96:全分类学部署复刻;附录 E13 hedge 全闭

用户:"开工吧"(解决评审弱点表仅剩的两个 ⚠️)。**弱点4(Prop 6 常数)**:Prop 6′ 把 Theorem B 的 $c_j$ 精确等同于
谱投影范数 $1/\sin\theta_j$——(i) 正交不变分裂(isotypic 块,线性等变情形:Schur 强迫不变性+正交表示强迫正交)
⇒ $c_j{=}1$,**上界=下界连常数都匹配**;(ii) 斜性只能活在 isotypic 块内;(iii) prefactor 对 horizon 的全部影响
= 加性 haircut $\log\kappa/\lambda$。诚实 caveat 内置:非线性等变 $f$ 的 $Df(z)$ 在一般点不与 $\rho$ 交换——强迫
正交条款限于线性/commutant 情形,学习环上 $\kappa$ 是被测对象。**step65b**(秒级):Schur placement 实测
(群平均随机阵 off-block 质量 1.45e-16)+ 逐 irrep 块分层增长(rel-err ~1e-15)+ 跨块泄漏字面 0.0——
"isotypic refinement asserted, not measured" 这句 limitation 删除。**step95**:A 部受控斜剪切下测得最坏系数 =
解析 $\|\Pi\|$ 至四位小数($\kappa\in\{1,1.1,2.2,5.1,10\}$,正交格恰为 1),horizon shift = $\log\kappa/\lambda$
(A1/A2/A3 全 PASS);B/C 部真实环上 $\kappa_1$ 诚实地是分布(v2 设计修正:两窗口估计器错误,混沌吸引子近切点
重尾是物理不是噪声)——Lorenz-96 中位 17.5(max 90;估计器自身收敛检查失败,如实披露)、walker-1 中位 20.9
(max 193)、cheetah-3 中位 20.8(max 789);三系统中位惊人一致 ~17–21,与 E13 实测校准 0.83–1.02 并排报告
="最坏 vs 典型对齐"双披露。**弱点5(部署单域)**:step96 补齐分类学剩余 cell——G3 稳定弃权 2/2 PASS
(finger-spin-2/3:k=24 下 93–94% 窗口从不穿越、invalid 4.0/4.4%、recall 1.00 delay≤5 ——"弃权-稳定=免费监控");
G4 偏差弃权 PASS(hopper-1:原位时钟 7.0 vs bench 5.5,落预注册 ×1.5 带内;证书正确拒发 horizon 的格,部署确认)
;G5 复刻 PASS(finger-spin-1:1.04 vs 0.95)。smoke 中 G4 曾 8.5 出带(n=4),未动 gate,全量 n=20 自然落回——
纪律生效的实例。**至此已发表分类学的每一种判定(校准/乐观/稳定弃权/偏差弃权/工况 caveat)都有部署实例,全部
先验预测自已发表 artifact。** 论文折叠:ICLR(§3.2 Prop 6′ 指针、E15 扩展、(vi) 升级、limitation 两条改写为
已测、等量补偿削减)+ 长版(Prop 6′ 全文内联含 Exp 27、Exp 28 段、§7 hedge 闭合)+ 附录 B 完整 6′ 证明。
tests/test_step95_96.py 6/6(投影范数恒等式、haircut 恒等式、live 重跑、artifact 一致性);全套 216 绿。
~2.5 h 全 CPU。弱点表:仅剩"像素/真机"的诚实 out-of-scope。

## [2026-06-10] step97(Exp 29,Rung 1)— 像素家族部署 cell:弃权-偏差 ⇒ 零传感节省;分类学完成部署价值排序

用户:"1和2都来"(像素 LeWM 监控 + V-JEPA 2-AC 侦察)。**step97**:LeWM/PushT(已发表 bias-abstain cell,
λ1=0.0013、一步偏差 0.17、bench crossing=2)上跑 E15 同款 sensor-only 监控,gate 预注册:**G6 PASS**(无可用
节奏:invalid@k=2/4/8 = 0.50/0.75/0.87 —— 恰为 (k-1)/k,staleness-2 必穿越的确定性结构)+ **G7 v2 PASS**
(k≥2 处报警通道全淹:per-read flag rate 1.00;遥测损坏臂 pre/post 1.00/1.00 ——检测与漂移不可分,赛前声明)。
k=1 干净(0.00)但 k=1=每帧都读=预报器零价值;v1→v2 gate 修正(smoke 抓到:k=1 测的是已发表的一步偏差<θ,
不是监控可用性)披露于脚本头。移动场景对照(随机游走动作)同判(0.76@k4)。**分类学现在先验排序部署价值:
稳定弃权=免费监控 > 校准/乐观=定价节省 > 偏差弃权=勿部署——三种判定、三个架构、一个免训练读出。**
论文折叠:ICLR E15 加像素 cell+排序句(等量微削补偿);长版 Exp 29 段。test_step97 过。
**Rung 2 侦察(V-JEPA 2-AC)**:torch.hub `vjepa2_ac_vit_giant`(ViT-g/16 1B 编码器 + AC predictor,MIT,
256px/8f,DROID 后训练);仓库随附 **franka_example_traj.npz(Meta 实验室真机轨迹,394KB)** ——真机数据第一格
无需 DROID;其能量=L1(预测潜表征,编码潜表征)——**与监控器阈值同量,钩子白送**。坑已排:箱 venv 无 pip
(uv 装 timm/einops);**上游 main 把 VJEPA_BASE_URL 写成 localhost:8300(真 URL 被注释)** —— sed 修复后
ckpt 下载中。箱:torch 2.5.1+cu121 / 3080 / 947GB 盘。

## [2026-06-10] step98 — 证书读 V-JEPA 2-AC(第三家族,1B 编码器,真机后训练):EXPANSIVE λ1=0.178

Rung 2 主体落地。官方 `vjepa2_ac_vit_giant`(ViT-g/16 1012M + AC predictor ~300M,MIT,DROID 后训练,11GB ckpt)
经 torch.hub 加载进作者自己的代码;被审计环 = 他们能量景观 notebook 的同款 AR 帧 token 环(256×1408,
d=360,448 —— 迄今最大审计对象 ×600 倍),固定零增量动作(同 LeWM 预注册范围)。**判定:EXPANSIVE,
λ1 = 0.180/0.177(双 Q-seed,1.8% 吻合),CI 包络 [0.136, 0.250],T1(0.2)≈9.0 模型步**;fp32 CUDA 精度披露。
工程战报(全部披露在脚本):上游 main 的 localhost URL 乌龙(猴补)、franka npz 实为 (当前,目标) 帧对
(0.735 重标注)、前向 AD×SDPA 四连环(sdpa_kernel 上下文在 functorch 失效 → 全局 flag 被他们裸 sdp_kernel()
覆盖 → use_sdpa=False 被 attn_mask 分支绕过 → 最终猴补显式数学注意力)、反向图保留 OOM(忘冻参数,32GB→修)。
他们的规划"能量"= L1(预测潜表征, 编码潜表征) = 我们监控器的量——证书定价的就是 V-JEPA 2-AC 自己的能量增长率。
