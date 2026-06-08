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
