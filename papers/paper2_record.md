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

Unit-test isolation: `tests/conftest.py` pins float32 around every test; float64 experiments opt in. Full suite **passes** (`test_step61.py` adds the SO(2) pose-planner equivariance + model-rollout orbit-invariance guards).
Multi-seed reproducibility: `experiments/aggregate_seeds.py` re-runs steps 50/51/52/53/57/58/59/60/61 at seeds {0,1,2} and commits per-seed `papers/figures/step5*_seeds.json` + `step59/60/61_*_seeds.json`; every range quoted above and in the draft is the seed min–max from those files. Single-seed `step5*.json/.png` stay canonical at the default seed 0.

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
- **T3 — pixel latent** ⏳ (S1). · **T4 — hinge slow=conserved as a theorem** ⏳ (hard). · **T5 — hero figure +
  positioning table + one-command repro** ⏳.
