# Session summary — paper2's three axes + first draft (2026-06-02, continued)

> Follows `MORNING_SUMMARY_2026-06-02.md`. You said **"1. push 2. paper2 只管推进就好"** — so I pushed and
> then advanced paper2 autonomously. **Everything below is committed AND pushed** (you'd already approved push
> for the paper2 line). All gates are honest (a run prints `INCONCLUSIVE` rather than loosen a threshold).

## TL;DR
paper2 went from *"keystone result + proposal"* to **all three axes empirically anchored + a full first draft**.
Four new decisive experiments (Steps 49–52), three unit tests, a test-isolation fix (81 tests green together),
every result folded into the proposal, and a complete paper draft assembled.

## The four new experiments (each 3 seeds unless noted, each with a figure + JSON)

| # | What it nails | Headline result |
|---|---|---|
| **49** | Config axis, *exponential* | Train on **6 generators** of $\mathbb{Z}_2^6$ (I Ching) → certified over **all 64** compositions to machine zero ($\sim10^{-33}$); MLP degrades $1.6\text{e-}5\to0.59$. *(1 seed; deterministic.)* |
| **50** | The **Noether hinge** | On $\mathrm{SO}(2)$ central force: conserved $(E,L)$ land in the invariant block ($R^2$ **0.92–0.99** vs $\le0.01$); slowest invariant mode $\approx0.01\ll0.145$ equivariant; invariant subspace group-residual **$10^{-16}$** (∀SO(2)) vs MLP **1.17**. slow ⊆ invariant. |
| **51** | **Structure vs scale** (the tagline) | Wedge-train / full-circle-test: equivariant **flat** over orbit (ratio **1.1–1.2**); $84\times$-scaled MLP buys in-wedge interpolation ($30\text{–}166\times$, *beats* equiv in-dist) but climbs **170–2700×** out-of-wedge, stays **10–155×** above the equiv floor. |
| **52** | Horizon × resolution (Thm B, 推背图) | Learned model recovers chaotic **$\hat\lambda=0.69$ to 0.1%**; certified horizon $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ (slope **1.3–1.6 ≈ 1/λ**); chaotic detail 3–10 steps, slow/invariant ≥90 (full rollout). |

Figures: `papers/figures/step{49,50,51,52}_*.png`.

## Tests + isolation
- New unit tests: `test_step50_noether_hinge.py`, `test_step52_horizon_resolution.py` (architectural halves need no training, like the Thm-A test).
- **Found + fixed a real test-isolation bug**: Steps 50/51/52 use `float64`; `torch.set_default_dtype` is global+persistent, so a float64 test was leaving float64 active and breaking the next float32 test (`float != double` in `rotate_points`). Added `tests/conftest.py` (autouse reset to float32; float64 tests opt in). **Full suite: 81 passed together** (was 3 failing as one session).

## The draft
`papers/paper2_certified_world_models.md` — **"A Predictability Certificate for Equivariant World Models"**:
abstract, 3-axis master theorem (closure lemma + Thm A + spectral Thm B + certified-region corollary), the
Noether hinge §3, an Experiments section with all 4 figures + multi-seed numbers, related-work delta (the
$[A]/[B]/[C]$ device is the $|w|{=}1$ single-resolution corner; BRo/UWM/LeJEPA/NWP positioning), honest
Limitations, conclusion, reproducibility table. **Additive** — the three v1/v2 papers and frozen arXiv bundles
are untouched; **not wired into the arXiv build** (a separate explicit step).

## Proposal status (`papers/proposals/paper2-certified-compositional-jepa.md`)
- **P0 ✅** master theorem backbone (closure test + Thm A + Thm B spectrum + hinge measured). *Remaining: tighten §2–3 proof prose.*
- **P1 ✅** config axis (Steps 47 + 49).
- **P2 ✅** hinge (Step 50) + $T_j(\epsilon)$ staircase (Step 52). *Remaining: lift to approximate-symmetry embodied model.*
- **P3 ✅** structure vs scale (Step 51). *Remaining: replicate scale-invariance on the discrete config axis.*
- **P4 / P5** not started (approximate-symmetry degradation; discovery+generation).

## Commits (all pushed to `main`)
`204a0a5` Step 49 · `14cf87a` Step 50 + hinge test · `3be4fd0` Step 51 · `620845b` Step 52 ·
`217ed0d` Step 52 test + conftest dtype fix · `1b56ba8` paper2 draft.

## ⟶ Your decisions / next moves
1. **Read the draft** `papers/paper2_certified_world_models.md` — the FOR-REVIEW block flags every judgment call
   (title, empirical scope, hinge framing, relationship to the old paper). **Title** is the main one to confirm.
2. **P4 — approximate symmetry** (task #153): the #1 reviewer attack. Tunable equivariance-error knob → measure
   certificate degradation vs Theorem B's $\epsilon_{\max}$ term; and lift the hinge to the contact-rich embodied
   model. The key remaining honesty experiment — I can do this next on your word.
3. **Wire the draft into the arXiv build** (task #154) — only after you sign off on title + framing; preserves
   frozen v1/v2.
4. **arXiv v2** of the *old* paper still awaits your manual upload (`arxiv_upload_v2.tar.gz`) — unchanged from
   this session; I can't submit for you.

## Honest open hinges (flagged in the draft, unchanged)
- The Noether hinge is **measured & confirmed on a controlled $\mathrm{SO}(2)$ system**, with the honest
  non-converse (invariant $\not\Rightarrow$ slow). Lifting it to *approximate* symmetry is open (P4).
- All five experiments are **toy/1-GPU-scale** proofs of principle — a *mechanism* paper, stated openly.
