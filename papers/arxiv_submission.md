# arXiv submission prep

Working notes + ready-to-use metadata for putting this work on arXiv. **Nothing here is
submitted automatically** — the items under "Decisions needed" require your input.

---

## Recommendation: submit the *core* paper

Submit **`equivariance_generalization_core.md`** (the focused [A]+[B]+[C] write-up), **not**
`geometric_payoff.md` (the 27-step internal results log — too long and log-shaped for a venue/arXiv
paper). The payoff log can live in the repo and be cited as "full results log" / supplementary.

The core paper is conversion-friendly: 1067 lines, 21 sections, **3 figures** (`killer_figure`,
`where_the_bet_pays`, `step23_indist_largeN` — all in `papers/figures/`, the first two also as PDF),
and only **2 wikilinks** (both pointing to `geometric_payoff.md`). All math is standard `$...$` /
`$$...$$`, which pandoc converts natively.

---

## arXiv abstract (plain-text field, trimmed to fit the ~1920-char limit)

> A latent world model built from an equivariant encoder and an equivariant predictor inherits a
> provable symmetry of its training loss: when the world's dynamics carries a group $G$ acting on
> latents by an orthogonal representation, one-step prediction error is exactly invariant across the
> whole group, so fitting on a thin slice of orientations mathematically determines the model on the
> entire orbit (zero-shot generalisation across the group). We verify this end-to-end at laptop scale
> (CPU/MPS, fully seeded), in both 2D SO(2) on real PushT and 3D SE(3) on point clouds. [A] The
> learned symmetry survives a real Muon/AdamW + EMA + VICReg training run -- the composed
> encode-predict residual is ~1e-6 after optimisation, not just at initialisation. [B] One-step error
> is flat to five digits across the group (OOD/seen ratio 1.00x), a theorem, while a same-hypothesis-
> class non-equivariant baseline fits the seen slice but breaks out-of-distribution (13.8x in 2D
> latent, 17.2x in 3D, 157x over the full SE(3) ladder) -- with the equivariant model 4.5-7.4x smaller
> and frequently better in-distribution. [C] Under a matching equivariant planner the closed-loop
> control error is orientation-invariant: float-floor-exact in 2D (paired bootstrap) and statistically
> flat in 3D SE(3). We are explicit about scope: everything is laptop-scale and silent on whether scale
> eventually beats the prior, and the across-group payoff is not an in-distribution sample-efficiency
> edge.

(The in-paper abstract is the fuller 287-word version; this one is trimmed for the arXiv metadata
field, which truncates around 1920 characters.)

---

## Suggested metadata

- **Title:** *Exact equivariance, kept through training, buys zero-shot generalisation across the
  symmetry group* (matches the paper's H1; consider a shorter variant if a venue caps title length).
- **Primary category:** `cs.LG` (Machine Learning).
- **Cross-list:** `cs.AI`, `cs.RO` (closed-loop PushT control), `stat.ML`. Optionally `cs.CV` (3D
  point clouds).
- **Comments:** e.g. `Laptop-scale (CPU/MPS), fully seeded and deterministic; 3 figures. Code: <repo URL>. Full 27-step results log included as supplementary.`
- **MSC / ACM (optional):** ACM `I.2.6` (Learning), `I.2.9` (Robotics); MSC `68T07`.

---

## Conversion path (Markdown -> LaTeX -> arXiv)

Neither `pandoc` nor a LaTeX engine is currently installed. arXiv compiles submitted **LaTeX source**
on its end, so a local LaTeX engine is only needed to proof-compile.

```bash
# 1. install the converter (one-time)
brew install pandoc
#    optional, to proof-compile locally before submitting:
brew install tectonic            # lightweight, or: brew install --cask mactex-no-gui

# 2. convert the core paper to standalone LaTeX
cd ~/Workspace/se3-ejepa/papers
pandoc equivariance_generalization_core.md \
  -o arxiv/main.tex --standalone \
  --metadata title="Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group"

# 3. proof-compile (optional)
cd arxiv && tectonic main.tex      # or: pdflatex main.tex

# 4. submit main.tex + the 3 figures from papers/figures/ to arXiv
#    (arxiv accepts PNG; killer_figure.pdf / where_the_bet_pays.pdf also exist)
```

**Hand-cleanup after pandoc (small, ~15 min):**
- The 2 `[[geometric_payoff.md]]` wikilinks -> plain text "the full results log" or a footnote (pandoc
  leaves them as literal `[[...]]`).
- Confirm the 3 `\includegraphics` paths resolve (copy the 3 figures next to `main.tex`).
- Spot-check that dense inline math (e.g. `$\mathrm{SE}(3)\rtimes S_O$`, `$\rho(g)$`) survived.
- Add `\author{...}` and `\date{}` (pandoc leaves these empty).

---

## Decisions needed (yours)

1. **Which paper** — core paper as the submission, payoff log as supplementary? (recommended)
2. **Authorship** — name(s) + affiliation as you want them listed on arXiv.
3. **Primary category** — `cs.LG` recommended; confirm cross-lists.
4. **Conversion path** — install pandoc and convert to LaTeX (arXiv-preferred), or submit a PDF?
5. **License** — arXiv default (perpetual, non-exclusive) vs. CC BY 4.0 (more open).
6. **Code release** — is the repo going public (so the "Code:" URL in Comments resolves)? Add a
   `LICENSE` file if so.

---

*Prepared 2026-05-30. Update the repo URL / author block before submitting.*
