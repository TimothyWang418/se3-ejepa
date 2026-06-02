# arXiv submission prep

Working notes + ready-to-use metadata for putting this work on arXiv. The structural decisions are
now **made** (recorded below); only **two free-text inputs** remain (author block + repo URL), after
which the Markdown -> LaTeX conversion can run.

---

## Plan (decided): submit *the three documents as one*

One arXiv submission, one PDF, stitched by `arxiv/build.py` (core → payoff appendix → LeJEPA
supplement, each part separated by a hard `\clearpage`):

- **Main body** = `equivariance_generalization_core.md` — the focused [A]+[B]+[C] write-up (the
  reviewable claim).
- **Appendix** = `geometric_payoff.md` — the full results log, attached under `\appendix` (the
  complete evidence trail).
- **Supplement** = `equivariant_lejepa.md` — the forward-looking equivariant-LeJEPA theory note: the
  symmetry-structured identifiability line (block-isotropy + gauge refinement, an equivariant
  predictor/dynamics, the $S_n\times\mathrm{SO}(3)$ compositional product group, and $G$-invariant
  planning — research **Directions 1–4**). It **does ship in the bundle**, labelled "Supplement" (not
  "Appendix B"), with its own manual §1–§10 numbering coexisting with the core's §1–§6 and the payoff's
  §0–§31 under `secnumdepth=-1`.

Why one document and not core-only-with-a-supplement: the appendix and supplement then live *inside*
the timestamped arXiv record (better for priority), while a reader still meets the core claim first and
only descends into the full results log / theory note if they want the receipts.

**Scope: what is in v1 vs. v2.** The frozen upload bundle (`arxiv/arxiv_upload.tar.gz`, the one
submitted manually) is the **Steps 3–38** build: core main body + payoff appendix (Steps 3–38) + the
LeJEPA supplement (Directions 1–4). Since that bundle was cut, the *source* program has advanced to
**Step 46** (Step 42 tensor-product-message ladder, Step 43 encoder-capacity / lossless-oracle ladder,
Step 44 encoder-output-budget confirmation, Step 46 multi-head equivariant attention-pool cure) and absorbed the review-driven revisions; those are **v2**,
regenerated from source via `arxiv/build.py` *without* disturbing the frozen v1 tarball. So **v1 =
Steps 3–38 + Directions 1–4**; **Steps 42–46 + the review fixes = v2**. (v1 ships the full
`equivariance_generalization_core.md` as the main body; the slimmer `main_compact.md` is an additive
draft — a candidate v2 main text, not wired into `build.py` today.)

Combined size: core 1408 lines / 3 figure embeds + 2 cross-references; payoff 3327 lines / 11 figure
embeds / 0 wikilinks; the LeJEPA supplement adds **no new figure files**. **Fifteen unique figure files total** (`killer_figure`, `where_the_bet_pays`,
`step23_indist_largeN`, `step24_object_interaction`, `step32_tp_degree_ladder`,
`step33_symmetry_discovery`, `step34_active_inference_noisy`, `step35_many_body`,
`step36_discover_exploit`, `step37_active_inference_search`, `step38_latent_goal_reaching`,
`step42_tp_message_ladder`, `step43_encoder_ladder`, `step44_encoder_output_budget`,
`step45_augmented_vs_exact`, `step46_pooling_cure` — all in
`papers/figures/` as PNG; `killer_figure`, `where_the_bet_pays`, `step24_object_interaction` also as
PDF). The core's 3 embeds (`killer_figure`, `where_the_bet_pays`, `step23_indist_largeN`) are all a
subset of the payoff's set, so the union is 15 distinct files. All math is standard `$...$` / `$$...$$`,
which pandoc converts natively.

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
> latent, 17.2x in 3D, and 157x for a raw-coordinate baseline over the full SE(3) rotation+translation
> ladder, an extrapolation regime) -- with the equivariant model 4.5-7.4x smaller,
> though with no in-distribution edge (a wash-to-loss at scale). [C] Under a matching equivariant planner the closed-loop
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
- **Authors:** Hongbo Wang (Department of Mathematics, Stony Brook University, Stony Brook, NY 11794, USA).
- **Primary category:** `cs.LG` (Machine Learning).
- **Cross-list:** `cs.AI`, `cs.RO` (closed-loop PushT control), `stat.ML`, and `math.RT`
  (Representation Theory) — the LeJEPA supplement's identifiability result is a Schur's-lemma /
  orthogonal-commutant statement, a genuine `math.RT` fit now that the supplement ships. Optionally
  `cs.CV` (3D point clouds).
- **License:** **CC BY 4.0** (decided — see "Licensing").
- **Comments:** e.g. `Laptop-scale (CPU/MPS), fully seeded and deterministic; 11 figures; 95 pp. Full results log (Steps 3-38) as an appendix; equivariant-LeJEPA identifiability theory note as a supplement. Code (Apache-2.0): https://github.com/TimothyWang418/se3-ejepa`
- **MSC / ACM (optional):** ACM `I.2.6` (Learning), `I.2.9` (Robotics); MSC `68T07`.

---

## Licensing (decided)

A deliberate split — permissive on both sides, but the right instrument for each artifact:

| Artifact | License | Why |
|---|---|---|
| **Paper** (this arXiv PDF + figures) | **CC BY 4.0** | Most open for prose/figures; only obligation on reusers is to credit you. Selected on the arXiv license screen. |
| **Code** (the repo) | **Apache 2.0** | Permissive like MIT but adds an explicit patent grant + contributor terms — a bit more protection. `LICENSE` (canonical Apache 2.0 text) is already in the repo root. |

---

## Priority / credit strategy (decided)

The principle: **maximise spread with permissive licenses, establish priority with a timestamped
public record, and convert that spread into credit via an attribution requirement.** Concretely:

1. **arXiv submission** — itself a dated, citable record; this is the primary priority anchor.
2. **Zenodo DOI** — archive the repo at a tagged release (`v0.1`) to mint a DOI, so the *software* is
   independently citable. (Connect the GitHub repo to Zenodo, then cut a release; Zenodo captures the
   tarball + assigns the DOI automatically.)
3. **Public commit history** — the repo goes public with its full history intact: an independent,
   third-party-timestamped trail of when each result landed.
4. **BibTeX in the README** — an explicit citation block (+ `CITATION.cff`) so anyone building on the
   work has a one-click way to cite it. Attribution-by-default is what turns reach into credit.

---

## Conversion path (Markdown -> LaTeX -> arXiv), via pandoc

> **Superseded by `arxiv/build.py`** (documented under "Inputs (now resolved)" below) — this is the
> original by-hand recipe, kept to show what the script automates. Two differences from what actually
> ships: (i) the build stitches **three** parts (core + payoff appendix + LeJEPA supplement), not two;
> (ii) it emits a **single self-contained `main.tex`** with the preamble inlined and the OTF fonts
> shipped alongside, not the three-file `core_body.tex`/`payoff_body.tex` split sketched below.

arXiv compiles the submitted **LaTeX source** on its end, so a local LaTeX engine is only needed to
proof-compile. For "all as one" we convert each Markdown file to a LaTeX *fragment* (no preamble) and
stitch them with a thin wrapper that puts the payoff log under `\appendix` and the LeJEPA note after it.

```bash
# 1. install the converter (one-time)
brew install pandoc
#    optional, to proof-compile locally before submitting:
brew install tectonic            # lightweight, or: brew install --cask mactex-no-gui

# 2. convert each paper to a LaTeX *fragment* (body only, no \documentclass)
cd ~/Workspace/se3-ejepa/papers
mkdir -p arxiv && cp figures/killer_figure.png figures/where_the_bet_pays.png \
  figures/step23_indist_largeN.png figures/step24_object_interaction.png \
  figures/step32_tp_degree_ladder.png figures/step33_symmetry_discovery.png \
  figures/step34_active_inference_noisy.png figures/step35_many_body.png \
  figures/step36_discover_exploit.png figures/step37_active_inference_search.png \
  figures/step38_latent_goal_reaching.png arxiv/
pandoc equivariance_generalization_core.md -o arxiv/core_body.tex
pandoc geometric_payoff.md                 -o arxiv/payoff_body.tex

# 3. write a thin arxiv/main.tex wrapper:
#    \documentclass{article}
#    \usepackage{amsmath,amssymb,graphicx,hyperref,cleveref}
#    \graphicspath{{./}}            % figures copied next to main.tex in step 2
#    \title{Exact equivariance, kept through training, buys zero-shot
#           generalisation across the symmetry group}
#    \author{Hongbo Wang \\ Department of Mathematics, Stony Brook University}
#    \date{}
#    \begin{document}\maketitle
#    \input{core_body}
#    \appendix
#    \section{Full results log}\label{app:payoff}   % build.py auto-titles this from the payoff H1
#    \input{payoff_body}
#    \end{document}

# 4. proof-compile (optional)
cd arxiv && tectonic main.tex      # or: pdflatex main.tex

# 5. submit main.tex + core_body.tex + payoff_body.tex + the 11 figures to arXiv
#    (arxiv accepts PNG; killer_figure / where_the_bet_pays / step24 also have PDF)
```

**Hand-cleanup after pandoc (still small — the appendix is cleaner than expected):**
- The 2 `[[geometric_payoff.md]]` cross-references in the core body (now pointing *at the appendix*) ->
  `\Cref{app:payoff}` / "see the full results log in the appendix" (pandoc leaves them literal
  `[[...]]`).
- The payoff log has **0 wikilinks** and **11 figure embeds**; the core has **3**. `killer_figure`,
  `where_the_bet_pays`, `step23_indist_largeN` appear in both -> they render twice (body + appendix);
  leave it, or drop one.
- Confirm all `\includegraphics` paths resolve (step 2 copies all 11 figures next to `main.tex`).
- Spot-check that dense inline math (e.g. `$\mathrm{SE}(3)\rtimes S_O$`, `$\rho(g)$`,
  `$\mathbf 1\otimes\mathbf 1\to\mathbf 1$`) survived.
- `\author{...}` and `\date{}` are filled in the wrapper (pandoc would leave them empty).

---

## Inputs (now resolved)

Both free-text values are filled in across this doc, the README, `LICENSE`/copyright, and
`CITATION.cff`:

1. **Author block** — Hongbo Wang, Department of Mathematics, Stony Brook University, Stony Brook, NY
   11794, USA.
2. **Repo URL** — `https://github.com/TimothyWang418/se3-ejepa` (created **private + empty** to lock the
   address; flip to **public** at submission time, since arXiv's "Code:" link must resolve when v1 goes
   live). *Note:* the GitHub handle **`TimothyWang418`** is the author's own (Hongbo Wang), not a third
   party — the repo, the `LICENSE`/copyright header, and `CITATION.cff` all attribute to Hongbo Wang, so
   the handle/name difference is cosmetic, not an authorship question a reviewer should flag.

The only value still outstanding is the **arXiv ID itself** — it doesn't exist until the moment you
submit, and it back-fills the BibTeX `eprint` (currently `XXXX.XXXXX`) and the `CITATION.cff`
`preferred-citation`.

**Toolchain ready:** `pandoc 3.9.0.2` and `tectonic 0.16.9` are both installed — the conversion is
unblocked. The built bundle lives in `papers/arxiv/`. `build.py` regenerates the whole thing (combine ->
pandoc -> copy figures -> ship fonts -> tarball) in one command, run from `papers/`:

```bash
python3 arxiv/build.py        # writes _combined.md, main.tex, 00README.json, figures/, fonts, arxiv_upload.tar.gz
cd arxiv && tectonic main.tex  # proof-compile -> main.pdf (95 pp, XeLaTeX for the 举一反三 glyphs)
```

> **v1-freeze caveat.** `build.py` rewrites `arxiv_upload.tar.gz` as its last step. The v1 tarball is
> **frozen** (submitted manually, reused as-is), so for v2 prep regenerate `_combined.md`/`main.tex` for
> proofing but **preserve the existing v1 tarball** (e.g. `git restore` it afterward, or skip
> `make_tarball`). Do not let a routine rebuild silently replace the bundle that gets uploaded.

The ready-to-upload source tarball is `papers/arxiv/arxiv_upload.tar.gz` (**20 members**: `main.tex` +
`00README.json` + the 11 figures + **7 OTF fonts** — 5 TeX Gyre Termes faces + 2 Fandol Song faces,
shipped and loaded by filename via `Path=./` so the compile is self-contained on arXiv;
`preamble_extra.tex` is *not* shipped because pandoc's `-H` already inlined it into `main.tex`). Last
proof-compile: **95 pp**, ~1.8 MB, exit 0 (only cosmetic overfull-hbox warnings), Termes + Fandol fonts
embedded.

---

*Prepared 2026-05-30; refreshed 2026-05-31 (bundle built: Steps 3–38 + LeJEPA supplement, 11 unique
figures; pandoc + tectonic installed); de-staled 2026-06-01 (source advanced to Step 44 + Directions
1–4 and absorbed the review revisions — these are **v2**; **v1 = the frozen Steps-3–38 bundle**).
Decisions locked: three-documents-as-one, pandoc->LaTeX, paper CC BY 4.0, code Apache 2.0, repo public.
Author + repo URL filled. Bundle built (95 pp, 20-member tarball); user submits manually, reusing the
existing tarball.*
