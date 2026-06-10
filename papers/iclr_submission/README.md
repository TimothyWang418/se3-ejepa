# ICLR 2026 submission build

Compiles `papers/iclr_certified_horizons.md` into the **official ICLR 2026 conference format**
(`iclr2026_conference.sty` from [github.com/ICLR/Master-Template](https://github.com/ICLR/Master-Template))
as an **anonymous double-blind submission** that fits the strict **9-page main-text limit**.

## Build

```bash
.venv/bin/python papers/iclr_submission/build_iclr.py
```

The script: (1) downloads the official ICLR 2026 author kit if absent; (2) splits the markdown into
title / abstract / body; (3) splits the in-source appendices (D, E) out of the main body and relocates all but three load-bearing figures into an unlimited Appendix A so the main text meets the strict 9-page submission limit (verified: References starts at the top of p.10); (4) runs pandoc (default xelatex template, so its `\tightlist`/`\pandocbounded`
helpers are defined) with a header-include preamble that loads the ICLR `.sty` and supplies a Unicode-safe
Times clone (TeX Gyre Termes) via `fontspec` — the official `times` package is 8-bit `mathptmx` and chokes on
`§ í ú – —` under XeTeX/tectonic; (5) compiles with `tectonic`. Output: `main.pdf`.

Sections keep their **manual numbers** (`secnumdepth=-1`), so every in-text `§3.2` / `E2` reference stays valid.

## Layout

- **Main text (≤ 9 pp):** §1 intro · §2 setup · §3 certificate (Thm A, Lem 1/2, Thm B, Prop 6/7, Noether Prop 4/5) ·
  §4 experiments E1–E8 · §5 related work · §6 limitations · §7 conclusion.
- **Inline figures (4):** hero schematic · Prop-6 tightness construction · **Lorenz horizon lift (E2)** ·
  real-contact-dynamics certificate (E5). The other five figures are in **Appendix A**.
- **References + Appendix:** unlimited, do not count toward the 9 pages.

## Before you submit

- **Year.** This targets **ICLR 2026** (the most recent official kit). For the next cycle, swap to
  `iclr2027_conference.*` in the kit and re-run — the `.sty` is stable year-to-year.
- **Anonymity.** The build is anonymous (`Anonymous authors / Paper under double-blind review`, line numbers on).
  For a camera-ready / arXiv version, uncomment `\iclrfinalcopy` and restore the author block.
- **Citations.** In-text citations are `natbib` author-year (`\citet`/`\citep`) and the bibliography is typeset by the
  official `iclr2026_conference.bst` from `papers/iclr_refs.bib`. The shared markdown uses pandoc `[@key]` syntax; the
  generic `make iclr-build` renders the same keys via `--citeproc`. **Verify the classic-reference details**
  (Eaton/Lehmann/Berger/Oseledets/Pilyugin/Tucker — venue/volume/pages) in `iclr_refs.bib` before a real submission;
  the arXiv-numbered entries are exact.
- This is a *generated* artifact; the single source of truth is `papers/iclr_certified_horizons.md`.
