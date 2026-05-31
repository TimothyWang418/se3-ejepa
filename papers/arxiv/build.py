#!/usr/bin/env python3
"""Reproducible build: two Markdown papers -> one arXiv LaTeX source.

Pipeline
--------
1. Concatenate the focused core write-up (main body) with the full results log
   (Steps 3--38, appendix), separated by a hard ``\\clearpage``.
2. Rewrite the cross-references that used to point at a *separate* file
   ``geometric_payoff.md`` so they now point at the in-document appendix.
3. Keep the recurring Chinese idiom 举一反三 as Chinese glyphs (the project's
   nickname for "zero-shot generalisation across the symmetry group"), adding a
   one-time parenthetical gloss at first use: pinyin + an English rendering. The
   CJK glyphs are typeset with ``xeCJK`` + the TeXLive-bundled Fandol font, so
   the source compiles with **XeLaTeX** (tectonic locally; arXiv auto-detects
   xelatex from the fontspec/xeCJK load).
4. Run pandoc to emit a standalone ``main.tex``.
5. Copy exactly the figures the combined document embeds into ``arxiv/figures/``.
6. Pack ``main.tex`` + ``00README.json`` + ``figures/`` into the upload tarball.

Run from papers/:  python3 arxiv/build.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # papers/arxiv
PAPERS = HERE.parent                             # papers
CORE = PAPERS / "equivariance_generalization_core.md"
PAYOFF = PAPERS / "geometric_payoff.md"
SRC_FIGURES = PAPERS / "figures"                 # papers/figures (source of truth)
COMBINED = HERE / "_combined.md"
MAIN_TEX = HERE / "main.tex"
PREAMBLE = HERE / "preamble_extra.tex"
README_JSON = HERE / "00README.json"             # arXiv engine selector
ARXIV_FIGURES = HERE / "figures"                 # papers/arxiv/figures (shipped)
TARBALL = HERE / "arxiv_upload.tar.gz"           # ready-to-upload source bundle

TITLE = ("Exact equivariance, kept through training, buys zero-shot "
         "generalisation across the symmetry group")

# --- idiom handling -------------------------------------------------------
# We KEEP 举一反三 in Chinese throughout (typeset via xeCJK + Fandol). Only the
# very first occurrence (core abstract) gets a one-time parenthetical gloss:
# pinyin + an English rendering.
IDIOM = "举一反三"
GLOSS_FROM = f"({IDIOM})"
GLOSS_TO = f"({IDIOM}; *jǔ yī fǎn sān*, “from one example, infer the rest”)"


def annotate_idiom(text: str) -> str:
    """Gloss the first ``(举一反三)`` once; leave every other occurrence as-is."""
    return text.replace(GLOSS_FROM, GLOSS_TO, 1)


def _assert_no_math_digit_bug(text: str) -> None:
    """Fail loudly if any inline-math span *closes* on a ``$`` followed by a
    digit -- pandoc would not treat that ``$`` as a closer, leaking the math
    into text mode. Skips fenced/inline code and escaped ``\\$``."""
    i, n, in_math, in_code = 0, len(text), False, False
    while i < n:
        if text[i:i + 3] == "```":
            in_code = not in_code; i += 3; continue
        if in_code:
            i += 1; continue
        c = text[i]
        if c == "`":
            j = text.find("`", i + 1)
            i = (j + 1) if j != -1 else n; continue
        if c == "\\":
            i += 2; continue
        if c == "$":
            if text[i:i + 2] == "$$":
                in_math = not in_math; i += 2; continue
            if in_math and i + 1 < n and text[i + 1].isdigit():
                ctx = text[max(0, i - 40):i + 15]
                raise AssertionError(f"math closes before a digit: ...{ctx!r}")
            in_math = not in_math
        i += 1


def build_combined() -> None:
    core = CORE.read_text(encoding="utf-8")
    payoff = PAYOFF.read_text(encoding="utf-8")

    # --- core: drop the H1 (title comes from pandoc metadata) ---
    core_lines = core.split("\n")
    assert core_lines[0].startswith("# "), "core line 1 is not an H1"
    core_body = "\n".join(core_lines[1:])

    # --- core: retarget the old separate-file references at the appendix ---
    # (a) abstract blurb: "...the full results log\n> [[geometric_payoff.md]] and..."
    core_body = core_body.replace(
        "the full results log\n> [[geometric_payoff.md]]",
        "the full results log in the appendix",
    )
    # (b) "Full treatment...: `geometric_payoff.md` §17–§17.1."
    core_body = core_body.replace("`geometric_payoff.md` §17–§17.1", "the appendix")
    # (c) any remaining bare wikilink -> "the appendix"
    core_body = core_body.replace("[[geometric_payoff.md]]", "the appendix")

    # --- payoff: becomes the appendix; prefix its H1 ---
    payoff_lines = payoff.split("\n")
    assert payoff_lines[0].startswith("# "), "payoff line 1 is not an H1"
    payoff_lines[0] = "# Appendix --- " + payoff_lines[0][2:]
    payoff_body = "\n".join(payoff_lines)
    payoff_body = payoff_body.replace("[[geometric_payoff.md]]", "the full results log")

    # --- stitch with a hard page break (raw LaTeX block) ---
    combined = core_body + "\n\n```{=latex}\n\\clearpage\n```\n\n" + payoff_body

    # --- fix a pandoc-incompatible inline-math fragment ---------------------
    # Pandoc does not close inline math on a ``$`` that is immediately followed
    # by a digit (so it won't read "$20,000" as math). The source has one such
    # spot, ``$\sim$2–4$\times$`` -> the closing ``$`` after \sim precedes "2",
    # so \sim leaks into text mode. Merge it into a single math span. (KaTeX in
    # Obsidian is lenient and renders the original fine, which is why it slipped
    # through; only the pandoc/LaTeX path needs this.)
    combined = combined.replace(r"($\sim$2–4$\times$)", r"($\sim 2\text{–}4\times$)")

    # --- gloss the idiom's first occurrence; keep all Chinese glyphs ---
    combined = annotate_idiom(combined)

    _assert_no_math_digit_bug(combined)

    COMBINED.write_text(combined, encoding="utf-8")
    # Extra preamble (pandoc -H). This build targets **XeLaTeX** (the xeCJK load
    # below requires it; tectonic is XeTeX-based, and arXiv auto-detects xelatex
    # from the fontspec/xeCJK packages):
    #  * manual section numbers in the prose are authoritative -> kill auto-numbering
    #  * a handful of Unicode symbols appear in *prose* (text mode); Latin Modern
    #    lacks them, so map each to its math equivalent via newunicodechar. (→ is
    #    NOT mapped: Latin Modern renders it fine.)
    #  * xeCJK + Fandol typesets the 举一反三 glyphs. Fandol is bundled in TeXLive,
    #    so tectonic fetches it locally and arXiv's server already has it -> the
    #    same font on both ends, no install. BoldFont is needed because some
    #    occurrences of 举一反三 sit inside **bold** prose.
    PREAMBLE.write_text(
        "\\setcounter{secnumdepth}{-1}\n"
        "\\usepackage{newunicodechar}\n"
        "\\newunicodechar{≈}{\\ensuremath{\\approx}}\n"
        "\\newunicodechar{↔}{\\ensuremath{\\leftrightarrow}}\n"
        "\\newunicodechar{⇒}{\\ensuremath{\\Rightarrow}}\n"
        "\\newunicodechar{✓}{\\ensuremath{\\checkmark}}\n"
        "\\usepackage{xeCJK}\n"
        "\\setCJKmainfont{FandolSong-Regular.otf}"
        "[BoldFont=FandolSong-Bold.otf, AutoFakeSlant=0.15]\n",
        encoding="utf-8",
    )

    # sanity: the idiom is kept (CJK present), glossed once, no stale wikilinks
    n_cjk = sum(1 for c in combined if "㐀" <= c <= "鿿")
    n_idiom = combined.count(IDIOM)
    assert n_cjk > 0, "expected 举一反三 glyphs, found none"
    assert "jǔ yī fǎn sān" in combined, "first-use pinyin gloss missing"
    assert "[[" not in combined, "stale wikilink remains"
    print(f"_combined.md: {combined.count(chr(10)) + 1} lines, "
          f"{n_idiom} 举一反三 uses ({n_cjk} CJK glyphs), 1 glossed")


def run_pandoc() -> None:
    cmd = [
        "pandoc", str(COMBINED), "--standalone", "-o", str(MAIN_TEX),
        "-H", str(PREAMBLE),
        "--metadata", f"title={TITLE}",
        "--metadata", "author=Hongbo Wang",
        "--metadata", "date=",
    ]
    subprocess.run(cmd, check=True)

    # pandoc escapes backslashes passed via --metadata, so inject the
    # affiliation (which needs a real ``\\`` line break) here instead.
    tex = MAIN_TEX.read_text(encoding="utf-8")
    tex = tex.replace(
        "\\author{Hongbo Wang}",
        "\\author{Hongbo Wang \\\\\n  \\small Department of Mathematics, "
        "Stony Brook University, Stony Brook, NY 11794, USA}",
        1,
    )
    MAIN_TEX.write_text(tex, encoding="utf-8")
    print(f"main.tex: {tex.count(chr(10)) + 1} lines")


def write_arxiv_readme() -> None:
    """Emit ``00README.json`` so arXiv compiles with XeLaTeX explicitly.

    The document loads ``fontspec``/``xeCJK``, which already makes arXiv's
    AutoTeX pick a Unicode engine -- but pinning the compiler removes all doubt
    (arXiv added an explicit ``xelatex`` option in Nov 2025). Schema per
    https://info.arxiv.org/help/00README.html .
    """
    readme = {
        "spec_version": 1,
        "process": {"compiler": "xelatex"},
        "sources": [{"filename": MAIN_TEX.name, "usage": "toplevel"}],
    }
    README_JSON.write_text(json.dumps(readme, indent=2) + "\n", encoding="utf-8")
    print(f"{README_JSON.name}: pinned compiler=xelatex")


def copy_figures() -> list[str]:
    """Copy *exactly* the figures the combined document embeds.

    Parses ``_combined.md`` for ``figures/<name>`` paths (the markdown embeds and
    the pandoc-emitted ``\\includegraphics`` both use this prefix), de-dupes, and
    mirrors that set from ``papers/figures/`` into ``papers/arxiv/figures/``. The
    destination is wiped first so a shrinking figure set never leaves stale files
    in the upload bundle. Returns the sorted list of basenames shipped.
    """
    combined = COMBINED.read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"figures/([\w.-]+\.(?:png|pdf|jpg|jpeg))", combined)))
    if ARXIV_FIGURES.exists():
        shutil.rmtree(ARXIV_FIGURES)
    ARXIV_FIGURES.mkdir(parents=True)
    missing = []
    for name in refs:
        src = SRC_FIGURES / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, ARXIV_FIGURES / name)
    assert not missing, f"referenced figures absent from papers/figures/: {missing}"
    print(f"figures/: copied {len(refs)} embedded figures -> {ARXIV_FIGURES}")
    return refs


def make_tarball(figures: list[str]) -> None:
    """Pack the arXiv source bundle: ``main.tex`` + ``00README.json`` + figures.

    ``preamble_extra.tex`` is *not* shipped -- pandoc's ``-H`` already inlined its
    content into ``main.tex``'s preamble, so it would be dead weight. Arcnames are
    repo-root-relative (``main.tex``, ``figures/<name>``) so arXiv sees a flat
    top-level ``main.tex`` exactly as ``00README.json`` declares.
    """
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(README_JSON, arcname=README_JSON.name)
        tar.add(MAIN_TEX, arcname=MAIN_TEX.name)
        for name in figures:
            tar.add(ARXIV_FIGURES / name, arcname=f"figures/{name}")
    size_kb = TARBALL.stat().st_size / 1024
    print(f"{TARBALL.name}: {2 + len(figures)} members, {size_kb:.0f} KiB")


if __name__ == "__main__":
    build_combined()
    run_pandoc()
    write_arxiv_readme()
    figs = copy_figures()
    make_tarball(figs)
