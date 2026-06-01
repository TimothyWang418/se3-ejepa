#!/usr/bin/env python3
"""Reproducible build: three Markdown documents -> one arXiv LaTeX source.

Pipeline
--------
1. Concatenate three parts, each separated by a hard ``\\clearpage``:
   the focused core write-up (main body), the full results log (Steps 3--38,
   "Appendix"), and the forward-looking equivariant-LeJEPA theory note
   ("Supplement").
2. Rewrite the cross-references that used to point at a *separate* file
   ``geometric_payoff.md`` so they now point at the in-document appendix.
3. Keep the recurring Chinese idiom 举一反三 as Chinese glyphs (the project's
   nickname for "zero-shot generalisation across the symmetry group"), adding a
   one-time parenthetical gloss at first use: pinyin + an English rendering. The
   CJK glyphs are typeset with ``xeCJK`` + the Fandol font, which is *shipped in
   the bundle* and loaded by filename (``Path=./``) -- self-contained, with no
   dependency on arXiv's server having the font. The source compiles with
   **XeLaTeX** (tectonic locally; arXiv auto-detects xelatex from the
   fontspec/xeCJK load).
4. Run pandoc to emit a standalone ``main.tex``.
5. Copy exactly the figures the combined document embeds into ``arxiv/figures/``.
6. Pack ``main.tex`` + ``00README.json`` + ``figures/`` + the Termes & Fandol
   OTFs into the upload tarball.

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
LEJEPA = PAPERS / "equivariant_lejepa.md"   # forward-looking theory supplement
SRC_FIGURES = PAPERS / "figures"                 # papers/figures (source of truth)
COMBINED = HERE / "_combined.md"
MAIN_TEX = HERE / "main.tex"
PREAMBLE = HERE / "preamble_extra.tex"
README_JSON = HERE / "00README.json"             # arXiv engine selector
ARXIV_FIGURES = HERE / "figures"                 # papers/arxiv/figures (shipped)
TARBALL = HERE / "arxiv_upload.tar.gz"           # ready-to-upload source bundle

# --- fonts shipped with the bundle ---------------------------------------
# We typeset in TeX Gyre Termes (a Times clone -> the NeurIPS/JEPA look). The
# OTFs are *shipped in the bundle* and loaded by filename, not by family name:
#  * tectonic resolves fonts from its own bundle by filename and does NOT query
#    the system fontconfig, so ``\setmainfont{TeX Gyre Termes}`` (by family)
#    fails locally even though TeXLive has the font.
#  * shipping the files makes rendering byte-identical locally and on arXiv.
# These live in TeXLive's tex-gyre / tex-gyre-math trees; we locate them once
# at build time and mirror them next to main.tex.
FONT_BASENAMES = [
    "texgyretermes-regular.otf",
    "texgyretermes-bold.otf",
    "texgyretermes-italic.otf",
    "texgyretermes-bolditalic.otf",
    "texgyretermes-math.otf",
]

# --- CJK font shipped with the bundle ------------------------------------
# The idiom 举一反三 is set in Fandol Song. We *ship* the OTFs and load them by
# filename via Path=./ (see the xeCJK line in the preamble) for the same reason
# as Termes above: it makes the upload self-contained, so the compile never
# depends on arXiv's server (or tectonic's bundle) happening to carry Fandol.
# These two faces (Regular + Bold) are all the document needs -- Bold is loaded
# because some 举一反三 occurrences sit inside **bold** prose. The OTFs live in
# TeXLive's fandol tree; copy_fonts() locates them the same way as Termes.
CJK_FONT_BASENAMES = [
    "FandolSong-Regular.otf",
    "FandolSong-Bold.otf",
]

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
    lejepa = LEJEPA.read_text(encoding="utf-8")

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

    # --- lejepa: becomes a forward-looking supplement; prefix its H1 ---
    # This is the third part: a theory note connecting the program to the very
    # recent LeJEPA identifiability line (arXiv:2511.08544 / 2605.26379). Labelled
    # "Supplement" (not "Appendix B"): the payoff is the unlettered results-log
    # "Appendix", and the core carries its own internal reproducibility "Appendix
    # A", so a lettered scheme would collide. Its §1--10 are manual numbers that
    # coexist with the core's §1--6 and payoff's §0--31 under secnumdepth=-1.
    lejepa_lines = lejepa.split("\n")
    assert lejepa_lines[0].startswith("# "), "lejepa line 1 is not an H1"
    lejepa_lines[0] = "# Supplement --- " + lejepa_lines[0][2:]
    lejepa_body = "\n".join(lejepa_lines)
    # Retarget the Sources line's separate-file names at their in-document homes.
    # The "(core paper §4)" prose refs (in §3/§7 bodies) are left as-is: the core
    # *is* the main body here, so "core paper §4" reads correctly and keeps them
    # distinct from the supplement's own §4.
    lejepa_body = lejepa_body.replace(
        "`equivariance_generalization_core.md` (flatness theorem, §4), `geometric_payoff.md`",
        "the core paper (flatness theorem, §4) and the appendix",
    )

    # --- stitch the three parts with hard page breaks (raw LaTeX blocks) ---
    pagebreak = "\n\n```{=latex}\n\\clearpage\n```\n\n"
    combined = core_body + pagebreak + payoff_body + pagebreak + lejepa_body

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
    #  * a handful of Unicode symbols appear in *prose* (text mode); the text font
    #    may lack them, so map each to its math equivalent via newunicodechar
    #    (including → now that we typeset in TeX Gyre Termes).
    #  * xeCJK + Fandol typesets the 举一反三 glyphs. The Fandol OTFs are *shipped*
    #    and loaded by filename via Path=./ (like Termes), so the same font is
    #    used locally and on arXiv with no dependency on either side carrying
    #    Fandol. BoldFont is needed because some 举一反三 occurrences sit inside
    #    **bold** prose.
    PREAMBLE.write_text(
        # JEPA/NeurIPS-style layout + fonts. The pandoc template already loads
        # fontspec+unicode-math under XeLaTeX; we only widen the column and pin
        # the font families. Fonts load BY FILENAME from the bundle dir (Path=./)
        # so tectonic (no fontconfig) and arXiv render identically.
        "\\usepackage[letterpaper,margin=1in]{geometry}\n"
        "\\setmainfont{texgyretermes}["
        "Path=./, Extension=.otf, "
        "UprightFont=*-regular, BoldFont=*-bold, "
        "ItalicFont=*-italic, BoldItalicFont=*-bolditalic]\n"
        "\\setmathfont{texgyretermes-math.otf}[Path=./]\n"
        "\\fvset{fontsize=\\small}\n"
        "\\setcounter{secnumdepth}{-1}\n"
        # Long snake_case identifiers in \texttt{} (e.g.
        # test_step26_optimizer_equivariance) overflow narrow table columns
        # because \_ is non-breaking; make each underscore a legal break point.
        "\\renewcommand{\\_}{\\textunderscore\\allowbreak}\n"
        "\\usepackage{newunicodechar}\n"
        "\\newunicodechar{≈}{\\ensuremath{\\approx}}\n"
        "\\newunicodechar{↔}{\\ensuremath{\\leftrightarrow}}\n"
        "\\newunicodechar{⇒}{\\ensuremath{\\Rightarrow}}\n"
        "\\newunicodechar{→}{\\ensuremath{\\rightarrow}}\n"
        "\\newunicodechar{✓}{\\ensuremath{\\checkmark}}\n"
        # NB: the prime U+2032 ("Prop. 1$'$", "[B/A$'$]") is NOT mapped here -- it
        # is written as an explicit math prime in the source. unicode-math claims
        # U+2032 as a math symbol at \begin{document}, overriding any text-mode
        # newunicodechar binding, so a mapping would lose and error in text mode.
        "\\usepackage{xeCJK}\n"
        "\\setCJKmainfont{FandolSong-Regular.otf}"
        "[Path=./, BoldFont=FandolSong-Bold.otf, AutoFakeSlant=0.15]\n",
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


def copy_fonts() -> list[str]:
    """Locate the Termes + Fandol OTFs and mirror them next to ``main.tex``.

    tectonic resolves fonts from its bundle by filename and ignores system
    fontconfig, so both the Times-clone body families *and* the Fandol CJK font
    must *travel in the build dir* and be loaded by filename (see the preamble's
    ``Path=./`` font specs). We find each OTF under the local TeXLive opentype
    trees -- with a few system font dirs as a fallback -- and copy it into
    ``papers/arxiv/``. Idempotent: a file already present is left in place.
    Returns the basenames shipped.
    """
    wanted = FONT_BASENAMES + CJK_FONT_BASENAMES
    roots = sorted(Path("/usr/local/texlive").glob(
        "*/texmf-dist/fonts/opentype/public"))
    roots += [Path("/Library/Fonts"), Path.home() / "Library" / "Fonts",
              Path("/System/Library/Fonts")]
    missing = []
    for name in wanted:
        dst = HERE / name
        if dst.exists():
            continue  # already shipped -> skip the search (idempotent + fast)
        found = next(
            (h for root in roots if root.exists()
             for h in root.rglob(name)),
            None,
        )
        if found is None:
            missing.append(name)
            continue
        shutil.copy2(found, dst)
    assert not missing, (
        f"required OTFs not found on this machine: {missing}. "
        "Install the tex-gyre / tex-gyre-math / fandol TeXLive packages, or copy "
        "the OTFs into papers/arxiv/ manually."
    )
    print(f"fonts/: shipped {len(FONT_BASENAMES)} Termes + "
          f"{len(CJK_FONT_BASENAMES)} Fandol OTFs -> {HERE}")
    return wanted


def make_tarball(figures: list[str]) -> None:
    """Pack the arXiv source bundle: ``main.tex`` + ``00README.json`` + figures
    + the Termes & Fandol OTFs.

    ``preamble_extra.tex`` is *not* shipped -- pandoc's ``-H`` already inlined its
    content into ``main.tex``'s preamble, so it would be dead weight. Arcnames are
    repo-root-relative (``main.tex``, ``figures/<name>``, ``<font>.otf``) so arXiv
    sees a flat top-level ``main.tex`` -- with the fonts alongside it, matching the
    preamble's ``Path=./`` -- exactly as ``00README.json`` declares.
    """
    fonts = FONT_BASENAMES + CJK_FONT_BASENAMES
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(README_JSON, arcname=README_JSON.name)
        tar.add(MAIN_TEX, arcname=MAIN_TEX.name)
        for name in figures:
            tar.add(ARXIV_FIGURES / name, arcname=f"figures/{name}")
        for name in fonts:
            tar.add(HERE / name, arcname=name)
    size_kb = TARBALL.stat().st_size / 1024
    n = 2 + len(figures) + len(fonts)
    print(f"{TARBALL.name}: {n} members, {size_kb:.0f} KiB")


if __name__ == "__main__":
    build_combined()
    run_pandoc()
    write_arxiv_readme()
    figs = copy_figures()
    copy_fonts()
    make_tarball(figs)
