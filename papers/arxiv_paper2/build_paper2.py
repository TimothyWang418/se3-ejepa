#!/usr/bin/env python3
r"""Build the **paper2** arXiv source bundle (separate from the old-paper arxiv/ build).

Single source: ``papers/paper2_certified_world_models.md``. Mirrors ``papers/arxiv/build.py`` (XeLaTeX target;
ships Termes + Fandol OTFs by filename so 推背图 / 举一反三 render identically locally and on arXiv) but for one
document. Strips the FOR-REVIEW block (meant to be deleted before submission). Does NOT touch the old paper's
arxiv/ frozen bundles.

Run:  .venv/bin/python papers/arxiv_paper2/build_paper2.py
Then compile a local QA PDF:  cd papers/arxiv_paper2 && tectonic main.tex
Upload:  papers/arxiv_paper2/arxiv_paper2_upload.tar.gz
"""

import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE.parent
SRC_MD = PAPERS / "paper2_certified_world_models.md"
SRC_FIGURES = PAPERS / "figures"
OLD_ARXIV = PAPERS / "arxiv"                       # reuse its shipped OTFs (already located there)
COMBINED = HERE / "_paper2_combined.md"
MAIN_TEX = HERE / "main.tex"
PREAMBLE = HERE / "preamble_extra.tex"
README_JSON = HERE / "00README.json"
ARXIV_FIGURES = HERE / "figures"
TARBALL = HERE / "arxiv_paper2_upload.tar.gz"
TITLE = "A Predictability Certificate for Equivariant World Models"

FONT_BASENAMES = ["texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf",
                  "texgyretermes-bolditalic.otf", "texgyretermes-math.otf"]
CJK_FONT_BASENAMES = ["FandolSong-Regular.otf", "FandolSong-Bold.otf"]


def build_combined() -> None:
    r"""Read the paper2 draft, drop the YAML frontmatter + the FOR-REVIEW blockquote, keep everything from the
    Abstract onward. (Title/subtitle are re-supplied via pandoc metadata.)"""
    md = SRC_MD.read_text(encoding="utf-8")
    # strip leading YAML frontmatter
    md = re.sub(r"\A---\n.*?\n---\n", "", md, count=1, flags=re.DOTALL)
    # drop everything before "## Abstract" (the FOR-REVIEW blockquote)
    idx = md.find("## Abstract")
    assert idx != -1, "could not find '## Abstract' anchor"
    body = md[idx:]
    assert "FOR REVIEW" not in body, "FOR-REVIEW block leaked into the bundle"
    assert "[[" not in body, "stale wikilink remains"
    COMBINED.write_text(body, encoding="utf-8")
    n_cjk = sum(1 for c in body if "㐀" <= c <= "鿿")
    print(f"_paper2_combined.md: {body.count(chr(10)) + 1} lines, {n_cjk} CJK glyphs")


def write_preamble() -> None:
    PREAMBLE.write_text(
        "\\usepackage[letterpaper,margin=1in]{geometry}\n"
        "\\setmainfont{texgyretermes}[Path=./, Extension=.otf, "
        "UprightFont=*-regular, BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]\n"
        "\\setmathfont{texgyretermes-math.otf}[Path=./]\n"
        # NB: no \fvset here — paper2 has no fenced code blocks, so pandoc does not load fancyvrb.
        "\\setcounter{secnumdepth}{-1}\n"
        "\\renewcommand{\\_}{\\textunderscore\\allowbreak}\n"
        "\\usepackage{newunicodechar}\n"
        "\\newunicodechar{≈}{\\ensuremath{\\approx}}\n"
        "\\newunicodechar{⇒}{\\ensuremath{\\Rightarrow}}\n"
        "\\newunicodechar{→}{\\ensuremath{\\rightarrow}}\n"
        "\\newunicodechar{×}{\\ensuremath{\\times}}\n"
        "\\newunicodechar{≪}{\\ensuremath{\\ll}}\n"
        "\\newunicodechar{≤}{\\ensuremath{\\leq}}\n"
        "\\newunicodechar{≥}{\\ensuremath{\\geq}}\n"
        "\\newunicodechar{∝}{\\ensuremath{\\propto}}\n"
        "\\newunicodechar{⊆}{\\ensuremath{\\subseteq}}\n"
        "\\newunicodechar{∀}{\\ensuremath{\\forall}}\n"
        "\\newunicodechar{✓}{\\ensuremath{\\checkmark}}\n"
        "\\newunicodechar{✅}{\\ensuremath{\\checkmark}}\n"
        "\\newunicodechar{↗}{\\ensuremath{\\nearrow}}\n"
        "\\newunicodechar{⊕}{\\ensuremath{\\oplus}}\n"
        "\\newunicodechar{⊗}{\\ensuremath{\\otimes}}\n"
        "\\newunicodechar{ℓ}{\\ensuremath{\\ell}}\n"
        "\\newunicodechar{⚠}{\\ensuremath{\\triangle}}\n"   # only reached by the review-copy FOR-REVIEW block
        "\\usepackage{xeCJK}\n"
        "\\setCJKmainfont{FandolSong-Regular.otf}[Path=./, BoldFont=FandolSong-Bold.otf, AutoFakeSlant=0.15]\n",
        encoding="utf-8",
    )


def run_pandoc() -> None:
    subprocess.run([
        "pandoc", str(COMBINED), "--standalone", "-o", str(MAIN_TEX),
        "-H", str(PREAMBLE),
        "--metadata", f"title={TITLE}",
        "--metadata", "subtitle=Scale buys interpolation, structure buys a certificate — "
                      "across configuration, horizon, and resolution",
        "--metadata", "author=Hongbo Wang",
        "--metadata", "date=",
    ], check=True)
    tex = MAIN_TEX.read_text(encoding="utf-8")
    tex = tex.replace(
        "\\author{Hongbo Wang}",
        "\\author{Hongbo Wang \\\\\n  \\small Department of Mathematics, "
        "Stony Brook University, Stony Brook, NY 11794, USA}", 1)
    MAIN_TEX.write_text(tex, encoding="utf-8")
    print(f"main.tex: {tex.count(chr(10)) + 1} lines")


def build_review_pdf() -> None:
    r"""Also emit the **tracked** standalone review copy ``papers/paper2_certified_world_models.pdf`` from the SAME
    single source — but KEEPING the FOR-REVIEW block that the arXiv bundle strips. Obsidian wikilinks are flattened
    to plain text and the ⚠️ variation selector is dropped so the block typesets cleanly. Soft-fails (warns + skips)
    if tectonic is unavailable, so the arXiv-bundle build never depends on a local LaTeX engine."""
    review_md = HERE / "_paper2_review.md"
    review_tex = HERE / "review.tex"
    review_pdf = HERE / "review.pdf"
    standalone = PAPERS / "paper2_certified_world_models.pdf"

    md = SRC_MD.read_text(encoding="utf-8")
    md = re.sub(r"\A---\n.*?\n---\n", "", md, count=1, flags=re.DOTALL)   # strip YAML, KEEP FOR-REVIEW + body
    assert "FOR REVIEW" in md, "review copy must retain the FOR-REVIEW block"
    md = md.replace("️", "")                                         # drop emoji variation selector (no glyph)
    # flatten Obsidian wikilinks [[path/name.md]] -> name (basename, no .md) so they read cleanly on paper
    md = re.sub(r"\[\[([^\]]+)\]\]", lambda m: m.group(1).rsplit("/", 1)[-1].removesuffix(".md"), md)
    review_md.write_text(md, encoding="utf-8")

    subprocess.run([
        "pandoc", str(review_md), "--standalone", "-o", str(review_tex), "-H", str(PREAMBLE),
        "--metadata", f"title={TITLE}",
        "--metadata", "subtitle=Scale buys interpolation, structure buys a certificate — "
                      "across configuration, horizon, and resolution",
        "--metadata", "author=Hongbo Wang", "--metadata", "date=",
    ], check=True)
    review_tex.write_text(review_tex.read_text(encoding="utf-8").replace(
        "\\author{Hongbo Wang}",
        "\\author{Hongbo Wang \\\\\n  \\small Department of Mathematics, "
        "Stony Brook University, Stony Brook, NY 11794, USA}", 1), encoding="utf-8")

    try:
        subprocess.run(["tectonic", review_tex.name], cwd=HERE, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        tail = (getattr(e, "stderr", "") or str(e)).strip().splitlines()
        print(f"  [skip] standalone review PDF not rebuilt (tectonic unavailable/failed): {tail[-1] if tail else e}")
        for f in (review_md, review_tex):
            f.unlink(missing_ok=True)
        return
    shutil.copy2(review_pdf, standalone)
    print(f"{standalone.name}: review copy rebuilt (FOR-REVIEW kept), {standalone.stat().st_size / 1024:.0f} KiB")
    for f in (review_md, review_tex, review_pdf, HERE / "review.log", HERE / "review.xdv"):
        f.unlink(missing_ok=True)


def write_arxiv_readme() -> None:
    README_JSON.write_text(json.dumps(
        {"spec_version": 1, "process": {"compiler": "xelatex"},
         "sources": [{"filename": MAIN_TEX.name, "usage": "toplevel"}]}, indent=2) + "\n", encoding="utf-8")
    print(f"{README_JSON.name}: pinned compiler=xelatex")


def copy_figures() -> list[str]:
    body = COMBINED.read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"figures/([\w.-]+\.(?:png|pdf|jpg|jpeg))", body)))
    if ARXIV_FIGURES.exists():
        shutil.rmtree(ARXIV_FIGURES)
    ARXIV_FIGURES.mkdir(parents=True)
    missing = []
    for name in refs:
        src = SRC_FIGURES / name
        (missing.append(name) if not src.exists() else shutil.copy2(src, ARXIV_FIGURES / name))
    assert not missing, f"referenced figures absent from papers/figures/: {missing}"
    print(f"figures/: copied {len(refs)} embedded figures -> {ARXIV_FIGURES}")
    return refs


def copy_fonts() -> list[str]:
    wanted = FONT_BASENAMES + CJK_FONT_BASENAMES
    missing = []
    for name in wanted:
        dst = HERE / name
        if dst.exists():
            continue
        src = OLD_ARXIV / name                     # the old build already located/shipped these
        (shutil.copy2(src, dst) if src.exists() else missing.append(name))
    assert not missing, f"OTFs not found in {OLD_ARXIV}: {missing} (run the old arxiv build first)"
    print(f"fonts/: shipped {len(FONT_BASENAMES)} Termes + {len(CJK_FONT_BASENAMES)} Fandol OTFs -> {HERE}")
    return wanted


def make_tarball(figures: list[str]) -> None:
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
    n = 2 + len(figures) + len(fonts)
    print(f"{TARBALL.name}: {n} members, {TARBALL.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    build_combined()
    write_preamble()
    run_pandoc()
    write_arxiv_readme()
    figs = copy_figures()
    copy_fonts()
    make_tarball(figs)
    build_review_pdf()   # also refresh the tracked standalone review PDF from the same single source
