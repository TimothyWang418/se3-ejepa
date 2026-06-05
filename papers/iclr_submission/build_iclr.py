#!/usr/bin/env python3
r"""Build the **ICLR 2026 submission** version of the focused horizon paper.

Single source: ``papers/iclr_certified_horizons.md``. Wraps it in the **official ICLR 2026 conference style**
(``iclr2026_conference.sty`` from github.com/ICLR/Master-Template) for an anonymous double-blind submission.
Pandoc (default xelatex template, so its helper macros \tightlist/\pandocbounded are defined) converts the body;
we inject the ICLR ``.sty`` + a fontspec Times-clone (TeX Gyre Termes, Unicode-safe — the official ``times`` package
is 8-bit mathptmx and chokes on §/í/ú/– under XeTeX) via a header-include preamble. Sections keep their manual
numbers (``secnumdepth=-1``) so every in-text ``§3.2`` / ``E2`` reference stays valid.

Run:    .venv/bin/python papers/iclr_submission/build_iclr.py
Output: papers/iclr_submission/main.tex  + main.pdf (compiled with tectonic)
"""

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE.parent
SRC_MD = PAPERS / "iclr_certified_horizons.md"
SRC_FIGURES = PAPERS / "figures"
OLD_ARXIV = PAPERS / "arxiv"                       # ships the Termes OTFs
ICLR_KIT = HERE / "iclr2026"                       # unzipped official template
COMBINED = HERE / "_iclr_combined.md"
PREAMBLE = HERE / "preamble_iclr.tex"
MAIN_TEX = HERE / "main.tex"
FIGDIR = HERE / "figures"
TITLE = "Certified Predictability Horizons for Equivariant World Models"
FONT_BASENAMES = ["texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf",
                  "texgyretermes-bolditalic.otf", "texgyretermes-math.otf"]


def ensure_template() -> None:
    r"""Download the official ICLR 2026 author kit (github.com/ICLR/Master-Template) if absent. Idempotent."""
    sty = ICLR_KIT / "iclr2026_conference.sty"
    if sty.exists():
        return
    zip_path = HERE / "iclr2026.zip"
    print("downloading official ICLR 2026 template ...")
    subprocess.run(["curl", "-sL", "-o", str(zip_path),
                    "https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip"], check=True)
    subprocess.run(["unzip", "-oq", str(zip_path), "-d", str(HERE)], check=True)
    assert sty.exists(), "ICLR template download/unzip failed"


def split_source() -> tuple[str, str]:
    r"""Return (abstract_md, body_md): abstract = text between '## Abstract' and the following '---';
    body = from '## 1. Introduction' to end. Title comes from the constant above."""
    md = SRC_MD.read_text(encoding="utf-8")
    md = re.sub(r"\A---\n.*?\n---\n", "", md, count=1, flags=re.DOTALL)   # strip YAML frontmatter
    m = re.search(r"## Abstract\s*\n(.*?)\n\s*---\s*\n", md, flags=re.DOTALL)
    assert m, "could not isolate the abstract block"
    abstract = m.group(1).strip()
    idx = md.find("## 1. Introduction")
    assert idx != -1, "could not find '## 1. Introduction'"
    body = md[idx:]
    assert "[[" not in body and "[[" not in abstract, "stale wikilink leaked"
    return abstract, body


# Keep only the load-bearing figures inline (the at-a-glance picture, the tightness proof = "central claim",
# the Lorenz headline, and the real-contact-dynamics result); relocate the rest to an unlimited appendix so the
# MAIN TEXT meets ICLR's strict 9-page limit. (Appendix + references do not count toward the 9 pages.)
MAIN_FIGS = ["hero_certified_region", "step65_horizon_tightness", "step70_lorenz_horizon", "step59_pusht_certificate"]


def relocate_figures(body: str) -> str:
    r"""Move every embedded figure whose basename is NOT in MAIN_FIGS into an appendix after the references; leave a
    one-line pointer where it stood. Returns the rewritten body (with an appended '## A  Additional figures')."""
    fig_re = re.compile(r"^!\[(?P<cap>.*?)\]\(figures/(?P<name>[\w.-]+)\.(?:png|pdf|jpe?g)\)\s*$", re.MULTILINE)
    moved: list[str] = []

    def repl(m: re.Match) -> str:
        stem = m.group("name")
        if any(stem.startswith(k) for k in MAIN_FIGS):
            return m.group(0)                                  # keep inline
        n = len(moved) + 1
        moved.append(f"![Figure A{n}. {m.group('cap')}](figures/{stem}.png)")
        return f"*(Figure A{n}, Appendix A.)*"                 # inline pointer

    new_body = fig_re.sub(repl, body)
    if moved:
        appendix = ("\n\n## A. Additional figures\n\n"
                    "Supporting figures relocated from the main text (the main-text figures are the hero schematic, "
                    "the tightness construction, the Lorenz horizon lift, and the real-contact-dynamics certificate).\n\n"
                    + "\n\n".join(moved) + "\n")
        new_body = new_body + appendix
    print(f"figures: {len(MAIN_FIGS)} kept inline, {len(moved)} relocated to Appendix A")
    return new_body


def build_combined(abstract: str, body: str) -> None:
    r"""YAML frontmatter (title + abstract block scalar) + body. Pandoc renders title/abstract from metadata so the
    abstract goes inside ICLR's \begin{abstract} with math/markdown intact."""
    body = relocate_figures(body)
    ab_indented = "\n".join("  " + ln for ln in abstract.splitlines())
    fm = f'---\ntitle: "{TITLE}"\nauthor: "Anonymous authors"\nabstract: |\n{ab_indented}\n---\n\n'
    COMBINED.write_text(fm + body, encoding="utf-8")
    print(f"_iclr_combined.md: {(fm + body).count(chr(10)) + 1} lines")


def write_preamble() -> None:
    PREAMBLE.write_text(
        # official ICLR 2026 conference style (controls geometry, headings, running header, line numbers)
        "\\usepackage{iclr2026_conference}\n"
        # Times-clone via fontspec (pandoc already loaded fontspec under xelatex); Unicode-safe unlike `times`.
        "\\setmainfont{texgyretermes}[Path=./, Extension=.otf, "
        "UprightFont=*-regular, BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]\n"
        "\\setmathfont{texgyretermes-math.otf}[Path=./]\n"
        "\\setcounter{secnumdepth}{-1}\n"          # keep manual heading numbers; in-text §3.2/E2 stay valid
        "\\renewcommand{\\_}{\\textunderscore\\allowbreak}\n",
        encoding="utf-8",
    )
    print("preamble_iclr.tex written")


def run_pandoc() -> None:
    subprocess.run([
        "pandoc", str(COMBINED), "--standalone", "-o", str(MAIN_TEX),
        "-H", str(PREAMBLE),
        "--pdf-engine=tectonic",
    ], check=True)
    tex = MAIN_TEX.read_text(encoding="utf-8")
    # ICLR anonymous double-blind author block
    tex = re.sub(r"\\author\{Anonymous authors\}",
                 "\\\\author{Anonymous authors \\\\\\\\ Paper under double-blind review}", tex, count=1)
    MAIN_TEX.write_text(tex, encoding="utf-8")
    print(f"main.tex: {tex.count(chr(10)) + 1} lines")


def copy_assets() -> None:
    # ICLR style files (only the .sty is strictly needed; tectonic auto-fetches fancyhdr/natbib/eso-pic)
    for name in ["iclr2026_conference.sty", "iclr2026_conference.bst"]:
        shutil.copy2(ICLR_KIT / name, HERE / name)
    # Termes OTFs
    for name in FONT_BASENAMES:
        if not (HERE / name).exists():
            shutil.copy2(OLD_ARXIV / name, HERE / name)
    # figures referenced by the body
    body = COMBINED.read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"figures/([\w.-]+\.(?:png|pdf|jpg|jpeg))", body)))
    if FIGDIR.exists():
        shutil.rmtree(FIGDIR)
    FIGDIR.mkdir(parents=True)
    missing = [n for n in refs if not (SRC_FIGURES / n).exists()]
    assert not missing, f"missing figures: {missing}"
    for n in refs:
        shutil.copy2(SRC_FIGURES / n, FIGDIR / n)
    print(f"assets: ICLR .sty/.bst + {len(FONT_BASENAMES)} Termes OTFs + {len(refs)} figures")


def compile_pdf() -> None:
    try:
        subprocess.run(["tectonic", MAIN_TEX.name], cwd=HERE, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        tail = (getattr(e, "stderr", "") or str(e)).strip().splitlines()
        print("  [tectonic FAILED]\n   " + "\n   ".join(tail[-25:]))
        return
    pdf = HERE / "main.pdf"
    print(f"main.pdf: compiled, {pdf.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    ensure_template()
    ab, body = split_source()
    build_combined(ab, body)
    write_preamble()
    run_pandoc()
    copy_assets()
    compile_pdf()
