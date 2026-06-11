#!/usr/bin/env python3
r"""Build the **non-anonymous arXiv v1** of the certified-horizons paper (flag-planting variant).

Differences from the double-blind submission build (which this wraps):
1. ``\iclrfinalcopy`` — real author block, no line numbers;
2. the ICLR running header "Published as a conference paper at ICLR 2026" is patched to "Preprint."
   (we must not claim publication or review status);
3. the Appendix-C sentence "Anonymized code accompanies the submission" becomes a public code link
   (REMINDER: the repo must be made public before the arXiv submission goes live);
4. output = ``arxiv_build/arxiv_iclr_upload.tar.gz`` — main.tex + main.bbl + patched .sty + .bst + .bib +
   00README.json (xelatex) + figures + Termes OTFs — the exact bundle arXiv's AutoTeX needs (fontspec => XeLaTeX,
   same recipe as the paper-1 bundle).

Run (box):  cd ~/se3-ejepa && python3 papers/iclr_submission/build_arxiv.py
Then: upload the tarball at arxiv.org/submit (primary cs.LG; suggested cross-lists cs.RO, math.DS).
"""
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE / "arxiv_build"
AUTHOR = r"Hongbo Wang\\ \texttt{whb591347285@gmail.com}"
CODE_URL = "https://github.com/TimothyWang418/se3-ejepa"
FONTS = ["texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf",
         "texgyretermes-bolditalic.otf", "texgyretermes-math.otf"]


def run_submission_build() -> None:
    subprocess.run([sys.executable, str(HERE / "build_iclr.py")], check=True)


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    # --- main.tex: de-anonymize + preprint footer-safe + code link ---
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    tex = tex.replace(r"\author{Anonymous authors \\ Paper under double-blind review}",
                      "\\iclrfinalcopy\n\\author{" + AUTHOR + "}")
    assert "Anonymous" not in tex, "anonymous block survived"
    n = tex.count("Anonymized code accompanies the submission")
    tex = tex.replace("Anonymized code accompanies the submission",
                      r"Code: \url{" + CODE_URL + "}. The anonymized artifact accompanies the conference submission")
    print(f"code-link patch applied x{n}")
    (STAGE / "main.tex").write_text(tex, encoding="utf-8")
    # --- style: "Published as a conference paper" -> "Preprint." ---
    sty = (HERE / "iclr2026_conference.sty").read_text(encoding="utf-8")
    n = sty.count("Published as a conference paper at ICLR 2026")
    sty = sty.replace("Published as a conference paper at ICLR 2026", "Preprint.")
    print(f"sty header patch applied x{n}")
    (STAGE / "iclr2026_conference.sty").write_text(sty, encoding="utf-8")
    # --- static assets ---
    shutil.copy2(HERE / "iclr2026_conference.bst", STAGE / "iclr2026_conference.bst")
    shutil.copy2(HERE / "iclr_refs.bib", STAGE / "iclr_refs.bib")
    for f in FONTS:
        shutil.copy2(HERE / f, STAGE / f)
    shutil.copytree(HERE / "figures", STAGE / "figures")
    (STAGE / "00README.json").write_text(json.dumps(
        {"spec_version": 1, "process": {"compiler": "xelatex"},
         "sources": [{"filename": "main.tex", "usage": "toplevel"}]}, indent=2))


def compile_and_check() -> None:
    subprocess.run(["tectonic", "--keep-intermediates", "main.tex"], cwd=STAGE, check=True,
                   capture_output=True, text=True)
    pdf = STAGE / "main.pdf"
    raw = pdf.read_bytes()
    print(f"main.pdf: {pdf.stat().st_size/1024:.0f} KiB")
    # text-level sanity via pdftotext-free heuristic: decompress check happens at upload; assert via tex instead
    tex = (STAGE / "main.tex").read_text(encoding="utf-8")
    assert "\\iclrfinalcopy" in tex and "Hongbo Wang" in tex
    assert "Anonymous" not in tex
    sty = (STAGE / "iclr2026_conference.sty").read_text(encoding="utf-8")
    assert "Published as a conference paper" not in sty
    print("sanity: finalcopy + real author + preprint header OK")


def pack() -> None:
    out = STAGE / "arxiv_iclr_upload.tar.gz"
    members = ["main.tex", "iclr2026_conference.sty", "iclr2026_conference.bst", "iclr_refs.bib",
               "00README.json"] + FONTS
    if (STAGE / "main.bbl").exists():
        members.append("main.bbl")
    with tarfile.open(out, "w:gz") as tar:
        for m in members:
            tar.add(STAGE / m, arcname=m)
        for fig in sorted((STAGE / "figures").iterdir()):
            tar.add(fig, arcname=f"figures/{fig.name}")
    print(f"bundle: {out.name} ({out.stat().st_size/1024:.0f} KiB, "
          f"{len(members) + len(list((STAGE/'figures').iterdir()))} members)")


if __name__ == "__main__":
    run_submission_build()
    stage()
    compile_and_check()
    pack()
    print("REMINDER: make the GitHub repo public BEFORE the arXiv listing goes live (the PDF links it).")
