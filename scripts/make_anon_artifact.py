r"""Build the anonymized reproducibility artifact for submission: dist/anon_artifact.zip.

Collects experiments/, tests/, docs/specs/ (pre-registration evidence), papers/figures/*.json (all result
artifacts) + the 5 main-text figures, scrubs identifying strings, writes ANON_README.md with exact repro commands,
and FAILS LOUDLY if any identifying token survives in the zip.

Run: .venv/bin/python scripts/make_anon_artifact.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "anon_artifact.zip"

# Sensitive literals are assembled from parts so this (public) source never contains them verbatim;
# extra site-local patterns (e.g. host addresses) can be appended via the untracked scripts/.scrub_local
# (one regex per line).
_HOST_IP = r"\.".join(("100", "71", "116", "12"))
_USR = "wh" + "b"
SCRUB = [(re.compile(p, re.I), r) for p, r in [
    (r"/Users/hongbowang[^\s'\")]*", "/ANON/path"),
    (r"hongbo\s*wang", "ANONYMOUS"),
    (r"hongbowang", "ANON"),
    (r"TimothyWang418", "ANON"),
    (_USR + r"@[\w.\-]+", "anon@anon"),
    (r"\b" + _USR + r"\b", "anon"),
    (_HOST_IP, "ANON-HOST"),
]]
_local = Path(__file__).with_name(".scrub_local")
if _local.exists():
    for ln in _local.read_text().splitlines():
        if ln.strip():
            SCRUB.append((re.compile(ln.strip(), re.I), "ANON-LOCAL"))
LEAK = re.compile(r"hongbo|timothywang|" + _USR + "@|" + _HOST_IP, re.I)

README = """# Anonymized artifact — Certified Predictability for Equivariant World Models

Code + result artifacts + pre-registered design specs for every experiment (E1–E13 / steps 52–90).

## Environment
python 3.11 · torch >= 2.5 (float64 CPU paths required) · numpy · matplotlib · gymnasium[classic_control]
· for E13: mujoco==3.3.2, dm-control==1.0.28; TD-MPC2 checkpoints are downloaded BY URL (not vendored):
https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/{task}-{seed}.pt -> models/tdmpc2/

## Reproduce (each writes papers/figures/*.json and prints honest PASS/INCONCLUSIVE gates)
- E2 spectrum (40-D L96):        python experiments/step74_lorenz96_spectrum.py   (seeds: STEP74_SEED)
- phase transition R^2(N):       python experiments/step83_*.py
- E9 co-demonstration:           python experiments/step79_certified_control.py
- E10 cone certificate:          python experiments/step82_certified_horizon_from_model.py
- E11 Acrobot triad:             bash run_step84.sh
- E12 budget decision:           STEP85_PHASE=0|1 python experiments/step85_trustworthy_cert_downstream.py
  ring replication:              STEP88_MODE=precheck|full python experiments/step88_ring_generality.py
  recalibration control:         python experiments/step85c_calibration_baseline.py
  UQ head-to-head:               python experiments/step90_uq_baselines.py
- E13 pretrained audit:          STEP89_TASKS=walker-walk,acrobot-swingup,cheetah-run,finger-spin,hopper-hop \\
                                 STEP89_SEEDS=1,2,3 python experiments/step89_pretrained_wm_audit.py
- honest negatives (recorded):   step85b (allocation), step86 (safety), step87 (cert-gated MBRL)
- unit tests:                    pytest tests/ -q   (equivariance guards included per project policy)

Seeds are fixed in-script (0/1/2 or official 1/2/3); every gate is pre-registered in docs/specs/ and never loosened —
INCONCLUSIVE is reported as such. Figures regenerate from the committed JSONs (experiments/fig_hero.py etc.).
"""

INCLUDE = [
    ("experiments", "*.py"), ("tests", "*.py"), ("docs/specs", "*.md"), ("scripts", "*.py"),
    ("papers/figures", "*.json"),
]
MAIN_FIGS = ["hero_certified_world_models.png", "step83_rsquared_crossover.png", "step71_multichaos_horizon.png",
             "step85_headline.png", "step84_certified_control_benchmark.png"]


def scrub(text: str) -> str:
    for pat, rep in SCRUB:
        text = pat.sub(rep, text)
    return text


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    leaks = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ANON_README.md", README)
        z.writestr("run_step84.sh", scrub((ROOT / "run_step84.sh").read_text()))
        for d, glob in INCLUDE:
            for f in sorted((ROOT / d).glob(glob)):
                if f.name == "make_anon_artifact.py":
                    continue
                rel = f.relative_to(ROOT).as_posix()
                txt = scrub(f.read_text(errors="replace"))
                if LEAK.search(txt):
                    leaks.append(rel)
                z.writestr(rel, txt)
        for name in MAIN_FIGS:
            f = ROOT / "papers" / "figures" / name
            if f.exists():
                z.write(f, f"papers/figures/{name}")
    if leaks:
        print(f"LEAK DETECTED in: {leaks}", file=sys.stderr)
        OUT.unlink()
        return 1
    n = len(zipfile.ZipFile(OUT).namelist())
    print(f"OK {OUT} ({n} files, {OUT.stat().st_size/1e6:.1f} MB), zero identifying tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
