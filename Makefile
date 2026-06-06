# Reproduction entry points for paper2 ("Certified World Models").
# All experiments are CPU/MPS (no CUDA); they self-configure a headless SDL for the PushT env.
# Usage:  make paper2        # full reproduction (multi-seed runs + figures + tests + PDF; slow, ~1-2h CPU)
#         make paper2-quick  # figures + PDF only, from committed JSONs (fast)
#         make paper2-test   # the equivariance + certificate test suite
#         make paper2-build  # rebuild the standalone PDF from the markdown + figures

PY := .venv/bin/python

.PHONY: paper2 paper2-seeds paper2-hero paper2-test paper2-build paper2-quick help

help:
	@grep -E '^##' Makefile | sed 's/## //'

## paper2          full reproduction: multi-seed experiments -> figures -> tests -> PDF (slow)
paper2: paper2-seeds paper2-hero paper2-test paper2-build

## paper2-seeds    re-run the multi-seed paper2 experiments at seeds {0,1,2} and (re)write per-seed JSONs
paper2-seeds:
	$(PY) experiments/aggregate_seeds.py

## paper2-hero     regenerate the Figure-1 concept schematic (papers/figures/hero_certified_region.png)
paper2-hero:
	$(PY) papers/figures/make_hero.py

## paper2-test     run the test suite (encoder/predictor/planner equivariance + certificate guards)
paper2-test:
	$(PY) -m pytest tests/ -q

## paper2-build    rebuild the standalone PDF (papers/paper2_certified_world_models.pdf) from clean source
paper2-build:
	$(PY) papers/arxiv_paper2/build_paper2.py

## paper2-quick    figures + PDF only, reusing committed experiment JSONs (skips the slow re-runs)
paper2-quick: paper2-hero paper2-build

## iclr-build      build the focused ICLR extraction draft (text+math only, no figures) -> QA PDF
iclr-build:
	pandoc papers/iclr_certified_horizons.md -o papers/iclr_certified_horizons.pdf \
	  --pdf-engine=tectonic --resource-path=papers -V geometry:margin=1in -V fontsize=10pt \
	  -V colorlinks=true -V linkcolor=NavyBlue \
	  --citeproc --bibliography=papers/iclr_refs.bib --metadata reference-section-title=References

# Official ICLR 2026 conference-format submission (anonymous, 9-page main text, natbib + official .bst)
iclr-submission:
	.venv/bin/python papers/iclr_submission/build_iclr.py
