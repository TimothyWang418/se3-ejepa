# wm-audit — certify a pretrained world model's trustworthy horizon (quickstart)

`scripts/wm_audit.py` reads the **leading Lyapunov band of a world model's latent loop** (forward-mode Benettin-QR)
and issues a **certified predictability horizon** $T_1(\epsilon)=\log(1/\epsilon)/\lambda_1$ per resolution — or
**abstains** when the $\lambda_1$ CI straddles zero (a contracting/neutral loop's residual divergence is bias, not
amplification — outside a Lyapunov certificate's jurisdiction). Training-free: no fine-tuning, no environment access
on the certified side, one forward-AD sweep of the checkpoint you already have.

The machinery, scope theory (Props 7–11), and the 15-cell + multitask-ladder + JEPA-family audits it reproduces are
in `papers/iclr_certified_horizons.md` (E13–E15).

## Install

```bash
git clone <this repo> && cd se3-ejepa
python -m venv .venv && .venv/bin/pip install -e .          # torch, numpy
.venv/bin/pip install dm_control                            # for the tdmpc2 / tdmpc2-mt families
.venv/bin/pip install "transformers<5" einops               # for the lewm family (classic ViT naming)
```

## Get checkpoints (pinned URLs — never vendored)

```bash
mkdir -p models/tdmpc2 models/tdmpc2_mt models/lewm
# TD-MPC2 single-task DMC (nicklashansen/tdmpc2, MIT):
curl -L -o models/tdmpc2/walker-walk-1.pt \
  https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/walker-walk-1.pt
# TD-MPC2 multitask ladder (same repo): .../multitask/mt30-317M.pt etc.
# LeWM (quentinll/lewm-*): weights + config.json per model card.
```

## CLI

```bash
# single-task TD-MPC2: certify the policy-prior latent loop g(z) = d(z, tanh(mu_pi(z)))
.venv/bin/python scripts/wm_audit.py tdmpc2    models/tdmpc2/walker-walk-1.pt --task walker-walk
# multitask ladder cell
.venv/bin/python scripts/wm_audit.py tdmpc2-mt models/tdmpc2_mt/mt30-317M.pt  --task walker-walk
# LeWM (pixels JEPA; authors' own code, bit-faithful load)
.venv/bin/python scripts/wm_audit.py lewm      models/lewm/pusht-weights.pt   --config models/lewm/pusht-config.json
# options: --eps 0.05,0.1,0.2   --steps 600   --k 12   --json report.json
```

Output is a JSON report:

```json
{ "lambda1": 0.268, "lambda1_ci": [0.217, 0.305],
  "leading_band": [...],
  "horizons": [{"eps": 0.2, "T1_steps": 6.0}, ...],
  "verdict": "EXPANSIVE — certified horizons issued" }
```

**Estimator note (honest):** `--steps` is the Benettin orbit length; short windows are noisy (120 steps can move
$\lambda_1$ by ~0.1). The published reference numbers (below) use 600 steps + 120 warmup + block-bootstrap CI; treat
anything under 300 as a smoke test. The CI **is** the uncertainty statement — Proposition 10 is its rate theorem
($n\asymp\sigma_\infty^2\log(1/\delta)/\varepsilon^2$).

## Python API — audit *any* deterministic differentiable latent map

```python
import torch
from scripts.wm_audit import audit_map

def g(z):                       # your latent loop: R^d -> R^d, float64, C^1
    return my_dynamics(z, my_policy(z))

report = audit_map(g, z0.double(), eps_list=[0.05, 0.1, 0.2], n_steps=600)
print(report["verdict"], report["horizons"])
```

## How to read the verdict — the scope taxonomy (validated cell-by-cell, E13/E14)

| latent-loop regime | certificate behaviour | reference cells (published) |
|---|---|---|
| strongly expansive ($\lambda_1\gtrsim0.25$) | **calibrated** — measured/certified $0.83$–$1.02$ | walker-walk 3/3 ($0.94/0.95/1.02$), cheetah-3 ($0.83$) |
| weakly expansive ($\lambda_1\approx0.05$–$0.2$) | **optimistic** — bias outpaces amplification (Prop 7 degeneracy) | cheetah-1/2 ($0.43/0.50$), hopper-hop ($0.13/0.38$) |
| contracting / neutral (CI $\ni 0$) | **abstain** — correctly, in both sub-cases | acrobot 3/3, finger-spin 2/3 (stable); LeWM (bias-driven) |

Deployment evidence: a sensor-only monitor forecasting with the certified loop replicates this map **cell-by-cell,
out-of-sample, at zero new estimation** (in-situ vs bench ratios $0.43/0.43$, $0.50/0.50$, $0.67/0.83$ — E15,
`experiments/step94_budgeted_monitor.py`).

## Scope — what this certifies and what it does not

- The certified object is the **latent loop you pass in** (for TD-MPC2: the policy-prior loop, *not* the MPPI
  planner). Decision value concentrates where the decided quantity IS the certified quantity (Proposition 11(i));
  through a task map it dilutes by the mis-resolution penalty $|\log(\epsilon/\theta^{\ast})|/\lambda_1$
  (Proposition 11(ii)) — the certificate cannot supply the task's implicit tolerance $\theta^{\ast}$.
- The horizon prices the **median** staleness clock; tail-quantile budgets bind earlier (quantile choice is yours).
- For a generic (non-equivariant) model the spectrum can be silently wrong while predictions stay good — cross-check
  against held-out divergence once (the E13 protocol). Structure (equivariance) is what removes that requirement
  where it holds; that exclusivity (Lemma 2) is the paper's thesis, not a property of this script.

## Reproduce the published audits

```bash
.venv/bin/python experiments/step89_pretrained_wm_audit.py     # 15-cell single-task map
.venv/bin/python experiments/step91_lewm_audit.py              # LeWM (JEPA family)
.venv/bin/python experiments/step92_scale_sweep.py             # 1M–317M multitask ladder
.venv/bin/python experiments/step94_budgeted_monitor.py        # deployed monitor (E15)
.venv/bin/python -m pytest tests/ -q                           # gates + artifact consistency
```

Per-cell JSONs land in `papers/figures/`; every number quoted above is reproducible from them.
