# SE(3)-Equivariant JEPA — a geometric latent world model

> **The bet.** If the world carries a symmetry group $G$, *hard-wiring* that symmetry into a
> latent world model should let it learn from far fewer interactions and generalise **zero-shot**
> to configurations it never saw (举一反三) — instead of paying for that generalisation with scale.
> This repo tests that bet end-to-end, at laptop scale (CPU / Apple-MPS, **no CUDA**), fully seeded.

A latent **JEPA** (Joint-Embedding Predictive Architecture) built from an **equivariant encoder** $E$
and an **equivariant predictor** $f$. When the dynamics genuinely carries a group $G$ acting on latents
by an *orthogonal* representation $\rho(g)$, the one-step prediction loss is **provably invariant across
the whole group** — so fitting the dynamics on a thin slice of orientations *mathematically determines*
it on the entire orbit. We verify this in 2D ($\mathrm{SO}(2)$, on real **PushT**) and 3D
($\mathrm{SE}(3)$, on point clouds), and pit the equivariant model against a $4.5$–$7.4\times$
*parameter-richer* non-equivariant baseline of the **same hypothesis class**.

---

## Headline results

Every structural claim is guarded by an equivariance/invariance unit test **at initialisation _and_
after a real training run** (`tests/test_*.py`), and each fails the non-equivariant control.

| | Claim | Result |
|---|---|---|
| **[A]** | The learned symmetry **survives optimisation** | composed encode→predict residual $\sim\!10^{-6}$ *post-training*, not just at init — and under *any* optimiser (Muon = Adam = SGD = floor), because Vector-Neuron / `e3nn` weights parametrise the intertwiner space **intrinsically** (Step 26) |
| **[B]** | One-step prediction error is **exactly flat across the group** (举一反三) | VN OOD/seen $= \times1.00$ (a theorem) vs the baseline's $\times13.8$ (2D latent), $\times17.2$ (3D $\mathrm{SO}(3)$), $\times157$ (full $\mathrm{SE}(3)$ ladder) — with the equivariant model $4.5$–$7.4\times$ **smaller** and frequently *better* in-distribution |
| **[C]** | Closed-loop control error is **orientation-invariant** | float-floor-exact in 2D (paired $K{=}48$ PushT, seen-vs-OOD change $=0$); statistically flat in 3D $\mathrm{SE}(3)$ (VN $[0.977,0.999]$, disjoint from the MLP's $[1.049,1.234]$) |

**Locating the boundary (not overselling it).** The later steps map *where* the bet pays:

- **Sample-efficiency frontier** (Step 21): the equivariant payoff is a descending whole-group learning
  curve against a baseline **wall** — its across-group error never drops below the wedge.
- **Symmetry-break $\times$ data phase diagram** (Step 22, 5 seeds): the prior wins **24/25** cells
  across the group — near-total and **data-proof** — yet is a **wash-to-loss in-distribution**; the lone
  exception is a statistical tie on the *most-broken* row, not a clean corner.
- **Large-$N$ in-distribution** (Step 23): the in-distribution gap does **not** run away with the
  break, even at $N{=}2048$ (the wall is a *sample-efficiency* barrier, not an impossibility).
- **Object-interaction rung** (Step 24 → Step 27): coupling objects with an equivariant torque collapses
  the scene group to the global diagonal $\mathrm{SE}(3)\rtimes S_O$ and exposes the
  **interpolation/extrapolation flip** — a plain MLP fits the bilinear coupling *better* in-distribution
  (an honest degree-1 Vector-Neuron cap) yet blows up $\times17$ across the group while the equivariant
  model stays $\times1.00$. **Step 27** then *builds the named fix* — a tensor-product (cross-product
  $\mathbf 1\otimes\mathbf 1\to\mathbf 1$) message — and recovers **42%** of that cap ($\times1.45$ better
  fit) while staying exactly $\times1.00$. *Enrich the equivariant class, don't drop the prior.*
- **Active inference** (Steps 20, 25): the agent's curiosity (ensemble disagreement) is an exact
  $\mathrm{SE}(3)$-invariant, and in an ambiguous-goal cue-foraging **POMDP** an Expected-Free-Energy
  planner in the equivariant latent cuts a reward-only planner's true-goal error by **55%** — past a
  *provable* hedge floor — purely by seeking information, with the whole loop staying equivariant.

---

## The papers

- **[`papers/equivariance_generalization_core.md`](papers/equivariance_generalization_core.md)** — the
  focused, submission-targeted write-up of the most robust result: the **[A] + [B] + [C]** core in both
  $\mathrm{SO}(2)$ and $\mathrm{SE}(3)$.
- **[`papers/geometric_payoff.md`](papers/geometric_payoff.md)** — the full per-step results log (27
  steps): every experiment, table, caveat, and the boundary-locating analyses above.

Result dumps and figures for every step live in [`papers/figures/`](papers/figures/) (`*.json` numeric
dumps + `*.png` / `*.pdf`).

---

## Repository layout

```
se3-ejepa/
├── src/
│   ├── geometry/so2.py          # SO(2) group utilities
│   ├── models/
│   │   ├── structured.py        # Vector-Neuron primitives: VNLinear, VNReLU, VNTensorProduct, predictors
│   │   ├── se3.py               # e3nn SE(3) point-cloud encoder
│   │   └── eqjepa.py            # JEPA wrapper + jointly-equivariant latent predictor
│   └── training/
│       ├── jepa.py              # EMA-target + VICReg training loop
│       └── muon.py              # Muon/AdamW optimiser routing
├── experiments/                 # step1..step27 — one runnable script per result
├── tests/                       # equivariance guards (init + post-training)
├── papers/                      # the two write-ups + figures/
├── configs/                     # Hydra configs
└── requirements.txt             # pinned laptop environment
```

---

## Setup

Python **3.11**, CPU / Apple-MPS only (**no CUDA**). Dependencies are managed with
[`uv`](https://github.com/astral-sh/uv) (not pip) and pinned in `requirements.txt`:

```bash
cd se3-ejepa
uv venv                                   # creates .venv with Python 3.11
uv pip install -r requirements.txt        # this alone runs all of tests/ + every synthetic step + the figures
```

**One optional dependency is *not* bundled here.** `stable-worldmodel` (the **real-PushT**
simulator) is an external, non-PyPI package kept out of git (see the `requirements.txt` header); it
is needed *only* by the experiments that drive the real PushT environment — the ones that call
`make_env()` / `swm.World` (e.g. `step8`–`step10`, `step12`, `step14`). If you have it, install it
editable:

```bash
uv pip install -e <path-to>/stable-worldmodel   # only for the real-PushT steps
```

Everything that does **not** touch real PushT — the full `tests/` suite, the synthetic
$\mathrm{SO}(2)$/$\mathrm{SO}(3)$/$\mathrm{SE}(3)$ experiments, and the figures — imports and runs on
`requirements.txt` alone.

All experiments are deterministic given their seeds; the **[A]/[B]** claims are *theorems* and hold at
init and post-training regardless of seed.

---

## Reproduce

Most steps finish in minutes on a single CPU; the heavier 3D steps accept `STEP{n}_SMOKE=1` for a fast
wiring check. The SDL/objc dylib warnings on macOS are harmless (filter with `2>/dev/null`).

> The two **real-PushT** steps below (`step8`, `step12`) need the optional `stable-worldmodel`
> dependency from **Setup**; everything else here — all of `tests/`, the synthetic
> $\mathrm{SO}(2)$/$\mathrm{SO}(3)$/$\mathrm{SE}(3)$ steps, and the figures — runs on `requirements.txt` alone.

```bash
PRE="SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1"

# --- [A]+[B]+[C] core, in order of strength ---
$PRE .venv/bin/python experiments/step8_sample_efficiency.py        # synthetic SO(2): efficiency + 举一反三   [needs stable-worldmodel]
$PRE .venv/bin/python experiments/step12_pose_control.py            # real PushT SO(2): decomposed mechanism    [needs stable-worldmodel]
$PRE .venv/bin/python experiments/step13_se3_latent_jepa.py         # 3D SO(3): the lift (STEP13_SMOKE=1)
$PRE .venv/bin/python experiments/step18_se3_closed_loop.py         # 3D SE(3) closed-loop [C], paired bootstrap (STEP18_SMOKE=1)

# --- locating the boundary ---
$PRE .venv/bin/python experiments/step21_sample_efficiency_frontier.py  # frontier: VN whole-group curve vs MLP wall
$PRE .venv/bin/python experiments/step22_symmetry_data_phase.py         # (g x N) phase diagram (5 seeds)
$PRE .venv/bin/python experiments/step24_object_interaction.py          # interaction rung: interp/extrap flip x1.00 vs x17
$PRE .venv/bin/python experiments/step27_tensor_product_message.py      # tensor-product fix: recovers 42% of the cap, stays x1.00
$PRE .venv/bin/python experiments/step25_active_inference_task.py       # POMDP: EFE planner beats reward-only past the hedge floor

# --- headline figures (no retraining) ---
$PRE .venv/bin/python experiments/make_bet_figures.py

# --- equivariance guards (init + post-training) ---
$PRE .venv/bin/python tests/test_planner_equivariance.py            # single-plan SE(3) theorem: plan(g.x)=g.plan(x)
$PRE .venv/bin/python tests/test_set_equivariance.py                # scene-group SE(3)^O |x| S_O
$PRE .venv/bin/python tests/test_step26_optimizer_equivariance.py   # intrinsic stays exact under any optimiser
$PRE .venv/bin/python tests/test_step27_tensor_product.py           # VNTensorProduct degree-2, SO(3)-equiv, pseudovector
```

The full command list (every step + every guard) is in the **Reproduce** section of
`papers/equivariance_generalization_core.md`.

---

## Audit your own world model (`wm-audit`)

The certificate machinery doubles as a standalone, **training-free trustworthiness audit** for pretrained
world models — it reads the latent loop's leading Lyapunov band and issues a certified horizon per
resolution, or honestly **abstains** where a Lyapunov certificate has no jurisdiction:

```bash
.venv/bin/python scripts/wm_audit.py tdmpc2 models/tdmpc2/walker-walk-1.pt --task walker-walk
# python API for ANY deterministic differentiable latent map:
#   from scripts.wm_audit import audit_map
```

Quickstart, checkpoint URLs, the validated scope taxonomy (calibrated / optimistic / abstain), and what the
certificate does *not* promise: [`docs/wm_audit_quickstart.md`](docs/wm_audit_quickstart.md).

---

## Honest scope

This is a focused empirical claim, not a scaling result. Deliberately **out of scope**:

- **Scale.** Everything is laptop-scale; we say nothing about whether brute-force scaling eventually
  beats the prior (Sutton's *Bitter Lesson* is the standing caveat).
- **Where the guarantee bites.** The exact flatness holds only where the world's symmetry is *real*; the
  across-group payoff is **not** an in-distribution sample-efficiency edge — a higher-capacity baseline
  fits the seen slice *better*, then breaks out-of-distribution.
- **Constructed settings.** The active-inference win (Step 25) is on a *constructed* POMDP over a
  synthetic teacher; what is proven is the mechanism (an invariant epistemic drive converted into a win a
  reward-only planner provably cannot match), not a benchmark victory in the wild.

---

## Citation

If you use this work, please cite it. (The arXiv `eprint` ID will be filled in once the preprint is
posted; a Zenodo DOI for the code archive will be added at the first tagged release.)

```bibtex
@misc{se3ejepa2026,
  title         = {Exact equivariance, kept through training, buys zero-shot
                   generalisation across the symmetry group},
  author        = {Wang, Hongbo},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  howpublished  = {\url{https://github.com/TimothyWang418/se3-ejepa}}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

---

## License

A deliberate split — permissive on both sides, the right instrument for each artifact:

- **Code** — [Apache License 2.0](LICENSE) (permissive + an explicit patent grant).
- **Paper & figures** (everything under [`papers/`](papers/)) — **CC BY 4.0** (reuse freely; just
  credit the author).

Copyright © 2026 Hongbo Wang.

---

*Laptop-scale (CPU/MPS), fully seeded and deterministic. Last updated: 2026-05-30.*
