# Step 79 — Certified-Control Co-Demonstration: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On controlled Lorenz-96 (one chaotic $\mathbb{Z}_N$-symmetric system), demonstrate simultaneously: (C) orbit-flat control from an equivariant CEM-MPC planner, (H) a certified horizon $T_j(\epsilon)$ with bootstrap error bars, and (D) a certificate-aware planner ($H{=}T_j$) matching/beating a horizon-blind one at suppressing chaos to $x_i{=}F$.

**Architecture:** Reuse `experiments/step74_lorenz96_spectrum.py` (dynamics, $\mathbb{Z}_N$-conv & MLP, Benettin–QR spectrum) and `experiments/step78_certified_horizon_ci.py` (bootstrap CI). Add a control input $u_i$ per site, an action-conditioned world model, an equivariant CEM-MPC planner, and a closed-loop runner. One new experiment file + one new test file, mirroring the `step76/77/78` structure.

**Tech Stack:** Python 3.11, PyTorch (CPU/MPS, float64 for spectra), matplotlib. Repo: `~/se3-ejepa`, `.venv/bin/python`.

**Repo rules (non-negotiable):** NEVER loosen a gate (report `INCONCLUSIVE`). Stage files by name (never `git add -A`). Commit locally; confirm before push. HEREDOC commit messages ending `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Honest math in LaTeX.

**Files:**
- Create: `experiments/step79_certified_control.py` — the experiment (dynamics+control, action-conditioned WM training, certificate read-off, equivariant CEM-MPC, closed-loop runner, gates, figure).
- Create: `tests/test_step79.py` — controlled-dynamics $\mathbb{Z}_N$-equivariance + planner equivariance/orbit-invariance.
- Outputs: `papers/figures/step79_certified_control{_seedS}.{json,png}`.
- Reuse (import, do not modify): `step74_lorenz96_spectrum`, `step78_certified_horizon_ci`.

---

## Phase 0 — Controlled dynamics + $\mathbb{Z}_N$-equivariance (TDD)

### Task 0.1: Failing equivariance test for the controlled field

- [ ] **Step 1: Write the failing test** in `tests/test_step79.py`

```python
import sys; from pathlib import Path
import torch
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "experiments")))
import step79_certified_control as s79  # noqa: E402
DT = torch.float64

def test_controlled_dynamics_is_ZN_equivariant():
    torch.manual_seed(0); N = 12
    x = torch.randn(N, dtype=DT); u = torch.randn(N, dtype=DT)
    for s in (1, 3, 7):
        lhs = s79.l96_controlled_rhs(torch.roll(x, s), torch.roll(u, s))
        rhs = torch.roll(s79.l96_controlled_rhs(x, u), s)
        assert torch.allclose(lhs, rhs, atol=1e-12), f"shift {s}: field not Z_N-equivariant"
        # and the full RK4 map
        mlhs = s79.rk4_controlled(torch.roll(x, s), torch.roll(u, s))
        mrhs = torch.roll(s79.rk4_controlled(x, u), s)
        assert torch.allclose(mlhs, mrhs, atol=1e-12), f"shift {s}: RK4 map not equivariant"
    print("PASS: controlled Lorenz-96 field and RK4 map are Z_N-equivariant.")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/se3-ejepa && .venv/bin/python -m pytest tests/test_step79.py -k equivariant -q`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError: l96_controlled_rhs`).

- [ ] **Step 3: Implement the controlled dynamics** at the top of `experiments/step79_certified_control.py`

```python
r"""Step 79 — certified-control co-demonstration on controlled Lorenz-96 (see docs/specs/2026-06-07-...)."""
import json, os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as s74  # noqa: E402  (l96_rhs, attractor_traj, lyapunov_spectrum, DTMAP, kaplan_yorke)
import step78_certified_horizon_ci as s78  # noqa: E402  (qr_logR_series, bootstrap_spectrum_ci, horizon_interval)

DTYPE = torch.float64
F_FORCE = 8.0
DT = 0.01

def l96_controlled_rhs(x, u, F=F_FORCE):
    r"""$\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F+u_i$. Z_N-equivariant in (x,u) jointly (roll commutes with roll)."""
    return (torch.roll(x, -1, -1) - torch.roll(x, 2, -1)) * torch.roll(x, 1, -1) - x + F + u

def rk4_controlled(x, u, dt=DT, F=F_FORCE):
    r"""One RK4 step with a zero-order-hold control u (u constant over the step)."""
    k1 = l96_controlled_rhs(x, u, F); k2 = l96_controlled_rhs(x + 0.5 * dt * k1, u, F)
    k3 = l96_controlled_rhs(x + 0.5 * dt * k2, u, F); k4 = l96_controlled_rhs(x + dt * k3, u, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
```

- [ ] **Step 4: Run to verify it passes**

Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/se3-ejepa
git add experiments/step79_certified_control.py tests/test_step79.py
git commit -F- <<'EOF'
Step 79 phase 0: controlled Lorenz-96 dynamics + Z_N-equivariance test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

**REVIEW CHECKPOINT 0:** controlled field is equivariant; fixed point $x_i{=}F$ verified ($\dot x=0$ at $u=0$).

---

## Phase 1 — Action-conditioned world model (reuse step74 architecture)

### Task 1.1: Equivariant + baseline world models that take $(x,u)$

The model predicts $x_{t+1}$ from $(x_t, u_t)$. The $\mathbb{Z}_N$-conv treats $u$ as a second input channel (so the map is jointly equivariant); the dense MLP concatenates $[x;u]$ (no structure). Both reuse step74's residual + circular-conv pattern.

- [ ] **Step 1:** Add to `step79_certified_control.py` an `L96ControlledConv` (mirror `s74.L96CyclicConv` but `Conv1d(2, ...)` first layer, input channels `[x, u]`, residual on `x`) and `L96ControlledMLP` (`Linear(2N, hidden)`, residual on `x`). Shape annotation: `(x:(...,N), u:(...,N)) -> (...,N)`.
- [ ] **Step 2:** Add a unit test `test_world_model_conv_is_equivariant`: an untrained `L96ControlledConv` satisfies `f(roll x, roll u) == roll f(x,u)` to `1e-10` (the architectural guarantee), and the MLP does NOT. Run, confirm.
- [ ] **Step 3:** Add `collect_data(N, n, seed)` — roll the TRUE controlled dynamics from attractor states under random bounded controls `u ~ U[-u_max, u_max]` (reuse `s74.attractor_traj` for init), returning `(x, u, x')` triples in normalized coords.
- [ ] **Step 4:** Add `train_wm(kind, data, ...)` — multi-step rollout loss (mirror `s74.train_model`, but the rollout feeds a fixed/zero control or the data's controls). Return model + one-step relMSE.
- [ ] **Step 5:** Commit `Step 79 phase 1: action-conditioned equivariant + baseline world models + equivariance test`.

**REVIEW CHECKPOINT 1:** both WMs train to relMSE $<10^{-3}$ on a smoke config; conv is exactly equivariant.

---

## Phase 2 — Certificate read-off (reuse step74 QR + step78 CI)

### Task 2.1: $T_j(\epsilon)$ with bootstrap CI from the trained WM at the operating point

- [ ] **Step 1:** Add `certificate(model, x_star, eps)` — form the autonomous map `g(x) = model(x, u=0)`; reuse `s78.qr_logR_series` + `s78.bootstrap_spectrum_ci` to get $\{\lambda_j\}\pm$CI at the operating point near $x_i{=}F$; convert to $T_j(\epsilon)$ via `s78.horizon_interval`. Return the binding (fastest positive) channel's $T_1(\epsilon)$ and its CI.
- [ ] **Step 2:** Unit-test that on the TRUE controlled map at $u=0$ the certificate's $\lambda_1$ CI is positive (chaotic) — reuse the `test_step78` calibration idea.
- [ ] **Step 3:** Commit `Step 79 phase 2: certified horizon T_j with bootstrap CI from the action-conditioned WM`.

**REVIEW CHECKPOINT 2:** $T_1(\epsilon)$ and its error bar are read off the learned WM; sanity vs the true-map value.

---

## Phase 3 — Equivariant CEM-MPC planner (TDD)

### Task 3.1: Planner + equivariance/orbit-invariance test

The planner optimizes a control sequence $u_{0:H}$ by CEM, scoring candidates by the WM rollout cost $\sum_t \lVert \hat x_t - F\mathbf 1\rVert^2$ (which is $\mathbb{Z}_N$-invariant). Because the WM is equivariant and the cost is invariant, the planner is orbit-equivariant.

- [ ] **Step 1: Write the failing planner-equivariance test** (adapt `tests/test_step73_planner_equivariance.py`):

```python
def test_planner_is_orbit_equivariant():
    torch.manual_seed(0); N = 12
    model = s79.make_equivariant_wm(N)            # small untrained equivariant WM is enough (it's equivariant by construction)
    x0 = torch.randn(N, dtype=DT)
    for s in (1, 5):
        u_a = s79.cem_plan(model, x0, seed=0)                       # control sequence (H, N)
        u_b = s79.cem_plan(model, torch.roll(x0, s), seed=0)        # SAME cem noise seed, shifted IC
        # with matched noise, the optimal-control map commutes with the shift up to CEM stochasticity:
        assert torch.allclose(u_b, torch.roll(u_a, s, dims=-1), atol=1e-6), f"shift {s}: planner not equivariant"
    print("PASS: equivariant CEM-MPC planner commutes with the Z_N shift (matched noise).")
```

(Note for implementer: to make this exact, `cem_plan` must shift its sampled noise by `s` when the IC is shifted — i.e., draw noise in a canonical frame. Implement CEM so the candidate sampling is itself equivariant; if exactness is impossible, weaken the assert to orbit-*invariance of the achieved cost* `|cost_a - cost_b| < tol` and document why.)

- [ ] **Step 2:** Run, confirm FAIL (`cem_plan` undefined).
- [ ] **Step 3:** Implement `make_equivariant_wm(N)` (thin wrapper) and `cem_plan(model, x0, H, n_iter, n_samp, u_max, seed)` — standard CEM over $u_{0:H}$, equivariant noise sampling, return the mean control sequence.
- [ ] **Step 4:** Run, confirm PASS (or document the weakened orbit-invariance assert).
- [ ] **Step 5:** Commit `Step 79 phase 3: equivariant CEM-MPC planner + equivariance test`.

**REVIEW CHECKPOINT 3:** planner commutes with the shift (or achieves orbit-invariant cost), validated.

---

## Phase 4 — Closed-loop runner + orbit-flatness (config axis, C)

- [ ] **Step 1:** Add `closed_loop(model, x0, H, n_steps, u_max)` — MPC: each step, `cem_plan` to depth `H`, apply the FIRST control to the TRUE dynamics, repeat; return the realized trajectory + time-averaged cost $\overline{\lVert x - F\mathbf 1\rVert}$ + the control trajectory.
- [ ] **Step 2:** Add `orbit_flatness(model, x0, H, s)` — run `closed_loop` from `x0` and from `roll(x0, s)`; return ratio of costs and the max control mismatch $\lVert u_t(\text{roll }x_0) - \text{roll }u_t(x_0)\rVert$. Gate: ratio $<1.05$, mismatch small.
- [ ] **Step 3:** Commit `Step 79 phase 4: closed-loop MPC runner + orbit-flatness of control (config axis)`.

**REVIEW CHECKPOINT 4:** equivariant planner ⇒ orbit-flat control (ratio $\to 1.000$); a baseline (non-equivariant WM) planner drifts.

---

## Phase 5 — Certificate-aware vs. horizon-blind (the decision, D)

- [ ] **Step 1:** Add `sweep_blind(model, x0, H_list, ...)` — run `closed_loop` for each fixed `H` in a sweep (e.g. `[T_j, 2·T_j, 4·T_j, full]`); record cost-vs-H.
- [ ] **Step 2:** Add the contrast: cert-aware uses `H = round(T_1(eps))` (from Phase 2, NO tuning); blind uses the best-swept fixed `H`. Gate (D): cert-aware cost $\le$ best-swept-blind cost AND strictly $<$ the naive long-$H$ blind, on 3/3 seeds.
- [ ] **Step 3:** Commit `Step 79 phase 5: certificate-aware vs horizon-blind MPC contrast (the decision)`.

**REVIEW CHECKPOINT 5:** the cost-vs-H curve shows blind degrades for too-long H (trusts the WM past T_j); cert-aware sits at/near the minimum without tuning.

---

## Phase 6 — Gates, figure, run, honest read

- [ ] **Step 1:** Add `run(seed)` assembling (C)+(H)+(D) into one result dict with honest gates (PASS iff all three pass; else `INCONCLUSIVE`), mirroring `step77/78` messaging. Env knobs `STEP79_{SMOKE,SEED,N,DEVICE}`.
- [ ] **Step 2:** Add `_save` — the 3-panel headline figure: (a) orbit-flatness ratio bar (equiv vs baseline); (b) $T_j\pm$ error bar; (c) cost-vs-H curve with the cert-aware point marked. JSON alongside.
- [ ] **Step 3:** Smoke: `STEP79_SMOKE=1 .venv/bin/python experiments/step79_certified_control.py` — fix any crashes.
- [ ] **Step 4:** Full run seeds 0/1/2 (background if slow); **read the honest result** (PASS / INCONCLUSIVE / which gate failed). Do NOT loosen gates.
- [ ] **Step 5:** Run all tests: `test_step74`, `test_step78`, `test_step79`. Commit `Step 79 phase 6: gates + 3-panel figure + 3-seed run (honest result: <...>)`.

**REVIEW CHECKPOINT 6 (DECISION GATE):** read the 3-seed result together. If PASS → Phase 7 (fold). If INCONCLUSIVE → diagnose (Phase 6b: tune regime per spec §8 risks, or honestly scope/park).

---

## Phase 7 — Fold into both papers (only if it lands; else honest scope)

- [ ] **Step 1:** paper2: new `### 5.x` experiment subsection (the co-demonstration) + embed `step79` figure + ledger entry in `paper2_record.md`.
- [ ] **Step 2:** ICLR: a compact E-paragraph (or fold into E7 closed-loop) — **re-verify main text ≤9pp** after the add; trim to fit.
- [ ] **Step 3:** Rebuild both PDFs; verify ICLR ≤9pp + 0 unresolved cites, paper2 0 stray `[@`.
- [ ] **Step 4:** Commit `Step 79: certified-control co-demonstration folded into both papers`. **Confirm with user before push.**

**REVIEW CHECKPOINT 7:** both papers rebuilt clean; the two axes now meet + the certificate drives a decision, on one system.

---

## Phase 8 (STRETCH, parked if not landing) — corroborator class-lift

Only after the anchor PASSES. Each is a separate experiment file mirroring step79, parked (like the ESN) if it doesn't train/control cleanly.

- [ ] **8a:** `step80` — $\mathbb{Z}_N$ coupled-pendulum ring (build dynamics + verify chaotic + verify $\mathbb{Z}_N$-equivariant + repeat phases 1–6). Embodied narrative.
- [ ] **8b:** `step81` — chaotic double pendulum ($\mathbb{Z}_2$ reflection) (familiar benchmark; repeat phases 1–6).
- [ ] **8c:** If $\ge$1 corroborator lands, upgrade the paper claim to "across a class of chaotic symmetric systems"; else keep the anchor + note the class-lift as future work.

---

## Self-Review (against the spec)

- **Spec coverage:** (C) Phase 3–4; (H) Phase 2; (D) Phase 5; honest gates Phase 6; tests Phase 0/1/3; reproducibility (seeds/CPU/float64) throughout; staged corroborators Phase 8; figure Phase 6; fold Phase 7. ✔ all spec sections mapped.
- **Placeholders:** the inherently-empirical steps (train/run/read) are marked as run-and-read, not code-placeholders; the deterministic load-bearing parts (dynamics, equivariance tests, planner skeleton) have concrete code. ✔
- **Type/name consistency:** `l96_controlled_rhs`, `rk4_controlled`, `L96ControlledConv/MLP`, `collect_data`, `train_wm`, `certificate`, `cem_plan`, `make_equivariant_wm`, `closed_loop`, `orbit_flatness`, `sweep_blind`, `run`, `_save` — used consistently across phases. ✔
- **Known risk (per spec §8):** the (D) effect needs a regime where $T_j$ is genuinely short vs the task; if Phase 6 shows no separation, Phase 6b tunes the regime or reports INCONCLUSIVE honestly. ✔
