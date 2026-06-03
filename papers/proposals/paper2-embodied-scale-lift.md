---
title: "paper2 — the embodied / scale lift (what it would take to reach top-tier)"
status: roadmap (not started); the honest path beyond the laptop-scale mechanism paper
created: 2026-06-03
---

# What this is

paper2 ("Certified World Models") is, today, a clean **mechanism-and-theory** paper with a real-contact-dynamics
anchor (Experiment 9, PushT). Its honest ceiling, stated in §7, is fourfold:

1. **structured state, not pixels** — the learned latent is the engineered 8-D PushT state, not raw observation;
2. **a single $\mathrm{SO}(2)$ task** — one planar group;
3. **a modest out-of-distribution gap** ($2\text{–}4\times$) — because one-step PushT prediction is easy in-dist;
4. **laptop scale** — CPU/MPS, $\le10^{5.5}$ parameters.

Top-tier venues (NeurIPS/ICLR main track) reward **a downstream-task result at non-trivial scale on raw
observations**, not a 1-step-relMSE certificate on a structured toy. This document is the concrete, staged plan to
close that gap — what each stage *proves*, the setup, the compute, the risk, and the tier it unlocks. It is the
experiment the paper itself names as its open frontier.

The governing principle stays the project's: **honest, gated, reproducible**; a stage that comes back
`INCONCLUSIVE` is reported as such, never loosened.

---

# The staged plan

Ordered by **leverage ÷ cost**. S1–S2 are laptop-feasible and are the highest-value next steps; S3–S5 need real
compute and are the genuine "top-tier" tier.

## S1 — Pixels (kill ceiling #1).  *Laptop-feasible.*

**Claim it establishes.** The certificate holds for a latent **learned from raw pixels**, not from an engineered
state vector — so the symmetry prior is not smuggled in via hand-built coordinates.

**Setup.** Render PushT to images; train an $\mathrm{SO}(2)$-steerable **pixel** JEPA — the repo already has
`src/models/eqjepa.py::SteerableEncoder` (e2cnn, $C_N$) + the JEPA trainer `src/training/jepa.py::train_jepa`. Run
the Experiment-7/9 wedge protocol on the pixel latent: train on a $\pm50^\circ$ orientation wedge, test zero-shot
across the orbit; equivariant-pixel vs scaled non-equivariant-CNN baselines.

**Metric.** Orbit-flatness of the rollout error on the pixel latent + the structure-vs-scale floor (as in Exp 9).

**Honest risk.** $C_N$ steerable convs are *exactly* equivariant only at the $N$ grid angles; off-grid angles hit a
bilinear-resampling floor (~$10^{-1}$, documented in the repo). So the pixel certificate will be **approximate**
($\epsilon_{\max}>0$) — which is fine and *on-thesis* (it lands in Theorem B's regime, with $\epsilon_{\max}$
measured), but the headline "exactly flat" weakens to "flat up to the resampling floor." Use a large $N$ (e.g.
$C_{16}$) and report the floor honestly. **Cost:** hours on MPS.

**Tier impact.** Removes the single most-cited reviewer objection ("structured state"). Moves TMLR/workshop from
strong to very strong; necessary-but-not-sufficient for main track.

## S2 — Task success, not prediction error (kill ceiling #3).  *Laptop-feasible.*

**Claim it establishes.** The certificate predicts **zero-shot closed-loop task success** on unseen orientations —
a downstream metric — not just low 1-step relMSE. *This is the single highest-leverage laptop experiment.*

**Setup.** The repo has CEM-MPC closed-loop on real PushT (`experiments/step10/step12`, `EqJEPA.criterion`/SWM
`Costable`). Train the world model on a wedge; deploy the equivariant **planner** zero-shot at orientations across
the orbit; baseline = scaled non-equivariant planner. Measure **success rate / fraction-of-goal-gap-closed vs
scene orientation**.

**Metric.** Success(orientation): equivariant should be **flat over the orbit** (the closed-loop $[C]$ guarantee,
Theorem A under assumption (A5)); the baseline should fall off-wedge. Report the paired seen-vs-unseen success
ratio with seeds (the repo's Step-14 power-analysis style).

**Honest risk.** Planning amplifies model error; the equivariant model must be a good-enough *controller*, not just
predictor. If success is low everywhere, the result is "flat but low" — report it (it is the honest "flat ≠ good"
case on a task). **Cost:** hours on MPS (CEM is the bottleneck).

**Tier impact.** A *task-success* certificate is the kind of result main-track reviewers reward. Combined with S1,
this is the difference between "mechanism paper" and "the certificate matters for control."

## S3 — A larger group on a real 3D task (kill ceiling #2).  *Needs a 3D sim + GPU.*

**Claim it establishes.** The certificate is not an $\mathrm{SO}(2)$ artifact — it holds for $\mathrm{SE}(3)$ on a
real 3D contact task.

**Setup.** A real 3D manipulation sim with genuine $\mathrm{SE}(3)$ structure — **ManiSkill** or **RLBench** (not
currently installed; both need a GPU + MuJoCo/SAPIEN). Use the repo's `src/models/se3.py::SE3PointEncoder` (e3nn) on
point-cloud observations. Same wedge protocol over an $\mathrm{SO}(3)$ orbit.

**Honest risk.** Real 3D contact (friction, multi-body) is only *approximately* $\mathrm{SE}(3)$-equivariant
(gravity breaks it to $\mathrm{SE}(2)\times$…); pick a task where the relevant symmetry survives (e.g. in-plane
manipulation = $\mathrm{SE}(2)$, or a micro-gravity/free-floating setup for full $\mathrm{SE}(3)$). **Cost:** days
on a GPU.

**Tier impact.** Generality across groups + a recognizable 3D benchmark — a main-track expectation.

## S4 — Scale the contrast (kill ceiling #4).  *Needs a GPU.*

**Claim it establishes.** The structure-vs-scale gap does **not** close as the non-equivariant baseline grows to a
meaningful size on a non-trivial task — the empirical complement to the §3.3 information-theoretic separation.

**Setup.** On the S1/S3 task, sweep the baseline across $\ge3$ orders of magnitude of parameters and data; track
the out-of-orbit floor-penalty vs scale. (§3.3 *proves* it cannot reach the floor; this shows it empirically at
scale, answering "you only swept to 337k.")

**Cost:** GPU-days. **Tier impact.** Directly rebuts the "scale was only swept to 337k" objection with data.

## S5 — The embodied Noether hinge (the real frontier).  *Research-grade; may stay open.*

**Claim it would establish.** On a *learned* latent of a real, contact-rich embodied task, the conserved/slow
content organizes into the invariant ($\ell{=}0$) $\oplus$ conserved-equivariant blocks — lifting the hinge
(Step 57 was `INCONCLUSIVE` on the two-body toy; §4 Proposition 4 proves only the *placement* of given charges, not
that the learned slow modes coincide with them).

**Setup.** Probe the S1/S3 learned latent: regress the task's (approximate) conserved quantities from the $\ell$-typed
blocks; measure the slow subspace; test the "slow = conserved" coincidence the paper currently only measures on
toys.

**Honest risk.** Real dissipative contact dynamics may have *no* clean conserved quantity → the hinge may simply
not lift, and the honest outcome is a negative/`INCONCLUSIVE` result that sharpens the hinge's scope. This is
genuinely open and might not resolve favorably; that is the nature of the bet.

---

# Recommended sequence

1. **S2 (task success on PushT)** — highest leverage, laptop-feasible, reuses existing CEM-MPC. Do first.
2. **S1 (pixels on PushT)** — laptop-feasible, kills the loudest objection. Do second (or in parallel).
3. *Gate:* if S1+S2 land cleanly, the paper is a strong TMLR / workshop-oral submission and a *plausible* main-track
   one. Decide then whether the GPU tier is worth it.
4. **S3 → S4** (GPU): the main-track tier — a larger group, a real 3D benchmark, scale. Requires committing real
   compute.
5. **S5**: attempt opportunistically on the S1/S3 latents; report honestly either way.

# Honest framing of the ceiling on the bet

Even with all of S1–S5, the project's contrarian thesis (geometric inductive bias beats scaled brute force on
sample efficiency) is a 5–7 year bet, and Sutton's Bitter Lesson is the standing counter-argument. These stages
make paper2 a *strong, well-evidenced* contribution; they do not, by themselves, settle the bet. The math learned
en route (Lie groups, equivariance, information geometry) is permanent capital regardless — as the project's own
charter notes.
