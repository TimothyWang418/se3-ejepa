# P4-step1 — pixel-PushT pipeline + base ladder + probe rig (E-P4.1 first pass) — seed spec

*Date: 2026-06-10 · Status: seed (W1 recon DONE, build-ready) · Owner: paper3 / Certified Subgoal Spacing · Compute: Mac CPU/MPS (train f32; audits stay CPU f64), no GPU*

> First build of paper3. Proposal (registered claims + data protocol v1.1):
> `papers/proposals/paper3-phase4-certified-subgoal-spacing.md`. Recon facts:
> `docs/specs/2026-06-10-p4-w1-recon.md`. This step builds the κ=0 substrate: data pipelines (both
> corpora), the {eq, plain-ladder, aug} bases, the PoR-style probe rig, and the seed-0 pass of
> E-P4.1. It creates `papers/paper3_record.md`. **No claim gate is evaluated here** — C2's tiers
> need 3 seeds; step1 sizes the compute and proves the machinery.

## Verified facts (W1 recon — file:line in the recon doc)

- Env `swm/PushT-v1` (vendored `third_party/stable-worldmodel`): obs dict
  {`pixels` 0–255 (ctor `resolution`, default 224), `state` (block pose; velocities last 2)};
  action 2-d PD target; **`damping` ctor param** (default `space.damping=0`).
- `WeakPolicy` collector + solver suite (`cem/icem/mppi`) vendored.
- `train_jepa(model, obs, act, nxt, ..., device=)` ready; `SteerableEncoder(in_channels,
  latent_dim, n_rot, width)` — $C_N$ regular rep; $128 = 16\times8$ regular fields ✓; plain-CNN
  baseline class in the same file.
- lerobot/pusht = 206 eps (the κ=0 *robustness row* — **not** in step1; step1 is WeakPolicy +
  oracle-CEM only).

## Pre-registered gates (G0–G4 — do NOT drift)

- **G0a env smoke**: `gym.make("swm/PushT-v1", resolution=96)` reset/step/render OK; **`damping`
  kwarg pass-through asserted** (`make(..., damping=0.95)` ⇒ `env.unwrapped.space.damping == 0.95`
  after reset) — the κ lane's load-bearing assumption, tested before anything else.
- **G0b collection**: 200 WeakPolicy episodes @ κ=0, 96×96; assert per-step state exposes block pose
  ($x,y,\theta$ extractable) + dtypes/ranges; fixed seeds.
- **G0c oracle demos**: true-env state-reset CEM produces successful episodes at κ=0;
  **infrastructure gate ≥80% success within the compute budget** (solver choice cem→icem→mppi is
  free — solver identity is not a claim).
- **G1 training health**: five bases — eq ($C_{16}$, latent 128), plain ladder ×3 (param-matched:
  $\{\sim0.5\times, \sim1\times, \sim2\times\}$ eq's count), aug (= plain-1× + rotation
  augmentation) — each: `latent_std` collapse-witness above floor, `pred_loss` decreasing.
- **G2 equivariance unit test** (repo standing rule): new `tests/test_p4_step1.py` — pixel-config
  $C_{16}$ encoder exact at grid angles; **off-grid bilinear floor measured and recorded** (expected
  $\sim10^{-1}$, feeds $\hat\epsilon_{\max}$ later).
- **G3 probe-rig sanity**: probes $[D\to256\to128\to\cdot]$ for $(x,y,\theta)$ + $\Delta\theta$
  effect probe, trained on 40 held-out episodes, evaluated on 20 further held-out; sanity (NOT a
  claim): full-data $\theta$-$R^2 > 0.5$ for at least one base — the task is probeable at all.
- **G4 record**: `papers/paper3_record.md` created, honest-verdict format, step1 entry.

## Spec-level decisions (registered now)

- **Render 96×96** (cost; matches the human-demo row for later). R-lewm cells re-render at the LeWM
  checkpoint's native input when E-P4.2 reaches them (faithful = 224; decide there, not here).
- Fraction grid (episodes): $\{2,10,20,60,200\}$; probe budgets fixed (40 train / 20 eval held-out)
  identical across all cells.
- Wedge-restricted variants NOT in step1 (land with E-P4.2 / C1b prep).
- Seed-0 full grid only ($5$ bases $\times$ $5$ fractions); 3-seed replication is a later step once
  compute per cell is known.

## Build plan (fresh session, est. 3–5 h build + training wall-time)

`experiments/p4_step1_pipeline.py`: WeakPolicy collection → oracle-CEM demo generation → tensorize
(obs, act, nxt) → train $5\times5$ grid (seed 0) → probe rig → JSON artifact
(`p4_step1_probes.json`) + per-dim $R^2$-vs-fraction figure + record entry.
`tests/test_p4_step1.py`: G0a damping assert, G2 equivariance + floor, dataset shape/dtype
contracts, probe-rig shape contract. Deps: none new (everything vendored).

## Risks (pre-registered responses)

1. **WeakPolicy corpus too weak for θ-probeability** (G3 fails): the honest move is a protocol
   v1.2 amendment (e.g., mixing oracle-CEM frames into the encoder corpus) — logged and registered
   BEFORE any claim readout, never after.
2. **MPS float64 unsupported**: train f32 on MPS; all certificate audits stay CPU f64 (repo
   convention).
3. **Oracle-CEM slow**: offline generation, overnight acceptable; iCEM fallback.
4. **Param-matching ambiguity** (steerable vs plain conv params): match total trainable params
   within ±10%, report counts verbatim in the record.

## Open items carried (not step1's)

- "Dynamic Push-T" query (one targeted search at step2, cite-and-differentiate if physics-dialing).
- LeWM native render resolution check (at E-P4.2).
- 3-D lane (ManiSkill3 + VN-DGCNN, own venv, GPU box) — parallel track, descope-protected, separate
  seed spec.
