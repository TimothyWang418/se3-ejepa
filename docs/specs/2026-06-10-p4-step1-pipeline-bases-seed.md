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

## Protocol v1.2 amendments (registered 2026-06-10 post-diagnosis, BEFORE any 3-seed claim run)

Forced by the seed-0 readout + the step1b diagnosis (`papers/figures/p4_step1b_diagnosis.json`):

1. **R-aug v1.2** — the seed-0 aug collapse traced to an *implementation artifact*, not the Brehmer
   control's substance: `augment_x4` drew angles **per copy, not per sample** (the whole dataset
   had 4 discrete angles), and zeros-padding gave each angle a corner-wedge signature ⇒ 4-cluster
   collapse (high variance, perfectly predictable, content-free; the continuous-angle probe reads
   it at only 0.49 because 4-discrete-angles training is OOD for it). Fix: **per-sample angles +
   a fixed circular mask applied to ALL frames everywhere** (augmented and not, train and probe —
   metric fairness; kills the corner cue at the source). Re-run the step1b diagnosis after the fix;
   only then may R-aug enter any C2 reading.
2. **Transitions v1.2 — stride-5 action chunks** (the θ-readability response): both within-episode
   rotation (median 70.6°/episode — hypothesis "corpus lacks rotation" is DEAD) and training
   length (60 epochs made θ WORSE, −0.40, and xy decayed 0.70→0.44 — "undertraining" is DEAD)
   are exonerated; the live hypothesis is that 1-step prediction at 10 Hz never needs orientation
   (~0.7°/step). Amendment: transitions become **5-step action chunks** (the LeWM/FF-JEPA CHUNK
   convention; ~3.5° median motion per chunk makes orientation a needed signal). Predictor change:
   `CNRegularPredictor` lifts each of the 5 sub-actions separately (10 lifted fields; equivariance
   preserved per sub-action — the unit test extends accordingly).
3. **Content-decay-with-epochs registered as a monitored axis** (not a fix): linear readability
   decays with training (xy 0.70→0.44, θ +0.03→−0.40 at 20→60 epochs; the @200 column's 0.55→0.31
   drop is the same shape). Epochs stay 20; v1.2 runs log probe-vs-epoch curves. Relates to the
   repo's own Step-64 predictability–variance tension and PoR's temporal-collapse axis — possibly
   reportable, not yet theorized.
4. Corpus and all C2 gate definitions unchanged. **G3 sanity reruns under v1.2 before the 3-seed
   grid**; if θ remains unreadable under stride-5, the next registered lever is corpus surgery
   (oracle-demo frames — blocked on the G0c solver fix).

## Protocol v1.3 amendments (registered 2026-06-10 post-v1.2-readout, BEFORE any 3-seed run)

The v1.2 grid (`p4_step1_v12.json`) surfaced a **training-budget confound**: chunking cut
transitions/cell 5× while epochs stayed 20 ⇒ optimizer steps per cell collapsed at small
fractions (std 0.002–0.16 at @2–@20) and the fraction axis conflated data quantity with training
steps; jointly with the content-decay axis (@60/@200 most steps, xy turned negative), one
mechanism explains the board: **linear content peaks early in training and decays; cells differ
chiefly in step count.** Amendments:

1. **Equal optimizer-step budget per cell** (replaces fixed epochs): 3,000 steps per cell at batch
   64, every base, every fraction — the data axis becomes a pure data axis.
2. **Probe-vs-step curves become primary instrumentation**: probes evaluated at checkpoints
   $\{300, 1000, 3000\}$ steps per cell (the monitored decay axis is now measured, not suspected);
   the C2 readout uses the registered step count (3,000) — any early-stop-by-probe rule would
   require a *held-out probe-validation* design and is NOT adopted now (circularity guarded).
3. **aug continuous-angle check** (one probe): the v1.2 aug still posts suspiciously low pred_loss
   (0.0012 vs plain 0.35) with zero content — per-sample drawing removed the discrete clusters but
   the angle remains a shared o/o₂ channel (scene orientation: walls/goal zone). Probe aug@200 for
   the continuous angle; if confirmed, the honest R-aug conclusion hardens: **scene-rotation
   augmentation in JEPA injects a predictable nuisance channel by construction** — reportable, and
   the Brehmer-control comparison must say so rather than pretend an aug arm exists that doesn't.
4. G3 unchanged (θ still unreadable in v1.2 — the stride-5 bet alone was insufficient; the
   probe-vs-step curves will show whether θ ever peaks early before deciding on corpus surgery).

## Protocol v1.4 amendment (registered 2026-06-11 post-Stage-2κ, before any run)

The κ=0.8 lane showed block velocity is unobservable in single frames ⇒ non-Markov latent
dynamics drown both bases (normalized δ̂: eq's 6.3× κ=0 advantage inverts to 1.6× WORSE at
κ=0.8, confounded with partial variance collapse, latent_std 0.562).

1. **Frame-pair encoder input** for κ>0 bases: channel-stack $(o_{t-1}, o_t)$ (6-ch input; both
   frames rotate together ⇒ steerable equivariance trivially preserved; ConvEncoder likewise).
   Restores Markovianity; the v1.4 run answers: *with velocity observable, which side of 1× does
   the normalized δ̂ ratio land on, per regime?*
2. **Registered cross-base normalizer**: δ̂ comparisons across bases are quoted normalized by
   (per-dim latent_std × $\sqrt D$), raw values alongside; the normalizer choice is reportable.
3. Recipe-stability check at κ>0 (latent_std floor) reported per base before any moat statement.

## Protocol v1.5 (registered 2026-06-11 post-v1.4: the stability pass — now the binding constraint for BOTH lanes)

Four collapse observations (content decay with steps; @2-fraction collapse; eq@κ0.8 std 0.562;
eq/6-ch std 0.307) converge on the recipe's variance–predictability tension, with a directional
pattern: **eq always collapses before plain.** Two registered hypotheses, ordered:

- **H-v1.5a (isotropy conflict — the sharper one, from the 06-11 review):** the variance floor
  `relu(1 − std)` enforces per-dim std = 1 — an ISOTROPY prior. The eq latent is a $C_{16}$
  regular representation whose content is naturally anisotropic across isotypic components;
  forcing isotropy pushes noise into structurally-empty dimensions and fights the equivariant
  structure — plain has no structure to fight, hence survives. **Principled fix candidate: an
  isotypic-aware variance floor** (per-field norm floor, or per-isotypic-component variance) —
  connects directly to paper2's Prop 4 (isotypic placement).
- **H-v1.5b:** `predictability_gated_var=True` (the repo's own Step-64 lever for exactly this
  tension), and/or var_coef adjustment.

Protocol: small sweep {floor-type × gating} on eq/6-ch@κ0 (the worst collapse cell), **gate:
latent_std ≥ 0.7 (or per-field equivalent) on ALL cells** before the moat question, the
in-jurisdiction cell, or any C2 probe is re-asked. Equivariance unit tests re-run for any new
floor (a per-field norm floor is manifestly equivariant; assert anyway).

## Open items carried (not step1's)

- "Dynamic Push-T" query (one targeted search at step2, cite-and-differentiate if physics-dialing).
- LeWM native render resolution check (at E-P4.2).
- 3-D lane (ManiSkill3 + VN-DGCNN, own venv, GPU box) — parallel track, descope-protected, separate
  seed spec.
