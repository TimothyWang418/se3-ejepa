# P4-step3 — `wm_audit` gap mode: $(\hat\delta, \hat\lambda_1, \hat\epsilon_{\max})$ of a trained base along the demo distribution — seed spec

*Date: 2026-06-10 · Status: seed (build next session, consumes step1 checkpoints) · Owner: paper3 ·
Compute: Mac CPU f64 (audit convention), minutes per base*

> The spine's certificate $\widehat{\mathrm{Err}}(H) = \hat\delta \sum_{t<H} e^{\hat\lambda_1 t}$
> (+ Thm B's $m\,\hat\epsilon_{\max}$ for group-transported claims) consumes exactly three measured
> quantities. This step builds the instrument that measures them — the action-conditioned,
> demo-distribution analogue of step89/91's free-running audit. Prop 11 alignment is the design
> axiom again: the instrument measures precisely what $H^*(\epsilon)$ consumes.

## Inputs / API (registered)

`audit_gap(model, frames, actions, *, group=None) -> dict` — new module (do **not** churn
`scripts/wm_audit.py`'s paper2 surface mid-submission; the gap mode imports its Benettin/QR
internals where importable, else self-contains them). Bases come from step1 checkpoints
(`data/p4_step1/ckpt_*.pt`; the original overnight run predates checkpointing — re-materialize by
seed-determinism if needed, registered in the record).

## The three measurements (registered definitions)

1. **$\hat\delta$ (one-step bias):** $\lVert f(E(o_t), a_t) - E(o_{t+1})\rVert$ over held-out demo
   transitions; report mean, q50, q90 + moving-block-bootstrap CI. Metric = raw latent norm (the
   certificate consumes the same metric); z-scored variant reported alongside for cross-base
   context, never substituted.
2. **$\hat\lambda_1$ (leading exponent of the action-conditioned loop):** Benettin/JVP along demo
   action sequences — windows of length $W{=}40$, leading band $k{=}4$, float64 CPU (step89/91 cost
   class); per-window finite-time exponents → mean + Prop 10 CI. Units: per control step.
3. **$\hat\epsilon_{\max}$ (equivariance residual, eq base only):**
   $\max_g \lVert E(g\cdot x) - \rho(g) E(x)\rVert$ over data, $g$ ranging over $C_N$ generators
   ($\rho$ = e2cnn's transform; grid angles exact, off-grid carries the documented bilinear floor —
   report both). For plain/aug bases **no canonical $\rho$ exists (Lemma 2)** — their
   wedge-transfer cells are evaluated by direct in/out-of-wedge $\hat\delta$ contrast instead
   (C1b's registered design); the instrument records `eps_max: null` for them, never a fake number.

## Instrument validation gates (pre-registered; the instrument is certified before it certifies)

- **G-I (known-system recovery):** a synthetic linear latent loop with planted spectrum
  ($\lambda_1^{\text{true}}$ set by construction) — recovered within the Prop 10 CI, 3/3 planted
  values spanning $\{-0.05, 0, +0.08\}$ (the κ-gate's measured range).
- **G-II (determinism):** bit-identical artifact across two runs.
- **G-III (orbit-invariance witness, eq base):** $\hat\lambda_1$ on $g$-rotated demo data equals
  the unrotated value within CI (a free consistency check of both instrument and equivariance).

## Outputs

Per (base, κ) JSON: the triple with CIs + the certified curve $\widehat{\mathrm{Err}}(H)$ on
$H \in \{1..60\}$ (+ the $m\,\hat\epsilon_{\max}$ band for eq) → these curves ARE the certificate
side of E-P4.3's feasibility comparison. Artifact `papers/figures/p4_step3_gap_audit.json`.

## Order of consumption (post-step2 reality)

The spine's regime contrast is **static vs κ=0.8** (step2 verdict FALLBACK-2PT). Gap mode runs on:
eq/plain-match/aug @ κ=0 full-fraction bases (step1) first; κ=0.8 bases require a step1-style
training pass at κ=0.8 (WeakPolicy corpus regenerates there by construction; **the κ>0 oracle-CEM
demo debt — 10-d extended-state setter — falls due before G-training at κ=0.8, not before gap
mode**).

## Risks

JVP through e2cnn layers (basis-expansion autograd path unverified in forward-mode) — fallback:
finite-difference Jacobian-vector products at f64 (cost acceptable at latent dim 128, W=40);
recorded if used. MPS never touches the audit (f64 CPU convention).
