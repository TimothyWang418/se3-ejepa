# Step 91 — LeWM audit (E13's second model family) — seed spec

*Date: 2026-06-10 · Status: seed (recon DONE, build-ready) · Owner: SE3-EJEPA / Certified World Models · Compute: Mac CPU (float64 QR), no GPU needed*

> Extends E13/step89 (TD-MPC2 scope map) to a SECOND public pretrained family: **LeWM** (Maes, Le Lidec, Scieur,
> LeCun, Balestriero; arXiv:2603.19312). Recon verified (sub-agent, 2026-06-10): **GO-with-caveats**. Two model
> families × the same training-free Jacobian certificate = E13 upgrades from "a model" to "a methodology."

## Verified facts (recon)

- **Checkpoints**: HF `quentinll/lewm-{pusht,cube,tworooms,reacher}`, each `weights.pt` (72.3 MB, plain state dict,
  `weights_only=True` loadable) + `config.json` (full Hydra spec). **MIT.**
  URL: `https://huggingface.co/quentinll/lewm-pusht/resolve/main/weights.pt`.
- **Architecture (audit-relevant)**: encoder = HF **ViT-tiny** (hidden 192, 12L/3H, patch 14, img 224, CLS token) →
  `projector` MLP 192→2048→192 (BatchNorm1d) ⇒ $z\in\mathbb{R}^{192}$. Transition = `Predictor` transformer (depth 6,
  heads 16, dim_head 64, mlp 2048; causal SDPA; learnable pos-emb over **num_frames=3**; actions via **AdaLN**,
  zero-init) → `pred_proj` MLP 192→2048→192 (BN). Action = 5-step chunk ×2 dims = 10-d → `Embedder`
  (Conv1d k1 + Linear–SiLU–Linear). **Deterministic + differentiable in eval** (dropout off, BN=affine; SDPA math
  kernel in float64). No SimNorm/sphere on the latent.
- **Canonical code**: `stable-worldmodel` (PyPI 0.1.1) `stable_worldmodel/wm/lewm/{lewm.py,module.py}`; state-dict
  prefixes `encoder./predictor./action_encoder./projector./pred_proj.`. Encoder: use `transformers.ViTModel` directly
  (the upstream code path — more faithful than a replica).
- **Audit loop**: delay-embedded state $s_t=(z_{t-2},z_{t-1},z_t)\in\mathbb{R}^{576}$; autonomous map
  $g(s)=\mathrm{shift}(s)\oplus d(s,a^\*)$ with a FIXED action chunk ($a^\*{=}0\in[-1,1]^{10}$ in-support, or
  dataset-mean). 576-d Benettin-QR = step89's cost class. **No policy prior exists** (LeWM plans by CEM).
- **Measured side**: PushT env is pymunk+pygame+cv2, pip-clean on macOS (`swm/PushT-v1`, render 224 = encoder input);
  roll true env under the same fixed chunks, encode frames, divergence horizon. No 13 GB dataset needed.

## Pre-registered scope & readings (do NOT drift)

1. **Scope caveat up front**: the certificate covers the fixed-action autonomous loop — weaker than step89's policy
   loop $g(z)=d(z,\tanh\mu_\pi(z))$; LeWM has no policy prior, so this IS its natural free-running mode. State it.
2. **Structural bands expected**: companion/shift structure ⇒ ~2/3 of the 576-d spectrum is shift-copy artifact
   (analogue of step89's SimNorm band) + possible near-zero BN scales. Report, don't hide.
3. **G0 (strict-load + sanity)**: `strict=True` state-dict load on the real checkpoint AND a one-step prediction
   sanity check vs the upstream `rollout` semantics (sliding window; verify action-chunk normalization raw-vs-stats)
   BEFORE certifying. Any mismatch = stop, fix the replica, never fudge.
4. **Readings (same taxonomy as E13)**: $\lambda_1>0$ ⇒ certified $T_1(\epsilon)$ vs measured divergence (encoder on
   true rollouts) — calibrated/optimistic per amplification-vs-bias; $\lambda_1\le0$ ⇒ abstain, classify
   stable-vs-bias-driven by censoring. Any cell outcome is content; ratios at $\epsilon\in\{0.05,0.1,0.2\}$.
5. **Order**: PushT first (1 ckpt, 72 MB); cube/reacher (MuJoCo, also macOS-fine) as follow-ups if PushT lands.

## Build plan (fresh session, est. 1.5–2.5 h)

`experiments/step91_lewm_audit.py` mirroring step89's skeleton: slice loader (ViTModel + faithful
Predictor/Embedder/MLP replicas from `module.py`) → G0 strict-load + one-step sanity → 576-d QR certificate
(`step78` unchanged) → PushT true-rollout divergence → JSON + fold into E13 ("two families") + record.
Deps to add: `transformers`, `stable-worldmodel` (or vendor `envs/pusht/env.py` + `expert_policy.py`).
Risks (recon): ViTConfig drift (qkv-bias/LN-eps — strict-load catches), action normalization (sanity check catches),
BN float64 conversion. `tests/test_step91.py`: replica-vs-upstream one-step parity on random weights + load contract.

## Why it matters

E13 currently audits one family (TD-MPC2, MLP/SimNorm latents). LeWM is architecturally disjoint (ViT + transformer
predictor + BN, JEPA-trained, pixel-based) and from the lab whose research program ours speaks to. Same read-out
working across both ⇒ "training-free trustworthiness audit" becomes a *method*, not a trick — and either outcome
(calibrated/optimistic/abstain) extends the scope map. Honest confidence: ~0.6 the audit lands cleanly on PushT
(replica-fidelity is the main risk), ~0.9 that whatever lands is publishable content for E13.
