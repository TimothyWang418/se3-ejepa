# RunPod deploy — step101 bridge (online TD-MPC2, eq vs dense). READINESS CHECKLIST.

Spec: `docs/specs/2026-06-13-step101-online-tdmpc2-bridge-seed.md`. Principle (B300 post-mortem): rent for
**parallelism across K×2 runs**, not per-run speed (env-bound; 4090 ≈ 3080 per run). Develop+smoke LOCAL
first; rented GPUs run only validated full-length jobs.

## Pre-rental (local, free — IN PROGRESS / next build block)

- [ ] `EqWorldModel` written (swap `_encoder`/`_dynamics` to S₂-tied; reward/Q invariant; π equivariant) —
      integration points in the spec. Reuses step89 NormedLinear/SimNorm + step100b tying + step100 ρ-action.
- [ ] G-EQ passes locally (eq equivariance defect ≤ 1e-5).
- [ ] Local smoke: ~5k steps on Mac/3080, confirm it trains + task return rises + latent λ trends → 0.25.
      ONLY after this green do we rent.

## Rental (you: spin up; me: deploy + launch)

1. **Instances**: ~5 × RunPod "PyTorch 2.5 / CUDA 12.4" Community Cloud (4090 or 3090, ~$0.3–0.5/h).
   Use **/root** local disk, not the /workspace network volume. Give me ssh (host/port/key) per instance.
2. **Transfer** (NOT the SSH proxy — 0.4MB/s trap): use `runpodctl`:
   - local:  `runpodctl send se3-ejepa-bridge.tar.gz`  → prints a code
   - pod:    `runpodctl receive <code>`  (~12MB/s)
   The tarball = `external/tdmpc2` (with the EqWorldModel patch) + `scripts/runpod_bridge_setup.sh`.
3. **Per-instance setup**: `bash runpod_bridge_setup.sh` (one-time, ~3–5 min).
4. **Launch one (arm,seed) per instance** (MUJOCO_GL=egl), e.g.:
   ```
   cd tdmpc2/tdmpc2 && MUJOCO_GL=egl python train.py task=walker-walk steps=1_000_000 \
       exp_name=bridge_<arm>_s<seed> seed=<seed> model=<eq|dense>   # model flag added by the patch
   ```
   K=5 seeds × {eq,dense} = 10 runs over ~2 waves of 5 → ~one-run wall-clock (~4–6 h).
5. **Pull back** the 10 checkpoints via `runpodctl send` (pod→local), then **release the pods** (don't leave
   them billing — the B300 lesson).

## Post (local, free)

- [ ] Audit all 10 checkpoints with the step89/step100 machinery (Benettin λ₁ + measured crossings).
- [ ] G-BR-CAL / G-BR-VAR / G-BR-TAME / G-BR-PERF per spec; bootstrap CI on the eq-vs-dense variance ratio.
- [ ] Honest verdict → E17 (appendix) or recorded null.

## Cost / time / payoff (eyes open)

- Cost: ~5 GPU × ~$0.4/h × ~6 h ≈ **$12–15** (K=5); double for K=10.
- Time: ~1 day fork+smoke (local) + ~6 h rented + audit.
- Payoff: **uncertain** — offline showed equivariance doesn't help one-step accuracy; the bridge bets on
  cross-seed calibration *reliability* (E12 mechanism) holding online. Real chance of an honest null.
