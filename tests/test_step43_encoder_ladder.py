r"""Step 43 — the **encoder-capacity ladder + lossless ORACLE bypass** (localise the residual cap).

Steps 32 (predictor degree) and 42 (message content) both saturated ~0.2-0.27 in-distribution relMSE,
far above the unconstrained MLP's ~0.074. By elimination the residual interaction cap should be the
shared encoder's **lossy** translation-invariant latent (it pools $P{=}24$ points per object onto
$N_{\text{out}}{=}16$ type-1 vectors). Step 43 tests that by (A) sweeping the encoder's internal
capacity at a fixed latent budget and (B) bypassing the encoder with a **lossless, parameter-free**
oracle — the per-object centred point cloud $\tilde x_k=x_k-\bar x$ — fed to the SAME degree-3 VN-TP
predictor. Whether the oracle *recovers* the cap is the experiment's empirical question
(``experiments/step43_encoder_ladder.py``); this test certifies the three structural invariants the
experiment relies on:

1. **Joint $\mathrm{SE}(3)\rtimes S_O$ equivariance** (to the float floor) of the WHOLE pipeline at
   **every** equivariant variant — the four encoder rungs E0..E3 (lmax/mul swept) AND both oracle
   variants (raw / unit message) — with the message recomputed from the transformed scene. So
   bypassing the encoder, like enriching the message in Step 42, does **not** spend equivariance: the
   across-group [G] flatness and the [A'] exactness claims hold for the oracle too. The MLP-MP anchor is
   verified to be the genuine *non*-equivariant control (large SE(3) residual, but permutation-exact
   from its shared-weight slots).

2. **Oracle = lossless + equivariant** — the two properties that make it the decisive control. The
   centred-point latent (a) reproduces $S-\bar x$ exactly (a *lossless* witness: it carries the full
   per-object geometry, $D_{\text{obj}}{=}P\cdot 3{=}72 > 48$, with zero learned compression), and
   (b) transforms as $\tilde x_k(RS{+}t)=R\,\tilde x_k(S)$ to the float floor (translation-invariant,
   $\mathrm{SO}(3)$-equivariant). Its predictor is a degree-3-capable :class:`VNTPPredictor` (two
   cross-product blocks) — the SAME class the encoder rungs use — so the oracle test is **not**
   confounded by predictor degree: only the latent's losslessness differs.

3. a **unit-message witness** (shared with Step 42): $\hat r$ is exactly unit-norm and scale-invariant
   ($\hat r(\lambda r)=\hat r(r)$ for $\lambda>0$), the normalisation the homogeneous predictor cannot
   synthesise — which ORACLE-unit supplies on top of the lossless geometry (its best case).

Run:
    .venv/bin/python tests/test_step43_encoder_ladder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step43_encoder_ladder import (  # noqa: E402
    ENC_RUNGS,
    MLP_ANCHOR,
    NAMES,
    ORACLES,
    P,
    VARIANTS,
    VN_VARIANTS,
    CenteredPointEncoder,
    N_OBJ,
    _d_obj,
    _unit,
    build_model,
    make_interacting_transitions,
    perm_resid,
    rand_so3,
    rotate_points,
    se3_resid,
)
from src.models.structured import VNTPPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    print("Step 43 — encoder ladder + ORACLE: joint SE(3)|xS_O equivariance + lossless-oracle witness\n")

    # a small batch of interacting scenes + a global (R, t)
    S, A_self, _ = make_interacting_transitions(48, seed=0)
    R = rand_so3(gen)
    t = torch.randn(3, generator=gen)

    # ---- 1. whole-pipeline equivariance at every variant (at init) ----------------------
    print(f"  {'variant':>11s} | {'config':>18s} | {'SE(3) resid':>12s} | {'perm resid':>11s}")
    print("  " + "-" * 62)
    for nm in NAMES:
        torch.manual_seed(0)                       # identical init across variants for a clean read
        model = build_model(nm)
        s = se3_resid(nm, model, S, A_self, R, t)
        p = perm_resid(nm, model, S, A_self)
        print(f"  {nm:>11s} | {VARIANTS[nm]['label']:>18s} | {s:12.2e} | {p:11.2e}")
        if nm == MLP_ANCHOR:
            # the non-equivariant control: SE(3) must genuinely BREAK (else it is not a control),
            # while its shared-weight slots keep permutation exact.
            assert s > 1e-1, f"MLP-MP should NOT be SE(3)-equivariant (it is the control), got {s:.2e}"
            assert p < 1e-4, f"MLP-MP shared slots should stay permutation-exact, got {p:.2e}"
        else:
            assert s < 1e-3, f"variant {nm} broke global SE(3) equivariance: {s:.2e}"
            assert p < 1e-3, f"variant {nm} broke permutation equivariance: {p:.2e}"

    # ---- 2. ORACLE witness: lossless AND exactly SE(3)-equivariant ----------------------
    print("\n  ORACLE witness (the two properties that make it the decisive control):")
    enc = CenteredPointEncoder(N_OBJ, P)
    z = enc(S).reshape(S.shape[0], N_OBJ, P, 3)         # (B,O,P,3) the centred-point latent
    centred = S - S.mean(dim=2, keepdim=True)            # the definition: per-object centred cloud
    lossless_err = (z - centred).abs().max().item()      # latent reproduces the full geometry exactly
    # equivariance of the latent itself: tilde_x(R S + t) == R tilde_x(S)
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    z_r = enc(Sr).reshape(S.shape[0], N_OBJ, P, 3)
    z_rot = rotate_points(z, R)                          # R applied to every latent 3-vector
    equiv_err = (z_r - z_rot).abs().max().item()
    d_obj_oracle = _d_obj("ORACLE-unit")
    d_obj_enc = _d_obj("E0-base")
    print(f"    |latent - (S - centroid)| = {lossless_err:.2e}   (oracle latent IS the full geometry -> lossless)")
    print(f"    |latent(RS+t) - R latent| = {equiv_err:.2e}   (translation-invariant + SO(3)-equivariant)")
    print(f"    D_obj: oracle {d_obj_oracle} > encoder {d_obj_enc}   (P*3 lossless vs 16-vector compressed budget)")
    assert lossless_err < 1e-5, f"oracle latent should equal the centred cloud, got {lossless_err:.2e}"
    assert equiv_err < 1e-5, f"oracle latent should be SE(3)-equivariant, got {equiv_err:.2e}"
    assert d_obj_oracle > d_obj_enc, "oracle latent must be wider (lossless) than the encoder budget"

    # the oracle predictor must be degree-3 capable (two TP blocks), the SAME class the encoder rungs
    # use -> the oracle isolates the LATENT, not the predictor degree (Step 32 already swept that).
    for nm in ORACLES + ("E0-base",):
        pred = build_model(nm).predictor.obj
        assert isinstance(pred, VNTPPredictor), f"{nm} predictor must be a (degree-3) VNTPPredictor"
    print("    oracle predictor = VNTPPredictor (two cross-product blocks, degree-3 capable),"
          " same class as the encoder rungs => predictor degree is NOT the confound")

    # ---- 3. unit-message witness (the normalisation ORACLE-unit supplies; shared with Step 42) ----
    print("\n  unit-message witness (the normalisation the VN predictor cannot form, fed to ORACLE-unit):")
    c = S.mean(dim=2)                                    # (B,O,3) centroids
    r = c[:, 1] - c[:, 0]                                # (B,3) a relative vector
    rhat = _unit(r)
    unit_err = (rhat.norm(dim=-1) - 1.0).abs().max().item()
    lam = 3.0
    rhat_scaled = _unit(lam * r)
    rhat_scale_drift = (rhat_scaled - rhat).abs().max().item()
    raw_scale_drift = (lam * r - r).abs().max().item()
    print(f"    ||r_hat|| - 1            = {unit_err:.2e}   (r_hat is exactly unit-norm)")
    print(f"    |r_hat(3r) - r_hat(r)|   = {rhat_scale_drift:.2e}   (r_hat is scale-INVARIANT)")
    print(f"    |3r - r|                 = {raw_scale_drift:.2e}   (raw r is NOT -- it carries ||r||)")
    assert unit_err < 1e-5, f"r_hat should be unit-norm, got {unit_err:.2e}"
    assert rhat_scale_drift < 1e-5, f"r_hat should be scale-invariant, got {rhat_scale_drift:.2e}"
    assert raw_scale_drift > 1e-1, "raw r should change under scaling (it carries the magnitude)"

    print(f"\n    variants checked: encoder rungs {list(ENC_RUNGS)} + oracles {list(ORACLES)} "
          f"({len(VN_VARIANTS)} equivariant) + {MLP_ANCHOR} (control)")
    print("\nPASS: every equivariant variant (encoder ladder AND lossless oracle) keeps the pipeline exactly")
    print("SE(3)|xS_O-equivariant (< 1e-3); the oracle latent is the lossless, exactly-equivariant centred")
    print("cloud fed to a degree-3 predictor => Step 43 isolates the ENCODER's latent losslessness cleanly,")
    print("with predictor degree (Step 32) and message normalisation (Step 42) held as non-confounds.")
    print("Whether the lossless latent RECOVERS the in-distribution cap is the experiment's question.")


def test_step43_encoder_ladder_equivariance_and_oracle_witness() -> None:
    """pytest: every variant stays SE(3)|xS_O-equivariant; the oracle is lossless + equivariant."""
    main()


if __name__ == "__main__":
    main()
