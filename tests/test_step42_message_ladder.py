r"""Step 42 — the **tensor-product MESSAGE ladder** (enrich the message, hold the predictor fixed).

Step 32 swept the *predictor's* representable degree and found the interaction-cap recovery saturates;
Step 42 instead enriches the **message** — the per-object action channel — and asks whether supplying
the unit relative direction $\hat r_{ij}=r_{ij}/\lVert r_{ij}\rVert$ recovers the capacity the predictor
degree cannot (it must form $1/\lVert r\rVert$, which is non-polynomial and homogeneity-breaking, so no
representable degree reaches it). The empirical recovery curve lives in
``experiments/step42_tp_message_ladder.py``; this test certifies the two structural invariants the
experiment relies on, for **every** message variant M0/M1/M2:

1. **Joint $\mathrm{SE}(3)\rtimes S_O$ equivariance** (to the float floor) of the WHOLE VN-TP pipeline
   — encoder + per-slot tensor-product predictor — with the message recomputed from the transformed
   scene. The message is a translation-invariant, $\mathrm{SO}(3)$-equivariant, permutation-covariant
   type-1 vector at every variant, so enriching it does **not** spend equivariance — the structural
   half of the "enrich the message, keep 举一反三 free" claim (whether that enrichment *recovers* the
   in-distribution cap is the experiment's empirical question, and Step 42 finds it does NOT);
2. a **message-content witness**: the unit message $\hat r$ is exactly unit-norm and **scale-invariant**
   ($\hat r(\lambda r)=\hat r(r)$ for $\lambda>0$), while the raw message $r$ is not — so M1 genuinely
   carries the normalisation $1/\lVert r\rVert$ that M0 lacks and that the homogeneous VN predictor
   cannot synthesise at any degree. (The teacher's $\hat r$ enters the cross product
   $\hat r\times a$ linearly, which is why M2 — raw + unit — is predicted to saturate.)

Run:
    .venv/bin/python tests/test_step42_message_ladder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step42_tp_message_ladder import (  # noqa: E402
    MESSAGES,
    VN_VARIANTS,
    _build_message,
    _unit,
    build_model,
    make_interacting_transitions,
    perm_resid,
    rand_so3,
    se3_resid,
)

torch.set_default_dtype(torch.float32)


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    print("Step 42 — message ladder: joint SE(3)|xS_O equivariance + message-content witness\n")

    # a small batch of interacting scenes + a global (R, t) and an aggressive scaling check
    S, A_self, _ = make_interacting_transitions(48, seed=0)
    R = rand_so3(gen)
    t = torch.randn(3, generator=gen)

    # ---- 1. whole-pipeline equivariance at every message variant (at init) ----
    print(f"  {'variant':>7s} | {'message':>13s} | {'SE(3) resid':>12s} | {'perm resid':>11s}")
    print("  " + "-" * 52)
    for nm in VN_VARIANTS:
        torch.manual_seed(0)                       # identical init across variants for a clean read
        model = build_model(nm)
        s = se3_resid(nm, model, S, A_self, R, t)
        p = perm_resid(nm, model, S, A_self)
        print(f"  {nm:>7s} | {MESSAGES[nm][2]:>13s} | {s:12.2e} | {p:11.2e}")
        assert s < 1e-4, f"variant {nm} broke global SE(3) equivariance: {s:.2e}"
        assert p < 1e-4, f"variant {nm} broke permutation equivariance: {p:.2e}"

    # ---- 2. message-content witness: r_hat is unit-norm and scale-invariant; raw r is neither ----
    print("\n  message-content witness (the normalisation the VN predictor cannot form):")
    c = S.mean(dim=2)                              # (B,O,3) centroids
    r = c[:, 1] - c[:, 0]                          # (B,3) a relative vector
    rhat = _unit(r)
    unit_err = (rhat.norm(dim=-1) - 1.0).abs().max().item()
    # scale the WHOLE scene about the origin by lambda>1: raw r scales, r_hat is invariant
    lam = 3.0
    r_scaled = lam * r
    rhat_scaled = _unit(r_scaled)
    rhat_scale_drift = (rhat_scaled - rhat).abs().max().item()
    raw_scale_drift = (r_scaled - r).abs().max().item()
    print(f"    ||r_hat|| - 1            = {unit_err:.2e}   (r_hat is exactly unit-norm)")
    print(f"    |r_hat(3r) - r_hat(r)|   = {rhat_scale_drift:.2e}   (r_hat is scale-INVARIANT)")
    print(f"    |3r - r|                 = {raw_scale_drift:.2e}   (raw r is NOT -- it carries ||r||)")
    assert unit_err < 1e-5, f"r_hat should be unit-norm, got {unit_err:.2e}"
    assert rhat_scale_drift < 1e-5, f"r_hat should be scale-invariant, got {rhat_scale_drift:.2e}"
    assert raw_scale_drift > 1e-1, "raw r should change under scaling (it carries the magnitude)"

    # ---- 3. M2 message strictly contains M0 and M1 content (raw + unit) ----
    m0 = _build_message(S, A_self, MESSAGES["M0-raw"][1])
    m1 = _build_message(S, A_self, MESSAGES["M1-unit"][1])
    m2 = _build_message(S, A_self, MESSAGES["M2-both"][1])
    assert m0.shape[-1] == m1.shape[-1], "M0 and M1 must have identical width (pure content swap)"
    assert m2.shape[-1] > m1.shape[-1], "M2 must be wider (it appends the raw magnitude back)"
    print(f"\n    widths: M0={m0.shape[-1]}  M1={m1.shape[-1]} (== M0, content swap)  M2={m2.shape[-1]} (raw+unit)")

    print("\nPASS: every message variant keeps the VN-TP pipeline exactly SE(3)|xS_O-equivariant (< 1e-4);")
    print("the unit message r_hat supplies the scale-invariant 1/||r|| the homogeneous predictor cannot form")
    print("=> the message can be ENRICHED without spending the equivariance prior. Whether that recovers")
    print("   in-distribution capacity is the experiment's question -- Step 42 finds it does NOT (x1.01,")
    print("   within seed noise), localising the residual interaction cap to the encoder's lossy latent.")


def test_step42_message_equivariance_and_witness() -> None:
    """pytest: every message variant stays SE(3)|xS_O-equivariant; r_hat supplies the normalisation."""
    main()


if __name__ == "__main__":
    main()
