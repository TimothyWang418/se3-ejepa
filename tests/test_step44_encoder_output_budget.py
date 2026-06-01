r"""Step 44 — the **encoder OUTPUT-budget sweep** (readout width vs the upstream pooling cap).

Steps 32 (predictor degree), 42 (message content) and 43A (encoder *internal* capacity) all saturated
~0.2 in-distribution relMSE; the lossless oracle (43B) solved the task. Step 43 named the one un-pulled
lever — the encoder's **output budget** (the readout ``lin_out: h -> 3*n_out_vec``). Step 44 pulls it,
sweeping ``n_out_vec in {16,24,32,48}`` at FIXED internals (``lmax=2, mul=8``). Whether widening the
readout drops the floor or saturates at the pooled-descriptor width is the experiment's empirical
question (``experiments/step44_encoder_output_budget.py``); this test certifies the structural invariants
the experiment relies on:

1. **Joint $\mathrm{SE}(3)\rtimes S_O$ equivariance** (to the float floor) of the WHOLE pipeline at
   **every** equivariant variant — all four output-budget rungs B16..B48 AND the lossless oracle — with
   the message recomputed from the transformed scene. So widening the output budget, like enriching the
   message (Step 42) or bypassing the encoder (Step 43), does **not** spend equivariance: the
   across-group [G] flatness and [A'] exactness claims hold at every budget. The MLP-MP anchor is verified
   to be the genuine *non*-equivariant control (large SE(3) residual, permutation-exact slots).

2. **The sweep moves ONLY the readout, downstream of a FIXED pooled descriptor.** :class:`SE3PointEncoder`
   sum-pools the $P$ per-point messages into a hidden descriptor $h$ of width
   $\dim(h)=\mathrm{mul}\cdot\sum_{\ell\le\ell_{\max}}(2\ell+1)$, then reads out $3\,n_{\text{out}}$ via a
   linear ``lin_out``. We assert (a) every budget rung's encoder has the SAME ``irreps_hidden.dim`` (the
   pool is fixed; only ``lin_out`` widens), and (b) the rung's flattened latent width equals
   $N_{\text{obj}}\cdot3\,n_{\text{out}}$ as declared — so a rung labelled ``D48/D72/D96/D144`` really has
   that per-object width and its VN-TP predictor is wired to match. This is what makes Step 44 a clean
   *output*-budget probe (vs Step 43A's *internal*-capacity probe).

3. **The B24-vs-oracle width coincidence** that makes the sweep's crispest contrast meaningful: the
   $n_{\text{out}}{=}24$ rung has per-object width $72 = P\cdot3$, exactly the lossless oracle's width —
   so B24-vs-oracle is *ordered-lossless vs pooled-lossy at the same width*, not a width difference.

Run:
    .venv/bin/python tests/test_step44_encoder_output_budget.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step44_encoder_output_budget import (  # noqa: E402
    BUDGET_RUNGS,
    HIDDEN_DIM,
    MLP_ANCHOR,
    NAMES,
    ORACLES,
    P,
    VARIANTS,
    VN_VARIANTS,
    CenteredPointEncoder,
    N_OBJ,
    _d_obj,
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

    print("Step 44 — output-budget sweep: joint SE(3)|xS_O equivariance + readout-only / fixed-pool witness\n")

    # a small batch of interacting scenes + a global (R, t)
    S, A_self, _ = make_interacting_transitions(48, seed=0)
    R = rand_so3(gen)
    t = torch.randn(3, generator=gen)

    # ---- 1. whole-pipeline equivariance at every variant (at init) ----------------------
    print(f"  {'variant':>11s} | {'config':>16s} | {'SE(3) resid':>12s} | {'perm resid':>11s}")
    print("  " + "-" * 60)
    for nm in NAMES:
        torch.manual_seed(0)                       # identical init across variants for a clean read
        model = build_model(nm)
        s = se3_resid(nm, model, S, A_self, R, t)
        p = perm_resid(nm, model, S, A_self)
        print(f"  {nm:>11s} | {VARIANTS[nm]['label']:>16s} | {s:12.2e} | {p:11.2e}")
        if nm == MLP_ANCHOR:
            # the non-equivariant control: SE(3) must genuinely BREAK (else it is not a control),
            # while its shared-weight slots keep permutation exact.
            assert s > 1e-1, f"MLP-MP should NOT be SE(3)-equivariant (it is the control), got {s:.2e}"
            assert p < 1e-4, f"MLP-MP shared slots should stay permutation-exact, got {p:.2e}"
        else:
            assert s < 1e-3, f"variant {nm} broke global SE(3) equivariance: {s:.2e}"
            assert p < 1e-3, f"variant {nm} broke permutation equivariance: {p:.2e}"

    # ---- 2. the sweep moves ONLY the readout, downstream of a FIXED pooled descriptor ----
    print("\n  readout-only witness (n_out_vec sweeps lin_out; the pooled descriptor h is held fixed):")
    print(f"    {'rung':>11s} | {'n_out_vec':>9s} | {'dim(h) pool':>11s} | {'3*n_out readout':>15s} | {'latent_dim':>10s}")
    print("    " + "-" * 66)
    for nm in BUDGET_RUNGS:
        k = VARIANTS[nm]["n_out_vec"]
        enc = build_model(nm).encoder.obj_enc            # the shared per-object SE3PointEncoder
        pool_dim = enc.irreps_hidden.dim                  # width of the summed descriptor h (fixed knob)
        readout = 3 * k                                   # lin_out output width (the swept knob)
        latent = build_model(nm).encoder.latent_dim       # scene latent = N_OBJ * 3 * n_out_vec
        print(f"    {nm:>11s} | {k:>9d} | {pool_dim:>11d} | {readout:>15d} | {latent:>10d}")
        # (a) the pooled descriptor is the SAME for every rung -> only the readout widens
        assert pool_dim == HIDDEN_DIM, (
            f"{nm}: pooled descriptor dim {pool_dim} != fixed {HIDDEN_DIM}; the sweep must move ONLY "
            f"the readout, not the internal pool (that was Step 43A)")
        # (b) the declared per-object / scene width really is 3*n_out_vec (predictor is wired to match)
        assert _d_obj(nm) == readout, f"{nm}: per-object width {_d_obj(nm)} != 3*n_out_vec {readout}"
        assert latent == N_OBJ * readout, f"{nm}: scene latent {latent} != N_OBJ*3*n_out_vec {N_OBJ * readout}"
        # (c) the predictor is the SAME degree-3 VN-TP class, just wider -> only the budget differs
        pred = build_model(nm).predictor.obj
        assert isinstance(pred, VNTPPredictor), f"{nm} predictor must be a (degree-3) VNTPPredictor"
    print(f"    => dim(h)={HIDDEN_DIM} fixed across all rungs; only lin_out (3*n_out_vec) and the matched")
    print(f"       VN-TP predictor width change. Step 44 probes the OUTPUT budget, not internal capacity.")

    # ---- 3. the B24-vs-oracle width coincidence (ordered-lossless vs pooled-lossy at SAME width) ----
    print("\n  B24-vs-oracle witness (same width, different latent quality):")
    enc_oracle = CenteredPointEncoder(N_OBJ, P)
    z = enc_oracle(S).reshape(S.shape[0], N_OBJ, P, 3)
    centred = S - S.mean(dim=2, keepdim=True)
    lossless_err = (z - centred).abs().max().item()
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    z_r = enc_oracle(Sr).reshape(S.shape[0], N_OBJ, P, 3)
    equiv_err = (z_r - rotate_points(z, R)).abs().max().item()
    d_b24 = _d_obj("B24-nout24")
    d_oracle = _d_obj("ORACLE-unit")
    print(f"    |oracle latent - centred cloud| = {lossless_err:.2e}   (oracle IS the full ordered geometry)")
    print(f"    |oracle(RS+t) - R oracle|       = {equiv_err:.2e}   (translation-inv + SO(3)-equivariant)")
    print(f"    D_obj: B24 {d_b24} == oracle {d_oracle}   (SAME width -> B24 vs oracle isolates pooled-vs-ordered)")
    assert lossless_err < 1e-5, f"oracle latent should equal the centred cloud, got {lossless_err:.2e}"
    assert equiv_err < 1e-5, f"oracle latent should be SE(3)-equivariant, got {equiv_err:.2e}"
    assert d_b24 == d_oracle == P * 3, f"B24 ({d_b24}) and oracle ({d_oracle}) should both be P*3={P * 3}"

    print(f"\n    variants checked: output-budget rungs {list(BUDGET_RUNGS)} + oracle {list(ORACLES)} "
          f"({len(VN_VARIANTS)} equivariant) + {MLP_ANCHOR} (control)")
    print("\nPASS: every output-budget rung (and the lossless oracle) keeps the pipeline exactly")
    print("SE(3)|xS_O-equivariant (< 1e-3); n_out_vec sweeps ONLY the readout downstream of a FIXED")
    print(f"dim(h)={HIDDEN_DIM} pool; B24 matches the oracle's width so B24-vs-oracle isolates pooled-lossy")
    print("vs ordered-lossless. Whether widening the readout RECOVERS the cap is the experiment's question.")


def test_step44_output_budget_equivariance_and_readout_only_witness() -> None:
    """pytest: every budget rung stays SE(3)|xS_O-equivariant; the sweep moves only the readout."""
    main()


if __name__ == "__main__":
    main()
