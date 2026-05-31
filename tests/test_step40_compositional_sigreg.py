r"""Step 40 — compositional bi-block-SIGReg on an $S_O\times SO(3)$-equivariant scene latent.

These tests pin the *mechanism* behind ``experiments/step40_compositional_sigreg.py`` so the
experiment's compositional headline cannot silently rot. A scene of $O$ distinguishable objects
(each $n_0$ scalars + $n_1$ vectors) carries the outer-tensor representation
$P\boxtimes\rho_{SE3}$ of $S_O\times SO(3)$. Because $\mathbb R^O=\mathbb 1\oplus\mathbf{std}$,
the scene latent splits into **four** bi-isotypic blocks $(\mathbb 1,0e),(\mathbb 1,1o),
(\mathbf{std},0e),(\mathbf{std},1o)$ — Prop. 1'. The guards:

1. **Bibasis geometry is orthogonal.** The Helmert change-of-basis $U\otimes I$ (object axis)
   followed by the contiguous gather is an orthogonal map, so :func:`spectral_gauge`
   (eigenvalues) is basis-invariant. Block widths are exactly $(2,6,6,18)$ summing to $n=32$.
2. **Scene equivariance (init AND post-training).** The encoder is per-object SE(3)-equivariant
   (scalars invariant, vectors equivariant under a global $R$), $S_O$-permutation-equivariant,
   and translation-invariant — to the float floor, and LeJEPA training must not break it.
3. **SIGReg objectives finite + differentiable** (isotropic / se3-block / bi-block / budget /
   relational task) — they are trained against.
4. **bi-block-SIGReg discriminates** (anti-vacuity): flat on valid bi-block-isotropic laws at
   ANY 4-scale split, but spikes on a spatially-anisotropic $(\mathbf{std},1o)$ block (outside
   Prop. 1', $\operatorname{cov}\not\propto\mathbf I_3$).
5. **COMPOSITIONAL separation (the new rung).** On a within-type $S_O$ split (trivial vs
   standard scaled differently at fixed SO(3)-type budget), the Step-39 *se3*-block-SIGReg
   GROWS (it cannot represent the split), while bi-block stays flat — the objective-level proof
   that bi-block strictly refines se3-block on the compositional DOF.
6. **Deterministic gauge ladder.** se3-type law $\to\dim O(8)\times O(24)=304$; a 4-distinct-scale
   bi-type law $\to O(2)\times O(6)\times O(6)\times O(18)=184$, robust across the clustering
   ``gap_factor`` inside the separating window.
7. **Prop. 1' + negative control.** A FRESH equivariant encoder on a Haar+permute (hence
   $S_O\times SO(3)$-invariant) law is bi-block-isotropic (cross$\to0$); the SAME encoder on a
   FIXED-SLOT law (rotation-invariant but NOT $S_O$-invariant) FAILS (cross spikes) — [C] *can*
   fail, and fails exactly when the $S_O$ premise is removed.

Run:
    .venv/bin/python tests/test_step40_compositional_sigreg.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from experiments.step40_compositional_sigreg import (  # noqa: E402
    GAUGE_BI,
    GAUGE_COMMUTANT,
    GAUGE_ISO,
    GAUGE_SE3,
    N_OBJ,
    N_SCALAR,
    N_SCENE,
    N_VEC,
    _dim_on,
    _make_basis,
    build_scene_eq,
    build_scene_mlp,
    equiv_errs_scene,
    make_scenes,
    perm_err_scene,
    prop1_bi_metrics,
    rel_target_std,
    rel_task_loss,
    scene_bibasis,
    spectral_gauge,
    synth_biblock_iso,
    synth_spatial_aniso,
    to_isobasis,
    train_lejepa,
    trans_inv_err_scene,
)
from src.training.sigreg import (  # noqa: E402
    sigreg_block,
    sigreg_isotropic,
    variance_budget_penalty,
)


def _worst_scene_equiv(enc, S) -> tuple[float, float, float]:
    r"""Worst (scalar-inv, vector-equiv, perm) residual over several Haar $R$ and permutations."""
    inv, equ, pe = 0.0, 0.0, 0.0
    for k in range(5):
        R = o3.rand_matrix()  # global SO(3) — equivariance must hold for ALL R
        i, e = equiv_errs_scene(enc, S, R)
        sigma = torch.randperm(N_OBJ, generator=torch.Generator().manual_seed(100 + k))
        inv, equ, pe = max(inv, i), max(equ, e), max(pe, perm_err_scene(enc, S, sigma))
    return inv, equ, pe


# --------------------------------------------------------------------------- #
# [1] bibasis geometry: orthogonality, block widths, gauge-ladder arithmetic.
# --------------------------------------------------------------------------- #
def test_bibasis_is_orthogonal_and_well_shaped() -> None:
    r"""Helmert $U$ orthogonal; ``to_isobasis`` norm-preserving; widths $(2,6,6,18)$ sum $32$."""
    U, bi_perm, bi_blocks, (se3_perm, se3_blocks) = scene_bibasis()
    o = N_OBJ
    assert torch.allclose(U @ U.T, torch.eye(o), atol=1e-5), "Helmert matrix not orthogonal"
    # row 0 of Helmert is the (normalised) mean = the trivial S_O mode
    assert torch.allclose(U[0], torch.full((o,), 1.0 / o**0.5), atol=1e-5), "U row 0 != mean mode"

    # to_isobasis (U on the object axis, then gather) preserves the L2 norm of every row
    z = torch.randn(64, N_SCENE)
    zb = to_isobasis(z, U, bi_perm)
    assert torch.allclose(z.norm(dim=1), zb.norm(dim=1), atol=1e-4), "to_isobasis not norm-preserving"

    widths = [b.width for b in bi_blocks]
    assert widths == [2, 6, 6, 18] and sum(widths) == N_SCENE, f"bi widths {widths}"
    se3_widths = [b.width for b in se3_blocks]
    assert se3_widths == [8, 24] and sum(se3_widths) == N_SCENE, f"se3 widths {se3_widths}"
    # bi_perm / se3_perm are genuine permutations of range(n)
    assert sorted(bi_perm.tolist()) == list(range(N_SCENE)), "bi_perm not a permutation"
    assert sorted(se3_perm.tolist()) == list(range(N_SCENE)), "se3_perm not a permutation"


def test_gauge_ladder_arithmetic() -> None:
    r"""$O(32)=496\to O(8)\times O(24)=304\to O(2)O(6)O(6)O(18)=184\to O(2)^4=4$."""
    assert GAUGE_ISO == _dim_on(32) == 496, GAUGE_ISO
    assert GAUGE_SE3 == _dim_on(8) + _dim_on(24) == 304, GAUGE_SE3
    assert GAUGE_BI == _dim_on(2) + _dim_on(6) + _dim_on(6) + _dim_on(18) == 184, GAUGE_BI
    assert GAUGE_COMMUTANT == 2 * _dim_on(N_SCALAR) + 2 * _dim_on(N_VEC) == 4, GAUGE_COMMUTANT
    # the compositional rung is a strict drop the single-object (se3) block cannot reach
    assert GAUGE_ISO > GAUGE_SE3 > GAUGE_BI > GAUGE_COMMUTANT


# --------------------------------------------------------------------------- #
# [2] scene equivariance: global SO(3) + S_O permutation + translation, init AND post-train.
# --------------------------------------------------------------------------- #
def test_scene_equivariance_init_and_post() -> None:
    r"""Scalars invariant, vectors equivariant (global $R$), perm-equivariant, trans-invariant —
    at init and after the faithful LeJEPA loss (bi-block variant)."""
    torch.manual_seed(0)
    enc = build_scene_eq()
    S = make_scenes(48, seed=11, orient="haar", permute=True)
    inv0, equ0, pe0 = _worst_scene_equiv(enc, S)
    ti0 = trans_inv_err_scene(enc, S, torch.Generator().manual_seed(1))
    assert max(inv0, equ0, pe0, ti0) < 1e-4, (
        f"init equivariance broke: inv={inv0:.2e} equ={equ0:.2e} perm={pe0:.2e} trans={ti0:.2e}"
    )
    # a few epochs of faithful LeJEPA must not damage exact equivariance
    train_lejepa(enc, S, variant="bi", basis=_make_basis(), epochs=3, batch=24, seed=0)
    inv1, equ1, pe1 = _worst_scene_equiv(enc, S)
    assert max(inv1, equ1, pe1) < 1e-4, (
        f"post-train equivariance broke: inv={inv1:.2e} equ={equ1:.2e} perm={pe1:.2e}"
    )


def test_mlp_breaks_rotation_but_keeps_permutation() -> None:
    r"""Honest control: the slot-MLP is perm-equivariant + translation-invariant by construction,
    but has NO rotation prior — so global-$R$ equivariance must clearly fail."""
    torch.manual_seed(0)
    mlp = build_scene_mlp()
    S = make_scenes(48, seed=11, orient="haar", permute=True)
    minv, mequ, mpe = _worst_scene_equiv(mlp, S)
    assert mpe < 1e-4, f"MLP should still be perm-equivariant: {mpe:.2e}"
    assert max(minv, mequ) > 1e-2, f"MLP should NOT be rotation-equivariant: inv={minv:.2e} equ={mequ:.2e}"


# --------------------------------------------------------------------------- #
# [3] SIGReg objectives finite + differentiable.
# --------------------------------------------------------------------------- #
def test_sigreg_objectives_finite_and_differentiable() -> None:
    r"""Each regulariser / the relational task returns a finite scalar with a finite gradient."""
    torch.manual_seed(0)
    U, bi_perm, bi_blocks, (se3_perm, se3_blocks) = scene_bibasis()
    checks = [
        ("isotropic", lambda z: sigreg_isotropic(z, n_proj=32)),
        ("se3-block", lambda z: sigreg_block(to_isobasis(z, U, se3_perm), se3_blocks, n_proj=32)),
        ("bi-block", lambda z: sigreg_block(to_isobasis(z, U, bi_perm), bi_blocks, n_proj=32)),
        ("budget", lambda z: variance_budget_penalty(z)),
    ]
    for name, fn in checks:
        z = torch.randn(256, N_SCENE, requires_grad=True)
        val = fn(z)
        assert val.ndim == 0 and torch.isfinite(val) and val.item() >= 0.0, f"{name} not a finite >=0 scalar"
        val.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all(), f"{name} gradient non-finite"

    # relational task loss: differentiable through the latent in bibasis
    S = make_scenes(32, seed=3, orient="haar", permute=True)
    enc = build_scene_eq()
    z = to_isobasis(enc(S), U, bi_perm)
    z = z.detach().requires_grad_(True)
    y_rel = rel_target_std(S, U)
    t = rel_task_loss(z, y_rel, bi_blocks)
    assert t.ndim == 0 and torch.isfinite(t) and t.item() >= 0.0, "rel_task_loss not finite >=0 scalar"
    t.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all(), "rel_task_loss gradient non-finite"


# --------------------------------------------------------------------------- #
# [4] anti-vacuity: bi-block-SIGReg flat on valid laws, spikes on a non-Prop.1' law.
# --------------------------------------------------------------------------- #
def test_bi_block_sigreg_discriminates() -> None:
    r"""bi-block-SIGReg is flat on *valid* bi-block-isotropic laws at ANY 4-scale split, yet
    **spikes** on a spatially-anisotropic $(\mathbf{std},1o)$ block (outside Prop. 1'). Were it
    flat on that too, its flatness on the [A] table would be vacuous (flat-on-everything)."""
    _, _, bi_blocks, _ = scene_bibasis()
    g = torch.Generator().manual_seed(7)
    g11 = lambda: torch.Generator().manual_seed(11)  # noqa: E731 (fixed sketch directions)
    valid_iso = float(sigreg_block(synth_biblock_iso(20000, (1, 1, 1, 1), g), bi_blocks, n_proj=128, generator=g11()))
    valid_split = float(sigreg_block(synth_biblock_iso(20000, (0.5, 1, 2, 4), g), bi_blocks, n_proj=128, generator=g11()))
    aniso = float(sigreg_block(synth_spatial_aniso(20000, 4.0, g), bi_blocks, n_proj=128, generator=g11()))
    assert max(valid_iso, valid_split) < 1e-3, (
        f"bi-block-SIGReg not flat on valid laws: iso={valid_iso:.2e}, split={valid_split:.2e}"
    )
    assert aniso > 1e-3 and aniso > 20.0 * max(valid_iso, valid_split), (
        f"bi-block did not spike on the anisotropic (non-Prop.1') law: "
        f"aniso={aniso:.2e} vs valid floor {max(valid_iso, valid_split):.2e}"
    )


# --------------------------------------------------------------------------- #
# [5] THE COMPOSITIONAL RUNG: se3-block grows on the within-type S_O split; bi-block flat.
# --------------------------------------------------------------------------- #
def test_compositional_separation() -> None:
    r"""On a within-type $S_O$ split — trivial vs standard scaled differently at a fixed SO(3)-type
    budget — the Step-39 *se3*-block-SIGReg cannot represent the split and GROWS, while bi-block
    stays flat. This is the objective-level proof that bi-block strictly refines se3-block on the
    compositional degree of freedom (the $304\to184$ rung)."""
    basis = _make_basis()
    bi_blocks, se3_blocks = basis["bi_blocks"], basis["se3_blocks"]
    bi_to_se3 = basis["bi_to_se3"]  # regroup bibasis [A|B|C|D] -> se3 [A|C|B|D] = [scalars|vectors]
    g = lambda: torch.Generator().manual_seed(7)  # noqa: E731
    g11 = lambda: torch.Generator().manual_seed(11)  # noqa: E731

    # synth laws are emitted in bibasis order; the se3 baseline regroups them to [scalars|vectors]
    # with a plain index permutation, EXACTLY as the experiment's [A] table (NOT via to_isobasis,
    # which Helmert-rotates the object axis of *raw object-stacked* encoder output — applying it to
    # already-bibasis synthetic data would corrupt the law).
    def se3_val(scales):
        Z = synth_biblock_iso(20000, scales, g())
        return float(sigreg_block(Z[:, bi_to_se3], se3_blocks, n_proj=128, generator=g11()))

    def bi_val(scales):
        Z = synth_biblock_iso(20000, scales, g())
        return float(sigreg_block(Z, bi_blocks, n_proj=128, generator=g11()))

    # se3-type law: scalar != vector, but trivial == standard WITHIN each SO(3) type -> se3 flat.
    se3_on_se3type = se3_val((1, 4, 1, 4))
    # bi-type law: all four distinct -> within-type S_O split that se3-block cannot see.
    se3_on_bitype = se3_val((0.5, 1, 2, 4))
    assert se3_on_bitype > 1e-3 and se3_on_bitype > 5.0 * max(se3_on_se3type, 1e-9), (
        f"se3-block did not grow on the within-type S_O split: "
        f"se3(bi-type)={se3_on_bitype:.2e} vs se3(se3-type)={se3_on_se3type:.2e}"
    )
    # bi-block stays flat on BOTH (it targets the full 4-block class)
    assert max(bi_val((1, 4, 1, 4)), bi_val((0.5, 1, 2, 4))) < 1e-3, "bi-block not flat on both laws"


# --------------------------------------------------------------------------- #
# [6] deterministic gauge ladder: se3-type -> 304; bi-type (4 distinct scales) -> 184.
# --------------------------------------------------------------------------- #
def test_gauge_ladder_synth_and_robust() -> None:
    r"""se3-type law $\to 304$ (clusters $[24,8]$); a well-separated 4-distinct-scale bi-type law
    $\to 184$ (clusters $[18,6,6,2]$), robust for ANY ``gap_factor`` inside the separating
    window — not an artefact of one tuned threshold."""
    Z_se3 = synth_biblock_iso(20000, (1.0, 4.0, 1.0, 4.0), torch.Generator().manual_seed(21))
    # geometric scales (var ratios = 9 between adjacent blocks) -> window (1, 9): gap in {1.5..4} is interior
    Z_bi = synth_biblock_iso(20000, (1.0, 3.0, 9.0, 27.0), torch.Generator().manual_seed(21))
    g_se3 = spectral_gauge(Z_se3)
    g_bi = spectral_gauge(Z_bi)
    assert g_se3["gauge_dim"] == GAUGE_SE3 == 304, f"se3-type gauge {g_se3}"
    assert g_bi["gauge_dim"] == GAUGE_BI == 184, f"bi-type gauge {g_bi}"
    assert sorted(g_se3["cluster_sizes"], reverse=True) == [24, 8], f"se3 clusters {g_se3['cluster_sizes']}"
    assert sorted(g_bi["cluster_sizes"], reverse=True) == [18, 6, 6, 2], f"bi clusters {g_bi['cluster_sizes']}"
    for gf in (1.5, 2.0, 3.0, 4.0):
        assert spectral_gauge(Z_se3, gap_factor=gf)["gauge_dim"] == 304, f"se3@gf={gf}"
        assert spectral_gauge(Z_bi, gap_factor=gf)["gauge_dim"] == 184, f"bi@gf={gf}"


# --------------------------------------------------------------------------- #
# [7] Prop. 1' on a learned/fresh latent + negative control (remove S_O invariance).
# --------------------------------------------------------------------------- #
def test_prop1_holds_on_invariant_law_fails_on_fixed_slot() -> None:
    r"""Block-isotropy follows from equivariance **and** an $S_O\times SO(3)$-invariant law, so a
    FRESH equivariant encoder on a Haar+permute law is bi-block-isotropic (the six cross-block
    couplings $\to0$). The SAME encoder on a FIXED-SLOT law (still rotation-invariant, but NOT
    $S_O$-invariant) FAILS — cross-coupling between the trivial and standard $S_O$-reps no longer
    has to vanish. So [C] *can* fail, and fails exactly when the $S_O$ premise is removed. (The
    per-vector $3\times3$ isotropy follows from the surviving global SO(3) invariance, so it is
    NOT the discriminator here — ``cross`` is.)"""
    torch.manual_seed(0)
    enc = build_scene_eq()
    _, _, bi_blocks, _ = scene_bibasis()
    U, bi_perm = _make_basis()["U"], _make_basis()["bi_perm"]
    N = 8192
    with torch.no_grad():
        z_inv = to_isobasis(enc(make_scenes(N, seed=777, orient="haar", permute=True)), U, bi_perm)
        z_fix = to_isobasis(enc(make_scenes(N, seed=777, orient="haar", permute=False)), U, bi_perm)
        m_inv = prop1_bi_metrics(z_inv, bi_blocks)
        m_fix = prop1_bi_metrics(z_fix, bi_blocks)
    # S_O x SO(3)-invariant law -> bi-block-isotropic (small cross, each 1o block ~isotropic)
    assert m_inv["cross"] < 0.15, f"invariant law should decouple the blocks: cross={m_inv['cross']:.3f}"
    assert m_inv["iso_rel"] < 1.5 and m_inv["iso_agg"] < 1.5, f"1o blocks not isotropic: {m_inv}"
    # remove the S_O invariance -> the trivial/standard cross-coupling need not vanish: cross spikes
    assert m_fix["cross"] > 2.5 * m_inv["cross"], (
        f"fixed-slot (non-S_O-invariant) law should break bi-block decoupling: "
        f"cross {m_inv['cross']:.3f} -> {m_fix['cross']:.3f}"
    )


def main() -> None:
    torch.manual_seed(0)
    print("Step 40 — compositional bi-block-SIGReg mechanism guards\n")
    line = "    " + "-" * 60

    print("[1] bibasis geometry orthogonal + gauge-ladder arithmetic")
    test_bibasis_is_orthogonal_and_well_shaped()
    test_gauge_ladder_arithmetic()
    print(f"        Helmert orthogonal, widths [2,6,6,18]; ladder {GAUGE_ISO}->{GAUGE_SE3}->{GAUGE_BI}->{GAUGE_COMMUTANT}  OK")
    print(line)

    print("[2] scene equivariance: SO(3)+S_O+translation (init + post-train)")
    test_scene_equivariance_init_and_post()
    test_mlp_breaks_rotation_but_keeps_permutation()
    print("        eq exact & survives training; MLP perm-equiv but not rotation     OK")
    print(line)

    print("[3] SIGReg / task objectives finite + differentiable")
    test_sigreg_objectives_finite_and_differentiable()
    print("        isotropic / se3 / bi / budget / rel-task: finite scalar + grad    OK")
    print(line)

    print("[4] bi-block-SIGReg discriminates (anti-vacuity positive control)")
    test_bi_block_sigreg_discriminates()
    print("        flat on valid 4-scale laws, spikes on anisotropic (non-Prop.1')   OK")
    print(line)

    print("[5] COMPOSITIONAL rung: se3-block grows on within-type S_O split")
    test_compositional_separation()
    print("        se3-block grows >5x on the S_O split it can't see; bi-block flat   OK")
    print(line)

    print("[6] deterministic gauge ladder + gap_factor robustness")
    test_gauge_ladder_synth_and_robust()
    print(f"        se3-type->O(8)xO(24)={GAUGE_SE3}; bi-type->O(2)O(6)O(6)O(18)={GAUGE_BI}     OK")
    print(line)

    print("[7] Prop. 1' + negative control (remove S_O invariance)")
    test_prop1_holds_on_invariant_law_fails_on_fixed_slot()
    print("        block-isotropic on Haar+perm; FAILS on fixed-slot (premise gone)   OK")
    print(line)

    print("\nPASS: compositional bi-block-SIGReg mechanism is intact (orthogonal bibasis,")
    print("SO(3)+S_O equivariance, non-vacuous objective, the 304->184 compositional rung,")
    print("and Prop. 1' fails exactly when the S_O premise is removed).")


if __name__ == "__main__":
    main()
