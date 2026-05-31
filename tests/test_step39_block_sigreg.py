r"""Step 39 — block-SIGReg on a mixed-type equivariant latent: correctness guards.

These tests pin the *mechanism* behind ``experiments/step39_block_sigreg.py`` so the
experiment's headline cannot silently rot. Four independent checks:

1. **Mixed-type equivariance (init AND post-training).** The ``n_out_scalar>0`` encoder
   must emit an SO(3)-**invariant** scalar block ($s(Rx)=s(x)$) and an **equivariant**
   vector block ($v(Rx)=R\,v(x)$) — to the float floor, for arbitrary $R\in\mathrm{SO}(3)$,
   and the LeJEPA training loop must not break it.
2. **SIGReg objectives are finite and differentiable.** Both regularisers and the budget
   penalty return finite scalars with finite gradients (they are trained against).
3. **Deterministic gauge ladder (Prop. 1's target class).** On synthetic block-isotropic
   Gaussians, the EQUAL-scale law (what vanilla SIGReg forces) has spectral gauge
   $\dim O(22)=231$ (one eigenvalue cluster), while a DISTINCT-scale law (which block-SIGReg
   admits) splits into the $[18,4]$ eigenspaces, $\dim O(18)\times O(4)=159$.
4. **Haar regression guard.** ``rand_so3`` must sample the **Haar** measure — the data law
   is $G$-invariant (and Prop. 1 applies) only then. A clean witness: $\mathbb E[R]=\mathbf 0$
   for Haar, whereas the previous axis-uniform/angle-uniform sampler gave $\mathbb E[R]=\tfrac13\mathbf I$.
   We also check the learned latent is block-isotropic on Haar data (cross$\to0$, vec-iso$\to1$).

Run:
    .venv/bin/python tests/test_step39_block_sigreg.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math  # noqa: E402

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from experiments.step39_block_sigreg import (  # noqa: E402
    _dim_on,
    build_eq,
    equiv_errs,
    make_clouds,
    prop1_metrics,
    rand_so3,
    spectral_gauge,
    synth_block_iso,
    synth_spatial_aniso,
    train_lejepa,
)
from src.training.sigreg import (  # noqa: E402
    sigreg_block,
    sigreg_isotropic,
    variance_budget_penalty,
)


def _worst_mixed_type_errs(enc, X) -> tuple[float, float]:
    r"""Worst (scalar-invariance, vector-equivariance) residual over several Haar $R$."""
    inv, equ = 0.0, 0.0
    for _ in range(5):
        R = o3.rand_matrix()  # e3nn Haar rotation — equivariance must hold for ALL R
        i, e = equiv_errs(enc, X, R)
        inv, equ = max(inv, i), max(equ, e)
    return inv, equ


def test_mixed_type_equivariance_init_and_post() -> None:
    r"""Scalars invariant + vectors equivariant, at init and after LeJEPA training."""
    torch.manual_seed(0)
    enc = build_eq()
    blocks = enc.irrep_blocks()
    assert [b.name for b in blocks] == ["0e", "1o"], "mixed-type layout [scalars | vectors]"

    X = make_clouds(64, seed=11, orient="haar")
    inv0, equ0 = _worst_mixed_type_errs(enc, X)
    assert max(inv0, equ0) < 1e-4, f"init equivariance broke: inv={inv0:.2e} equ={equ0:.2e}"

    # a few epochs of the faithful LeJEPA loss must not damage equivariance
    train_lejepa(enc, X, variant="block", blocks=blocks, epochs=3, batch=32, seed=0)
    inv1, equ1 = _worst_mixed_type_errs(enc, X)
    assert max(inv1, equ1) < 1e-4, f"post-train equivariance broke: inv={inv1:.2e} equ={equ1:.2e}"


def test_sigreg_objectives_finite_and_differentiable() -> None:
    r"""Each regulariser returns a finite scalar with a finite gradient."""
    torch.manual_seed(0)
    blocks = build_eq().irrep_blocks()
    for name, fn in (
        ("isotropic", lambda z: sigreg_isotropic(z, n_proj=32)),
        ("block", lambda z: sigreg_block(z, blocks, n_proj=32)),
        ("budget", lambda z: variance_budget_penalty(z)),
    ):
        z = torch.randn(256, 22, requires_grad=True)
        val = fn(z)
        assert val.ndim == 0 and torch.isfinite(val) and val.item() >= 0.0, f"{name} not a finite >=0 scalar"
        val.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all(), f"{name} gradient non-finite"


def test_block_sigreg_discriminates() -> None:
    r"""Anti-vacuity (positive control): block-SIGReg is flat on *valid* block-isotropic laws
    at ANY per-irrep scale split, yet **spikes** on a spatially-anisotropic vector block — a
    law outside Prop. 1's class ($\operatorname{cov}\not\propto\mathbf I_3$). Were it flat on
    that too, its flatness on the [A] table would be meaningless (flat-on-everything)."""
    blocks = build_eq().irrep_blocks()
    g = torch.Generator().manual_seed(7)
    g11 = lambda: torch.Generator().manual_seed(11)  # noqa: E731 (fixed sketch directions)
    valid1 = float(sigreg_block(synth_block_iso(20000, 1.0, g), blocks, n_proj=128, generator=g11()))
    valid4 = float(sigreg_block(synth_block_iso(20000, 4.0, g), blocks, n_proj=128, generator=g11()))
    aniso = float(sigreg_block(synth_spatial_aniso(20000, 4.0, g), blocks, n_proj=128, generator=g11()))
    assert max(valid1, valid4) < 1e-3, f"block-SIGReg not flat on valid laws: {valid1:.2e}, {valid4:.2e}"
    assert aniso > 1e-3 and aniso > 20.0 * max(valid1, valid4), (
        f"block-SIGReg did not spike on the anisotropic (non-Prop.1) law: "
        f"aniso={aniso:.2e} vs valid floor {max(valid1, valid4):.2e}"
    )


def test_gauge_ladder_robust_to_gap_factor() -> None:
    r"""The $231\to159$ ladder is a $\sim16\times$ eigenvalue gap (ratio$^2$ at ratio $4$), so it
    must be recovered for ANY clustering ``gap_factor`` in $(\sim1.13, 16)$ — i.e. the gauge
    claim is not an artefact of one tuned threshold."""
    z_eq = synth_block_iso(20000, 1.0, torch.Generator().manual_seed(21))
    z_di = synth_block_iso(20000, 4.0, torch.Generator().manual_seed(21))
    for gf in (1.5, 2.0, 3.0, 4.0, 8.0):
        assert spectral_gauge(z_eq, gap_factor=gf)["gauge_dim"] == _dim_on(22) == 231, f"equal@gf={gf}"
        assert spectral_gauge(z_di, gap_factor=gf)["gauge_dim"] == _dim_on(18) + _dim_on(4) == 159, (
            f"distinct@gf={gf}"
        )


def test_synth_gauge_ladder() -> None:
    r"""Equal-scale law -> gauge $\dim O(22)=231$; distinct-scale law -> $O(18)\times O(4)=159$."""
    g_eq = spectral_gauge(synth_block_iso(20000, 1.0, torch.Generator().manual_seed(21)))
    g_di = spectral_gauge(synth_block_iso(20000, 4.0, torch.Generator().manual_seed(21)))
    assert g_eq["gauge_dim"] == _dim_on(22) == 231, f"equal-scale gauge {g_eq}"
    assert g_di["gauge_dim"] == _dim_on(18) + _dim_on(4) == 159, f"distinct-scale gauge {g_di}"
    # the distinct-scale spectrum must actually separate into the [18, 4] isotypic eigenspaces
    assert sorted(g_di["cluster_sizes"], reverse=True) == [18, 4], f"clusters {g_di['cluster_sizes']}"


def test_rand_so3_is_haar() -> None:
    r"""Regression guard: $\mathbb E_{\mathrm{Haar}}[R]=\mathbf 0$ (old non-Haar sampler gave $\tfrac13\mathbf I$)."""
    gen = torch.Generator().manual_seed(0)
    n = 8000
    acc = torch.zeros(3, 3)
    dets = 0.0
    for _ in range(n):
        R = rand_so3(gen)
        acc += R
        dets += float(torch.det(R))
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), "rand_so3 not orthogonal"
    mean_norm = (acc / n).norm().item()
    assert mean_norm < 0.05, f"E[R] not ~0 (got ||E[R]||={mean_norm:.3f}); sampler is not Haar"
    assert abs(dets / n - 1.0) < 1e-4, "rand_so3 must lie in SO(3) (det=+1)"


def test_learned_latent_block_isotropic_on_haar() -> None:
    r"""On a Haar (hence $G$-invariant) law the equivariant latent satisfies Prop. 1.

    Block-isotropy follows from equivariance + invariant data *regardless of training*, so we
    check it on a fresh encoder at a large covariance sample (cross $\sim1/\sqrt N\to0$,
    each vector channel's $3\times3$ covariance $\to\mathbf I_3$, ratio $\to1$).
    """
    torch.manual_seed(0)
    enc = build_eq()
    blocks = enc.irrep_blocks()
    N = 8192
    with torch.no_grad():
        m = prop1_metrics(enc(make_clouds(N, seed=777, orient="haar")), blocks)
    rt = math.sqrt(3.0 / N)
    iso_floor = (1 + rt) ** 2 / (1 - rt) ** 2  # finite-sample isotropy ceiling
    assert m["cross"] < 0.05, f"cross-irrep coupling not ~0: {m['cross']:.3f}"
    assert m["vec_iso"] < 1.5 * iso_floor, f"vector channels not isotropic: {m['vec_iso']:.3f} (floor {iso_floor:.3f})"


def test_prop1_fails_on_non_invariant_law() -> None:
    r"""Negative control for Prop. 1: the **same** equivariant encoder, fed a non-$G$-invariant
    (wedge — $z$-rotations in $[0,90°)$) law, must FAIL block-isotropy. Block-isotropy follows
    from equivariance **and** an invariant data law; remove the latter and it must break — so
    [C] *can* fail, and fails exactly when the premise is removed (not because the metric is
    lax). Holds at init: it is a structural consequence, not a training artefact."""
    torch.manual_seed(0)
    enc = build_eq()
    blocks = enc.irrep_blocks()
    N = 8192
    with torch.no_grad():
        m_haar = prop1_metrics(enc(make_clouds(N, seed=777, orient="haar")), blocks)
        m_wedge = prop1_metrics(enc(make_clouds(N, seed=777, orient="wedge")), blocks)
    # G-invariant (Haar) law -> block-isotropic; remove the invariance -> clearly not.
    assert m_haar["cross"] < 0.05 and m_haar["vec_iso"] < 1.35, f"Haar should be block-isotropic: {m_haar}"
    assert m_wedge["cross"] > 0.3 and m_wedge["vec_iso"] > 3.0, (
        f"wedge (non-G-invariant) law should break block-isotropy: {m_wedge}"
    )


def main() -> None:
    torch.manual_seed(0)
    print("Step 39 — block-SIGReg mechanism guards\n")
    line = "    " + "-" * 56

    print("[1] mixed-type equivariance (init + post-training)")
    test_mixed_type_equivariance_init_and_post()
    print("        scalars invariant + vectors equivariant, survives training  OK")
    print(line)

    print("[2] SIGReg objectives finite + differentiable")
    test_sigreg_objectives_finite_and_differentiable()
    print("        isotropic / block / budget: finite scalar, finite grad      OK")
    print(line)

    print("[2b] block-SIGReg discriminates (anti-vacuity positive control)")
    test_block_sigreg_discriminates()
    print("        flat on valid laws, spikes on anisotropic (non-Prop.1) law  OK")
    print(line)

    print("[3] deterministic gauge ladder")
    test_synth_gauge_ladder()
    print(f"        equal-scale -> O(22)={_dim_on(22)}; distinct -> O(18)xO(4)={_dim_on(18) + _dim_on(4)}   OK")
    print(line)

    print("[3b] gauge ladder robust to clustering gap_factor")
    test_gauge_ladder_robust_to_gap_factor()
    print("        231/159 stable across gap_factor in {1.5,2,3,4,8}            OK")
    print(line)

    print("[4] Haar regression guard + learned block-isotropy")
    test_rand_so3_is_haar()
    test_learned_latent_block_isotropic_on_haar()
    print("        E[R]~0 (Haar), latent block-isotropic on Haar data          OK")
    print(line)

    print("[5] Prop. 1 negative control (non-G-invariant wedge law)")
    test_prop1_fails_on_non_invariant_law()
    print("        block-isotropy FAILS on wedge (premise removed -> [C] fails) OK")
    print(line)

    print("\nPASS: block-SIGReg mechanism is intact (equivariance, objectives non-vacuous,")
    print("gauge ladder robust, Haar law, and Prop. 1 fails exactly when its premise is removed).")


if __name__ == "__main__":
    main()
