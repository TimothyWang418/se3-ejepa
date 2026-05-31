r"""Step 41 — equivariant latent dynamics (C2 / Prop. 2): correctness guards.

These tests pin the *mechanism* behind ``experiments/step41_equivariant_dynamics.py`` so the
experiment's headline — *a $G$-equivariant world resolves the gauge pure SSL leaves free* —
cannot silently rot. They mirror Step 39's guard suite (same Haar sampler, same gauge-ladder
robustness style, same negative-control discipline), lifted from the static covariance (Prop. 1)
to the **dynamics** (Prop. 2):

1. **Predictor equivariance (init AND post-training).** The mixed-type predictor commutes with
   $\rho(R)=\mathbf I_4\oplus(\mathbf I_6\otimes R)$ to the float floor for arbitrary
   $R\in\mathrm{SO}(3)$, and one-step-MSE training must not break it; the MLP control does not.
2. **[A1] Schur drift + commutant projection (deterministic).** The headline drift
   $A=r_0\mathbf I_4\oplus r_1\mathbf I_{18}$ commutes with $\rho(R)$ (lies in the commutant);
   a generic dense drift does not; the commutant projection makes any matrix commute, is
   idempotent, and is faithful on $A$ ($P_C(A)=A$).
3. **Stationarity + degenerate static spectrum.** The OU with $q_i=\sigma^2(1-r_i^2)$ is
   stationary at $\Sigma_\infty=\sigma^2\mathbf I_{22}$, so the *static* covariance spectrum is
   one $22$-fold cluster — gauge $\dim O(22)=231$ (the "stuck" regime Step 39 calls underdetermined).
4. **[A2] dynamical gauge ladder (deterministic).** The *drift* spectrum of the equivariant OU
   splits into the isotypic eigenspaces $[18,4]$ — gauge $\dim O(18)\times O(4)=159$ — while the
   static spectrum stays $231$. This is the headline, computed from sampled OU pairs.
5. **[A2] robust to clustering gap_factor.** $231/159$ is recovered for ANY ``gap_factor`` in a
   wide band — not an artefact of one tuned threshold.
6. **[A3] orbit-constant Bayes error (C2 flatness).** On the anisotropic $z_t$ law (where orbit
   transport is discriminating), the *commuting* drift's one-step Bayes error is orbit-constant
   (to the float floor) while a *non-commuting* (spatially-anisotropic) drift varies — the
   positive control for C2's flatness theorem.
7. **Haar regression guard.** ``rand_so3`` samples the Haar measure ($\mathbb E[R]=\mathbf 0$,
   $\det=+1$); orbit transport is only a uniform group average under Haar.
8. **[C] Prop. 2 structure (deterministic / init).** Under a $G$-invariant law the equivariant
   transition's cross-time second moment $C_1=\mathbb E[f(z)z^\top]$ is block-diagonal (cross
   $\to0$) — at init, a structural consequence of equivariance + invariant law. Negative control
   (removing equivariance): a fresh MLP's $C_1$ has large cross-block coupling. Premise witness
   (PSD, no sign quirk): the input law's own covariance is $3\times3$-isotropic on the invariant
   law and anisotropic on the aniso law.
9. **[C] learned recovery + non-invariant-law negative control.** A briefly-trained equivariant
   predictor recovers the true per-irrep $\hat r_i\approx(0.2,0.9)$ with a near-Schur $C_1$ on the
   invariant law; the SAME map on a non-$G$-invariant (anisotropic) law breaks $3\times3$-isotropy
   — so [C] *can* fail, and fails exactly when Prop. 2's premise is removed.

Run:
    .venv/bin/python tests/test_step41_equivariant_dynamics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from experiments.step41_equivariant_dynamics import (  # noqa: E402
    BLOCKS,
    MLPPredictor,
    MixedTypePredictor,
    R0,
    R1,
    _dim_on,
    anisotropic_drift,
    apply_rho,
    bayes_orbit_errors,
    block_vec_iso,
    commutant_project,
    commute_err,
    cross_block_coupling,
    dynamical_gauge,
    equivariant_drift,
    generic_drift,
    per_block_r,
    predictor_equiv_err,
    rand_so3,
    sample_ou_pairs,
    static_gauge,
    stationary_noise,
    train_predictor,
    transition_second_moment,
)


def _worst_equiv_err(model, z, n_rot: int = 5, seed: int = 0) -> float:
    r"""Worst predictor-equivariance residual $\max\lVert\rho f(z)-f(\rho z)\rVert$ over Haar $R$."""
    gen = torch.Generator().manual_seed(seed)
    return max(predictor_equiv_err(model, z, rand_so3(gen)) for _ in range(n_rot))


def _cov(z: torch.Tensor) -> torch.Tensor:
    zc = z - z.mean(0, keepdim=True)
    return (zc.T @ zc) / (z.shape[0] - 1)


def test_predictor_equivariance_init_and_post() -> None:
    r"""Mixed-type predictor commutes with $\rho$ at init and after one-step-MSE training; MLP not.

    Equivariance is a structural property ($f(\rho z)=\rho f(z)$ by construction), and the training
    loop must not damage it — the exact analogue of Step 39's init+post encoder-equivariance guard.
    """
    torch.manual_seed(0)
    assert [b.name for b in BLOCKS] == ["0e", "1o"], "mixed-type layout [scalars | vectors]"
    A, q = equivariant_drift(), stationary_noise()
    zt, ztp1 = sample_ou_pairs(2000, A, q, torch.Generator().manual_seed(1), law="invariant")
    z_ev, _ = sample_ou_pairs(1024, A, q, torch.Generator().manual_seed(2), law="invariant")

    eq, mlp = MixedTypePredictor(), MLPPredictor()
    assert _worst_equiv_err(eq, z_ev) < 1e-4, "eq predictor not equivariant at init"
    assert _worst_equiv_err(mlp, z_ev) > 1e-2, "MLP unexpectedly equivariant at init"

    # a few epochs of one-step MSE must not break equivariance of the eq predictor
    train_predictor(eq, zt, ztp1, epochs=15, seed=0)
    train_predictor(mlp, zt, ztp1, epochs=15, seed=0)
    assert _worst_equiv_err(eq, z_ev) < 1e-4, "eq predictor lost equivariance after training"
    assert _worst_equiv_err(mlp, z_ev) > 1e-2, "MLP should remain non-equivariant after training"


def test_schur_drift_and_commutant_projection() -> None:
    r"""[A1] (deterministic): the headline drift is Schur (commutes with $\rho$); a generic drift is
    not; the commutant projection commutes, is idempotent, and is faithful on $A$ ($P_C(A)=A$)."""
    A = equivariant_drift()
    gen = torch.Generator().manual_seed(0)
    G = generic_drift(gen)
    Rs = [rand_so3(torch.Generator().manual_seed(s)) for s in range(4)]

    # headline drift already lies in the commutant
    assert max(commute_err(A, R) for R in Rs) < 1e-5, "headline drift A does not commute with rho"
    # a generic dense drift does not
    assert max(commute_err(G, R) for R in Rs) > 1.0, "generic drift unexpectedly in commutant"
    # its commutant projection does commute (to the float floor)
    PG = commutant_project(G)
    assert max(commute_err(PG, R) for R in Rs) < 1e-5, "commutant-projected drift does not commute"
    # projection is idempotent and faithful on A
    assert (commutant_project(PG) - PG).abs().max().item() < 1e-5, "P_C not idempotent"
    assert (commutant_project(A) - A).abs().max().item() < 1e-5, "P_C(A) != A (not faithful on A)"


def test_ou_stationary_static_spectrum_degenerate() -> None:
    r"""The OU with $q_i=\sigma^2(1-r_i^2)$ is stationary at $\Sigma_\infty=\mathbf I_{22}$, so the
    static spectrum is one $22$-fold cluster — gauge $\dim O(22)=231$ (the degenerate / stuck regime).
    """
    A, q = equivariant_drift(), stationary_noise()
    zt, ztp1 = sample_ou_pairs(20000, A, q, torch.Generator().manual_seed(0), law="invariant")
    # stationarity: cov(z_t) ~ I AND cov(z_{t+1}) ~ I (the equal-stationary-scale choice)
    assert (_cov(zt) - torch.eye(22)).abs().max().item() < 0.1, "z_t marginal not ~ I"
    assert (_cov(ztp1) - torch.eye(22)).abs().max().item() < 0.1, "OU not stationary (cov(z_{t+1}) != I)"
    # degenerate static spectrum -> full O(22)
    sg = static_gauge(zt)
    assert sg["cluster_sizes"] == [22], f"static spectrum not degenerate: {sg['cluster_sizes']}"
    assert sg["gauge_dim"] == _dim_on(22) == 231, f"static gauge {sg['gauge_dim']} != 231"


def test_dynamical_gauge_ladder() -> None:
    r"""[A2] HEADLINE: while the static spectrum is degenerate ($231$), the *dynamical* (drift)
    spectrum of the equivariant OU separates into the isotypic eigenspaces $[18,4]$ — gauge
    $\dim O(18)\times O(4)=159$. The world model resolves what SSL leaves free."""
    A, q = equivariant_drift(), stationary_noise()
    zt, ztp1 = sample_ou_pairs(20000, A, q, torch.Generator().manual_seed(0), law="invariant")
    assert static_gauge(zt)["gauge_dim"] == 231, "static gauge should be the degenerate 231"
    dg = dynamical_gauge(zt, ztp1)
    assert sorted(dg["cluster_sizes"], reverse=True) == [18, 4], f"drift clusters {dg['cluster_sizes']}"
    assert dg["gauge_dim"] == _dim_on(18) + _dim_on(4) == 159, f"dynamical gauge {dg['gauge_dim']} != 159"


def test_gauge_ladder_robust_to_gap_factor() -> None:
    r"""The $231\to159$ dynamical ladder must be recovered for ANY clustering ``gap_factor`` in a
    wide band (the AR gap $r_1/r_0=4.5$ is comfortably separated) — not an artefact of one threshold.
    """
    A, q = equivariant_drift(), stationary_noise()
    zt, ztp1 = sample_ou_pairs(20000, A, q, torch.Generator().manual_seed(0), law="invariant")
    for gf in (1.5, 2.0, 3.0, 4.0):
        assert static_gauge(zt, gap_factor=gf)["gauge_dim"] == 231, f"static@gf={gf}"
        assert dynamical_gauge(zt, ztp1, gap_factor=gf)["gauge_dim"] == 159, f"dynamical@gf={gf}"


def test_bayes_error_orbit_constant() -> None:
    r"""[A3] C2 flatness: on the anisotropic $z_t$ law the one-step Bayes error
    $\mathbb E\lVert\rho(R)z'-A\,\rho(R)z\rVert^2$ is orbit-constant for the *commuting* drift and
    *varies* for a non-commuting (spatially-anisotropic) drift — the positive control. (On the
    isotropic law the error collapses to a rotation-invariant Frobenius norm, so a wrong drift would
    look flat too; the aniso law is what makes the test discriminating.)"""
    A, q = equivariant_drift(), stationary_noise()
    gen = torch.Generator().manual_seed(5)
    zt, ztp1 = sample_ou_pairs(256, A, q, gen, law="aniso")
    Rs = [torch.eye(3)] + [rand_so3(gen) for _ in range(8)]

    def spread(errs):
        mean = sum(errs) / len(errs)
        return (max(errs) - min(errs)) / max(mean, 1e-12)

    errs_c = bayes_orbit_errors(A, zt, ztp1, Rs)  # commuting -> orbit-constant (per-sample exact)
    errs_g = bayes_orbit_errors(anisotropic_drift(), zt, ztp1, Rs)  # non-commuting -> varies
    assert spread(errs_c) < 1e-4, f"commuting drift not orbit-constant: spread={spread(errs_c):.2e}"
    assert spread(errs_g) > 0.05, f"non-commuting drift did not vary along orbits: spread={spread(errs_g):.2e}"


def test_rand_so3_is_haar() -> None:
    r"""Regression guard: $\mathbb E_{\mathrm{Haar}}[R]=\mathbf 0$ and $\det R=+1$ (orbit transport
    is a uniform group average only under the Haar measure)."""
    gen = torch.Generator().manual_seed(0)
    n = 8000
    acc = torch.zeros(3, 3)
    dets = 0.0
    for _ in range(n):
        R = rand_so3(gen)
        acc += R
        dets += float(torch.det(R))
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), "rand_so3 not orthogonal"
    assert (acc / n).norm().item() < 0.05, f"E[R] not ~0 (||E[R]||={(acc / n).norm().item():.3f}); not Haar"
    assert abs(dets / n - 1.0) < 1e-4, "rand_so3 must lie in SO(3) (det=+1)"


def test_prop2_schur_structure_at_init() -> None:
    r"""[C] structure (deterministic / init): under a $G$-invariant law the equivariant transition's
    cross-time second moment $C_1=\mathbb E[f(z)z^\top]$ is block-diagonal (cross $\to0$) — a
    structural consequence of equivariance + invariant law, so it holds for a *fresh* predictor.

    Negative control (remove equivariance): a fresh MLP's $C_1$ has large cross-block coupling.
    Premise witness (PSD, no sign quirk): the input law's own covariance is $3\times3$-isotropic on
    the invariant law and clearly anisotropic on the aniso law.
    """
    torch.manual_seed(0)
    A, q = equivariant_drift(), stationary_noise()
    zt, _ = sample_ou_pairs(8192, A, q, torch.Generator().manual_seed(777), law="invariant")
    zt_an, _ = sample_ou_pairs(8192, A, q, torch.Generator().manual_seed(778), law="aniso")

    # eq predictor -> Schur (cross -> 0); MLP -> large cross-block coupling (equivariance removed)
    cr_eq = cross_block_coupling(transition_second_moment(MixedTypePredictor(), zt))
    cr_mlp = cross_block_coupling(transition_second_moment(MLPPredictor(), zt))
    assert cr_eq < 0.12, f"eq C_1 not block-diagonal at init: cross={cr_eq:.3f}"
    assert cr_mlp > 0.5 and cr_mlp > 4.0 * cr_eq, f"MLP C_1 should break Schur: cross={cr_mlp:.3f} vs eq {cr_eq:.3f}"

    # premise witness: the aniso law is genuinely non-G-invariant at the (PSD) covariance level
    assert block_vec_iso(_cov(zt)) < 1.35, "invariant law should be 3x3-isotropic"
    assert block_vec_iso(_cov(zt_an)) > 3.0, "aniso law should be 3x3-anisotropic (premise removed)"


def test_prop2_recovers_r_and_fails_on_non_invariant_law() -> None:
    r"""[C] learned recovery + non-invariant-law negative control. A briefly-trained equivariant
    predictor recovers the true per-irrep $\hat r_i\approx(r_0,r_1)$ with a near-Schur $C_1$ on the
    $G$-invariant law; the SAME map on a non-$G$-invariant (anisotropic) law breaks $3\times3$-
    isotropy — [C] *can* fail, exactly when Prop. 2's premise (invariant law) is removed."""
    torch.manual_seed(0)
    A, q = equivariant_drift(), stationary_noise()
    zt_tr, ztp1_tr = sample_ou_pairs(3000, A, q, torch.Generator().manual_seed(1), law="invariant")
    zt, _ = sample_ou_pairs(8192, A, q, torch.Generator().manual_seed(777), law="invariant")
    zt_an, _ = sample_ou_pairs(8192, A, q, torch.Generator().manual_seed(778), law="aniso")

    eq = MixedTypePredictor()
    train_predictor(eq, zt_tr, ztp1_tr, epochs=20, seed=0)

    # recovers the true per-irrep AR coefficients with a near-Schur C_1 on the invariant law
    r = per_block_r(eq, zt)
    C1 = transition_second_moment(eq, zt)
    assert abs(r["0e"] - R0) < 0.06, f"r_hat(0e)={r['0e']:.3f} far from {R0}"
    assert abs(r["1o"] - R1) < 0.12, f"r_hat(1o)={r['1o']:.3f} far from {R1}"
    assert cross_block_coupling(C1) < 0.12, "learned C_1 not block-diagonal on invariant law"
    assert block_vec_iso(C1) < 1.35, "learned C_1 not 3x3-isotropic on invariant law"

    # negative control: the SAME equivariant map on a non-G-invariant law breaks 3x3-isotropy
    iso_an = block_vec_iso(transition_second_moment(eq, zt_an))
    assert iso_an > 3.0, f"eq map should break isotropy on non-invariant law: iso={iso_an:.3f}"


def main() -> None:
    torch.manual_seed(0)
    print("Step 41 — equivariant latent dynamics (C2 / Prop. 2) mechanism guards\n")
    line = "    " + "-" * 60

    print("[1] predictor equivariance (init + post-training)")
    test_predictor_equivariance_init_and_post()
    print("        eq commutes with rho (init + post-train); MLP does not        OK")
    print(line)

    print("[2] [A1] Schur drift + commutant projection (deterministic)")
    test_schur_drift_and_commutant_projection()
    print("        A commutes; generic does not; P_C commutes, idempotent, faithful OK")
    print(line)

    print("[3] OU stationary -> degenerate static spectrum (231)")
    test_ou_stationary_static_spectrum_degenerate()
    print("        Sigma_inf ~ I_22, static spectrum [22] -> O(22)=231           OK")
    print(line)

    print("[4] [A2] dynamical gauge ladder (231 -> 159)")
    test_dynamical_gauge_ladder()
    print(f"        static 231 / dynamical [18,4] -> O(18)xO(4)={_dim_on(18) + _dim_on(4)}        OK")
    print(line)

    print("[4b] [A2] gauge ladder robust to clustering gap_factor")
    test_gauge_ladder_robust_to_gap_factor()
    print("        231/159 stable across gap_factor in {1.5,2,3,4}               OK")
    print(line)

    print("[5] [A3] orbit-constant Bayes error (C2 flatness)")
    test_bayes_error_orbit_constant()
    print("        commuting drift orbit-flat; non-commuting drift varies         OK")
    print(line)

    print("[6] Haar regression guard")
    test_rand_so3_is_haar()
    print("        E[R]~0 (Haar), det=+1                                          OK")
    print(line)

    print("[7] [C] Prop. 2 Schur structure at init + premise witness")
    test_prop2_schur_structure_at_init()
    print("        eq C_1 block-diagonal; MLP breaks it; aniso law non-isotropic  OK")
    print(line)

    print("[8] [C] learned r-recovery + non-invariant-law negative control")
    test_prop2_recovers_r_and_fails_on_non_invariant_law()
    print("        eq recovers r_i, near-Schur C_1; breaks on non-invariant law   OK")
    print(line)

    print("\nPASS: Step 41 mechanism intact (predictor equivariance, Schur drift + commutant,")
    print("stationary degenerate static spectrum, dynamical gauge ladder 231->159 robust to")
    print("gap_factor, C2 orbit-flatness, Haar law, and Prop. 2 fails exactly when its premise")
    print("(equivariant map + G-invariant law) is removed).")


if __name__ == "__main__":
    main()
