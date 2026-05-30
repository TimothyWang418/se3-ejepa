r"""Intrinsic vs extrinsic $\mathrm{SE}(3)$-equivariance under the optimiser (Step 26).

A recent result (Lau & Su, *A Symmetry-Compatible Principle for Optimizer Design*, arXiv:2605.18106)
warns that Adam / AdamW / RMSProp are **geometry-blind**: their element-wise $1/\sqrt{v_t}$
preconditioner does not commute with a group action on weight space, so they can *silently break* an
equivariance constraint one step at a time. This test pins the project-specific resolution, which is
a clean dichotomy decided by the **parametrisation**, not the optimiser:

  * **Intrinsic** (the project's ``VNLinear`` / e3nn layers): the effective weight is $W=M\otimes I_d$
    for *any* channel-mixing $M$, so $W$ lies in the commutant $\mathcal C=\{W:W\rho(g)=\rho(g)W\}$
    **by construction**. The equivariance residual is *identically zero* (to the float floor) for any
    parameters — hence immune to **any** optimiser under **any** label noise.
  * **Extrinsic** (a free dense ``nn.Linear`` merely *initialised* in $\mathcal C$): equivariance is a
    measure-zero subspace constraint. On a noiseless realisable target even Adam *heals*, but under
    realistic **label noise** the stochastic gradient carries an off-$\mathcal C$ component; SGD
    (symmetry-compatible) damps it and Adam *sustains a larger* drift — the Lau–Su effect, but only a
    **modest** ($\sim$2–4$\times$) gap. The decisive axis is the *row* (intrinsic vs extrinsic).

The guards mirror ``experiments/step26_optimizer_equivariance.py``:
  [math]  the commutant construction is exact: $\mathrm{kron}(M,I_3)$ commutes with
          $\mathrm{kron}(I_2,R)$, and $P_{\mathcal C}$ is an idempotent projector recovering $M$.
  [B]     intrinsic ``VNLinear`` off-$\mathcal C$ $=0$ for Adam AND SGD under noise; extrinsic
          ``nn.Linear`` drifts off $\mathcal C$ under Adam; SGD drifts strictly less (sym-compatible).
  [A]     the real Step-13 VN ``EqJEPA`` is optimiser-agnostic (Muon/AdamW = Adam = SGD = float floor)
          at init AND post-train; a non-equivariant MLP control breaks (the metric is not vacuous).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python tests/test_step26_optimizer_equivariance.py
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                  # for `src.*`
sys.path.insert(0, str(ROOT / "experiments"))  # for the Step 26 module + its Step 13 backbone

import torch  # noqa: E402

from step13_se3_latent_jepa import rand_so3  # noqa: E402
from step18_se3_closed_loop import EVAL_DTYPE  # noqa: E402  (float64 probe dtype)
from step26_optimizer_equivariance import (  # noqa: E402
    _BREAK_A,
    _BREAK_B,
    _FLOOR_A,
    _FLOOR_B,
    _I2,
    _I3,
    _commutant_proj,
    _equiv_resid,
    _kron,
    _offcommutant,
    _rho,
    panel_a,
    panel_b,
)

# Short configs (the panels are seed-fixed; these match the experiment's SMOKE behaviour closely).
_EPOCHS_A = 4       # real Muon/AdamW + EMA-target + VICReg training for Panel A (kept test-cheap)
_STEPS_B = 1500     # Panel B reaches its stationary off-commutant fluctuation well before this
_BATCH_B = 64
_NOISE_B = 0.05     # label noise sigma -- WITHOUT it a realisable target lets Adam heal (no sustained break)


# --------------------------------------------------------------------------- #
# [math] the commutant construction itself (no training; fast + deterministic)
# --------------------------------------------------------------------------- #
def test_commutant_construction_is_exact() -> None:
    r"""$W=M\otimes I_3$ commutes with $\rho(R)=I_2\otimes R$, and $P_{\mathcal C}$ recovers $M$ exactly.

    This pins the geometry the experiment relies on: the intrinsic image $\{M\otimes I_3\}$ *is* the
    commutant of $\rho(R)=R\oplus R$ (Schur), and the orthogonal projection $P_{\mathcal C}$ is an
    idempotent that leaves it fixed — so the intrinsic layer is exactly equivariant by construction.
    """
    gen = torch.Generator().manual_seed(0)
    for _ in range(16):
        M = torch.randn(2, 2, generator=gen, dtype=EVAL_DTYPE)
        W = _kron(M, _I3)                                   # an exact intertwiner
        # (i) it really lies in the commutant: residual at the float64 floor.
        assert _equiv_resid(W, gen) < _FLOOR_B, "kron(M, I3) is not in the commutant"
        assert _offcommutant(W) < _FLOOR_B, "kron(M, I3) has nonzero off-commutant distance"
        # (ii) the projector recovers M exactly and is idempotent (P W = W for W in C).
        Wc, Mhat = _commutant_proj(W)
        assert (Mhat - M).abs().max().item() < _FLOOR_B, "P_C did not recover M"
        assert (Wc - W).abs().max().item() < _FLOOR_B, "P_C is not identity on the commutant"
        # (iii) a generic dense W is genuinely OFF the commutant (the metric is not vacuous).
        Wdense = torch.randn(6, 6, generator=gen, dtype=EVAL_DTYPE)
        assert _offcommutant(Wdense) > 1e-2, "a generic dense 6x6 should be far off the commutant"
        assert _equiv_resid(Wdense, gen) > 1e-2, "a generic dense 6x6 should break equivariance"


# --------------------------------------------------------------------------- #
# Panel B: intrinsic immune to any optimiser; extrinsic drifts (Adam worse than SGD)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _panel_b() -> dict:
    r"""The $2\times2$ commutant probe {intrinsic, extrinsic} x {Adam, SGD} under label noise. Cached."""
    return panel_b(_STEPS_B, _BATCH_B, _NOISE_B)


def test_intrinsic_vnlinear_is_exactly_equivariant_under_any_optimizer() -> None:
    r"""Intrinsic ``VNLinear``: off-$\mathcal C$ $=0$ for BOTH Adam and SGD, even under label noise.

    The effective weight is $W=M\otimes I_3$ for any $M$, so equivariance is a property of the
    *parametrisation*, not of where the optimiser lands — Adam's geometry-blindness cannot touch it.
    """
    b = _panel_b()
    for opt in ("adam", "sgd"):
        r = b[f"intrinsic_{opt}"]
        assert r["final_resid"] < _FLOOR_B, (
            f"intrinsic VNLinear lost equivariance under {opt}: {r['final_resid']:.2e} (>= {_FLOOR_B})"
        )
        assert r["offcommutant"] < _FLOOR_B, (
            f"intrinsic VNLinear drifted off the commutant under {opt}: {r['offcommutant']:.2e}; "
            "W = M (x) I_3 must be in C by construction for any M"
        )


def test_extrinsic_linear_drifts_off_commutant_under_adam() -> None:
    r"""Extrinsic ``nn.Linear`` (init in $\mathcal C$): Adam drives it off $\mathcal C$ under noise — while fitting.

    This is the Lau–Su effect made concrete on the project's representation: the break is real
    (off-$\mathcal C$ $\gg$ floor) and not mere divergence (the held-out fit explains the variance).
    """
    b = _panel_b()
    r = b["extrinsic_adam"]
    assert r["init_resid"] < _FLOOR_B, (
        f"extrinsic layer was not initialised in C: {r['init_resid']:.2e}"
    )
    assert r["offcommutant"] > _BREAK_B, (
        f"extrinsic+Adam did NOT drift off the commutant: {r['offcommutant']:.2e} (<= {_BREAK_B}); "
        "the Symmetry-Compatible warning would be inapplicable here"
    )
    assert r["fit_loss"] < 0.3, (
        f"extrinsic+Adam did not actually train (fit_loss {r['fit_loss']:.2e} >= 0.3); the off-C "
        "drift would then be divergence, not the optimiser's geometry"
    )


def test_symmetry_compatible_sgd_drifts_less_than_adam() -> None:
    r"""Among extrinsic layers, symmetry-compatible SGD drifts strictly LESS off $\mathcal C$ than Adam.

    The honest, *modest* ($\sim$2–4$\times$) column gap that Lau–Su predict — SGD's update keeps the
    restoring force toward $\mathcal C$ aligned, Adam's element-wise rescaling distorts it. (Neither
    reaches the floor under noise; only the intrinsic parametrisation does — that is the row gap.)
    """
    b = _panel_b()
    adam_off = b["extrinsic_adam"]["offcommutant"]
    sgd_off = b["extrinsic_sgd"]["offcommutant"]
    assert sgd_off < adam_off, (
        f"symmetry-compatible SGD did not beat Adam off-commutant: sgd={sgd_off:.2e} >= adam={adam_off:.2e}"
    )
    # ...and the decisive ROW gap dwarfs this COL gap: extrinsic-Adam is orders above intrinsic-Adam.
    row_gap = adam_off / max(b["intrinsic_adam"]["offcommutant"], 1e-18)
    col_gap = adam_off / max(sgd_off, 1e-18)
    assert row_gap > 1e3 * col_gap, (
        f"row gap (x{row_gap:.1e}) should dwarf col gap (x{col_gap:.2f}); parametrisation must dominate"
    )


# --------------------------------------------------------------------------- #
# Panel A: the real VN EqJEPA is optimiser-agnostic at init AND post-train
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _panel_a() -> dict:
    r"""Train the real Step-13 VN ``EqJEPA`` under {Muon/AdamW, Adam, SGD}; probe SE(3) at init+post. Cached."""
    return panel_a(_EPOCHS_A)


def test_vn_eqjepa_is_optimizer_agnostic() -> None:
    r"""The project's real VN world model keeps its $\sim10^{-6}$ SE(3)-equivariance under EVERY optimiser.

    Including **pure Adam on every parameter** (the geometry-blind optimiser the paper warns against) —
    at init AND after a real EMA-target + VICReg training run. The headline equivariance is earned by
    the intrinsic parametrisation, *not* by the choice of optimiser; Muon is used for optimisation
    quality, not to protect equivariance.
    """
    a = _panel_a()
    for opt in ("muon_adamw", "adam", "sgd"):
        r = a[opt]
        assert r["init"] < _FLOOR_A and r["post"] < _FLOOR_A, (
            f"VN EqJEPA equivariance is NOT {opt}-agnostic: init={r['init']:.2e}, post={r['post']:.2e} "
            f"(>= {_FLOOR_A})"
        )


def test_mlp_control_breaks_se3_equivariance() -> None:
    r"""Non-equivariant MLP control under Adam breaks SE(3) — so the Panel A assertion is not vacuous."""
    a = _panel_a()
    post = a["mlp_adam"]["post"]
    assert post > _BREAK_A, (
        f"MLP control unexpectedly equivariant ({post:.2e} <= {_BREAK_A}); the optimiser-agnostic "
        "claim for the VN model would then be vacuous"
    )


def main() -> None:
    """Print the Step 26 panels (math + B + A) and run all assertions."""
    line = "-" * 78
    print("Step 26: intrinsic vs extrinsic SE(3)-equivariance under the optimiser\n")

    # math
    test_commutant_construction_is_exact()
    print("    [math] commutant construction exact: kron(M,I3) in C, P_C idempotent, dense W off C.")

    # Panel B
    b = _panel_b()
    print(f"\n    [B] commutant probe (label noise sigma={_NOISE_B}, {_STEPS_B} steps, batch {_BATCH_B}):")
    print(f"    {'parametrisation':>20} | {'opt':>5} | {'final resid':>11} | {'off-C':>9} | {'fit':>9}")
    print("    " + line)
    for label, key in (("intrinsic VNLinear", "intrinsic"), ("extrinsic nn.Linear", "extrinsic")):
        for opt in ("adam", "sgd"):
            r = b[f"{key}_{opt}"]
            print(f"    {label:>20} | {opt:>5} | {r['final_resid']:>11.2e} | "
                  f"{r['offcommutant']:>9.2e} | {r['fit_loss']:>9.2e}")
    row_gap = b["extrinsic_adam"]["offcommutant"] / max(b["intrinsic_adam"]["offcommutant"], 1e-18)
    col_gap = b["extrinsic_adam"]["offcommutant"] / max(b["extrinsic_sgd"]["offcommutant"], 1e-18)
    print(f"    => ROW gap x{row_gap:.1e} (absolute) dwarfs COL gap x{col_gap:.2f} (modest).")

    # Panel A
    a = _panel_a()
    print(f"\n    [A] real VN EqJEPA composed SE(3) residual (float64), {_EPOCHS_A} epochs:")
    print(f"    {'optimiser':>12} | {'init':>9} | {'post':>9}")
    print("    " + line)
    for opt in ("muon_adamw", "adam", "sgd"):
        r = a[opt]
        print(f"    {opt:>12} | {r['init']:>9.2e} | {r['post']:>9.2e}")
    print(f"    {'MLP/adam':>12} | {'n/a':>9} | {a['mlp_adam']['post']:>9.2e}  <- non-equivariant control")

    test_intrinsic_vnlinear_is_exactly_equivariant_under_any_optimizer()
    test_extrinsic_linear_drifts_off_commutant_under_adam()
    test_symmetry_compatible_sgd_drifts_less_than_adam()
    test_vn_eqjepa_is_optimizer_agnostic()
    test_mlp_control_breaks_se3_equivariance()
    print("\nPASS: intrinsic VNLinear is exactly equivariant under any optimiser (off-C == 0); extrinsic")
    print("      nn.Linear drifts under Adam, SGD drifts less (modest); the real VN EqJEPA is")
    print("      optimiser-agnostic and the MLP control breaks => parametrisation dominates, optimiser second.")


if __name__ == "__main__":
    main()
