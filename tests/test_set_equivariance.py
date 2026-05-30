r"""Scene-group equivariance of the object-centric latent world model (Step 19).

The single-object tests (``test_se3_equivariance.py``, ``test_post_training_equivariance.py``)
verify $\mathrm{SE}(3)$ for *one* rigid body. A *scene* of $O$ objects carries a larger
symmetry, $\mathrm{SE}(3)^{\,O}\rtimes S_O$ — per-object rigid motions **and** object
relabelings — built from *two logically independent* architectural priors:

  1. **Factorization** (shared-weight per-object slots): buys exact **permutation
     equivariance** $E(\sigma\!\cdot\!S)=\sigma\!\cdot\!E(S)$, **leakage-freedom** (object
     $i$'s latent block is independent of object $j$'s state), and — with a centered
     per-object encoder — **translation invariance**.
  2. **Per-object $\mathrm{SE}(3)$-equivariance**: buys the global-$\mathrm{SO}(3)$ relation
     $f_\phi(\rho(R)E(S),RA)=\rho(R)f_\phi(E(S),A)$ on the *whole* scene world model.

This test pins the **structural** half of Step 19's 2×2 (the part that must hold to the
float floor, by construction, independent of the weights) at init **and after a real
Muon/AdamW + EMA-target + VICReg training run** — Step 19's *learned* half (the
orientation-OOD relMSE blow-up of the non-equivariant slot model) lives in the experiment,
not here. We check all three Step-19 models so the test is a faithful unit-level mirror of
the ablation and is **not vacuous**:

  | model       | factorization (perm/leak/trans) | per-object SE(3) (global $\rho(R)$) |
  |-------------|---------------------------------|-------------------------------------|
  | VN-Set      | exact (floor)                   | exact (floor)                       |
  | MLP-Slot    | exact (floor)                   | **fails** (no rotation prior)       |
  | MLP-Global  | **fails** (entangled, leaks)    | **fails**                           |

So VN-Set passing *both* columns while MLP-Slot fails only the SE(3) column and MLP-Global
fails *both* is exactly the structural content of "you need both priors", and the two
negative controls guarantee each assertion bites.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python tests/test_set_equivariance.py
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # for `src.*`
sys.path.insert(0, str(ROOT / "experiments")) # for the Step 19 scene model + its Step 13 backbone

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from src.training.jepa import train_jepa  # noqa: E402
from step19_object_centric import (  # noqa: E402
    build_mlp_global,
    build_mlp_slot,
    build_vn_set,
    composed_perm_err,
    composed_se3_err,
    encoder_perm_err,
    leakage_err,
    make_scene_transitions,
    trans_inv_err,
)

# Tolerances. The factorization/equivariance residuals are exact-by-construction (integer
# reshape/index ops + orthogonal $\rho(R)$), so they sit at the float floor; the broken
# controls miss it by orders of magnitude, so every threshold has a wide margin.
_FLOOR = 1e-4    # structural equivariance residual (SE(3) composed, perm, translation)
_EXACT = 1e-5    # leakage: a literal sub-network identity for a slotted encoder => 0
_BROKEN = 1e-2   # a prior that is ABSENT must produce a residual this large (negative control)
_LEAKS = 5e-2    # a non-factorized encoder must leak at least this much


@torch.no_grad()
def _probe(model, S: torch.Tensor, A: torch.Tensor) -> dict[str, float]:
    r"""All four structural residuals for one model on a held-out scene batch.

    ``se3`` is worst-over-5-rotations of the end-to-end global-$\mathrm{SO}(3)$ residual
    $\max\lVert\rho(R)f(E(S),A)-f(E(RS),RA)\rVert_\infty$; ``perm_enc``/``perm_comp`` the
    encoder / whole-model permutation residuals; ``trans`` the global-shift invariance of the
    encoder; ``leak`` the relative change of object-1's latent block when object 2 is moved.
    """
    return {
        "se3": max(composed_se3_err(model, S, A, o3.rand_matrix()) for _ in range(5)),
        "perm_enc": encoder_perm_err(model.encoder, S),
        "perm_comp": composed_perm_err(model, S, A),
        "trans": trans_inv_err(model.encoder, S, torch.Generator().manual_seed(1)),
        "leak": leakage_err(model, S, torch.Generator().manual_seed(2)),
    }


@lru_cache(maxsize=1)
def _trained() -> tuple[dict, dict]:
    r"""Build all three Step-19 models, probe at init, train each, probe post-train.

    Returns ``(init, post)``, each a ``{model_name: residual_dict}``. Cached so the three
    short training runs happen once across all pytest entry points. A real
    Muon/AdamW + EMA-target + VICReg run (the same ``train_jepa`` the experiment uses) so this
    is a genuine "symmetry survives optimisation" regression guard, not an init-only check.
    """
    S, A, S2 = make_scene_transitions(300, seed=0)
    St, At, _ = make_scene_transitions(96, seed=7)

    builders = {"VN-Set": build_vn_set, "MLP-Slot": build_mlp_slot, "MLP-Global": build_mlp_global}
    init: dict[str, dict] = {}
    post: dict[str, dict] = {}
    for name, build in builders.items():
        torch.manual_seed(0)
        model = build()
        init[name] = _probe(model, St, At)
        train_jepa(model, S, A, S2, epochs=12, batch_size=128, var_coef=0.1, seed=0, log_every=10**9)
        post[name] = _probe(model, St, At)
    return init, post


def test_vn_set_carries_both_priors() -> None:
    r"""VN-Set: exact permutation/leakage/translation **and** exact global $\mathrm{SO}(3)$ — init + post."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["VN-Set"]
        assert r["se3"] < _FLOOR, f"VN-Set global SE(3) broke @{phase}: {r['se3']:.2e}"
        assert r["perm_enc"] < _FLOOR, f"VN-Set encoder permutation broke @{phase}: {r['perm_enc']:.2e}"
        assert r["perm_comp"] < _FLOOR, f"VN-Set whole-model permutation broke @{phase}: {r['perm_comp']:.2e}"
        assert r["trans"] < _FLOOR, f"VN-Set translation invariance broke @{phase}: {r['trans']:.2e}"
        assert r["leak"] < _EXACT, f"VN-Set leaked @{phase}: {r['leak']:.2e}"


def test_mlp_slot_is_factorized_but_not_equivariant() -> None:
    r"""MLP-Slot isolates factorization: exact perm/leak/trans, but **no** $\mathrm{SE}(3)$ prior."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["MLP-Slot"]
        # factorization => exact (identical to VN-Set on this column)
        assert r["perm_enc"] < _FLOOR, f"MLP-Slot encoder permutation broke @{phase}: {r['perm_enc']:.2e}"
        assert r["perm_comp"] < _FLOOR, f"MLP-Slot whole-model permutation broke @{phase}: {r['perm_comp']:.2e}"
        assert r["trans"] < _FLOOR, f"MLP-Slot translation invariance broke @{phase}: {r['trans']:.2e}"
        assert r["leak"] < _EXACT, f"MLP-Slot leaked @{phase}: {r['leak']:.2e}"
        # but the SE(3) prior is ABSENT — this control must FAIL the rotation relation
        assert r["se3"] > _BROKEN, (
            f"MLP-Slot unexpectedly SE(3)-equivariant @{phase} ({r['se3']:.2e}); "
            "the equivariance assertion would be vacuous"
        )


def test_mlp_global_carries_neither_prior() -> None:
    r"""MLP-Global isolates the absence of factorization: not permutation-equivariant, leaks, not $\mathrm{SE}(3)$."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["MLP-Global"]
        assert r["perm_enc"] > _BROKEN, (
            f"MLP-Global unexpectedly permutation-equivariant @{phase} ({r['perm_enc']:.2e}); "
            "the factorization assertion would be vacuous"
        )
        assert r["leak"] > _LEAKS, f"MLP-Global unexpectedly leakage-free @{phase} ({r['leak']:.2e})"
        assert r["se3"] > _BROKEN, f"MLP-Global unexpectedly SE(3)-equivariant @{phase} ({r['se3']:.2e})"


def main() -> None:
    """Print the structural 2×2 (init + post-train) and run all three assertions."""
    init, post = _trained()
    print("Step 19 scene-group equivariance: SE(3)^O |x| S_O, structural residuals\n")
    hdr = f"    {'model':>10} | {'phase':>10} | {'SE(3) comp':>11} | {'perm enc':>9} | {'trans':>9} | {'leak':>9}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in ("VN-Set", "MLP-Slot", "MLP-Global"):
        for phase, tbl in (("init", init), ("post-train", post)):
            r = tbl[name]
            print(f"    {name:>10} | {phase:>10} | {r['se3']:>11.2e} | {r['perm_enc']:>9.2e} | "
                  f"{r['trans']:>9.2e} | {r['leak']:>9.2e}")

    test_vn_set_carries_both_priors()
    test_mlp_slot_is_factorized_but_not_equivariant()
    test_mlp_global_carries_neither_prior()
    print("\nPASS: VN-Set is exactly SE(3)^O |x| S_O-equivariant through training; MLP-Slot is")
    print("      factorized-only (perm/leak/trans exact, SE(3) broken); MLP-Global is neither.")
    print("=> the two compositional priors are independent and each control bites — Step 19 payoff.")


if __name__ == "__main__":
    main()
