r"""Noether-hinge unit test (Step 50, paper2 §3 / P2).

paper2's horizon axis rests on the conjecture (§3) that an equivariant world model's *slow*
(long-horizon-certifiable) latent channels are its *group-invariant* ones — the Noether bridge. Two parts,
tested at two costs:

1. **Architectural (no training).** The encoder's scalar ($\ell{=}0$) block is $\mathrm{SO}(2)$-invariant *by
   construction*: rotating the input leaves the scalars fixed and rotates the vector ($\ell{=}1$) block by
   $R(\theta)$, to the float floor — at init, independent of weights. Hence the invariant subspace's OOD
   group-action residual is $\approx 0$ *for all of* $\mathrm{SO}(2)$ — the certificate's location is given by
   structure, for free. A randomly-initialised non-equivariant MLP has no such subspace (its directions drift
   under rotation), so the test discriminates the property it claims.
2. **Learned (brief training).** A trained equivariant model spontaneously places the conserved quantities
   $(E,L)$ in the invariant block — regression $R^2$ from the scalar block ≫ from the vector block — and the
   slowest mode the invariant block admits is far slower than anything the equivariant block admits
   (slow ⊆ invariant). [Smoke-scale here; the full multi-seed result is `experiments/step50_noether_hinge.py`.]

Run:  .venv/bin/python tests/test_step50_noether_hinge.py
"""

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step50_noether_hinge import (  # noqa: E402
    DS, DV, EquivWorldModel, MLPWorldModel, make_orbits, ood_group_residual, r2_regress, rot2,
    slowest_mode_rate, train, true_invariants,
)


def test_certificate_is_architectural_for_equivariant_model() -> None:
    r"""At init (no training): encoder scalars are SO(2)-invariant and vectors rotate, to the float floor;
    the invariant subspace's OOD group residual is ~0 for the whole group. A fresh MLP's directions are not."""
    torch.set_default_dtype(torch.float64)               # step50's math is float64 (robust to test import order)
    torch.manual_seed(0)
    eq = EquivWorldModel().eval()
    St = make_orbits(64, 6, seed=7)[0]                   # test states (first of the (s_t, s_{t+1}) pair)
    with torch.no_grad():
        R = rot2(0.6)
        rs = torch.cat([St[:, :2] @ R.T, St[:, 2:] @ R.T], 1)
        s0, V0 = eq.encode(St)
        s1, V1 = eq.encode(rs)
        scal_resid = (s1 - s0).abs().max().item()
        vec_resid = (V1 - torch.einsum("ij,bnj->bni", R, V0)).abs().max().item()
    print(f"encoder @init: scalar Δ {scal_resid:.2e} (should be ~0), vector rot-Δ {vec_resid:.2e} (should be ~0)")
    assert scal_resid < 1e-6, f"encoder scalars not SO(2)-invariant: {scal_resid:.2e}"
    assert vec_resid < 1e-6, f"encoder vectors do not transform as ℓ=1: {vec_resid:.2e}"

    eye = torch.eye(DS + 2 * DV)
    eq_resid = float(ood_group_residual(eq, St, eye[:, :DS], equiv=True).mean())
    print(f"equivariant invariant-subspace OOD group residual @init: {eq_resid:.2e}")
    assert eq_resid < 1e-6, f"invariant subspace should be exactly group-invariant: {eq_resid:.2e}"

    torch.manual_seed(0)
    mlp = MLPWorldModel().eval()
    mlp_resid = float(ood_group_residual(mlp, St, torch.eye(DS + 2 * DV)[:, :2], equiv=False).mean())
    print(f"MLP arbitrary-subspace OOD group residual @init: {mlp_resid:.2e}")
    assert mlp_resid > 1e-2, (
        f"a non-equivariant MLP should NOT have a group-invariant subspace, but residual was {mlp_resid:.2e} "
        "— the certificate test would not discriminate"
    )


def test_noether_content_and_slow_subset_invariant() -> None:
    r"""After brief training: the invariant (scalar) block recovers $(E,L)$ far better than the vector block,
    and the slowest mode it admits is far slower than the equivariant block's (slow ⊆ invariant)."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    S_t, S_tp1 = make_orbits(200, 12, seed=0)
    St, St1 = make_orbits(120, 12, seed=999)
    EL = true_invariants(St)

    eq = EquivWorldModel()
    train(eq, S_t, S_tp1, epochs=15, equiv=True)
    with torch.no_grad():
        s_te, V_te = eq.encode(St)
        s_te1, V_te1 = eq.encode(St1)
    vecf, vecf1 = V_te.reshape(V_te.shape[0], -1), V_te1.reshape(V_te1.shape[0], -1)

    r2_scal, r2_vec = r2_regress(s_te, EL), r2_regress(vecf, EL)
    print(f"(E,L) recovery: invariant R²={r2_scal:.3f}  vs  equivariant R²={r2_vec:.3f}")
    assert r2_scal > 0.8, f"invariant block should carry the conserved (E,L): R²={r2_scal:.3f}"
    assert r2_scal > r2_vec + 0.3, f"invariants should beat vectors at (E,L): {r2_scal:.3f} vs {r2_vec:.3f}"

    rate_scal = slowest_mode_rate(s_te, s_te1)
    rate_vec = slowest_mode_rate(vecf, vecf1)
    print(f"slowest mode: invariant block {rate_scal:.4f}  vs  equivariant block {rate_vec:.4f}")
    assert rate_scal < 0.5 * rate_vec, (
        f"slow ⊆ invariant broken: invariant-block slowest mode {rate_scal:.4f} not ≪ "
        f"equivariant-block {rate_vec:.4f}"
    )


def main() -> None:
    print("Step 50 — Noether hinge (group ⇒ slow), architectural + learned\n")
    test_certificate_is_architectural_for_equivariant_model()
    print()
    test_noether_content_and_slow_subset_invariant()
    print("\nPASS: the equivariant model's invariant subspace is exactly group-invariant (architectural) and,")
    print("      once trained, holds the conserved/slowest quantities — slow ⊆ invariant. paper2 §3 hinge.")


if __name__ == "__main__":
    main()
