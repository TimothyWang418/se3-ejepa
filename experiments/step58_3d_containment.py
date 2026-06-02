r"""Step 58: the **3D-aware containment** — slow ⊆ (invariant ⊕ conserved-equivariant), measured.

Step 57 found the honest 3D subtlety: lifting the Noether hinge to a 3D two-body contact system, the
*Noether-content* half lifts (conserved $E$ lands in the invariant $\ell{=}0$ block) but the clean 2D
containment "slow ⊆ invariant" does **not**, because **in 3D the conserved angular momentum
$L=\sum_i r_i\times v_i$ is an $\ell{=}1$ *vector* (equivariant)** — so a slow conserved mode also lives in the
equivariant block. This step turns that observation into a *positive, measured* statement:

    in 3D,  slow  ⊆  (invariant ⊕ conserved-equivariant),

by showing the two conserved quantities split **cleanly by isotypic type** in the learned latent:
  - **energy $E$** (an $\mathrm{SO}(3)$-invariant scalar) is recovered from the **$\ell{=}0$** block, not $\ell{=}1$;
  - **angular momentum $L$** (a conserved $\ell{=}1$ vector) is recovered from the **$\ell{=}1$** block, not $\ell{=}0$;
  - both are conserved by the true dynamics (per-step drift ≪ scale), hence both are *slow*.

So the slow/long-horizon subspace is not contained in the invariant block (the 2D statement) but in the
**direct sum of the invariant block and the conserved part of the equivariant block** — Schur-clean, and exactly
what Noether predicts once $L$ is recognised as a vector in 3D.

Reuses the Step 57 system + model (3D SO(3) two-body contact, hand-rolled 3D Vector-Neuron ℓ=0⊕ℓ=1 latent).

Run (full ~2-3 min):  .venv/bin/python experiments/step58_3d_containment.py
Smoke:  STEP58_SMOKE=1 .venv/bin/python experiments/step58_3d_containment.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from step50_noether_hinge import r2_regress, slowest_mode_rate  # noqa: E402
from step57_embodied_hinge import (  # noqa: E402
    DS, DV, EquivWM, contact_dist, energy, make_pairs, step_state, train,
)

torch.set_default_dtype(torch.float64)
SMOKE = bool(os.environ.get("STEP58_SMOKE"))
SEED = int(os.environ.get("STEP58_SEED", "0"))
BETA = 0.0                                    # exact SO(3) (so E and L are both conserved)


def angular_momentum(s: torch.Tensor) -> torch.Tensor:
    r"""Total angular momentum $L=\sum_i r_i\times v_i$ — a conserved $\ell{=}1$ vector at $\beta{=}0$. (B,12)->(B,3)."""
    r1, v1, r2, v2 = s[:, 0:3], s[:, 3:6], s[:, 6:9], s[:, 9:12]
    return torch.cross(r1, v1, dim=-1) + torch.cross(r2, v2, dim=-1)


def cross_features(V: torch.Tensor) -> torch.Tensor:
    r"""All pairwise cross products $V_i\times V_j$ of the $d_v$ latent $\ell{=}1$ vectors — the **degree-2,
    $\mathrm{SO}(3)$-equivariant** readout. $L=r\times v$ is bilinear (a pseudovector), so it is NOT a *linear*
    function of the latent vectors; it lives in this cross-product (degree-2) part — the same primitive the old
    paper's tensor-product message supplies. (B,dv,3) -> (B, 3*C(dv,2))."""
    dv = V.shape[1]
    feats = [torch.cross(V[:, i], V[:, j], dim=-1) for i in range(dv) for j in range(i + 1, dv)]
    return torch.cat(feats, dim=-1)


def conservation_drift(fn, beta, n=400, seed=2025):
    r"""Relative one-step drift of a conserved quantity along the TRUE dynamics: std(Δq)/std(q). ~0 ⇒ conserved."""
    s, _ = make_pairs(n, beta, seed=seed)
    q0 = fn(s)
    q1 = fn(step_state(s, beta))
    d = (q1 - q0)
    if d.dim() == 1:
        return float(d.std() / q0.std().clamp_min(1e-9))
    return float(d.norm(dim=-1).std() / q0.norm(dim=-1).std().clamp_min(1e-9))


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 94
    n_train, epochs = (1000, 20) if SMOKE else (6000, 90)
    epochs = int(os.environ.get("STEP58_EPOCHS", epochs))
    print(line)
    print(f"STEP 58  3D-aware containment: slow ⊆ (invariant ⊕ conserved-equivariant)  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    S, S2 = make_pairs(n_train, BETA, seed=SEED)
    St, St1 = make_pairs(800, BETA, seed=999)

    # conservation check on the true dynamics (β=0): both E and L are conserved (≈0 drift)
    eE = conservation_drift(energy_beta(BETA), BETA)
    eL = conservation_drift(angular_momentum, BETA)
    print(f"    conservation (true-dynamics relative one-step drift, ≈0 ⇒ conserved): "
          f"E {eE:.3f}, L {eL:.3f}")

    torch.manual_seed(SEED)
    eq = EquivWM(); loss = train(eq, S, S2, epochs, seed=SEED)
    with torch.no_grad():
        s_te, V_te = eq.encode(St)
    scal = s_te                                              # ℓ=0 invariant block
    vecf = V_te.reshape(V_te.shape[0], -1)                   # ℓ=1 equivariant block
    E = energy(St, BETA).unsqueeze(1)
    L = angular_momentum(St)                                 # (B,3)
    Lmag = L.norm(dim=-1, keepdim=True)

    # isotypic placement: E (invariant) → ℓ=0;  L (equivariant vector) → ℓ=1
    with torch.no_grad():
        cross = cross_features(V_te)                         # degree-2 equivariant readout of the ℓ=1 block
    r2_E_inv, r2_E_eq = r2_regress(scal, E), r2_regress(vecf, E)
    r2_L_lin = r2_regress(vecf, L)                           # L from LINEAR ℓ=1 (should fail — L is bilinear)
    r2_L_inv = r2_regress(scal, L)                           # L from ℓ=0 (should fail — L is equivariant)
    r2_L_cross = r2_regress(cross, L)                        # L from the degree-2 cross-product readout
    r2_Lmag_inv = r2_regress(scal, Lmag)                     # |L| is invariant ⇒ partly in ℓ=0
    print(f"    train loss {loss:.2e}")
    print(f"    energy E (invariant scalar):       R²(ℓ=0)={r2_E_inv:.3f}   R²(ℓ=1 lin)={r2_E_eq:.3f}")
    print(f"    ang. mom. L (conserved ℓ=1 vector): R²(ℓ=0)={r2_L_inv:.3f}   R²(ℓ=1 lin)={r2_L_lin:.3f}   "
          f"R²(ℓ=1 cross/deg-2)={r2_L_cross:.3f}")
    print(f"    |L| (invariant scalar):            R²(ℓ=0)={r2_Lmag_inv:.3f}")

    with torch.no_grad():
        s_te1, V_te1 = eq.encode(St1)
    rate_inv = slowest_mode_rate(s_te, s_te1)
    rate_eq = slowest_mode_rate(vecf, V_te1.reshape(V_te1.shape[0], -1))

    print(f"\n{line}\nSTEP 58 SUMMARY\n{line}")
    print(f"    the conserved physics splits by isotypic TYPE *and* by polynomial DEGREE:")
    print(f"        E (invariant, deg-2): linearly in the ℓ=0 block (R² {r2_E_inv:.2f} ≫ {r2_E_eq:.2f} in ℓ=1)")
    print(f"        L (conserved ℓ=1 vector, BILINEAR): not linear in either block "
          f"(ℓ=0 {r2_L_inv:.2f}, ℓ=1-lin {r2_L_lin:.2f}); recovered by the degree-2 cross-product readout "
          f"of ℓ=1 (R² {r2_L_cross:.2f})")
    print(f"    both conserved (drift E {eE:.2f}, L {eL:.2f}) ⇒ both slow; per-block slow modes "
          f"{rate_inv:.3f}(ℓ=0)/{rate_eq:.3f}(ℓ=1)")
    ok = (eE < 0.2 and eL < 0.2
          and r2_E_inv > 0.8 and r2_E_inv > r2_E_eq + 0.2
          and r2_L_cross > 0.6 and r2_L_cross > max(r2_L_lin, r2_L_inv) + 0.2)
    if ok:
        print(f"\n    CONFIRMED: in 3D the conserved physics is contained in (invariant ⊕ conserved-equivariant),")
        print(f"        NOT in the invariant block alone. Energy (invariant) is linear in ℓ=0; angular momentum")
        print(f"        (a conserved ℓ=1 vector) is **bilinear** — it lives in the degree-2 cross-product part of")
        print(f"        the ℓ=1 block (R² {r2_L_cross:.2f}), not any linear latent direction (the same r×v")
        print(f"        cross-product the old paper's degree-1 VN could not form). So the 2D containment")
        print(f"        slow ⊆ invariant becomes, in 3D, slow ⊆ (invariant ⊕ conserved-equivariant) with the")
        print(f"        equivariant conserved part accessed at degree 2 — Step 57's subtlety made precise.")
    else:
        print(f"\n    INCONCLUSIVE: gate not met; reported as-is (no thresholds loosened).")

    out = dict(passed=bool(ok), train_loss=loss, drift_E=eE, drift_L=eL,
               r2_E_invariant=r2_E_inv, r2_E_equivariant=r2_E_eq,
               r2_L_invariant=r2_L_inv, r2_L_linear_eq=r2_L_lin, r2_L_cross_deg2=r2_L_cross,
               r2_Lmag_invariant=r2_Lmag_inv,
               slow_invariant=rate_inv, slow_equivariant=rate_eq, beta=BETA, smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step58_3d_containment_smoke" if SMOKE else "step58_3d_containment"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = ["E ←\nℓ=0 (lin)", "E ←\nℓ=1 (lin)", "L ←\nℓ=0 (lin)", "L ←\nℓ=1 (lin)", "L ←\nℓ=1 (deg-2 cross)"]
    vals = [r2_E_inv, r2_E_eq, r2_L_inv, r2_L_lin, r2_L_cross]
    cols = ["C2", "C7", "C7", "C7", "C3"]
    ax.bar(labels, vals, color=cols)
    ax.set_ylabel("R² recovering the conserved quantity"); ax.set_ylim(0, 1.05)
    ax.set_title("Step 58 — 3D conserved physics: E invariant (linear ℓ=0), L equivariant + BILINEAR (deg-2 ℓ=1)\n"
                 "slow ⊆ (invariant ⊕ conserved-equivariant); L = r×v needs the cross-product the deg-1 VN lacks")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


def energy_beta(beta):
    r"""Curry energy() to a single-arg fn for conservation_drift."""
    return lambda s: energy(s, beta)


if __name__ == "__main__":
    main()
