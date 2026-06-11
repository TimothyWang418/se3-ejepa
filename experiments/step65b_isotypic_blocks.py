r"""Step 65b — the isotypic refinement, MEASURED (closes "asserted, not separately measured").

Two claims, both forced by Schur's lemma for a linear equivariant map, both verified to machine precision on
$G=\mathbb{Z}_4$ acting on $\mathbb R^4$ by the orthogonal representation
$\rho = \mathbf{1} \oplus \mathrm{sgn} \oplus R(90^\circ)$ (trivial $\oplus$ sign $\oplus$ 2-D rotation irrep):

  (P) **Placement is forced**: the group average $\bar M=\tfrac14\sum_k\rho^k M\rho^{-k}$ of a RANDOM matrix lands in
      the commutant — off-isotypic-block mass $<10^{-14}$ (relative) for every random draw. (Proposition 4's
      mechanism, measured rather than asserted.)
  (S) **Per-block horizon stratification with zero leakage**: with equivariant dynamics
      $\Phi=e^{0.7}\oplus e^{-0.4}\,\mathrm{sgn\text{-}block}\oplus e^{0.35}R(1.0)$ (a generic commutant element) and
      a single encoder defect $\epsilon u$ confined to block $B$, the $T$-step orbit-error-variation in block $B$
      equals $\epsilon\,e^{\lambda_B T}$ (rel. err $<10^{-6}$) while every OTHER block's variation stays at the
      numerical floor ($<10^{-12}\cdot$scale) — channels cannot mix across isotypic blocks, so the per-block
      certificate constants are exactly $1$ there (the step95 obliqueness can only live INSIDE a block).

Gates (pre-registered): (P) rel off-block mass < 1e-14 on 100 draws; (S1) defect-block growth matches
$\epsilon e^{\lambda_B T}$ to rel 1e-6 for each of the three blocks; (S2) cross-block leakage < 1e-12 relative.

Run: .venv/bin/python experiments/step65b_isotypic_blocks.py     (CPU, seconds)
Writes: papers/figures/step65b_isotypic_blocks.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# rho(g) for the generator of Z_4: trivial (1) ⊕ sign (-1) ⊕ rotation by 90°
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
RHO = np.block([[np.eye(1), np.zeros((1, 3))],
                [np.zeros((1, 1)), -np.eye(1), np.zeros((1, 2))],
                [np.zeros((2, 2)), R90]])
BLOCKS = {"trivial": [0], "sign": [1], "rot": [2, 3]}
LAMS = {"trivial": 0.70, "sign": -0.40, "rot": 0.35}


def rho_pow(k):
    out = np.eye(4)
    for _ in range(k % 4):
        out = RHO @ out
    return out


def commutant_dynamics():
    th = 1.0
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Phi = np.zeros((4, 4))
    Phi[0, 0] = np.exp(LAMS["trivial"])
    Phi[1, 1] = np.exp(LAMS["sign"])
    Phi[2:, 2:] = np.exp(LAMS["rot"]) * rot
    # sanity: commutes with rho
    assert np.abs(Phi @ RHO - RHO @ Phi).max() < 1e-12
    return Phi


def part_p(n_draws=100) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_draws):
        M = rng.standard_normal((4, 4))
        Mbar = sum(rho_pow(k) @ M @ rho_pow(4 - k) for k in range(4)) / 4.0
        off = Mbar.copy()
        for idx in BLOCKS.values():
            off[np.ix_(idx, idx)] = 0.0
        worst = max(worst, np.abs(off).max() / max(np.abs(Mbar).max(), 1e-300))
    ok = worst < 1e-14
    print(f"[step65b:P] worst relative off-isotypic-block mass over {n_draws} group-averaged random matrices: "
          f"{worst:.2e} | gate(<1e-14)={ok}", file=sys.stderr)
    return {"worst_rel_offblock": float(worst), "n_draws": n_draws, "G_P_placement_forced": bool(ok)}


def part_s(T_max=30, eps=1e-3) -> dict:
    Phi = commutant_dynamics()
    out = {"blocks": {}}
    ok_growth, ok_leak = True, True
    for bname, idx in BLOCKS.items():
        u = np.zeros(4)
        u[idx[0]] = 1.0                                              # unit defect confined to block B
        rows = []
        for T in range(0, T_max + 1, 3):
            PhiT = np.linalg.matrix_power(Phi, T)
            v = PhiT @ (eps * u)                                     # propagated defect = orbit-error-variation vector
            grown = np.linalg.norm(v[idx])
            pred = eps * np.exp(LAMS[bname] * T)
            leak = max(np.linalg.norm(v[j]) for jn, j in BLOCKS.items() if jn != bname)
            rows.append({"T": T, "block_var": float(grown), "predicted": float(pred),
                         "rel_err": float(abs(grown - pred) / pred), "leak": float(leak)})
            ok_growth &= rows[-1]["rel_err"] < 1e-6
            ok_leak &= leak < 1e-12 * max(grown, eps)
        out["blocks"][bname] = rows
        print(f"[step65b:S] block={bname:7s} lam={LAMS[bname]:+.2f}: max rel-err "
              f"{max(r['rel_err'] for r in rows):.1e}, max leak {max(r['leak'] for r in rows):.1e}", file=sys.stderr)
    out["G_S1_per_block_growth"] = bool(ok_growth)
    out["G_S2_zero_leakage"] = bool(ok_leak)
    return out


if __name__ == "__main__":
    res = {"P_placement": part_p(), "S_stratification": part_s()}
    ok = (res["P_placement"]["G_P_placement_forced"] and res["S_stratification"]["G_S1_per_block_growth"]
          and res["S_stratification"]["G_S2_zero_leakage"])
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "step65b_isotypic_blocks.json").write_text(json.dumps(res, indent=2))
    print(f"[step65b] {'PASS' if ok else 'INCONCLUSIVE'}", file=sys.stderr)
    raise SystemExit(0 if ok else 1)
