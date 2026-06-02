r"""Step 54 (A4 redesign): the equivariant latent's covariance is a **$G$-invariant operator** — clean version.

Step 48 tried to show that an $\mathrm{SO}(3)$-equivariant latent's covariance is "pinned to the group
representation $\rho$" ($\rho\Sigma\rho^\top=\Sigma$), but on the interacting-teacher data — which is **not**
rotation-symmetric — it did not separate from a non-equivariant control (residual $0.52$ vs $0.88$, $\times2$),
and was honestly shelved. The fix is to run the diagnostic on a **genuinely rotation-symmetric** distribution.
Step 50's $\mathrm{SO}(2)$ central-force data samples initial conditions $\mathrm{SO}(2)$-isotropically, so the
latent distribution of an equivariant encoder is exactly $\rho$-invariant and its covariance must commute with
$\rho$ — by Schur's lemma a *prescriptive*, checkable signature.

Two diagnostics (equivariant Step-50 model vs an identically-trained MLP):
  1. **$\rho$-invariance residual** $\mathbb{E}_R\lVert\rho(R)\,\Sigma\,\rho(R)^\top-\Sigma\rVert_F/\lVert\Sigma\rVert_F$,
     where $\rho(R)$ fixes the scalar ($\ell{=}0$) block and rotates each $\ell{=}1$ vector. Small ⇒ the
     covariance is pinned to the group rep. (Applied to the MLP latent it is the *hypothesised* rep the MLP has
     no reason to respect.)
  2. **$\ell{=}1$ block isotropy** (equivariant-only prescriptive signature): each vector channel's $2\times2$
     self-covariance should be $\propto I$ (no preferred direction). Anisotropy $=(\lambda_{\max}-\lambda_{\min})/
     (\lambda_{\max}+\lambda_{\min})$, averaged over channels; $\approx0$ for the equivariant latent.

Run (full ~1 min):  .venv/bin/python experiments/step54_latent_spectrum_v2.py
Smoke:  STEP54_SMOKE=1 .venv/bin/python experiments/step54_latent_spectrum_v2.py
"""

import json
import math
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

from step50_noether_hinge import DS, DV, EquivWorldModel, MLPWorldModel, make_orbits, rot2, train  # noqa: E402

torch.set_default_dtype(torch.float64)
SMOKE = bool(os.environ.get("STEP54_SMOKE"))
SEED = int(os.environ.get("STEP54_SEED", "0"))


def rho_matrix(R: torch.Tensor) -> torch.Tensor:
    r"""Hypothesised latent rep $\rho(R)$: identity on the $d_s$ scalar block, $R$ on each of the $d_v$ vectors.
    For the equivariant model this is the *actual* rep; for the MLP it is the rep its latent has no reason to obey."""
    dim = DS + 2 * DV
    M = torch.eye(dim)
    for b in range(DV):
        i = DS + 2 * b
        M[i:i + 2, i:i + 2] = R
    return M


def covariance(Z: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(0, keepdim=True)
    return (Zc.T @ Zc) / max(Zc.shape[0] - 1, 1)


@torch.no_grad()
def rho_invariance_residual(cov: torch.Tensor, n_rot=24, seed=300) -> float:
    g = torch.Generator().manual_seed(seed)
    denom = cov.norm().clamp_min(1e-12)
    res = []
    for _ in range(n_rot):
        M = rho_matrix(rot2(float(2 * math.pi * torch.rand(1, generator=g))))
        res.append(((M @ cov @ M.T) - cov).norm() / denom)
    return float(torch.stack(res).mean())


@torch.no_grad()
def ell1_block_anisotropy(cov: torch.Tensor) -> float:
    r"""Mean anisotropy of the $d_v$ vector-channel $2\times2$ self-covariances (0 = isotropic = $\propto I$)."""
    aniso = []
    for b in range(DV):
        i = DS + 2 * b
        block = cov[i:i + 2, i:i + 2]
        ev = torch.linalg.eigvalsh(block).clamp_min(0)
        aniso.append(float((ev.max() - ev.min()) / (ev.sum().clamp_min(1e-12))))
    return float(np.mean(aniso))


@torch.no_grad()
def latent_over_symmetric_set(model, S, n_rot, equiv, gen):
    r"""Latents over the test set + several random rotations of it (a rotation-symmetric sample)."""
    def enc_flat(x):
        if equiv:
            s, V = model.encode(x)
            return torch.cat([s, V.reshape(x.shape[0], -1)], 1)
        return model.encode(x)
    zs = [enc_flat(S)]
    for _ in range(n_rot):
        R = rot2(float(2 * math.pi * torch.rand(1, generator=gen)))
        zs.append(enc_flat(torch.cat([S[:, :2] @ R.T, S[:, 2:] @ R.T], 1)))
    return torch.cat(zs, 0)


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 90
    n_train, epochs, n_rot = (200, 12, 8) if SMOKE else (800, 80, 24)
    epochs = int(os.environ.get("STEP54_EPOCHS", epochs))
    print(line)
    print(f"STEP 54 (A4)  equivariant latent covariance is a G-invariant operator (clean SO(2) data)  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    S_t, S_tp1 = make_orbits(n_train, 30 if not SMOKE else 12, seed=SEED)
    St = make_orbits(400, 30 if not SMOKE else 12, seed=999)[0]
    gen = torch.Generator().manual_seed(50 + SEED)

    res = {}
    for nm, equiv, Model in [("equivariant", True, EquivWorldModel), ("MLP", False, MLPWorldModel)]:
        torch.manual_seed(SEED)
        model = Model()
        train(model, S_t, S_tp1, epochs, equiv=equiv)
        cov = covariance(latent_over_symmetric_set(model, St, n_rot, equiv, gen))
        rinv = rho_invariance_residual(cov, seed=300 + SEED)
        aniso = ell1_block_anisotropy(cov) if equiv else float("nan")
        res[nm] = dict(rho_inv_residual=rinv, ell1_anisotropy=aniso)
        extra = f", ℓ=1 block anisotropy {aniso:.3f} (0=isotropic)" if equiv else ""
        print(f"  {nm:12s}: ρ-invariance residual ||ρΣρ^T-Σ||/||Σ|| = {rinv:.4f}{extra}")

    eq, mlp = res["equivariant"]["rho_inv_residual"], res["MLP"]["rho_inv_residual"]
    ratio = mlp / max(eq, 1e-9)
    print(f"\n{line}\nSTEP 54 SUMMARY\n{line}")
    print(f"    ρ-invariance residual: equivariant {eq:.4f}  vs  MLP {mlp:.4f}  ({ratio:.1f}× larger)")
    print(f"    equivariant ℓ=1 block anisotropy {res['equivariant']['ell1_anisotropy']:.3f} "
          f"(≈0 ⇒ isotropic, the prescriptive group-given signature)")
    ok = (eq < 0.15 and ratio > 3.0 and res["equivariant"]["ell1_anisotropy"] < 0.25)
    if ok:
        print(f"\n    CONFIRMED: on genuinely rotation-symmetric data the equivariant latent's covariance is")
        print(f"        PINNED to the group rep ρ (residual {eq:.3f}, {ratio:.0f}× below the MLP) and its ℓ=1")
        print(f"        blocks are isotropic — a prescriptive, group-given second-moment signature the MLP lacks.")
    else:
        print(f"\n    INCONCLUSIVE: gate not met; reported as-is (no thresholds loosened).")

    res.update(passed=bool(ok), ratio=ratio, smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step54_latent_spectrum_v2_smoke" if SMOKE else "step54_latent_spectrum_v2"
    (fig_dir / f"{stem}.json").write_text(json.dumps(res, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(["equivariant\n(ℓ=0 ⊕ ℓ=1)", "MLP"], [max(eq, 1e-4), mlp], color=["C2", "C3"])
    ax.set_yscale("log"); ax.set_ylabel(r"$\|\rho\Sigma\rho^\top-\Sigma\|/\|\Sigma\|$  (lower = pinned to $\rho$)")
    ax.set_title(f"Step 54 (A4) — equivariant latent covariance is a $G$-invariant operator\n"
                 f"on rotation-symmetric data ({ratio:.0f}× below the MLP; "
                 f"ℓ=1 anisotropy {res['equivariant']['ell1_anisotropy']:.2f})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
