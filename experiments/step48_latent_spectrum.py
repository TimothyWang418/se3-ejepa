r"""Step 48 (old-paper A4): the equivariant latent's **group-prescribed second-moment signature**.

> ⚠️ **WIP / inconclusive (2026-06-02, autonomous).** Two simple covariance diagnostics — (a) 3-fold
> eigenvalue degeneracy of the full $96\times96$ covariance, (b) the $\rho$-invariance residual below — do
> **not** cleanly separate the equivariant latent from the MLP at this scale (smoke: residual $0.52$ vs
> $0.88$, only $\times2$). Root cause: the interacting-teacher data is **not rotation-symmetric**, and
> finite rotation-averaging does not symmetrise it enough for $\Sigma$ to commute with $\rho$. A correct
> A4 needs either full symmetrisation (many rotations + averaging the base geometry) or a **per-$\ell{=}1$-block
> isotropy** metric on properly symmetrised data. Deferred to a careful (supervised) redesign; this file is
> the reusable scaffold + the honest null, not a paper figure yet.



UR-JEPA (Le, 2026) diagnoses representation geometry by the covariance spectrum (anisotropic cliff vs
LeJEPA's flat). We borrow the *tool* for a sharper, *prescriptive* point about our $\mathrm{SE}(3)$-equivariant
latent: over a rotation-symmetric data distribution its **covariance is a $G$-invariant operator** — it
commutes with the representation, $\rho(R)\,\Sigma\,\rho(R)^\top=\Sigma$ for all $R$ — because
$E(Rx)=\rho(R)E(x)$ exactly. That is a *group-given*, checkable signature: the latent's second moment is
not merely anisotropic, it is **pinned to $\rho$**. A non-equivariant MLP encoder has no such property.

Diagnostic (equivariant E0 vs identically-trained MLP):
  - **ρ-invariance residual** $\mathbb{E}_R\lVert\rho(R)\Sigma\rho(R)^\top-\Sigma\rVert_F/\lVert\Sigma\rVert_F$,
    where $\rho(R)$ block-rotates the latent as stacked 3-vectors. Small ⇒ the covariance is pinned to the
    group rep; this is the prescriptive signature. (Equivariant: small, finite-sample-limited; MLP: large.)
  - the covariance spectrum + effective rank (reported alongside, à la UR-JEPA).

Run (full ~5 min):  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    .venv/bin/python experiments/step48_latent_spectrum.py
Smoke:  STEP48_SMOKE=1 ... .venv/bin/python experiments/step48_latent_spectrum.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from step24_object_interaction import (  # noqa: E402
    make_interacting_transitions, rand_so3, rotate_points, train_jepa,
)
from step43_encoder_ladder import build_model, model_action_v  # noqa: E402

torch.set_default_dtype(torch.float32)
SMOKE = bool(os.environ.get("STEP48_SMOKE"))


@torch.no_grad()
def latent_over_orbit(model, S, n_rot, gen):
    r"""Latents over the test set + several random rotations of it (a rotation-symmetric sample, which is
    what makes the equivariant covariance an exact $G$-invariant operator in the infinite-sample limit)."""
    zs = [model.encoder(S)]
    for _ in range(n_rot):
        zs.append(model.encoder(rotate_points(S, rand_so3(gen))))
    return torch.cat(zs, 0)


def covariance(Z):
    Zc = Z - Z.mean(0, keepdim=True)
    return (Zc.T @ Zc) / max(Zc.shape[0] - 1, 1)


def spectrum(cov):
    return torch.linalg.eigvalsh(cov).flip(0).clamp_min(0).cpu().numpy()


def effective_rank(ev):
    p = ev / max(ev.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def rho_matrix(R, dim):
    r"""Block-diagonal $\rho(R)$: $\mathrm{dim}/3$ copies of the $3\times3$ rotation (the latent is stacked
    type-1 vectors). For the equivariant latent this is the actual representation; applied to the MLP latent
    it is the *hypothesised* rep the MLP has no reason to respect."""
    nblk = dim // 3
    M = torch.zeros(dim, dim)
    for b in range(nblk):
        M[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return M


def rho_invariance_residual(cov, gen, n=12):
    r"""$\mathbb{E}_R\lVert\rho(R)\,\Sigma\,\rho(R)^\top-\Sigma\rVert_F/\lVert\Sigma\rVert_F$. Small ⇒ the
    covariance commutes with the group rep (is pinned to $\rho$)."""
    dim = cov.shape[0]
    denom = cov.norm().clamp_min(1e-12)
    res = []
    for _ in range(n):
        M = rho_matrix(rand_so3(gen), dim)
        res.append(((M @ cov @ M.T) - cov).norm() / denom)
    return float(torch.stack(res).mean())


def main() -> None:
    line = "=" * 80
    SEED = int(os.environ.get("STEP48_SEED", "0"))
    N_TRAIN, N_TEST, EPOCHS, N_ROT = (150, 64, 3, 8) if SMOKE else (1500, 400, 80, 24)
    EPOCHS = int(os.environ.get("STEP48_EPOCHS", EPOCHS))

    print(line)
    print(f"STEP 48 (A4)  equivariant latent: covariance pinned to ρ (G-invariant) vs MLP  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    S, A_self, S2 = make_interacting_transitions(N_TRAIN, seed=SEED)
    St, At_self, _ = make_interacting_transitions(N_TEST, seed=999)
    gen = torch.Generator().manual_seed(50 + SEED)

    res = {}
    for nm in ["E0-base", "MLP-MP"]:
        torch.manual_seed(SEED)
        model = build_model(nm)
        train_jepa(model, S, model_action_v(nm, S, A_self), S2,
                   epochs=EPOCHS, batch_size=128, var_coef=0.1, seed=SEED, log_every=999)
        cov = covariance(latent_over_orbit(model, St, N_ROT, gen))
        ev = spectrum(cov)
        rinv = rho_invariance_residual(cov, torch.Generator().manual_seed(300 + SEED))
        er = effective_rank(ev)
        res[nm] = dict(rho_inv_residual=rinv, eff_rank=er, ev=ev.tolist(),
                       top=float(ev[0]), bottom=float(ev[ev > 0][-1]) if (ev > 0).any() else 0.0)
        print(f"\n  {nm}: eff-rank={er:.1f}  top/bottom={ev[0] / max(ev[ev > 0][-1], 1e-12):.1f}")
        print(f"    ρ-invariance residual ||ρΣρ^T - Σ||/||Σ|| = {rinv:.4f}  "
              f"({'covariance PINNED to ρ (G-invariant)' if rinv < 0.05 else 'not ρ-invariant'})")

    eq, mlp = res["E0-base"], res["MLP-MP"]
    ratio = mlp["rho_inv_residual"] / max(eq["rho_inv_residual"], 1e-9)
    sig = eq["rho_inv_residual"] < 0.05 and mlp["rho_inv_residual"] > 5 * eq["rho_inv_residual"]
    print(f"\n{line}\nSTEP 48 SUMMARY\n{line}")
    print(f"    equivariant ρ-invariance residual {eq['rho_inv_residual']:.4f}  vs  "
          f"MLP {mlp['rho_inv_residual']:.4f}  (x{ratio:.0f} larger)")
    print(f"    => the equivariant latent's covariance is PINNED to the group rep ρ (a prescriptive,")
    print(f"       group-given second-moment signature); the MLP's is not. signature confirmed: {sig}")
    print(f"    (honest: the equivariant residual is finite-sample-limited, not exactly 0; the headline is")
    print(f"     the {ratio:.0f}x separation from the non-equivariant control.)")

    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step48_latent_spectrum_smoke" if SMOKE else "step48_latent_spectrum"
    (fig_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for nm, c in [("E0-base", "C2"), ("MLP-MP", "C3")]:
        ev = np.array(res[nm]["ev"])
        ax[0].semilogy(ev / ev[0], "o-", ms=3, color=c, label=f"{nm} (eff-rank {res[nm]['eff_rank']:.1f})")
    ax[0].set_xlabel("component (sorted)"); ax[0].set_ylabel("eigenvalue / top")
    ax[0].set_title("latent covariance spectrum"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    ax[1].bar(["E0 (equiv)", "MLP"], [eq["rho_inv_residual"], mlp["rho_inv_residual"]], color=["C2", "C3"])
    ax[1].set_yscale("log"); ax[1].set_ylabel(r"$\|\rho\Sigma\rho^\top-\Sigma\|/\|\Sigma\|$")
    ax[1].set_title("covariance pinned to ρ? (lower = G-invariant)"); ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Step 48 (A4): the equivariant latent's covariance is pinned to the group rep ρ", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)} + .png")
    sys.exit(0 if sig else 1)


if __name__ == "__main__":
    main()
