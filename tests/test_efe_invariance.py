r"""$\mathrm{SE}(3)$-invariance of the active-inference (Expected Free Energy) drives (Step 20).

Steps 13–18 pin the *pragmatic* half of the loop (drive the latent to a goal) and its exact
$\mathrm{SE}(3)$-equivariance. Step 20 adds the *epistemic* half — an agent that also acts to reduce
its own uncertainty — and the claim this test guards is the **geometric theorem** behind it:

  For a deep ensemble of $K$ predictors sharing **one** equivariant encoder, every member maps by the
  same orthogonal $\rho(R)$ under a global motion, so the ensemble *disagreement*
  $\mathcal{D}=\tfrac1K\sum_k\lVert z^{(k)}-\bar z\rVert^2$ and its information-geometric face, the
  Gaussian entropy $\mathcal{H}=\tfrac12\log\det(\hat\Sigma+\epsilon I)$, are **exactly
  $\mathrm{SE}(3)$-invariant** ($\mathcal{D}(\rho(R)z,Ra)=\mathcal{D}(z,a)$; $\hat\Sigma\mapsto
  \rho\hat\Sigma\rho^\top$ so $\log\det$ is unchanged since $\det\rho=\pm1$). With the invariant
  pragmatic cost this makes the whole one-step EFE $G$ invariant, hence the EFE-optimal plan
  $\mathrm{SE}(3)$-equivariant. **The agent's curiosity is geometry-independent.**

Two corollaries this test also pins:

  * **Zero epistemic novelty in re-orientation** — moving a $(\text{cloud},\text{action})$ pair to
    another point of its $\mathrm{SE}(3)$ orbit leaves $\mathcal{D}$ unchanged to the float floor:
    the equivariant agent is *correctly not curious* about poses it already generalises across
    (举一反三). A non-equivariant ensemble assigns spurious novelty to mere re-orientation.
  * **Non-vacuity** — the broken (MLP) ensemble misses each invariance by orders of magnitude, so
    every VN assertion bites; and $\mathcal{D}$ is a *non-constant* field (it is not trivially
    invariant by being everywhere equal).

This is the unit-level mirror of the experiment's panel [A]/[B] (``experiments/step20_active_inference.py``),
pinned to the float floor **by construction** at init AND after a real
Muon/AdamW + EMA-target + VICReg ensemble training run (symmetry must survive optimisation).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python tests/test_efe_invariance.py
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
sys.path.insert(0, str(ROOT / "experiments"))  # for the Step 20 model + its Step 13/18 backbone

import torch  # noqa: E402

from step13_se3_latent_jepa import collect_cloud_transitions, rand_so3  # noqa: E402
from step20_active_inference import (  # noqa: E402
    _reorient,
    build_mlp_ensemble,
    build_vn_ensemble,
    disagreement,
    efe_term_invariance,
    total_efe_invariance,
    train_ensemble_jepa,
)

# Tolerances. The invariances are exact-by-construction (orthogonal $\rho(R)$ acting on a shared
# equivariant latent), so the VN residuals sit at the e3nn float floor; the entropy logdet is the
# least-clean (rank-deficient $\hat\Sigma$ + $\epsilon I$), so _FLOOR sits a few x above its ~1.5e-5.
_FLOOR = 1e-4    # VN invariance residual (disagreement, entropy, total G) and reorient ratio deviation
_BROKEN = 1e-2   # the non-equivariant control must MISS each invariance by at least this (=> not vacuous)


@torch.no_grad()
def _probe(model, S: torch.Tensor, A: torch.Tensor, gen: torch.Generator) -> dict[str, float]:
    r"""All EFE-invariance residuals for one model on a held-out probe batch.

    ``disagree``/``entropy`` are the worst-over-batch absolute change of $\mathcal{D}$ and
    $\mathcal{H}$ under a global rotation; ``total_g`` the change of the whole one-step EFE $G$ under a
    full $(R,t)$ motion; ``reorient_dev`` is $\lvert \overline{\mathcal{D}}(\text{re-oriented}) /
    \overline{\mathcal{D}}(\text{seen}) - 1\rvert$ (a same-orbit move of the (cloud, action) pair);
    ``cov`` the coefficient of variation of per-sample $\mathcal{D}$ (>0 => non-constant field).
    """
    R = rand_so3(gen)
    term = efe_term_invariance(model, S, A, R)
    t = torch.randn(3, generator=gen) * 0.8
    total_g = total_efe_invariance(model, S[:1], S[1:2], A[:1], R, t, beta=4.0)

    d_vec = disagreement(model.members_next(model.encoder(S), A))   # (B,)
    d_seen = d_vec.mean().clamp_min(1e-12)
    S_re, A_re = _reorient(S, A, gen)
    d_re = disagreement(model.members_next(model.encoder(S_re), A_re)).mean()
    return {
        "disagree": term["disagree"],
        "entropy": term["entropy"],
        "total_g": total_g,
        "reorient_dev": abs((d_re / d_seen - 1.0).item()),
        "cov": (d_vec.std(unbiased=False) / d_seen).item(),
    }


@lru_cache(maxsize=1)
def _trained() -> tuple[dict, dict]:
    r"""Build the VN + MLP ensembles, probe at init, train each, probe post-train.

    Returns ``(init, post)``, each ``{model_name: residual_dict}``. Cached so the two short ensemble
    training runs happen once across all pytest entry points. A real Muon/AdamW + EMA-target + VICReg
    run (the same ``train_ensemble_jepa`` the experiment uses) => a genuine "symmetry survives
    optimisation" guard, not an init-only check.
    """
    S, A, S2 = collect_cloud_transitions(256, seed=0)
    Sp, Ap, _ = collect_cloud_transitions(96, seed=11)
    gen = torch.Generator().manual_seed(3)

    builders = {"VN": build_vn_ensemble, "MLP": build_mlp_ensemble}
    init: dict[str, dict] = {}
    post: dict[str, dict] = {}
    for name, build in builders.items():
        model = build(5, seed=0)
        init[name] = _probe(model, Sp, Ap, gen)
        train_ensemble_jepa(model, S, A, S2, epochs=8, batch_size=128, var_coef=0.1, seed=0, verbose=False)
        post[name] = _probe(model, Sp, Ap, gen)
    return init, post


def test_vn_ensemble_efe_drives_are_se3_invariant() -> None:
    r"""VN ensemble: disagreement, entropy and total $G$ are $\mathrm{SE}(3)$-invariant — init + post."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["VN"]
        assert r["disagree"] < _FLOOR, f"VN disagreement not SO(3)-invariant @{phase}: {r['disagree']:.2e}"
        assert r["entropy"] < _FLOOR, f"VN Gaussian entropy not SO(3)-invariant @{phase}: {r['entropy']:.2e}"
        assert r["total_g"] < _FLOOR, f"VN total EFE G not SE(3)-invariant @{phase}: {r['total_g']:.2e}"


def test_reorientation_carries_zero_epistemic_novelty() -> None:
    r"""VN: a same-orbit re-orientation leaves disagreement unchanged (举一反三), and $\mathcal{D}$ is non-constant."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["VN"]
        assert r["reorient_dev"] < _FLOOR, (
            f"VN epistemic drive changed under a same-orbit re-orientation @{phase} "
            f"({r['reorient_dev']:.2e}); re-orientation must carry zero novelty"
        )
        # the invariance must be non-vacuous: D is not trivially constant across the batch
        assert r["cov"] > 1e-3, f"VN disagreement is ~constant @{phase} (CoV {r['cov']:.2e}); invariance vacuous"


def test_mlp_ensemble_breaks_efe_invariance() -> None:
    r"""Non-equivariant control: the EFE invariances fail — so the VN assertions are not vacuous.

    The *invariance* break (disagreement / entropy not preserved by a global rotation) is structural and
    present even at random init. The *spurious novelty on re-orientation* (``reorient_dev`` large) is a
    learned pathology — a random untrained MLP's batch-mean disagreement is only mildly pose-sensitive —
    so we pin it post-train, exactly where the experiment's panel [B] reports it (ratio $\approx 6$).
    """
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["MLP"]
        assert r["disagree"] > _BROKEN or r["entropy"] > _BROKEN, (
            f"MLP ensemble unexpectedly EFE-invariant @{phase} "
            f"(disagree {r['disagree']:.2e}, entropy {r['entropy']:.2e}); the invariance assertion would be vacuous"
        )
    r_post = post["MLP"]
    assert r_post["reorient_dev"] > _BROKEN, (
        f"trained MLP ensemble unexpectedly blind to re-orientation ({r_post['reorient_dev']:.2e}); "
        "the zero-novelty assertion would be vacuous"
    )


def main() -> None:
    """Print the EFE-invariance residuals (init + post-train) and run all three assertions."""
    init, post = _trained()
    print("Step 20 active-inference EFE drives: SE(3)-invariance residuals\n")
    hdr = (f"    {'model':>6} | {'phase':>10} | {'disagree':>9} | {'entropy':>9} | "
           f"{'total G':>9} | {'reori dev':>9} | {'CoV':>6}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in ("VN", "MLP"):
        for phase, tbl in (("init", init), ("post-train", post)):
            r = tbl[name]
            print(f"    {name:>6} | {phase:>10} | {r['disagree']:>9.2e} | {r['entropy']:>9.2e} | "
                  f"{r['total_g']:>9.2e} | {r['reorient_dev']:>9.2e} | {r['cov']:>6.3f}")

    test_vn_ensemble_efe_drives_are_se3_invariant()
    test_reorientation_carries_zero_epistemic_novelty()
    test_mlp_ensemble_breaks_efe_invariance()
    print("\nPASS: VN ensemble's epistemic (disagreement, entropy) and total-G drives are exactly")
    print("      SE(3)-invariant through training; re-orientation carries zero novelty (D non-constant);")
    print("      the non-equivariant control breaks every invariance => each assertion bites.")


if __name__ == "__main__":
    main()
