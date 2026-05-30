r"""$\mathrm{SE}(3)$-invariance of the partial-observability salience drive + EFE plan (Step 25).

Step 20 proved the *epistemic* drive (ensemble disagreement) is $\mathrm{SE}(3)$-invariant as a
**mechanism**. Step 25 makes the epistemic drive earn a **task win** in an ambiguous-goal
cue-foraging POMDP, and the claim this test guards is the geometric theorem that survives the move to
partial observability:

  The cue *salience* is the imagined probability of sensing the hidden-goal cue,
  $\eta=1-\prod_h\big(1-e^{-\lVert\hat z_h-z_c\rVert^2/2\delta^2}\big)$, a function of the **latent
  distance** to the cue only. Under a global motion $x\mapsto Rx+t$ the equivariant encoder sends
  every latent by the same orthogonal $\rho(R)$ ($E(Rx+t)=\rho(R)E(x)$), so each
  $\lVert\hat z_h-z_c\rVert$ is preserved and $\eta$ — hence the whole Expected Free Energy and the
  EFE-optimal plan — is **exactly $\mathrm{SE}(3)$-invariant / -equivariant**. *The agent's curiosity
  about where the information is does not depend on the global pose of the scene.* A non-equivariant
  encoder conflates pose with cue-proximity and breaks it (the control).

This is the unit-level mirror of the experiment's panel [C]
(``experiments/step25_active_inference_task.py``), pinned to the e3nn float floor at init AND after a
real Muon/AdamW + EMA-target + VICReg training run (the symmetry must survive optimisation). Probing
is done at :data:`EVAL_DTYPE` (float64), exactly as the experiment promotes its weights, so the VN
residuals sit at the double-precision floor and CEM tie-flips do not muddy the plan-equivariance read.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python tests/test_step25_salience_invariance.py
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
sys.path.insert(0, str(ROOT / "experiments"))  # for the Step 25 model + its Step 13/18 backbone

import torch  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    rand_so3,
)
from step18_se3_closed_loop import EVAL_DTYPE, _apply, centroid  # noqa: E402
from step25_active_inference_task import (  # noqa: E402
    efe_pomdp_plan,
    make_cue_tasks,
    salience_invariance,
)
from src.training.jepa import train_jepa  # noqa: E402

# Tolerances. The salience invariance is exact-by-construction (orthogonal $\rho(R)$ acting on a
# shared equivariant latent), so the VN residual sits at the float64 floor; the EFE plan equivariance
# inherits the CEM tie-flip floor (~the experiment's guard), a touch looser.
_FLOOR = 1e-4    # VN salience-field SE(3)-invariance residual (exact-by-construction; float64 floor)
_PLAN = 1e-2     # VN EFE-plan equivariance ||plan(Rx) - R.plan(x)||_inf (CEM tie-flip floor); the
                 # non-equivariant control must MISS this by a wide margin (the non-vacuity guard)

_CEM = dict(H=10, n_samples=96, n_iters=4, n_elite=12, sigma0=0.6, w_run=0.3)
_BETA = 12.0
_DELTA_FRAC = 0.6


@torch.no_grad()
def _plan_equiv(model, task: tuple, R: torch.Tensor, *, seed: int = 321) -> float:
    r"""$\lVert\text{plan}(Rx)-R\cdot\text{plan}(x)\rVert_\infty$ for the Phase-1 ($p=\tfrac12$) EFE plan.

    Same CEM seed for the seen and rotated runs, with the rotated run's exploration noise pre-rotated
    by ``R`` (``R_noise=R``) — so for the equivariant model the two plans coincide to the float floor.
    """
    X0, Xc, Xgp, Xgm, _b = task

    def plan(X0_, Xc_, Xgp_, Xgm_, Rn, g):
        delta = _DELTA_FRAC * float((model.encoder(Xc_) - model.encoder(X0_)).norm()) + 1e-9
        return efe_pomdp_plan(
            model, X0_, p=0.5, zgp=model.encoder(Xgp_), zgm=model.encoder(Xgm_),
            cgp=centroid(Xgp_), cgm=centroid(Xgm_), zc=model.encoder(Xc_), delta=delta,
            beta=_BETA, R_noise=Rn, w_t=0.5, gen=g, **_CEM,
        )

    z = torch.zeros(3, dtype=EVAL_DTYPE)
    base = plan(X0, Xc, Xgp, Xgm, None, torch.Generator().manual_seed(seed))
    rot = plan(_apply(X0, R, z), _apply(Xc, R, z), _apply(Xgp, R, z), _apply(Xgm, R, z),
               R, torch.Generator().manual_seed(seed))
    return (rot - torch.einsum("ij,hj->hi", R, base)).abs().max().item()


@torch.no_grad()
def _probe(model, task: tuple, gen: torch.Generator) -> dict[str, float]:
    r"""Salience-field SE(3)-invariance and EFE-plan equivariance residuals for one model."""
    X0, Xc = task[0], task[1]
    R = rand_so3(gen).to(EVAL_DTYPE)
    t = ((torch.rand(3, generator=gen) * 2 - 1) * 0.8).to(EVAL_DTYPE)
    g = torch.Generator().manual_seed(99)
    sal = salience_invariance(model, X0, Xc, R, t, H=_CEM["H"], gen=g)
    plan_eq = _plan_equiv(model, task, rand_so3(gen).to(EVAL_DTYPE))
    return {"salience": sal, "plan_equiv": plan_eq}


@lru_cache(maxsize=1)
def _trained() -> tuple[dict, dict]:
    r"""Build VN + MLP latent JEPAs, probe at init, train each (short), probe post-train.

    Returns ``(init, post)``, each ``{model_name: residual_dict}``. Cached so the two short training
    runs happen once across all pytest entry points. A real ``train_jepa`` (Muon/AdamW + EMA target +
    VICReg) run => a genuine "symmetry survives optimisation" guard, not an init-only check.
    """
    S, A, S2 = collect_cloud_transitions(256, seed=0)
    task = make_cue_tasks(1, seed=321)[0]
    gen = torch.Generator().manual_seed(3)

    init: dict[str, dict] = {}
    post: dict[str, dict] = {}
    for name, build in (("VN", build_eq_jepa), ("MLP", build_mlp_jepa)):
        model = build()
        model = model.to(EVAL_DTYPE)
        init[name] = _probe(model, task, gen)
        model = model.to(torch.float32)
        train_jepa(model, S, A, S2, epochs=8, batch_size=128, var_coef=0.1, seed=0, verbose=False)
        model = model.to(EVAL_DTYPE)
        post[name] = _probe(model, task, gen)
    return init, post


def test_vn_salience_field_is_se3_invariant() -> None:
    r"""VN: the cue salience $\eta$ is $\mathrm{SE}(3)$-invariant — at init AND post-train."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["VN"]
        assert r["salience"] < _FLOOR, (
            f"VN salience field not SE(3)-invariant @{phase}: {r['salience']:.2e} (>= {_FLOOR}); "
            "the epistemic drive must be geometry-independent"
        )


def test_vn_efe_plan_is_se3_equivariant() -> None:
    r"""VN: the EFE-optimal cue-foraging plan is $\mathrm{SE}(3)$-equivariant — init AND post-train."""
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["VN"]
        assert r["plan_equiv"] < _PLAN, (
            f"VN EFE plan not SE(3)-equivariant @{phase}: {r['plan_equiv']:.2e} (>= {_PLAN}); "
            "rotating the whole POMDP must rotate the plan"
        )


def test_mlp_breaks_se3_structure() -> None:
    r"""Non-equivariant control: the EFE *plan* is not $\mathrm{SE}(3)$-equivariant — VN claim not vacuous.

    We pin the break on the **EFE-plan** equivariance (the end-to-end decision quantity) rather than the
    salience scalar. The salience $\eta=1-\prod_h(1-s_h)$ can *saturate* (to $0$, or over many steps to
    $1$) for a collapsed / lightly-trained non-equivariant latent — the short 8-epoch control here —
    making it look vacuously invariant *in that one scalar* even though the model is not equivariant
    (the VN invariance is exact for any $\delta$). The plan equivariance breaks robustly at both phases;
    the full 60-epoch experiment additionally shows the salience break directly (MLP $\approx0.9$).
    """
    init, post = _trained()
    for phase, tbl in (("init", init), ("post-train", post)):
        r = tbl["MLP"]
        assert r["plan_equiv"] > _PLAN, (
            f"MLP EFE plan unexpectedly SE(3)-equivariant @{phase}: {r['plan_equiv']:.2e} (<= {_PLAN}); "
            "the VN plan-equivariance assertion would be vacuous"
        )


def main() -> None:
    """Print the salience-invariance + plan-equivariance residuals (init + post) and run all assertions."""
    init, post = _trained()
    print("Step 25 cue-foraging salience: SE(3)-invariance + EFE-plan equivariance residuals\n")
    hdr = f"    {'model':>6} | {'phase':>10} | {'salience-inv':>12} | {'plan-equiv':>11}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in ("VN", "MLP"):
        for phase, tbl in (("init", init), ("post-train", post)):
            r = tbl[name]
            print(f"    {name:>6} | {phase:>10} | {r['salience']:>12.2e} | {r['plan_equiv']:>11.2e}")

    test_vn_salience_field_is_se3_invariant()
    test_vn_efe_plan_is_se3_equivariant()
    test_mlp_breaks_se3_structure()
    print("\nPASS: VN cue salience is exactly SE(3)-invariant and its EFE plan SE(3)-equivariant through")
    print("      training; the non-equivariant control breaks the EFE-plan equivariance => assertions bite.")


if __name__ == "__main__":
    main()
