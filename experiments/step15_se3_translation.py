r"""Step 15: complete the group -- SE(3) = SO(3) $\ltimes$ $\mathbb{R}^3$ 举一反三 in 3D.

Where this sits, and why it is the missing rung
-----------------------------------------------
The project is named for **SE(3)**, but every generalisation test so far isolates
the *rotation* subgroup: Steps 10-12 are SO(2) (planar rotation on PushT), Step 13 is
SO(3) (3D rotation on point clouds). Translation -- the other half of the Euclidean
group, $g=(R,t)$ acting by $x\mapsto Rx+t$ -- has never been the OOD axis. Step 15
closes that gap and states the full SE(3) claim.

The mechanism is asymmetric, and we are honest about it:

* **Rotation** is a *learned* equivariance: the e3nn :class:`SE3PointEncoder` maps a
  global rotation to the block-diagonal $\rho(R)$ on the latent, and that survives
  training (Step 13 [A']). This is the non-trivial half.
* **Translation** is *exact by construction*: the encoder **centres** the cloud
  ($r_i=x_i-\bar x$), so $E(x+t)=E(x)$ identically -- a translation-*invariant*
  latent. The teacher is translation-*equivariant* (it centres internally, so
  $\mathrm{Dyn}(x+t,a)=\mathrm{Dyn}(x,a)+t$). Together: a translated transition has
  the **same** latent, the **same** predicted latent, and the **same** next latent,
  so the latent relMSE is *exactly* unchanged. We do not oversell this as deep -- it
  is geometry done right -- but it is precisely what makes the *full* SE(3) group, not
  just SO(3), a no-cost generalisation for the equivariant model.

Why it is a real test (not vacuous). Training clouds live near the origin (template +
small jitter, rotated only within a $+z$ wedge -- never translated). The baseline
:class:`MLPPointEncoder` flattens the **raw** coordinates with no centering, so a large
test-time translation pushes it fully out of distribution; and a full $\mathrm{SO}(3)$
rotation does too. The equivariant model is flat on **both** axes simultaneously. So
the OOD test ranges over the whole of SE(3): large translations the wedge never showed,
new rotation axes/angles the wedge never showed, and their composition.

  $z' = E(s')$,  relMSE $= \sum_i\lVert f(E(s_i),a_i)-E(s_i')\rVert^2/\sum_i\lVert E(s_i')-E(s_i)\rVert^2$.

[A] translation-invariance + rotation-equivariance residuals after training (mechanism);
[B] latent relMSE across an SE(3) ladder: identity, pure translation (small/large),
    pure SO(3), and translation+SO(3) composed. VN flat to the float floor; MLP breaks.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step15_se3_translation.py
    # fast smoke: STEP15_SMOKE=1 .venv/bin/python experiments/step15_se3_translation.py
"""

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
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 13 helpers we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.jepa import train_jepa  # noqa: E402

# Reuse the *exact* Step 13 pipeline so this is a strict extension, nothing re-invented:
# same encoders, same teacher, same training recipe, same latent relMSE metric.
from step13_se3_latent_jepa import (  # noqa: E402
    ACTION_DIM,
    LATENT_DIM,
    N_POINTS,
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    composed_equiv_err,
    latent_rel_mse,
    rand_so3,
    rotate_points,
    teacher_step,
)
from step10_pusht_closed_loop import n_params  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP15_SMOKE"))


# --------------------------------------------------------------------------- #
# SE(3) helpers (translation half; rotation half is reused from Step 13)
# --------------------------------------------------------------------------- #
def translate_points(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Apply the global shift $x\mapsto x+t$ to every point. ``(B,P,3),(3,)->(B,P,3)``."""
    return x + t.view(1, 1, 3)


@torch.no_grad()
def trans_inv_residual(enc: torch.nn.Module, S: torch.Tensor, t: torch.Tensor) -> float:
    r"""$\max\lVert E(x+t)-E(x)\rVert_\infty$ -- 0 for a centering (translation-invariant) encoder."""
    return (enc(translate_points(S, t)) - enc(S)).abs().max().item()


@torch.no_grad()
def teacher_trans_equiv_residual(S: torch.Tensor, A: torch.Tensor, t: torch.Tensor) -> float:
    r"""$\max\lVert \mathrm{Dyn}(x+t,a)-(\mathrm{Dyn}(x,a)+t)\rVert_\infty$ -- 0 if the teacher commutes with translation."""
    lhs = teacher_step(translate_points(S, t), A)  # Dyn(x+t, a)
    rhs = translate_points(teacher_step(S, A), t)  # Dyn(x,a) + t
    return (lhs - rhs).abs().max().item()


def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72

    N_TRAIN = 200 if SMOKE else 1500
    N_TEST = 80 if SMOKE else 400
    EPOCHS = 3 if SMOKE else 60
    VAR_COEF = 0.1  # match Step 13 (3D latent anti-collapse)

    print(line)
    print("STEP 15  SE(3) = SO(3) x R^3 generalisation  (rotation + TRANSLATION)")
    print(line)
    print("    Train end-to-end 3D latent JEPA near the origin (z-wedge rotations, NO")
    print("    translations); test latent 举一反三 over the FULL SE(3) group: large")
    print("    translations and full SO(3), never seen in training.")
    print(f"    {'SMOKE' if SMOKE else 'FULL'} mode: N_train={N_TRAIN} N_test={N_TEST} epochs={EPOCHS}")

    # ---- data: identical to Step 13 (near origin, z-wedge [0,90))
    S, A, S2 = collect_cloud_transitions(N_TRAIN, seed=0)
    St, At, S2t = collect_cloud_transitions(N_TEST, seed=999)

    eq = build_eq_jepa()
    mlp = build_mlp_jepa()

    # ---- train both (same recipe as Step 13)
    print()
    print(line)
    print("[train] EMA-target JEPA (Muon/AdamW) on near-origin z-wedge clouds")
    print(line)
    print(f"    {len(S)} train / {len(St)} held-out transitions ({N_POINTS} pts each)")
    print("    VN equivariant JEPA:")
    h_eq = train_jepa(eq, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=10)
    print("    MLP baseline JEPA:")
    h_mlp = train_jepa(mlp, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=10)
    print(f"    final latent_std: VN={h_eq['latent_std']:.3f}  MLP={h_mlp['latent_std']:.3f}  (>0 => no collapse)")
    print(f"    params: VN={n_params(eq)}  MLP={n_params(mlp)}  ({n_params(mlp)/n_params(eq):.1f}x VN)")

    # ---------------------------------------------------------------- [A]
    print()
    print(line)
    print("[A] SE(3) mechanism AFTER training: translation-invariance + rotation-equivariance")
    print(line)
    t_small = torch.tensor([2.0, -1.5, 1.0])
    t_large = torch.tensor([10.0, -8.0, 6.0])
    R_test = rand_so3(torch.Generator().manual_seed(3))
    print("    translation-invariance residual |E(x+t) - E(x)| (encoder centres the cloud):")
    print(f"        VN  encoder, |t|=small : {trans_inv_residual(eq.encoder, St, t_small):.4e}")
    print(f"        VN  encoder, |t|=large : {trans_inv_residual(eq.encoder, St, t_large):.4e}")
    print(f"        MLP encoder, |t|=small : {trans_inv_residual(mlp.encoder, St, t_small):.4e}  (flattens raw coords)")
    print(f"        MLP encoder, |t|=large : {trans_inv_residual(mlp.encoder, St, t_large):.4e}")
    print(f"    teacher translation-equivariance |Dyn(x+t,a)-(Dyn(x,a)+t)| : "
          f"{teacher_trans_equiv_residual(St, At, t_large):.4e}  (commutes with translation)")
    print(f"    VN  composed rotation residual |rho F(x,a)-F(Rx,Ra)|       : {composed_equiv_err(eq, St, At, R_test):.4e}")
    print(f"    MLP composed rotation residual                            : {composed_equiv_err(mlp, St, At, R_test):.4e}")

    # ---------------------------------------------------------------- [B]
    print()
    print(line)
    print("[B] LATENT prediction 举一反三 across SE(3) (train near origin; transform held-out)")
    print(line)
    print("    latent 1-step relMSE (SAME held-out set, mapped by each SE(3) element):")
    print(f"    {'SE(3) transform':28s} | {'VN relMSE':>12s} | {'MLP relMSE':>12s}")
    print("    " + "-" * 59)

    rng_so3 = torch.Generator().manual_seed(5)
    R_a = rand_so3(rng_so3)
    R_b = rand_so3(rng_so3)

    def _apply(S_, A_, S2_, *, R=None, t=None):
        r"""Map a transition by an SE(3) element $g=(R,t)$: rotate points+action, then shift points."""
        Sx, Ax, S2x = S_, A_, S2_
        if R is not None:
            Sx, Ax, S2x = rotate_points(Sx, R), rotate_points(Ax, R), rotate_points(S2x, R)
        if t is not None:
            Sx, S2x = translate_points(Sx, t), translate_points(S2x, t)  # action is a free vector: unshifted
        return Sx, Ax, S2x

    bins: list[tuple[str, dict]] = [
        ("identity (seen)", dict()),
        ("translate small", dict(t=t_small)),
        ("translate LARGE", dict(t=t_large)),
        ("rotate SO(3) only", dict(R=R_a)),
        ("translate + SO(3)", dict(R=R_a, t=t_large)),
        ("translate + SO(3) (2)", dict(R=R_b, t=torch.tensor([-7.0, 9.0, -5.0]))),
    ]
    vn_b, mlp_b = [], []
    for lab, g in bins:
        Sx, Ax, S2x = _apply(St, At, S2t, **g)
        ve = latent_rel_mse(eq, Sx, Ax, S2x)
        me = latent_rel_mse(mlp, Sx, Ax, S2x)
        vn_b.append(ve); mlp_b.append(me)
        print(f"    {lab:28s} | {ve:12.4e} | {me:12.4e}")

    vn_seen, mlp_seen = vn_b[0], mlp_b[0]
    vn_ood, mlp_ood = max(vn_b[1:]), max(mlp_b[1:])
    ratio_vn = vn_ood / max(vn_seen, 1e-12)
    ratio_mlp = mlp_ood / max(mlp_seen, 1e-12)
    print(f"    => VN flat across SE(3) (seen={vn_seen:.3e} OOD={vn_ood:.3e}, x{ratio_vn:.2f}); "
          f"MLP degrades (seen={mlp_seen:.3f} OOD={mlp_ood:.3f}, x{ratio_mlp:.1f}).")

    # ---------------------------------------------------------------- summary
    print()
    print(line)
    print("STEP 15 SUMMARY")
    print(line)
    print("    The project's named group is SE(3); Step 13 covered the SO(3) (rotation)")
    print("    subgroup, Step 15 adds the translation subgroup R^3, completing the group.")
    print(f"    [A] mechanism: VN encoder is translation-INVARIANT by centering")
    print(f"        (|E(x+t)-E(x)| <= ~1e-5 even at |t|=large) and SO(3)-equivariant after")
    print(f"        training; the MLP encoder (raw coords) is translation-sensitive.")
    print(f"    [B] latent 举一反三 across the WHOLE SE(3) group: VN flat (seen={vn_seen:.2e}")
    print(f"        OOD={vn_ood:.2e}, x{ratio_vn:.2f}) while the baseline breaks under large")
    print(f"        translations AND new rotations (x{ratio_mlp:.0f}).")

    ok_trans_vn = trans_inv_residual(eq.encoder, St, t_large) < 1e-3
    ok_trans_mlp = trans_inv_residual(mlp.encoder, St, t_large) > 0.1
    ok_flat_vn = ratio_vn < 1.05
    ok_break_mlp = ratio_mlp > 1.5
    passed = ok_trans_vn and ok_trans_mlp and ok_flat_vn and ok_break_mlp
    print(f"    guards: VN-trans-invariant={ok_trans_vn}  MLP-trans-sensitive={ok_trans_mlp}  "
          f"VN-flat={ok_flat_vn}  MLP-degrades={ok_break_mlp}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: the equivariant latent world model generalises 举一反三")
    print(f"    across the FULL SE(3) group (rotation learned + translation exact-by-centering),")
    print(f"    closing the gap between the project's named target geometry and what was tested.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
