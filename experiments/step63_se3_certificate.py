r"""Step 63 — the predictability certificate lifted from SO(2) to the non-abelian SO(3), on 3D point clouds.

Experiments 1–9 establish the configuration certificate for SO(2) / discrete groups; Experiment 9 grounds it on
real PushT contact physics. This step lifts the *same* multi-step rollout certificate to **SO(3) on 3D point
clouds** — a genuinely larger, **non-commutative** group, where the orbit is two-dimensional (axis on $S^2$ times
angle) rather than a circle. It reuses the laptop-proven 3D machinery of Steps 13/18 (an e3nn ``SE3PointEncoder``
whose latent is a stack of type-$\ell{=}1$ vectors, and a jointly-equivariant ``VNPredictor``), and the
rotate-the-orbit protocol of Step 59.

Setup. The dynamics is a constructed exactly-SO(3)-equivariant teacher (drift + torque + anisotropic stretch on a
24-point cloud); like Experiments 1–7 this is a *toy* — the real-data anchor remains Experiment 9 (PushT). What is
NOT vacuous here is that the **learned** equivariant model (encoder + predictor) keeps the symmetry after training
and its **multi-step latent rollout generalizes across all of SO(3) from a $z$-wedge of training rotations**, while a
non-equivariant MLP — trained on identical data — degrades out of the wedge. We train one-step on a $z$-wedge
($\varphi\in[0,90^\circ)$), then roll each model out $H$ steps and measure rollout relMSE over the SO(3) orbit:
the in-wedge identity, and OOD rotations (off-axis $90^\circ$, the antipode, random SO(3)).

Honest gate at horizon $H$ (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   equivariant encoder+predictor stay SO(3)-equivariant after training: eq_resid < 1e-3;
  (ii)  equivariant rollout flat over the SO(3) orbit:                        eq_orbit_ratio < 1.15;
  (iii) equivariant competitive in-distribution:                             eq_in < 1.5 * mlp_in;
  (iv)  the non-equivariant baseline degrades out of the wedge:              mlp_orbit_ratio > 2.0.

Run:        .venv/bin/python experiments/step63_se3_certificate.py
Seeded:     STEP63_SEED=0|1|2 .venv/bin/python experiments/step63_se3_certificate.py
Writes:     papers/figures/step63_se3_certificate.{json,png}
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from step13_se3_latent_jepa import (  # noqa: E402
    ACTION_DIM,
    LATENT_DIM,
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    rand_so3,
    rotate_points,
    teacher_step,
)
from src.training.jepa import train_jepa  # noqa: E402

SEED = int(os.environ.get("STEP63_SEED", "0"))
SMOKE = os.environ.get("STEP63_SMOKE", "0") == "1"

N_TRAIN = 200 if SMOKE else 1500
N_TRAJ = 64 if SMOKE else 256
EPOCHS = 4 if SMOKE else 60
H = 5
HS = [1, 3, 5]
VAR_COEF = 0.1                              # 3D needs more than 2D's 0.04 (Step 13)
FIG = ROOT / "papers" / "figures"


def axis_angle_R(axis, deg):
    r"""A single named rotation matrix (degrees) about a unit axis, for the orbit bins."""
    th = math.radians(deg)
    ax = np.asarray(axis, dtype=np.float64); ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return torch.tensor(R, dtype=torch.float32)


def collect_cloud_trajs(n_traj: int, horizon: int, *, seed: int):
    r"""``n_traj`` length-``horizon`` teacher trajectories on the $z$-wedge: clouds ``(n,H+1,P,3)`` and actions
    ``(n,H,3)``. Start clouds come from :func:`collect_cloud_transitions` (the $z$-wedge template); the exactly
    SO(3)-equivariant ``teacher_step`` rolls them, so the rotated trajectory is itself a valid teacher trajectory."""
    S0, _, _ = collect_cloud_transitions(n_traj, seed=seed, phi_lo=0.0, phi_hi=90.0)  # (n,P,3) z-wedge starts
    g = torch.Generator().manual_seed(seed + 1)
    clouds, acts = [S0], []
    s = S0
    for _ in range(horizon):
        a = torch.randn(n_traj, ACTION_DIM, generator=g) * 0.6
        s = teacher_step(s, a)
        clouds.append(s); acts.append(a)
    return torch.stack(clouds, dim=1), torch.stack(acts, dim=1)   # (n,H+1,P,3), (n,H,3)


def rotate_traj_3d(S: torch.Tensor, A: torch.Tensor, R: torch.Tensor):
    r"""Rotate a whole cloud trajectory by $R\in\mathrm{SO}(3)$: clouds $X\mapsto XR^\top$, actions $a\mapsto Ra$."""
    Sr = rotate_points(S, R)                                       # (n,H+1,P,3)
    Ar = torch.einsum("ij,nhj->nhi", R, A)                        # (n,H,3)
    return Sr, Ar


@torch.no_grad()
def rollout_relmse(model, S, A, horizon: int) -> float:
    z = model.encoder(S[:, 0])
    for t in range(horizon):
        z = model.predictor(z, A[:, t])
    z_true = model.encoder(S[:, horizon])
    rel = ((z - z_true) ** 2).sum(1) / (z_true ** 2).sum(1).clamp_min(1e-9)
    return float(rel.mean())


@torch.no_grad()
def equiv_residual(model, S, A) -> float:
    r"""$\max\lVert f(\rho(R)E(x),Ra)-\rho(R)f(E(x),a)\rVert$ on the encoder+predictor at a random SO(3) (the
    architectural floor for the equivariant model; large for the ordinary one)."""
    g = torch.Generator().manual_seed(7)
    R = rand_so3(g)
    x0, a0 = S[:64, 0], A[:64, 0]
    z = model.encoder(x0)
    lhs = model.predictor(z, a0)
    lhs = rotate_points(lhs.reshape(lhs.shape[0], -1, 3), R).reshape(lhs.shape[0], -1)   # ρ(R) f(z,a)
    zr = model.encoder(rotate_points(x0, R))
    ar = torch.einsum("ij,bj->bi", R, a0)
    rhs = model.predictor(zr, ar)                                                        # f(ρ(R)z, Ra)
    return float((lhs - rhs).abs().max())


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)

    S_tr, A_tr, S2_tr = collect_cloud_transitions(N_TRAIN, seed=SEED, phi_lo=0.0, phi_hi=90.0)
    S_te, A_te = collect_cloud_trajs(N_TRAJ, H, seed=9000 + SEED)
    print(f"[step63] seed={SEED} train={tuple(S_tr.shape)} traj={tuple(S_te.shape)} latent={LATENT_DIM}",
          file=sys.stderr)

    eq = build_eq_jepa()
    mlp = build_mlp_jepa()
    train_jepa(eq, S_tr, A_tr, S2_tr, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=SEED, verbose=False)
    train_jepa(mlp, S_tr, A_tr, S2_tr, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=SEED, verbose=False)

    # SO(3) orbit: identity (in-wedge seen) + OOD rotations (off-axis 90, antipode, random SO(3)) ------
    g = torch.Generator().manual_seed(100 + SEED)
    seen = [("identity", torch.eye(3))]
    far = [("z 180", axis_angle_R([0, 0, 1], 180.0)),
           ("x 90", axis_angle_R([1, 0, 0], 90.0)),
           ("y 90", axis_angle_R([0, 1, 0], 90.0))]
    far += [(f"rand{i}", rand_so3(g)) for i in range(3)]
    orbit = seen + far
    orbit_traj = {lab: rotate_traj_3d(S_te, A_te, R) for lab, R in orbit}

    def curves(model):
        return {h: {lab: rollout_relmse(model, Sb, Ab, h) for lab, (Sb, Ab) in orbit_traj.items()} for h in HS}

    eq_c, mlp_c = curves(eq), curves(mlp)
    seen_labs = [l for l, _ in seen]; far_labs = [l for l, _ in far]
    mean_in = lambda c, h: float(np.mean([c[h][l] for l in seen_labs]))    # noqa: E731
    mean_far = lambda c, h: float(np.mean([c[h][l] for l in far_labs]))    # noqa: E731

    eq_in = {h: mean_in(eq_c, h) for h in HS}
    eq_far = {h: mean_far(eq_c, h) for h in HS}
    mlp_in = {h: mean_in(mlp_c, h) for h in HS}
    mlp_far = {h: mean_far(mlp_c, h) for h in HS}
    eq_ratio = {h: eq_far[h] / max(eq_in[h], 1e-12) for h in HS}
    mlp_ratio = {h: mlp_far[h] / max(mlp_in[h], 1e-12) for h in HS}
    eq_resid = equiv_residual(eq, S_te, A_te)
    mlp_resid = equiv_residual(mlp, S_te, A_te)
    floor_pen = {h: mlp_far[h] / max(eq_far[h], 1e-12) for h in HS}

    ok_equiv = eq_resid < 1e-3
    ok_flat = eq_ratio[H] < 1.15
    ok_compete = eq_in[H] < 1.5 * mlp_in[H]
    ok_degrade = mlp_ratio[H] > 2.0
    passed = ok_equiv and ok_flat and ok_compete and ok_degrade

    result = {
        "passed": passed,
        "gate": {"equiv": ok_equiv, "flat": ok_flat, "compete": ok_compete, "degrade": ok_degrade},
        "eq_resid": eq_resid, "mlp_resid": mlp_resid,
        "eq_in": eq_in, "eq_far": eq_far, "mlp_in": mlp_in, "mlp_far": mlp_far,
        "eq_orbit_ratio": eq_ratio, "mlp_orbit_ratio": mlp_ratio, "floor_penalty": floor_pen,
        "eq_curves": {str(h): eq_c[h] for h in HS}, "mlp_curves": {str(h): mlp_c[h] for h in HS},
        "horizons": HS, "headline_H": H, "smoke": SMOKE, "seed": SEED,
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "step63_se3_certificate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # figure -----------------------------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    labs = seen_labs + far_labs
    xs = np.arange(len(labs))
    axL.bar(xs - 0.2, [eq_c[H][l] for l in labs], 0.4, color="C2", label="SO(3)-equivariant")
    axL.bar(xs + 0.2, [mlp_c[H][l] for l in labs], 0.4, color="C3", label="MLP baseline")
    axL.axvline(0.5, color="0.6", ls=":")
    axL.set_yscale("log"); axL.set_xticks(xs); axL.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    axL.set_ylabel(f"{H}-step rollout relMSE"); axL.legend(fontsize=8)
    axL.set_title(f"SO(3) orbit (seen | OOD): equivariant flat (x{eq_ratio[H]:.2f}), MLP climbs (x{mlp_ratio[H]:.1f})")
    axR.plot(HS, [floor_pen[h] for h in HS], "s-", color="C3", label="MLP out-of-wedge / equiv floor")
    axR.axhline(1.0, color="C2", ls="--", lw=2, label="equivariant floor")
    axR.set_xlabel("rollout horizon H"); axR.set_ylabel("OOD relMSE / equivariant floor")
    axR.set_title("no baseline reaches the equivariant floor (any horizon)"); axR.legend(fontsize=8)
    fig.suptitle("Step 63 — the predictability certificate on SO(3) (3D point clouds, constructed teacher)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "step63_se3_certificate.png", dpi=130)

    print(f"[step63] eq-resid {eq_resid:.1e} (mlp {mlp_resid:.1e}); H={H} "
          f"eq in/far {eq_in[H]:.4f}/{eq_far[H]:.4f} (ratio {eq_ratio[H]:.2f}); "
          f"mlp in/far {mlp_in[H]:.4f}/{mlp_far[H]:.4f} (ratio {mlp_ratio[H]:.1f}); floor-pen {floor_pen[H]:.1f}x",
          file=sys.stderr)
    if passed:
        print(f"[step63] CERTIFICATE on SO(3): learned equivariant model flat over the non-abelian orbit "
              f"(x{eq_ratio[H]:.2f}, resid {eq_resid:.0e}) and competitive; MLP degrades x{mlp_ratio[H]:.1f} OOD "
              f"({floor_pen[H]:.1f}x above the floor).", file=sys.stderr)
    else:
        bad = [k for k, v in result["gate"].items() if not v]
        print(f"[step63] INCONCLUSIVE: gate not met ({bad}); reported as-is (no thresholds loosened).", file=sys.stderr)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
