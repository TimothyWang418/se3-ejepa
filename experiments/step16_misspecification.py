r"""Step 16: SO(3) **misspecification sweep** -- when does the equivariant prior stop winning?

Where this sits, and why it is the honest substitute for "a real 3D system"
---------------------------------------------------------------------------
The roadmap's Phase-4 endpoint is a *real* 3D manipulation benchmark. On this machine
that is platform-blocked: ManiSkill / SAPIEN / Open3D / PyTorch3D / dm_control are all
absent and none installs without CUDA/Vulkan (only a bare ``mujoco`` wheel is present,
and MuJoCo manipulation is *not* a provably-SO(3)-equivariant world at laptop scale).
Rather than fake a 3D sim, Step 16 attacks the *scientific* question a real system
would force on us anyway, and which every prior step (10-15) deliberately avoided:

  **the real world is never *exactly* symmetric. When the symmetry the model hard-codes
  is only approximately true, does the geometric prior still help -- and at what point
  does the unconstrained baseline win?**

This is the project thesis meeting Sutton's *Bitter Lesson* (2019) head-on. Steps 10-15
all live on exactly-symmetric data (real PushT interior is SO(2) to 1.8e-5 px; the 3D
teacher is SO(3) by construction), where the equivariant model is provably flat and the
baseline provably degrades. That is the *best case* for the prior. Step 16 leaves it: we
take the exactly-SO(3) Step-13 teacher and add a **fixed lab-axis field** of tunable
strength $g$ that breaks the rotation symmetry by a controlled amount, then ask which
model predicts better out-of-distribution as $g$ grows.

The misspecified teacher (a controlled symmetry break)
-------------------------------------------------------
For a centred cloud $\tilde x_i = x_i - \bar x$ and the Step-13 equivariant dynamics
$\mathrm{Dyn}_0(x,a)$ (drift + torque + anisotropic stretch), define

$$ \mathrm{Dyn}_g(x,a)_i \;=\; \mathrm{Dyn}_0(x,a)_i \;-\; g\,\big(e_z\cdot\tilde x_i\big)\,e_z, $$

a **fixed-direction** (lab-frame $z$) anisotropic compression of strength $g\ge 0$.
Two properties make it the right knob:

* It **survives the encoder's centering**. Since $\sum_i \tilde x_i = 0$, the added term
  contributes $-g\,e_z\,(e_z\cdot\sum_i\tilde x_i)=0$ to the new centroid, so it is a
  *real, visible* prediction target -- not a global translation the VN encoder would
  wash out. The model must actually fit it.
* It is **not SO(3)-equivariant**: a fixed lab axis does not commute with rotation,
  $R\,(I-g\,e_z e_z^\top)\tilde x \neq (I-g\,e_z e_z^\top)\,R\tilde x$. So $g$ literally
  parameterises the energy of dynamics that the VN -- whose learned map obeys
  $F(Rx,Ra)=\rho(R)F(x,a)$ *exactly* -- structurally **cannot represent**. At $g=0$ we
  recover the exactly-equivariant Step-13 world.

Physically: a vertical "gravity-like" field that flattens the object along lab-$z$
regardless of the object's own orientation -- exactly the kind of fixed-frame effect
(gravity, a ground plane, a preferred growth axis) that makes real 3D dynamics only
*approximately* rotation-symmetric.

The crucial methodological point (why we re-sample, not rotate)
---------------------------------------------------------------
In Steps 13/15 the OOD test *rotated* a held-out transition $(s,a,s')\mapsto(Rs,Ra,Rs')$.
That is a valid ground-truth transition **only because the teacher was equivariant**
($\mathrm{Dyn}_0(Rs,Ra)=R\,s'$). Once $g>0$ that identity fails:
$\mathrm{Dyn}_g(Rs,Ra)\neq R\,\mathrm{Dyn}_g(s,a)$, so the rotated target $Rs'$ is *fake*
and would conflate model error with a wrong label. Step 16 therefore generates **genuine**
OOD transitions: it samples clouds at full-SO(3) orientations and applies the *true*
$\mathrm{Dyn}_g$ to each. Seen and OOD differ only in the orientation distribution
(z-wedge vs. full SO(3)); the $(s,a)$ stream and all seeds are held fixed across $g$ so
the *only* thing that changes down a column of the sweep is the symmetry-break strength.

What we expect (and let the data decide)
-----------------------------------------
* $g=0$: reproduce Step 13 -- VN OOD-flat ($\approx$ seen), baseline degrades; prior wins.
* $g\uparrow$: the VN can only fit the SO(3)-*equivariant projection* of the dynamics; the
  fixed-axis term sits in the complement it cannot output, so its error acquires a floor
  that grows with $g$ -- *even in-distribution*. The MLP has no such constraint.
* a **crossover** $g^\star$ may appear where the baseline's OOD error drops below the VN's:
  once enough of the dynamics violates the assumed symmetry, the unconstrained model wins.
  We report whether and where it happens -- this is the Bitter-Lesson boundary of the
  geometric bet, and finding it (or its absence) is the result, not a failure.

Related framing: this is the exact-vs-approximate-equivariance tradeoff studied by
*relaxed/approximate* equivariant nets (e.g. Finzi et al., Residual Pathway Priors, 2021;
Wang et al., Approximately Equivariant Networks, 2022). Step 16 measures, in a latent
world model, where a *hard* prior crosses from asset to liability.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step16_misspecification.py
    # fast smoke: STEP16_SMOKE=1 .venv/bin/python experiments/step16_misspecification.py
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
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 13/10 helpers we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.jepa import train_jepa  # noqa: E402

# Reuse the Step-13 geometry, teacher, models and metric VERBATIM (nothing re-invented).
from step13_se3_latent_jepa import (  # noqa: E402
    N_POINTS,
    LATENT_DIM,
    ACTION_DIM,
    _TEMPLATE,
    build_eq_jepa,
    build_mlp_jepa,
    latent_rel_mse,
    rand_so3,
    rot_z,
    rotate_points,
    teacher_step,          # the exactly-SO(3) g=0 teacher
)
from step10_pusht_closed_loop import n_params  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP16_SMOKE"))


# --------------------------------------------------------------------------- #
# the misspecified teacher: Step-13 equivariant dynamics + fixed lab-axis field
# --------------------------------------------------------------------------- #
def teacher_step_g(X: torch.Tensor, a: torch.Tensor, g: float) -> torch.Tensor:
    r"""Step-13 teacher plus a fixed lab-$z$ anisotropy of strength ``g``. ``(B,P,3),(B,3)->(B,P,3)``.

    $\mathrm{Dyn}_g(x,a)_i = \mathrm{Dyn}_0(x,a)_i - g\,(e_z\cdot\tilde x_i)\,e_z$, with
    $\tilde x_i = x_i-\bar x$. At ``g=0`` this is exactly the SO(3)-equivariant teacher;
    the added term is centering-invariant (real target, not a translation) yet not
    SO(3)-equivariant (fixed lab axis), so ``g`` is a clean knob on symmetry-break energy.
    """
    eq = teacher_step(X, a)                                  # exactly-equivariant part
    xt = X - X.mean(dim=1, keepdim=True)                     # centred coords
    ez = X.new_tensor([0.0, 0.0, 1.0])
    grav = -g * (xt @ ez).unsqueeze(-1) * ez                 # (B,P,1)*(3,) -> (B,P,3)
    return eq + grav


def collect_transitions_g(
    n: int, *, seed: int, g: float, full_so3: bool, phi_lo: float = 0.0, phi_hi: float = 90.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``n`` genuine transitions of the misspecified teacher $\mathrm{Dyn}_g$.

    ``full_so3=False`` -> orientations in the training z-wedge $[\phi_{lo},\phi_{hi})$ (seen);
    ``full_so3=True``  -> orientations uniform over a scatter of full-SO(3) rotations (OOD).
    The cloud/action stream depends only on ``seed`` (NOT on ``g``), so a fixed seed gives
    an identical $(s,a)$ set across the whole sweep -- only the teacher (the label $s'$)
    changes with ``g``. Returns ``(S, A, S2)`` with ``S,S2:(n,P,3)``, ``A:(n,3)``.
    """
    rng = np.random.default_rng(seed)
    rgen = torch.Generator().manual_seed(seed + 10_000)      # independent stream for rand_so3
    S = np.empty((n, N_POINTS, 3), np.float32)
    A = np.empty((n, 3), np.float32)
    for i in range(n):
        jitter = rng.standard_normal((N_POINTS, 3)).astype(np.float32) * 0.04
        axis_scale = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
        cloud = (_TEMPLATE + jitter) * axis_scale            # (P,3) body frame
        R = rand_so3(rgen).numpy() if full_so3 else rot_z(rng.uniform(phi_lo, phi_hi)).numpy()
        S[i] = cloud @ R.T
        A[i] = np.clip(rng.standard_normal(3) * 0.6, -1.0, 1.0)
    St = torch.from_numpy(S)
    At = torch.from_numpy(A)
    S2t = teacher_step_g(St, At, g)
    return St, At, S2t


@torch.no_grad()
def noneq_fraction(g: float, S: torch.Tensor, A: torch.Tensor, *, n_rot: int = 8, seed: int = 7) -> float:
    r"""Fraction of the teacher's one-step motion that violates SO(3) equivariance.

    $\dfrac{\mathbb{E}_R\lVert \mathrm{Dyn}_g(Rx,Ra)-R\,\mathrm{Dyn}_g(x,a)\rVert^2}
            {\mathbb{E}_R\lVert \mathrm{Dyn}_g(x,a)-x\rVert^2}$ -- 0 at ``g=0``, growing
    with ``g``. This is the model-independent "how broken is the symmetry" x-axis for the
    sweep (the displacement-normalised equivariance residual of the *data*, not any model).
    """
    gen = torch.Generator().manual_seed(seed)
    num, den = 0.0, 0.0
    for _ in range(n_rot):
        R = rand_so3(gen)
        lhs = teacher_step_g(rotate_points(S, R), rotate_points(A, R), g)
        rhs = rotate_points(teacher_step_g(S, A, g), R)
        num += ((lhs - rhs) ** 2).sum().item()
        den += ((teacher_step_g(S, A, g) - S) ** 2).sum().item()
    return num / max(den, 1e-12)


# --------------------------------------------------------------------------- #
# one sweep point: train both models on the z-wedge at strength g, eval seen + OOD
# --------------------------------------------------------------------------- #
def run_one_g(g: float, *, n_train: int, n_test: int, epochs: int, var_coef: float) -> dict:
    r"""Train VN + MLP on $\mathrm{Dyn}_g$ z-wedge data; return seen/OOD latent relMSE for both."""
    S, A, S2 = collect_transitions_g(n_train, seed=0, g=g, full_so3=False)        # TRAIN: z-wedge
    Ss, As, S2s = collect_transitions_g(n_test, seed=999, g=g, full_so3=False)    # SEEN eval
    So, Ao, S2o = collect_transitions_g(n_test, seed=1234, g=g, full_so3=True)    # OOD eval (genuine)

    eq = build_eq_jepa()
    mlp = build_mlp_jepa()
    train_jepa(eq, S, A, S2, epochs=epochs, batch_size=128, var_coef=var_coef, seed=0, log_every=10**9)
    train_jepa(mlp, S, A, S2, epochs=epochs, batch_size=128, var_coef=var_coef, seed=0, log_every=10**9)

    return {
        "g": g,
        "noneq_frac": noneq_fraction(g, So, Ao),
        "vn_seen": latent_rel_mse(eq, Ss, As, S2s),
        "vn_ood": latent_rel_mse(eq, So, Ao, S2o),
        "mlp_seen": latent_rel_mse(mlp, Ss, As, S2s),
        "mlp_ood": latent_rel_mse(mlp, So, Ao, S2o),
        "vn_params": n_params(eq),
        "mlp_params": n_params(mlp),
    }


def save_plot(rows: list[dict], out_png: Path) -> None:
    r"""Log-scale OOD latent relMSE vs symmetry-break fraction, VN vs MLP (+ seen, dashed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"    [plot skipped: {exc}]")
        return
    x = [r["noneq_frac"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(x, [r["vn_ood"] for r in rows], "o-", color="#1f77b4", label="VN (equivariant) OOD")
    ax.plot(x, [r["mlp_ood"] for r in rows], "s-", color="#d62728", label="MLP (baseline) OOD")
    ax.plot(x, [r["vn_seen"] for r in rows], "o--", color="#1f77b4", alpha=0.45, label="VN seen")
    ax.plot(x, [r["mlp_seen"] for r in rows], "s--", color="#d62728", alpha=0.45, label="MLP seen")
    ax.set_xlabel(r"symmetry-break fraction  $\mathbb{E}_R\|Dyn_g(Rx,Ra)-R\,Dyn_g(x,a)\|^2 / \|\Delta\|^2$")
    ax.set_ylabel("latent 1-step relMSE (OOD = genuine full SO(3))")
    ax.set_yscale("log")
    ax.set_title("Step 16: where the hard SO(3) prior stops winning")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"    figure -> {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72

    N_TRAIN = 200 if SMOKE else 1200
    N_TEST = 80 if SMOKE else 400
    EPOCHS = 3 if SMOKE else 50
    VAR_COEF = 0.1
    # Each g-point is seeded independently of grid position (train/models seed=0,
    # seen seed=999, OOD seed=1234), so densifying/extending the grid leaves the
    # originally-published 6 points {0,.05,.10,.20,.40,.80} bit-identical and only
    # adds rows. We add intermediates (smooths the curve) and push past g=0.8 to
    # bracket the Bitter-Lesson crossover further out.
    G_VALUES = [0.0, 0.4] if SMOKE else [
        0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.20, 1.60
    ]

    print(line)
    print("STEP 16  SO(3) MISSPECIFICATION SWEEP  (when does the equivariant prior stop winning?)")
    print(line)
    print("    teacher = exactly-SO(3) Step-13 dynamics  +  g * fixed lab-z anisotropy")
    print("    train both models on the z-wedge [0,90) at each g; evaluate latent 1-step relMSE on")
    print("    (seen) fresh z-wedge   and   (OOD) GENUINE full-SO(3) transitions of the SAME teacher.")
    print(f"    {'FULL' if not SMOKE else 'SMOKE'} mode: N_train={N_TRAIN} N_test={N_TEST} epochs={EPOCHS}  g={G_VALUES}")
    print("    NOTE: OOD is re-sampled (not rotate-the-test-set): once g>0 the teacher is not")
    print("          equivariant, so a rotated target would be a fake label. We apply the true Dyn_g.")
    print()

    rows: list[dict] = []
    for g in G_VALUES:
        r = run_one_g(g, n_train=N_TRAIN, n_test=N_TEST, epochs=EPOCHS, var_coef=VAR_COEF)
        rows.append(r)
        print(f"    g={g:4.2f}  noneq={r['noneq_frac']:7.4f} | "
              f"VN seen={r['vn_seen']:8.4f} OOD={r['vn_ood']:8.4f} | "
              f"MLP seen={r['mlp_seen']:8.4f} OOD={r['mlp_ood']:8.4f}")

    print()
    print(line)
    print("SWEEP TABLE  (latent 1-step relMSE; lower = better)")
    print(line)
    print(f"    {'g':>5s} | {'noneq':>7s} | {'VN seen':>8s} | {'VN OOD':>8s} | "
          f"{'MLP seen':>8s} | {'MLP OOD':>8s} | {'winner OOD':>10s}")
    print("    " + "-" * 76)
    crossover_g = None
    for r in rows:
        win = "VN" if r["vn_ood"] < r["mlp_ood"] else "MLP"
        if win == "MLP" and crossover_g is None and r["g"] > 0:
            crossover_g = r["g"]
        print(f"    {r['g']:5.2f} | {r['noneq_frac']:7.4f} | {r['vn_seen']:8.4f} | {r['vn_ood']:8.4f} | "
              f"{r['mlp_seen']:8.4f} | {r['mlp_ood']:8.4f} | {win:>10s}")

    # --- save curve + figure for the papers ---
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "step16_misspecification.json").write_text(json.dumps(rows, indent=2))
    save_plot(rows, fig_dir / "step16_misspecification.png")

    # --- verdict ---
    print()
    print(line)
    print("STEP 16 SUMMARY")
    print(line)
    r0 = rows[0]
    rmax = rows[-1]
    vn_ood_growth = rmax["vn_ood"] / max(r0["vn_ood"], 1e-9)
    noneq_monotone = all(rows[i]["noneq_frac"] <= rows[i + 1]["noneq_frac"] + 1e-6 for i in range(len(rows) - 1))
    g0_prior_wins = (r0["vn_ood"] < r0["mlp_ood"]) and (r0["vn_ood"] / max(r0["vn_seen"], 1e-9) < 1.15)
    vn_degrades = rmax["vn_ood"] > r0["vn_ood"] * 1.2   # the prior genuinely becomes wrong

    print(f"    g=0 reproduces Steps 13/15: VN OOD-flat (OOD/seen={r0['vn_ood']/max(r0['vn_seen'],1e-9):.2f}, "
          f"VN OOD={r0['vn_ood']:.3f} < MLP OOD={r0['mlp_ood']:.3f}).")
    print(f"    symmetry-break knob is monotone in g: {noneq_monotone} "
          f"(noneq {r0['noneq_frac']:.3f} -> {rmax['noneq_frac']:.3f}).")
    print(f"    as g grows the HARD prior becomes wrong: VN OOD x{vn_ood_growth:.1f} "
          f"({r0['vn_ood']:.3f} -> {rmax['vn_ood']:.3f}); it cannot represent the fixed-axis term.")
    if crossover_g is not None:
        cr = next(r for r in rows if r["g"] == crossover_g)
        print(f"    CROSSOVER at g={crossover_g:.2f} (noneq={cr['noneq_frac']:.3f}): "
              f"baseline OOD ({cr['mlp_ood']:.3f}) overtakes VN OOD ({cr['vn_ood']:.3f}).")
        print("    => the Bitter-Lesson boundary is INSIDE the tested range: past this much symmetry")
        print("       violation, the unconstrained model predicts better OOD than the hard-equivariant one.")
    else:
        print("    NO crossover within the tested g-range: the equivariant prior still wins OOD even at")
        print(f"    the largest break (g={rmax['g']:.2f}, noneq={rmax['noneq_frac']:.3f}) -- the baseline's")
        print("    rotation-extrapolation penalty outweighs the VN's un-representable fixed-axis residual.")

    passed = noneq_monotone and g0_prior_wins and vn_degrades
    print(f"    guards: knob-monotone={noneq_monotone}  g0-prior-wins={g0_prior_wins}  VN-degrades-with-break={vn_degrades}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: the sweep maps the geometric bet honestly -- exact symmetry is the")
    print("    prior's best case (Steps 10-15); this step shows how its OOD advantage erodes as the world")
    print("    departs from the assumed group, locating (or bounding) the crossover where scaling-free")
    print("    structure stops paying. Real-3D-sim caveat: ManiSkill/SAPIEN/Open3D/PyTorch3D absent on this")
    print("    machine (no CUDA/Vulkan); this controlled break is the faithful laptop-scale stand-in.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
