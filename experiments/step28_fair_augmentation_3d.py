r"""Step 28 (3D arm): the fair-augmentation baseline, lifted from $\mathrm{SO}(2)$ to $\mathrm{SO}(3)$.

The 2D arm (``experiments/step28_fair_augmentation_baseline.py``) established, on the
exactly-SO(2)-equivariant teacher: rotation augmentation **closes the across-group TASK metric**
(full coverage $\to$ OOD/seen $\times1.06$, $\approx$ the VN's $\times1.00$) but **never reaches the
architecture's EXACT equivariance** ($\Delta_{\mathrm{eq}}$ plateaus $\sim\!10^5\times$ above the
float floor). This arm confirms the **same conclusion in 3D $\mathrm{SO}(3)$**, where the worry is that
the richer group (3 rotational DoF, not 1) might change the picture.

What "coverage" means in 3D
---------------------------
In 2D the augmentation coverage is an arc $[0,\theta_{\max})$ of $\mathrm{SO}(2)$. In 3D we use the
natural analogue: a **geodesic ball** of rotation angle $\le\theta_{\max}$ about a *random axis*
(axis uniform on $S^2$, angle $\sim U[0,\theta_{\max}]$). Since every rotation has an angle in
$[0,180°]$, $\theta_{\max}=180°$ covers **all of $\mathrm{SO}(3)$**. We sweep
$\theta_{\max}\in\{90°,135°,180°\}$ (all $\ge90°$, so the seen $z$-wedge $[0,90°)$ is always inside the
ball, keeping the OOD/seen ratio's denominator genuinely in-distribution).

Setup (mirrors step8/step28-2D, lifted to dim=3; reuses step13's $\mathrm{SO}(3)$ helpers)
---------------------------------------------------------------------------------------
A frozen random **Vector-Neuron teacher** with dim=3 (exactly $\mathrm{SO}(3)$-equivariant by
construction: VN layers are dimension-agnostic), a **VN student** (dim=3, equivariance hard-wired),
and a plain **MLP baseline** on the flattened 3D state. VN + MLP(no-aug) train on a thin $z$-axis wedge
$[0,90°)$ (= step13's protocol); MLP+aug($\theta_{\max}$) augments the same base scenes with random
$\mathrm{SO}(3)$ rotations of angle $\le\theta_{\max}$. The heavy e3nn point-cloud pipeline was already
validated in Steps 13/18; this controlled arm isolates the **augmentation** question cheaply.

Measurements (identical to the 2D arm)
--------------------------------------
  [1] **task metric** -- OOD/seen relMSE ratio (OOD = random global $\mathrm{SO}(3)$ off the wedge);
      expected to *fall toward 1* as $\theta_{\max}\to180°$.
  [2] **exactness** -- $\Delta_{\mathrm{eq}}=\max_R\lVert f(Rx)-Rf(x)\rVert/\lVert f(x)\rVert$ over random
      $\mathrm{SO}(3)$ probes; expected to **plateau orders of magnitude above** the VN's float floor.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step28_fair_augmentation_3d.py
Smoke (~40 s):
    STEP28_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step28_fair_augmentation_3d.py
"""

import json
import math
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

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models.structured import VNLinear, VNReLU  # noqa: E402

torch.set_default_dtype(torch.float32)

N_STATE = 6
DIM = 3
SMOKE = bool(os.environ.get("STEP28_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- step8/step28 lifted to dim=3 (VN layers are dimension-agnostic and
# exactly equivariant in 3D; cf. src/models/structured.py).
# --------------------------------------------------------------------------- #
class EquivariantWorld(nn.Module):
    r"""Frozen random *world*: an exactly $\mathrm{SO}(3)$-equivariant one-step map (one VN nonlinearity).

    ``forward: (B, n_state, 3), (B, 1, 3) -> (B, n_state, 3)`` with $T(Rs, Ra) = R\,T(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 8):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 3)
        return self.l2(self.a1(self.l1(x)))


class VNDynamics(nn.Module):
    r"""Equivariant **student** (dim=3), two VN nonlinearities; symmetry hard-wired.

    ``forward: (B, n_state, 3), (B, 1, 3) -> (B, n_state, 3)`` and $T_\theta(Rs, Ra)=R\,T_\theta(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 32):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, hidden)
        self.a2 = VNReLU(hidden)
        self.l3 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 3)
        return self.l3(self.a2(self.l2(self.a1(self.l1(x)))))


class MLPDynamics(nn.Module):
    r"""Non-equivariant **baseline**: a plain MLP on the flattened 3D state+action (~5x VN params)."""

    def __init__(self, n_state: int = N_STATE, hidden: int = 128):
        super().__init__()
        self.n_state = n_state
        in_dim, out_dim = n_state * DIM + DIM, n_state * DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        x = torch.cat([s.reshape(b, -1), a.reshape(b, -1)], dim=1)
        return self.net(x).reshape(b, self.n_state, DIM)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# SO(3) geometry -- single-rotation helpers mirror step13; batched variants added
# for per-sample augmentation. Kept self-contained (no e3nn/swm import).
# --------------------------------------------------------------------------- #
def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    r"""Batched skew-symmetric matrices $[\,\hat n\,]_\times$. ``(n, 3) -> (n, 3, 3)``."""
    n = v.shape[0]
    K = torch.zeros(n, 3, 3, dtype=v.dtype)
    K[:, 0, 1] = -v[:, 2]; K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2];  K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]; K[:, 2, 1] = v[:, 0]
    return K


def axis_angle_batch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    r"""Batched Rodrigues: $R=\mathbf I+\sin\theta\,[\hat n]_\times+(1-\cos\theta)[\hat n]_\times^2$.

    ``axis: (n, 3), angle: (n,) -> (n, 3, 3)``.
    """
    nrm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K = _skew_batch(axis / nrm)
    eye = torch.eye(3, dtype=axis.dtype).expand(axis.shape[0], 3, 3)
    s = torch.sin(angle).view(-1, 1, 1)
    c = (1.0 - torch.cos(angle)).view(-1, 1, 1)
    return eye + s * K + c * (K @ K)


def rand_so3(gen: torch.Generator) -> torch.Tensor:
    r"""A single random rotation (axis uniform, angle $\sim U[0,2\pi)$). ``-> (3, 3)`` (= step13)."""
    axis = torch.randn(3, generator=gen)
    angle = torch.full((1,), 2.0 * math.pi * torch.rand((), generator=gen).item())
    return axis_angle_batch(axis.unsqueeze(0), angle)[0]


def rot_z_batch(phi: torch.Tensor) -> torch.Tensor:
    r"""Per-sample rotation about $+z$ by ``phi`` (radians). ``(n,) -> (n, 3, 3)``."""
    z = torch.tensor([0.0, 0.0, 1.0]).expand(phi.shape[0], 3)
    return axis_angle_batch(z, phi)


def rand_so3_cap_batch(n: int, theta_max: float, gen: torch.Generator) -> torch.Tensor:
    r"""``n`` random rotations in the **geodesic ball** of radius ``theta_max`` (radians).

    Axis uniform on $S^2$ (normalised Gaussian), angle $\sim U[0,\theta_{\max}]$. At
    $\theta_{\max}=\pi$ this covers all of $\mathrm{SO}(3)$. ``-> (n, 3, 3)``.
    """
    axis = torch.randn(n, 3, generator=gen)
    angle = torch.rand(n, generator=gen) * theta_max
    return axis_angle_batch(axis, angle)


def rotate_ps(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Per-sample rotation. ``x: (n, c, 3), R: (n, 3, 3) -> (n, c, 3)``."""
    return torch.einsum("nij,ncj->nci", R, x)


def rotate_g(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Global rotation by a single ``R``. ``x: (..., 3), R: (3, 3) -> (..., 3)``."""
    return x @ R.transpose(-1, -2)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rel_mse(model: nn.Module, s, a, y) -> float:
    err = (model(s, a) - y).pow(2).mean().item()
    return err / (y.pow(2).mean().item() + 1e-12)


@torch.no_grad()
def equiv_residual(model: nn.Module, s, a, Rs) -> float:
    r"""Worst-case relative equivariance residual over a set of $\mathrm{SO}(3)$ probe rotations.

    $\Delta_{\mathrm{eq}} = \max_R \lVert f(Rs, Ra) - R\,f(s,a)\rVert_F / \lVert f(s,a)\rVert_F$.
    """
    base = model(s, a)
    den = base.pow(2).sum().item() + 1e-12
    worst = 0.0
    for R in Rs:
        lhs = model(rotate_g(s, R), rotate_g(a, R))
        rhs = rotate_g(base, R)
        worst = max(worst, ((lhs - rhs).pow(2).sum().item() / den) ** 0.5)
    return worst


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model: nn.Module, s, a, y, *, epochs=1200, lr=3e-3, wd=1e-4) -> nn.Module:
    r"""Full-batch AdamW on a FIXED dataset (VN and the no-aug MLP)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = (model(s, a) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


def train_augmented(
    model: nn.Module, s_base, a_base, target_fn, theta_max: float,
    *, epochs=1200, lr=3e-3, wd=1e-4, gen: torch.Generator,
) -> nn.Module:
    r"""Full-batch AdamW with on-the-fly $\mathrm{SO}(3)$ augmentation in the radius-``theta_max`` ball.

    Each epoch the base scenes are rotated by fresh per-sample rotations sampled from the geodesic ball
    (angle $\le\theta_{\max}$, random axis); the target is recomputed via the frozen equivariant teacher,
    so the augmented label is exact.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    model.train()
    for _ in range(epochs):
        R = rand_so3_cap_batch(n, theta_max, gen)  # (n, 3, 3)
        s_aug, a_aug = rotate_ps(s_base, R), rotate_ps(a_base, R)
        with torch.no_grad():
            y_aug = target_fn(s_aug, a_aug)
        opt.zero_grad()
        loss = (model(s_aug, a_aug) - y_aug).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


# --------------------------------------------------------------------------- #
# Evaluation: seen = z-wedge [0,90); OOD = random global SO(3) off the wedge
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_so3(model, target_fn, s_seen, a_seen, gen: torch.Generator, k_ood: int):
    r"""Return ``(seen_relMSE, ood_relMSE, ratio)``.

    ``seen`` is scored on the $z$-wedge scenes; ``ood`` averages relMSE over ``k_ood`` random global
    $\mathrm{SO}(3)$ rotations of those same scenes (so content is fixed, only orientation moves).
    """
    seen = rel_mse(model, s_seen, a_seen, target_fn(s_seen, a_seen))
    vals = []
    for _ in range(k_ood):
        R = rand_so3(gen)
        s_o, a_o = rotate_g(s_seen, R), rotate_g(a_seen, R)
        vals.append(rel_mse(model, s_o, a_o, target_fn(s_o, a_o)))
    ood = sum(vals) / len(vals)
    return seen, ood, ood / (seen + 1e-12)


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)


# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 72
    torch.manual_seed(0)

    if SMOKE:
        SEEDS, THETAS_DEG, EPOCHS, N_BASE, N_TEST, K_OOD = [0, 1], [90, 180], 250, 128, 400, 4
    else:
        SEEDS, THETAS_DEG, EPOCHS, N_BASE, N_TEST, K_OOD = [0, 1, 2, 3, 4], [90, 135, 180], 1200, 256, 2000, 8

    print(line)
    print(f"STEP 28 (3D)  fair augmentation baseline in SO(3): does aug buy what equivariance gives free?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    # --- the world: frozen, exactly SO(3)-equivariant VN teacher --------------------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, DIM), torch.randn(8000, 1, DIM)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()

    def target(s, a):
        with torch.no_grad():
            return teacher(s, a) / rms

    probe_gen = torch.Generator().manual_seed(11)
    PROBE_R = [rand_so3(probe_gen) for _ in range(5)]
    s_chk, a_chk = torch.randn(64, N_STATE, DIM), torch.randn(64, 1, DIM)
    teacher_eq = equiv_residual(lambda s, a: target(s, a), s_chk, a_chk, PROBE_R)
    assert teacher_eq < 1e-4, "teacher world is not SO(3)-equivariant"
    print(f"    world: frozen equivariant SO(3) teacher, residual {teacher_eq:.1e}  "
          f"(targets exactly symmetric -> augmentation labels exact)")

    # --- fixed anisotropic canonical layout (no rotational symmetry) ----------------------
    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, DIM) * 1.5
    mu_a = torch.randn(1, DIM) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, DIM, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, DIM, generator=gen)
        return s, a

    vn_p, mlp_p = n_params(VNDynamics()), n_params(MLPDynamics())
    print(f"    params: VN(equivariant)={vn_p}   MLP(baseline)={mlp_p}  ({mlp_p / vn_p:.1f}x VN)")
    print(f"    protocol: VN/MLP train z-wedge [0,90); aug fills the SO(3) ball of radius theta_max; "
          f"OOD=random SO(3); {len(SEEDS)} seeds")

    acc = {
        "VN": {"ratio": [], "eq": [], "seen": [], "ood": []},
        "MLP_noaug": {"ratio": [], "eq": [], "seen": [], "ood": []},
        "MLP_aug": {th: {"ratio": [], "eq": [], "seen": [], "ood": []} for th in THETAS_DEG},
    }

    for sd in SEEDS:
        data_gen = torch.Generator().manual_seed(2000 + sd)
        aug_gen = torch.Generator().manual_seed(3000 + sd)
        test_gen = torch.Generator().manual_seed(4000 + sd)

        # base canonical (unrotated) scenes; wedge models see these z-rotated into [0,90)
        s_c, a_c = aniso_inputs(N_BASE, data_gen)
        phi_w = torch.rand(N_BASE, generator=data_gen) * (math.pi / 2)  # z-wedge [0,90)
        Rz_w = rot_z_batch(phi_w)
        s_w, a_w = rotate_ps(s_c, Rz_w), rotate_ps(a_c, Rz_w)
        y_w = target(s_w, a_w)

        # held-out z-wedge test scenes (= "seen") + a probe set for the equivariance residual
        s_te0, a_te0 = aniso_inputs(N_TEST, test_gen)
        phi_te = torch.rand(N_TEST, generator=test_gen) * (math.pi / 2)
        Rz_te = rot_z_batch(phi_te)
        s_seen, a_seen = rotate_ps(s_te0, Rz_te), rotate_ps(a_te0, Rz_te)
        s_pr, a_pr = aniso_inputs(min(N_TEST, 1000), test_gen)

        torch.manual_seed(10 + sd)
        vn = train(VNDynamics(), s_w, a_w, y_w, epochs=EPOCHS)
        seen, ood, ratio = eval_so3(vn, target, s_seen, a_seen, test_gen, K_OOD)
        acc["VN"]["ratio"].append(ratio); acc["VN"]["seen"].append(seen)
        acc["VN"]["ood"].append(ood); acc["VN"]["eq"].append(equiv_residual(vn, s_pr, a_pr, PROBE_R))

        torch.manual_seed(20 + sd)
        mlp0 = train(MLPDynamics(), s_w, a_w, y_w, epochs=EPOCHS)
        seen, ood, ratio = eval_so3(mlp0, target, s_seen, a_seen, test_gen, K_OOD)
        acc["MLP_noaug"]["ratio"].append(ratio); acc["MLP_noaug"]["seen"].append(seen)
        acc["MLP_noaug"]["ood"].append(ood); acc["MLP_noaug"]["eq"].append(equiv_residual(mlp0, s_pr, a_pr, PROBE_R))

        for th in THETAS_DEG:
            torch.manual_seed(30 + sd)
            mlpa = train_augmented(MLPDynamics(), s_c, a_c, target, th * math.pi / 180.0,
                                   epochs=EPOCHS, gen=aug_gen)
            seen, ood, ratio = eval_so3(mlpa, target, s_seen, a_seen, test_gen, K_OOD)
            acc["MLP_aug"][th]["ratio"].append(ratio); acc["MLP_aug"][th]["seen"].append(seen)
            acc["MLP_aug"][th]["ood"].append(ood)
            acc["MLP_aug"][th]["eq"].append(equiv_residual(mlpa, s_pr, a_pr, PROBE_R))

    def summ(d):
        return {k: mean_std(d[k]) for k in ("ratio", "eq", "seen", "ood")}

    vn_s, mlp0_s = summ(acc["VN"]), summ(acc["MLP_noaug"])
    aug_s = {th: summ(acc["MLP_aug"][th]) for th in THETAS_DEG}

    # --- [1] task metric ------------------------------------------------------------------
    print()
    print(line)
    print("[1] TASK METRIC -- OOD/seen relMSE ratio (OOD = random global SO(3); 1.00 = flat = 举一反三)")
    print(line)
    print(f"    {'model':>22s} | {'seen relMSE':>13s} | {'OOD/seen ratio':>16s}")
    print("    " + "-" * 58)
    print(f"    {'VN (exact, no aug)':>22s} | {vn_s['seen'][0]:13.3e} | "
          f"x{vn_s['ratio'][0]:6.3f} +-{vn_s['ratio'][1]:.2f}")
    print(f"    {'MLP (no aug)':>22s} | {mlp0_s['seen'][0]:13.3e} | "
          f"x{mlp0_s['ratio'][0]:6.2f} +-{mlp0_s['ratio'][1]:.2f}   <- step13-style baseline")
    for th in THETAS_DEG:
        a = aug_s[th]
        tag = "   <- full SO(3)" if th == 180 else ""
        print(f"    {('MLP + aug ball<=' + str(th)):>22s} | {a['seen'][0]:13.3e} | "
              f"x{a['ratio'][0]:6.3f} +-{a['ratio'][1]:.2f}{tag}")
    print(f"    => coverage helps: ratio falls x{aug_s[THETAS_DEG[0]]['ratio'][0]:.2f} "
          f"(<= {THETAS_DEG[0]} deg) -> x{aug_s[THETAS_DEG[-1]]['ratio'][0]:.2f} (full SO(3)); "
          f"VN is x{vn_s['ratio'][0]:.2f} at zero coverage.")

    # --- [2] exactness --------------------------------------------------------------------
    print()
    print(line)
    print("[2] EXACTNESS -- residual equivariance Delta_eq = max_R ||f(Rx)-Rf(x)|| / ||f(x)|| over SO(3)")
    print(line)
    print(f"    {'model':>22s} | {'Delta_eq':>12s}")
    print("    " + "-" * 38)
    print(f"    {'VN (exact)':>22s} | {vn_s['eq'][0]:12.2e}   <- float floor, weight-independent")
    print(f"    {'MLP (no aug)':>22s} | {mlp0_s['eq'][0]:12.2e}")
    for th in THETAS_DEG:
        print(f"    {('MLP + aug ball<=' + str(th)):>22s} | {aug_s[th]['eq'][0]:12.2e}")
    aug_eq_full = aug_s[THETAS_DEG[-1]]["eq"][0]
    eq_factor = aug_eq_full / max(vn_s["eq"][0], 1e-12)
    print(f"    => even at FULL SO(3) coverage the augmented MLP is only APPROXIMATELY equivariant: "
          f"Delta_eq x{eq_factor:.0f} the VN floor.")

    # --- verdict --------------------------------------------------------------------------
    print()
    print(line)
    print("STEP 28 (3D) SUMMARY")
    print(line)
    ratio_drop = aug_s[THETAS_DEG[-1]]["ratio"][0] < aug_s[THETAS_DEG[0]]["ratio"][0]
    task_closed = aug_s[THETAS_DEG[-1]]["ratio"][0] < 1.5
    never_exact = aug_eq_full > 50.0 * max(vn_s["eq"][0], 1e-12)

    if task_closed:
        print(f"    [1] task: full-SO(3) augmentation DOES flatten the MLP "
              f"(x{aug_s[THETAS_DEG[-1]]['ratio'][0]:.2f}, near the VN's x{vn_s['ratio'][0]:.2f}).")
    else:
        print(f"    [1] task: even full-SO(3) augmentation does NOT reach VN-level flatness "
              f"(x{aug_s[THETAS_DEG[-1]]['ratio'][0]:.2f} vs VN x{vn_s['ratio'][0]:.2f}).")
    print(f"    [2] exactness: augmentation NEVER reaches exact SO(3) equivariance "
          f"(Delta_eq x{eq_factor:.0f} the VN floor, at full coverage).")
    print(f"    headline (matches the 2D arm): augmentation APPROXIMATES the symmetry; "
          f"the architecture IS the symmetry.")

    assert vn_s["ratio"][0] < 1.5, "VN must stay ~flat across SO(3)"
    assert vn_s["eq"][0] < 1e-4, "VN must stay exactly SO(3)-equivariant"
    assert mlp0_s["ratio"][0] > 3.0, "no-aug MLP must break across SO(3)"
    assert ratio_drop, "more SO(3) coverage must reduce the OOD ratio"
    assert never_exact, "augmentation must NOT reach exact SO(3) equivariance"
    print(f"\n    guards: vn-flat={vn_s['ratio'][0] < 1.5}  vn-exact={vn_s['eq'][0] < 1e-4}  "
          f"noaug-breaks={mlp0_s['ratio'][0] > 3.0}  coverage-helps={ratio_drop}  "
          f"aug-never-exact={never_exact}")
    print("    PASS: the 3D fair-augmentation baseline confirms the 2D conclusion.")

    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "group": "SO(3)",
        "config": dict(seeds=SEEDS, thetas_deg=THETAS_DEG, epochs=EPOCHS,
                       n_base=N_BASE, n_test=N_TEST, k_ood=K_OOD, n_state=N_STATE),
        "params": {"VN": vn_p, "MLP": mlp_p},
        "vn": vn_s,
        "mlp_noaug": mlp0_s,
        "mlp_aug": {str(th): aug_s[th] for th in THETAS_DEG},
        "derived": {"eq_factor_full_vs_vn": eq_factor,
                    "task_closed": bool(task_closed),
                    "coverage_helps": bool(ratio_drop),
                    "aug_never_exact": bool(never_exact)},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step28_fair_augmentation_3d_smoke.json" if SMOKE
                          else "step28_fair_augmentation_3d.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
