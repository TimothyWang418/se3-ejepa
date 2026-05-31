r"""Step 31 (3D arm): does one-step $\mathrm{SO}(3)$ equivariance buy multi-step *rollout* generalisation
for free?

The 2D arm (``experiments/step31_rollout_horizon.py``) established, in $\mathrm{SO}(2)$, that equivariance
is closed under composition: a Vector-Neuron rollout is across-group **flat** (OOD/seen ratio $\times1.00$)
and **exactly equivariant** ($\Delta_{\mathrm{eq}}^{(H)}$ at the float floor) at *every* horizon $H$, while
a free MLP that matches it one-step carries a large across-group gap at every horizon and its rollout
operator drifts **monotonically** further from equivariance as $H$ grows. This arm reruns the identical
story in the richer group $\mathrm{SO}(3)$ (3 rotational DoF), where the rollout is genuinely 3D.

The theorem (group-agnostic)
----------------------------
If the one-step rollout operator $\Phi_\theta(s)=s+v_\theta(s,a)$ is $\mathrm{SO}(3)$-equivariant --
$\Phi_\theta(Rs)=R\,\Phi_\theta(s)$ with $a$ carried as $Ra$ -- then by induction so is the $H$-fold
composition $\Phi_\theta^{(H)}$:

$$ \Phi_\theta^{(H)}(Rs)=\Phi_\theta^{(H-1)}\!\big(R\,\Phi_\theta(s)\big)=R\,\Phi_\theta^{(H)}(s). $$

A Vector-Neuron rollout therefore inherits exact across-group flatness and float-floor exactness at every
horizon; the non-equivariant MLP re-injects its extrapolation error every step.

Setup (exact $\mathrm{SO}(3)$ teacher, no symmetry break)
---------------------------------------------------------
The world is kept *exactly* $\mathrm{SO}(3)$-equivariant (the variable under study is the horizon). The
teacher is a velocity field $s_{t+1}=s_t+\tau\,\widehat{\mathrm{Dyn}}_0(s_t,a)$,
$\widehat{\mathrm{Dyn}}_0=\mathrm{Dyn}_0/\mathrm{rms}$, built from the frozen $\mathrm{SO}(3)$ VN world. Both
students learn the one-step velocity with rotation augmentation confined to the **seen ball** (geodesic
radius $\le\theta_{\max}$, random axis); we then roll out $H$ steps from initial conditions rotated into
the seen ball versus the uncovered OOD **complement shell** (angle $\in(\theta_{\max},180^\circ]$) -- genuine
across-group extrapolation. Two models only: hard VN, free MLP.

Three metrics over the horizon $H$ (identical to the 2D arm)
------------------------------------------------------------
  [1] **seen rollout fidelity** (final-state relMSE over the ball): accumulates with $H$ for both -- the
      honest baseline.
  [2] **OOD/seen rollout ratio**: VN $\times1.00$ flat at every $H$ (composition theorem); the MLP carries
      a large across-group gap at every $H$.
  [3] **composed exactness** $\Delta_{\mathrm{eq}}^{(H)}=\max_R\lVert \Phi^{(H)}(Rx)-R\,\Phi^{(H)}(x)\rVert/
      \lVert \Phi^{(H)}(x)\rVert$: VN at the float floor for every $H$; the MLP's residual **compounds**
      monotonically with the horizon.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step31_rollout_horizon_3d.py
Smoke (~90 s):
    STEP31_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step31_rollout_horizon_3d.py
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
SMOKE = bool(os.environ.get("STEP31_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- step28/29/30 lifted to dim=3 (the two extremes of the dial).
# --------------------------------------------------------------------------- #
class EquivariantWorld(nn.Module):
    r"""Frozen random *world*: an exactly $\mathrm{SO}(3)$-equivariant one-step map.

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
    r"""Equivariant **student** (dim=3, hard, $\sim\!3.5$k params); symmetry hard-wired and scale-free.

    Learns the one-step velocity $v_\theta(s,a)$; ``forward: (B, n_state, 3), (B, 1, 3) -> (B, n_state, 3)``
    with $v_\theta(Rs, Ra)=R\,v_\theta(s,a)$, so the rollout operator $\Phi_\theta(s)=s+v_\theta(s,a)$ is
    exactly equivariant at every step.
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
    r"""Non-equivariant **baseline** velocity on the flattened 3D state+action (no symmetry prior)."""

    def __init__(self, n_state: int = N_STATE, hidden: int = 256):
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
# SO(3) geometry -- mirrors step29/step30 3D (self-contained).
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
    r"""A single random rotation (axis uniform, angle $\sim U[0,2\pi)$). ``-> (3, 3)``."""
    axis = torch.randn(3, generator=gen)
    angle = torch.full((1,), 2.0 * math.pi * torch.rand((), generator=gen).item())
    return axis_angle_batch(axis.unsqueeze(0), angle)[0]


def rand_so3_cap_batch(n: int, theta_max: float, gen: torch.Generator) -> torch.Tensor:
    r"""``n`` random rotations in the **geodesic ball** of radius ``theta_max`` (axis uniform). ``-> (n,3,3)``."""
    axis = torch.randn(n, 3, generator=gen)
    angle = torch.rand(n, generator=gen) * theta_max
    return axis_angle_batch(axis, angle)


def rand_so3_shell_batch(n: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    r"""``n`` random rotations in the **complement shell**: angle $\sim U[\text{lo},\text{hi}]$, axis uniform."""
    axis = torch.randn(n, 3, generator=gen)
    angle = torch.rand(n, generator=gen) * (hi - lo) + lo
    return axis_angle_batch(axis, angle)


def rotate_ps(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Per-sample rotation. ``x: (n, c, 3), R: (n, 3, 3) -> (n, c, 3)``."""
    return torch.einsum("nij,ncj->nci", R, x)


def rotate_g(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Global rotation by a single ``R``. ``x: (..., 3), R: (3, 3) -> (..., 3)``."""
    return x @ R.transpose(-1, -2)


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)


def fresh(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- #
# Rollout + metrics (final-state relMSE over a horizon, from one forward pass)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_relmse_curve(model, s0, a, Hs, teacher_step):
    r"""Final-state relMSE at each horizon $h\in$ ``Hs`` from a single lock-step rollout.

    Teacher flow $s\!\to\!s+\tau\widehat{\mathrm{Dyn}}_0(s,a)$ (``teacher_step``); model flow
    $s\!\to\!s+v_\theta(s,a)$ (``model`` outputs the increment). Returns ``{h: relMSE_h}``.
    """
    Hmax = max(Hs)
    Hset = set(Hs)
    s_star = s0
    s_hat = s0
    out = {}
    for h in range(1, Hmax + 1):
        s_star = teacher_step(s_star, a)
        s_hat = s_hat + model(s_hat, a)
        if h in Hset:
            out[h] = ((s_hat - s_star).pow(2).mean().item()
                      / (s_star.pow(2).mean().item() + 1e-12))
    return out


@torch.no_grad()
def eval_rollout_seen_ood(model, s_te0, a_te0, Hs, cov, teacher_step, gen):
    r"""Rollout relMSE on the seen ball ($\le$ ``cov``) vs genuine OOD shell ($(\mathrm{cov},180^\circ]$).

    The initial condition is rotated once into each regime; the autonomous rollout (fixed $a$, carried as
    $Ra$) then stays in that frame. Returns ``(seen, ood, ratio)`` dicts keyed by horizon.
    """
    n = s_te0.shape[0]
    R_seen = rand_so3_cap_batch(n, cov, gen)
    s0s, a0s = rotate_ps(s_te0, R_seen), rotate_ps(a_te0, R_seen)
    seen = rollout_relmse_curve(model, s0s, a0s, Hs, teacher_step)
    R_ood = rand_so3_shell_batch(n, cov, math.pi, gen)
    s0o, a0o = rotate_ps(s_te0, R_ood), rotate_ps(a_te0, R_ood)
    ood = rollout_relmse_curve(model, s0o, a0o, Hs, teacher_step)
    ratio = {h: ood[h] / (seen[h] + 1e-12) for h in Hs}
    return seen, ood, ratio


@torch.no_grad()
def rollout_equiv_curve(model, s0, a, Hs, Rs):
    r"""Composed exactness $\Delta_{\mathrm{eq}}^{(H)}$ of the $H$-step *model* rollout operator over SO(3).

    $\max_R\lVert \Phi_\theta^{(H)}(Rs_0, Ra)-R\,\Phi_\theta^{(H)}(s_0,a)\rVert/\lVert
    \Phi_\theta^{(H)}(s_0,a)\rVert$ at each horizon $h\in$ ``Hs``.
    """
    Hmax = max(Hs)
    Hset = set(Hs)
    base = {}
    s = s0
    for h in range(1, Hmax + 1):
        s = s + model(s, a)
        if h in Hset:
            base[h] = s
    worst = {h: 0.0 for h in Hs}
    for R in Rs:
        sR, aR = rotate_g(s0, R), rotate_g(a, R)
        s = sR
        for h in range(1, Hmax + 1):
            s = s + model(s, aR)
            if h in Hset:
                rhs = rotate_g(base[h], R)
                den = base[h].pow(2).sum().item() + 1e-12
                worst[h] = max(worst[h], ((s - rhs).pow(2).sum().item() / den) ** 0.5)
    return worst


# --------------------------------------------------------------------------- #
# Training -- plain one-step MSE on the velocity target (no RPP/beta here)
# --------------------------------------------------------------------------- #
def train_one(
    model: nn.Module, s_base, a_base, target_fn, theta_max: float,
    *, steps: int, batch: int, lr=3e-3, wd=1e-6, gen: torch.Generator,
) -> nn.Module:
    r"""Minibatch AdamW on the one-step velocity with $\mathrm{SO}(3)$ ball augmentation ($\le\theta_{\max}$).

    ``target_fn`` is recomputed on the rotated inputs each step (genuine labels); the model output is the
    per-step increment.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        sb, ab = s_base[idx], a_base[idx]
        R = rand_so3_cap_batch(batch, theta_max, gen)
        s_aug, a_aug = rotate_ps(sb, R), rotate_ps(ab, R)
        with torch.no_grad():
            y_aug = target_fn(s_aug, a_aug)
        opt.zero_grad()
        loss = (model(s_aug, a_aug) - y_aug).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 78
    torch.manual_seed(0)

    COVERAGE_DEG = 90   # partial coverage ball (same as Step 29/30 3D); (90,180] shell is OOD
    TAU = 0.05          # per-step size: keeps the seen rollout faithful across the whole horizon
    if SMOKE:
        SEEDS, HVALS, N_DATA, STEPS, BATCH, N_TEST = ([0, 1], [1, 4, 16], 512, 400, 128, 400)
    else:
        SEEDS, HVALS, N_DATA, STEPS, BATCH, N_TEST = (
            [0, 1, 2, 3, 4], [1, 2, 4, 8, 16], 1024, 2000, 256, 2000)

    print(line)
    print(f"STEP 31 (3D)  multi-step rollout: across-group ratio over horizon H  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    teacher = exact SO(3) VN velocity field s' = s + tau * Dyn0(s,a)/rms (no symmetry break)")
    print(f"    train one-step on seen ball <= {COVERAGE_DEG} deg; roll out H steps; measure OOD shell/seen")
    print(f"    models: hard VN, free MLP; {len(SEEDS)} seeds; tau={TAU}; horizons H in {HVALS}")

    # --- frozen, exactly SO(3)-equivariant teacher world ---------------------------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, DIM), torch.randn(8000, 1, DIM)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()

    def teacher_step(s, a):
        r"""One step of the equivariant teacher flow $\Phi^\ast(s)=s+\tau\widehat{\mathrm{Dyn}}_0(s,a)$."""
        with torch.no_grad():
            return s + TAU * (teacher(s, a) / rms)

    def target_v(s, a):
        r"""One-step velocity label $v^\ast(s,a)=\tau\widehat{\mathrm{Dyn}}_0(s,a)$ (the training target)."""
        with torch.no_grad():
            return TAU * (teacher(s, a) / rms)

    probe_gen = torch.Generator().manual_seed(11)
    PROBE_R = [rand_so3(probe_gen) for _ in range(5)]
    # init-time equivariance unit test: the teacher one-step flow must be exactly SO(3)-equivariant
    s_chk, a_chk = torch.randn(64, N_STATE, DIM), torch.randn(64, 1, DIM)
    teacher_eq = rollout_equiv_curve(lambda s, a: target_v(s, a), s_chk, a_chk, [1], PROBE_R)[1]
    assert teacher_eq < 1e-4, "teacher one-step flow is not SO(3)-equivariant"

    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, DIM) * 1.5
    mu_a = torch.randn(1, DIM) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, DIM, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, DIM, generator=gen)
        return s, a

    vn_p, mlp_p = n_params(VNDynamics()), n_params(MLPDynamics())
    print(f"    params: VN={vn_p}  MLP={mlp_p}")
    print(f"    budget: {STEPS} AdamW steps, batch {BATCH}, N_data={N_DATA}\n")

    cov = COVERAGE_DEG * math.pi / 180.0
    MODELS = ["VN", "MLP"]
    acc = {m: {k: {h: [] for h in HVALS} for k in ("seen", "ood", "ratio", "eq")} for m in MODELS}

    for sd in SEEDS:
        base_gen = fresh(2000 + sd)
        s_te0, a_te0 = aniso_inputs(N_TEST, base_gen)
        s_eq, a_eq = aniso_inputs(min(N_TEST, 256), base_gen)  # exactness probe ICs
        s_c, a_c = aniso_inputs(N_DATA, base_gen)              # one-step training scenes

        def run(model, name):
            m = train_one(model, s_c, a_c, target_v, cov,
                          steps=STEPS, batch=BATCH, gen=fresh(3000 + sd))
            seen, ood, ratio = eval_rollout_seen_ood(m, s_te0, a_te0, HVALS, cov, teacher_step,
                                                     fresh(4000 + sd))
            eq = rollout_equiv_curve(m, s_eq, a_eq, HVALS, PROBE_R)
            for h in HVALS:
                acc[name]["seen"][h].append(seen[h])
                acc[name]["ood"][h].append(ood[h])
                acc[name]["ratio"][h].append(ratio[h])
                acc[name]["eq"][h].append(eq[h])

        torch.manual_seed(100 + sd)
        run(VNDynamics(), "VN")
        torch.manual_seed(200 + sd)
        run(MLPDynamics(), "MLP")

    def cell(name, key, h):
        return mean_std(acc[name][key][h])

    hcols = "".join(f"     H={h:<3d}" for h in HVALS)

    def print_table(title, key, fmt):
        print(line)
        print(title)
        print(line)
        print(f"    {'model':>6s} |{hcols}")
        print("    " + "-" * (8 + len(hcols)))
        for m in MODELS:
            row = f"    {m:>6s} |"
            for h in HVALS:
                row += fmt(cell(m, key, h))
            print(row)

    # --- [1] seen rollout fidelity -------------------------------------------------------
    print_table("[1] SEEN ROLLOUT -- final-state relMSE on the seen ball (accumulates with H; lower better)",
                "seen", lambda v: f"  {v[0]:9.2e}")
    vn_seen_H1 = cell("VN", "seen", HVALS[0])[0]
    vn_seen_Hm = cell("VN", "seen", HVALS[-1])[0]
    mlp_seen_H1 = cell("MLP", "seen", HVALS[0])[0]
    mlp_seen_Hm = cell("MLP", "seen", HVALS[-1])[0]
    print(f"    => rollout error accumulates for BOTH: VN seen {vn_seen_H1:.2e}(H={HVALS[0]}) -> "
          f"{vn_seen_Hm:.2e}(H={HVALS[-1]}); MLP {mlp_seen_H1:.2e} -> {mlp_seen_Hm:.2e}.")

    # --- [2] OOD / seen rollout ratio (headline) -----------------------------------------
    print()
    print_table("[2] GENERALISATION -- genuine across-group OOD shell rollout relMSE",
                "ood", lambda v: f"  {v[0]:9.2e}")
    print()
    print_table("    ... OOD/seen rollout ratio (1.00 = flat across orientations at that horizon)",
                "ratio", lambda v: f"  x{v[0]:7.2f}")
    vn_ratio_H1 = cell("VN", "ratio", HVALS[0])[0]
    vn_ratio_Hm = cell("VN", "ratio", HVALS[-1])[0]
    mlp_ratio_H1 = cell("MLP", "ratio", HVALS[0])[0]
    mlp_ratio_Hm = cell("MLP", "ratio", HVALS[-1])[0]
    vn_ratio_max = max(cell("VN", "ratio", h)[0] for h in HVALS)
    mlp_ratio_min = min(cell("MLP", "ratio", h)[0] for h in HVALS)
    mlp_ratio_max = max(cell("MLP", "ratio", h)[0] for h in HVALS)
    print(f"    => VN ratio flat across H (<= x{vn_ratio_max:.2f}): the composition theorem holds -- one-step")
    print(f"       equivariance propagates through the rollout. The MLP carries a large across-group gap at")
    print(f"       EVERY horizon (x{mlp_ratio_min:.1f}-x{mlp_ratio_max:.1f}); it peaks early then compresses as the OOD rollout")
    print(f"       saturates (relMSE ceiling) -- the *monotone* compounding signal is the exactness [3].")

    # --- [3] composed exactness ----------------------------------------------------------
    print()
    print_table("[3] EXACTNESS -- composed Delta_eq^(H) = max_R ||Phi^H(Rx)-R Phi^H(x)|| / ||Phi^H(x)|| over SO(3)",
                "eq", lambda v: f"  {v[0]:9.2e}")
    vn_eq_max = max(cell("VN", "eq", h)[0] for h in HVALS)
    mlp_eq_min = min(cell("MLP", "eq", h)[0] for h in HVALS)
    mlp_eq_H1 = cell("MLP", "eq", HVALS[0])[0]
    mlp_eq_Hm = cell("MLP", "eq", HVALS[-1])[0]
    print(f"    => VN composed exactness at the float floor for EVERY H (<= {vn_eq_max:.1e}): the rollout")
    print(f"       operator is structurally equivariant. MLP residual grows {mlp_eq_H1:.2e}(H={HVALS[0]}) -> "
          f"{mlp_eq_Hm:.2e}(H={HVALS[-1]}).")

    # --- verdict -------------------------------------------------------------------------
    print()
    print(line)
    print("STEP 31 (3D) SUMMARY")
    print(line)
    vn_rollout_flat = vn_ratio_max < 1.30
    vn_rollout_exact = vn_eq_max < 1e-3
    mlp_gap_persists = mlp_ratio_min > 3.0 * max(vn_ratio_max, 1e-12)
    mlp_exactness_compounds = mlp_eq_min > 50.0 * max(vn_eq_max, 1e-12) and mlp_eq_Hm > 1.5 * mlp_eq_H1
    rollout_accumulates = max(vn_seen_Hm / max(vn_seen_H1, 1e-12),
                              mlp_seen_Hm / max(mlp_seen_H1, 1e-12)) > 1.5

    print(f"    [1] rollout error accumulates for everyone (seen relMSE grows with H) -- the honest baseline:")
    print(f"        rollout is hard regardless of the prior.")
    print(f"    [2] HEADLINE: in SO(3) too the VN rollout is across-group FLAT at every horizon (ratio <= "
          f"x{vn_ratio_max:.2f}) --")
    print(f"        one-step equivariance composes, so multi-step across-group generalisation is FREE. The free")
    print(f"        MLP carries a large gap at EVERY horizon (>= x{mlp_ratio_min:.1f}).")
    print(f"    [3] the VN rollout operator stays exactly equivariant at every H ({vn_eq_max:.1e}); the MLP's")
    print(f"        composed residual is >= x{mlp_eq_min / max(vn_eq_max, 1e-12):.0f} the floor and COMPOUNDS monotonically with the")
    print(f"        horizon ({mlp_eq_H1:.1e} at H={HVALS[0]} -> {mlp_eq_Hm:.1e} at H={HVALS[-1]}) -- each step re-injects the break.")
    print(f"    headline (matches the 2D arm): equivariance is closed under composition, so the one-step prior")
    print(f"    pays at the rollout horizon the world model is actually used at -- a multi-step guarantee.")

    assert vn_rollout_flat, "VN OOD/seen rollout ratio must stay flat across all horizons"
    assert vn_rollout_exact, "VN composed rollout must stay exactly SO(3)-equivariant at all horizons"
    assert mlp_gap_persists, "the free MLP must carry an across-group gap at every horizon"
    assert mlp_exactness_compounds, "the free MLP rollout must be non-equivariant and compound with the horizon"
    assert rollout_accumulates, "rollout error must visibly accumulate with the horizon (honest baseline)"
    print(f"\n    guards: vn-rollout-flat={vn_rollout_flat}  vn-rollout-exact={vn_rollout_exact}  "
          f"mlp-gap-persists={mlp_gap_persists}")
    print(f"            mlp-exactness-compounds={mlp_exactness_compounds}  rollout-accumulates={rollout_accumulates}")
    print("    PASS: in SO(3) too, one-step equivariance composes into a multi-step guarantee; the MLP gap persists & compounds.")

    # --- dump JSON -----------------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "group": "SO(3)",
        "config": dict(seeds=SEEDS, hvals=HVALS, tau=TAU, n_data=N_DATA, steps=STEPS,
                       batch=BATCH, coverage_deg=COVERAGE_DEG, n_test=N_TEST, n_state=N_STATE),
        "params": {"VN": vn_p, "MLP": mlp_p},
        "models": {m: {k: {str(h): cell(m, k, h) for h in HVALS}
                       for k in ("seen", "ood", "ratio", "eq")} for m in MODELS},
        "derived": {"vn_ratio_max": vn_ratio_max, "vn_eq_max": vn_eq_max,
                    "vn_ratio_H1": vn_ratio_H1, "vn_ratio_Hm": vn_ratio_Hm,
                    "mlp_ratio_H1": mlp_ratio_H1, "mlp_ratio_Hm": mlp_ratio_Hm,
                    "mlp_ratio_min": mlp_ratio_min, "mlp_ratio_max": mlp_ratio_max,
                    "mlp_eq_min": mlp_eq_min, "mlp_eq_H1": mlp_eq_H1, "mlp_eq_Hm": mlp_eq_Hm,
                    "vn_seen_H1": vn_seen_H1, "vn_seen_Hm": vn_seen_Hm,
                    "mlp_seen_H1": mlp_seen_H1, "mlp_seen_Hm": mlp_seen_Hm,
                    "vn_rollout_flat": bool(vn_rollout_flat), "vn_rollout_exact": bool(vn_rollout_exact),
                    "mlp_gap_persists": bool(mlp_gap_persists),
                    "mlp_exactness_compounds": bool(mlp_exactness_compounds),
                    "rollout_accumulates": bool(rollout_accumulates)},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step31_rollout_horizon_3d_smoke.json" if SMOKE else "step31_rollout_horizon_3d.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")

    # --- figure (guarded) ----------------------------------------------------------------
    if not SMOKE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            colors = {"VN": "C2", "MLP": "C3"}
            fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.3))

            for m in MODELS:
                axA.plot(HVALS, [cell(m, "seen", h)[0] for h in HVALS], "o-", label=m, color=colors[m])
            axA.set_yscale("log"); axA.set_xscale("log", base=2)
            axA.set_xlabel("rollout horizon H"); axA.set_ylabel("seen rollout relMSE")
            axA.set_title("[1] rollout error accumulates (both)")
            axA.legend(fontsize=9); axA.grid(alpha=0.3, which="both")

            for m in MODELS:
                axB.plot(HVALS, [cell(m, "ratio", h)[0] for h in HVALS], "o-", label=m, color=colors[m])
            axB.axhline(1.0, color="C2", ls=":", alpha=0.6)
            axB.set_xscale("log", base=2)
            axB.set_xlabel("rollout horizon H"); axB.set_ylabel("OOD / seen rollout ratio")
            axB.set_title("[2] VN flat at x1; MLP gap at every H (SO(3))")
            axB.legend(fontsize=9); axB.grid(alpha=0.3, which="both")

            for m in MODELS:
                axC.plot(HVALS, [cell(m, "eq", h)[0] for h in HVALS], "o-", label=m, color=colors[m])
            axC.set_yscale("log"); axC.set_xscale("log", base=2)
            axC.set_xlabel("rollout horizon H"); axC.set_ylabel("composed  Delta_eq^(H)")
            axC.set_title("[3] only VN holds the floor over H (SO(3))")
            axC.legend(fontsize=9); axC.grid(alpha=0.3, which="both")

            fig.tight_layout()
            fig_path = fig_dir / "step31_rollout_horizon_3d.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"    wrote {fig_path.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    (figure skipped: {e})")

    sys.exit(0)


if __name__ == "__main__":
    main()
