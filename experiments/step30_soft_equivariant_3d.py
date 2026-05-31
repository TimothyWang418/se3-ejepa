r"""Step 30 (3D arm): the soft-equivariant model (RPP) under a controlled symmetry break, in
$\mathrm{SO}(3)$.

The 2D arm (``experiments/step30_soft_equivariant.py``) established, in $\mathrm{SO}(2)$, that the
Residual Pathway Prior $f_\beta=f_{\mathrm{VN}}+f_{\mathrm{free}}$ is a continuous **dial** between the
hard prior and the free model: relaxing the prior buys the capacity to absorb a broken symmetry, but
every step costs across-group generalisation **and** forfeits the float-floor exactness -- only the
hard architecture holds the exact corner. This arm reruns the identical story in the richer group
$\mathrm{SO}(3)$ (3 rotational DoF), the Bitter-Lesson's best case for "more capacity will win."

The controlled break (= Step 16, the faithful 3D form)
------------------------------------------------------
The teacher is the exactly-$\mathrm{SO}(3)$ Vector-Neuron world $\mathrm{Dyn}_0$ plus a fixed lab-$z$
anisotropy of strength $g$:

$$ \mathrm{Dyn}_g(s,a)_c \;=\; \mathrm{Dyn}_0(s,a)_c \;-\; g\,(s_c\!\cdot e_z)\,e_z, \qquad e_z=(0,0,1). $$

At $g=0$ this is exactly equivariant; the term references the fixed lab $z$-axis, so it is **not**
$\mathrm{SO}(3)$-equivariant ($g$ is a clean knob on symmetry-break energy, measured by
``noneq_fraction``). A subtlety unique to 3D: this break is *invariant* under rotations **about** $z$,
so a $z$-wedge would let even the hard VN fit it. We therefore train and evaluate "seen" over the **full
coverage ball** (rotations of angle $\le\theta_{\max}$ about a *random* axis): over random axes the break
genuinely violates equivariance, so the hard VN cannot fit it -- exactly the 2D situation. The OOD set
is the uncovered **complement shell** (angle $\in(\theta_{\max},180^\circ]$), genuinely re-sampled from
the *same* broken teacher (not rotated targets).

Three metrics, one honest tradeoff (identical to the 2D arm)
------------------------------------------------------------
For five models -- hard VN, RPP at three softness levels, free MLP -- swept over the break $g$:

  [1] **capacity** (seen relMSE over the ball): the hard VN can only fit the equivariant projection of
      $\mathrm{Dyn}_g$, so its seen error rises with $g$; the residual pathway recovers it.
  [2] **generalisation** (genuine OOD shell relMSE / OOD-seen ratio): the free MLP pays the
      rotation-extrapolation penalty; the hard VN generalises the symmetry for free; RPP interpolates.
  [3] **exactness** ($\Delta_{\mathrm{eq}}=\max_R\lVert f(Rx)-Rf(x)\rVert/\lVert f(x)\rVert$): the hard VN
      sits at the float floor for **every** $g$; the soft model leaves it the instant the residual is
      active -- even at $g=0$, where the symmetry is intact.

Headline: in $\mathrm{SO}(3)$ too, the soft architecture is a continuous dial, not a free lunch -- it buys
capacity by spending across-group reach and float-floor exactness. Only the hard architecture holds the
exact corner; at $g=0$ it dominates (same fit, exact, free).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step30_soft_equivariant_3d.py
Smoke (~90 s):
    STEP30_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step30_soft_equivariant_3d.py
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
SMOKE = bool(os.environ.get("STEP30_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- step28/step29 lifted to dim=3; RPP is the soft middle.
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
    r"""Non-equivariant **baseline** on the flattened 3D state+action (no symmetry prior)."""

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


class ResidualPathwayPrior(nn.Module):
    r"""Soft-equivariant model (RPP; Finzi et al. 2021): $f_\beta = f_{\mathrm{VN}} + f_{\mathrm{free}}$.

    The exact pathway ``self.vn`` carries the $\mathrm{SO}(3)$ structure; the residual pathway
    ``self.free`` is an unconstrained MLP. Softness is the residual *output-energy* penalty $\beta$
    (see :func:`train_one`): $\beta\to\infty$ recovers the hard VN, $\beta\to0$ recovers the free MLP.
    """

    def __init__(self, n_state: int = N_STATE, vn_hidden: int = 32, mlp_hidden: int = 256):
        super().__init__()
        self.vn = VNDynamics(n_state, hidden=vn_hidden)
        self.free = MLPDynamics(n_state, hidden=mlp_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.vn(s, a) + self.free(s, a)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# SO(3) geometry -- mirrors step29_scaling_sweep_3d.py (self-contained).
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


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rel_mse(model: nn.Module, s, a, y) -> float:
    err = (model(s, a) - y).pow(2).mean().item()
    return err / (y.pow(2).mean().item() + 1e-12)


@torch.no_grad()
def equiv_residual(model: nn.Module, s, a, Rs) -> float:
    r"""Worst-case $\Delta_{\mathrm{eq}} = \max_R \lVert f(Rs, Ra) - R\,f(s,a)\rVert_F / \lVert f(s,a)\rVert_F$."""
    base = model(s, a)
    den = base.pow(2).sum().item() + 1e-12
    worst = 0.0
    for R in Rs:
        lhs = model(rotate_g(s, R), rotate_g(a, R))
        rhs = rotate_g(base, R)
        worst = max(worst, ((lhs - rhs).pow(2).sum().item() / den) ** 0.5)
    return worst


@torch.no_grad()
def free_fraction(model: ResidualPathwayPrior, s, a) -> float:
    r"""Residual-pathway share $\rho=\mathbb{E}\lVert f_{\mathrm{free}}\rVert/\mathbb{E}\lVert f_\beta\rVert$."""
    total = model(s, a).norm().item()
    free = model.free(s, a).norm().item()
    return free / (total + 1e-12)


@torch.no_grad()
def noneq_fraction(target_g, s, a, Rs) -> float:
    r"""Model-independent symmetry-break energy of the teacher (the sweep's x-axis).

    $\dfrac{\mathbb{E}_R\lVert \mathrm{Dyn}_g(Rs,Ra)-R\,\mathrm{Dyn}_g(s,a)\rVert^2}
            {\mathbb{E}_R\lVert \mathrm{Dyn}_g(s,a)\rVert^2}$ -- 0 at $g=0$, growing with $g$.
    """
    base = target_g(s, a)
    num = 0.0
    den = 0.0
    for R in Rs:
        lhs = target_g(rotate_g(s, R), rotate_g(a, R))
        rhs = rotate_g(base, R)
        num += (lhs - rhs).pow(2).sum().item()
        den += base.pow(2).sum().item()
    return num / max(den, 1e-12)


# --------------------------------------------------------------------------- #
# Training -- one routine for all three families; ``beta`` activates the RPP prior.
# --------------------------------------------------------------------------- #
def train_one(
    model: nn.Module, s_base, a_base, target_fn, theta_max: float,
    *, beta: float, steps: int, batch: int, lr=3e-3, wd=1e-6, gen: torch.Generator,
) -> nn.Module:
    r"""Minibatch AdamW with on-the-fly $\mathrm{SO}(3)$ ball augmentation on the (broken) teacher.

    For a :class:`ResidualPathwayPrior` with ``beta`` $>0$ the loss adds $\beta\,\mathbb{E}\lVert
    f_{\mathrm{free}}\rVert^2$; for VN / MLP it is plain MSE. ``target_fn`` is recomputed on the rotated
    inputs each step (genuine labels, never rotated targets).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    is_rpp = isinstance(model, ResidualPathwayPrior)
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
        if is_rpp and beta > 0.0:
            loss = loss + beta * model.free(s_aug, a_aug).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


# --------------------------------------------------------------------------- #
# Evaluation: seen = coverage ball (<=cov); OOD = complement shell (cov,180] = extrapolation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_seen_ood(model, target_fn, s_te0, a_te0, cov: float, gen: torch.Generator):
    r"""Return ``(seen_relMSE, ood_relMSE, ood/seen)``; OOD = genuine re-sample of the broken teacher.

    ``seen`` rotates the canonical test scenes by the **full coverage ball** (angle $\le$ ``cov``, random
    axis) -- the training distribution, over which the lab-$z$ break is genuinely non-equivariant. ``ood``
    rotates them by the complement **shell** (angle $\in(\text{cov},180^\circ]$) -- pure extrapolation.
    """
    n = s_te0.shape[0]
    R_seen = rand_so3_cap_batch(n, cov, gen)
    s_s, a_s = rotate_ps(s_te0, R_seen), rotate_ps(a_te0, R_seen)
    seen = rel_mse(model, s_s, a_s, target_fn(s_s, a_s))
    R_ood = rand_so3_shell_batch(n, cov, math.pi, gen)
    s_o, a_o = rotate_ps(s_te0, R_ood), rotate_ps(a_te0, R_ood)
    ood = rel_mse(model, s_o, a_o, target_fn(s_o, a_o))
    return seen, ood, ood / (seen + 1e-12)


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)


def fresh(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 78
    torch.manual_seed(0)

    COVERAGE_DEG = 90  # partial coverage: a radius-90 ball; (90,180] shell is extrapolation
    if SMOKE:
        SEEDS, GVALS, BETAS, N_DATA, STEPS, BATCH, N_TEST = (
            [0, 1], [0.0, 0.6], [1.0, 1e-3], 512, 400, 128, 400)
    else:
        SEEDS, GVALS, BETAS, N_DATA, STEPS, BATCH, N_TEST = (
            [0, 1, 2, 3, 4], [0.0, 0.2, 0.4, 0.8], [1.0, 1e-2, 1e-4], 1024, 2000, 256, 2000)
    BETA_LABELS = {b: lab for b, lab in zip(BETAS, ("hi", "mid", "lo"))}

    print(line)
    print(f"STEP 30 (3D)  soft-equivariant model (RPP) under a controlled SO(3) symmetry break  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    teacher = exact SO(3) VN world + g * fixed lab-z anisotropy; sweep break g x softness beta")
    print(f"    models: hard VN, RPP(beta in {BETAS}), free MLP; {len(SEEDS)} seeds; coverage ball<={COVERAGE_DEG} deg")

    # --- frozen, exactly SO(3)-equivariant teacher world ---------------------------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, DIM), torch.randn(8000, 1, DIM)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()
    e_lab = torch.tensor([0.0, 0.0, 1.0])  # fixed lab axis for the break

    def make_target_g(g: float):
        r"""Broken teacher $\mathrm{Dyn}_g(s,a)_c = \mathrm{Dyn}_0(s,a)_c - g\,(s_c\!\cdot e_z)\,e_z$ (normalised)."""
        def target_g(s, a):
            with torch.no_grad():
                eqv = teacher(s, a) / rms
                proj = (s * e_lab).sum(dim=-1, keepdim=True) * e_lab  # (B,n_state,3): (s_c.ez) ez
                return eqv - g * proj
        return target_g

    probe_gen = torch.Generator().manual_seed(11)
    PROBE_R = [rand_so3(probe_gen) for _ in range(5)]
    # init-time equivariance unit test: the g=0 teacher must be exactly SO(3)-equivariant
    s_chk, a_chk = torch.randn(64, N_STATE, DIM), torch.randn(64, 1, DIM)
    teacher_eq = equiv_residual(lambda s, a: make_target_g(0.0)(s, a), s_chk, a_chk, PROBE_R)
    assert teacher_eq < 1e-4, "teacher world is not SO(3)-equivariant at g=0"

    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, DIM) * 1.5
    mu_a = torch.randn(1, DIM) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, DIM, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, DIM, generator=gen)
        return s, a

    vn_p, mlp_p, rpp_p = n_params(VNDynamics()), n_params(MLPDynamics()), n_params(ResidualPathwayPrior())
    print(f"    params: VN={vn_p}  MLP={mlp_p}  RPP={rpp_p} (VN+MLP pathways)")
    print(f"    budget: {STEPS} AdamW steps, batch {BATCH}, N_data={N_DATA}\n")

    cov = COVERAGE_DEG * math.pi / 180.0
    MODELS = ["VN"] + [f"RPP-{BETA_LABELS[b]}" for b in BETAS] + ["MLP"]
    acc = {m: {g: {"seen": [], "ood": [], "ratio": [], "eq": [], "rho": []} for g in GVALS} for m in MODELS}
    nf = {g: [] for g in GVALS}

    for sd in SEEDS:
        base_gen = fresh(2000 + sd)
        s_te0, a_te0 = aniso_inputs(N_TEST, base_gen)
        s_pr, a_pr = aniso_inputs(min(N_TEST, 1000), base_gen)
        s_c, a_c = aniso_inputs(N_DATA, base_gen)  # base scenes, identical across g

        for g in GVALS:
            target_g = make_target_g(g)
            nf[g].append(noneq_fraction(target_g, s_pr, a_pr, PROBE_R))

            def run(model, beta, name):
                m = train_one(model, s_c, a_c, target_g, cov, beta=beta,
                              steps=STEPS, batch=BATCH, gen=fresh(3000 + sd))
                seen, ood, ratio = eval_seen_ood(m, target_g, s_te0, a_te0, cov, fresh(4000 + sd))
                eq = equiv_residual(m, s_pr, a_pr, PROBE_R)
                rho = free_fraction(m, s_pr, a_pr) if isinstance(m, ResidualPathwayPrior) else float("nan")
                d = acc[name][g]
                d["seen"].append(seen); d["ood"].append(ood); d["ratio"].append(ratio)
                d["eq"].append(eq); d["rho"].append(rho)

            torch.manual_seed(100 + sd)
            run(VNDynamics(), 0.0, "VN")
            for b in BETAS:
                torch.manual_seed(300 + sd)
                run(ResidualPathwayPrior(), b, f"RPP-{BETA_LABELS[b]}")
            torch.manual_seed(200 + sd)
            run(MLPDynamics(), 0.0, "MLP")

    def cell(name, g, key):
        return mean_std(acc[name][g][key])

    nf_mean = {g: mean_std(nf[g])[0] for g in GVALS}
    gcols = "".join(f"  g={g:<4.2f}(nf={nf_mean[g]:4.2f})" for g in GVALS)

    def print_table(title, key, fmt):
        print(line)
        print(title)
        print(line)
        print(f"    {'model':>8s} |{gcols}")
        print("    " + "-" * (10 + len(gcols)))
        for m in MODELS:
            row = f"    {m:>8s} |"
            for g in GVALS:
                row += fmt(cell(m, g, key))
            print(row)

    # --- [1] capacity --------------------------------------------------------------------
    print_table("[1] CAPACITY -- seen relMSE (in-ball fit of the broken world; lower = better)",
                "seen", lambda v: f"  {v[0]:8.4f}      ")
    vn_seen0 = cell("VN", GVALS[0], "seen")[0]
    vn_seenG = cell("VN", GVALS[-1], "seen")[0]
    soft_seenG = min(cell("MLP", GVALS[-1], "seen")[0], cell(f"RPP-{BETA_LABELS[BETAS[-1]]}", GVALS[-1], "seen")[0])
    print(f"    => VN seen {vn_seen0:.4f} (g=0) -> {vn_seenG:.4f} (g={GVALS[-1]}): the hard prior cannot fit the")
    print(f"       fixed-axis break (irreducible floor). Softest model at g={GVALS[-1]}: {soft_seenG:.4f} -- capacity recovered.")

    # --- [2] generalisation --------------------------------------------------------------
    print()
    print_table("[2] GENERALISATION -- genuine across-group OOD shell relMSE",
                "ood", lambda v: f"  {v[0]:8.4f}      ")
    print()
    print_table("    ... OOD/seen ratio (1.00 = flat across orientations)",
                "ratio", lambda v: f"  x{v[0]:7.2f}     ")
    vn_ratio = max(cell("VN", g, "ratio")[0] for g in GVALS)
    mlp_ratio = max(cell("MLP", g, "ratio")[0] for g in GVALS)
    print(f"    => across all g: VN ratio <= x{vn_ratio:.2f} (generalises the symmetry); "
          f"MLP up to x{mlp_ratio:.1f} (extrapolation). RPP interpolates.")

    # --- [3] exactness -------------------------------------------------------------------
    print()
    print_table("[3] EXACTNESS -- residual equivariance Delta_eq = max_R ||f(Rx)-Rf(x)|| / ||f(x)|| over SO(3)",
                "eq", lambda v: f"  {v[0]:9.2e}    ")
    vn_floor = max(cell("VN", g, "eq")[0] for g in GVALS)
    rpp_eq0 = cell(f"RPP-{BETA_LABELS[BETAS[-1]]}", GVALS[0], "eq")[0]  # softest, g=0
    print(f"    => VN Delta_eq <= {vn_floor:.1e} for EVERY g (structural, break-independent). Softest RPP at g=0:")
    print(f"       {rpp_eq0:.2e} = x{rpp_eq0 / max(vn_floor, 1e-12):.0f} the floor -- the residual pathway forfeits")
    print(f"       exactness the instant it is active, even where the symmetry is intact (g=0).")

    # --- [dial] free-fraction ------------------------------------------------------------
    print()
    print(line)
    print("[dial] FREE-FRACTION rho = ||f_free|| / ||f_beta||  (RPP only; the softness readout)")
    print(line)
    print(f"    {'model':>8s} |{gcols}")
    print("    " + "-" * (10 + len(gcols)))
    for b in BETAS:
        m = f"RPP-{BETA_LABELS[b]}"
        row = f"    {m:>8s} |"
        for g in GVALS:
            row += f"  {cell(m, g, 'rho')[0]:8.3f}      "
        print(row)
    rho_hi = cell(f"RPP-{BETA_LABELS[BETAS[0]]}", GVALS[-1], "rho")[0]
    rho_lo = cell(f"RPP-{BETA_LABELS[BETAS[-1]]}", GVALS[-1], "rho")[0]
    print(f"    => at g={GVALS[-1]}: beta={BETAS[0]:g} -> rho={rho_hi:.3f} (near-VN); "
          f"beta={BETAS[-1]:g} -> rho={rho_lo:.3f} (near-free). The prior dials the pathway.")

    # --- verdict -------------------------------------------------------------------------
    print()
    print(line)
    print("STEP 30 (3D) SUMMARY")
    print(line)
    vn_exact_all_g = vn_floor < 1e-4
    vn_capacity_floor = vn_seenG > 3.0 * max(vn_seen0, 1e-9)
    soft_recovers_capacity = soft_seenG < 0.5 * vn_seenG
    soft_breaks_exactness = rpp_eq0 > 50.0 * max(vn_floor, 1e-12)
    dial_monotone = rho_hi < rho_lo

    print(f"    [1] capacity: the hard VN's seen error rises x{vn_seenG / max(vn_seen0, 1e-9):.1f} "
          f"({vn_seen0:.4f}->{vn_seenG:.4f}) as the break grows -- structurally blind to it; the")
    print(f"        soft/free pathway recovers it ({soft_seenG:.4f} at g={GVALS[-1]}, x{vn_seenG / max(soft_seenG, 1e-9):.1f} better).")
    print(f"    [2] generalisation: VN stays flat across SO(3) (ratio <= x{vn_ratio:.2f}); the free MLP")
    print(f"        pays up to x{mlp_ratio:.1f} for extrapolation. Capacity is bought with across-group reach.")
    print(f"    [3] exactness: VN at the float floor (<= {vn_floor:.1e}) for EVERY break; the soft model leaves")
    print(f"        the floor the instant the residual is active (x{rpp_eq0 / max(vn_floor, 1e-12):.0f} at g=0). Capacity is bought with exactness.")
    print(f"    headline (matches the 2D arm): in SO(3) too the soft prior is a continuous dial, not a free")
    print(f"    lunch -- it buys capacity by spending across-group reach AND float-floor exactness. Only the")
    print(f"    hard architecture holds the exact corner; at g=0 it dominates (same fit, exact, free).")

    assert vn_exact_all_g, "VN must stay exactly SO(3)-equivariant (structural) at every break g"
    assert vn_capacity_floor, "the break must be strong enough that the hard VN visibly cannot fit it"
    assert soft_recovers_capacity, "the soft/free pathway must recover the in-distribution capacity"
    assert soft_breaks_exactness, "the soft model must forfeit float-floor exactness when the residual is active"
    assert dial_monotone, "the softness knob must move the free-fraction monotonically"
    print(f"\n    guards: vn-exact-all-g={vn_exact_all_g}  vn-capacity-floor={vn_capacity_floor}  "
          f"soft-recovers-capacity={soft_recovers_capacity}")
    print(f"            soft-breaks-exactness={soft_breaks_exactness}  dial-monotone={dial_monotone}")
    print("    PASS: in SO(3) too, the soft middle is a tunable tradeoff; the exact corner is the architecture's.")

    # --- dump JSON -----------------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "group": "SO(3)",
        "config": dict(seeds=SEEDS, gvals=GVALS, betas=BETAS, n_data=N_DATA, steps=STEPS,
                       batch=BATCH, coverage_deg=COVERAGE_DEG, n_test=N_TEST, n_state=N_STATE),
        "params": {"VN": vn_p, "MLP": mlp_p, "RPP": rpp_p},
        "noneq_fraction": {str(g): nf_mean[g] for g in GVALS},
        "models": {m: {str(g): {k: cell(m, g, k) for k in ("seen", "ood", "ratio", "eq", "rho")}
                       for g in GVALS} for m in MODELS},
        "derived": {"vn_floor": vn_floor, "vn_seen0": vn_seen0, "vn_seenG": vn_seenG,
                    "soft_seenG": soft_seenG, "rpp_eq0_softest": rpp_eq0,
                    "rho_hi_gmax": rho_hi, "rho_lo_gmax": rho_lo,
                    "vn_exact_all_g": bool(vn_exact_all_g), "vn_capacity_floor": bool(vn_capacity_floor),
                    "soft_recovers_capacity": bool(soft_recovers_capacity),
                    "soft_breaks_exactness": bool(soft_breaks_exactness),
                    "dial_monotone": bool(dial_monotone)},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step30_soft_equivariant_3d_smoke.json" if SMOKE else "step30_soft_equivariant_3d.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")

    # --- figure (guarded) ----------------------------------------------------------------
    if not SMOKE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [nf_mean[g] for g in GVALS]
            colors = {"VN": "C2", "MLP": "C3"}
            fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.3))

            for m in MODELS:
                c = colors.get(m, "C0")
                style = "o-" if m in ("VN", "MLP") else "s--"
                axA.plot(xs, [cell(m, g, "seen")[0] for g in GVALS], style, label=m, color=c if m in colors else None)
            axA.set_xlabel("symmetry-break fraction (model-independent)")
            axA.set_ylabel("seen relMSE (in-ball capacity)")
            axA.set_title("[1] capacity: hard VN is blind to the break")
            axA.legend(fontsize=8); axA.grid(alpha=0.3)

            for m in MODELS:
                c = colors.get(m, "C0")
                style = "o-" if m in ("VN", "MLP") else "s--"
                axB.plot(xs, [cell(m, g, "eq")[0] for g in GVALS], style, label=m, color=c if m in colors else None)
            axB.axhline(vn_floor, color="C2", ls=":", alpha=0.6)
            axB.set_yscale("log"); axB.set_xlabel("symmetry-break fraction")
            axB.set_ylabel("residual equivariance  Delta_eq")
            axB.set_title("[3] exactness: only VN holds the floor (SO(3))")
            axB.legend(fontsize=8); axB.grid(alpha=0.3, which="both")

            gx = GVALS[-1]
            for m in MODELS:
                c = colors.get(m, "C0")
                axC.scatter(cell(m, gx, "eq")[0], cell(m, gx, "seen")[0],
                            s=70, label=m, color=c if m in colors else None,
                            marker=("o" if m in ("VN", "MLP") else "s"))
            axC.set_xscale("log")
            axC.set_xlabel("Delta_eq (exactness; left = better)")
            axC.set_ylabel("seen relMSE (capacity; lower = better)")
            axC.set_title(f"the dial at break g={gx}: RPP between VN & MLP")
            axC.legend(fontsize=8); axC.grid(alpha=0.3, which="both")

            fig.tight_layout()
            fig_path = fig_dir / "step30_soft_equivariant_3d.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"    wrote {fig_path.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    (figure skipped: {e})")

    sys.exit(0)


if __name__ == "__main__":
    main()
