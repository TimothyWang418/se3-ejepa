r"""Step 30 (2D arm): the soft-equivariant model -- the missing middle between the hard prior
and the free baseline, tested under a *controlled symmetry break*.

Completing the Tier-1 arc
-------------------------
Steps 28-29 asked whether two scaling-style substitutes match the equivariant architecture:

  * **Step 28** (augmentation): with full coverage, rotation augmentation closes the across-group
    *task* metric but **never** reaches the architecture's *exactness* ($\Delta_{\mathrm{eq}}$
    plateaus $\sim\!10^5\times$ above the float floor). "Augmentation approximates the symmetry;
    the architecture *is* the symmetry."
  * **Step 29** (scale): at *partial* coverage, scaling MLP size $\times16$ and data $\times16$
    substitutes for neither the missing coverage (task) nor the architecture (exactness).

Both compared only the two **extremes**: a hard-equivariant Vector-Neuron map (exact, weight-free
symmetry) versus a free MLP (no symmetry). The honest question they leave open is the **middle**:
what if the world's symmetry is only *approximate*, and we relax the prior just enough to absorb the
violation? That is the **Residual Pathway Prior** (RPP; Finzi, Benton, Wilson, NeurIPS 2021) -- a
*soft*-equivariant model

$$ f_\beta(s,a) \;=\; \underbrace{f_{\mathrm{VN}}(s,a)}_{\text{exact pathway}}
                    \;+\; \underbrace{f_{\mathrm{free}}(s,a)}_{\text{residual pathway}}, $$

trained with a prior of strength $\beta$ that penalises the residual-pathway *output energy*
$\beta\,\mathbb{E}\lVert f_{\mathrm{free}}\rVert^2$. As $\beta\to\infty$ the residual is suppressed and
$f_\beta\to f_{\mathrm{VN}}$ (hard, exact); as $\beta\to0$ the residual is free and $f_\beta$ behaves
like the unconstrained MLP. $\beta$ is therefore a continuous **dial** between the two extremes Steps
28-29 measured, and we report the realised *free-fraction*
$\rho=\mathbb{E}\lVert f_{\mathrm{free}}\rVert/\mathbb{E}\lVert f_\beta\rVert$ as its model-side readout.

The controlled break (= Step 16, 2D)
-------------------------------------
The teacher is the exactly-$\mathrm{SO}(2)$ Vector-Neuron world $\mathrm{Dyn}_0$ plus a fixed lab-axis
anisotropy of strength $g$ (the 2D analogue of Step 16's lab-$z$ term):

$$ \mathrm{Dyn}_g(s,a)_c \;=\; \mathrm{Dyn}_0(s,a)_c \;-\; g\,(s_c\!\cdot e)\,e, \qquad e=(0,1). $$

At $g=0$ this is exactly equivariant; the added term references a fixed lab axis, so it is **not**
$\mathrm{SO}(2)$-equivariant -- $g$ is a clean, model-independent knob on symmetry-break energy
(measured by ``noneq_fraction``). Crucially the OOD set is **genuinely re-sampled** transitions of the
*same* broken teacher (not rotated targets): once $g>0$ the teacher is not equivariant, so a rotated
target would be a fake label.

Three metrics, one honest tradeoff
-----------------------------------
For five models -- hard VN, RPP at three softness levels, free MLP -- swept over the break $g$:

  [1] **capacity** (seen relMSE, in the training wedge): can the model represent the broken world?
      The hard VN has an *irreducible floor* -- it can only fit the equivariant projection of
      $\mathrm{Dyn}_g$, so its seen error **rises with $g$**. The residual pathway absorbs the break,
      so RPP/MLP stay low. *The soft model recovers the in-distribution capacity the hard VN lacks.*
  [2] **generalisation** (genuine across-group OOD relMSE / the OOD/seen ratio): the free MLP pays the
      rotation-extrapolation penalty; the hard VN generalises the symmetry for free; RPP interpolates.
      *Capacity is bought with across-group generalisation.*
  [3] **exactness** ($\Delta_{\mathrm{eq}}=\max_\alpha\lVert f(R_\alpha x)-R_\alpha f(x)\rVert/\lVert f(x)\rVert$):
      the hard VN sits at the float floor for **every** $g$ (structural, weight-independent); the instant
      the residual pathway is active, RPP leaves the floor -- **even at $g=0$**, where the symmetry is
      intact and exactness would be free. *Capacity is also bought with exactness.*

Headline: the soft architecture is a continuous dial, not a free lunch. Relaxing the prior buys the
capacity to absorb a broken symmetry, but every step costs across-group generalisation **and** forfeits
the float-floor exactness. Only the hard architecture occupies the exact corner -- and at $g=0$ it
dominates outright (same fit, exact, free). Augmentation (28), scale (29), and soft relaxation (30) each
give at most an *approximation* of what hard-wiring gives exactly.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step30_soft_equivariant.py
Smoke (~60 s):
    STEP30_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step30_soft_equivariant.py
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

from src.geometry.so2 import rot_matrix, rotate_vectors  # noqa: E402
from src.models.structured import VNLinear, VNReLU  # noqa: E402

torch.set_default_dtype(torch.float32)

N_STATE = 6
SMOKE = bool(os.environ.get("STEP30_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- VN / MLP identical to step28/step29; RPP is the new soft middle.
# --------------------------------------------------------------------------- #
class EquivariantWorld(nn.Module):
    r"""Frozen random *world*: an exactly $\mathrm{SO}(2)$-equivariant one-step map (one VN nonlinearity).

    ``forward: (B, n_state, 2), (B, 1, 2) -> (B, n_state, 2)`` with $T(R_\alpha s, R_\alpha a)=R_\alpha T(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 8):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 2)
        return self.l2(self.a1(self.l1(x)))


class VNDynamics(nn.Module):
    r"""Equivariant **student** (hard, $\sim\!3.5$k params); symmetry hard-wired and scale-free.

    ``forward: (B, n_state, 2), (B, 1, 2) -> (B, n_state, 2)`` and $T_\theta(R_\alpha s,R_\alpha a)=R_\alpha T_\theta(s,a)$.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 32):
        super().__init__()
        self.l1 = VNLinear(n_state + 1, hidden)
        self.a1 = VNReLU(hidden)
        self.l2 = VNLinear(hidden, hidden)
        self.a2 = VNReLU(hidden)
        self.l3 = VNLinear(hidden, n_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)  # (B, n_state+1, 2)
        return self.l3(self.a2(self.l2(self.a1(self.l1(x)))))


class MLPDynamics(nn.Module):
    r"""Non-equivariant **baseline** on the flattened state+action (no symmetry prior)."""

    def __init__(self, n_state: int = N_STATE, hidden: int = 256):
        super().__init__()
        self.n_state = n_state
        in_dim, out_dim = n_state * 2 + 2, n_state * 2
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
        return self.net(x).reshape(b, self.n_state, 2)


class ResidualPathwayPrior(nn.Module):
    r"""Soft-equivariant model (RPP; Finzi et al. 2021): $f_\beta = f_{\mathrm{VN}} + f_{\mathrm{free}}$.

    The exact pathway ``self.vn`` carries the $\mathrm{SO}(2)$ structure; the residual pathway
    ``self.free`` is an unconstrained MLP. Softness is imposed at train time by penalising the residual
    *output energy* with strength $\beta$ (see :func:`train_one`): $\beta\to\infty$ recovers the hard VN,
    $\beta\to0$ recovers the free MLP. ``forward`` shapes match the other two models.
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
# Geometry / metrics
# --------------------------------------------------------------------------- #
def rotate_per_sample(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    r"""Rotate each sample by its own angle. ``x: (n, c, 2), alpha: (n,) -> (n, c, 2)``."""
    R = rot_matrix(alpha)  # (n, 2, 2)
    return torch.einsum("nij,ncj->nci", R, x)


@torch.no_grad()
def rel_mse(model: nn.Module, s, a, y) -> float:
    err = (model(s, a) - y).pow(2).mean().item()
    return err / (y.pow(2).mean().item() + 1e-12)


@torch.no_grad()
def equiv_residual(model: nn.Module, s, a, angles) -> float:
    r"""Worst-case relative equivariance residual $\Delta_{\mathrm{eq}}$ over a set of probe angles."""
    base = model(s, a)
    den = base.pow(2).sum().item() + 1e-12
    worst = 0.0
    for al in angles:
        lhs = model(rotate_vectors(s, al), rotate_vectors(a, al))
        rhs = rotate_vectors(base, al)
        worst = max(worst, ((lhs - rhs).pow(2).sum().item() / den) ** 0.5)
    return worst


@torch.no_grad()
def free_fraction(model: ResidualPathwayPrior, s, a) -> float:
    r"""Residual-pathway share $\rho=\mathbb{E}\lVert f_{\mathrm{free}}\rVert/\mathbb{E}\lVert f_\beta\rVert$."""
    total = model(s, a).norm().item()
    free = model.free(s, a).norm().item()
    return free / (total + 1e-12)


@torch.no_grad()
def noneq_fraction(target_g, s, a, angles) -> float:
    r"""Model-independent symmetry-break energy of the *teacher* (the sweep's x-axis).

    $\dfrac{\mathbb{E}_\alpha\lVert \mathrm{Dyn}_g(R_\alpha s,R_\alpha a)-R_\alpha\,\mathrm{Dyn}_g(s,a)\rVert^2}
            {\mathbb{E}_\alpha\lVert \mathrm{Dyn}_g(s,a)\rVert^2}$ -- 0 at $g=0$, growing with $g$.
    """
    base = target_g(s, a)
    num = 0.0
    den = 0.0
    for al in angles:
        lhs = target_g(rotate_vectors(s, al), rotate_vectors(a, al))
        rhs = rotate_vectors(base, al)
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
    r"""Minibatch AdamW with on-the-fly augmentation in $[0,\theta_{\max})$ on the (broken) teacher.

    For a :class:`ResidualPathwayPrior` with ``beta`` $>0$ the loss adds $\beta\,\mathbb{E}\lVert
    f_{\mathrm{free}}\rVert^2$, the soft-equivariance prior that dials the residual pathway. For VN / MLP
    (or ``beta=0``) it is plain MSE. Fixed step budget; ``target_fn`` is recomputed on the rotated inputs
    each step (genuine labels, never rotated targets).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    is_rpp = isinstance(model, ResidualPathwayPrior)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        sb, ab = s_base[idx], a_base[idx]
        alpha = torch.rand(batch, generator=gen) * theta_max
        s_aug, a_aug = rotate_per_sample(sb, alpha), rotate_per_sample(ab, alpha)
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
# Evaluation: seen wedge [0,cov) vs genuine re-sampled OOD complement [cov,2pi)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_seen_ood(model, target_fn, s_te, a_te, cov: float, gen: torch.Generator):
    r"""Return ``(seen_relMSE, ood_relMSE, ood/seen)``; OOD = genuine re-sample of the broken teacher."""
    n = s_te.shape[0]
    al_seen = torch.rand(n, generator=gen) * cov
    s_s, a_s = rotate_per_sample(s_te, al_seen), rotate_per_sample(a_te, al_seen)
    seen = rel_mse(model, s_s, a_s, target_fn(s_s, a_s))
    al_ood = torch.rand(n, generator=gen) * (2 * math.pi - cov) + cov
    s_o, a_o = rotate_per_sample(s_te, al_ood), rotate_per_sample(a_te, al_ood)
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

    COVERAGE_DEG = 180  # same partial coverage as Step 29; the new swept axis is the break g
    if SMOKE:
        SEEDS, GVALS, BETAS, N_DATA, STEPS, BATCH, N_TEST = (
            [0, 1], [0.0, 0.6], [1.0, 1e-3], 512, 400, 128, 400)
    else:
        SEEDS, GVALS, BETAS, N_DATA, STEPS, BATCH, N_TEST = (
            [0, 1, 2, 3, 4], [0.0, 0.2, 0.4, 0.8], [1.0, 1e-2, 1e-4], 1024, 2000, 256, 2000)
    PROBE_ANGLES = [0.37, 1.2345, 2.71, 3.94, 5.40]
    BETA_LABELS = {b: lab for b, lab in zip(BETAS, ("hi", "mid", "lo"))}

    print(line)
    print(f"STEP 30  soft-equivariant model (RPP) under a controlled symmetry break  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    teacher = exact SO(2) VN world + g * fixed lab-axis anisotropy; sweep break g x softness beta")
    print(f"    models: hard VN, RPP(beta in {BETAS}), free MLP; {len(SEEDS)} seeds; coverage [0,{COVERAGE_DEG}) deg")

    # --- frozen, exactly SO(2)-equivariant teacher world (= step28/29) -------------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, 2), torch.randn(8000, 1, 2)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()
    e_lab = torch.tensor([0.0, 1.0])  # fixed lab axis for the break

    def make_target_g(g: float):
        r"""Broken teacher $\mathrm{Dyn}_g(s,a)_c = \mathrm{Dyn}_0(s,a)_c - g\,(s_c\!\cdot e)\,e$ (normalised)."""
        def target_g(s, a):
            with torch.no_grad():
                eqv = teacher(s, a) / rms
                proj = (s * e_lab).sum(dim=-1, keepdim=True) * e_lab  # (B,n_state,2): (s_c.e) e
                return eqv - g * proj
        return target_g

    # init-time equivariance unit test: the g=0 teacher must be exactly equivariant
    s_chk, a_chk = torch.randn(64, N_STATE, 2), torch.randn(64, 1, 2)
    teacher_eq = equiv_residual(lambda s, a: make_target_g(0.0)(s, a), s_chk, a_chk, PROBE_ANGLES)
    assert teacher_eq < 1e-4, "teacher world is not equivariant at g=0"

    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, 2) * 1.5
    mu_a = torch.randn(1, 2) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, 2, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, 2, generator=gen)
        return s, a

    vn_p, mlp_p, rpp_p = n_params(VNDynamics()), n_params(MLPDynamics()), n_params(ResidualPathwayPrior())
    print(f"    params: VN={vn_p}  MLP={mlp_p}  RPP={rpp_p} (VN+MLP pathways)")
    print(f"    budget: {STEPS} AdamW steps, batch {BATCH}, N_data={N_DATA}\n")

    cov = COVERAGE_DEG * math.pi / 180.0
    MODELS = ["VN"] + [f"RPP-{BETA_LABELS[b]}" for b in BETAS] + ["MLP"]
    # acc[model][g] -> dict of metric-lists over seeds; nf[g] -> list over seeds
    acc = {m: {g: {"seen": [], "ood": [], "ratio": [], "eq": [], "rho": []} for g in GVALS} for m in MODELS}
    nf = {g: [] for g in GVALS}

    for sd in SEEDS:
        base_gen = fresh(2000 + sd)
        s_te, a_te = aniso_inputs(N_TEST, base_gen)
        s_pr, a_pr = aniso_inputs(min(N_TEST, 1000), base_gen)
        s_c, a_c = aniso_inputs(N_DATA, base_gen)  # base scenes, identical across g

        for g in GVALS:
            target_g = make_target_g(g)
            nf[g].append(noneq_fraction(target_g, s_pr, a_pr, PROBE_ANGLES))

            def run(model, beta, name):
                m = train_one(model, s_c, a_c, target_g, cov, beta=beta,
                              steps=STEPS, batch=BATCH, gen=fresh(3000 + sd))
                seen, ood, ratio = eval_seen_ood(m, target_g, s_te, a_te, cov, fresh(4000 + sd))
                eq = equiv_residual(m, s_pr, a_pr, PROBE_ANGLES)
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

    def print_table(title, key, fmt, footer=None):
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
        if footer:
            print(footer)

    # --- [1] capacity --------------------------------------------------------------------
    print_table("[1] CAPACITY -- seen relMSE (in-wedge fit of the broken world; lower = better)",
                "seen", lambda v: f"  {v[0]:8.4f}      ")
    vn_seen0 = cell("VN", GVALS[0], "seen")[0]
    vn_seenG = cell("VN", GVALS[-1], "seen")[0]
    soft_seenG = min(cell("MLP", GVALS[-1], "seen")[0], cell(f"RPP-{BETA_LABELS[BETAS[-1]]}", GVALS[-1], "seen")[0])
    print(f"    => VN seen {vn_seen0:.4f} (g=0) -> {vn_seenG:.4f} (g={GVALS[-1]}): the hard prior cannot fit the")
    print(f"       fixed-axis break (irreducible floor). Softest model at g={GVALS[-1]}: {soft_seenG:.4f} -- capacity recovered.")

    # --- [2] generalisation --------------------------------------------------------------
    print()
    print_table("[2] GENERALISATION -- genuine across-group OOD relMSE  (and OOD/seen ratio)",
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
    print_table("[3] EXACTNESS -- residual equivariance Delta_eq = max_a ||f(Rx)-Rf(x)|| / ||f(x)||",
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
    print("STEP 30 SUMMARY")
    print(line)
    vn_exact_all_g = vn_floor < 1e-4
    vn_capacity_floor = vn_seenG > 3.0 * max(vn_seen0, 1e-9)
    soft_recovers_capacity = soft_seenG < 0.5 * vn_seenG
    soft_breaks_exactness = rpp_eq0 > 50.0 * max(vn_floor, 1e-12)
    dial_monotone = rho_hi < rho_lo

    print(f"    [1] capacity: the hard VN's seen error rises x{vn_seenG / max(vn_seen0, 1e-9):.1f} "
          f"({vn_seen0:.4f}->{vn_seenG:.4f}) as the break grows -- it is structurally blind to it; the")
    print(f"        soft/free pathway recovers it ({soft_seenG:.4f} at g={GVALS[-1]}, x{vn_seenG / max(soft_seenG, 1e-9):.1f} better).")
    print(f"    [2] generalisation: VN stays flat across orientations (ratio <= x{vn_ratio:.2f}); the free MLP")
    print(f"        pays up to x{mlp_ratio:.1f} for extrapolation. Capacity is bought with across-group reach.")
    print(f"    [3] exactness: VN at the float floor (<= {vn_floor:.1e}) for EVERY break; the soft model leaves")
    print(f"        the floor the instant the residual is active (x{rpp_eq0 / max(vn_floor, 1e-12):.0f} at g=0). Capacity is bought with exactness.")
    print(f"    headline: the soft prior is a continuous dial, not a free lunch -- it buys capacity to absorb a")
    print(f"    broken symmetry but spends across-group generalisation AND float-floor exactness. Only the hard")
    print(f"    architecture holds the exact corner; at g=0 it dominates (same fit, exact, free).")

    assert vn_exact_all_g, "VN must stay exactly equivariant (structural) at every break g"
    assert vn_capacity_floor, "the break must be strong enough that the hard VN visibly cannot fit it"
    assert soft_recovers_capacity, "the soft/free pathway must recover the in-distribution capacity"
    assert soft_breaks_exactness, "the soft model must forfeit float-floor exactness when the residual is active"
    assert dial_monotone, "the softness knob must move the free-fraction monotonically"
    print(f"\n    guards: vn-exact-all-g={vn_exact_all_g}  vn-capacity-floor={vn_capacity_floor}  "
          f"soft-recovers-capacity={soft_recovers_capacity}")
    print(f"            soft-breaks-exactness={soft_breaks_exactness}  dial-monotone={dial_monotone}")
    print("    PASS: the soft middle is a tunable tradeoff; the exact corner belongs to the architecture alone.")

    # --- dump JSON -----------------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "group": "SO(2)",
        "config": dict(seeds=SEEDS, gvals=GVALS, betas=BETAS, n_data=N_DATA, steps=STEPS,
                       batch=BATCH, coverage_deg=COVERAGE_DEG, n_test=N_TEST, n_state=N_STATE,
                       probe_angles=PROBE_ANGLES),
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
    out_path = fig_dir / ("step30_soft_equivariant_smoke.json" if SMOKE else "step30_soft_equivariant.json")
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
            axA.set_ylabel("seen relMSE (in-wedge capacity)")
            axA.set_title("[1] capacity: hard VN is blind to the break")
            axA.legend(fontsize=8); axA.grid(alpha=0.3)

            for m in MODELS:
                c = colors.get(m, "C0")
                style = "o-" if m in ("VN", "MLP") else "s--"
                axB.plot(xs, [cell(m, g, "eq")[0] for g in GVALS], style, label=m, color=c if m in colors else None)
            axB.axhline(vn_floor, color="C2", ls=":", alpha=0.6)
            axB.set_yscale("log"); axB.set_xlabel("symmetry-break fraction")
            axB.set_ylabel("residual equivariance  Delta_eq")
            axB.set_title("[3] exactness: only VN holds the floor")
            axB.legend(fontsize=8); axB.grid(alpha=0.3, which="both")

            # dial: seen (capacity) vs Delta_eq (exactness) at the largest break -- RPP interpolates
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
            fig_path = fig_dir / "step30_soft_equivariant.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"    wrote {fig_path.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    (figure skipped: {e})")

    sys.exit(0)


if __name__ == "__main__":
    main()
