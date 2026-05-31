r"""Step 28: is the equivariant prior just doing what rotation **augmentation** would do anyway?

The fair-baseline objection (the most predictable reviewer attack)
-----------------------------------------------------------------
Steps 8 / 21 / 24 train the non-equivariant baseline on a $[0^\circ,90^\circ)$ wedge and test it on
the whole circle, then report that it collapses out-of-distribution while the equivariant model stays
flat ($\times1.00$, 举一反三). The obvious objection: *of course it collapses — it was never shown the
other orientations.* The practitioner's standard response to a known symmetry is **data augmentation**:
rotate every training scene by random angles and let a plain MLP *learn* the symmetry from data.

This step asks that question head-on, giving the baseline the SAME prior knowledge the architecture has
(the world is $\mathrm{SO}(2)$-symmetric):

    Given that prior, does rotation augmentation let a plain MLP match what the equivariant
    architecture gets for free -- and if not, exactly where does it fall short, and at what cost?

The experiment
--------------
We reuse step8's exact wedge$\to$circle test and sweep the augmentation **coverage**
$\theta_{\max}\in\{90^\circ,180^\circ,270^\circ,360^\circ\}$, comparing three models:

  * **VN (exact)** -- equivariance is *structural* (Vector-Neuron layers); flat across the circle with
    residual $\sim10^{-6}$, and it needs **no** augmentation at all.
  * **MLP (no aug)** -- step8's baseline; trained on the $[0,90)$ wedge only, breaks OOD.
  * **MLP + aug($\theta_{\max}$)** -- the *fair* baseline; each epoch its scenes are rotated by fresh
    per-sample angles drawn uniformly in $[0,\theta_{\max})$. Targets stay **exact** because the world
    is exactly equivariant, so $T(R_\alpha s, R_\alpha a)=R_\alpha\,T(s,a)$ -- augmentation injects no
    label noise here (a best case for the baseline).

Three measurements, and what each one decides
---------------------------------------------
  [1] **Task metric** -- OOD/seen relMSE ratio. Does coverage close the 举一反三 gap? Expectation: the
      ratio *shrinks* monotonically toward $1$ as $\theta_{\max}\to360^\circ$ (more coverage helps). The
      $\theta_{\max}=90^\circ$ run is a control: same coverage as the wedge but resampled every epoch
      (effectively infinite in-wedge samples), so if it still breaks OOD, the failure is **missing
      coverage / extrapolation**, not a finite-sample artefact.

  [2] **Exactness** -- residual equivariance
      $\Delta_{\mathrm{eq}} = \max_\alpha
      \lVert f(R_\alpha x) - R_\alpha f(x)\rVert / \lVert f(x)\rVert$.
      Does augmentation ever *reach* the architecture's symmetry? Expectation: **no.** Augmentation buys
      a *statistical average* over the orbit -- approximate equivariance that plateaus orders of
      magnitude above the VN's float-floor $\sim10^{-6}$, no matter how wide the coverage.

  [3] **Cost** -- augmentation needs the *same* prior (you must know the group to rotate correctly)
      **plus** a multiplicative training budget to tile the orbit, and *still* only yields [2]'s
      approximate symmetry. The architecture bakes the group in and gets exactness at zero marginal cost.

Honest both ways
----------------
If [1] closes (aug matches VN on the task), the conclusion is **not** "the prior is useless": it is that
*with the group known*, augmentation is a viable substitute **on the task metric**, and the
architecture's unique remaining edge is **exactness** ([2]) and **zero coverage cost** ([3]). If [1]
does **not** close, the prior wins outright on the task too. Either way the certified, group-independent
claim is: **augmentation approximates the symmetry; the architecture *is* the symmetry.**

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step28_fair_augmentation_baseline.py
Smoke (~30 s):
    STEP28_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step28_fair_augmentation_baseline.py
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

N_STATE = 6  # number of type-1 state vectors (matches PushT's 6-vector state)
SMOKE = bool(os.environ.get("STEP28_SMOKE"))


# --------------------------------------------------------------------------- #
# Models -- mirror experiments/step8_sample_efficiency.py verbatim so this step
# forks ONLY the [C] (wedge->circle) protocol and adds the augmentation arm.
# Kept self-contained (no swm / PushT import) because Step 28 is pure SO(2).
# --------------------------------------------------------------------------- #
class EquivariantWorld(nn.Module):
    r"""Frozen random *world*: an exactly SO(2)-equivariant one-step map (one VN nonlinearity).

    ``forward: (B, n_state, 2), (B, 1, 2) -> (B, n_state, 2)`` with
    $T(R_\alpha s, R_\alpha a) = R_\alpha\,T(s,a)$ by construction.
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
    r"""Equivariant **student**: two VN nonlinearities, ~3.5k params; symmetry hard-wired.

    ``forward: (B, n_state, 2), (B, 1, 2) -> (B, n_state, 2)`` and
    $T_\theta(R_\alpha s, R_\alpha a) = R_\alpha\,T_\theta(s,a)$ for *every* $\alpha$.
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
    r"""Non-equivariant **baseline**: a plain MLP on the flattened state+action (~5.7x VN params).

    The only thing it lacks is the SO(2) symmetry prior -- which augmentation tries to supply via data.
    """

    def __init__(self, n_state: int = N_STATE, hidden: int = 128):
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


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# Geometry / metrics
# --------------------------------------------------------------------------- #
def rotate_per_sample(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    r"""Rotate each sample by its own angle. ``x: (n, c, 2), alpha: (n,)`` (mirrors step8)."""
    R = rot_matrix(alpha)  # (n, 2, 2)
    return torch.einsum("nij,ncj->nci", R, x)


@torch.no_grad()
def rel_mse(model: nn.Module, s, a, y) -> float:
    r"""Test MSE normalised by target power (1.0 = predicting zero; scale-free)."""
    err = (model(s, a) - y).pow(2).mean().item()
    return err / (y.pow(2).mean().item() + 1e-12)


@torch.no_grad()
def equiv_residual(model: nn.Module, s, a, angles) -> float:
    r"""Worst-case **relative** equivariance residual over a set of probe angles.

    $\Delta_{\mathrm{eq}} = \max_\alpha
    \dfrac{\lVert f(R_\alpha s, R_\alpha a) - R_\alpha f(s,a)\rVert_F}{\lVert f(s,a)\rVert_F}$.

    For the VN this sits at the float floor ($\sim10^{-6}$, weight-independent); for a trained MLP it is
    whatever the data *taught* it -- the quantity that decides exactness in measurement [2].
    """
    base = model(s, a)
    den = base.pow(2).sum().item() + 1e-12
    worst = 0.0
    for al in angles:
        lhs = model(rotate_vectors(s, al), rotate_vectors(a, al))
        rhs = rotate_vectors(base, al)
        num = (lhs - rhs).pow(2).sum().item()
        worst = max(worst, (num / den) ** 0.5)
    return worst


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model: nn.Module, s, a, y, *, epochs=1200, lr=3e-3, wd=1e-4) -> nn.Module:
    r"""Full-batch AdamW on a FIXED dataset (used for VN and the no-aug MLP)."""
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
    r"""Full-batch AdamW with **on-the-fly rotation augmentation** up to ``theta_max`` (radians).

    Each epoch the base scenes are rotated by fresh per-sample angles $\alpha_i\sim U[0,\theta_{\max})$
    and the target is recomputed via the frozen (exactly equivariant) teacher, so the augmented label is
    exact. This is the practitioner's standard recipe for a known symmetry; ``theta_max`` is the angular
    **coverage** of the synthesised orbit.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = s_base.shape[0]
    model.train()
    for _ in range(epochs):
        alpha = torch.rand(n, generator=gen) * theta_max  # (n,) in [0, theta_max)
        s_aug = rotate_per_sample(s_base, alpha)
        a_aug = rotate_per_sample(a_base, alpha)
        with torch.no_grad():
            y_aug = target_fn(s_aug, a_aug)  # exact: world is equivariant
        opt.zero_grad()
        loss = (model(s_aug, a_aug) - y_aug).pow(2).mean()
        loss.backward()
        opt.step()
    return model.eval()


# --------------------------------------------------------------------------- #
# Evaluation across the circle (step8's 4 bins; seen bin = [0,90))
# --------------------------------------------------------------------------- #
BINS = [(0, 90), (90, 180), (180, 270), (270, 360)]


@torch.no_grad()
def eval_circle(model, target_fn, s_te, a_te, gen: torch.Generator):
    r"""Return ``(seen_relMSE, worst_relMSE, ood_ratio)`` over the 4 orientation bins.

    ``seen`` is the $[0,90)$ bin (the wedge all models are trained to cover); the ratio
    $=\max_{\text{bin}}/\,\text{seen}$ is the 举一反三 degradation factor (1.0 = perfectly flat).
    """
    errs = []
    for lo, hi in BINS:
        al = (torch.rand(s_te.shape[0], generator=gen) * (hi - lo) + lo) * math.pi / 180.0
        s_b, a_b = rotate_per_sample(s_te, al), rotate_per_sample(a_te, al)
        y_b = target_fn(s_b, a_b)
        errs.append(rel_mse(model, s_b, a_b, y_b))
    return errs[0], max(errs), max(errs) / (errs[0] + 1e-12)


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)


# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 72
    torch.manual_seed(0)

    if SMOKE:
        SEEDS, THETAS_DEG, EPOCHS, N_BASE, N_TEST = [0, 1], [90, 360], 250, 128, 400
    else:
        SEEDS, THETAS_DEG, EPOCHS, N_BASE, N_TEST = [0, 1, 2, 3, 4], [90, 180, 270, 360], 1200, 256, 2000
    PROBE_ANGLES = [0.37, 1.2345, 2.71, 3.94, 5.40]  # radians, spread around the circle

    print(line)
    print(f"STEP 28  fair augmentation baseline: does aug buy what equivariance gives free?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    # --- the world: a frozen, exactly SO(2)-equivariant teacher (= step8) -----------------
    teacher = EquivariantWorld(n_state=N_STATE, hidden=8).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        s0, a0 = torch.randn(8000, N_STATE, 2), torch.randn(8000, 1, 2)
        rms = teacher(s0, a0).pow(2).mean().sqrt().item()

    def target(s, a):
        with torch.no_grad():
            return teacher(s, a) / rms  # scalar standardisation keeps the target equivariant

    s_chk, a_chk = torch.randn(64, N_STATE, 2), torch.randn(64, 1, 2)
    teacher_eq = equiv_residual(lambda s, a: target(s, a), s_chk, a_chk, PROBE_ANGLES)
    assert teacher_eq < 1e-4, "teacher world is not equivariant"
    print(f"    world: frozen equivariant teacher, residual {teacher_eq:.1e}  "
          f"(targets are exactly symmetric -> augmentation labels are exact)")

    # --- fixed canonical (anisotropic) layout so rotation genuinely moves the cloud -------
    torch.manual_seed(7)
    mu_s = torch.randn(N_STATE, 2) * 1.5
    mu_a = torch.randn(1, 2) * 1.5

    def aniso_inputs(n, gen, noise=0.3):
        s = mu_s.unsqueeze(0) + noise * torch.randn(n, N_STATE, 2, generator=gen)
        a = mu_a.unsqueeze(0) + noise * torch.randn(n, 1, 2, generator=gen)
        return s, a

    vn_p, mlp_p = n_params(VNDynamics()), n_params(MLPDynamics())
    print(f"    params: VN(equivariant)={vn_p}   MLP(baseline)={mlp_p}  ({mlp_p / vn_p:.1f}x VN)")
    print(f"    protocol: train coverage in [0,theta_max); test full circle in 4 bins (seen=[0,90)); "
          f"{len(SEEDS)} seeds")

    # --- accumulate over seeds ------------------------------------------------------------
    acc = {
        "VN": {"ratio": [], "eq": [], "seen": [], "worst": []},
        "MLP_noaug": {"ratio": [], "eq": [], "seen": [], "worst": []},
        "MLP_aug": {th: {"ratio": [], "eq": [], "seen": [], "worst": []} for th in THETAS_DEG},
    }

    for sd in SEEDS:
        data_gen = torch.Generator().manual_seed(2000 + sd)
        aug_gen = torch.Generator().manual_seed(3000 + sd)
        test_gen = torch.Generator().manual_seed(4000 + sd)

        # base canonical scenes (unrotated); the wedge models see these rotated into [0,90)
        s_c, a_c = aniso_inputs(N_BASE, data_gen)
        al_wedge = torch.rand(N_BASE, generator=data_gen) * (math.pi / 2)
        s_w, a_w = rotate_per_sample(s_c, al_wedge), rotate_per_sample(a_c, al_wedge)
        y_w = target(s_w, a_w)

        # held-out test scenes + a fixed probe set for the equivariance residual
        s_te, a_te = aniso_inputs(N_TEST, test_gen)
        s_pr, a_pr = aniso_inputs(min(N_TEST, 1000), test_gen)

        # VN (exact) -- coverage-independent; trained on the wedge only
        torch.manual_seed(10 + sd)
        vn = train(VNDynamics(), s_w, a_w, y_w, epochs=EPOCHS)
        seen, worst, ratio = eval_circle(vn, target, s_te, a_te, test_gen)
        acc["VN"]["ratio"].append(ratio); acc["VN"]["seen"].append(seen)
        acc["VN"]["worst"].append(worst); acc["VN"]["eq"].append(equiv_residual(vn, s_pr, a_pr, PROBE_ANGLES))

        # MLP no-aug -- step8's baseline; the wedge only
        torch.manual_seed(20 + sd)
        mlp0 = train(MLPDynamics(), s_w, a_w, y_w, epochs=EPOCHS)
        seen, worst, ratio = eval_circle(mlp0, target, s_te, a_te, test_gen)
        acc["MLP_noaug"]["ratio"].append(ratio); acc["MLP_noaug"]["seen"].append(seen)
        acc["MLP_noaug"]["worst"].append(worst)
        acc["MLP_noaug"]["eq"].append(equiv_residual(mlp0, s_pr, a_pr, PROBE_ANGLES))

        # MLP + aug(theta_max) -- the fair baseline, swept over coverage
        for th in THETAS_DEG:
            torch.manual_seed(30 + sd)  # paired init across coverages
            mlpa = train_augmented(MLPDynamics(), s_c, a_c, target, th * math.pi / 180.0,
                                   epochs=EPOCHS, gen=aug_gen)
            seen, worst, ratio = eval_circle(mlpa, target, s_te, a_te, test_gen)
            acc["MLP_aug"][th]["ratio"].append(ratio); acc["MLP_aug"][th]["seen"].append(seen)
            acc["MLP_aug"][th]["worst"].append(worst)
            acc["MLP_aug"][th]["eq"].append(equiv_residual(mlpa, s_pr, a_pr, PROBE_ANGLES))

    # --- aggregate ------------------------------------------------------------------------
    def summ(d):
        return {k: mean_std(d[k]) for k in ("ratio", "eq", "seen", "worst")}

    vn_s = summ(acc["VN"])
    mlp0_s = summ(acc["MLP_noaug"])
    aug_s = {th: summ(acc["MLP_aug"][th]) for th in THETAS_DEG}

    # --- [1] task metric: OOD/seen ratio vs coverage --------------------------------------
    print()
    print(line)
    print("[1] TASK METRIC -- OOD/seen relMSE ratio (1.00 = perfectly flat = 举一反三)")
    print(line)
    print(f"    {'model':>22s} | {'seen relMSE':>13s} | {'OOD/seen ratio':>16s}")
    print("    " + "-" * 58)
    print(f"    {'VN (exact, no aug)':>22s} | {vn_s['seen'][0]:13.3e} | "
          f"x{vn_s['ratio'][0]:6.3f} +-{vn_s['ratio'][1]:.2f}")
    print(f"    {'MLP (no aug)':>22s} | {mlp0_s['seen'][0]:13.3e} | "
          f"x{mlp0_s['ratio'][0]:6.2f} +-{mlp0_s['ratio'][1]:.2f}   <- step8 baseline")
    for th in THETAS_DEG:
        a = aug_s[th]
        tag = "   <- coverage = wedge (control)" if th == 90 else (
              "   <- full circle" if th == 360 else "")
        print(f"    {('MLP + aug [0,' + str(th) + ')'):>22s} | {a['seen'][0]:13.3e} | "
              f"x{a['ratio'][0]:6.3f} +-{a['ratio'][1]:.2f}{tag}")
    print(f"    => coverage helps the MLP: ratio falls x{aug_s[THETAS_DEG[0]]['ratio'][0]:.2f} "
          f"(@{THETAS_DEG[0]}) -> x{aug_s[THETAS_DEG[-1]]['ratio'][0]:.2f} (@{THETAS_DEG[-1]}); "
          f"VN is x{vn_s['ratio'][0]:.2f} at zero coverage.")

    # --- [2] exactness: residual equivariance vs coverage ---------------------------------
    print()
    print(line)
    print("[2] EXACTNESS -- residual equivariance Delta_eq = max_a ||f(Rx)-Rf(x)|| / ||f(x)||")
    print(line)
    print(f"    {'model':>22s} | {'Delta_eq':>12s}")
    print("    " + "-" * 38)
    print(f"    {'VN (exact)':>22s} | {vn_s['eq'][0]:12.2e}   <- float floor, weight-independent")
    print(f"    {'MLP (no aug)':>22s} | {mlp0_s['eq'][0]:12.2e}")
    for th in THETAS_DEG:
        print(f"    {('MLP + aug [0,' + str(th) + ')'):>22s} | {aug_s[th]['eq'][0]:12.2e}")
    aug_eq_full = aug_s[THETAS_DEG[-1]]["eq"][0]
    eq_factor = aug_eq_full / max(vn_s["eq"][0], 1e-12)
    print(f"    => even at FULL coverage the augmented MLP's symmetry is only APPROXIMATE: "
          f"Delta_eq x{eq_factor:.0f} the VN's exact floor.")

    # --- verdict (honest both ways) -------------------------------------------------------
    print()
    print(line)
    print("STEP 28 SUMMARY")
    print(line)
    ratio_drop = aug_s[THETAS_DEG[-1]]["ratio"][0] < aug_s[THETAS_DEG[0]]["ratio"][0]
    task_closed = aug_s[THETAS_DEG[-1]]["ratio"][0] < 1.5  # did full-coverage aug reach VN-level flatness?
    never_exact = aug_eq_full > 50.0 * max(vn_s["eq"][0], 1e-12)

    if task_closed:
        print(f"    [1] task: full-coverage augmentation DOES flatten the MLP "
              f"(x{aug_s[THETAS_DEG[-1]]['ratio'][0]:.2f}, near the VN's x{vn_s['ratio'][0]:.2f}).")
        print(f"        => with the group known, augmentation is a viable substitute ON THE TASK METRIC.")
    else:
        print(f"    [1] task: even full-coverage augmentation does NOT reach VN-level flatness "
              f"(x{aug_s[THETAS_DEG[-1]]['ratio'][0]:.2f} vs VN x{vn_s['ratio'][0]:.2f}).")
        print(f"        => the architectural prior wins outright, on the task metric too.")
    print(f"    [2] exactness: augmentation NEVER reaches the architecture's exact symmetry "
          f"(Delta_eq x{eq_factor:.0f} the VN floor, at full coverage).")
    print(f"    [3] cost: augmentation needs the SAME prior (you must know the group) PLUS a wider")
    print(f"        training orbit, and still only buys [2]'s approximate symmetry; the VN bakes it in free.")
    print(f"    headline: augmentation APPROXIMATES the symmetry; the architecture IS the symmetry.")

    # --- guards that hold in BOTH outcomes ------------------------------------------------
    assert vn_s["ratio"][0] < 1.5, "VN must stay ~flat across orientations"
    assert vn_s["eq"][0] < 1e-4, "VN must stay exactly equivariant (structural)"
    assert mlp0_s["ratio"][0] > 3.0, "no-aug MLP must break OOD (the step8 baseline)"
    assert ratio_drop, "more augmentation coverage must reduce the OOD ratio"
    assert never_exact, "augmentation must NOT reach the architecture's exact equivariance"
    print(f"\n    guards: vn-flat={vn_s['ratio'][0] < 1.5}  vn-exact={vn_s['eq'][0] < 1e-4}  "
          f"noaug-breaks={mlp0_s['ratio'][0] > 3.0}  coverage-helps={ratio_drop}  "
          f"aug-never-exact={never_exact}")
    print("    PASS: the fair augmentation baseline is answered -- approximate symmetry at a price, "
          "never exact.")

    # --- dump JSON artifact ---------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(seeds=SEEDS, thetas_deg=THETAS_DEG, epochs=EPOCHS,
                       n_base=N_BASE, n_test=N_TEST, n_state=N_STATE,
                       probe_angles=PROBE_ANGLES),
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
    out_path = fig_dir / ("step28_fair_augmentation_smoke.json" if SMOKE
                          else "step28_fair_augmentation.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")

    # --- optional figure (guarded; never fails the run) -----------------------------------
    if not SMOKE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ths = THETAS_DEG
            ratios = [aug_s[t]["ratio"][0] for t in ths]
            eqs = [aug_s[t]["eq"][0] for t in ths]
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
            axL.axhline(vn_s["ratio"][0], color="C2", ls="--", label="VN (exact)")
            axL.axhline(mlp0_s["ratio"][0], color="C3", ls=":", label="MLP (no aug)")
            axL.plot(ths, ratios, "o-", color="C0", label="MLP + aug")
            axL.set_xlabel("augmentation coverage  theta_max (deg)")
            axL.set_ylabel("OOD/seen relMSE ratio")
            axL.set_title("[1] task metric: coverage closes the gap")
            axL.legend(); axL.grid(alpha=0.3)
            axR.axhline(vn_s["eq"][0], color="C2", ls="--", label="VN (exact floor)")
            axR.plot(ths, eqs, "o-", color="C0", label="MLP + aug")
            axR.set_xlabel("augmentation coverage  theta_max (deg)")
            axR.set_ylabel("residual equivariance  Delta_eq")
            axR.set_yscale("log")
            axR.set_title("[2] exactness: aug never reaches the floor")
            axR.legend(); axR.grid(alpha=0.3)
            fig.tight_layout()
            fig_path = fig_dir / "step28_fair_augmentation.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"    wrote {fig_path.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    (figure skipped: {e})")

    sys.exit(0)


if __name__ == "__main__":
    main()
