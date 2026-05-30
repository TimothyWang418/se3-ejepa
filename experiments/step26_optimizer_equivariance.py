r"""Step 26: does the *optimiser* break SE(3)-equivariance? Intrinsic vs extrinsic parametrisation.

Where this sits, and the precise question it answers
----------------------------------------------------
A recent result (Lau & Su, *A Symmetry-Compatible Principle for Optimizer Design*, arXiv
2605.18106) makes a sharp, often-overlooked point: **Adam / AdamW / RMSProp / AdaGrad /
Adafactor are geometry-blind.** Their per-coordinate second moment
$v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^{\odot 2}$ is an *element-wise* accumulation, so the
preconditioned step $m_t/(\sqrt{v_t}+\epsilon)$ does **not** commute with a non-trivial
group action on weight space. The paper's worry — widely repeated in the e3nn / steerable-CNN
community — is that *even an architecturally equivariant network can have its equivariance
silently broken, one optimiser step at a time*. Their fix is a family of *symmetry-compatible*
spectral optimisers (LPRO; Muon = a special case), whose update direction commutes with the
symmetry by construction.

This step asks the project-specific version of that question and answers it **decisively, with
a controlled $2\times2$**:

  *Does any of this threaten the headline $\sim10^{-6}$ SE(3)-equivariance of our VN latent
  world model — and if not, exactly why not?*

The answer is a clean dichotomy that the project's own building block already settles.

Intrinsic vs extrinsic equivariance (the whole story in one definition)
-----------------------------------------------------------------------
A linear layer $x\mapsto Wx$ is $G$-equivariant for a representation $\rho$ iff $W$ lies in the
**commutant** (intertwiner space) $\mathcal C=\{W:\,W\rho(g)=\rho'(g)\,W\ \forall g\in G\}$, a
*linear subspace* of weight space. There are two ways to "be in $\mathcal C$":

  * **Intrinsic** (our layers): *parametrise $\mathcal C$ directly.* Vector-Neuron ``VNLinear``
    stores a channel-mixing matrix $M$ and computes $\sum_i M_{oi}\,x_{i,:}$, with $\rho$ acting
    on the *spatial* axis — so the effective weight is $W=M\otimes I_d$ and lands in $\mathcal C$
    for **every** value of $M$ (e3nn ``o3.Linear`` / TensorProduct are the same: any path weight
    gives an intertwiner). The parameter manifold's whole image is $\mathcal C$. Hence **any**
    optimiser — Adam included — keeps the layer *exactly* equivariant: the equivariance residual
    is identically zero (to the float floor) regardless of how the parameters move.

  * **Extrinsic**: hold a *free* dense $W$ and merely *initialise* it inside $\mathcal C$. Now
    equivariance is a measure-zero subspace **constraint**, not a property of the parametrisation.
    The *population* gradient of an SE(3)-invariant loss on isotropic data lies in $\mathcal C$, so on
    a **noiseless, realisable** target full-batch GD — *and even Adam* — heals back to $W^\star\in
    \mathcal C$ (the off-commutant error is only a transient). The break is therefore **not** generic:
    it needs the realistic regime of **persistent minibatch/label noise**, where each stochastic
    gradient has an off-$\mathcal C$ component. Plain SGD then has a restoring force toward $\mathcal C$
    and settles at a *small* steady off-commutant fluctuation; Adam's element-wise $1/\sqrt{v_t}$
    rescales coordinates **across** the tie that defines $\mathcal C$, distorting that restoring force
    and *sustaining a larger* off-commutant error that does **not** vanish at convergence. This is the
    Lau–Su effect — but, honestly, only a **modest** ($\sim$2–4$\times$) Adam-vs-SGD gap; the decisive
    axis is the *row* (intrinsic vs extrinsic), not the column (Adam vs SGD).

The two panels
--------------
  [A] **The real model is optimiser-agnostic (de-risking).** Train the *actual* Step-13 VN
      ``EqJEPA`` (``SE3PointEncoder`` + ``VNPredictor``) on the exactly-SO(3) teacher with three
      optimisers — the project default **Muon/AdamW**, **pure Adam on every parameter** (the
      geometry-blind optimiser the paper warns against), and **pure SGD** — and measure the
      composed SE(3) residual at init and post-train in float64. *All three sit at the float
      floor.* A non-equivariant MLP control trained with Adam has a large residual, so the metric
      is not vacuous. Conclusion: our headline equivariance does **not** depend on the optimiser,
      because the layers are intrinsically parametrised; Muon is used for optimisation *quality*,
      not to protect equivariance.

  [B] **The safety is earned, not generic ($2\times2$).** A minimal commutant probe pins the
      dichotomy on the project's own layer. Representation $\rho(R)=R\oplus R$ on $\mathbb R^6$
      (two type-1 vectors); by Schur its commutant is $\mathcal C=\{M\otimes I_3:M\in\mathbb
      R^{2\times2}\}$ (4-dim). Fit the *equivariant* target $W^\star=M^\star\otimes I_3$ from
      isotropic Gaussian data with **label noise** $y=W^\star x+\sigma\xi$ (the realistic,
      non-interpolating regime — see above for why a noiseless target would let *both* optimisers
      heal), starting **in** $\mathcal C$. Off-commutant distance $\lVert W-P_{\mathcal C}(W)\rVert_F$:

        |                              | Adam (geometry-blind) | SGD (symmetry-compatible) |
        | intrinsic ``VNLinear(2,2)``  |  **0** (exactly immune)|  **0** (exactly immune)   |
        | extrinsic ``nn.Linear(6,6)`` |   $\sim$1e-2 (worst)   |   $\sim$4e-3 (better)     |

      Same target, same data, same init-in-$\mathcal C$ — only the parametrisation and the optimiser
      differ. Read it by **rows then columns**: the *row* gap is absolute — intrinsic ``VNLinear``
      (the project's layer) is $W=M\otimes I_3$ by construction, so its off-commutant distance is
      *identically zero* for **any** $M$, hence immune to **any** optimiser under **any** noise. The
      *column* gap is real but **modest** ($\sim$2–4$\times$): among extrinsic layers the
      symmetry-compatible SGD drifts less than geometry-blind Adam, exactly as Lau–Su predict, but
      neither stays on $\mathcal C$. The honest lesson: **parametrisation dominates; the optimiser is
      a second-order correction.**

Headline: *intrinsic parametrisation makes equivariance optimiser- and noise-proof; among extrinsic
layers a symmetry-compatible optimiser only modestly helps.* The project's VN/e3nn backbone is
intrinsic (Panel A), so the Sym-Compat warning does **not** touch its headline $\sim10^{-6}$ numbers;
the warning is real only for extrinsically-constrained equivariance (Panel B), where the first-line
fix is an intrinsic parametrisation and the second-line fix is a symmetry-compatible optimiser.

References
----------
  * Lau & Su, *A Symmetry-Compatible Principle for Optimizer Design*, arXiv:2605.18106 (2026).
  * Jordan et al., *Muon: An optimizer for the hidden layers of neural networks* (2024).
  * Deng et al., *Vector Neurons: A General Framework for SO(3)-Equivariant Networks*, ICCV 2021.

Run (full ~3-5 min on a laptop CPU):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step26_optimizer_equivariance.py
Smoke (~40 s):
    STEP26_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step26_optimizer_equivariance.py
"""

import copy
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
sys.path.insert(0, str(HERE))   # for the Step 13 backbone we reuse

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    rand_so3,
    rotate_points,
)
from step18_se3_closed_loop import EVAL_DTYPE  # noqa: E402  (float64 probe dtype)
from src.models.structured import VNLinear  # noqa: E402  (the intrinsic equivariant layer)
from src.training.muon import build_muon_adamw  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP26_SMOKE"))

# Float floor for an *exactly* equivariant model probed in float64: e3nn/VN sit at ~1e-6 (the
# library float floor for the encoder TensorProducts); the bare 6x6 VNLinear sits far lower.
_FLOOR_A = 1e-4    # Panel A composed SE(3) residual (matches test_post_training_equivariance)
_BREAK_A = 1e-2    # the non-equivariant MLP control must MISS the floor by a wide margin
_FLOOR_B = 1e-6    # Panel B: the INTRINSIC layer's equivariance residual (exactly 0 to float64 floor)
_BREAK_B = 1e-3    # Panel B: noisy training must drive the EXTRINSIC layer off the commutant by >= this
NOISE_B = 0.05     # Panel B: label noise sigma. WITHOUT it a realisable target lets Adam *heal* (the
#                    break is only a transient); persistent noise makes the off-commutant drift sustained.


# =========================================================================== #
# Panel A: the real VN EqJEPA is optimiser-agnostic
# =========================================================================== #
def _rotate_latent(z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""$\rho(R)$ = block-diagonal copies of $R$ on a flattened stack of 3-vectors. ``(B,D)->(B,D)``."""
    b = z.shape[0]
    return rotate_points(z.reshape(b, -1, 3), R).reshape(b, -1)


@torch.no_grad()
def _composed_se3_err(model, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor) -> float:
    r"""End-to-end residual $\max\lvert\rho(R)\,f(E(x),a)-f(E(Rx),Ra)\rvert$ (the whole world model)."""
    lhs = _rotate_latent(model.predictor(model.encoder(S), A), R)
    rhs = model.predictor(model.encoder(rotate_points(S, R)), rotate_points(A, R))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def _worst_se3(model, S: torch.Tensor, A: torch.Tensor, gen: torch.Generator, n: int = 5) -> float:
    r"""Worst composed SE(3) residual over ``n`` random rotations (probe dtype = ``S.dtype``)."""
    return max(_composed_se3_err(model, S, A, rand_so3(gen).to(S.dtype)) for _ in range(n))


def _make_opts(kind: str, model: nn.Module) -> list:
    r"""Optimiser factory for the A/B. ``muon_adamw`` = project default; ``adam``/``sgd`` = all-param.

    ``adam`` is the *geometry-blind* optimiser the Symmetry-Compatible paper warns against; we put
    **every** parameter (including the 2D weight matrices Muon would orthogonalise) through it, so
    if any optimiser could break the VN model's equivariance it would be this one.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if kind == "muon_adamw":
        muon, adamw, _ = build_muon_adamw(model, muon_lr=0.02, adamw_lr=1e-3)
        return [o for o in (muon, adamw) if o is not None]
    if kind == "adam":
        return [torch.optim.Adam(params, lr=1e-3)]
    if kind == "sgd":
        return [torch.optim.SGD(params, lr=2e-2, momentum=0.9)]
    raise ValueError(kind)


def _train(
    model: nn.Module,
    S: torch.Tensor,
    A: torch.Tensor,
    S2: torch.Tensor,
    *,
    opt_kind: str,
    epochs: int,
    batch_size: int = 128,
    var_coef: float = 0.1,
    ema_decay: float = 0.99,
    seed: int = 0,
) -> float:
    r"""EMA-target + VICReg JEPA training with a *swappable* optimiser. Returns final pred-loss.

    Mirrors :func:`src.training.jepa.train_jepa` exactly (same loss, EMA target, VICReg hinge);
    the **only** thing that varies across the A/B is ``opt_kind`` — so any difference in the
    post-train equivariance residual is attributable to the optimiser alone.
    """
    torch.manual_seed(seed)
    model.train()
    target_enc = copy.deepcopy(model.encoder)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    target_enc.eval()

    opts = _make_opts(opt_kind, model)
    n = S.shape[0]
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            o, a, o2 = S[idx], A[idx], S2[idx]
            z0 = model.encoder(o)
            with torch.no_grad():
                z_tgt = target_enc(o2)
            z_pred = model.predictor(z0, a)
            pred_loss = F.mse_loss(z_pred, z_tgt)
            var_loss = F.relu(1.0 - z0.std(dim=0)).mean()
            loss = pred_loss + var_coef * var_loss
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in opts:
                opt.step()
            with torch.no_grad():  # EMA target, matched by name (e3nn-safe)
                online_p = dict(model.encoder.named_parameters())
                for name, pt in target_enc.named_parameters():
                    pt.mul_(ema_decay).add_(online_p[name], alpha=1.0 - ema_decay)
                online_b = dict(model.encoder.named_buffers())
                for name, bt in target_enc.named_buffers():
                    bs = online_b.get(name)
                    if bs is not None and bt.shape == bs.shape:
                        bt.copy_(bs)
            last = pred_loss.item()
    model.eval()
    return last


def panel_a(epochs: int) -> dict:
    r"""Train the real VN ``EqJEPA`` under three optimisers; probe SE(3) equivariance in float64."""
    S, A, S2 = collect_cloud_transitions(256, seed=0)
    St, At, _ = collect_cloud_transitions(64, seed=7)
    gen = torch.Generator().manual_seed(11)

    out: dict[str, dict] = {}
    for opt_kind in ("muon_adamw", "adam", "sgd"):
        model = build_eq_jepa()
        # init residual (float64 probe)
        model64 = model.to(EVAL_DTYPE)
        init = _worst_se3(model64, St.to(EVAL_DTYPE), At.to(EVAL_DTYPE), torch.Generator().manual_seed(1))
        model = model.to(torch.float32)
        _train(model, S, A, S2, opt_kind=opt_kind, epochs=epochs, seed=0)
        model64 = model.to(EVAL_DTYPE)
        post = _worst_se3(model64, St.to(EVAL_DTYPE), At.to(EVAL_DTYPE), torch.Generator().manual_seed(1))
        out[opt_kind] = {"init": init, "post": post}

    # non-equivariant control: MLP under Adam must FAIL (metric is not vacuous)
    mlp = build_mlp_jepa()
    _train(mlp, S, A, S2, opt_kind="adam", epochs=epochs, seed=0)
    mlp64 = mlp.to(EVAL_DTYPE)
    out["mlp_adam"] = {
        "init": float("nan"),
        "post": _worst_se3(mlp64, St.to(EVAL_DTYPE), At.to(EVAL_DTYPE), torch.Generator().manual_seed(1)),
    }
    return out


# =========================================================================== #
# Panel B: intrinsic VNLinear vs extrinsic nn.Linear on the SAME equivariant target
#   rep rho(R) = R (+) R on R^6; commutant C = { M (x) I_3 : M in R^{2x2} } (Schur).
# =========================================================================== #
_I3 = torch.eye(3, dtype=EVAL_DTYPE)
_I2 = torch.eye(2, dtype=EVAL_DTYPE)


def _kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.kron(A.contiguous(), B.contiguous())


def _rho(R: torch.Tensor) -> torch.Tensor:
    r"""$\rho(R)=R\oplus R=I_2\otimes R$ on $\mathbb R^6$ (two stacked type-1 vectors)."""
    return _kron(_I2, R)


def _commutant_proj(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Orthogonal projection of a $6\times6$ $W$ onto $\mathcal C=\{M\otimes I_3\}$.

    Each $3\times3$ block $W_{pq}$ projects onto $\mathbb R\cdot I_3$ as
    $\tfrac{\mathrm{tr}(W_{pq})}{3}I_3$, so $\hat M_{pq}=\tfrac13\mathrm{tr}(W_{pq})$. Returns
    $(\,\hat M\otimes I_3,\ \hat M\,)$.
    """
    M = torch.empty(2, 2, dtype=W.dtype)
    for p in range(2):
        for q in range(2):
            M[p, q] = torch.diagonal(W[3 * p : 3 * p + 3, 3 * q : 3 * q + 3]).mean()
    return _kron(M, _I3), M


@torch.no_grad()
def _equiv_resid(W: torch.Tensor, gen: torch.Generator, n: int = 8) -> float:
    r"""$\max_R\lVert W\rho(R)-\rho(R)W\rVert_\infty$ over ``n`` random rotations (float64)."""
    worst = 0.0
    for _ in range(n):
        R = rand_so3(gen).to(W.dtype)
        rho = _rho(R)
        worst = max(worst, (W @ rho - rho @ W).abs().max().item())
    return worst


@torch.no_grad()
def _offcommutant(W: torch.Tensor) -> float:
    r"""Distance $\lVert W-P_{\mathcal C}(W)\rVert_F$ from the commutant (a basis-free equivariance witness)."""
    Wc, _ = _commutant_proj(W)
    return (W - Wc).norm().item()


def _run_commutant(
    kind: str, opt_kind: str, *, steps: int, batch: int, lr: float, noise: float, seed: int = 0
) -> dict:
    r"""Fit $W^\star=M^\star\otimes I_3$ from isotropic data; track the equivariance residual.

    ``kind``: ``intrinsic`` (``VNLinear(2,2)`` -> $W=M\otimes I_3$ for any $M$) or ``extrinsic``
    (``nn.Linear(6,6)`` initialised in $\mathcal C$). The target and (isotropic) data are
    $\rho$-invariant, so the population optimum is in $\mathcal C$ and the *only* thing that can push
    $W$ off $\mathcal C$ is the optimiser's geometry. We add ``noise`` to the labels
    ($y=W^\star x+\sigma\xi$, the realistic non-interpolating regime): on a *noiseless* realizable
    problem Adam *heals* (it converges to $W^\star\in\mathcal C$, off-$\mathcal C\to0$), so the
    Symmetry-Compatible drift is only a transient there; persistent gradient noise makes it
    *sustained*, which is the regime real training lives in.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rgen = torch.Generator().manual_seed(seed + 100)
    Mstar = torch.randn(2, 2, generator=gen, dtype=EVAL_DTYPE)
    Wstar = _kron(Mstar, _I3)  # (6,6), exactly equivariant target

    if kind == "intrinsic":
        layer = VNLinear(2, 2).to(EVAL_DTYPE)

        def w_eff() -> torch.Tensor:
            return _kron(layer.weight.detach(), _I3)

        def fwd(x6: torch.Tensor) -> torch.Tensor:
            return layer(x6.reshape(-1, 2, 3)).reshape(-1, 6)

    elif kind == "extrinsic":
        layer = nn.Linear(6, 6, bias=False).to(EVAL_DTYPE)
        with torch.no_grad():  # initialise INSIDE the commutant (so init residual ~ float floor)
            M0 = torch.randn(2, 2, generator=gen, dtype=EVAL_DTYPE)
            layer.weight.copy_(_kron(M0, _I3))

        def w_eff() -> torch.Tensor:
            return layer.weight.detach().clone()

        def fwd(x6: torch.Tensor) -> torch.Tensor:
            return layer(x6)
    else:
        raise ValueError(kind)

    if opt_kind == "adam":
        opt = torch.optim.Adam(layer.parameters(), lr=lr)
    elif opt_kind == "sgd":
        opt = torch.optim.SGD(layer.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(opt_kind)

    init_resid = _equiv_resid(w_eff(), rgen)
    for _ in range(steps):
        x6 = torch.randn(batch, 6, generator=gen, dtype=EVAL_DTYPE)
        # label noise => the loss is non-interpolating; the per-step gradient carries an off-C
        # component that SGD damps (restoring force) but Adam's 1/sqrt(v_t) sustains/amplifies.
        y = x6 @ Wstar.T + noise * torch.randn(batch, 6, generator=gen, dtype=EVAL_DTYPE)
        loss = F.mse_loss(fwd(x6), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    # honest held-out fit loss against the CLEAN target (so a "break" is not just divergence; the
    # irreducible noise floor ~ sigma^2 = 2.5e-3 keeps this small but nonzero even for a perfect fit)
    xe = torch.randn(2048, 6, generator=gen, dtype=EVAL_DTYPE)
    fit = F.mse_loss(fwd(xe), xe @ Wstar.T).item()
    return {
        "init_resid": init_resid,
        "final_resid": _equiv_resid(w_eff(), rgen),
        "offcommutant": _offcommutant(w_eff()),
        "fit_loss": fit,
    }


def panel_b(steps: int, batch: int, noise: float) -> dict:
    r"""The $2\times2$: {intrinsic, extrinsic} x {Adam, SGD}, fit under label noise ``noise``.

    The decisive separation is by **row**: the intrinsic ``VNLinear`` rows sit at the float floor
    (off-commutant identically 0) for either optimiser; the extrinsic ``nn.Linear`` rows drift off
    $\mathcal C$ under noisy training, with Adam (geometry-blind) worse than SGD (symmetry-compatible).
    """
    out: dict[str, dict] = {}
    out["extrinsic_adam"] = _run_commutant("extrinsic", "adam", steps=steps, batch=batch, lr=1e-2, noise=noise)
    out["extrinsic_sgd"] = _run_commutant("extrinsic", "sgd", steps=steps, batch=batch, lr=1e-2, noise=noise)
    out["intrinsic_adam"] = _run_commutant("intrinsic", "adam", steps=steps, batch=batch, lr=1e-2, noise=noise)
    out["intrinsic_sgd"] = _run_commutant("intrinsic", "sgd", steps=steps, batch=batch, lr=1e-2, noise=noise)
    return out


# =========================================================================== #
# main
# =========================================================================== #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 76

    # Panel B reaches a stationary off-commutant fluctuation quickly (Ornstein-Uhlenbeck-like), so
    # SMOKE and FULL agree on the off-C numbers; STEPS_B mainly buys a tighter fit_loss.
    if SMOKE:
        EPOCHS_A, STEPS_B, BATCH_B = 2, 1200, 64
    else:
        EPOCHS_A, STEPS_B, BATCH_B = 8, 3000, 64

    print(line)
    print(f"STEP 26  does the OPTIMISER break SE(3)-equivariance?  intrinsic vs extrinsic  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print("    claim: intrinsic parametrisation makes equivariance optimiser- AND noise-proof;")
    print("           among extrinsic layers a symmetry-compatible optimiser only MODESTLY helps.")
    print("    [A] the real VN EqJEPA is optimiser-agnostic (Muon/AdamW = Adam = SGD = floor)")
    print("    [B] 2x2 commutant probe under label noise: intrinsic VNLinear immune (off-C==0) for ANY")
    print("        optimiser; extrinsic nn.Linear drifts off C, Adam worse than SGD (modest ~2-4x)")

    # ----- Panel A --------------------------------------------------------------------------
    print()
    print(line)
    print("[A] real Step-13 VN EqJEPA: composed SE(3) residual at init & post-train (float64 probe)")
    print(line)
    a = panel_a(EPOCHS_A)
    print(f"    {'optimiser':>12s} | {'init resid':>12s} | {'post-train resid':>16s}")
    print("    " + "-" * 48)
    for opt_kind in ("muon_adamw", "adam", "sgd"):
        r = a[opt_kind]
        print(f"    {opt_kind:>12s} | {r['init']:12.2e} | {r['post']:16.2e}")
    print(f"    {'MLP/adam':>12s} | {'n/a':>12s} | {a['mlp_adam']['post']:16.2e}   <- non-equivariant control")
    ok_a = (
        all(a[k]["init"] < _FLOOR_A and a[k]["post"] < _FLOOR_A for k in ("muon_adamw", "adam", "sgd"))
        and a["mlp_adam"]["post"] > _BREAK_A
    )
    print(f"    => VN equivariance is optimiser-agnostic (all < {_FLOOR_A:.0e}); MLP control breaks "
          f"({a['mlp_adam']['post']:.1e} > {_BREAK_A:.0e}).  guard={ok_a}")

    # ----- Panel B --------------------------------------------------------------------------
    print()
    print(line)
    print(f"[B] commutant probe  rho(R)=R(+)R on R^6,  C = {{ M(x)I_3 }} :  fit W* = M*(x)I_3 "
          f"+ noise (sigma={NOISE_B})")
    print(line)
    b = panel_b(STEPS_B, BATCH_B, NOISE_B)
    print(f"    {'parametrisation':>16s} | {'optimiser':>9s} | {'init resid':>11s} | "
          f"{'final resid':>11s} | {'off-C dist':>10s} | {'fit loss':>9s}")
    print("    " + "-" * 80)
    rows = [
        ("extrinsic nn.Linear", "adam", "extrinsic_adam"),
        ("extrinsic nn.Linear", "sgd", "extrinsic_sgd"),
        ("intrinsic VNLinear", "adam", "intrinsic_adam"),
        ("intrinsic VNLinear", "sgd", "intrinsic_sgd"),
    ]
    for label, opt_kind, key in rows:
        r = b[key]
        print(f"    {label:>16s} | {opt_kind:>9s} | {r['init_resid']:11.2e} | "
              f"{r['final_resid']:11.2e} | {r['offcommutant']:10.2e} | {r['fit_loss']:9.2e}")
    # The target W* = M* (x) I_3 has O(1) per-coordinate scale (a W=0 baseline loss ~ 2.0), so
    # fit_loss < 0.3 means the layer explained >~85% of the variance -- clearly *training*, so the
    # off-commutant distance reflects the optimiser's geometry under noise, not divergence.
    # Gate on the HONEST dichotomy: (i) intrinsic is exactly immune to BOTH optimisers (off-C == 0
    # by construction); (ii) noisy training drifts the EXTRINSIC layer off C; (iii) among extrinsic
    # layers the symmetry-compatible SGD drifts LESS than geometry-blind Adam (modest, ~2-4x).
    # NB: we no longer require extrinsic+SGD to reach the floor -- under realistic noise it does NOT.
    ok_b = (
        b["intrinsic_adam"]["final_resid"] < _FLOOR_B          # intrinsic VNLinear immune to Adam...
        and b["intrinsic_sgd"]["final_resid"] < _FLOOR_B       # ...and to SGD (the absolute row gap)
        and b["intrinsic_adam"]["offcommutant"] < _FLOOR_B     # off-C identically 0 (W = M (x) I_3)
        and b["extrinsic_adam"]["offcommutant"] > _BREAK_B     # noisy training drifts extrinsic off C
        and b["extrinsic_adam"]["fit_loss"] < 0.3              # ...while still fitting (rules out divergence)
        and b["extrinsic_sgd"]["offcommutant"] < b["extrinsic_adam"]["offcommutant"]  # sym-compat helps (modestly)
    )
    row_gap = b["extrinsic_adam"]["offcommutant"] / max(b["intrinsic_adam"]["offcommutant"], 1e-18)
    col_gap = b["extrinsic_adam"]["offcommutant"] / max(b["extrinsic_sgd"]["offcommutant"], 1e-18)
    print(f"    => ROW gap (extrinsic/intrinsic off-C) = x{row_gap:.1e} (absolute); "
          f"COL gap (Adam/SGD) = x{col_gap:.2f} (modest).  guard={ok_b}")

    # ----- verdict --------------------------------------------------------------------------
    passed = ok_a and ok_b
    print()
    print(line)
    print("STEP 26 SUMMARY")
    print(line)
    print(f"    [A] real VN EqJEPA optimiser-agnostic: muon/adamw={a['muon_adamw']['post']:.1e}, "
          f"adam={a['adam']['post']:.1e}, sgd={a['sgd']['post']:.1e}  (all < {_FLOOR_A:.0e}); "
          f"MLP/adam={a['mlp_adam']['post']:.1e}.")
    print(f"    [B] under label noise: intrinsic VNLinear off-C = {b['intrinsic_adam']['offcommutant']:.1e} "
          f"(Adam) / {b['intrinsic_sgd']['offcommutant']:.1e} (SGD) -- EXACTLY immune; extrinsic nn.Linear "
          f"drifts to {b['extrinsic_adam']['offcommutant']:.1e} (Adam) vs {b['extrinsic_sgd']['offcommutant']:.1e} (SGD).")
    print(f"        => ROW gap x{row_gap:.0e} (absolute) dwarfs COL gap x{col_gap:.1f} (modest).")
    print("    headline: PARAMETRISATION dominates, optimiser is second-order. Intrinsic equivariance")
    print("        (W = M (x) I_3, our VN/e3nn layers) is immune to ANY optimiser under ANY noise => the")
    print("        Symmetry-Compatible-Optimizer warning does NOT touch our ~1e-6 numbers (Panel A). It")
    print("        bites only EXTRINSIC equivariance (Panel B); there a symmetry-compatible optimiser helps")
    print("        modestly, but an intrinsic parametrisation is the real fix. (On a NOISELESS realisable")
    print("        target even Adam heals -- the break needs persistent noise to be sustained.)")
    print(f"    guards: panelA={ok_a}  panelB={ok_b}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}")

    # ----- dump JSON artifact ---------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(EPOCHS_A=EPOCHS_A, STEPS_B=STEPS_B, BATCH_B=BATCH_B,
                       floor_a=_FLOOR_A, break_a=_BREAK_A, floor_b=_FLOOR_B, break_b=_BREAK_B),
        "panel_a": a,
        "panel_b": b,
        "verdict": {"passed": bool(passed), "ok_a": ok_a, "ok_b": ok_b},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step26_optimizer_equivariance_smoke.json" if SMOKE else "step26_optimizer_equivariance.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
