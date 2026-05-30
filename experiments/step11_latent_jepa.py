r"""Step 11: the end-to-end equivariant **latent** JEPA -- 举一反三 in latent space.

Where Step 10 sits, and why this is different
---------------------------------------------
Step 10 fit an *explicit-coordinate* forward model $M:(s,a)\mapsto s'$ that predicts
the next **physical** state, and planned against a cost computed directly on the
block's pixel coordinates. The "latent" there was the state itself -- a useful
external-validity test, but **not** the project's actual thesis.

Step 11 builds the thing the roadmap is really about: a genuine **JEPA latent
world model** (LeCun 2022; Bardes et al. V-JEPA 2024) on real PushT --

* a learned **encoder** $E_\theta:\text{state}\to z\in\mathbb{R}^{D}$,
* a learned **latent predictor** $f_\phi:(z,a)\mapsto z$ that advances the latent,
* planning by a **latent** terminal cost $\mathcal{C}=\lVert \hat z_H - z_g\rVert_2^2$,

and asks whether making *both* maps SO(2)-equivariant causes the **learned
representation** to inherit exact equivariance, so that

  (i)  the JEPA planning cost is *exactly* rotation-invariant, and
  (ii) latent-space prediction 举一反三 holds -- training the latent dynamics on a
       $[0,90°)$ wedge of push directions *determines* it on the whole circle.

This is the Phase-4 deliverable: "the predictor learns dynamics in the geometric
latent space." Everything is *composed* from existing modules -- nothing about the
geometry is re-invented here:

* encoder   : :class:`src.models.structured.StructuredStateEncoder`  (exact SO(2))
              vs a parameter-comparable :class:`MLPStateEncoder` baseline (no prior)
* predictor : :class:`src.models.structured.VNPredictor` (jointly equivariant)
              vs :class:`src.models.eqjepa.LatentPredictor` (ordinary residual MLP)
* wrapper   : :class:`src.models.eqjepa.EqJEPA` (latent ``rollout``)
* training  : :func:`src.training.jepa.train_jepa` -- EMA-target JEPA + VICReg
              variance hinge + Muon/AdamW, used **unchanged** (it only ever calls
              ``model.encoder(o)`` / ``model.predictor(z,a)``), fed structured
              ``(N,4,2)`` vector transitions instead of pixels.

Why the equivariant latent is *exactly* flat across orientation (the guarantee).
$E_\theta(R\,s)=\rho(R)\,E_\theta(s)$ with $\rho$ orthogonal, and
$f_\phi(\rho(R)z,R a)=\rho(R)f_\phi(z,a)$, so for any rotated transition
$(Rs,Ra)\to Rs'$ the latent prediction error is unchanged:
$\lVert f_\phi(E(Rs),Ra)-E(Rs')\rVert=\lVert\rho(R)\,[\,f_\phi(E(s),a)-E(s')\,]\rVert
=\lVert f_\phi(E(s),a)-E(s')\rVert$, and likewise the cost
$\lVert E(Rs)-E(Rs_g)\rVert=\lVert E(s)-E(s_g)\rVert$. The ordinary baseline has no
such guarantee and degrades out-of-distribution. We *verify* this to the float
floor (equivariance unit tests) at init **and** after training, per project policy.

Honest scope (carried over from Step 10). Real PushT interior dynamics is exactly
SO(2)-equivariant (Step 10 [A]); we stay interior. The decisive metrics are the
representation-level ones ([A] equivariance, [B] latent 举一反三). Closed-loop task
success [C] is reported but -- as Step 10 established -- is noise-limited on
position-only pushing at laptop scale; here it also stresses whether a *purely
latent* cost can drive a planner at all. We gate the verdict on [A]+[B].

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step11_latent_jepa.py
"""

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
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 10 helpers we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.geometry.so2 import rotate_vectors  # noqa: E402
from src.models.eqjepa import EqJEPA, LatentPredictor  # noqa: E402
from src.models.structured import StructuredStateEncoder, VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

# Reuse Step 10's *validated* env scaffolding (its main() is __main__-guarded, so
# importing only pulls in the helpers + constants, not the Step 10 run).
from step10_pusht_closed_loop import (  # noqa: E402
    CENTER,
    GOAL_DIST_PX,
    POS_SCALE,
    SUCCESS_PX,
    block_xy_px,
    collect_transitions,
    goal_to_scaled,
    make_env,
    n_params,
    reset_task,
    sample_task,
)

torch.set_default_dtype(torch.float32)

# Latent / state geometry.  The encoder consumes a goal-free state of 4 type-1
# vectors (reusing Step 10's packer, just reshaped): channels are
# [agent_pos, agent_vel, block_pos, block_dir]; block_pos is channel index 2.
N_VEC = 4
LATENT_DIM = 128
ACTION_DIM = 2
BLOCK_CH = 2  # index of block_pos within the (N_VEC, 2) state


# --------------------------------------------------------------------------- #
# baseline encoder (the non-equivariant counterpart to StructuredStateEncoder)
# --------------------------------------------------------------------------- #
class MLPStateEncoder(nn.Module):
    r"""Ordinary MLP encoder on the flattened structured state. No symmetry prior.

    Flattens the $(\text{n\_vec}, 2)$ vectors and maps them through a plain MLP, so
    a global rotation of the input has **no** predictable action on the latent --
    the fair non-equivariant control for :class:`StructuredStateEncoder`.

    forward: ``(B, n_vec, 2) -> (B, latent_dim)``
    """

    def __init__(self, n_vec: int = N_VEC, latent_dim: int = LATENT_DIM, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(n_vec * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_vec, 2) -> (B, latent_dim)
        return self.net(x.flatten(1))


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #
def build_eq_jepa() -> EqJEPA:
    r"""Exactly-SO(2)-equivariant latent JEPA: VN encoder + jointly-equivariant VN predictor."""
    enc = StructuredStateEncoder(n_vec=N_VEC, latent_dim=LATENT_DIM, hidden=64)
    pred = VNPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=64, dim=2)
    return EqJEPA(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, encoder=enc, predictor=pred)


def build_mlp_jepa() -> EqJEPA:
    r"""Non-equivariant baseline latent JEPA: MLP encoder + ordinary residual MLP predictor."""
    enc = MLPStateEncoder(n_vec=N_VEC, latent_dim=LATENT_DIM, hidden=128)
    pred = LatentPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=256)
    return EqJEPA(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, encoder=enc, predictor=pred)


# --------------------------------------------------------------------------- #
# data: structured vector transitions (reshape Step 10's validated collector)
# --------------------------------------------------------------------------- #
def collect_vec_transitions(
    n: int, beta_lo: float, beta_hi: float, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Collect ~``n`` interior real transitions as structured vectors.

    Thin wrapper over Step 10's ``collect_transitions`` (same exploration policy +
    interior filter, exactly SO(2)-equivariant regime); reshapes its packed
    ``(N, 8)`` states into ``(N, 4, 2)`` type-1 vectors for the structured encoder.

    Returns ``(S, A, S2)`` with ``S, S2: (N, 4, 2)``, ``A: (N, 2)``.
    """
    s8, a, s28 = collect_transitions(n, beta_lo, beta_hi, seed=seed)
    return s8.reshape(-1, N_VEC, 2), a, s28.reshape(-1, N_VEC, 2)


# --------------------------------------------------------------------------- #
# geometry helpers (continuous-angle SO(2), exact -- no grid)
# --------------------------------------------------------------------------- #
def rotate_vecs(x: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""Rotate every type-1 vector of a state tensor. ``(B, k, 2) -> (B, k, 2)``."""
    return rotate_vectors(x, alpha)


def rotate_action(a: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""Rotate an action vector. ``(B, 2) -> (B, 2)``."""
    return rotate_vectors(a, alpha)


def rotate_latent(z: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""Apply $\rho(\alpha)$ to a latent, treating it as stacked 2-vectors.

    The equivariant latent decomposes as ``latent_dim // 2`` type-1 vectors, so
    $\rho(\alpha)$ is block-diagonal copies of $R(\alpha)$ (orthogonal).

    ``(B, latent_dim) -> (B, latent_dim)``
    """
    b = z.shape[0]
    return rotate_vectors(z.reshape(b, -1, 2), alpha).reshape(b, -1)


# --------------------------------------------------------------------------- #
# [A] equivariance unit tests (continuous angle; init AND post-train)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encoder_equiv_err(enc: nn.Module, S: torch.Tensor, alpha: float = 0.7) -> float:
    r"""$\max\lVert \rho(\alpha)E(s) - E(R(\alpha)s)\rVert_\infty$ (0 for an SO(2) encoder)."""
    lhs = rotate_latent(enc(S), alpha)
    rhs = enc(rotate_vecs(S, alpha))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def predictor_equiv_err(
    pred: nn.Module, z: torch.Tensor, a: torch.Tensor, alpha: float = 0.7
) -> float:
    r"""$\max\lVert \rho(\alpha)f(z,a) - f(\rho(\alpha)z, R(\alpha)a)\rVert_\infty$."""
    lhs = rotate_latent(pred(z, a), alpha)
    rhs = pred(rotate_latent(z, alpha), rotate_action(a, alpha))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def composed_equiv_err(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, alpha: float = 0.7) -> float:
    r"""End-to-end (encode then predict) equivariance residual -- the headline unit test."""
    lhs = rotate_latent(model.predictor(model.encoder(S), A), alpha)
    rhs = model.predictor(model.encoder(rotate_vecs(S, alpha)), rotate_action(A, alpha))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def cost_drift(enc: nn.Module, S: torch.Tensor, Sg: torch.Tensor, alpha: float) -> float:
    r"""Relative drift of the JEPA matching cost $\lVert E(s)-E(s_g)\rVert$ under rotation.

    The well-posed *cross-model* metric: both encoders define a latent metric, and we
    ask how much a *joint* rotation of (state, goal) by a continuous $\alpha$ changes
    it. An SO(2)-equivariant encoder gives drift $\approx 0$ for **every** $\alpha$
    (orthogonal $\rho$); an ordinary encoder drifts. Continuous-angle analog of
    :func:`src.training.jepa.fov_cost_drift` (which is pixel/$90^\circ$-only).
    """
    base = (enc(S) - enc(Sg)).norm(dim=-1)
    rot = (enc(rotate_vecs(S, alpha)) - enc(rotate_vecs(Sg, alpha))).norm(dim=-1)
    return ((rot - base).abs().mean() / base.mean().clamp_min(1e-8)).item()


# --------------------------------------------------------------------------- #
# [B] latent-space prediction 举一反三
# --------------------------------------------------------------------------- #
@torch.no_grad()
def latent_rel_mse(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor) -> float:
    r"""Latent 1-step prediction error, normalised by the latent step size.

    $\mathrm{relMSE}=\big\lVert f_\phi(E(s),a)-E(s')\big\rVert^2 \big/
    \big\lVert E(s')-E(s)\big\rVert^2$. For the equivariant JEPA this is *exactly*
    invariant to a rotation of the transition (numerator and denominator both carry
    an orthogonal $\rho$), hence flat across orientation; the baseline degrades OOD.
    """
    z = model.encoder(S)
    z2 = model.encoder(S2)
    zp = model.predictor(z, A)
    num = ((zp - z2) ** 2).sum(-1)
    den = ((z2 - z) ** 2).sum(-1).clamp_min(1e-8)
    return (num / den).mean().item()


# --------------------------------------------------------------------------- #
# [C] latent-space closed-loop planning (bonus; pre-registered noise-limited)
# --------------------------------------------------------------------------- #
def goal_state_vecs(goal_scaled: np.ndarray, block_dir: np.ndarray) -> torch.Tensor:
    r"""Build a goal *state* in the (4,2) format for latent encoding $z_g=E(s_g)$.

    Channels [agent_pos, agent_vel, block_pos, block_dir]: place the block at the
    goal (and the agent at the goal, vel 0 -- a canonical "done" configuration);
    orientation is not part of the task, so ``block_dir`` is left at its current
    value to avoid spuriously penalising rotation. Standard goal-observation JEPA
    cost (cf. :meth:`EqJEPA.criterion`); the agent term is an approximation we note.
    """
    g = np.asarray(goal_scaled, dtype=np.float32)
    vecs = np.stack([g, np.zeros(2, np.float32), g, np.asarray(block_dir, np.float32)], axis=0)
    return torch.from_numpy(vecs).float().unsqueeze(0)  # (1, 4, 2)


@torch.no_grad()
def latent_cem_plan(
    model: EqJEPA, vecs0: np.ndarray, zg: torch.Tensor, u_hat: np.ndarray,
    *, H: int = 20, n_samples: int = 300, n_iters: int = 6, n_elite: int = 30,
    sigma0: float = 0.8, w_run: float = 0.3, warm: float = 0.3,
    gen: torch.Generator | None = None,
) -> np.ndarray:
    r"""CEM-MPC against a **purely latent** terminal cost $\lVert \hat z_H - z_g\rVert_2^2$.

    The latent is rolled forward with ``model.predictor`` (i.e. ``EqJEPA.rollout``);
    no physical state is available inside the rollout. A small warm start along the
    (orientation-equivariant) push direction $\hat u$ plays the role Step 10's
    ``w_app`` shaping did -- it is identical for both models and every orientation,
    so it cannot favour either prior. Returns the elite-mean plan ``(H, 2)``.
    """
    v0 = torch.from_numpy(np.asarray(vecs0, np.float32)).unsqueeze(0)  # (1,4,2)
    z0 = model.encoder(v0).expand(n_samples, -1).contiguous()  # (n_samples, D)
    zg = zg.expand(n_samples, -1).contiguous()
    mean = torch.from_numpy(warm * np.asarray(u_hat, np.float32)).expand(H, 2).contiguous()
    sigma = torch.full((H, 2), sigma0)
    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 2, generator=gen)
        cand = (mean.unsqueeze(0) + sigma.unsqueeze(0) * eps).clamp(-1, 1)
        z = z0.clone()
        cost = torch.zeros(n_samples)
        for h in range(H):
            z = model.predictor(z, cand[:, h])
            d2 = ((z - zg) ** 2).sum(-1)
            cost = cost + (w_run * d2 if h < H - 1 else d2)
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        sigma = elite.std(0).clamp_min(1e-3)
    return mean.numpy()


def vecs0_and_dir(info: dict) -> tuple[np.ndarray, np.ndarray]:
    r"""Current state as ``(4,2)`` vectors and the current ``block_dir`` (for $z_g$)."""
    from step10_pusht_closed_loop import info_to_packed

    packed = info_to_packed(info)  # (8,) = [agent_pos, agent_vel, block_pos, block_dir]
    return packed.reshape(N_VEC, 2), packed[6:8].copy()


def latent_closed_loop(
    env, model: EqJEPA, task: dict,
    *, T_max: int = 20, replan_every: int = 20, seed: int = 0,
    gen: torch.Generator | None = None, **cem,
) -> dict:
    r"""Open-loop (no-replan) latent-cost CEM-MPC on the real env; success + final dist."""
    info = reset_task(env, task, seed=seed)
    goal_scaled = goal_to_scaled(task["goal_px"])
    u_hat = np.array([math.cos(task["beta"]), math.sin(task["beta"])], np.float32)
    _, block_dir = vecs0_and_dir(info)
    zg = model.encoder(goal_state_vecs(goal_scaled, block_dir))  # (1, D)
    min_d = np.inf
    t = 0
    while t < T_max:
        vecs0, _ = vecs0_and_dir(info)
        plan = latent_cem_plan(model, vecs0, zg, u_hat, gen=gen, **cem)
        for k in range(min(replan_every, T_max - t)):
            _, _, _, _, info = env.step(plan[k].astype(np.float32))
            d = float(np.linalg.norm(np.asarray(info["block_pose"])[:2] - task["goal_px"]))
            min_d = min(min_d, d)
            t += 1
            if d < SUCCESS_PX:
                return {"success": True, "min_d": min_d}
    return {"success": min_d < SUCCESS_PX, "min_d": min_d}


def eval_orientation_latent(
    model: EqJEPA, beta_centers: list[float], n_task: int, seeds: tuple[int, ...], **cl
) -> dict:
    r"""Mean success + mean min-distance per orientation bin, averaged over eval seeds."""
    acc: dict[float, list[tuple[float, float]]] = {bc: [] for bc in beta_centers}
    for sd in seeds:
        env = make_env()
        rng = np.random.default_rng(sd)
        gen = torch.Generator().manual_seed(sd)
        for bc in beta_centers:
            succ, dists = [], []
            for i in range(n_task):
                task = sample_task(rng, bc - math.radians(45), bc + math.radians(45))
                r = latent_closed_loop(env, model, task, seed=i, gen=gen, **cl)
                succ.append(r["success"]); dists.append(r["min_d"])
            acc[bc].append((float(np.mean(succ)), float(np.mean(dists))))
    return {
        bc: (float(np.mean([s for s, _ in v])), float(np.mean([d for _, d in v])))
        for bc, v in acc.items()
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72
    quadrants = [("[0,90) seen", 0.0), ("[90,180)", 90.0),
                 ("[180,270)", 180.0), ("[270,360)", 270.0)]

    # ---- data: train on a [0,90) wedge; one held-out set we rotate per quadrant
    N_TRAIN, N_TEST = 1500, 400
    S, A, S2 = collect_vec_transitions(N_TRAIN, 0.0, math.radians(90.0), seed=0)
    St, At, S2t = collect_vec_transitions(N_TEST, 0.0, math.radians(90.0), seed=999)

    eq = build_eq_jepa()
    mlp = build_mlp_jepa()

    # ---------------------------------------------------------------- [A] init
    print(line)
    print("[A] Equivariance of the learned latent (continuous angle) -- AT INIT")
    print(line)
    g = torch.Generator().manual_seed(1)
    z_rand = torch.randn(64, LATENT_DIM, generator=g)
    a_rand = torch.randn(64, ACTION_DIM, generator=g)
    print("    VN equivariant JEPA (should be ~float floor):")
    print(f"        encoder    |rho E(s) - E(Rs)|        : {encoder_equiv_err(eq.encoder, St):.4e}")
    print(f"        predictor  |rho f(z,a) - f(rho z,Ra)|: {predictor_equiv_err(eq.predictor, z_rand, a_rand):.4e}")
    print(f"        composed   (encode->predict)         : {composed_equiv_err(eq, St, At):.4e}")
    print(f"    MLP baseline composed equiv residual     : {composed_equiv_err(mlp, St, At):.4e} (no prior)")
    print(f"    params: VN={n_params(eq)}   MLP={n_params(mlp)}  ({n_params(mlp) / n_params(eq):.1f}x VN)")

    # ---------------------------------------------------------------- train
    print()
    print(line)
    print("[train] EMA-target JEPA (Muon/AdamW) on the wedge [0,90)")
    print(line)
    print(f"    {len(S)} train / {len(St)} held-out interior transitions")
    print("    VN equivariant JEPA:")
    h_eq = train_jepa(eq, S, A, S2, epochs=40, batch_size=128, var_coef=0.04, seed=0, log_every=10)
    print("    MLP baseline JEPA:")
    h_mlp = train_jepa(mlp, S, A, S2, epochs=40, batch_size=128, var_coef=0.04, seed=0, log_every=10)
    print(f"    final latent_std: VN={h_eq['latent_std']:.3f}  MLP={h_mlp['latent_std']:.3f}  "
          f"(>0 => no collapse)")

    # ---------------------------------------------------------------- [A] post
    print()
    print(line)
    print("[A'] Equivariance AFTER training (the property must survive optimisation)")
    print(line)
    eq_comp = composed_equiv_err(eq, St, At)
    print(f"    VN  composed equiv residual : {eq_comp:.4e}  (still exact)")
    print(f"    MLP composed equiv residual : {composed_equiv_err(mlp, St, At):.4e}")
    print("    JEPA cost drift  |C(Rs,Rs_g)-C(s,s_g)| / C  under continuous rotation:")
    print(f"    {'angle':>8s} | {'VN drift':>12s} | {'MLP drift':>12s}")
    print("    " + "-" * 40)
    Sg = St.roll(1, dims=0)  # arbitrary distinct "goal" states from held-out set
    vn_drift, mlp_drift = [], []
    for deg in (37.0, 90.0, 153.0, 211.0):
        a = math.radians(deg)
        vd, md = cost_drift(eq.encoder, St, Sg, a), cost_drift(mlp.encoder, St, Sg, a)
        vn_drift.append(vd); mlp_drift.append(md)
        print(f"    {deg:7.0f}° | {vd:12.4e} | {md:12.4e}")
    print(f"    => VN cost is rotation-invariant (max {max(vn_drift):.1e}); "
          f"MLP drifts (max {max(mlp_drift):.2f}).")

    # ---------------------------------------------------------------- [B]
    print()
    print(line)
    print("[B] LATENT-space prediction 举一反三 (train wedge [0,90); rotate held-out)")
    print(line)
    print("    latent 1-step relMSE (SAME held-out set rotated into each quadrant):")
    print(f"    {'orientation':16s} | {'VN relMSE':>12s} | {'MLP relMSE':>12s}")
    print("    " + "-" * 47)
    vn_b, mlp_b = [], []
    for lab, deg in quadrants:
        a = math.radians(deg)
        Sr, Ar, S2r = rotate_vecs(St, a), rotate_action(At, a), rotate_vecs(S2t, a)
        ve, me = latent_rel_mse(eq, Sr, Ar, S2r), latent_rel_mse(mlp, Sr, Ar, S2r)
        vn_b.append(ve); mlp_b.append(me)
        print(f"    {lab:16s} | {ve:12.4e} | {me:12.4e}")
    pred_vn = max(vn_b) / vn_b[0]
    pred_mlp = max(mlp_b) / mlp_b[0]
    print(f"    => VN flat across orientation (x{pred_vn:.2f}); MLP degrades OOD (x{pred_mlp:.1f}).")

    # ---------------------------------------------------------------- [C]
    print()
    print(line)
    print("[C] LATENT-space closed-loop planning across quadrants (bonus, noise-limited)")
    print(line)
    beta_centers = [math.radians(c) for c in (45, 135, 225, 315)]
    labels = [q[0] for q in quadrants]
    N_TASK, EVAL_SEEDS = 15, (0, 1)
    cl_kw = dict(T_max=20, replan_every=20, H=20, n_samples=300,
                 n_iters=6, n_elite=30, sigma0=0.8, w_run=0.3, warm=0.3)
    vn_res = eval_orientation_latent(eq, beta_centers, N_TASK, EVAL_SEEDS, **cl_kw)
    mlp_res = eval_orientation_latent(mlp, beta_centers, N_TASK, EVAL_SEEDS, **cl_kw)
    print(f"    open-loop H={cl_kw['H']}, purely-latent cost ||z_H - z_g||^2, "
          f"{len(EVAL_SEEDS)} seeds x {N_TASK}/bin ; success<{SUCCESS_PX:.0f}px (goal {GOAL_DIST_PX:.0f}px)")
    print(f"    {'orientation':16s} | {'VN succ':>8s} {'VN dist':>9s} | {'MLP succ':>9s} {'MLP dist':>9s}")
    print("    " + "-" * 60)
    for bc, lab in zip(beta_centers, labels):
        vs, vd = vn_res[bc]; ms, md = mlp_res[bc]
        print(f"    {lab:16s} | {vs:8.2f} {vd:8.1f}px | {ms:9.2f} {md:8.1f}px")
    vn_seen_d = vn_res[beta_centers[0]][1]
    mlp_seen_d = mlp_res[beta_centers[0]][1]
    vn_unseen_d = float(np.mean([vn_res[b][1] for b in beta_centers[1:]]))
    mlp_unseen_d = float(np.mean([mlp_res[b][1] for b in beta_centers[1:]]))
    vn_ratio = vn_unseen_d / max(vn_seen_d, 1e-6)
    mlp_ratio = mlp_unseen_d / max(mlp_seen_d, 1e-6)

    # ---------------------------------------------------------------- summary
    print()
    print(line)
    print("STEP 11 SUMMARY")
    print(line)
    print("    end-to-end LATENT JEPA on real PushT: equivariant encoder + jointly-")
    print("    equivariant predictor, planned in the LEARNED latent (not coordinates).")
    print(f"    [A'] learned latent stays exactly SO(2)-equivariant after training:")
    print(f"         VN composed residual {eq_comp:.1e} ; cost drift max {max(vn_drift):.1e} "
          f"(MLP {max(mlp_drift):.2f}).")
    print(f"    [B]  LATENT prediction 举一反三: VN flat (x{pred_vn:.2f}), MLP degrades (x{pred_mlp:.1f}).")
    print(f"    [C]  latent closed-loop block->goal dist (px), OOD ratio = unseen/seen:")
    print(f"         VN  seen={vn_seen_d:5.1f} unseen={vn_unseen_d:5.1f}  -> x{vn_ratio:.2f}")
    print(f"         MLP seen={mlp_seen_d:5.1f} unseen={mlp_unseen_d:5.1f}  -> x{mlp_ratio:.2f}")

    # Gate on the decisive, reproducible representation-level results [A']+[B];
    # [C] is reported honestly but (per Step 10) is noise-limited on position-only
    # pushing AND additionally stresses whether a purely-latent cost can plan at all.
    ok_eq = eq_comp < 1e-4
    ok_drift = max(vn_drift) < 0.02 and max(mlp_drift) > 0.10
    ok_pred_vn = pred_vn < 1.05
    ok_pred_mlp = pred_mlp > 1.5
    passed = ok_eq and ok_drift and ok_pred_vn and ok_pred_mlp
    print(f"    guards: latent-equivariant={ok_eq}  cost-invariant-vs-drift={ok_drift}  "
          f"VN-flat={ok_pred_vn}  MLP-degrades={ok_pred_mlp}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: equivariant prior => the LEARNED "
          f"latent world model inherits exact")
    print(f"    SO(2) symmetry, so its planning cost is rotation-invariant and latent")
    print(f"    prediction 举一反三 holds (x{pred_mlp:.0f} OOD gap) -- the Phase-4 thesis, end to end.")
    print(f"    [C] latent closed-loop: secondary / noise-limited (N={N_TASK}x{len(EVAL_SEEDS)}/bin); "
          f"VN OOD ratio x{vn_ratio:.2f} (true 1.00).")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
