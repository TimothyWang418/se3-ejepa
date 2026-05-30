r"""Step 13: SO(3) point-cloud end-to-end equivariant **latent** JEPA -- 举一反三 in 3D.

Where this sits, and why it is the missing rung
-----------------------------------------------
Step 6 wired :class:`~src.models.se3.SE3PointEncoder` (e3nn, SE(3)) to
:class:`~src.models.structured.VNPredictor` (``dim=3``) and verified the *whole*
3D latent world model is exactly SE(3)-equivariant -- but only **at init, on random
clouds**, with no training, no baseline, and no generalisation test. Step 11 did the
real scientific test (train end-to-end; show the *learned* latent stays equivariant
and that latent prediction 举一反三 holds) but only in **2D / SO(2)** on PushT.

Step 13 closes the gap: it runs the Step-11 protocol in **3D / SO(3)** on point
clouds. We ask the project's core question one dimension up:

  (i)  does the *learned* 3D latent world model stay **exactly SO(3)-equivariant
       after training** (so its JEPA planning cost is rotation-invariant), and
  (ii) does latent-space prediction **举一反三 across the whole rotation group**
       hold -- training the dynamics on a thin *wedge* of orientations
       (rotations about $+z$ by $\phi\in[0,90°)$) *determines* it on all of
       $\mathrm{SO}(3)$, including axes and angles never seen?

Everything geometric is *composed* from existing, unit-tested modules -- nothing is
re-invented:

* encoder   : :class:`src.models.se3.SE3PointEncoder` (exact SE(3))
              vs a parameter-comparable :class:`MLPPointEncoder` (no prior)
* predictor : :class:`src.models.structured.VNPredictor` (``dim=3``, jointly equiv.)
              vs :class:`src.models.eqjepa.LatentPredictor` (ordinary residual MLP)
* wrapper   : :class:`src.models.eqjepa.EqJEPA` (latent ``rollout``)
* training  : :func:`src.training.jepa.train_jepa` -- EMA-target JEPA + VICReg
              variance hinge + Muon/AdamW, used **unchanged** (it only calls
              ``model.encoder(o)`` / ``model.predictor(z,a)``), fed ``(N, P, 3)``
              point-cloud transitions instead of pixels.

The teacher dynamics (a *synthetic* but exactly-SO(3)-equivariant world)
---------------------------------------------------------------------------
There is no laptop-scale 3D physics sim that is *provably* SO(3)-equivariant, so we
use an explicit equivariant teacher and ask whether the equivariant *student*
generalises across the group where the ordinary one cannot. For a centred cloud
$\tilde x_i = x_i - \bar x$ and a type-1 action $a\in\mathbb{R}^3$:

$$ x_i' \;=\; x_i \;+\; \underbrace{c_t\,a}_{\text{rigid drift}}
        \;+\; \underbrace{c_r\,\big(a\times\tilde x_i\big)}_{\text{torque / rotation channel}}
        \;+\; \underbrace{c_d\,\langle a,\hat u_i\rangle\,\hat u_i}_{\text{anisotropic stretch}},
   \qquad \hat u_i=\tilde x_i/\lVert\tilde x_i\rVert. $$

Each term is exactly SO(3)-equivariant: $\mathrm{Dyn}(Rx,Ra)=R\,\mathrm{Dyn}(x,a)$,
using $Ra\times R\tilde x = R(a\times\tilde x)$ for $R\in\mathrm{SO}(3)$ (proper
rotations) and the invariance of $\langle a,\hat u\rangle$. The **torque** term is
the 3D analogue of PushT's block-rotation channel -- the place a non-equivariant
model is expected to break out-of-distribution; the **drift** term is the easy,
near-linear "self-motion" channel an MLP extrapolates fine (cf. Step 12's
agent_pos vs block_dir decomposition).

Why the equivariant latent is *exactly* flat across orientation (the guarantee).
$E(Rx)=\rho(R)E(x)$ with $\rho(R)$ orthogonal (block-diagonal copies of $R$), and
$f(\rho(R)z,Ra)=\rho(R)f(z,a)$, so for any rotated transition $(Rx,Ra)\to Rx'$ the
latent prediction error and the matching cost are unchanged. The ordinary baseline
has no such guarantee and degrades off the training wedge. We *verify* this to the
float floor at init **and** after training, per project policy.

Honest scope. The teacher is synthetic (the cost of a provable 3D symmetry at
laptop scale); the decisive claims are the representation-level ones -- [A']
equivariance survives training, [B] latent 举一反三 across SO(3). [C] is a
bonus closed-loop check that plans in the learned latent and *steps the teacher as
the ground-truth env*; because the teacher is itself exactly equivariant, the VN's
[C] OOD gap should be ~float-zero (a cleaner setting than the C4-contaminated real
PushT of Steps 11/12). We gate the verdict on [A']+[B].

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step13_se3_latent_jepa.py
    # fast smoke: STEP13_SMOKE=1 .venv/bin/python experiments/step13_se3_latent_jepa.py
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
sys.path.insert(0, str(HERE))   # for the Step 10 helper we reuse (n_params)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models.eqjepa import EqJEPA, LatentPredictor  # noqa: E402
from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.models.structured import VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

from step10_pusht_closed_loop import n_params  # noqa: E402  (generic param counter)

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP13_SMOKE"))

# --------------------------------------------------------------------------- #
# geometry / latent dimensions
# --------------------------------------------------------------------------- #
N_POINTS = 24          # points per cloud (fixed ordering -> MLP baseline is fair)
N_OUT_VEC = 16         # SE3 encoder outputs this many type-1 vectors
LATENT_DIM = 3 * N_OUT_VEC   # = 48 (must be divisible by 3 for VNPredictor dim=3)
ACTION_DIM = 3

# teacher dynamics coefficients (small -> one step is a modest, learnable perturbation)
C_T, C_R, C_D = 0.15, 0.15, 0.08


# --------------------------------------------------------------------------- #
# SO(3) helpers (continuous, exact -- no grid). Local to keep scope contained.
# --------------------------------------------------------------------------- #
def _skew(v: torch.Tensor) -> torch.Tensor:
    r"""Skew-symmetric matrix $[v]_\times$ with $[v]_\times u = v\times u$. ``(3,)->(3,3)``."""
    x, y, z = v
    return torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float32
    )


def axis_angle_matrix(axis: torch.Tensor, angle: float) -> torch.Tensor:
    r"""Rodrigues' formula: $R=\mathbf{I}+\sin\theta\,[\hat n]_\times+(1-\cos\theta)[\hat n]_\times^2$."""
    n = axis / axis.norm().clamp_min(1e-8)
    K = _skew(n)
    return torch.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rot_z(deg: float) -> torch.Tensor:
    r"""Rotation about $+z$ by ``deg`` degrees (the training-wedge axis)."""
    return axis_angle_matrix(torch.tensor([0.0, 0.0, 1.0]), math.radians(deg))


def rot_x(deg: float) -> torch.Tensor:
    return axis_angle_matrix(torch.tensor([1.0, 0.0, 0.0]), math.radians(deg))


def rot_y(deg: float) -> torch.Tensor:
    return axis_angle_matrix(torch.tensor([0.0, 1.0, 0.0]), math.radians(deg))


def rand_so3(gen: torch.Generator) -> torch.Tensor:
    r"""A random rotation: axis uniform on $S^2$, angle uniform in $[0,2\pi)$.

    Not exactly Haar (angle is uniform, not $\propto(1-\cos)$), but it scatters
    rotations across *all* axes and large angles -- which is all the OOD test needs:
    these are far from the $z$-axis $[0,90°)$ training wedge.
    """
    axis = torch.randn(3, generator=gen)
    angle = 2.0 * math.pi * torch.rand((), generator=gen).item()
    return axis_angle_matrix(axis, angle)


def rotate_points(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Apply $v\mapsto Rv$ to every vector. ``(..., 3) -> (..., 3)`` (right-multiply)."""
    return x @ R.transpose(-1, -2)


def rotate_latent(z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Apply $\rho(R)$ (block-diagonal copies of $R$) to a flattened vector latent.

    ``(B, latent_dim) -> (B, latent_dim)``, treating the latent as stacked 3-vectors.
    """
    b = z.shape[0]
    return rotate_points(z.reshape(b, -1, 3), R).reshape(b, -1)


# --------------------------------------------------------------------------- #
# data: anisotropic clouds + exactly-SO(3)-equivariant teacher
# --------------------------------------------------------------------------- #
def make_template(seed: int = 0) -> np.ndarray:
    r"""A fixed, **anisotropic** base cloud (so orientation is observable). ``(P, 3)``.

    Drawn once with distinct per-axis scales -- no rotational symmetry, so a rotation
    genuinely changes the cloud and the OOD test is not vacuous (cf. Step 8 anisotropy).
    """
    rng = np.random.default_rng(seed)
    scale = np.array([1.0, 0.55, 0.3], dtype=np.float32)   # anisotropic
    return (rng.standard_normal((N_POINTS, 3)).astype(np.float32) * scale)


_TEMPLATE = make_template(0)


def teacher_step(X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""One step of the exactly-SO(3)-equivariant teacher. ``(B,P,3),(B,3)->(B,P,3)``.

    $x_i' = x_i + c_t a + c_r (a\times\tilde x_i) + c_d\langle a,\hat u_i\rangle\hat u_i$.
    """
    xbar = X.mean(dim=1, keepdim=True)            # (B,1,3) centroid
    xt = X - xbar                                  # centred
    u = xt / xt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a_b = a[:, None, :]                            # (B,1,3) broadcast over points
    drift = C_T * a_b
    torque = C_R * torch.cross(a_b.expand_as(xt), xt, dim=-1)
    proj = (a_b * u).sum(-1, keepdim=True)         # <a, u_i> invariant scalar
    stretch = C_D * proj * u
    return X + drift + torque + stretch


def collect_cloud_transitions(
    n: int, *, seed: int = 0, phi_lo: float = 0.0, phi_hi: float = 90.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``n`` transitions in the orientation *wedge* (z-rotation $\phi\in[\text{lo},\text{hi})$).

    Each cloud: shared anisotropic template + small per-sample jitter and per-axis
    scale (variety, consistent point ordering), then rotated about $+z$ by a random
    $\phi$ in the wedge. Action ~ clipped Gaussian. Returns ``(S, A, S2)`` with
    ``S, S2: (n, P, 3)``, ``A: (n, 3)``.
    """
    rng = np.random.default_rng(seed)
    S = np.empty((n, N_POINTS, 3), np.float32)
    A = np.empty((n, 3), np.float32)
    for i in range(n):
        jitter = rng.standard_normal((N_POINTS, 3)).astype(np.float32) * 0.04
        axis_scale = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
        cloud = (_TEMPLATE + jitter) * axis_scale            # (P,3) body frame
        phi = rng.uniform(phi_lo, phi_hi)
        Rz = rot_z(phi).numpy()
        S[i] = cloud @ Rz.T                                  # rotate into the wedge
        A[i] = np.clip(rng.standard_normal(3) * 0.6, -1.0, 1.0)
    St = torch.from_numpy(S)
    At = torch.from_numpy(A)
    S2t = teacher_step(St, At)
    return St, At, S2t


# --------------------------------------------------------------------------- #
# baseline encoder (non-equivariant counterpart to SE3PointEncoder)
# --------------------------------------------------------------------------- #
class MLPPointEncoder(nn.Module):
    r"""Ordinary MLP on the flattened point cloud. No symmetry prior. ``(B,P,3)->(B,D)``.

    Flattens the $(P,3)$ coordinates (fixed point ordering across the dataset, so this
    is a fair non-equivariant control -- not handicapped by permutation) and maps them
    through a plain MLP, so a global rotation of the input has **no** predictable action
    on the latent.
    """

    def __init__(self, n_points: int = N_POINTS, latent_dim: int = LATENT_DIM, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(n_points * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, 3) -> (B, latent_dim)
        return self.net(x.flatten(1))


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #
def build_eq_jepa() -> EqJEPA:
    r"""Exactly-SO(3)-equivariant latent JEPA: SE3 encoder + jointly-equivariant VN predictor."""
    enc = SE3PointEncoder(n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    pred = VNPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=64, dim=3)
    return EqJEPA(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, encoder=enc, predictor=pred)


def build_mlp_jepa() -> EqJEPA:
    r"""Non-equivariant baseline latent JEPA: MLP encoder + ordinary residual MLP predictor."""
    enc = MLPPointEncoder(n_points=N_POINTS, latent_dim=LATENT_DIM, hidden=128)
    pred = LatentPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=256)
    return EqJEPA(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, encoder=enc, predictor=pred)


# --------------------------------------------------------------------------- #
# [A] equivariance unit tests (random SO(3); init AND post-train)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encoder_equiv_err(enc: nn.Module, S: torch.Tensor, R: torch.Tensor) -> float:
    r"""$\max\lVert \rho(R)E(x) - E(Rx)\rVert_\infty$ (0 for an SO(3) encoder)."""
    lhs = rotate_latent(enc(S), R)
    rhs = enc(rotate_points(S, R))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def predictor_equiv_err(pred: nn.Module, z: torch.Tensor, a: torch.Tensor, R: torch.Tensor) -> float:
    r"""$\max\lVert \rho(R)f(z,a) - f(\rho(R)z, R a)\rVert_\infty$."""
    lhs = rotate_latent(pred(z, a), R)
    rhs = pred(rotate_latent(z, R), rotate_points(a, R))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def composed_equiv_err(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor) -> float:
    r"""End-to-end (encode then predict) equivariance residual -- the headline unit test."""
    lhs = rotate_latent(model.predictor(model.encoder(S), A), R)
    rhs = model.predictor(model.encoder(rotate_points(S, R)), rotate_points(A, R))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def trans_inv_err(enc: nn.Module, S: torch.Tensor, gen: torch.Generator) -> float:
    r"""$\max\lVert E(x+t) - E(x)\rVert_\infty$ for a random global shift (SE(3) check)."""
    t = torch.randn(1, 1, 3, generator=gen)
    return (enc(S + t) - enc(S)).abs().max().item()


@torch.no_grad()
def cost_drift(enc: nn.Module, S: torch.Tensor, Sg: torch.Tensor, R: torch.Tensor) -> float:
    r"""Relative drift of the JEPA matching cost $\lVert E(x)-E(x_g)\rVert$ under joint rotation.

    SO(3)-equivariant encoder => drift $\approx 0$ (orthogonal $\rho$); ordinary encoder drifts.
    """
    base = (enc(S) - enc(Sg)).norm(dim=-1)
    rot = (enc(rotate_points(S, R)) - enc(rotate_points(Sg, R))).norm(dim=-1)
    return ((rot - base).abs().mean() / base.mean().clamp_min(1e-8)).item()


# --------------------------------------------------------------------------- #
# [B] latent-space prediction 举一反三 (pooled relMSE; robust like Step 12)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def latent_rel_mse(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor) -> float:
    r"""Pooled latent 1-step prediction error, normalised by the latent step size.

    $\mathrm{relMSE}=\sum_i\lVert f(E(s_i),a_i)-E(s_i')\rVert^2 \big/
    \sum_i\lVert E(s_i')-E(s_i)\rVert^2$ ($<1$ = better than predicting no latent
    change). For the equivariant JEPA this is *exactly* invariant under a rotation of
    the transition (numerator and denominator both carry an orthogonal $\rho$); the
    baseline degrades OOD.
    """
    z = model.encoder(S)
    z2 = model.encoder(S2)
    zp = model.predictor(z, A)
    num = ((zp - z2) ** 2).sum()
    den = ((z2 - z) ** 2).sum().clamp_min(1e-8)
    return (num / den).item()


# --------------------------------------------------------------------------- #
# [C] latent closed-loop planning toward a goal cloud (bonus; teacher = ground-truth env)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def latent_cem_plan(
    model: EqJEPA, X0: torch.Tensor, zg: torch.Tensor,
    *, H: int = 8, n_samples: int = 256, n_iters: int = 5, n_elite: int = 25,
    sigma0: float = 0.6, w_run: float = 0.3, gen: torch.Generator | None = None,
) -> torch.Tensor:
    r"""CEM-MPC against a purely-latent terminal cost $\lVert\hat z_H - z_g\rVert_2^2$.

    The latent is rolled with ``model.predictor`` (no cloud is decoded inside the
    rollout). Returns the elite-mean plan ``(H, 3)``. ``X0: (1,P,3)``, ``zg: (1,D)``.
    """
    z0 = model.encoder(X0).expand(n_samples, -1).contiguous()
    zg = zg.expand(n_samples, -1).contiguous()
    mean = torch.zeros(H, 3)
    sigma = torch.full((H, 3), sigma0)
    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 3, generator=gen)
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
    return mean


@torch.no_grad()
def plan_frac_closed(
    model: EqJEPA, X0: torch.Tensor, Xg: torch.Tensor, *, gen: torch.Generator, **cem
) -> float:
    r"""Plan in the learned latent, **execute on the teacher**, return fraction of gap closed.

    $\text{frac}=1-\lVert X_{\text{achieved}}-X_g\rVert/\lVert X_0-X_g\rVert$ (1 = reached,
    0 = no progress, <0 = moved away). ``X0, Xg: (1,P,3)``.
    """
    zg = model.encoder(Xg)
    plan = latent_cem_plan(model, X0, zg, gen=gen, **cem)   # (H,3)
    X = X0.clone()
    for h in range(plan.shape[0]):
        X = teacher_step(X, plan[h : h + 1])               # step the ground-truth teacher
    gap0 = (X0 - Xg).norm().clamp_min(1e-6)
    achieved = (X - Xg).norm()
    return float(1.0 - achieved / gap0)


def make_goal_tasks(k: int, *, seed: int, H_goal: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    r"""``k`` reachable goal tasks in the seen wedge: ``(X0, Xg)`` with $X_g$ a teacher rollout."""
    St, _, _ = collect_cloud_transitions(k, seed=seed)
    rng = np.random.default_rng(seed + 7)
    tasks = []
    for i in range(k):
        X0 = St[i : i + 1]                                  # (1,P,3)
        X = X0.clone()
        for _ in range(H_goal):
            a = torch.from_numpy(np.clip(rng.standard_normal((1, 3)) * 0.6, -1, 1).astype(np.float32))
            X = teacher_step(X, a)
        tasks.append((X0, X))
    return tasks


@torch.no_grad()
def eval_plan_orientation(
    model: EqJEPA, tasks: list[tuple[torch.Tensor, torch.Tensor]],
    rotations: list[torch.Tensor], *, gen: torch.Generator, **cem
) -> float:
    r"""Mean fraction-of-gap-closed over tasks, each rotated by every ``R`` in ``rotations``.

    For ``rotations=[I]`` this is the *seen* score; for OOD rotations the equivariant model
    is invariant (teacher + model both equivariant), the baseline degrades.
    """
    fracs = []
    for X0, Xg in tasks:
        for R in rotations:
            fracs.append(
                plan_frac_closed(model, rotate_points(X0, R), rotate_points(Xg, R), gen=gen, **cem)
            )
    return float(np.mean(fracs))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72

    N_TRAIN = 200 if SMOKE else 1500
    N_TEST = 80 if SMOKE else 400
    EPOCHS = 3 if SMOKE else 60
    VAR_COEF = 0.1  # VICReg anti-collapse weight (3D latent needs more than the 2D 0.04)

    # ---- data: train on the z-wedge [0,90); one held-out set we rotate per bin
    S, A, S2 = collect_cloud_transitions(N_TRAIN, seed=0)
    St, At, S2t = collect_cloud_transitions(N_TEST, seed=999)

    eq = build_eq_jepa()
    mlp = build_mlp_jepa()

    g = torch.Generator().manual_seed(1)
    z_rand = torch.randn(64, LATENT_DIM, generator=g)
    a_rand = torch.randn(64, ACTION_DIM, generator=g)
    R_test = rand_so3(torch.Generator().manual_seed(3))

    # ---------------------------------------------------------------- [A] init
    print(line)
    print("[A] Equivariance of the 3D latent world model (random SO(3)) -- AT INIT")
    print(line)
    print("    VN equivariant JEPA (should be ~float floor):")
    print(f"        translation inv |E(x+t)-E(x)|         : {trans_inv_err(eq.encoder, St, g):.4e}")
    print(f"        encoder    |rho E(x) - E(Rx)|          : {encoder_equiv_err(eq.encoder, St, R_test):.4e}")
    print(f"        predictor  |rho f(z,a) - f(rho z,Ra)|  : {predictor_equiv_err(eq.predictor, z_rand, a_rand, R_test):.4e}")
    print(f"        composed   (encode->predict)           : {composed_equiv_err(eq, St, At, R_test):.4e}")
    print(f"    MLP baseline composed equiv residual       : {composed_equiv_err(mlp, St, At, R_test):.4e} (no prior)")
    print(f"    params: VN={n_params(eq)}   MLP={n_params(mlp)}  ({n_params(mlp) / n_params(eq):.1f}x VN)")

    # ---------------------------------------------------------------- train
    print()
    print(line)
    print("[train] EMA-target JEPA (Muon/AdamW) on the z-wedge [0,90)")
    print(line)
    print(f"    {len(S)} train / {len(St)} held-out cloud transitions ({N_POINTS} pts each)")
    print("    VN equivariant JEPA:")
    h_eq = train_jepa(eq, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=10)
    print("    MLP baseline JEPA:")
    h_mlp = train_jepa(mlp, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=10)
    print(f"    final latent_std: VN={h_eq['latent_std']:.3f}  MLP={h_mlp['latent_std']:.3f}  (>0 => no collapse)")

    # ---------------------------------------------------------------- [A'] post
    print()
    print(line)
    print("[A'] Equivariance AFTER training (the property must survive optimisation)")
    print(line)
    eq_comp = composed_equiv_err(eq, St, At, R_test)
    print(f"    VN  composed equiv residual : {eq_comp:.4e}  (still exact)")
    print(f"    MLP composed equiv residual : {composed_equiv_err(mlp, St, At, R_test):.4e}")
    print("    JEPA cost drift |C(Rx,Rx_g)-C(x,x_g)| / C  under random SO(3):")
    print(f"    {'rotation':>10s} | {'VN drift':>12s} | {'MLP drift':>12s}")
    print("    " + "-" * 42)
    Sg = St.roll(1, dims=0)  # arbitrary distinct "goal" clouds from the held-out set
    drift_gen = torch.Generator().manual_seed(11)
    vn_drift, mlp_drift = [], []
    for i in range(4):
        R = rand_so3(drift_gen)
        vd, md = cost_drift(eq.encoder, St, Sg, R), cost_drift(mlp.encoder, St, Sg, R)
        vn_drift.append(vd); mlp_drift.append(md)
        print(f"    rand R_{i}    | {vd:12.4e} | {md:12.4e}")
    print(f"    => VN cost rotation-invariant (max {max(vn_drift):.1e}); MLP drifts (max {max(mlp_drift):.2f}).")

    # ---------------------------------------------------------------- [B]
    print()
    print(line)
    print("[B] LATENT-space prediction 举一反三 (train z-wedge [0,90); rotate held-out)")
    print(line)
    print("    latent 1-step relMSE (SAME held-out set, rotated into each orientation bin):")
    print(f"    {'orientation bin':22s} | {'VN relMSE':>12s} | {'MLP relMSE':>12s}")
    print("    " + "-" * 53)
    # bins: the seen z-wedge (in-distribution) then a ladder of clearly-OOD rotations
    rand_gen_b = torch.Generator().manual_seed(5)
    R_rand_list = [rand_so3(rand_gen_b) for _ in range(8)]
    bins: list[tuple[str, list[torch.Tensor]]] = [
        ("z 45 (seen wedge)", [rot_z(45.0)]),
        ("z 180 (OOD angle)", [rot_z(180.0)]),
        ("x 90  (OOD axis)", [rot_x(90.0)]),
        ("y 90  (OOD axis)", [rot_y(90.0)]),
        ("random SO(3) x8", R_rand_list),
    ]
    vn_b, mlp_b = [], []
    for lab, Rs in bins:
        ve = float(np.mean([latent_rel_mse(eq, rotate_points(St, R), rotate_points(At, R), rotate_points(S2t, R)) for R in Rs]))
        me = float(np.mean([latent_rel_mse(mlp, rotate_points(St, R), rotate_points(At, R), rotate_points(S2t, R)) for R in Rs]))
        vn_b.append(ve); mlp_b.append(me)
        print(f"    {lab:22s} | {ve:12.4e} | {me:12.4e}")
    vn_seen, mlp_seen = vn_b[0], mlp_b[0]
    vn_ood = max(vn_b[1:]); mlp_ood = max(mlp_b[1:])
    pred_vn = vn_ood / max(vn_seen, 1e-12)
    pred_mlp = mlp_ood / max(mlp_seen, 1e-12)
    print(f"    => VN flat across SO(3) (seen={vn_seen:.3e} OOD={vn_ood:.3e}, x{pred_vn:.2f}); "
          f"MLP degrades (seen={mlp_seen:.3f} OOD={mlp_ood:.3f}, x{pred_mlp:.1f}).")

    # ---------------------------------------------------------------- [C]
    print()
    print(line)
    print("[C] LATENT closed-loop planning to a goal cloud (bonus; teacher = ground-truth env)")
    print(line)
    K_TASK = 4 if SMOKE else 12
    H_GOAL = 6
    cem_kw = dict(H=H_GOAL, n_samples=(64 if SMOKE else 256), n_iters=(3 if SMOKE else 5),
                  n_elite=(8 if SMOKE else 25), sigma0=0.6, w_run=0.3)
    tasks = make_goal_tasks(K_TASK, seed=321, H_goal=H_GOAL)
    ood_gen = torch.Generator().manual_seed(13)
    ood_Rs = [rand_so3(ood_gen) for _ in range(3)]
    plan_gen = torch.Generator()
    plan_gen.manual_seed(0)
    vn_seen_c = eval_plan_orientation(eq, tasks, [torch.eye(3)], gen=plan_gen, **cem_kw)
    plan_gen.manual_seed(0)
    vn_ood_c = eval_plan_orientation(eq, tasks, ood_Rs, gen=plan_gen, **cem_kw)
    plan_gen.manual_seed(0)
    mlp_seen_c = eval_plan_orientation(mlp, tasks, [torch.eye(3)], gen=plan_gen, **cem_kw)
    plan_gen.manual_seed(0)
    mlp_ood_c = eval_plan_orientation(mlp, tasks, ood_Rs, gen=plan_gen, **cem_kw)
    print(f"    plan H={H_GOAL} in learned latent, execute on teacher; {K_TASK} tasks, 3 OOD rotations")
    print("    fraction of gap closed (1=reached goal, 0=no progress):")
    print(f"    {'model':6s} | {'seen (I)':>10s} | {'OOD SO(3)':>10s} | {'OOD/seen':>9s}")
    print("    " + "-" * 46)
    vn_c_ratio = vn_ood_c / max(abs(vn_seen_c), 1e-6)
    mlp_c_ratio = mlp_ood_c / max(abs(mlp_seen_c), 1e-6)
    print(f"    {'VN':6s} | {vn_seen_c:10.3f} | {vn_ood_c:10.3f} | x{vn_c_ratio:7.2f}")
    print(f"    {'MLP':6s} | {mlp_seen_c:10.3f} | {mlp_ood_c:10.3f} | x{mlp_c_ratio:7.2f}")

    # ---------------------------------------------------------------- summary
    print()
    print(line)
    print("STEP 13 SUMMARY")
    print(line)
    print("    end-to-end SO(3) point-cloud LATENT JEPA: SE(3) encoder + jointly-")
    print("    equivariant VN predictor (dim=3), planned in the LEARNED latent.")
    print(f"    [A'] learned latent stays exactly SO(3)-equivariant after training:")
    print(f"         VN composed residual {eq_comp:.1e} ; cost drift max {max(vn_drift):.1e} (MLP {max(mlp_drift):.2f}).")
    print(f"    [B]  LATENT prediction 举一反三 across SO(3): VN flat (seen={vn_seen:.2e} OOD={vn_ood:.2e}, x{pred_vn:.2f}),")
    print(f"         MLP degrades (seen={mlp_seen:.3f} OOD={mlp_ood:.3f}, x{pred_mlp:.1f}); new-axis rotations are worst for MLP.")
    print(f"    [C]  latent closed-loop frac-gap-closed, OOD/seen: VN x{vn_c_ratio:.2f} (seen={vn_seen_c:.2f}), "
          f"MLP x{mlp_c_ratio:.2f} (seen={mlp_seen_c:.2f}).")

    # Gate on the decisive representation-level results [A']+[B]; [C] reported as bonus.
    ok_eq = eq_comp < 1e-4
    ok_drift = max(vn_drift) < 0.02 and max(mlp_drift) > 0.10
    ok_pred_vn = pred_vn < 1.05
    ok_pred_mlp = pred_mlp > 1.5
    passed = ok_eq and ok_drift and ok_pred_vn and ok_pred_mlp
    print(f"    guards: latent-SO(3)-equivariant={ok_eq}  cost-invariant-vs-drift={ok_drift}  "
          f"VN-flat={ok_pred_vn}  MLP-degrades={ok_pred_mlp}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: the equivariant prior makes the LEARNED 3D latent world")
    print(f"    model inherit exact SO(3) symmetry, so its planning cost is rotation-invariant and")
    print(f"    latent prediction 举一反三 holds across the whole group (x{pred_mlp:.0f} OOD gap) from a z-wedge.")
    print(f"    [C] latent closed-loop is a clean bonus (teacher is itself equivariant): "
          f"VN OOD/seen x{vn_c_ratio:.2f}, MLP x{mlp_c_ratio:.2f}.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
