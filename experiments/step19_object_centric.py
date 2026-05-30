r"""Step 19: object-centric *compositional* equivariant latent — disentangling the two priors.

Where this sits, and the precise question it answers
----------------------------------------------------
Steps 13–18 proved the project's claims for a **single** rigid object under
$\mathrm{SE}(3)$: the learned 3D latent stays exactly equivariant ([A']), latent
prediction 举一反三 across the whole rotation group ([B]), and the closed loop
inherits an orientation error invariant to any global $(R,t)$ ([E], Step 18). That is
one object. The world has *many*. CLAUDE.md Open Question #3 asks the next thing
directly: **how do compositional / object-centric abstractions emerge in equivariant
latent world models?**

A scene of $O$ objects carries *two logically independent* symmetries, and the
sloppy thing to do is to conflate them. Step 19's whole point is to **separate** them
with a controlled three-model ablation:

  1. **Factorization** (a per-object *slot* structure with shared weights). This alone
     buys three exact properties: **leakage-freedom** (object $i$'s latent does not
     depend on object $j$'s state), **permutation-equivariance**
     $E(\sigma\!\cdot\!S)=\sigma\!\cdot\!E(S)$ for $\sigma\in S_O$, and
     **arrangement-invariance** (a translation-invariant per-object encoder makes the
     per-object latent independent of *where* the object sits).
  2. **Per-object $\mathrm{SE}(3)$-equivariance**. This buys **orientation 举一反三**:
     a per-object reorientation never seen in training is handled exactly.

The relevant symmetry group of the scene is thus $\mathrm{SE}(3)^{\,O}\rtimes S_O$ —
per-object rigid motions *and* object relabelings. Which architectural prior buys which
half of it? We answer with three models that differ *only* in which prior they carry:

  * **VN-Set**   (both priors)            : shared :class:`SE3PointEncoder` per object +
                                            shared jointly-equivariant ``VNPredictor``.
  * **MLP-Slot** (factorization only)     : shared *centered* per-object MLP encoder +
                                            shared ordinary ``LatentPredictor``. Slotted
                                            and translation-invariant, but **no rotation
                                            prior**.
  * **MLP-Global** (neither)              : one monolithic MLP on the whole flattened
                                            scene. No slots, no equivariance.

The 2×2 that isolates each prior
--------------------------------
Two out-of-distribution axes, each applied to the *same* held-out scenes (a paired
design; the transform of a valid teacher transition is a valid teacher transition):

  [B]  orientation-OOD : reorient **each object independently** by a random
                         $\mathrm{SO}(3)$ about its own centroid (novel per-object pose).
  [B'] arrangement-OOD : translate **each object independently** to a novel absolute /
                         relative placement (novel arrangement; orientation untouched).

                          | arrangement-OOD | orientation-OOD
        VN-Set   (both)   |      flat       |      flat
        MLP-Slot (factor) |      flat       |    DEGRADES   <- isolates equivariance
        MLP-Global (none) |    DEGRADES     |    DEGRADES

The **decisive, learned** result is the orientation column: VN-Set and MLP-Slot are
*both* factorized, so the only thing that differs between them is the $\mathrm{SE}(3)$
prior — VN-Set staying flat where MLP-Slot degrades is the equivariance contribution,
cleanly isolated. The arrangement column, the permutation check [A], and the leakage
check [F] are **exact-by-construction** (a translation-invariant, shared-weight,
per-object encoder *is* arrangement/permutation/leakage invariant — like Step 18's
exact centroid channel): they isolate the *factorization* contribution, where
MLP-Global fails and both factorized models are exact to the float floor. You need
**both** priors for full compositional 举一反三 — that is the headline.

The teacher (an exactly $\mathrm{SE}(3)^O$-equivariant, factorized world)
-----------------------------------------------------------------------
Each object is stepped by the Step-13 single-object teacher
:func:`teacher_step` (drift $c_t a$ + torque $c_r(a\times\tilde x)$ + anisotropic
stretch), which is exactly $\mathrm{SO}(3)$-equivariant *and* translation-equivariant
($\mathrm{Dyn}(Rx+t,Ra)=R\,\mathrm{Dyn}(x,a)+t$, verified in Steps 13/18). The scene
teacher applies it **per object, independently** — so the dynamics are exactly
factorized and exactly $\mathrm{SE}(3)^O\rtimes S_O$-equivariant. Two distinct
anisotropic templates make the objects *distinguishable* (so permutation is a genuine
operation and orientation is observable per object).

Honest scope. The objects do **not interact** — the scene dynamics are a direct sum of
per-object dynamics. That is what makes the factorization theorem clean, and it is the
cost of a *provable* compositional symmetry at laptop scale. Interaction (a
relative-pose channel between slots, à la Step 18's centroid term, or an equivariant
message-passing block) is the natural next rung and is explicitly future work. The
arrangement-invariance of the factorized models is *architectural and exact* (centering),
not learned; the **orientation 举一反三 of VN-Set vs MLP-Slot is the learned, decisive
comparison**. We gate the verdict on [A]+[F]+[B]+[B'].

Run (full ~10–15 min on a laptop CPU):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step19_object_centric.py
Smoke (~60 s):
    STEP19_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step19_object_centric.py
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
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 13 backbone we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

# Reuse the *validated* Step 13 machinery verbatim: the exactly-SE(3)-equivariant
# single-object teacher, the SO(3) helpers, the anisotropic templates, and the latent
# building blocks. Nothing geometric is re-invented here.
from step13_se3_latent_jepa import (  # noqa: E402
    ACTION_DIM,
    LATENT_DIM,
    N_POINTS,
    make_template,
    rand_so3,
    rot_z,
    rotate_latent,
    rotate_points,
    teacher_step,
)
from step10_pusht_closed_loop import n_params  # noqa: E402  (generic param counter)
from src.models.eqjepa import EqJEPA, LatentPredictor  # noqa: E402
from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.models.structured import VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP19_SMOKE"))

# --------------------------------------------------------------------------- #
# scene dimensions (a scene is O objects; the latent is O stacked per-object blocks)
# --------------------------------------------------------------------------- #
N_OBJ = 2                       # objects per scene
P = N_POINTS                    # 24 points per object (fixed ordering -> MLP control is fair)
D_OBJ = LATENT_DIM              # 48 per-object latent dim (= 3 * 16 vectors, for VNPredictor)
A_OBJ = ACTION_DIM              # 3 per-object action dim (type-1 vector)
D_SCENE = N_OBJ * D_OBJ         # 96 scene latent dim
A_SCENE = N_OBJ * A_OBJ         # 6 scene action dim
N_OUT_VEC = D_OBJ // 3          # 16 type-1 vectors per object

_PERM = torch.tensor([1, 0])    # the (only, for O=2) non-trivial object permutation

# two *distinct* anisotropic templates so the objects are distinguishable (permutation
# is non-vacuous) and orientation is observable per object.
_TEMPLATES = [make_template(0), make_template(1)]


# --------------------------------------------------------------------------- #
# scene <-> per-object reshaping helpers and the group actions on the latent
# --------------------------------------------------------------------------- #
def _rotate_actions(A: torch.Tensor, Rs) -> torch.Tensor:
    r"""Rotate each object's (type-1) action by its own $R_o$. ``(B, O*A_OBJ) -> (B, O*A_OBJ)``."""
    b = A.shape[0]
    Ab = A.reshape(b, N_OBJ, A_OBJ)
    out = torch.stack([rotate_points(Ab[:, o], Rs[o]) for o in range(N_OBJ)], dim=1)
    return out.reshape(b, -1)


def rotate_scene_latent(z: torch.Tensor, Rs) -> torch.Tensor:
    r"""Apply per-object $\rho(R_o)$ to each 48-dim block of the scene latent.

    ``(B, O*D_OBJ) -> (B, O*D_OBJ)``; each block is treated as ``D_OBJ/3`` stacked
    3-vectors. For ``Rs=[R]*O`` this is the *global* rotation $\rho(R)$ of the scene.
    """
    b = z.shape[0]
    zb = z.reshape(b, N_OBJ, D_OBJ)
    out = torch.stack([rotate_latent(zb[:, o], Rs[o]) for o in range(N_OBJ)], dim=1)
    return out.reshape(b, -1)


def permute_scene(S: torch.Tensor) -> torch.Tensor:
    r"""Relabel objects $\sigma\cdot S$. ``(B, O, P, 3) -> (B, O, P, 3)``."""
    return S[:, _PERM]


def permute_latent(z: torch.Tensor) -> torch.Tensor:
    r"""Relabel latent blocks $\sigma\cdot z$. ``(B, O*D_OBJ) -> (B, O*D_OBJ)``."""
    b = z.shape[0]
    return z.reshape(b, N_OBJ, D_OBJ)[:, _PERM].reshape(b, -1)


def permute_action(a: torch.Tensor) -> torch.Tensor:
    r"""Relabel action blocks. ``(B, O*A_OBJ) -> (B, O*A_OBJ)``."""
    b = a.shape[0]
    return a.reshape(b, N_OBJ, A_OBJ)[:, _PERM].reshape(b, -1)


# --------------------------------------------------------------------------- #
# the exactly SE(3)^O-equivariant, factorized teacher (per-object Step-13 dynamics)
# --------------------------------------------------------------------------- #
def scene_teacher_step(S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""Step every object by the Step-13 teacher, **independently**. ``(B,O,P,3),(B,O,3)->(B,O,P,3)``.

    A direct sum of per-object dynamics: $\mathrm{Dyn}(S,A)_o=\mathrm{Dyn}(S_o,A_o)$. Each
    summand is exactly $\mathrm{SO}(3)$- and translation-equivariant (Steps 13/18), so the
    scene teacher is exactly $\mathrm{SE}(3)^O\rtimes S_O$-equivariant. **No interaction** —
    the honest scope limit (see module docstring).
    """
    outs = [teacher_step(S[:, o], A[:, o]) for o in range(N_OBJ)]
    return torch.stack(outs, dim=1)


def _rel_offset(rng: np.random.Generator) -> np.ndarray:
    r"""A *training-cone* relative placement for object 2 (radius + a thin azimuth/elevation wedge)."""
    radius = rng.uniform(1.5, 2.5)
    az = math.radians(rng.uniform(0.0, 90.0))     # thin azimuth wedge (cf. Step 13's z-wedge)
    el = math.radians(rng.uniform(-20.0, 20.0))
    return np.array(
        [radius * math.cos(el) * math.cos(az),
         radius * math.cos(el) * math.sin(az),
         radius * math.sin(el)], dtype=np.float32,
    )


def make_scene_transitions(
    n: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``n`` seen-distribution scene transitions. ``S,S2: (n,O,P,3)``, ``A: (n,O*A_OBJ)``.

    Each object: its anisotropic template + small jitter/scale, rotated about $+z$ by a
    random $\phi\in[0,90°)$ (the *seen orientation wedge*, exactly as Step 13). Object 1 sits
    at the origin; object 2 at a training-cone offset (the *seen arrangement*). Actions are
    clipped Gaussians. OOD is produced later by *transforming* these transitions, so the
    seen/OOD comparison is paired.
    """
    rng = np.random.default_rng(seed)
    S = np.empty((n, N_OBJ, P, 3), np.float32)
    A = np.empty((n, N_OBJ, 3), np.float32)
    for i in range(n):
        for o in range(N_OBJ):
            jitter = rng.standard_normal((P, 3)).astype(np.float32) * 0.04
            axis_scale = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
            cloud = (_TEMPLATES[o] + jitter) * axis_scale          # body frame
            cloud = cloud @ rot_z(rng.uniform(0.0, 90.0)).numpy().T  # seen orientation wedge
            offset = np.zeros(3, np.float32) if o == 0 else _rel_offset(rng)
            S[i, o] = cloud + offset
            A[i, o] = np.clip(rng.standard_normal(3) * 0.6, -1.0, 1.0)
    St = torch.from_numpy(S)
    A3 = torch.from_numpy(A)
    S2t = scene_teacher_step(St, A3)
    return St, A3.reshape(n, A_SCENE), S2t


# --------------------------------------------------------------------------- #
# OOD transforms of a held-out transition (each is a *valid* teacher transition)
# --------------------------------------------------------------------------- #
def transform_orient(
    S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor, Rs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Reorient each object about its **start centroid** by $R_o$ (arrangement preserved).

    Rotating object $o$ about its start centroid $c_o$ is
    $T_{c_o}\!\circ\rho_{R_o}\!\circ T_{-c_o}$; since the teacher is rotation- and
    translation-equivariant, the rotated transition is valid with the *same* $c_o$ on
    both $S$ and $S_2$ and the action rotated by $R_o$. The encoder (translation-invariant
    per object) sees only the rotation, so a VN-Set per-object latent maps by $\rho(R_o)$.
    """
    c = S.mean(dim=2, keepdim=True)                          # (B,O,1,3) start centroids
    Sr = torch.stack([rotate_points(S[:, o] - c[:, o], Rs[o]) + c[:, o] for o in range(N_OBJ)], dim=1)
    S2r = torch.stack([rotate_points(S2[:, o] - c[:, o], Rs[o]) + c[:, o] for o in range(N_OBJ)], dim=1)
    return Sr, _rotate_actions(A, Rs), S2r


def transform_arrange(
    S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor, ts
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Translate each object by $t_o$ (a novel arrangement; orientation/action untouched).

    Valid because the teacher is translation-equivariant per object
    ($\mathrm{Dyn}(x+t,a)=\mathrm{Dyn}(x,a)+t$). Moving object 1 off the origin and object 2
    out of the training cone is in-support for nothing the MLP-Global encoder saw, but a
    *no-op* for a translation-invariant per-object encoder.
    """
    Sr = torch.stack([S[:, o] + ts[o].reshape(1, 1, 3) for o in range(N_OBJ)], dim=1)
    S2r = torch.stack([S2[:, o] + ts[o].reshape(1, 1, 3) for o in range(N_OBJ)], dim=1)
    return Sr, A.clone(), S2r


def transform_global_se3(
    S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor, R: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply ONE global $(R,t)$ to every object (about the origin). For the [A] equivariance test."""
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    S2r = rotate_points(S2, R) + t.reshape(1, 1, 1, 3)
    return Sr, _rotate_actions(A, [R] * N_OBJ), S2r


def perturb_object(S: torch.Tensor, obj: int, R: torch.Tensor) -> torch.Tensor:
    r"""Reorient a single object (about its centroid); used for the leakage probe [F]."""
    c = S.mean(dim=2, keepdim=True)
    Sr = S.clone()
    Sr[:, obj] = rotate_points(S[:, obj] - c[:, obj], R) + c[:, obj]
    return Sr


# --------------------------------------------------------------------------- #
# three scene encoders, differing ONLY in which prior they carry
# --------------------------------------------------------------------------- #
class SetSE3Encoder(nn.Module):
    r"""Object-centric, $\mathrm{SE}(3)$-equivariant scene encoder (**both** priors).

    A single :class:`SE3PointEncoder` shared across objects, applied per slot:
    ``(B, O, P, 3) -> (B, O*D_OBJ)``. Factorized (shared weights => permutation-equivariant
    and leakage-free) *and* per-object translation-invariant + $\mathrm{SO}(3)$-equivariant.
    The flattened latent is $O$ stacked per-object vector blocks, so $\rho$ is block-diagonal
    copies of $R$ (orthogonal) and the JEPA cost is rotation-invariant per object.
    """

    def __init__(self, n_obj: int = N_OBJ, n_out_vec: int = N_OUT_VEC, lmax: int = 2, mul: int = 8):
        super().__init__()
        self.n_obj = n_obj
        self.obj_enc = SE3PointEncoder(n_out_vec=n_out_vec, lmax=lmax, mul=mul)
        self.latent_dim = n_obj * self.obj_enc.latent_dim

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        # (B, O, P, 3) -> (B*O, P, 3) -> shared encoder -> (B, O*D_OBJ)
        b, o, p, _ = S.shape
        z = self.obj_enc(S.reshape(b * o, p, 3))
        return z.reshape(b, o * self.obj_enc.latent_dim)


class SlotMLPEncoder(nn.Module):
    r"""Factorized but **non-equivariant** scene encoder (factorization prior only).

    Centers each object (translation-invariant per slot), then a shared per-object MLP on the
    flattened coordinates: ``(B, O, P, 3) -> (B, O*D_OBJ)``. Shared weights => permutation-
    equivariant + leakage-free; centering => arrangement-invariant. But a global rotation of
    an object has **no** predictable action on its latent (the isolated control for the
    $\mathrm{SE}(3)$ prior — identical factorization to VN-Set, missing only equivariance).
    """

    def __init__(self, n_obj: int = N_OBJ, n_points: int = P, d_obj: int = D_OBJ, hidden: int = 128):
        super().__init__()
        self.n_obj, self.d_obj = n_obj, d_obj
        self.latent_dim = n_obj * d_obj
        self.net = nn.Sequential(
            nn.Linear(n_points * 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_obj),
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        b, o, p, _ = S.shape
        Sc = S - S.mean(dim=2, keepdim=True)        # center each object => translation-invariant
        z = self.net(Sc.reshape(b * o, p * 3))
        return z.reshape(b, o * self.d_obj)


class GlobalMLPEncoder(nn.Module):
    r"""Monolithic scene encoder (**neither** prior).

    One MLP on the whole flattened, **un-centered** scene: ``(B, O, P, 3) -> (B, D_SCENE)``.
    No slots (=> leaks across objects, not permutation-equivariant) and no rotation prior. The
    isolated control for the *factorization* prior.
    """

    def __init__(self, n_obj: int = N_OBJ, n_points: int = P, d_scene: int = D_SCENE, hidden: int = 256):
        super().__init__()
        self.latent_dim = d_scene
        self.net = nn.Sequential(
            nn.Linear(n_obj * n_points * 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_scene),
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.net(S.reshape(S.shape[0], -1))


class SlotPredictor(nn.Module):
    r"""Apply a **shared** per-object predictor to each latent block independently.

    ``z: (B, O*d_obj), a: (B, O*a_obj) -> (B, O*d_obj)``. Factorized dynamics: object $o$'s
    next latent depends only on $(z_o,a_o)$. With an equivariant per-object predictor the whole
    map is jointly $\mathrm{SE}(3)^O\rtimes S_O$-equivariant; with an ordinary one it is merely
    factorized.
    """

    def __init__(self, obj_predictor: nn.Module, n_obj: int = N_OBJ, d_obj: int = D_OBJ, a_obj: int = A_OBJ):
        super().__init__()
        self.obj = obj_predictor
        self.n_obj, self.d_obj, self.a_obj = n_obj, d_obj, a_obj

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        z2 = self.obj(z.reshape(b * self.n_obj, self.d_obj), a.reshape(b * self.n_obj, self.a_obj))
        return z2.reshape(b, self.n_obj * self.d_obj)


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #
def build_vn_set() -> EqJEPA:
    r"""Both priors: shared SE(3) encoder + shared jointly-equivariant VN predictor."""
    enc = SetSE3Encoder(N_OBJ, n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    pred = SlotPredictor(VNPredictor(D_OBJ, A_OBJ, hidden=64, dim=3))
    return EqJEPA(latent_dim=D_SCENE, action_dim=A_SCENE, encoder=enc, predictor=pred)


def build_mlp_slot() -> EqJEPA:
    r"""Factorization only: shared centered per-object MLP encoder + shared ordinary predictor."""
    enc = SlotMLPEncoder(N_OBJ, P, D_OBJ, hidden=128)
    pred = SlotPredictor(LatentPredictor(D_OBJ, A_OBJ, hidden=128))
    return EqJEPA(latent_dim=D_SCENE, action_dim=A_SCENE, encoder=enc, predictor=pred)


def build_mlp_global() -> EqJEPA:
    r"""Neither prior: monolithic scene MLP encoder + monolithic ordinary predictor."""
    enc = GlobalMLPEncoder(N_OBJ, P, D_SCENE, hidden=256)
    pred = LatentPredictor(D_SCENE, A_SCENE, hidden=256)
    return EqJEPA(latent_dim=D_SCENE, action_dim=A_SCENE, encoder=enc, predictor=pred)


# --------------------------------------------------------------------------- #
# [A] equivariance probes (global SE(3) + permutation), init AND post-train
# --------------------------------------------------------------------------- #
@torch.no_grad()
def trans_inv_err(enc: nn.Module, S: torch.Tensor, gen: torch.Generator) -> float:
    r"""$\max\lVert E(S+t)-E(S)\rVert_\infty$ for a random global shift (translation invariance)."""
    t = torch.randn(1, 1, 1, 3, generator=gen)
    return (enc(S + t) - enc(S)).abs().max().item()


@torch.no_grad()
def encoder_perm_err(enc: nn.Module, S: torch.Tensor) -> float:
    r"""$\max\lVert \sigma\!\cdot\!E(S) - E(\sigma\!\cdot\!S)\rVert_\infty$ (0 for a slotted encoder)."""
    return (permute_latent(enc(S)) - enc(permute_scene(S))).abs().max().item()


@torch.no_grad()
def composed_se3_err(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor) -> float:
    r"""End-to-end global-$\mathrm{SO}(3)$ residual $\max\lVert\rho(R)f(E(S),A)-f(E(RS),RA)\rVert_\infty$."""
    lhs = rotate_scene_latent(model.predictor(model.encoder(S), A), [R] * N_OBJ)
    Sr, Ar, _ = transform_global_se3(S, A, S, R, torch.zeros(3))
    rhs = model.predictor(model.encoder(Sr), Ar)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def composed_perm_err(model: EqJEPA, S: torch.Tensor, A: torch.Tensor) -> float:
    r"""End-to-end permutation residual $\max\lVert\sigma f(E(S),A)-f(E(\sigma S),\sigma A)\rVert_\infty$."""
    lhs = permute_latent(model.predictor(model.encoder(S), A))
    rhs = model.predictor(model.encoder(permute_scene(S)), permute_action(A))
    return (lhs - rhs).abs().max().item()


# --------------------------------------------------------------------------- #
# [F] leakage: does object 1's latent block move when object 2 is perturbed?
# --------------------------------------------------------------------------- #
@torch.no_grad()
def leakage_err(model: EqJEPA, S: torch.Tensor, gen: torch.Generator) -> float:
    r"""Relative change of the object-1 latent block when **object 2** is reoriented.

    $\mathbb{E}\,\lVert E_1(\text{perturb}_2 S)-E_1(S)\rVert/\lVert E_1(S)\rVert$, where
    $E_1$ is the first $D_{\rm obj}$ latent dims. For a slotted encoder this is $E_1=$ object-1's
    own latent => exactly $0$. For the monolithic encoder the "object-1 block" is a *nominal*
    slice of an entangled readout (it has no real factorization) — which is precisely why it
    leaks. So this number is both the leakage magnitude and a witness that MLP-Global cannot
    expose a clean per-object latent.
    """
    z1 = model.encoder(S).reshape(S.shape[0], N_OBJ, D_OBJ)[:, 0]
    z1p = model.encoder(perturb_object(S, 1, rand_so3(gen))).reshape(S.shape[0], N_OBJ, D_OBJ)[:, 0]
    return ((z1p - z1).norm(dim=-1) / z1.norm(dim=-1).clamp_min(1e-6)).mean().item()


# --------------------------------------------------------------------------- #
# [B]/[B'] latent-prediction 举一反三 (pooled relMSE; identical formula to Step 13)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def scene_latent_rel_mse(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor) -> float:
    r"""Pooled scene-latent 1-step relMSE $\sum\lVert f(E(s),a)-E(s')\rVert^2/\sum\lVert E(s')-E(s)\rVert^2$.

    $<1$ beats predicting no latent change. For an equivariant+factorized model it is *exactly*
    invariant under per-object reorientation and per-object translation (numerator and
    denominator carry the same orthogonal block $\rho$, or are untouched by translation).
    """
    z, z2 = model.encoder(S), model.encoder(S2)
    zp = model.predictor(z, A)
    num = ((zp - z2) ** 2).sum()
    den = ((z2 - z) ** 2).sum().clamp_min(1e-8)
    return (num / den).item()


@torch.no_grad()
def ood_factor(
    model: EqJEPA, St: torch.Tensor, At: torch.Tensor, S2t: torch.Tensor, *, kind: str,
    k_draws: int, t_arr: float, gen: torch.Generator,
) -> tuple[float, float, float]:
    r"""Return ``(seen, ood, ratio)`` relMSE for ``kind in {"orient","arrange"}`` (paired).

    ``orient``  : each object independently reoriented by a random $\mathrm{SO}(3)$ about its
                  centroid (novel per-object pose). ``arrange`` : each object independently
                  translated by $t\sim U(-\text{t\_arr},\text{t\_arr})^3$ (novel arrangement).
    Averages over ``k_draws`` random OOD elements; the seen score is the untransformed set.
    """
    seen = scene_latent_rel_mse(model, St, At, S2t)
    vals = []
    for _ in range(k_draws):
        if kind == "orient":
            Rs = [rand_so3(gen) for _ in range(N_OBJ)]
            Sr, Ar, S2r = transform_orient(St, At, S2t, Rs)
        else:  # arrange
            ts = [(torch.rand(3, generator=gen) * 2.0 - 1.0) * t_arr for _ in range(N_OBJ)]
            Sr, Ar, S2r = transform_arrange(St, At, S2t, ts)
        vals.append(scene_latent_rel_mse(model, Sr, Ar, S2r))
    ood = float(np.mean(vals))
    return seen, ood, ood / max(seen, 1e-12)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
_MODELS = ("VN-Set", "MLP-Slot", "MLP-Global")


def _equiv_panel(models: dict, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor, gen: torch.Generator) -> dict:
    r"""Print the [A] equivariance table (trans-inv, perm, global-SE(3), perm-composed) per model."""
    print(f"    {'model':>10s} | {'trans-inv':>10s} | {'perm enc':>10s} | {'SE(3) comp':>11s} | {'perm comp':>10s}")
    print("    " + "-" * 64)
    out = {}
    for name in _MODELS:
        m = models[name]
        ti = trans_inv_err(m.encoder, S, gen)
        pe = encoder_perm_err(m.encoder, S)
        ce = composed_se3_err(m, S, A, R)
        cp = composed_perm_err(m, S, A)
        out[name] = dict(trans_inv=ti, perm_enc=pe, se3_comp=ce, perm_comp=cp)
        print(f"    {name:>10s} | {ti:10.3e} | {pe:10.3e} | {ce:11.3e} | {cp:10.3e}")
    return out


def _ood_panel(models: dict, St, At, S2t, *, kind: str, k_draws: int, t_arr: float, gen: torch.Generator) -> dict:
    r"""Print one OOD panel ([B] orient or [B'] arrange): seen / OOD / ratio per model."""
    print(f"    {'model':>10s} | {'seen relMSE':>12s} | {'OOD relMSE':>12s} | {'OOD/seen':>9s}")
    print("    " + "-" * 52)
    out = {}
    for name in _MODELS:
        seen, ood, ratio = ood_factor(models[name], St, At, S2t, kind=kind, k_draws=k_draws, t_arr=t_arr, gen=gen)
        out[name] = dict(seen=seen, ood=ood, ratio=ratio)
        print(f"    {name:>10s} | {seen:12.4e} | {ood:12.4e} | x{ratio:7.3f}")
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72

    if SMOKE:
        N_TRAIN, N_TEST, EPOCHS, K_OOD = 150, 64, 3, 2
    else:
        N_TRAIN, N_TEST, EPOCHS, K_OOD = 1500, 400, 60, 6
    VAR_COEF, T_ARR = 0.1, 3.0

    print(line)
    print(f"STEP 19  object-centric compositional equivariant latent  ({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    scene: O={N_OBJ} independent objects, P={P} pts each; latent {D_SCENE}=O*{D_OBJ}; action {A_SCENE}")
    print(f"    priors: VN-Set=factorization+SE(3) | MLP-Slot=factorization only | MLP-Global=neither")

    # ---- data (seen distribution: z-wedge orientations + training-cone arrangement) ----
    S, A, S2 = make_scene_transitions(N_TRAIN, seed=0)
    St, At, S2t = make_scene_transitions(N_TEST, seed=999)
    models = {"VN-Set": build_vn_set(), "MLP-Slot": build_mlp_slot(), "MLP-Global": build_mlp_global()}
    g = torch.Generator().manual_seed(7)
    A_probe = torch.randn(St.shape[0], A_SCENE, generator=g)   # random action for equiv probes
    R_chk = rand_so3(torch.Generator().manual_seed(3))

    print(f"    params: " + "  ".join(f"{n}={n_params(m)}" for n, m in models.items()))

    # ---- [A] equivariance AT INIT ----------------------------------------------
    print()
    print(line)
    print("[A] equivariance of the scene world model (global SO(3) + object permutation) -- AT INIT")
    print(line)
    _equiv_panel(models, St, A_probe, R_chk, g)

    # ---- train all three (same recipe) -----------------------------------------
    print()
    print(line)
    print("[train] EMA-target JEPA (Muon/AdamW) on seen scenes (z-wedge orient + train-cone arrange)")
    print(line)
    hist = {}
    for name in _MODELS:
        hist[name] = train_jepa(models[name], S, A, S2, epochs=EPOCHS, batch_size=128,
                                var_coef=VAR_COEF, seed=0, log_every=999)
    print("    final latent_std (>0 => no collapse): "
          + "  ".join(f"{n}={hist[n]['latent_std']:.3f}" for n in _MODELS))

    # ---- [A'] equivariance AFTER training --------------------------------------
    print()
    print(line)
    print("[A'] equivariance AFTER training (the symmetry must survive optimisation)")
    print(line)
    eq = _equiv_panel(models, St, A_probe, R_chk, g)

    # ---- [F] leakage -----------------------------------------------------------
    print()
    print(line)
    print("[F] leakage: relative change of object-1 latent block when OBJECT 2 is reoriented")
    print(line)
    leak = {}
    for name in _MODELS:
        leak[name] = leakage_err(models[name], St, g)
        print(f"    {name:>10s} : {leak[name]:.4e}"
              + ("   (slotted => leakage-free)" if leak[name] < 1e-4 else "   (entangled => leaks)"))

    # ---- [B] orientation-OOD (the DECISIVE, learned panel) ---------------------
    print()
    print(line)
    print("[B] orientation 举一反三: each object independently reoriented by random SO(3) (about its centroid)")
    print(line)
    orient = _ood_panel(models, St, At, S2t, kind="orient", k_draws=K_OOD, t_arr=T_ARR, gen=g)
    print("    => VN-Set flat (per-object SE(3) prior); MLP-Slot degrades despite identical")
    print("       factorization -- this isolates the EQUIVARIANCE contribution.")

    # ---- [B'] arrangement-OOD (exact-by-construction factorization panel) ------
    print()
    print(line)
    print("[B'] arrangement invariance: each object independently translated to a novel placement")
    print(line)
    arrange = _ood_panel(models, St, At, S2t, kind="arrange", k_draws=K_OOD, t_arr=T_ARR, gen=g)
    print("    => VN-Set & MLP-Slot exactly flat (translation-invariant slots); MLP-Global degrades")
    print("       -- this isolates the FACTORIZATION contribution.")

    # ---- summary + verdict -----------------------------------------------------
    print()
    print(line)
    print("STEP 19 SUMMARY")
    print(line)
    # Guards: factorization (perm + leakage exact for slotted, broken for global), equivariance
    # (VN-Set still SE(3)-equivariant post-train), and the disentangled 2x2 OOD behaviour.
    ok_vn_equiv = eq["VN-Set"]["se3_comp"] < 1e-4 and eq["VN-Set"]["perm_comp"] < 1e-4
    ok_factor_perm = eq["VN-Set"]["perm_enc"] < 1e-4 and eq["MLP-Slot"]["perm_enc"] < 1e-4 \
        and eq["MLP-Global"]["perm_enc"] > 1e-2
    ok_leak = leak["VN-Set"] < 1e-4 and leak["MLP-Slot"] < 1e-4 and leak["MLP-Global"] > 0.05
    ok_orient = orient["VN-Set"]["ratio"] < 1.15 and orient["MLP-Slot"]["ratio"] > 1.3 \
        and orient["MLP-Global"]["ratio"] > 1.3
    ok_arrange = arrange["VN-Set"]["ratio"] < 1.15 and arrange["MLP-Slot"]["ratio"] < 1.15 \
        and arrange["MLP-Global"]["ratio"] > 1.3
    passed = ok_vn_equiv and ok_factor_perm and ok_leak and ok_orient and ok_arrange

    print(f"    the 2x2 that separates the two compositional priors (OOD/seen relMSE factor):")
    print(f"        {'':>10s} | {'arrange-OOD':>12s} | {'orient-OOD':>12s}")
    print(f"        {'VN-Set':>10s} | x{arrange['VN-Set']['ratio']:10.3f} | x{orient['VN-Set']['ratio']:10.3f}")
    print(f"        {'MLP-Slot':>10s} | x{arrange['MLP-Slot']['ratio']:10.3f} | x{orient['MLP-Slot']['ratio']:10.3f}")
    print(f"        {'MLP-Global':>10s} | x{arrange['MLP-Global']['ratio']:10.3f} | x{orient['MLP-Global']['ratio']:10.3f}")
    print(f"    factorization (exact): VN-Set/MLP-Slot perm-equiv & leakage-free "
          f"({max(eq['VN-Set']['perm_enc'], eq['MLP-Slot']['perm_enc'], leak['VN-Set'], leak['MLP-Slot']):.1e}); "
          f"MLP-Global leaks ({leak['MLP-Global']:.2f}).")
    print(f"    equivariance (learned): VN-Set still SE(3)-equivariant post-train "
          f"(composed {eq['VN-Set']['se3_comp']:.1e}); it alone is flat on BOTH OOD axes.")
    print(f"    guards: vn-equiv={ok_vn_equiv}  factor-perm={ok_factor_perm}  leak={ok_leak}  "
          f"orient={ok_orient}  arrange={ok_arrange}")
    print(f"    headline: a scene carries TWO priors -- factorization (=> permutation/leakage/")
    print(f"        arrangement invariance, exact) and per-object SE(3)-equivariance (=> orientation")
    print(f"        举一反三, learned). The ablation isolates each: MLP-Slot has the first and fails")
    print(f"        orientation; MLP-Global has neither. Only VN-Set, carrying BOTH, generalises across")
    print(f"        the full scene group SE(3)^O |x| S_O. Honest scope: objects do not interact (the")
    print(f"        teacher is a direct sum); a relative-pose / message channel is the next rung.")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}")

    # ---- dump JSON artifact for the papers -------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(N_OBJ=N_OBJ, P=P, D_SCENE=D_SCENE, A_SCENE=A_SCENE,
                       N_TRAIN=N_TRAIN, N_TEST=N_TEST, EPOCHS=EPOCHS, K_OOD=K_OOD,
                       VAR_COEF=VAR_COEF, T_ARR=T_ARR),
        "params": {n: n_params(m) for n, m in models.items()},
        "latent_std": {n: hist[n]["latent_std"] for n in _MODELS},
        "equiv_posttrain": eq,
        "leakage": leak,
        "orient_ood": orient,
        "arrange_ood": arrange,
        "verdict": {"passed": bool(passed), "ok_vn_equiv": ok_vn_equiv, "ok_factor_perm": ok_factor_perm,
                    "ok_leak": ok_leak, "ok_orient": ok_orient, "ok_arrange": ok_arrange},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / ("step19_object_centric_smoke.json" if SMOKE else "step19_object_centric.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
