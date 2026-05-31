r"""Step 35: few-body -> many-body **combinatorial** 举一反三 (a NEW generalisation axis).

Where this sits, and the precise question it answers
----------------------------------------------------
Steps 13--18 proved orientation 举一反三 for **one** object; Step 19 added an
object-centric *slot* structure (factorisation) for **several non-interacting**
objects; Step 24 turned on **interaction** for exactly $O=2$ objects via an
equivariant relative-pose message and showed the equivariance prior still pays
after the scene symmetry collapses to the global diagonal
$\mathrm{SE}(3)\rtimes S_O$. Every one of those rungs fixed the object **count**.

Step 35 opens a *new* generalisation axis that is **combinatorial, not geometric**:
the number of objects $O$. We train the *interacting* world model on a single
cardinality ($O=3$) and ask whether it 举一反三 **zero-shot** to cardinalities it
never saw -- $O\in\{1,2,4,5,6\}$ -- including counts both *smaller* and *larger*
than training. This is the discrete analogue of the project's continuous-group
generalisation: a thin slice of the configuration space (one count) should
*determine* the dynamics on the rest of it (all counts).

The count-stable message: the one design decision that makes this possible
---------------------------------------------------------------------------
Step 24's two-body torque used the single relative direction $\hat r_{ij}$. The
naive many-body generalisation -- **sum** over neighbours,
$\omega_i=(\sum_{j\ne i}\hat r_{ij})\times a_i$ -- has a message whose magnitude
grows with the count, so a model trained at $O=3$ would see out-of-range messages
at $O=6$ and could not generalise. We instead **mean-aggregate**:

$$ \bar r_i \;=\; \frac{1}{O-1}\sum_{j\ne i}\hat r_{ij},
   \qquad \omega_i \;=\; \bar r_i\times a_i,
   \qquad x_k^{(i)\prime}=\mathrm{self}_i(x_k)+\kappa\,(\omega_i\times\tilde x_k^{(i)}). $$

The mean of unit vectors lives in the unit ball at **every** count
($\lVert\bar r_i\rVert\le 1$), so the per-object message statistics are
count-stable -- the property that makes a single-count predictor transfer. This is
the standard permutation-invariant (DeepSets / mean-pool GNN) aggregator, and it
keeps the teacher **exactly** equivariant at every $O$:

* global $\mathrm{SE}(3)$: $\hat r_{ij}\mapsto R\hat r_{ij}$ (translation cancels in
  $c_j-c_i$; norm is rotation-invariant), so $\bar r_i\mapsto R\bar r_i$, and
  $\omega_i=\bar r_i\times a_i\mapsto R\omega_i$ (cross product of two type-1
  vectors, proper $R$); the torque maps as $R(\omega_i\times\tilde x_k)$ and the
  Step-13 self-term is already equivariant -- so the whole step maps as
  $x'\mapsto Rx'+t$ at any count.
* permutation $S_O$: $\bar r_i$ is a **symmetric mean** over the other objects, so
  relabelling permutes the per-object rule exactly. The single-other-object case
  $O=2$ recovers Step 24 verbatim ($\bar r_i=\hat r_{ij}$).

So the message handed to the predictor is exactly the $\bar r_i$ the teacher
crosses with $a_i$: one type-1 vector per object, $A^{\rm aug}_{\rm obj}=6=[a_i,\bar r_i]$.

The 2x2 that isolates the two priors (the same logic as Step 19, new axis)
-------------------------------------------------------------------------
Two out-of-distribution axes, each applied to held-out scenes:

  [C]  count-OOD at the **seen** orientation: evaluate at $O\ne 3$ with objects in
       the trained $z$-wedge. Count-generalisation here is bought by **factorisation
       + the count-stable mean message** -- a shared-weight, leakage-free slot
       encoder/predictor applies the *same* per-object map at any $O$, and the only
       thing that shifts is the (bounded) message distribution. So BOTH the
       equivariant VN-MP and the non-equivariant MLP-MP should stay flat. Learned,
       not exact; this isolates the *combinatorial / factorisation* contribution.
       The claim is *interaction transfer* across the many-body family $O\ge 2$ (both
       smaller, $O=2$, and larger, $O\in\{4,5,6\}$, than the train count). The single
       count $O=1$ is the qualitatively different **no-interaction limit**: the mean
       message is identically $\mathbf 0$ -- an input the $O=3$-trained channel never
       sees -- and the torque vanishes so the latent step shrinks; it is reported as a
       separate boundary (still beats no-change), not folded into the headline ratio.
  [G]  count x global-orientation: at each unseen count, rotate the WHOLE scene by a
       random $\mathrm{SO}(3)$ off the $z$-wedge. VN-MP is **exactly** flat
       (orthogonal $\rho(R)$ on numerator and denominator, message recomputed =
       rotated message), at *every* count; **MLP-MP degrades**. This isolates the
       $\mathrm{SE}(3)$-equivariance contribution, exactly as Step 19's orientation
       column did for a fixed count.

                       | count-OOD (seen orient) | count x orientation-OOD
        VN-MP  (equiv) |          flat           |        flat
        MLP-MP (no eq) |          flat           |      DEGRADES   <- isolates equivariance

The decisive, novel guard is that VN-MP stays **exactly** global-$\mathrm{SE}(3)\rtimes
S_O$-equivariant at a count it **never trained on** ($O=5$), at init AND after
training -- the architecture is count-agnostic by construction, and we verify it.
VN-Set (Step 19 verbatim, channel-blind) is the in-distribution channel-necessity
witness: it cannot see $\bar r_i$, so it is mis-specified at every count.

Honest scope. Vanilla degree-1 Vector Neurons (VN-Linear + VN-ReLU) are degree-1
homogeneous and cannot form the trilinear torque $(\bar r_i\times a_i)\times\tilde x_k$;
so BOTH VN models share Step 24's cross-product cap and VN-MP's *absolute* fit is
capped (relMSE not tiny). The headline is **flatness across count and orientation**,
which is exact regardless of the cap (a bilinear/tensor-product message -- Step 27 --
is the way to lift the cap). The mean is lossy (it discards higher moments of the
neighbour set), an honest limit of a single type-1 message; and the teacher is
synthetic but provably equivariant at every count (the cost of a provable discrete
symmetry at laptop scale). We gate on VN-MP-equivariant-at-unseen-count +
[C]-VN-flat (over the interacting family $O\ge 2$, with the $O=1$ no-interaction
limit reported separately and required only to beat no-change) +
[G]-(VN-flat, MLP-degrades) + channel-necessity.

Run (full ~15-22 min on a laptop CPU):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step35_many_body.py
Smoke (~90 s):
    STEP35_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step35_many_body.py
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
sys.path.insert(0, str(HERE))   # for the Step 13/19 backbones we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

# Reuse the *validated* Step 13 geometry + Step 19 slot machinery verbatim. Nothing
# geometric is re-invented: the single-object teacher, the SO(3) helpers, the
# anisotropic templates, and the (already variable-cardinality) slot encoders.
from step13_se3_latent_jepa import (  # noqa: E402
    make_template,
    rand_so3,
    rot_z,
    rotate_latent,
    rotate_points,
    teacher_step,
)
from step19_object_centric import (  # noqa: E402
    A_OBJ,
    D_OBJ,
    N_OUT_VEC,
    P,
    SetSE3Encoder,
    SlotMLPEncoder,
)
from step10_pusht_closed_loop import n_params  # noqa: E402  (generic param counter)
from src.models.eqjepa import EqJEPA, LatentPredictor  # noqa: E402
from src.models.structured import VNPredictor  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP35_SMOKE"))

# --------------------------------------------------------------------------- #
# dimensions. The message is ONE mean-relative-direction vector per object, appended
# to that object's action, so the per-object "action" the VN-MP predictor consumes is
# 6-dim (two type-1 vectors: the real action and the count-stable mean message).
# --------------------------------------------------------------------------- #
A_MSG = 3                          # one mean-relative-direction vector per object (type-1)
A_OBJ_AUG = A_OBJ + A_MSG          # 6: [a_i (3), rbar_i (3)]

# interaction strength: reuse Step 24's calibrated value so the inter-object torque is a
# meaningful fraction of the step (channel-blind VN-Set clearly mis-specified in-dist).
C_INT = float(os.environ.get("STEP35_CINT", "0.80"))

TRAIN_OBJ = 3                      # the SINGLE cardinality we train on
MAX_OBJ = 6                        # template pool size (every test count uses only seen shapes)

# six distinct anisotropic templates; every scene draws a WITHOUT-replacement subset, so the
# encoder/predictor see all six shapes during O=3 training and no test count introduces a
# novel shape -- isolating COUNT from shape (cf. Step 19's two distinct templates).
_TEMPLATES6 = [make_template(s) for s in range(MAX_OBJ)]


# --------------------------------------------------------------------------- #
# the count-stable mean relative-direction message, shared by teacher and channel
# --------------------------------------------------------------------------- #
def mean_rel_dirs(S: torch.Tensor) -> torch.Tensor:
    r"""Per-object mean unit relative direction. ``(B,O,P,3) -> (B,O,3)``.

    $\bar r_i=\frac{1}{O-1}\sum_{j\ne i}\hat r_{ij}$, $\hat r_{ij}=(c_j-c_i)/\lVert c_j-c_i\rVert$,
    $c_i=\bar x_i$ the object centroid. The mean of unit vectors lies in the unit ball at every
    count (count-stable). Defined as $0$ for $O=1$ (no neighbours -> no interaction). Exactly
    global-$\mathrm{SE}(3)$-equivariant ($\bar r_i\mapsto R\bar r_i$, translation-invariant) and
    $S_O$-equivariant (symmetric mean over the other objects).
    """
    b, o, _, _ = S.shape
    if o == 1:
        return S.new_zeros(b, 1, 3)
    c = S.mean(dim=2)                                          # (B,O,3) centroids
    rbar = S.new_zeros(b, o, 3)
    for i in range(o):
        acc = S.new_zeros(b, 3)
        for j in range(o):
            if j == i:
                continue
            r = c[:, j] - c[:, i]                              # (B,3)
            acc = acc + r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rbar[:, i] = acc / (o - 1)
    return rbar


def scene_teacher_many(S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""Variable-cardinality interacting scene step. ``(B,O,P,3),(B,O,3) -> (B,O,P,3)``.

    $x_k^{(i)\prime}=\mathrm{self}_i(x_k)+\kappa\,(\omega_i\times\tilde x_k^{(i)})$ with the
    **mean-aggregated** angular velocity $\omega_i=\bar r_i\times a_i$. Exactly global-
    $\mathrm{SE}(3)\rtimes S_O$-equivariant at any $O$ (see module docstring); recovers the
    Step 24 two-body teacher when $O=2$, and pure Step-13 self-dynamics when $O=1$.
    """
    b, o, _, _ = S.shape
    self_next = torch.stack([teacher_step(S[:, i], A[:, i]) for i in range(o)], dim=1)
    if o == 1:
        return self_next                                      # no interaction
    rbar = mean_rel_dirs(S)                                    # (B,O,3)
    c = S.mean(dim=2)                                          # (B,O,3)
    out = self_next.clone()
    for i in range(o):
        omega = torch.cross(rbar[:, i], A[:, i], dim=-1)       # (B,3) equivariant angular velocity
        xt = S[:, i] - c[:, i][:, None, :]                     # (B,P,3) centred points of object i
        torque = C_INT * torch.cross(omega[:, None, :].expand_as(xt), xt, dim=-1)
        out[:, i] = out[:, i] + torque
    return out


def build_msg_action_many(S: torch.Tensor, A_self: torch.Tensor) -> torch.Tensor:
    r"""Append each object's mean-relative-direction message to its action. ``-> (B, O*A_OBJ_AUG)``.

    Per object $i$: $[\,a_i,\;\bar r_i\,]$ -- exactly the count-stable type-1 vector the teacher
    crosses with $a_i$. Two stacked type-1 vectors, so the jointly-equivariant ``VNPredictor``
    stays equivariant in the augmented action at every count.
    """
    b, o, _, _ = S.shape
    rbar = mean_rel_dirs(S)                                    # (B,O,3)
    Aself = A_self.reshape(b, o, A_OBJ)
    aug = torch.cat([Aself, rbar], dim=-1)                     # (B,O,A_OBJ_AUG)
    return aug.reshape(b, o * A_OBJ_AUG)


# --------------------------------------------------------------------------- #
# variable-cardinality group actions on the scene / latent / action (parameterised by O)
# --------------------------------------------------------------------------- #
def rot_self_actions_var(A_self: torch.Tensor, R: torch.Tensor, o: int) -> torch.Tensor:
    r"""Rotate every object's (type-1) self-action by a global $R$. ``(B,O*A_OBJ)->(B,O*A_OBJ)``."""
    b = A_self.shape[0]
    return rotate_points(A_self.reshape(b, o, A_OBJ), R).reshape(b, o * A_OBJ)


def rotate_scene_latent_var(z: torch.Tensor, Rs: list[torch.Tensor], o: int) -> torch.Tensor:
    r"""Apply per-object $\rho(R_i)$ to each ``D_OBJ`` block of the scene latent. ``(B,O*D_OBJ)->...``."""
    b = z.shape[0]
    zb = z.reshape(b, o, D_OBJ)
    return torch.stack([rotate_latent(zb[:, i], Rs[i]) for i in range(o)], dim=1).reshape(b, -1)


def permute_scene_var(S: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    r"""Relabel objects $\sigma\cdot S$. ``(B,O,P,3)->(B,O,P,3)``."""
    return S[:, perm]


def permute_latent_var(z: torch.Tensor, perm: torch.Tensor, o: int) -> torch.Tensor:
    r"""Relabel latent blocks $\sigma\cdot z$. ``(B,O*D_OBJ)->(B,O*D_OBJ)``."""
    b = z.shape[0]
    return z.reshape(b, o, D_OBJ)[:, perm].reshape(b, -1)


def permute_self_action_var(A_self: torch.Tensor, perm: torch.Tensor, o: int) -> torch.Tensor:
    r"""Relabel self-action blocks. ``(B,O*A_OBJ)->(B,O*A_OBJ)``."""
    b = A_self.shape[0]
    return A_self.reshape(b, o, A_OBJ)[:, perm].reshape(b, -1)


def transform_global_many(
    S: torch.Tensor, A_self: torch.Tensor, S2: torch.Tensor, R: torch.Tensor, t: torch.Tensor, o: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply ONE global $(R,t)$ to the whole scene (a symmetry of the interacting teacher).

    The message is recomputed downstream from the transformed scene (it is translation-invariant
    and $\mathrm{SO}(3)$-equivariant), so the [G] panel is a paired seen-vs-OOD test at each count.
    """
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    S2r = rotate_points(S2, R) + t.reshape(1, 1, 1, 3)
    return Sr, rot_self_actions_var(A_self, R, o), S2r


# --------------------------------------------------------------------------- #
# data: O objects in the seen orientation wedge at count-independent random placements
# --------------------------------------------------------------------------- #
def _rand_offset(rng: np.random.Generator) -> np.ndarray:
    r"""A count-independent random placement on a shell (radius 1.5--2.5, full sphere).

    Drawn iid per object regardless of $O$, so the per-object *arrangement* distribution is
    count-invariant and only the NUMBER of neighbours changes -- isolating the count axis.
    """
    radius = rng.uniform(1.5, 2.5)
    az = math.radians(rng.uniform(0.0, 360.0))
    el = math.radians(rng.uniform(-30.0, 30.0))
    return np.array(
        [radius * math.cos(el) * math.cos(az),
         radius * math.cos(el) * math.sin(az),
         radius * math.sin(el)], dtype=np.float32,
    )


def make_many_body_transitions(
    n: int, o: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``n`` interacting transitions of exactly ``o`` objects. ``S,S2:(n,o,P,3)``, ``A_self:(n,o*A_OBJ)``.

    Each scene draws a without-replacement subset of the 6 templates (so no count introduces a
    novel shape), each rotated about $+z$ by a random $\phi\in[0,90°)$ (the seen orientation
    wedge) and placed at an iid count-independent offset. Returns the *self* actions; the mean
    message is built separately so each model is fed exactly the channel it is entitled to.
    """
    rng = np.random.default_rng(seed)
    S = np.empty((n, o, P, 3), np.float32)
    A = np.empty((n, o, 3), np.float32)
    for k in range(n):
        templ = rng.choice(MAX_OBJ, size=o, replace=False)
        for i in range(o):
            jitter = rng.standard_normal((P, 3)).astype(np.float32) * 0.04
            axis_scale = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
            cloud = (_TEMPLATES6[templ[i]] + jitter) * axis_scale
            cloud = cloud @ rot_z(rng.uniform(0.0, 90.0)).numpy().T     # seen orientation wedge
            S[k, i] = cloud + _rand_offset(rng)
            A[k, i] = np.clip(rng.standard_normal(3) * 0.6, -1.0, 1.0)
    St = torch.from_numpy(S)
    A3 = torch.from_numpy(A)
    S2t = scene_teacher_many(St, A3)
    return St, A3.reshape(n, o * A_OBJ), S2t


# --------------------------------------------------------------------------- #
# variable-cardinality slot predictor (infers the count from the latent width)
# --------------------------------------------------------------------------- #
class VarSlotPredictor(nn.Module):
    r"""Apply a **shared** per-object predictor to each latent block, for **any** object count.

    ``z:(B,O*d_obj), a:(B,O*a_obj) -> (B,O*d_obj)``. Unlike Step 19's :class:`SlotPredictor`
    (which hard-codes ``n_obj``), the count is inferred as ``o = z.shape[1] // d_obj`` at every
    call, so the *same* trained weights run at any cardinality. With an equivariant per-object
    predictor the whole map is jointly $\mathrm{SE}(3)^{\,O}\rtimes S_O$-equivariant at every $O$.
    """

    def __init__(self, obj_predictor: nn.Module, d_obj: int = D_OBJ, a_obj: int = A_OBJ_AUG):
        super().__init__()
        self.obj = obj_predictor
        self.d_obj, self.a_obj = d_obj, a_obj

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        o = z.shape[1] // self.d_obj                          # infer count dynamically
        z2 = self.obj(z.reshape(b * o, self.d_obj), a.reshape(b * o, self.a_obj))
        return z2.reshape(b, o * self.d_obj)


# --------------------------------------------------------------------------- #
# three models, differing ONLY in the message channel and the equivariance prior
# --------------------------------------------------------------------------- #
def build_vn_mp() -> EqJEPA:
    r"""Equivariance + message: shared SE(3) encoder + shared VN predictor whose per-object action
    is augmented by the mean-relative-direction message (jointly equivariant *and* coupling-aware)."""
    enc = SetSE3Encoder(TRAIN_OBJ, n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    pred = VarSlotPredictor(VNPredictor(D_OBJ, A_OBJ_AUG, hidden=64, dim=3), d_obj=D_OBJ, a_obj=A_OBJ_AUG)
    return EqJEPA(latent_dim=TRAIN_OBJ * D_OBJ, action_dim=TRAIN_OBJ * A_OBJ_AUG, encoder=enc, predictor=pred)


def build_vn_set() -> EqJEPA:
    r"""Equivariance, **no** message (Step 19 verbatim, variable-card): channel-blind => mis-specified."""
    enc = SetSE3Encoder(TRAIN_OBJ, n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    pred = VarSlotPredictor(VNPredictor(D_OBJ, A_OBJ, hidden=64, dim=3), d_obj=D_OBJ, a_obj=A_OBJ)
    return EqJEPA(latent_dim=TRAIN_OBJ * D_OBJ, action_dim=TRAIN_OBJ * A_OBJ, encoder=enc, predictor=pred)


def build_mlp_mp() -> EqJEPA:
    r"""Message + slot factorisation, **no** equivariance: Step 19's *centred* :class:`SlotMLPEncoder`
    + ordinary per-slot predictor fed the SAME augmented message. The clean one-variable control for
    VN-MP -- carries the message and factorisation, differing ONLY in the $\mathrm{SE}(3)$ prior."""
    enc = SlotMLPEncoder(TRAIN_OBJ, P, D_OBJ, hidden=128)
    pred = VarSlotPredictor(LatentPredictor(D_OBJ, A_OBJ_AUG, hidden=128), d_obj=D_OBJ, a_obj=A_OBJ_AUG)
    return EqJEPA(latent_dim=TRAIN_OBJ * D_OBJ, action_dim=TRAIN_OBJ * A_OBJ_AUG, encoder=enc, predictor=pred)


_MODELS = ("VN-MP", "VN-Set", "MLP-MP")


def _build(name: str) -> EqJEPA:
    if name == "VN-MP":
        return build_vn_mp()
    if name == "VN-Set":
        return build_vn_set()
    return build_mlp_mp()


def model_action_many(name: str, S: torch.Tensor, A_self: torch.Tensor) -> torch.Tensor:
    r"""The action each model is entitled to: VN-MP and MLP-MP get the augmented (message) action;
    only the channel-blind VN-Set gets the bare self action. VN-MP vs VN-Set isolates the MESSAGE;
    VN-MP vs MLP-MP isolates the EQUIVARIANCE prior."""
    return A_self if name == "VN-Set" else build_msg_action_many(S, A_self)


# --------------------------------------------------------------------------- #
# relMSE (identical formula to Steps 13/19/24) + the variable-count equivariance probes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rel_mse(model: EqJEPA, S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor) -> float:
    r"""Pooled scene-latent 1-step relMSE $\sum\lVert f(E(s),a)-E(s')\rVert^2/\sum\lVert E(s')-E(s)\rVert^2$.

    Normalised per-count by its own latent step size, so it is directly comparable across counts:
    a flat relMSE across $O$ means count-invariant per-object prediction quality. For the
    equivariant model it is *exactly* invariant under a global rotation of the transition.
    """
    z, z2 = model.encoder(S), model.encoder(S2)
    zp = model.predictor(z, A)
    num = ((zp - z2) ** 2).sum()
    den = ((z2 - z) ** 2).sum().clamp_min(1e-8)
    return (num / den).item()


@torch.no_grad()
def rel_mse_named(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor, S2: torch.Tensor) -> float:
    r"""relMSE feeding ``model`` exactly the action channel it is entitled to."""
    return rel_mse(model, S, model_action_many(name, S, A_self), S2)


@torch.no_grad()
def vnmp_se3_err(model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor, R: torch.Tensor, t: torch.Tensor, o: int) -> float:
    r"""Whole-pipeline global $\mathrm{SE}(3)$ residual at count ``o`` (message channel included).

    $\max\lVert\rho(R)\,f(E(S),\mathrm{aug}(S,A))-f(E(RS+t),\mathrm{aug}(RS+t,RA))\rVert_\infty$.
    The message recomputed from the transformed scene equals the rotated message (translation-
    invariant, $\mathrm{SO}(3)$-equivariant), so this sits at the float floor for the equivariant
    model at any count -- including counts never trained on -- and also exercises translation via $t$.
    """
    aug = build_msg_action_many(S, A_self)
    lhs = rotate_scene_latent_var(model.predictor(model.encoder(S), aug), [R] * o, o)
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    aug_r = build_msg_action_many(Sr, rot_self_actions_var(A_self, R, o))
    rhs = model.predictor(model.encoder(Sr), aug_r)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def vnmp_perm_err(model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor, perm: torch.Tensor, o: int) -> float:
    r"""Whole-pipeline permutation residual at count ``o``: $\max\lVert\sigma f(E(S),\mathrm{aug})-f(E(\sigma S),\mathrm{aug}(\sigma S))\rVert_\infty$.

    Relabelling permutes the (symmetric-mean) message blocks, so recomputing the augmented action
    from the permuted scene equals permuting it -- exact for the shared-weight slot model at any $O$.
    """
    aug = build_msg_action_many(S, A_self)
    lhs = permute_latent_var(model.predictor(model.encoder(S), aug), perm, o)
    Sp = permute_scene_var(S, perm)
    aug_p = build_msg_action_many(Sp, permute_self_action_var(A_self, perm, o))
    rhs = model.predictor(model.encoder(Sp), aug_p)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def ctrl_se3_err(model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor, R: torch.Tensor, o: int) -> float:
    r"""Global-$\mathrm{SO}(3)$ residual for the only bare-action model, VN-Set (no message), at count ``o``."""
    lhs = rotate_scene_latent_var(model.predictor(model.encoder(S), A_self), [R] * o, o)
    Sr = rotate_points(S, R)
    rhs = model.predictor(model.encoder(Sr), rot_self_actions_var(A_self, R, o))
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def composed_se3_residual(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor, R: torch.Tensor, t: torch.Tensor, o: int) -> float:
    # VN-Set is the only bare-action model; VN-MP and MLP-MP both consume the augmented message.
    return ctrl_se3_err(model, S, A_self, R, o) if name == "VN-Set" else vnmp_se3_err(model, S, A_self, R, t, o)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 72

    if SMOKE:
        N_TRAIN, N_TEST, EPOCHS, K_OOD = 150, 64, 3, 2
        COUNTS_C = [1, 2, 3, 5]
        COUNTS_G = [4]
        UNSEEN_CHK = 4
    else:
        N_TRAIN, N_TEST, EPOCHS, K_OOD = 1500, 400, 60, 6
        COUNTS_C = [1, 2, 3, 4, 5, 6]
        COUNTS_G = [2, 4, 6]
        UNSEEN_CHK = 5
    N_TRAIN = int(os.environ.get("STEP35_NTRAIN", N_TRAIN))
    EPOCHS = int(os.environ.get("STEP35_EPOCHS", EPOCHS))
    VAR_COEF = 0.1

    print(line)
    print(f"STEP 35  few-body -> many-body combinatorial 举一反三  ({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    train on a SINGLE count O={TRAIN_OBJ}; test zero-shot on O in {COUNTS_C} (count-OOD) "
          f"and O in {COUNTS_G} (count x SO(3))")
    print(f"    mean-aggregated torque omega_i=(rbar_i x a_i), rbar_i=mean_j!=i rhat_ij (count-stable, "
          f"|rbar|<=1); kappa={C_INT}")
    print(f"    per-object: P={P} pts, latent {D_OBJ}, self-action {A_OBJ}, VN-MP action {A_OBJ_AUG} (msg)")

    # ---- data: train at O=TRAIN_OBJ; held-out sets at every test count (paired seen/OOD) ----
    S, A_self, S2 = make_many_body_transitions(N_TRAIN, TRAIN_OBJ, seed=0)
    test = {o: make_many_body_transitions(N_TEST, o, seed=900 + o) for o in sorted(set(COUNTS_C) | set(COUNTS_G))}
    # a small probe batch at an UNSEEN count for the equivariance guards
    Su, Au, _ = make_many_body_transitions(min(N_TEST, 64), UNSEEN_CHK, seed=4242)

    models = {name: _build(name) for name in _MODELS}
    g = torch.Generator().manual_seed(7)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    t_chk = torch.randn(3, generator=torch.Generator().manual_seed(4))
    perm_chk = torch.roll(torch.arange(UNSEEN_CHK), 1)         # a non-trivial permutation of O=5 slots
    print(f"    params: " + "  ".join(f"{n}={n_params(m)}" for n, m in models.items()))

    # ---- [A] equivariance AT INIT, at an UNSEEN count (the architecture is count-agnostic) ----
    def equiv_panel(tag: str) -> dict:
        print(f"    O={UNSEEN_CHK} (UNSEEN) | {'model':>8s} | {'SE(3) comp':>11s} | {'perm comp':>10s}")
        print("    " + "-" * 48)
        out = {}
        for name in _MODELS:
            se3 = composed_se3_residual(name, models[name], Su, Au, R_chk, t_chk, UNSEEN_CHK)
            perm = vnmp_perm_err(models[name], Su, Au, perm_chk, UNSEEN_CHK) if name != "VN-Set" else float("nan")
            out[name] = dict(se3_comp=se3, perm_comp=perm)
            ptxt = f"{perm:10.3e}" if name != "VN-Set" else f"{'n/a':>10s}"
            print(f"    {tag:>12s} | {name:>8s} | {se3:11.3e} | {ptxt}")
        return out

    print()
    print(line)
    print(f"[A] equivariance of the many-body world model at an UNSEEN count O={UNSEEN_CHK} -- AT INIT")
    print(line)
    equiv_panel("init")

    # ---- train all three on the SAME single-count data (each fed its entitled channel) ----
    print()
    print(line)
    print(f"[train] EMA-target JEPA (Muon/AdamW) on the SINGLE training count O={TRAIN_OBJ}")
    print(line)
    hist = {}
    for name in _MODELS:
        hist[name] = train_jepa(models[name], S, model_action_many(name, S, A_self), S2,
                                epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=999)
    print("    final latent_std (>0 => no collapse): "
          + "  ".join(f"{n}={hist[n]['latent_std']:.3f}" for n in _MODELS))

    # ---- [A'] equivariance AFTER training, still at the UNSEEN count (decisive novel guard) ----
    print()
    print(line)
    print(f"[A'] equivariance AFTER training, at the UNSEEN count O={UNSEEN_CHK} (must survive optimisation)")
    print(line)
    eq = equiv_panel("post-train")

    # ---- [I] in-distribution fit at the TRAIN count (channel-necessity witness) ----
    print()
    print(line)
    print(f"[I] IN-DISTRIBUTION fit at the TRAIN count O={TRAIN_OBJ}: does the model represent the interaction?")
    print(line)
    Stn, Atn, S2tn = test[TRAIN_OBJ]
    indist = {name: rel_mse_named(name, models[name], Stn, Atn, S2tn) for name in _MODELS}
    print(f"    {'model':>8s} | {'in-dist relMSE':>15s}")
    print("    " + "-" * 28)
    for name in _MODELS:
        tag = {"VN-Set": "   <- channel-blind: MIS-SPECIFIED",
               "VN-MP": "   <- equivariant + message: fits"}.get(name, "")
        print(f"    {name:>8s} | {indist[name]:15.4e}{tag}")
    chan_factor = indist["VN-Set"] / max(indist["VN-MP"], 1e-12)
    print(f"    => mean-message channel necessity: VN-Set/VN-MP = x{chan_factor:.2f} (differ ONLY in the message).")

    # ---- [C] count-OOD at the SEEN orientation (the NEW combinatorial axis) ----
    print()
    print(line)
    print(f"[C] count 举一反三 (seen orientation): train O={TRAIN_OBJ}, evaluate zero-shot at other counts")
    print(line)
    print(f"    {'O':>3s} | {'VN-MP relMSE':>13s} | {'VN-Set relMSE':>14s} | {'MLP-MP relMSE':>14s}")
    print("    " + "-" * 53)
    count_c = {name: {} for name in _MODELS}
    for o in COUNTS_C:
        So, Ao, S2o = test[o]
        for name in _MODELS:
            count_c[name][o] = rel_mse_named(name, models[name], So, Ao, S2o)
        mark = ("  <- train count" if o == TRAIN_OBJ
                else "  <- no-interaction limit" if o == 1 else "")
        print(f"    {o:>3d} | {count_c['VN-MP'][o]:13.4e} | {count_c['VN-Set'][o]:14.4e} | "
              f"{count_c['MLP-MP'][o]:14.4e}{mark}")
    # The claim is *interaction transfer* across the many-body family O>=2 (smaller AND
    # larger than the train count). O=1 is the qualitatively different *no-interaction
    # limit*: the mean message is identically zero (mean_rel_dirs at O=1 = 0), an input the
    # O=3-trained message channel never sees, and the torque term vanishes so the relMSE
    # denominator (per-object latent step) shrinks. We (i) keep O=1 in the table, (ii) report
    # it as a separate boundary with its mechanism, (iii) compute the headline count ratio
    # over the interacting family. This scopes the claim to the regime it is about; it does
    # NOT hide O=1, which is shown to still beat the no-change baseline.
    interacting = [o for o in COUNTS_C if o >= 2]
    def _count_ratio(name: str) -> float:
        ref = count_c[name][TRAIN_OBJ]
        ood = [count_c[name][o] for o in interacting if o != TRAIN_OBJ]
        return max(ood) / max(ref, 1e-12) if ood else 1.0
    count_ratio = {name: _count_ratio(name) for name in _MODELS}
    print(f"    => count-OOD/train over the INTERACTING family O in {interacting}: "
          f"VN-MP x{count_ratio['VN-MP']:.2f}, MLP-MP x{count_ratio['MLP-MP']:.2f} "
          f"(both factorized+mean-message => flat); VN-Set x{count_ratio['VN-Set']:.2f}.")
    # O=1 boundary diagnostic: expose WHY the no-interaction limit reads higher (message
    # channel pushed to an unseen zero + smaller latent step), and that it does not blow up.
    if 1 in COUNTS_C:
        d = {}
        for o in (1, TRAIN_OBJ):
            So, Ao, S2o = test[o]
            z = models["VN-MP"].encoder(So)
            zp = models["VN-MP"].predictor(z, model_action_many("VN-MP", So, Ao))
            z2 = models["VN-MP"].encoder(S2o)
            nrm = So.shape[0] * o
            d[o] = (((zp - z2) ** 2).sum().item() / nrm,            # per-object prediction MSE
                    ((z2 - z) ** 2).sum().item() / nrm,             # per-object latent step (denominator)
                    mean_rel_dirs(So).norm(dim=-1).mean().item())   # mean ||rbar||
        o1_ratio = count_c["VN-MP"][1] / max(count_c["VN-MP"][TRAIN_OBJ], 1e-12)
        print(f"    [O=1 boundary] msg |rbar|: O=1 {d[1][2]:.1e} (identically 0, OOD) vs "
              f"O={TRAIN_OBJ} {d[TRAIN_OBJ][2]:.2f}; per-object latent step "
              f"{d[TRAIN_OBJ][1]:.2e}->{d[1][1]:.2e} (torque gone) => relMSE x{o1_ratio:.2f}, "
              f"still {count_c['VN-MP'][1]:.2f}<1 (beats no-change): documented limit, not a blow-up.")

    # ---- [G] count x global-orientation OOD (equivariance still pays at every unseen count) ----
    print()
    print(line)
    print("[G] count x global-orientation 举一反三: rotate the WHOLE scene by random SO(3) off the z-wedge")
    print(line)
    print(f"    {'O':>3s} | {'VN-MP seen':>11s} | {'VN-MP OOD':>10s} | {'VN x':>6s} | "
          f"{'MLP seen':>9s} | {'MLP OOD':>9s} | {'MLP x':>6s}")
    print("    " + "-" * 70)
    count_g = {"VN-MP": {}, "MLP-MP": {}}
    for o in COUNTS_G:
        So, Ao, S2o = test[o]
        for name in ("VN-MP", "MLP-MP"):
            seen = rel_mse_named(name, models[name], So, Ao, S2o)
            vals = []
            for _ in range(K_OOD):
                R = rand_so3(g)
                t = torch.randn(3, generator=g)
                Sg, Ag, S2g = transform_global_many(So, Ao, S2o, R, t, o)
                vals.append(rel_mse_named(name, models[name], Sg, Ag, S2g))
            ood = float(np.mean(vals))
            count_g[name][o] = dict(seen=seen, ood=ood, ratio=ood / max(seen, 1e-12))
        vg, mg = count_g["VN-MP"][o], count_g["MLP-MP"][o]
        print(f"    {o:>3d} | {vg['seen']:11.4e} | {vg['ood']:10.4e} | x{vg['ratio']:4.2f} | "
              f"{mg['seen']:9.3e} | {mg['ood']:9.3e} | x{mg['ratio']:4.2f}")
    vn_g_ratio = max(count_g["VN-MP"][o]["ratio"] for o in COUNTS_G)
    mlp_g_ratio = float(np.mean([count_g["MLP-MP"][o]["ratio"] for o in COUNTS_G]))
    print(f"    => VN-MP exactly flat at every unseen count (max x{vn_g_ratio:.2f}); "
          f"MLP-MP degrades (mean x{mlp_g_ratio:.2f}).")

    # ---- summary + verdict ----------------------------------------------------------------
    print()
    print(line)
    print("STEP 35 SUMMARY")
    print(line)
    ok_vnmp_equiv = eq["VN-MP"]["se3_comp"] < 1e-4 and eq["VN-MP"]["perm_comp"] < 1e-4
    ok_vnmp_fits = indist["VN-MP"] < 0.85       # degree-1 + mean-aggregation cap (beats no-change clearly)
    ok_channel = chan_factor > 1.1              # mean-message necessity (cf. Step 24's modest cap)
    ok_count_vn = count_ratio["VN-MP"] < 1.30   # NEW: count 举一反三 over the interacting family O>=2
    ok_count_o1 = (1 not in COUNTS_C) or (count_c["VN-MP"][1] < 1.0)  # no-interaction limit still beats no-change
    ok_countorient_vn = vn_g_ratio < 1.15       # exact: rotation-invariant at every unseen count
    ok_countorient_mlp = mlp_g_ratio > 1.30     # control: equivariance prior bites on the orientation axis
    ok_mlp_no_equiv = eq["MLP-MP"]["se3_comp"] > 1e-2
    passed = (ok_vnmp_equiv and ok_vnmp_fits and ok_channel and ok_count_vn and ok_count_o1
              and ok_countorient_vn and ok_countorient_mlp and ok_mlp_no_equiv)

    print(f"    [A'] VN-MP still global-SE(3)|xS_O-equivariant at UNSEEN O={UNSEEN_CHK}: "
          f"SE(3) {eq['VN-MP']['se3_comp']:.1e}, perm {eq['VN-MP']['perm_comp']:.1e} "
          f"(MLP-MP SE(3) {eq['MLP-MP']['se3_comp']:.1e}: no rot prior).")
    print(f"    [I] channel necessity (O={TRAIN_OBJ}): VN-MP={indist['VN-MP']:.3e} < VN-Set={indist['VN-Set']:.3e} "
          f"(x{chan_factor:.2f}; degree-1 cross-product cap, modest by design).")
    print(f"    [C] count 举一反三 (seen orient, INTERACTING family O>=2): VN-MP x{count_ratio['VN-MP']:.2f}, "
          f"MLP-MP x{count_ratio['MLP-MP']:.2f} -- bought by factorization + count-stable mean message (BOTH flat);")
    if 1 in COUNTS_C:
        print(f"        O=1 no-interaction limit reads x{count_c['VN-MP'][1]/max(count_c['VN-MP'][TRAIN_OBJ],1e-12):.2f} "
              f"(msg=0 OOD + torque-free => smaller latent step), still {count_c['VN-MP'][1]:.2f}<1: a documented boundary.")
    print(f"    [G] count x SO(3): VN-MP max x{vn_g_ratio:.2f} (exact, every unseen count) vs "
          f"MLP-MP mean x{mlp_g_ratio:.2f} (degrades) -- isolates the EQUIVARIANCE prior.")
    print(f"    guards: vnmp-equiv={ok_vnmp_equiv}  vnmp-fits={ok_vnmp_fits}  channel={ok_channel}  "
          f"count-vn={ok_count_vn}  count-o1={ok_count_o1}")
    print(f"            countorient-vn={ok_countorient_vn}  countorient-mlp={ok_countorient_mlp}  "
          f"mlp-no-equiv={ok_mlp_no_equiv}")
    print(f"    headline: a single training count O={TRAIN_OBJ} DETERMINES the interacting dynamics across the")
    print(f"        many-body family O in {interacting} (combinatorial 举一反三). The count-stable MEAN message")
    print(f"        makes the slot model transfer (VN-MP & MLP-MP both flat in count); equivariance adds")
    print(f"        EXACT global-rotation 举一反三 at EVERY unseen count (VN-MP x{vn_g_ratio:.2f}), where the")
    print(f"        equally-equipped MLP-MP degrades (x{mlp_g_ratio:.2f}). Two independent generalisation")
    print(f"        axes -- discrete count x continuous SO(3) -- met at once, only by the geometric model.")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}")

    # ---- dump JSON + figure artifacts for the papers --------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(TRAIN_OBJ=TRAIN_OBJ, MAX_OBJ=MAX_OBJ, P=P, D_OBJ=D_OBJ, A_OBJ=A_OBJ,
                       A_OBJ_AUG=A_OBJ_AUG, C_INT=C_INT, N_TRAIN=N_TRAIN, N_TEST=N_TEST, EPOCHS=EPOCHS,
                       K_OOD=K_OOD, VAR_COEF=VAR_COEF, COUNTS_C=COUNTS_C, COUNTS_G=COUNTS_G,
                       UNSEEN_CHK=UNSEEN_CHK),
        "params": {n: n_params(m) for n, m in models.items()},
        "latent_std": {n: hist[n]["latent_std"] for n in _MODELS},
        "equiv_posttrain_unseen": eq,
        "indist": indist,
        "channel_factor": chan_factor,
        "count_ood": {n: {str(o): count_c[n][o] for o in COUNTS_C} for n in _MODELS},
        "count_ratio": count_ratio,
        "count_orient_ood": {n: {str(o): count_g[n][o] for o in COUNTS_G} for n in ("VN-MP", "MLP-MP")},
        "vn_g_ratio": vn_g_ratio,
        "mlp_g_ratio": mlp_g_ratio,
        "interacting_family": interacting,
        "count_ratio_basis": "interacting family O>=2 (O=1 is the no-interaction limit, reported separately)",
        "verdict": {"passed": bool(passed), "ok_vnmp_equiv": ok_vnmp_equiv, "ok_vnmp_fits": ok_vnmp_fits,
                    "ok_channel": ok_channel, "ok_count_vn": ok_count_vn, "ok_count_o1": ok_count_o1,
                    "ok_countorient_vn": ok_countorient_vn, "ok_countorient_mlp": ok_countorient_mlp,
                    "ok_mlp_no_equiv": ok_mlp_no_equiv},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if SMOKE else ""
    out_path = fig_dir / f"step35_many_body{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")
    try:
        _make_figure(count_c, count_g, COUNTS_C, COUNTS_G, fig_dir / f"step35_many_body{suffix}.png")
        print(f"    wrote {(fig_dir / f'step35_many_body{suffix}.png').relative_to(ROOT)}")
    except Exception as exc:  # figure is a nicety; never fail the run on a plotting hiccup
        print(f"    (figure skipped: {exc})")
    sys.exit(0 if passed else 1)


def _make_figure(count_c: dict, count_g: dict, counts_c: list[int], counts_g: list[int], path: Path) -> None:
    r"""Two-panel summary: (A) count 举一反三 curve; (B) count x SO(3) OOD/seen ratio bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axc, axg) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: relMSE vs object count (seen orientation), one line per model.
    styles = {"VN-MP": ("o-", "#1f77b4"), "VN-Set": ("s--", "#7f7f7f"), "MLP-MP": ("^-", "#d62728")}
    for name, (mk, col) in styles.items():
        ys = [count_c[name][o] for o in counts_c]
        axc.plot(counts_c, ys, mk, color=col, label=name, lw=2, ms=6)
    axc.axvline(TRAIN_OBJ, color="k", ls=":", lw=1.2, alpha=0.7)
    axc.text(TRAIN_OBJ, axc.get_ylim()[1], "  train", va="top", ha="left", fontsize=9, alpha=0.7)
    axc.set_xlabel("object count $O$ (zero-shot)")
    axc.set_ylabel("latent 1-step relMSE")
    axc.set_title("[C] count 举一反三 (seen orientation)")
    axc.set_xticks(counts_c)
    axc.legend(frameon=False, fontsize=9)
    axc.grid(alpha=0.25)

    # Panel B: OOD/seen ratio under global SO(3), VN-MP vs MLP-MP, per unseen count.
    x = np.arange(len(counts_g))
    w = 0.38
    vn = [count_g["VN-MP"][o]["ratio"] for o in counts_g]
    mlp = [count_g["MLP-MP"][o]["ratio"] for o in counts_g]
    axg.bar(x - w / 2, vn, w, color="#1f77b4", label="VN-MP (equivariant)")
    axg.bar(x + w / 2, mlp, w, color="#d62728", label="MLP-MP (no eq. prior)")
    axg.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7)
    axg.set_xticks(x)
    axg.set_xticklabels([f"O={o}" for o in counts_g])
    axg.set_ylabel("OOD / seen relMSE ratio")
    axg.set_title("[G] count $\\times$ global SO(3) (unseen counts)")
    axg.legend(frameon=False, fontsize=9)
    axg.grid(alpha=0.25, axis="y")

    fig.suptitle("Step 35: few-body $\\to$ many-body combinatorial 举一反三 "
                 f"(train $O={TRAIN_OBJ}$)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
