r"""Step 43: the **encoder-capacity ladder + oracle bypass** — localise the residual interaction cap.

Where Steps 32 and 42 left us
-----------------------------
The interacting teacher (Step 24) applies a degree-3 torque
$\kappa\,(\hat r_{ij}\times a_i)\times\tilde x_k$ with the **unit** relative direction
$\hat r_{ij}=r_{ij}/\lVert r_{ij}\rVert$ and the per-object **centred points** $\tilde x_k=x_k-\bar x$.
Two levers have now been swept to recover the cap a degree-1 Vector-Neuron predictor leaves open, and
**both saturate**:

  * Step 32 — the **predictor degree** ladder ($d_{\max}=2^L$): recovery saturates after the first
    cross-product rung; climbing representable degree past $d_{\max}=2$ buys nothing.
  * Step 42 — the **message content** ladder (raw $r$ / unit $\hat r$ / both): a NULL. Supplying the
    very normalisation $1/\lVert r\rVert$ the homogeneous predictor cannot form does not measurably
    beat the raw message (x1.01, within seed noise).

Both levers stall at the same ~0.2-0.27 in-distribution relMSE floor, far above the unconstrained
MLP's ~0.074. By elimination the residual cap should live in the **third** component — the shared
:class:`SetSE3Encoder`, whose translation-invariant latent is **lossy**: it pools $P{=}24$ points per
object down to $N_{\text{out}}{=}16$ type-1 vectors ($\ell_{\max}{=}2$, mul $8$). If the per-object
geometry $\{\tilde x_k\}$ the torque needs is not faithfully carried in that latent, no predictor and
no message can recover it. This step tests that localisation **directly**, two ways.

(A) encoder-capacity ladder
---------------------------
Hold the VN-TP predictor, the raw message (M0 $[a, r]$), the teacher, the data and the training
**fixed**, and sweep only the encoder's *internal* capacity at a **fixed latent budget**
($N_{\text{out}}{=}16$, $D_{\text{scene}}{=}96$ unchanged):

    E0  lmax=2 mul=8   (the Step 27/32/42 baseline)
    E1  lmax=2 mul=16
    E2  lmax=2 mul=32
    E3  lmax=3 mul=8

If a richer encoder at the **same output budget** dropped the floor, the cap would be encoder
*under-capacity*; if it saturates (the expected echo of Steps 32/42), the cap is the **output budget**
itself — the lossy projection onto 16 vectors — which (A) holds fixed and so cannot probe. (B) does.

(B) the oracle / encoder-bypass control — the decisive test
-----------------------------------------------------------
Replace the learned encoder with a **lossless, parameter-free** one: the per-object **centred point
cloud** $\tilde x_k=x_k-\bar x$ itself ($D_{\text{obj}}{=}P\cdot 3{=}72$). This is still exactly
SE(3)-equivariant (translation-invariant via centering, $R\tilde x_k$ under rotation) and
permutation-covariant — a legitimate equivariant model, just with a latent that **loses nothing**.
Feed it to the SAME degree-3 VN-TP predictor:

    ORACLE-raw   lossless points + raw  message  [a, r]
    ORACLE-unit  lossless points + unit message  [a, r_hat]   <- the best case

With the lossless geometry $\{\tilde x_k\}$ in the latent AND the normalisation $\hat r$ supplied by the
message, the predictor has every primitive the teacher uses: it can form $\hat r_{ij}\times a_i$
(block 1) then $\,\cdot\,\times\tilde x_k$ (block 2), exactly — the degree-3 torque is in its class.

The falsifiable prediction
--------------------------
If the cap is the encoder's lossy latent, **ORACLE-unit should substantially close the E0->MLP gap**
(the predictor class fits the interaction once the geometry is lossless), while the encoder ladder (A)
saturates. If instead ORACLE-unit *also* stalls at ~0.2, the localisation is WRONG — the cap is
intrinsic to the equivariant predictor *class* on this task, not the encoder — and Step 43 reports
INCONCLUSIVE, no guard loosened.

Caveat (stated honestly). relMSE is measured in *each model's own latent space*: the oracle's is point
space, the encoders' is the 96-dim learned latent. So the oracle number is **not** an apples-to-apples
absolute against E0 — it is read as *"does the lossless model essentially SOLVE the one-step task"*
(relMSE -> the irreducible floor) *or not* (still ~0.2). The normalised relMSE ("fraction of the latent
change left unexplained") makes that near-zero / not-near-zero reading meaningful across spaces, but the
cross-space caveat is why ORACLE is a *localiser*, not a fourth point on the same recovery curve.

Across-group [G] and equivariance [A'] are checked at every variant — the oracle is equivariant too, so
it must stay x1.00 flat and float-floor exact, showing lossless + equivariant + 举一反三 coexist.

Gate (PASS): every VN variant exactly equivariant (init + post-train) AND the lossless oracle closes
>50% of the E0->MLP gap AND every VN variant stays global-flat AND the MLP still degrades.

Run (full; ~30-60 min on a laptop CPU, 3 seeds x 7 models):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step43_encoder_ladder.py
Smoke (~3-5 min):
    STEP43_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step43_encoder_ladder.py
"""

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
sys.path.insert(0, str(HERE))   # for the Step 24 machinery we reuse

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

# Reuse the *validated* Step 24 interacting-teacher bench VERBATIM (same data generator, global-OOD
# transform, whole-pipeline relMSE, the MLP-MP anchor, the raw message builder, and the
# self-action rotation helper). Step 43 changes ONLY the ENCODER (its internal capacity, or the
# lossless point-cloud bypass) and adds the unit-message variant for the oracle; the VN-TP predictor
# (degree-3) and the teacher/data/training are held at the Step-27/42 configuration.
from step24_object_interaction import (  # noqa: E402
    A_OBJ,
    A_SCENE,
    C_INT,
    D_OBJ,
    D_SCENE,
    N_OBJ,
    N_OUT_VEC,
    P,
    EqJEPA,
    SetSE3Encoder,
    SlotPredictor,
    _PERM,
    _rot_self_actions,
    build_mlp_mp,
    build_msg_action,
    make_interacting_transitions,
    n_params,
    permute_scene,
    rand_so3,
    rel_mse,
    rotate_points,
    train_jepa,
    transform_global,
)
from src.models.structured import VNTPPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP43_SMOKE"))


# --------------------------------------------------------------------------- #
# the lossless ORACLE encoder: per-object centred point cloud (parameter-free).
# --------------------------------------------------------------------------- #
class CenteredPointEncoder(nn.Module):
    r"""Parameter-free **lossless** oracle 'encoder': the per-object centred point cloud.

    ``(B, O, P, 3) -> (B, O*P*3)``. Subtracting each object's centroid $\bar x$ makes the latent
    translation-invariant; a global rotation $R$ maps $\tilde x_k = x_k-\bar x \mapsto R\tilde x_k$,
    so the flattened latent is a stack of type-1 vectors transforming by block-diagonal $R$ — exactly
    SE(3)-equivariant — and the object axis is preserved, so it is permutation-covariant. It carries
    the FULL per-object geometry the degree-3 torque $\kappa(\hat r_{ij}\times a_i)\times\tilde x_k$
    needs, with **zero** learned compression: the encoder-bypass control. It has no parameters, so the
    VICReg variance hinge is inert on it (nothing to collapse — the latent *is* the data) and only the
    predictor trains.
    """

    def __init__(self, n_obj: int, n_pts: int):
        super().__init__()
        self.n_obj = n_obj
        self.n_pts = n_pts
        self.latent_dim = n_obj * n_pts * 3

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        # S: (B, O, P, 3)
        c = S.mean(dim=2, keepdim=True)        # (B, O, 1, 3) per-object centroid
        xt = S - c                              # (B, O, P, 3) centred -> SE(3)-equivariant
        return xt.reshape(S.shape[0], -1)       # (B, O*P*3)


# --------------------------------------------------------------------------- #
# messages: raw [a, r] (reused VERBATIM from Step 24) and unit [a, r_hat] (the Step-42 M1 content).
# Both are width A_OBJ+3 = 6 per object, so every variant has the SAME action width.
# --------------------------------------------------------------------------- #
def _unit(r: torch.Tensor) -> torch.Tensor:
    r"""$\hat r = r/\lVert r\rVert$ — the non-polynomial normalisation no VN predictor degree can form."""
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def build_unit_msg(S: torch.Tensor, A_self: torch.Tensor) -> torch.Tensor:
    r"""Per object $i$: $[\,a_i,\; \hat r_{ij}\,]$, the **unit** relative direction. ``-> (B, O*(A_OBJ+3))``.

    Mirrors :func:`build_msg_action` but normalises the relative centroid. Translation-invariant,
    $R\hat r_{ij}$-equivariant, antisymmetric under the object swap ($\hat r_{ji}=-\hat r_{ij}$) — so
    the whole VN pipeline stays jointly $\mathrm{SE}(3)\rtimes S_O$-equivariant.
    """
    b = S.shape[0]
    c = S.mean(dim=2)                                          # (B, O, 3) input centroids
    Aself = A_self.reshape(b, N_OBJ, A_OBJ)
    blocks = []
    for i in range(N_OBJ):
        j = 1 - i
        blocks.append(torch.cat([Aself[:, i], _unit(c[:, j] - c[:, i])], dim=-1))
    return torch.stack(blocks, dim=1).reshape(b, -1)


# --------------------------------------------------------------------------- #
# the variant ladder. ENC_RUNGS sweep the encoder's internal capacity at a fixed latent
# budget (raw message); ORACLES bypass the encoder with the lossless centred cloud
# (raw + unit message); MLP-MP is the unconstrained, non-equivariant ceiling.
# --------------------------------------------------------------------------- #
VARIANTS = {
    "E0-base":     dict(kind="enc", lmax=2, mul=8,  msg="raw",  label="enc lmax2 mul8 [a,r]"),
    "E1-mul16":    dict(kind="enc", lmax=2, mul=16, msg="raw",  label="enc lmax2 mul16 [a,r]"),
    "E2-mul32":    dict(kind="enc", lmax=2, mul=32, msg="raw",  label="enc lmax2 mul32 [a,r]"),
    "E3-lmax3":    dict(kind="enc", lmax=3, mul=8,  msg="raw",  label="enc lmax3 mul8 [a,r]"),
    "ORACLE-raw":  dict(kind="oracle",              msg="raw",  label="points [a,r]"),
    "ORACLE-unit": dict(kind="oracle",              msg="unit", label="points [a,r_hat]"),
    "MLP-MP":      dict(kind="mlp",                 msg="raw",  label="MLP [a,r]"),
}
ENC_RUNGS = ("E0-base", "E1-mul16", "E2-mul32", "E3-lmax3")
ORACLES = ("ORACLE-raw", "ORACLE-unit")
VN_VARIANTS = ENC_RUNGS + ORACLES           # every equivariant variant
MLP_ANCHOR = "MLP-MP"
NAMES = list(VN_VARIANTS) + [MLP_ANCHOR]

AUG = A_OBJ + 3                              # per-object action width (one extra type-1 vector); = 6


def _d_obj(name: str) -> int:
    r"""Per-object latent width: ``P*3 = 72`` for the lossless oracle, else ``D_OBJ = 48``."""
    return P * 3 if VARIANTS[name]["kind"] == "oracle" else D_OBJ


def build_model(name: str) -> EqJEPA:
    r"""Build a variant. Encoder rungs share the Step-42 VN-TP predictor at $D_{\text{obj}}{=}48$; the
    oracle uses the same VN-TP class at $D_{\text{obj}}{=}72$ (the lossless point latent); MLP-MP is the
    non-equivariant ceiling. Action width is ``AUG = 6`` everywhere (raw or unit relative vector)."""
    v = VARIANTS[name]
    if v["kind"] == "mlp":
        return build_mlp_mp()                        # unconstrained ceiling (no equivariance)
    if v["kind"] == "oracle":
        d_obj = P * 3                                # 72: lossless centred points, no compression
        enc = CenteredPointEncoder(N_OBJ, P)
        pred = SlotPredictor(VNTPPredictor(d_obj, AUG, hidden=64, dim=3),
                             n_obj=N_OBJ, d_obj=d_obj, a_obj=AUG)
        return EqJEPA(latent_dim=N_OBJ * d_obj, action_dim=N_OBJ * AUG, encoder=enc, predictor=pred)
    # encoder-capacity rung: vary lmax/mul, hold the latent budget (N_OUT_VEC=16, D_SCENE=96) fixed
    enc = SetSE3Encoder(N_OBJ, n_out_vec=N_OUT_VEC, lmax=v["lmax"], mul=v["mul"])
    pred = SlotPredictor(VNTPPredictor(D_OBJ, AUG, hidden=64, dim=3), a_obj=AUG)
    return EqJEPA(latent_dim=D_SCENE, action_dim=N_OBJ * AUG, encoder=enc, predictor=pred)


def model_action_v(name: str, S: torch.Tensor, A_self: torch.Tensor) -> torch.Tensor:
    r"""The action each model is fed: the unit message for ORACLE-unit, else the raw Step-24 message
    (which is also exactly what MLP-MP consumes — its unconstrained MLP can form $1/\lVert r\rVert$ on
    its own, so the raw message is its true ceiling)."""
    if VARIANTS[name]["msg"] == "unit":
        return build_unit_msg(S, A_self)
    return build_msg_action(S, A_self)               # raw [a, r] (M0)


# --------------------------------------------------------------------------- #
# width-generic equivariance probes (the Step 24/42 helpers hard-code D_OBJ=48; the
# oracle latent is 72-wide, so we parameterise by per-object width). For a GLOBAL rotation
# all objects share R, so rho(R) on the scene latent is just "treat the whole latent as
# stacked 3-vectors and rotate" — width-agnostic.
# --------------------------------------------------------------------------- #
def _rotate_all(z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Global $\rho(R)$: treat the flattened latent as stacked 3-vectors and rotate each. Works for
    any latent width divisible by 3 (oracle 144, encoder 96, MLP 96)."""
    b = z.shape[0]
    return rotate_points(z.reshape(b, -1, 3), R).reshape(b, -1)


def _perm_latent(z: torch.Tensor, d_obj: int) -> torch.Tensor:
    r"""Relabel the two object blocks of a ``(B, N_OBJ*d_obj)`` latent."""
    b = z.shape[0]
    return z.reshape(b, N_OBJ, d_obj)[:, _PERM].reshape(b, -1)


@torch.no_grad()
def se3_resid(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor,
              R: torch.Tensor, t: torch.Tensor) -> float:
    r"""Whole-pipeline global $\mathrm{SE}(3)$ residual, message recomputed from the transformed scene:
    $\max\lVert\rho(R)f(E(S),m(S,a)) - f(E(RS{+}t), m(RS{+}t, Ra))\rVert_\infty$. Float-floor for the
    equivariant variants (encoder rungs AND the lossless oracle); large for the non-equivariant MLP."""
    m = model_action_v(name, S, A_self)
    lhs = _rotate_all(model.predictor(model.encoder(S), m), R)
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    m_r = model_action_v(name, Sr, _rot_self_actions(A_self, R))
    rhs = model.predictor(model.encoder(Sr), m_r)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def perm_resid(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor) -> float:
    r"""Whole-pipeline permutation residual: relabelling objects permutes the message blocks
    ($r_{ji}=-r_{ij}$, $\hat r_{ji}=-\hat r_{ij}$), exact for the shared-weight slot model."""
    b = S.shape[0]
    d_obj = _d_obj(name)
    m = model_action_v(name, S, A_self)
    lhs = _perm_latent(model.predictor(model.encoder(S), m), d_obj)
    Sp = permute_scene(S)
    A_self_p = A_self.reshape(b, N_OBJ, A_OBJ)[:, _PERM].reshape(b, A_SCENE)
    m_p = model_action_v(name, Sp, A_self_p)
    rhs = model.predictor(model.encoder(Sp), m_p)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def indist_relmse(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor,
                  S2: torch.Tensor) -> float:
    return rel_mse(model, S, model_action_v(name, S, A_self), S2)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 80

    if SMOKE:
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0,), 150, 64, 3, 2
    else:
        # EPOCHS=80 (vs Step 42's 60): the lossless oracle's 144-dim point-space regression is harder
        # than the 96-dim learned-latent one, so we give EVERY variant extra headroom (uniform => fair)
        # and verify convergence explicitly below (a loss plateau over the last 20% of training). This
        # de-risks the decisive failure mode: an UNDER-trained oracle would look falsely capped and could
        # flip the verdict to a spurious INCONCLUSIVE.
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0, 1, 2), 1500, 400, 80, 6
    SEEDS = tuple(int(s) for s in os.environ.get("STEP43_SEEDS", ",".join(map(str, SEEDS))).split(","))
    N_TRAIN = int(os.environ.get("STEP43_NTRAIN", N_TRAIN))
    EPOCHS = int(os.environ.get("STEP43_EPOCHS", EPOCHS))
    VAR_COEF = 0.1

    print(line)
    print(f"STEP 43  encoder-capacity ladder + ORACLE bypass: localise the residual interaction cap  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    same interacting teacher / VN-TP predictor as Step 27/32/42: O={N_OBJ} objs, "
          f"P={P} pts, kappa={C_INT}")
    print(f"    teacher torque omega_i=(r_hat_ij x a_i); Steps 32 (predictor degree) and 42 (message)")
    print(f"    BOTH saturated ~0.2-0.27 >> MLP ~0.074 -> by elimination the cap is the ENCODER's latent.")
    print(f"    (A) encoder ladder (raw msg, latent budget FIXED at N_out={N_OUT_VEC}, D_scene={D_SCENE}):")
    for nm in ENC_RUNGS:
        v = VARIANTS[nm]
        print(f"          {nm:>10s}  lmax={v['lmax']} mul={v['mul']:<2d}  [a, r]")
    print(f"    (B) ORACLE bypass (lossless centred points, D_obj={P * 3}; the decisive control):")
    print(f"          {'ORACLE-raw':>10s}  points + [a, r]      -- lossless geometry, raw message")
    print(f"          {'ORACLE-unit':>10s}  points + [a, r_hat]  -- lossless geometry + normalisation (best case)")
    print(f"    seeds={SEEDS}  N_train={N_TRAIN}  epochs={EPOCHS}  K_ood={K_OOD}")

    # ---- fixed eval data shared across seeds (paired seen vs global-OOD) ------------------
    St, At_self, S2t = make_interacting_transitions(N_TEST, seed=999)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    t_chk = torch.randn(3, generator=torch.Generator().manual_seed(4))

    # ---- param counts --------------------------------------------------------------------
    _probe = {nm: build_model(nm) for nm in NAMES}
    params = {nm: n_params(_probe[nm]) for nm in NAMES}
    print("    params: " + "  ".join(f"{nm}={params[nm]}" for nm in NAMES))
    print(f"            (oracle predictor is wider: it ingests {P * 3}-dim lossless points vs {D_OBJ}-dim latent)")

    # ---- [A'] equivariance AT INIT -------------------------------------------------------
    print()
    print(line)
    print("[A'] equivariance AT INIT (global SE(3) + permutation), message recomputed from transformed scene")
    print(line)
    print(f"    {'variant':>11s} | {'SE(3) resid':>12s} | {'perm resid':>11s}")
    print("    " + "-" * 42)
    for nm in NAMES:
        s = se3_resid(nm, _probe[nm], St, At_self, R_chk, t_chk)
        p = perm_resid(nm, _probe[nm], St, At_self)
        tag = "   <- non-equivariant control" if nm == MLP_ANCHOR else (
            "   <- lossless + still exactly equivariant" if nm == "ORACLE-unit" else "")
        print(f"    {nm:>11s} | {s:12.2e} | {p:11.2e}{tag}")
    del _probe

    # ---- per-seed training + evaluation --------------------------------------------------
    indist = {nm: [] for nm in NAMES}
    ood_ratio = {nm: [] for nm in NAMES}
    se3 = {nm: [] for nm in NAMES}
    perm = {nm: [] for nm in NAMES}
    conv = {nm: [] for nm in NAMES}        # convergence witness: |Δpred_loss| over the last 20% of epochs

    for seed in SEEDS:
        print()
        print(line)
        print(f"[seed {seed}] build fresh models, train one-step JEPA on the interacting seen scenes")
        print(line)
        S, A_self, S2 = make_interacting_transitions(N_TRAIN, seed=seed)
        g = torch.Generator().manual_seed(100 + seed)          # the global-OOD rotation stream

        for nm in NAMES:
            torch.manual_seed(seed)                            # paired init across variants
            model = build_model(nm)
            hist = train_jepa(model, S, model_action_v(nm, S, A_self), S2,
                              epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=seed, log_every=999)
            # convergence witness: relative change in pred_loss over the final 20% of training.
            # ~0 => plateaued (trustworthy eval); large => still descending (under-trained, suspect).
            pl = hist["pred_loss"]
            tail = max(1, EPOCHS // 5)
            conv_v = abs(pl[-1] - pl[-tail]) / max(abs(pl[-tail]), 1e-12) if len(pl) > tail else float("nan")
            conv[nm].append(conv_v)
            se3_v = se3_resid(nm, model, St, At_self, R_chk, t_chk)
            perm_v = perm_resid(nm, model, St, At_self)
            seen = indist_relmse(nm, model, St, At_self, S2t)
            ood_vals = []
            for _ in range(K_OOD):
                R = rand_so3(g)
                t = torch.randn(3, generator=g)
                Sg, Ag, S2g = transform_global(St, At_self, S2t, R, t)
                ood_vals.append(indist_relmse(nm, model, Sg, Ag, S2g))
            ood = float(np.mean(ood_vals))
            indist[nm].append(seen)
            ood_ratio[nm].append(ood / max(seen, 1e-12))
            se3[nm].append(se3_v)
            perm[nm].append(perm_v)
            print(f"    {nm:>11s} | in-dist relMSE {seen:.4e} | OOD/seen x{ood / max(seen, 1e-12):6.3f} "
                  f"| SE(3) {se3_v:.1e} | perm {perm_v:.1e} | conv(last20%) {conv_v:5.1%}")

    # ---- aggregate -----------------------------------------------------------------------
    def m(d, nm):
        return float(np.mean(d[nm]))

    def sd(d, nm):
        return float(np.std(d[nm]))

    indist_mean = {nm: m(indist, nm) for nm in NAMES}
    ratio_mean = {nm: m(ood_ratio, nm) for nm in NAMES}
    se3_mean = {nm: m(se3, nm) for nm in NAMES}
    perm_mean = {nm: m(perm, nm) for nm in NAMES}
    conv_mean = {nm: m(conv, nm) for nm in NAMES}

    cap = indist_mean["E0-base"]
    mlp = indist_mean[MLP_ANCHOR]
    gap = cap - mlp
    best_enc = min(indist_mean[nm] for nm in ENC_RUNGS)
    best_enc_nm = min(ENC_RUNGS, key=lambda nm: indist_mean[nm])
    oracle_raw = indist_mean["ORACLE-raw"]
    oracle_unit = indist_mean["ORACLE-unit"]
    enc_closed = (cap - best_enc) / gap if abs(gap) > 1e-9 else float("nan")
    oracle_unit_closed = (cap - oracle_unit) / gap if abs(gap) > 1e-9 else float("nan")
    oracle_raw_closed = (cap - oracle_raw) / gap if abs(gap) > 1e-9 else float("nan")

    # ---- [L] the localisation curve ------------------------------------------------------
    print()
    print(line)
    print("[L] LOCALISATION -- in-distribution relMSE (lower=better); does anything drop the E0 floor?")
    print(line)
    print(f"    {'variant':>11s} | {'config':>18s} | {'in-dist relMSE (mean +/- std)':>30s}")
    print("    " + "-" * 68)
    for nm in ENC_RUNGS:
        tag = "   <- baseline cap (Step 27/32/42)" if nm == "E0-base" else ""
        print(f"    {nm:>11s} | {VARIANTS[nm]['label']:>18s} | {indist_mean[nm]:.4e} +/- {sd(indist, nm):.1e}{tag}")
    print("    " + "-" * 68)
    for nm in ORACLES:
        tag = "   <- the decisive control" if nm == "ORACLE-unit" else ""
        print(f"    {nm:>11s} | {VARIANTS[nm]['label']:>18s} | {indist_mean[nm]:.4e} +/- {sd(indist, nm):.1e}{tag}")
    print(f"    {MLP_ANCHOR:>11s} | {VARIANTS[MLP_ANCHOR]['label']:>18s} | {mlp:.4e} +/- "
          f"{sd(indist, MLP_ANCHOR):.1e}   <- unconstrained ceiling (no equivariance)")
    print(f"    => (A) encoder ladder: best rung {best_enc_nm} {best_enc:.4e} closes "
          f"{100 * enc_closed:.0f}% of the E0->MLP gap (raw msg, latent budget fixed)")
    print(f"       (B) ORACLE-unit (lossless+normalised) {oracle_unit:.4e} closes "
          f"{100 * oracle_unit_closed:.0f}% of the gap; ORACLE-raw {100 * oracle_raw_closed:.0f}%")
    print(f"       [cross-space caveat: oracle relMSE is in POINT space, not the 96-d latent -- read as "
          f"'solved' vs 'still ~{cap:.2f}', not an absolute against E0]")
    # convergence witness: every variant must have plateaued for the eval to be trustworthy; the oracle
    # (harder 144-dim regression) is the one to watch -- an under-trained oracle would look falsely capped.
    conv_max = max(conv_mean[nm] for nm in NAMES)
    print(f"       convergence (|Δpred_loss| last 20% of {EPOCHS} epochs): "
          + "  ".join(f"{nm.split('-')[0]}={conv_mean[nm]:.1%}" for nm in NAMES))
    ok_converged = conv_max < 0.10
    print(f"       => all variants {'PLATEAUED (<10%, eval trustworthy)' if ok_converged else 'NOT all plateaued (>10% -- read with care)'}; "
          f"oracle-unit conv={conv_mean['ORACLE-unit']:.1%}")

    # ---- [G] across-group flatness -------------------------------------------------------
    print()
    print(line)
    print("[G] across-group 举一反三 -- global SE(3) OOD/seen ratio per variant (1.00=flat)")
    print(line)
    print(f"    {'variant':>11s} | {'OOD/seen ratio':>15s}")
    print("    " + "-" * 30)
    for nm in VN_VARIANTS:
        tag = "   <- lossless + equivariant + flat" if nm == "ORACLE-unit" else ""
        print(f"    {nm:>11s} | x{ratio_mean[nm]:13.3f}{tag}")
    print(f"    {MLP_ANCHOR:>11s} | x{ratio_mean[MLP_ANCHOR]:13.3f}   <- degrades (no equivariance)")
    vn_ratio_max = max(ratio_mean[nm] for nm in VN_VARIANTS)
    print(f"    => every VN variant flat (<= x{vn_ratio_max:.2f}); MLP carries "
          f"x{ratio_mean[MLP_ANCHOR]:.1f} -- bypass the encoder, 举一反三 stays free.")

    # ---- [A'] post-train equivariance ----------------------------------------------------
    print()
    print(line)
    print("[A'] post-training equivariance per variant (mean over seeds): VN stays exact, MLP does not")
    print(line)
    print(f"    {'variant':>11s} | {'SE(3) resid':>12s} | {'perm resid':>11s}")
    print("    " + "-" * 42)
    for nm in NAMES:
        tag = "   <- non-equivariant control" if nm == MLP_ANCHOR else ""
        print(f"    {nm:>11s} | {se3_mean[nm]:12.2e} | {perm_mean[nm]:11.2e}{tag}")

    # ---- verdict -------------------------------------------------------------------------
    vn_se3_max = max(se3_mean[nm] for nm in VN_VARIANTS)
    vn_perm_max = max(perm_mean[nm] for nm in VN_VARIANTS)
    ok_equiv = vn_se3_max < 1e-3 and vn_perm_max < 1e-3
    ok_oracle_closes = oracle_unit_closed > 0.5                # decisive: lossless oracle closes >50%
    ok_global_flat = vn_ratio_max < 1.15
    ok_mlp_degrades = ratio_mean[MLP_ANCHOR] > 1.3
    passed = ok_equiv and ok_oracle_closes and ok_global_flat and ok_mlp_degrades

    print()
    print(line)
    print("STEP 43 SUMMARY")
    print(line)
    print(f"    [L] localisation: E0 cap {cap:.3e}; encoder ladder best {best_enc:.3e} "
          f"({100 * enc_closed:.0f}% of gap to MLP {mlp:.3e}); ORACLE-unit {oracle_unit:.3e} "
          f"({100 * oracle_unit_closed:.0f}% of gap).")
    print(f"    [G] across-group OOD/seen: VN variants <= x{vn_ratio_max:.2f} (flat) vs MLP "
          f"x{ratio_mean[MLP_ANCHOR]:.1f} (degrades).")
    print(f"    [A'] post-train equivariance: VN SE(3) <= {vn_se3_max:.1e}, perm <= {vn_perm_max:.1e} "
          f"(exact at every variant, oracle included).")
    print(f"    guards: equiv={ok_equiv}  oracle-closes={ok_oracle_closes}  global-flat={ok_global_flat}  "
          f"mlp-degrades={ok_mlp_degrades}")
    print(f"    headline: Steps 32 (predictor degree) and 42 (message) both SATURATED ~0.2-0.27. By")
    print(f"        elimination the residual interaction cap should be the ENCODER's lossy latent. Step 43")
    print(f"        tests it: (A) richer encoders at a FIXED latent budget, (B) a lossless point-cloud")
    print(f"        ORACLE fed the same degree-3 VN-TP predictor.")
    if passed:
        print(f"    CONFIRMED: the encoder-capacity ladder saturates ({100 * enc_closed:.0f}% of the gap), but")
        print(f"        the lossless ORACLE closes {100 * oracle_unit_closed:.0f}% -- so the residual cap is the")
        print(f"        ENCODER's lossy translation-invariant latent (the 16-vector output budget), NOT the")
        print(f"        predictor degree (Step 32) or the message (Step 42). The triangulation is complete.")
        print(f"        And the oracle stays exactly equivariant (SE(3) {se3_mean['ORACLE-unit']:.1e}) and flat")
        print(f"        (x{ratio_mean['ORACLE-unit']:.2f}): lossless + equivariant + 举一反三 coexist.")
    else:
        print(f"    INCONCLUSIVE: the lossless ORACLE did NOT close the gap (ORACLE-unit "
              f"{100 * oracle_unit_closed:.0f}% of E0->MLP). So the residual interaction cap is NOT")
        print(f"        (only) the encoder's lossy latent -- with Steps 32 and 42 also saturating, the cap")
        print(f"        is intrinsic to the equivariant predictor CLASS on this task (the homogeneous")
        print(f"        Vector-Neuron dynamics), not any single removable component. Reported as-is, no")
        print(f"        guard loosened. Equivariance (<= {vn_se3_max:.0e}) and flatness (x{vn_ratio_max:.2f})")
        print(f"        stay exact at every variant, the oracle included -- the prior is never the cost.")

    # ---- dump JSON + figure --------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(ENC_RUNGS=list(ENC_RUNGS), ORACLES=list(ORACLES),
                       labels={k: VARIANTS[k]["label"] for k in VARIANTS},
                       SEEDS=list(SEEDS), N_TRAIN=N_TRAIN, N_TEST=N_TEST, EPOCHS=EPOCHS, K_OOD=K_OOD,
                       VAR_COEF=VAR_COEF, C_INT=C_INT, AUG=AUG, D_OBJ=D_OBJ, D_OBJ_ORACLE=P * 3),
        "params": params,
        "indist_per_seed": indist,
        "ood_ratio_per_seed": ood_ratio,
        "se3_per_seed": se3,
        "perm_per_seed": perm,
        "indist_mean": indist_mean,
        "indist_std": {nm: sd(indist, nm) for nm in NAMES},
        "ood_ratio_mean": ratio_mean,
        "se3_mean": se3_mean,
        "perm_mean": perm_mean,
        "conv_per_seed": conv,
        "conv_mean": conv_mean,
        "ok_converged": bool(conv_max < 0.10),
        "cap": cap, "mlp": mlp, "gap": gap,
        "best_enc": best_enc, "best_enc_variant": best_enc_nm, "enc_gap_closed_frac": enc_closed,
        "oracle_raw": oracle_raw, "oracle_unit": oracle_unit,
        "oracle_raw_gap_closed_frac": oracle_raw_closed,
        "oracle_unit_gap_closed_frac": oracle_unit_closed,
        "verdict": {"passed": bool(passed), "ok_equiv": ok_equiv, "ok_oracle_closes": ok_oracle_closes,
                    "ok_global_flat": ok_global_flat, "ok_mlp_degrades": ok_mlp_degrades},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step43_encoder_ladder_smoke" if SMOKE else "step43_encoder_ladder"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    _make_figure(fig_dir / f"{stem}.png", indist, indist_mean, ratio_mean, se3_mean, cap, mlp)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)}")
    print(f"    wrote {(fig_dir / f'{stem}.png').relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


def _make_figure(path, indist, indist_mean, ratio_mean, se3_mean, cap, mlp) -> None:
    r"""Three panels: localisation bars (encoder ladder + oracle), across-group flatness, exactness."""
    xs = np.arange(len(VN_VARIANTS))
    # encoder rungs in blue, oracle in green (the decisive controls)
    colors = ["C0"] * len(ENC_RUNGS) + ["C2"] * len(ORACLES)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A: localisation curve
    means = [indist_mean[nm] for nm in VN_VARIANTS]
    stds = [float(np.std(indist[nm])) for nm in VN_VARIANTS]
    ax[0].bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax[0].axhline(mlp, ls="--", color="C3", label=f"MLP-MP ceiling ({mlp:.3f})")
    ax[0].axhline(cap, ls=":", color="C7", label=f"E0 cap ({cap:.3f})")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([nm for nm in VN_VARIANTS], rotation=30, ha="right", fontsize=7)
    ax[0].set_ylabel("in-distribution relMSE")
    ax[0].set_title("[L] localisation: encoder ladder (blue) vs lossless oracle (green)")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].grid(alpha=0.3, axis="y")

    # Panel B: across-group OOD/seen ratio per variant
    ratios = [ratio_mean[nm] for nm in VN_VARIANTS]
    ax[1].bar(xs, ratios, color=colors, alpha=0.85)
    ax[1].axhline(ratio_mean[MLP_ANCHOR], ls="--", color="C3",
                  label=f"MLP-MP (x{ratio_mean[MLP_ANCHOR]:.1f})")
    ax[1].axhline(1.0, ls=":", color="k", alpha=0.5, label="flat (x1.00)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([nm for nm in VN_VARIANTS], rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("global OOD/seen ratio")
    ax[1].set_title("[G] across-group 举一反三 (flat at every variant)")
    ax[1].legend(fontsize=7, loc="best")
    ax[1].grid(alpha=0.3, which="both", axis="y")

    # Panel C: post-train equivariance residual per variant
    se3s = [se3_mean[nm] for nm in VN_VARIANTS]
    ax[2].bar(xs, se3s, color=colors, alpha=0.85)
    ax[2].axhline(se3_mean[MLP_ANCHOR], ls="--", color="C3", label="MLP-MP")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([nm for nm in VN_VARIANTS], rotation=30, ha="right", fontsize=7)
    ax[2].set_ylabel("post-train SE(3) residual")
    ax[2].set_title("[A'] exact equivariance (oracle included)")
    ax[2].legend(fontsize=7, loc="best")
    ax[2].grid(alpha=0.3, which="both", axis="y")

    fig.suptitle("Step 43 — encoder ladder + lossless oracle: localising the residual interaction cap",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
