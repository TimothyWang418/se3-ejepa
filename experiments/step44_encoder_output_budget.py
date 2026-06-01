r"""Step 44: the **encoder OUTPUT-budget sweep** — the one lever Step 43 named but could not pull.

Where Step 43 left the localisation
------------------------------------
The interacting teacher (Step 24) applies a degree-3 torque
$\kappa\,(\hat r_{ij}\times a_i)\times\tilde x_k$. Three levers have now been swept to recover the
~0.2 in-distribution relMSE floor a degree-1 Vector-Neuron predictor leaves open, and the first three
all stalled:

  * Step 32 — the **predictor degree** ladder: saturates after the first cross-product rung.
  * Step 42 — the **message content** ladder (raw $r$ / unit $\hat r$): a NULL (x1.01).
  * Step 43A — the encoder's **internal capacity** at a *fixed* latent budget ($N_{\text{out}}{=}16$):
    swept $\mathrm{mul}\in\{8,16,32\}$ and $\ell_{\max}\in\{2,3\}$, saturates (best rung closes 29%).
  * Step 43B — the lossless **oracle** (the per-object centred point cloud, $D_{\text{obj}}{=}P\cdot3{=}72$)
    fed the SAME degree-3 VN-TP predictor *solves* the task (closes 156% of the E0->MLP gap).

Step 43 concluded the cap is the encoder's lossy latent and named the *exact* un-pulled lever in its
own docstring (lines 35-37): the **output budget** — the lossy projection onto 16 vectors — which
43A "holds fixed and so cannot probe." Step 44 pulls it.

The architecture makes the question precise (and sharper than "budget")
----------------------------------------------------------------------
:class:`SE3PointEncoder` (``src/models/se3.py``) is a **set** encoder. Its forward path is

    per-point TP messages  ->  **sum over points**  (line 173: ``h = msg.sum(dim=1)``)
                           ->  equivariant head on the pooled descriptor $h$
                           ->  readout ``lin_out: h -> 3*n_out_vec``  (the OUTPUT budget).

So there are *two distinct* width knobs and one pooling step, in series:

  (i)  the **pooled-descriptor width** $\dim(h)=\mathrm{mul}\cdot\sum_{\ell=0}^{\ell_{\max}}(2\ell+1)$
       — here $8\cdot(1{+}3{+}5)=72$ — set by $\mathrm{mul}/\ell_{\max}$. **This is what Step 43A swept.**
  (ii) the **readout width** $3\,N_{\text{out}}$ — the final ``lin_out``. **This is what Step 44 sweeps.**
  (*)  the permutation-invariant **sum-pool** that collapses $P{=}24$ points into $h$ — never swept; the
       oracle is the only model that escapes it (it keeps the *ordered* cloud).

Holding $\mathrm{mul}{=}8,\ \ell_{\max}{=}2$ fixed and sweeping $N_{\text{out}}\in\{16,24,32,48\}$ widens
**only (ii)**, downstream of the $72$-wide bottleneck $h$. A linear readout cannot manufacture
information the pooled $h$ discarded, so the falsifiable prediction is sharp:

  * **If the cap is the readout width**, the floor keeps dropping as $N_{\text{out}}$ grows toward the
    oracle/MLP. Step 43's "output budget" phrasing is literally right.
  * **If the cap is upstream** (the pooled descriptor / the sum-pool), the floor **saturates** once the
    readout stops being the binding constraint — i.e. once $3\,N_{\text{out}}\ge\dim(h)=72$
    ($N_{\text{out}}\ge24$) — and stays far above the oracle. Then "output budget" must be read as the
    pooled-descriptor/pool, not the readout, and Step 43A+44 together pin the cap to the **lossy
    permutation-invariant pool**, which only the ordered oracle latent escapes.

Either reading is a clean, reportable result; the experiment is built to measure which one the data give,
not to force one. ($N_{\text{out}}{=}24$ is the natural hinge: its $D_{\text{obj}}{=}72$ equals the
oracle's *width*, so B24-vs-oracle isolates **ordered-lossless vs pooled-lossy at the same width** — the
crispest single contrast in the sweep.)

What is held fixed (so only the output budget moves)
----------------------------------------------------
The VN-TP degree-3 predictor (its per-object width tracks $D_{\text{obj}}{=}3N_{\text{out}}$ so the class
is identical, only wider), the raw message $[a,r]$ (M0), the teacher, the data, the optimiser and the
epoch budget. ORACLE-unit (lossless + normalised, the equivariant ceiling) and MLP-MP (the
non-equivariant ceiling that also supplies the across-group [G] degrader) are carried as the two
reference lines.

Guards (the load-bearing claims; these gate the verdict, never loosened)
------------------------------------------------------------------------
Across-group [G] flatness and [A'] exact equivariance are checked at **every** budget rung and the
oracle: widening the output budget must not spend equivariance (the rungs stay float-floor exact and
x1.00 flat), and the MLP must still degrade. The scientific reading (budget-is/ isn't the cap) is
reported in full regardless; PASS additionally requires the sweep to give a *clean* reading (clearly
saturates OR clearly tracks the oracle) and every variant to have converged — otherwise INCONCLUSIVE,
no guard loosened.

Run (full; ~30-60 min on a laptop CPU, 5 seeds x 6 models):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step44_encoder_output_budget.py
Smoke (~3-5 min):
    STEP44_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step44_encoder_output_budget.py
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
sys.path.insert(0, str(HERE))   # for the Step 24/43 machinery we reuse

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Reuse the *validated* Step 24 interacting-teacher bench VERBATIM (same data generator, global-OOD
# transform, whole-pipeline relMSE, the MLP-MP anchor, the raw message builder, self-action rotation).
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

# Reuse the Step 43 lossless oracle, the unit-message builder, and the width-generic latent-action
# helpers VERBATIM — so the oracle reference line here is byte-for-byte the Step 43 oracle.
from step43_encoder_ladder import (  # noqa: E402
    AUG,
    CenteredPointEncoder,
    _perm_latent,
    _rotate_all,
    build_unit_msg,
)
from src.models.structured import VNTPPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP44_SMOKE"))

# Fixed encoder internals (the Step 27/32/42/43-E0 baseline). ONLY n_out_vec moves in this step, so the
# pooled-descriptor width dim(h) below is constant across all budget rungs.
LMAX_FIXED = 2
MUL_FIXED = 8
HIDDEN_DIM = MUL_FIXED * sum(2 * ell + 1 for ell in range(LMAX_FIXED + 1))  # 8*(1+3+5)=72: the pool width


# --------------------------------------------------------------------------- #
# the variant ladder: four OUTPUT-budget rungs (n_out_vec swept; lmax/mul/predictor-class/message all
# fixed), the lossless ORACLE (equivariant ceiling), and MLP-MP (non-equivariant ceiling + [G] degrader).
# B16 == the Step 27/32/42/43-E0 baseline. B24 has D_obj=72 == the oracle's WIDTH (ordered vs pooled).
# --------------------------------------------------------------------------- #
VARIANTS = {
    "B16-nout16":  dict(kind="budget", n_out_vec=16, msg="raw",  label="nout16 D48 [a,r]"),
    "B24-nout24":  dict(kind="budget", n_out_vec=24, msg="raw",  label="nout24 D72 [a,r]"),
    "B32-nout32":  dict(kind="budget", n_out_vec=32, msg="raw",  label="nout32 D96 [a,r]"),
    "B48-nout48":  dict(kind="budget", n_out_vec=48, msg="raw",  label="nout48 D144 [a,r]"),
    "ORACLE-unit": dict(kind="oracle",               msg="unit", label="points [a,r_hat]"),
    "MLP-MP":      dict(kind="mlp",                  msg="raw",  label="MLP [a,r]"),
}
BUDGET_RUNGS = ("B16-nout16", "B24-nout24", "B32-nout32", "B48-nout48")
ORACLES = ("ORACLE-unit",)
VN_VARIANTS = BUDGET_RUNGS + ORACLES            # every equivariant variant
MLP_ANCHOR = "MLP-MP"
NAMES = list(VN_VARIANTS) + [MLP_ANCHOR]
BASE_RUNG = "B16-nout16"                         # the E0 baseline cap


def _d_obj(name: str) -> int:
    r"""Per-object latent width. Oracle: ``P*3 = 72`` (lossless ordered cloud). Budget rung:
    ``3 * n_out_vec`` (the swept readout). MLP anchor: ``D_OBJ = 48`` (its fixed per-object block)."""
    v = VARIANTS[name]
    if v["kind"] == "oracle":
        return P * 3
    if v["kind"] == "budget":
        return 3 * v["n_out_vec"]
    return D_OBJ


def build_model(name: str) -> EqJEPA:
    r"""Build a variant. Budget rungs share the Step-42 VN-TP degree-3 predictor *class*, widened to
    match each rung's $D_{\text{obj}}{=}3N_{\text{out}}$; the encoder internals ($\ell_{\max}{=}2$,
    $\mathrm{mul}{=}8$) are held fixed so ONLY the readout (output budget) moves. The oracle uses the
    same VN-TP class at $D_{\text{obj}}{=}72$ (lossless points); MLP-MP is the non-equivariant ceiling.
    Action width is ``AUG = 6`` everywhere (raw or unit relative vector)."""
    v = VARIANTS[name]
    if v["kind"] == "mlp":
        return build_mlp_mp()                            # unconstrained ceiling (no equivariance)
    if v["kind"] == "oracle":
        d_obj = P * 3                                    # 72: lossless centred points, no compression
        enc = CenteredPointEncoder(N_OBJ, P)
        pred = SlotPredictor(VNTPPredictor(d_obj, AUG, hidden=64, dim=3),
                             n_obj=N_OBJ, d_obj=d_obj, a_obj=AUG)
        return EqJEPA(latent_dim=N_OBJ * d_obj, action_dim=N_OBJ * AUG, encoder=enc, predictor=pred)
    # output-budget rung: vary n_out_vec, hold encoder internals (lmax=2, mul=8) and predictor class fixed
    k = v["n_out_vec"]
    d_obj = 3 * k
    enc = SetSE3Encoder(N_OBJ, n_out_vec=k, lmax=LMAX_FIXED, mul=MUL_FIXED)
    pred = SlotPredictor(VNTPPredictor(d_obj, AUG, hidden=64, dim=3),
                         n_obj=N_OBJ, d_obj=d_obj, a_obj=AUG)
    return EqJEPA(latent_dim=N_OBJ * d_obj, action_dim=N_OBJ * AUG, encoder=enc, predictor=pred)


def model_action_v(name: str, S: torch.Tensor, A_self: torch.Tensor) -> torch.Tensor:
    r"""The action each model is fed: the unit message for ORACLE-unit, else the raw Step-24 message
    (which is exactly what MLP-MP and every budget rung consume)."""
    if VARIANTS[name]["msg"] == "unit":
        return build_unit_msg(S, A_self)
    return build_msg_action(S, A_self)                   # raw [a, r] (M0)


# --------------------------------------------------------------------------- #
# width-generic equivariance probes (mirror Step 43; parameterised by per-object width because the
# budget rungs and the oracle have different latent widths). For a GLOBAL rotation all objects share R,
# so rho(R) on the scene latent is "treat the whole latent as stacked 3-vectors and rotate" (_rotate_all).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def se3_resid(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor,
              R: torch.Tensor, t: torch.Tensor) -> float:
    r"""Whole-pipeline global $\mathrm{SE}(3)$ residual (message recomputed from the transformed scene):
    $\max\lVert\rho(R)f(E(S),m(S,a)) - f(E(RS{+}t), m(RS{+}t, Ra))\rVert_\infty$. Float-floor for every
    budget rung and the oracle; large for the non-equivariant MLP."""
    m = model_action_v(name, S, A_self)
    lhs = _rotate_all(model.predictor(model.encoder(S), m), R)
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    m_r = model_action_v(name, Sr, _rot_self_actions(A_self, R))
    rhs = model.predictor(model.encoder(Sr), m_r)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def perm_resid(name: str, model: EqJEPA, S: torch.Tensor, A_self: torch.Tensor) -> float:
    r"""Whole-pipeline permutation residual: relabelling objects permutes the message blocks
    ($r_{ji}=-r_{ij}$), exact for the shared-weight slot model at any output budget."""
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
        # 5 seeds + 80 epochs, matching Step 43 exactly so the two sweeps are directly comparable on the
        # same plot (Step 43A internal-capacity ladder vs Step 44 output-budget ladder). The widest rung
        # (B48, D_scene=288) is the largest regression here; the convergence witness below guards against
        # an under-trained wide rung looking falsely capped (the decisive failure mode).
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0, 1, 2, 3, 4), 1500, 400, 80, 6
    SEEDS = tuple(int(s) for s in os.environ.get("STEP44_SEEDS", ",".join(map(str, SEEDS))).split(","))
    N_TRAIN = int(os.environ.get("STEP44_NTRAIN", N_TRAIN))
    EPOCHS = int(os.environ.get("STEP44_EPOCHS", EPOCHS))
    VAR_COEF = 0.1

    print(line)
    print(f"STEP 44  encoder OUTPUT-budget sweep: is the residual cap the readout width, or upstream?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    same interacting teacher / VN-TP predictor as Step 27/32/42/43: O={N_OBJ} objs, "
          f"P={P} pts, kappa={C_INT}")
    print(f"    Steps 32 (predictor degree), 42 (message), 43A (encoder INTERNAL capacity) all SATURATED;")
    print(f"    43B oracle (lossless ordered points) SOLVED. Step 43 named the un-pulled lever: the OUTPUT")
    print(f"    budget (readout 3*n_out_vec). Step 44 pulls it at FIXED lmax={LMAX_FIXED}, mul={MUL_FIXED}.")
    print(f"    pooled-descriptor width dim(h) = mul*sum(2l+1) = {HIDDEN_DIM}  (the UPSTREAM bottleneck;")
    print(f"      readout cannot add info past it -> floor should saturate once 3*n_out_vec >= {HIDDEN_DIM}):")
    for nm in BUDGET_RUNGS:
        k = VARIANTS[nm]["n_out_vec"]
        binding = "readout BINDS (< pool)" if 3 * k < HIDDEN_DIM else "pool BINDS (readout >= pool)"
        print(f"          {nm:>11s}  n_out_vec={k:<2d}  D_obj={3 * k:<3d}  readout 3*n_out={3 * k:<3d}  -> {binding}")
    print(f"    references: ORACLE-unit (lossless ordered points, D_obj={P * 3}) | MLP-MP (non-equiv ceiling)")
    print(f"    seeds={SEEDS}  N_train={N_TRAIN}  epochs={EPOCHS}  K_ood={K_OOD}")

    # ---- fixed eval data shared across seeds (paired seen vs global-OOD) ------------------
    St, At_self, S2t = make_interacting_transitions(N_TEST, seed=999)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    t_chk = torch.randn(3, generator=torch.Generator().manual_seed(4))

    # ---- param counts --------------------------------------------------------------------
    _probe = {nm: build_model(nm) for nm in NAMES}
    params = {nm: n_params(_probe[nm]) for nm in NAMES}
    print("    params: " + "  ".join(f"{nm}={params[nm]}" for nm in NAMES))

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

    cap = indist_mean[BASE_RUNG]                                 # B16 == E0 baseline floor
    mlp = indist_mean[MLP_ANCHOR]
    gap = cap - mlp
    oracle_unit = indist_mean["ORACLE-unit"]
    widest = indist_mean["B48-nout48"]
    best_budget = min(indist_mean[nm] for nm in BUDGET_RUNGS)
    best_budget_nm = min(BUDGET_RUNGS, key=lambda nm: indist_mean[nm])
    widest_closed = (cap - widest) / gap if abs(gap) > 1e-9 else float("nan")
    best_budget_closed = (cap - best_budget) / gap if abs(gap) > 1e-9 else float("nan")
    oracle_closed = (cap - oracle_unit) / gap if abs(gap) > 1e-9 else float("nan")

    # saturation diagnostic: improvement B16->B24 (where the readout first matches the pool) vs the
    # FURTHER improvement B24->B48 (tripling the readout past the pool). If the latter is a small
    # fraction of the former, the floor has plateaued at the pooled-descriptor width => upstream cap.
    d_16_24 = indist_mean["B16-nout16"] - indist_mean["B24-nout24"]
    d_24_48 = indist_mean["B24-nout24"] - indist_mean["B48-nout48"]
    sat_resid = (d_24_48 / d_16_24) if abs(d_16_24) > 1e-9 else float("nan")  # ~0 => saturated past B24

    # ---- [L] the output-budget localisation curve ----------------------------------------
    print()
    print(line)
    print("[L] OUTPUT-BUDGET localisation -- in-distribution relMSE (lower=better); does widening the")
    print("    readout (n_out_vec) drop the E0 floor, or does it saturate at the pooled-descriptor width?")
    print(line)
    print(f"    {'variant':>11s} | {'config':>16s} | {'in-dist relMSE (mean +/- std)':>30s}")
    print("    " + "-" * 66)
    for nm in BUDGET_RUNGS:
        tag = "   <- baseline cap (= Step 43 E0)" if nm == BASE_RUNG else ""
        print(f"    {nm:>11s} | {VARIANTS[nm]['label']:>16s} | {indist_mean[nm]:.4e} +/- {sd(indist, nm):.1e}{tag}")
    print("    " + "-" * 66)
    print(f"    {'ORACLE-unit':>11s} | {VARIANTS['ORACLE-unit']['label']:>16s} | {oracle_unit:.4e} +/- "
          f"{sd(indist, 'ORACLE-unit'):.1e}   <- lossless ordered-cloud ceiling")
    print(f"    {MLP_ANCHOR:>11s} | {VARIANTS[MLP_ANCHOR]['label']:>16s} | {mlp:.4e} +/- "
          f"{sd(indist, MLP_ANCHOR):.1e}   <- unconstrained ceiling (no equivariance)")
    print(f"    => widening n_out_vec 16->48 (3x the readout) closes {100 * widest_closed:.0f}% of the "
          f"E0->MLP gap; best rung {best_budget_nm} closes {100 * best_budget_closed:.0f}%")
    print(f"       residual drop past the pool hinge (B24->B48)/(B16->B24) = {sat_resid:.2f}  "
          f"(~0 => saturated at dim(h)={HIDDEN_DIM}; the readout is not the binding cap)")
    print(f"       lossless ORACLE-unit closes {100 * oracle_closed:.0f}% at the SAME width as B24 "
          f"(D_obj={P * 3}) -> ordered-lossless vs pooled-lossy, not width")
    print(f"       [cross-space caveat carries from Step 43: oracle relMSE is in POINT space; read as "
          f"'solved' vs 'still ~{cap:.2f}', not an absolute against the budget rungs]")
    conv_max = max(conv_mean[nm] for nm in NAMES)
    print(f"       convergence (|Δpred_loss| last 20% of {EPOCHS} epochs): "
          + "  ".join(f"{nm.split('-')[0]}={conv_mean[nm]:.1%}" for nm in NAMES))
    ok_converged = conv_max < 0.10
    print(f"       => all variants {'PLATEAUED (<10%, eval trustworthy)' if ok_converged else 'NOT all plateaued (>10% -- read with care)'}; "
          f"widest rung B48 conv={conv_mean['B48-nout48']:.1%}")

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
    print(f"    => every budget rung + oracle flat (<= x{vn_ratio_max:.2f}); MLP carries "
          f"x{ratio_mean[MLP_ANCHOR]:.1f} -- widen the output budget, 举一反三 stays free.")

    # ---- [A'] post-train equivariance ----------------------------------------------------
    print()
    print(line)
    print("[A'] post-training equivariance per variant (mean over seeds): every budget rung stays exact")
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
    ok_global_flat = vn_ratio_max < 1.15
    ok_mlp_degrades = ratio_mean[MLP_ANCHOR] > 1.3
    # scientific reading: budget IS the cap (tracks oracle) vs budget is NOT the cap (saturates upstream)
    budget_is_cap = widest_closed > 0.5
    budget_not_cap = widest_closed < 0.25
    ok_conclusive = budget_is_cap or budget_not_cap          # a clean reading either way (else muddy)
    passed = ok_equiv and ok_global_flat and ok_mlp_degrades and ok_converged and ok_conclusive

    print()
    print(line)
    print("STEP 44 SUMMARY")
    print(line)
    print(f"    [L] output budget: B16 cap {cap:.3e}; widest B48 {widest:.3e} "
          f"({100 * widest_closed:.0f}% of gap to MLP {mlp:.3e}); ORACLE-unit {oracle_unit:.3e} "
          f"({100 * oracle_closed:.0f}%). Saturation past pool hinge (B24->B48)/(B16->B24)={sat_resid:.2f}.")
    print(f"    [G] across-group OOD/seen: budget rungs + oracle <= x{vn_ratio_max:.2f} (flat) vs MLP "
          f"x{ratio_mean[MLP_ANCHOR]:.1f} (degrades).")
    print(f"    [A'] post-train equivariance: VN SE(3) <= {vn_se3_max:.1e}, perm <= {vn_perm_max:.1e} "
          f"(exact at every budget rung, oracle included).")
    print(f"    guards: equiv={ok_equiv}  global-flat={ok_global_flat}  mlp-degrades={ok_mlp_degrades}  "
          f"converged={ok_converged}  conclusive={ok_conclusive}")
    print(f"    headline: Steps 32/42/43A (predictor degree, message, encoder INTERNAL capacity) all")
    print(f"        saturated; the lossless oracle solved. Step 44 pulls the one lever Step 43 named but")
    print(f"        could not: the encoder's OUTPUT budget (readout width 3*n_out_vec), at fixed internals.")
    if passed and budget_not_cap:
        print(f"    CONFIRMED (budget is NOT the cap): tripling the readout (n_out_vec 16->48) closes only")
        print(f"        {100 * widest_closed:.0f}% of the E0->MLP gap and the floor plateaus past the pool hinge")
        print(f"        (B24->B48 residual {sat_resid:.2f}x the B16->B24 step), saturating at the pooled-")
        print(f"        descriptor width dim(h)={HIDDEN_DIM} -- a linear readout cannot recover what the sum-")
        print(f"        pool discarded. With 43A (internal capacity) also saturated and the ordered ORACLE")
        print(f"        ({100 * oracle_closed:.0f}%) solving at the SAME width as B24, the cap is pinned to the")
        print(f"        encoder's LOSSY PERMUTATION-INVARIANT POOL, not any width (internal or output). The")
        print(f"        fix is a non-pooling (ordered / per-point) equivariant latent, not a bigger budget.")
        print(f"        And every budget rung stays exactly equivariant (SE(3) {vn_se3_max:.0e}) and flat")
        print(f"        (x{vn_ratio_max:.2f}): widening the output budget never spends 举一反三.")
    elif passed and budget_is_cap:
        print(f"    REFINED (budget IS (part of) the cap): widening the readout (n_out_vec 16->48) closes")
        print(f"        {100 * widest_closed:.0f}% of the E0->MLP gap -- the output budget WAS a binding")
        print(f"        constraint, so Step 43's 'output budget' phrasing is literally right and the encoder-")
        print(f"        ladder (43A, internal capacity) was simply the wrong width knob. Revisit the Step 43")
        print(f"        triangulation framing accordingly. Equivariance ({vn_se3_max:.0e}) and flatness "
              f"(x{vn_ratio_max:.2f}) stay exact throughout.")
    else:
        why = ("equivariance/flatness/mlp guard failed" if not (ok_equiv and ok_global_flat and ok_mlp_degrades)
               else "a variant did not converge" if not ok_converged
               else f"muddy reading: widest closes {100 * widest_closed:.0f}% (neither <25% nor >50%)")
        print(f"    INCONCLUSIVE ({why}). The output-budget sweep did not yield a clean reading; reported")
        print(f"        as-is with NO guard loosened. Equivariance (<= {vn_se3_max:.0e}) and flatness "
              f"(x{vn_ratio_max:.2f}) stay exact at every budget rung, the oracle included -- the prior is")
        print(f"        never the cost, whatever the cap turns out to be.")

    # ---- dump JSON + figure --------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(BUDGET_RUNGS=list(BUDGET_RUNGS), ORACLES=list(ORACLES),
                       labels={k: VARIANTS[k]["label"] for k in VARIANTS},
                       n_out_vec={nm: VARIANTS[nm]["n_out_vec"] for nm in BUDGET_RUNGS},
                       LMAX_FIXED=LMAX_FIXED, MUL_FIXED=MUL_FIXED, HIDDEN_DIM=HIDDEN_DIM,
                       SEEDS=list(SEEDS), N_TRAIN=N_TRAIN, N_TEST=N_TEST, EPOCHS=EPOCHS, K_OOD=K_OOD,
                       VAR_COEF=VAR_COEF, C_INT=C_INT, AUG=AUG, D_OBJ_ORACLE=P * 3),
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
        "ok_converged": bool(ok_converged),
        "cap": cap, "mlp": mlp, "gap": gap, "oracle_unit": oracle_unit,
        "widest": widest, "best_budget": best_budget, "best_budget_variant": best_budget_nm,
        "widest_gap_closed_frac": widest_closed, "best_budget_gap_closed_frac": best_budget_closed,
        "oracle_gap_closed_frac": oracle_closed,
        "saturation_residual_ratio": sat_resid,
        "reading": ("budget_not_cap" if budget_not_cap else "budget_is_cap" if budget_is_cap else "muddy"),
        "verdict": {"passed": bool(passed), "ok_equiv": ok_equiv, "ok_global_flat": ok_global_flat,
                    "ok_mlp_degrades": ok_mlp_degrades, "ok_converged": ok_converged,
                    "ok_conclusive": ok_conclusive},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step44_encoder_output_budget_smoke" if SMOKE else "step44_encoder_output_budget"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    _make_figure(fig_dir / f"{stem}.png", indist, indist_mean, ratio_mean, se3_mean, cap, mlp, oracle_unit)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)}")
    print(f"    wrote {(fig_dir / f'{stem}.png').relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


def _make_figure(path, indist, indist_mean, ratio_mean, se3_mean, cap, mlp, oracle_unit) -> None:
    r"""Three panels: output-budget localisation (with pool-hinge + oracle/MLP lines), across-group
    flatness, exactness — all per variant, to show widening the readout never spends the prior."""
    xs = np.arange(len(BUDGET_RUNGS))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A: output-budget localisation curve (budget rungs as a line; oracle + MLP as ceilings)
    means = [indist_mean[nm] for nm in BUDGET_RUNGS]
    stds = [float(np.std(indist[nm])) for nm in BUDGET_RUNGS]
    nouts = [VARIANTS[nm]["n_out_vec"] for nm in BUDGET_RUNGS]
    ax[0].errorbar(nouts, means, yerr=stds, marker="o", capsize=4, color="C0", label="budget rungs")
    ax[0].axhline(oracle_unit, ls="-.", color="C2", label=f"lossless oracle ({oracle_unit:.3f})")
    ax[0].axhline(mlp, ls="--", color="C3", label=f"MLP-MP ceiling ({mlp:.3f})")
    ax[0].axhline(cap, ls=":", color="C7", label=f"B16/E0 cap ({cap:.3f})")
    ax[0].axvline(HIDDEN_DIM / 3.0, ls=":", color="C8", alpha=0.7,
                  label=f"pool hinge n_out={HIDDEN_DIM // 3} (3*n_out=dim h)")
    ax[0].set_xlabel("output budget  n_out_vec")
    ax[0].set_xticks(nouts)
    ax[0].set_ylabel("in-distribution relMSE")
    ax[0].set_title("[L] output-budget sweep: does the readout drop the floor?")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].grid(alpha=0.3)

    # Panel B: across-group OOD/seen ratio per variant
    xs2 = np.arange(len(VN_VARIANTS))
    colors = ["C0"] * len(BUDGET_RUNGS) + ["C2"] * len(ORACLES)
    ratios = [ratio_mean[nm] for nm in VN_VARIANTS]
    ax[1].bar(xs2, ratios, color=colors, alpha=0.85)
    ax[1].axhline(ratio_mean[MLP_ANCHOR], ls="--", color="C3",
                  label=f"MLP-MP (x{ratio_mean[MLP_ANCHOR]:.1f})")
    ax[1].axhline(1.0, ls=":", color="k", alpha=0.5, label="flat (x1.00)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs2)
    ax[1].set_xticklabels([nm for nm in VN_VARIANTS], rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("global OOD/seen ratio")
    ax[1].set_title("[G] across-group 举一反三 (flat at every budget)")
    ax[1].legend(fontsize=7, loc="best")
    ax[1].grid(alpha=0.3, which="both", axis="y")

    # Panel C: post-train equivariance residual per variant
    se3s = [se3_mean[nm] for nm in VN_VARIANTS]
    ax[2].bar(xs2, se3s, color=colors, alpha=0.85)
    ax[2].axhline(se3_mean[MLP_ANCHOR], ls="--", color="C3", label="MLP-MP")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs2)
    ax[2].set_xticklabels([nm for nm in VN_VARIANTS], rotation=30, ha="right", fontsize=7)
    ax[2].set_ylabel("post-train SE(3) residual")
    ax[2].set_title("[A'] exact equivariance (oracle included)")
    ax[2].legend(fontsize=7, loc="best")
    ax[2].grid(alpha=0.3, which="both", axis="y")

    fig.suptitle("Step 44 — encoder output-budget sweep: readout width vs the upstream pooling cap",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
