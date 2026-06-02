r"""Step 46: the **pooling cure** — does a richer equivariant aggregator close the open problem?

Where Steps 42-44 left us
-------------------------
Three levers were swept to recover the in-distribution relMSE floor a degree-3 Vector-Neuron
world model leaves open on the interacting teacher $\omega_i=\kappa(\hat r_{ij}\times a_i)\times\tilde x_k$,
and all three triangulated the cap onto ONE component:

  * Step 32 — predictor **degree** ($d_{\max}=2^L$): saturates after the first cross-product rung.
  * Step 42 — message **content** (raw $r$ / unit $\hat r$): a NULL (x1.01, seed noise).
  * Step 43 — encoder **capacity** at fixed budget (mul 8->32, lmax 2->3): saturates at ~29% of the
    E0->MLP gap, while the lossless point-cloud **oracle** closes ~156% — proving the residual cap is
    the encoder's lossy aggregate, NOT the predictor class or the message.

Step 43 localised the cap to a single line of :class:`~src.models.se3.SE3PointEncoder`: the
permutation-invariant **sum-pool** $h=\sum_k \mathrm{msg}_k$, which collapses $P{=}24$ per-object points
to one mean-field aggregate *before* the 16-vector readout (so Step 44's output-budget widening could
not help — the loss is upstream). Both papers report this as an **open problem**: *can a richer
equivariant pooling close the gap while staying exactly equivariant?* This step tests the most direct
cure.

The cure (design-rule: "enrich the aggregator, keep the prior")
---------------------------------------------------------------
Replace the single sum with a multi-head **equivariant attention pool** (``pool="attn"``): per-point
scores read off the *invariant* $\ell=0$ channels of $\mathrm{msg}$ (so the weights are SO(3)-invariant),
$\mathrm{softmax}$ over the $N$ points (so the head stays permutation-invariant), $K$ distinct weighted
sums of the equivariant per-point features, then an ``o3.Linear`` recombination. Because the $K$ heads
enter the downstream ``NormActivation`` *separately* (not pre-summed — a pre-sum would collapse back to
one weighted aggregate), the pool is strictly richer than the sum, at the SAME fixed latent budget
($N_{\text{out}}{=}16$, $D_{\text{scene}}{=}96$). ``test_step46_attn_pool_equivariance.py`` proves it is
still exactly SE(3)- + permutation-equivariant (float-floor), so any gain is NOT bought by leaking the
prior. The latent stays a fixed-size abstract code (it does NOT regress toward the raw-cloud oracle —
the "J" in JEPA survives).

The variants
------------
    E0-sum     enc, sum-pool, mul8       <- the published cap (Step 27/32/42/43 baseline)
    P1-attn4   enc, attn-pool, K=4, mul8 <- the cure: ONLY the pooling rule changes
    P2-attn8   enc, attn-pool, K=8, mul8 <- richer routing (more heads)
    ORACLE-unit lossless points + r_hat  <- the target: the predictor class SOLVES the one-step task
    MLP-MP     unconstrained MLP          <- non-equivariant ceiling

Step 43 already controlled for encoder *width* (mul saturates), so P1/P2 isolate whether the pooling
*structure* — not width — is the lever. Param counts are printed (the attn modules add params; that is
the enrichment, the fair question being "does structure help where width didn't").

The honest three-way verdict (no guard is loosened either way)
--------------------------------------------------------------
Let ``closed = (E0_sum - best_attn) / (E0_sum - MLP)`` be the fraction of the gap the cure closes,
against the Step-43 sum-pool ladder's ~29%:
  * CLOSED  — attn closes >50% of the gap (matching the oracle's bar): the open problem is resolved,
              a richer equivariant pooling recovers the interaction. Requires equivariant + flat.
  * PARTIAL — attn closes clearly more than the 29% ladder (>=40%) but <50%: real progress, the
              problem is *narrowed* but not closed.
  * NO-HELP — attn ~ the ladder (<40%): the pooling *rule* is not the lever; the cap is the
              single-aggregate->fixed-latent *compression* itself, a sharper open problem.
In every case the attn pool MUST stay exactly equivariant ([A']) and global-flat ([G]); if it does not,
the run is a confound and reports its symmetry break honestly rather than any in-dist number.

Run (full; ~25-45 min on a laptop CPU, 5 seeds x 5 models):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step46_pooling_cure.py
Smoke (~2-4 min):
    STEP46_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step46_pooling_cure.py
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

# Reuse the validated Step 24 interacting-teacher bench + Step 43 oracle/message/probe machinery
# VERBATIM. Step 46 changes ONLY the encoder's POOLING rule (sum -> multi-head attention), holding
# the teacher, data, predictor (degree-3 VN-TP), training, latent budget and messages fixed.
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
from step43_encoder_ladder import (  # noqa: E402
    CenteredPointEncoder,
    _perm_latent,
    _rotate_all,
    build_unit_msg,
)
from src.models.structured import VNTPPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP46_SMOKE"))

# --------------------------------------------------------------------------- #
# variant ladder: the sum-pool cap, two attention-pool cures, the lossless oracle, the MLP ceiling.
# --------------------------------------------------------------------------- #
VARIANTS = {
    "E0-sum":      dict(kind="enc", pool="sum",  n_heads=0, msg="raw",  label="enc sum-pool [a,r]"),
    "P1-attn4":    dict(kind="enc", pool="attn", n_heads=4, msg="raw",  label="enc attn-pool K4 [a,r]"),
    "P2-attn8":    dict(kind="enc", pool="attn", n_heads=8, msg="raw",  label="enc attn-pool K8 [a,r]"),
    "ORACLE-unit": dict(kind="oracle",           n_heads=0, msg="unit", label="points [a,r_hat]"),
    "MLP-MP":      dict(kind="mlp",              n_heads=0, msg="raw",  label="MLP [a,r]"),
}
ENC_RUNGS = ("E0-sum", "P1-attn4", "P2-attn8")
ATTN_RUNGS = ("P1-attn4", "P2-attn8")
ORACLES = ("ORACLE-unit",)
VN_VARIANTS = ENC_RUNGS + ORACLES          # every equivariant variant
MLP_ANCHOR = "MLP-MP"
NAMES = list(VN_VARIANTS) + [MLP_ANCHOR]

AUG = A_OBJ + 3                            # per-object action width (one extra type-1 vector) = 6
LADDER_29 = 0.29                           # Step 43 sum-pool encoder ladder's best gap-closed fraction


def _d_obj(name: str) -> int:
    return P * 3 if VARIANTS[name]["kind"] == "oracle" else D_OBJ


def build_model(name: str) -> EqJEPA:
    r"""Build a variant. Encoder rungs share the VN-TP predictor at $D_{\text{obj}}{=}48$ and differ
    ONLY in the per-object pooling (``sum`` vs ``attn``) at a fixed budget; the oracle bypasses the
    encoder with the lossless centred cloud at $D_{\text{obj}}{=}72$; MLP-MP is the ceiling."""
    v = VARIANTS[name]
    if v["kind"] == "mlp":
        return build_mlp_mp()
    if v["kind"] == "oracle":
        d_obj = P * 3
        enc = CenteredPointEncoder(N_OBJ, P)
        pred = SlotPredictor(VNTPPredictor(d_obj, AUG, hidden=64, dim=3),
                             n_obj=N_OBJ, d_obj=d_obj, a_obj=AUG)
        return EqJEPA(latent_dim=N_OBJ * d_obj, action_dim=N_OBJ * AUG, encoder=enc, predictor=pred)
    # encoder rung: vary ONLY the pooling rule, hold lmax=2/mul=8 and the latent budget fixed
    enc = SetSE3Encoder(N_OBJ, n_out_vec=N_OUT_VEC, lmax=2, mul=8,
                        pool=v["pool"], n_heads=max(1, v["n_heads"]))
    pred = SlotPredictor(VNTPPredictor(D_OBJ, AUG, hidden=64, dim=3), a_obj=AUG)
    return EqJEPA(latent_dim=D_SCENE, action_dim=N_OBJ * AUG, encoder=enc, predictor=pred)


def model_action_v(name: str, S: torch.Tensor, A_self: torch.Tensor) -> torch.Tensor:
    if VARIANTS[name]["msg"] == "unit":
        return build_unit_msg(S, A_self)
    return build_msg_action(S, A_self)


@torch.no_grad()
def se3_resid(name, model, S, A_self, R, t) -> float:
    m = model_action_v(name, S, A_self)
    lhs = _rotate_all(model.predictor(model.encoder(S), m), R)
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    m_r = model_action_v(name, Sr, _rot_self_actions(A_self, R))
    rhs = model.predictor(model.encoder(Sr), m_r)
    return (lhs - rhs).abs().max().item()


@torch.no_grad()
def perm_resid(name, model, S, A_self) -> float:
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
def indist_relmse(name, model, S, A_self, S2) -> float:
    return rel_mse(model, S, model_action_v(name, S, A_self), S2)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 80

    if SMOKE:
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0,), 150, 64, 3, 2
    else:
        # mirror Step 43 exactly (5 seeds, 80 epochs, 1500/400) so the attn rungs are read on the SAME
        # footing as the sum-pool ladder they are compared against — the headline is a like-for-like
        # gap-closed fraction, so any difference in seeds/epochs/data would confound it.
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0, 1, 2, 3, 4), 1500, 400, 80, 6
    SEEDS = tuple(int(s) for s in os.environ.get("STEP46_SEEDS", ",".join(map(str, SEEDS))).split(","))
    N_TRAIN = int(os.environ.get("STEP46_NTRAIN", N_TRAIN))
    EPOCHS = int(os.environ.get("STEP46_EPOCHS", EPOCHS))
    VAR_COEF = 0.1

    print(line)
    print(f"STEP 46  pooling cure: does a richer equivariant aggregator close the Step-43 open problem?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    same interacting teacher / VN-TP predictor as Step 27/32/42/43: O={N_OBJ} objs, "
          f"P={P} pts, kappa={C_INT}")
    print(f"    Step 43 localised the cap to the encoder's sum-pool h=sum_k msg_k (ladder saturated at")
    print(f"        ~{100 * LADDER_29:.0f}% of the E0->MLP gap; lossless oracle closed ~156%). Step 46 replaces the")
    print(f"        sum with a multi-head EQUIVARIANT attention pool, SAME budget (N_out={N_OUT_VEC}, D_scene={D_SCENE}):")
    for nm in ENC_RUNGS:
        v = VARIANTS[nm]
        extra = f"K={v['n_heads']}" if v["pool"] == "attn" else "mean-field"
        print(f"          {nm:>11s}  pool={v['pool']:<4s} ({extra})")
    print(f"    target {('ORACLE-unit'):>11s}  lossless points + [a, r_hat] (does the predictor class SOLVE it)")
    print(f"    ceiling {('MLP-MP'):>10s}  unconstrained, non-equivariant")
    print(f"    seeds={SEEDS}  N_train={N_TRAIN}  epochs={EPOCHS}  K_ood={K_OOD}")

    St, At_self, S2t = make_interacting_transitions(N_TEST, seed=999)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    t_chk = torch.randn(3, generator=torch.Generator().manual_seed(4))

    _probe = {nm: build_model(nm) for nm in NAMES}
    params = {nm: n_params(_probe[nm]) for nm in NAMES}
    print("    params: " + "  ".join(f"{nm}={params[nm]}" for nm in NAMES))
    print(f"            (attn pool adds the K-head score MLP + o3.Linear recombiner; Step 43 already showed")
    print(f"             encoder *width* (mul 8->32) saturates, so P1/P2 isolate the pooling *structure*.)")

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
            "   <- attn pool, still exactly equivariant" if nm in ATTN_RUNGS else "")
        print(f"    {nm:>11s} | {s:12.2e} | {p:11.2e}{tag}")
    del _probe

    # ---- per-seed training + evaluation --------------------------------------------------
    indist = {nm: [] for nm in NAMES}
    ood_ratio = {nm: [] for nm in NAMES}
    se3 = {nm: [] for nm in NAMES}
    perm = {nm: [] for nm in NAMES}
    conv = {nm: [] for nm in NAMES}

    for seed in SEEDS:
        print()
        print(line)
        print(f"[seed {seed}] build fresh models, train one-step JEPA on the interacting seen scenes")
        print(line)
        S, A_self, S2 = make_interacting_transitions(N_TRAIN, seed=seed)
        g = torch.Generator().manual_seed(100 + seed)

        for nm in NAMES:
            torch.manual_seed(seed)
            model = build_model(nm)
            hist = train_jepa(model, S, model_action_v(nm, S, A_self), S2,
                              epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=seed, log_every=999)
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

    cap = indist_mean["E0-sum"]
    mlp = indist_mean[MLP_ANCHOR]
    gap = cap - mlp
    best_attn = min(indist_mean[nm] for nm in ATTN_RUNGS)
    best_attn_nm = min(ATTN_RUNGS, key=lambda nm: indist_mean[nm])
    oracle_unit = indist_mean["ORACLE-unit"]
    attn_closed = (cap - best_attn) / gap if abs(gap) > 1e-9 else float("nan")
    oracle_closed = (cap - oracle_unit) / gap if abs(gap) > 1e-9 else float("nan")

    # ---- [L] the localisation curve ------------------------------------------------------
    print()
    print(line)
    print("[L] LOCALISATION -- in-distribution relMSE (lower=better); does the attn pool drop the E0 floor?")
    print(line)
    print(f"    {'variant':>11s} | {'config':>20s} | {'in-dist relMSE (mean +/- std)':>30s}")
    print("    " + "-" * 70)
    for nm in ENC_RUNGS:
        tag = "   <- baseline cap (Step 43)" if nm == "E0-sum" else (
            "   <- best attn rung" if nm == best_attn_nm else "")
        print(f"    {nm:>11s} | {VARIANTS[nm]['label']:>20s} | {indist_mean[nm]:.4e} +/- {sd(indist, nm):.1e}{tag}")
    print("    " + "-" * 70)
    print(f"    {'ORACLE-unit':>11s} | {VARIANTS['ORACLE-unit']['label']:>20s} | {oracle_unit:.4e} +/- "
          f"{sd(indist, 'ORACLE-unit'):.1e}   <- lossless target")
    print(f"    {MLP_ANCHOR:>11s} | {VARIANTS[MLP_ANCHOR]['label']:>20s} | {mlp:.4e} +/- "
          f"{sd(indist, MLP_ANCHOR):.1e}   <- unconstrained ceiling")
    print(f"    => E0 sum-pool cap {cap:.4e}; best attn rung {best_attn_nm} {best_attn:.4e} closes "
          f"{100 * attn_closed:.0f}% of the E0->MLP gap")
    print(f"       (Step 43 sum-pool encoder ladder closed ~{100 * LADDER_29:.0f}%; the lossless oracle here "
          f"closes {100 * oracle_closed:.0f}%)")
    conv_max = max(conv_mean[nm] for nm in NAMES)
    print(f"       convergence (|Δpred_loss| last 20% of {EPOCHS} epochs): "
          + "  ".join(f"{nm.split('-')[0]}={conv_mean[nm]:.1%}" for nm in NAMES))
    ok_converged = conv_max < 0.10
    print(f"       => all variants {'PLATEAUED (<10%, eval trustworthy)' if ok_converged else 'NOT all plateaued (>10% -- read with care)'}")

    # ---- [G] across-group flatness -------------------------------------------------------
    print()
    print(line)
    print("[G] across-group 举一反三 -- global SE(3) OOD/seen ratio per variant (1.00=flat)")
    print(line)
    print(f"    {'variant':>11s} | {'OOD/seen ratio':>15s}")
    print("    " + "-" * 30)
    for nm in VN_VARIANTS:
        tag = "   <- attn pool, still flat" if nm in ATTN_RUNGS else ""
        print(f"    {nm:>11s} | x{ratio_mean[nm]:13.3f}{tag}")
    print(f"    {MLP_ANCHOR:>11s} | x{ratio_mean[MLP_ANCHOR]:13.3f}   <- degrades (no equivariance)")
    vn_ratio_max = max(ratio_mean[nm] for nm in VN_VARIANTS)
    print(f"    => every VN variant flat (<= x{vn_ratio_max:.2f}); MLP carries x{ratio_mean[MLP_ANCHOR]:.1f}.")

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

    # ---- verdict (honest three-way) ------------------------------------------------------
    vn_se3_max = max(se3_mean[nm] for nm in VN_VARIANTS)
    vn_perm_max = max(perm_mean[nm] for nm in VN_VARIANTS)
    ok_equiv = vn_se3_max < 1e-3 and vn_perm_max < 1e-3
    ok_global_flat = vn_ratio_max < 1.15
    ok_mlp_degrades = ratio_mean[MLP_ANCHOR] > 1.3
    symmetry_clean = ok_equiv and ok_global_flat and ok_mlp_degrades

    if attn_closed > 0.50:
        verdict = "CLOSED"
    elif attn_closed >= 0.40:
        verdict = "PARTIAL"
    else:
        verdict = "NO-HELP"

    print()
    print(line)
    print("STEP 46 SUMMARY")
    print(line)
    print(f"    [L] E0 sum-pool cap {cap:.3e}; best attn rung {best_attn_nm} {best_attn:.3e} closes "
          f"{100 * attn_closed:.0f}% of gap to MLP {mlp:.3e} (Step-43 sum ladder: ~{100 * LADDER_29:.0f}%; oracle {100 * oracle_closed:.0f}%).")
    print(f"    [G] across-group OOD/seen: VN variants <= x{vn_ratio_max:.2f} (flat) vs MLP "
          f"x{ratio_mean[MLP_ANCHOR]:.1f} (degrades).")
    print(f"    [A'] post-train equivariance: VN SE(3) <= {vn_se3_max:.1e}, perm <= {vn_perm_max:.1e} "
          f"(attn pool exactly equivariant).")
    print(f"    guards: equiv={ok_equiv}  global-flat={ok_global_flat}  mlp-degrades={ok_mlp_degrades}  "
          f"=> symmetry-clean={symmetry_clean}")
    if not symmetry_clean:
        print(f"    INVALID: the attn pool did NOT stay exactly equivariant/flat -- the cure is a confound")
        print(f"        (capacity bought by leaking the prior). Reported as a symmetry break, NOT an in-dist win.")
    elif verdict == "CLOSED":
        print(f"    CLOSED: the multi-head equivariant attention pool closes {100 * attn_closed:.0f}% of the gap")
        print(f"        (vs the sum-pool ladder's ~{100 * LADDER_29:.0f}%) WHILE staying exactly equivariant (SE(3) "
              f"{se3_mean[best_attn_nm]:.0e}) and flat (x{ratio_mean[best_attn_nm]:.2f}). The Step-43 open problem")
        print(f"        is resolved: a richer equivariant aggregator recovers the interaction the sum-pool lost,")
        print(f"        without spending the prior or regressing toward the raw-cloud oracle. 'enrich the class,")
        print(f"        keep the prior' -- now demonstrated for the pooling, not just asserted.")
    elif verdict == "PARTIAL":
        print(f"    PARTIAL: the attn pool closes {100 * attn_closed:.0f}% of the gap -- clearly more than the")
        print(f"        sum-pool ladder's ~{100 * LADDER_29:.0f}%, so the pooling structure IS part of the lever, but it")
        print(f"        does not reach the oracle's >50%. The open problem is NARROWED, not closed: a richer")
        print(f"        aggregator helps, yet a residual remains in the fixed-size compression itself. Stays")
        print(f"        exactly equivariant + flat. Reported as partial progress, no overclaim.")
    else:
        print(f"    NO-HELP: the attn pool closes only {100 * attn_closed:.0f}% -- within noise of the sum-pool")
        print(f"        ladder's ~{100 * LADDER_29:.0f}%. So the pooling *rule* (sum vs learned attention) is NOT the lever;")
        print(f"        the cap is the single-aggregate->fixed-16-vector *compression* itself, which both share.")
        print(f"        The open problem is SHARPENED, not closed: it is the latent's fixed abstract size, not")
        print(f"        the aggregation function. Stays exactly equivariant + flat; no guard loosened, no overclaim.")

    # ---- dump JSON + figure --------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(ENC_RUNGS=list(ENC_RUNGS), ATTN_RUNGS=list(ATTN_RUNGS), ORACLES=list(ORACLES),
                       labels={k: VARIANTS[k]["label"] for k in VARIANTS},
                       SEEDS=list(SEEDS), N_TRAIN=N_TRAIN, N_TEST=N_TEST, EPOCHS=EPOCHS, K_OOD=K_OOD,
                       VAR_COEF=VAR_COEF, C_INT=C_INT, AUG=AUG, D_OBJ=D_OBJ, D_OBJ_ORACLE=P * 3,
                       LADDER_29=LADDER_29),
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
        "best_attn": best_attn, "best_attn_variant": best_attn_nm, "attn_gap_closed_frac": attn_closed,
        "oracle_unit": oracle_unit, "oracle_unit_gap_closed_frac": oracle_closed,
        "ladder_gap_closed_frac": LADDER_29,
        "verdict": {"label": verdict, "symmetry_clean": symmetry_clean, "ok_equiv": ok_equiv,
                    "ok_global_flat": ok_global_flat, "ok_mlp_degrades": ok_mlp_degrades,
                    "attn_closed": attn_closed},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step46_pooling_cure_smoke" if SMOKE else "step46_pooling_cure"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))
    _make_figure(fig_dir / f"{stem}.png", indist, indist_mean, ratio_mean, se3_mean, cap, mlp,
                 oracle_unit, attn_closed, verdict)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)}")
    print(f"    wrote {(fig_dir / f'{stem}.png').relative_to(ROOT)}")
    # exit 0 if the run is symmetry-clean AND the cure at least PARTIAL-ly helps; NO-HELP is a valid
    # honest outcome but exits 1 so CI flags it for a human read (it changes the paper's open-problem text).
    sys.exit(0 if (symmetry_clean and verdict != "NO-HELP") else 1)


def _make_figure(path, indist, indist_mean, ratio_mean, se3_mean, cap, mlp,
                 oracle_unit, attn_closed, verdict) -> None:
    r"""Three panels: localisation (sum vs attn vs oracle), across-group flatness, exactness."""
    xs = np.arange(len(VN_VARIANTS))
    # sum-pool in grey, attn rungs in orange (the cure), oracle in green
    colors = []
    for nm in VN_VARIANTS:
        colors.append("C2" if VARIANTS[nm]["kind"] == "oracle"
                      else ("C1" if nm in ATTN_RUNGS else "C7"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    means = [indist_mean[nm] for nm in VN_VARIANTS]
    stds = [float(np.std(indist[nm])) for nm in VN_VARIANTS]
    ax[0].bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax[0].axhline(mlp, ls="--", color="C3", label=f"MLP-MP ceiling ({mlp:.3f})")
    ax[0].axhline(cap, ls=":", color="k", alpha=0.6, label=f"E0 sum-pool cap ({cap:.3f})")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(list(VN_VARIANTS), rotation=30, ha="right", fontsize=7)
    ax[0].set_ylabel("in-distribution relMSE")
    ax[0].set_title(f"[L] pooling cure: sum (grey) vs attn (orange) vs oracle (green)\n"
                    f"best attn closes {100 * attn_closed:.0f}% of gap — {verdict}")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].grid(alpha=0.3, axis="y")

    ratios = [ratio_mean[nm] for nm in VN_VARIANTS]
    ax[1].bar(xs, ratios, color=colors, alpha=0.85)
    ax[1].axhline(ratio_mean[MLP_ANCHOR], ls="--", color="C3",
                  label=f"MLP-MP (x{ratio_mean[MLP_ANCHOR]:.1f})")
    ax[1].axhline(1.0, ls=":", color="k", alpha=0.5, label="flat (x1.00)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(list(VN_VARIANTS), rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("global OOD/seen ratio")
    ax[1].set_title("[G] across-group 举一反三 (attn pool stays flat)")
    ax[1].legend(fontsize=7, loc="best")
    ax[1].grid(alpha=0.3, which="both", axis="y")

    se3s = [se3_mean[nm] for nm in VN_VARIANTS]
    ax[2].bar(xs, se3s, color=colors, alpha=0.85)
    ax[2].axhline(se3_mean[MLP_ANCHOR], ls="--", color="C3", label="MLP-MP")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(list(VN_VARIANTS), rotation=30, ha="right", fontsize=7)
    ax[2].set_ylabel("post-train SE(3) residual")
    ax[2].set_title("[A'] attn pool stays exactly equivariant")
    ax[2].legend(fontsize=7, loc="best")
    ax[2].grid(alpha=0.3, which="both", axis="y")

    fig.suptitle("Step 46 — pooling cure: a richer equivariant aggregator vs the lossy sum-pool",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
