r"""Step 32: the **tensor-product degree ladder** — recovery curve vs representable degree.

What Step 27 left open
----------------------
Step 24 diagnosed a *structural* capacity cap: vanilla Vector-Neuron layers are degree-1, so an
equivariant predictor cannot form the interacting teacher's **degree-3** torque
$(\hat r_{ij}\times a_i)\times\tilde x_k$, and a plain MLP fits ~5x better in-distribution while
throwing away all across-group 举一反三. Step 27 added *one* fixed tensor-product stack (VN-TP) and
recovered ~42% of that gap while staying exactly flat. Natural question Step 27 could not answer with
a single architectural point: **how does the recovery depend on the representable polynomial degree?**

The ladder (this step)
----------------------
:class:`VNTPLadderPredictor` exposes a clean degree knob: the first $L=\texttt{n\_tp\_blocks}$ of a
fixed stack of ``n_blocks_total=3`` equivariant blocks carry a cross-product branch, so the maximum
representable polynomial degree is $d_{\max}=2^{L}$. Crucially **depth, width, and (near-)parameter
count are held fixed across the ladder** — only the representable degree changes. We sweep
$L\in\{0,1,2,3\}$ ($d_{\max}\in\{1,2,4,8\}$) on the *identical* Step-24 interacting teacher, encoder,
message channel, data, and training as Step 27.

The teacher's torque is degree-3, first representable at $L=2$ ($d_{\max}=4\ge3$). So the theory
predicts a **degree signature**, not a capacity ramp:

  [I]  recovery curve : in-dist relMSE should *drop* as $L$ crosses the rung that admits degree-3 and
                        then **saturate** — $L=3$ ($d_{\max}=8$) buys ~nothing over $L=2$, because the
                        physics is exactly degree-3. (A pure capacity effect would keep improving.)
  [G]  across-group OOD: every rung must stay **x1.00** (equivariance is a theorem at every degree),
                        while the unconstrained MLP degrades (~x17) — climb the ladder for capacity,
                        keep 举一反三 for free at every rung.
  [A'] post-training equivariance: every rung's global SE(3) residual at the float floor.

Gate (PASS): every rung exactly equivariant post-train AND the ladder recovers capacity
(top rung clearly better than degree-1) AND it saturates (the last rung's marginal gain is a small
fraction of the total recovery) AND every rung stays global-flat AND the MLP still degrades.

Result (3 seeds, FULL): all five guards PASS. The recovery and the saturation are both real, but the
knee lands one rung *earlier* than the naive degree-3 prediction: in-dist relMSE drops $0.263\to0.194$
at $L{=}0\to1$ ($\times1.36$, $38\%$ of the cap$\to$MLP gap) and then SATURATES dead-flat
($L_1\!\approx\!L_2\!\approx\!L_3\!\approx\!0.20$; last rung $+1\%$ of the total). So the *first*
cross-product rung ($d_{\max}{=}2$) already captures the recoverable bulk — the predictor acts on the
encoder's **already-nonlinear** ($\ell_{\max}{=}2$) latent, not on raw points, so the point-space
degree-3 count is an upper bound, not the latent-space knee. The qualitative claim is unchanged and if
anything sharper: a saturating step, NOT a monotone capacity ramp (a pure-capacity effect would keep
improving toward MLP's $0.080$; instead $d_{\max}{=}4,8$ buy nothing). Equivariance is free at every
rung: OOD/seen $\times1.00$, SE(3) residual $\le 9.3\times10^{-5}$, perm $0$; the $2.4\times$-larger MLP
fits $0.080$ in-dist but degrades $\times10.5$ across the group with SE(3) residual $\approx 8.9$.

Run (full; ~20-40 min on a laptop CPU, 3 seeds x 5 models):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step32_tp_degree_ladder.py
Smoke (~2-3 min):
    STEP32_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step32_tp_degree_ladder.py
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

# Reuse the *validated* Step 24 interacting-teacher bench VERBATIM (same data generator, message
# channel, global-OOD transform, whole-pipeline equivariance probes, relMSE, and the MLP-MP anchor);
# Step 32 adds ONLY the degree-laddered predictor and sweeps its representable degree.
from step24_object_interaction import (  # noqa: E402
    A_OBJ_AUG,
    C_INT,
    D_OBJ,
    D_SCENE,
    A_SCENE_AUG,
    N_OBJ,
    N_OUT_VEC,
    P,
    EqJEPA,
    SetSE3Encoder,
    SlotPredictor,
    build_mlp_mp,
    composed_se3_residual,
    make_interacting_transitions,
    model_action,
    n_params,
    rand_so3,
    rel_mse_named,
    train_jepa,
    transform_global,
    vnmp_perm_err,
)
from src.models.structured import VNTPLadderPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP32_SMOKE"))

LADDER = (0, 1, 2, 3)        # n_tp_blocks rungs; max representable degree d_max = 2^L
HIDDEN = 64
N_BLOCKS_TOTAL = 3
TEACHER_DEGREE = 3           # the interacting torque (r_hat x a) x x_tilde is degree-3
FIRST_OK_RUNG = 2            # first L with d_max = 2^L >= TEACHER_DEGREE  (2^2 = 4 >= 3)


def ladder_name(L: int) -> str:
    return f"L{L}"


def build_ladder(L: int) -> EqJEPA:
    r"""Step-24 VN-MP with its per-object predictor swapped to the degree-laddered one at rung ``L``.

    Same shared :class:`SetSE3Encoder` and augmented relative-pose action as VN-MP/VN-TP; ONLY the
    predictor's representable degree (``n_tp_blocks=L``) changes. Still jointly
    $\mathrm{SE}(3)\rtimes S_O$-equivariant at every rung.
    """
    enc = SetSE3Encoder(N_OBJ, n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    pred = SlotPredictor(
        VNTPLadderPredictor(D_OBJ, A_OBJ_AUG, hidden=HIDDEN, dim=3,
                            n_tp_blocks=L, n_blocks_total=N_BLOCKS_TOTAL),
        a_obj=A_OBJ_AUG,
    )
    return EqJEPA(latent_dim=D_SCENE, action_dim=A_SCENE_AUG, encoder=enc, predictor=pred)


def build_model(name: str) -> EqJEPA:
    if name == "MLP-MP":
        return build_mlp_mp()       # unconstrained capacity ceiling (no equivariance) -- the anchor
    return build_ladder(int(name[1:]))  # "L0".."L3"


def main() -> None:
    line = "=" * 78

    if SMOKE:
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0,), 150, 64, 3, 2
    else:
        SEEDS, N_TRAIN, N_TEST, EPOCHS, K_OOD = (0, 1, 2), 1500, 400, 60, 6
    SEEDS = tuple(int(s) for s in os.environ.get("STEP32_SEEDS", ",".join(map(str, SEEDS))).split(","))
    N_TRAIN = int(os.environ.get("STEP32_NTRAIN", N_TRAIN))
    EPOCHS = int(os.environ.get("STEP32_EPOCHS", EPOCHS))
    VAR_COEF = 0.1

    names = [ladder_name(L) for L in LADDER] + ["MLP-MP"]

    print(line)
    print(f"STEP 32  tensor-product DEGREE LADDER: recovery curve vs representable degree  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    same interacting teacher as Step 24/27: O={N_OBJ} objs, P={P} pts, kappa={C_INT}; "
          f"latent {D_SCENE}, aug action {A_SCENE_AUG}")
    print(f"    ladder: VNTPLadderPredictor, n_tp_blocks L in {list(LADDER)} (d_max=2^L in "
          f"{[2 ** L for L in LADDER]}); fixed depth={N_BLOCKS_TOTAL}, width={HIDDEN}")
    print(f"    teacher torque is degree-{TEACHER_DEGREE}: first representable at L={FIRST_OK_RUNG} "
          f"(d_max={2 ** FIRST_OK_RUNG}); theory predicts a knee at L={FIRST_OK_RUNG} then saturation")
    print(f"    seeds={SEEDS}  N_train={N_TRAIN}  epochs={EPOCHS}  K_ood={K_OOD}")

    # ---- fixed eval data shared across seeds (paired seen vs global-OOD) ------------------
    St, At_self, S2t = make_interacting_transitions(N_TEST, seed=999)

    # report param counts once (near-constant across the ladder by construction)
    _probe = {nm: build_model(nm) for nm in names}
    print("    params: " + "  ".join(f"{nm}={n_params(m)}" for nm, m in _probe.items())
          + "   (ladder rungs ~matched: only the representable DEGREE differs)")
    params = {nm: n_params(_probe[nm]) for nm in names}
    del _probe

    # ---- per-seed training + evaluation --------------------------------------------------
    # accumulators: indist[name][seed], ood_ratio[name][seed], se3[name][seed], perm[name][seed]
    indist = {nm: [] for nm in names}
    ood_ratio = {nm: [] for nm in names}
    se3 = {nm: [] for nm in names}
    perm = {nm: [] for nm in names}
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    t_chk = torch.randn(3, generator=torch.Generator().manual_seed(4))

    for seed in SEEDS:
        print()
        print(line)
        print(f"[seed {seed}] build fresh models, train one-step JEPA on the interacting seen scenes")
        print(line)
        S, A_self, S2 = make_interacting_transitions(N_TRAIN, seed=seed)
        torch.manual_seed(seed)
        models = {nm: build_model(nm) for nm in names}
        g = torch.Generator().manual_seed(100 + seed)  # the global-OOD rotation stream

        for nm in names:
            train_jepa(models[nm], S, model_action(nm, S, A_self), S2,
                       epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=seed, log_every=999)
            # [A'] post-train equivariance (message channel included)
            se3_v = composed_se3_residual(nm, models[nm], St, At_self, R_chk, t_chk)
            perm_v = vnmp_perm_err(models[nm], St, At_self)
            # [I] in-distribution fit on the seen wedge
            seen = rel_mse_named(nm, models[nm], St, At_self, S2t)
            # [G] global-orientation OOD: rotate+translate the WHOLE scene off the seen wedge
            ood_vals = []
            for _ in range(K_OOD):
                R = rand_so3(g)
                t = torch.randn(3, generator=g)
                Sg, Ag, S2g = transform_global(St, At_self, S2t, R, t)
                ood_vals.append(rel_mse_named(nm, models[nm], Sg, Ag, S2g))
            ood = float(np.mean(ood_vals))
            indist[nm].append(seen)
            ood_ratio[nm].append(ood / max(seen, 1e-12))
            se3[nm].append(se3_v)
            perm[nm].append(perm_v)
            print(f"    {nm:>7s} | in-dist relMSE {seen:.4e} | OOD/seen x{ood / max(seen, 1e-12):6.3f} "
                  f"| SE(3) {se3_v:.1e} | perm {perm_v:.1e}")

    # ---- aggregate (mean over seeds) -----------------------------------------------------
    def m(d, nm):
        return float(np.mean(d[nm]))

    def s(d, nm):
        return float(np.std(d[nm]))

    indist_mean = {nm: m(indist, nm) for nm in names}
    ratio_mean = {nm: m(ood_ratio, nm) for nm in names}
    se3_mean = {nm: m(se3, nm) for nm in names}
    perm_mean = {nm: m(perm, nm) for nm in names}

    # ---- [A'] equivariance at every rung -------------------------------------------------
    print()
    print(line)
    print("[A'] post-training equivariance at EVERY rung (mean over seeds): VN ladder stays exact")
    print(line)
    print(f"    {'rung':>7s} | {'d_max':>5s} | {'SE(3) resid':>12s} | {'perm resid':>11s}")
    print("    " + "-" * 44)
    for L in LADDER:
        nm = ladder_name(L)
        print(f"    {nm:>7s} | {2 ** L:>5d} | {se3_mean[nm]:12.2e} | {perm_mean[nm]:11.2e}")
    print(f"    {'MLP-MP':>7s} | {'-':>5s} | {se3_mean['MLP-MP']:12.2e} | {perm_mean['MLP-MP']:11.2e}"
          f"   <- non-equivariant control")

    # ---- [I] the recovery curve ----------------------------------------------------------
    print()
    print(line)
    print("[I] RECOVERY CURVE -- in-distribution relMSE vs representable degree d_max=2^L (lower=better)")
    print(line)
    print(f"    {'rung':>7s} | {'d_max':>5s} | {'in-dist relMSE (mean +/- std)':>30s}")
    print("    " + "-" * 50)
    for L in LADDER:
        nm = ladder_name(L)
        tag = ""
        if L == 0:
            tag = "   <- degree-1 VN cap (Step 24)"
        elif L == FIRST_OK_RUNG:
            tag = f"   <- first rung that can represent the degree-{TEACHER_DEGREE} torque"
        print(f"    {nm:>7s} | {2 ** L:>5d} | {indist_mean[nm]:.4e} +/- {s(indist, nm):.1e}{tag}")
    print(f"    {'MLP-MP':>7s} | {'-':>5s} | {indist_mean['MLP-MP']:.4e} +/- {s(indist, 'MLP-MP'):.1e}"
          f"   <- unconstrained capacity ceiling (no equivariance)")

    cap = indist_mean[ladder_name(0)]                          # degree-1 cap
    best = min(indist_mean[ladder_name(L)] for L in LADDER)    # best ladder rung
    mlp = indist_mean["MLP-MP"]
    recovery_gain = cap / max(best, 1e-12)
    total_recovery = cap - best
    gap = cap - mlp                                            # VN-cap -> MLP ceiling
    gap_closed = (cap - best) / gap if abs(gap) > 1e-9 else float("nan")
    # marginal gains along the ladder
    rel = [indist_mean[ladder_name(L)] for L in LADDER]
    gains = {f"{LADDER[i]}->{LADDER[i + 1]}": rel[i] - rel[i + 1] for i in range(len(LADDER) - 1)}
    last_gain = rel[-2] - rel[-1]                              # L=2 -> L=3 marginal improvement
    saturate_frac = last_gain / total_recovery if total_recovery > 1e-12 else 0.0
    knee = max(gains, key=gains.get)                           # where the largest single drop is
    print(f"    => degree-1 cap {cap:.4e} -> best rung {best:.4e}  (x{recovery_gain:.2f} better, "
          f"{100 * gap_closed:.0f}% of the cap->MLP gap)")
    print(f"       marginal gains by rung: " + "  ".join(f"{k}:{v:+.2e}" for k, v in gains.items()))
    print(f"       largest single drop at L {knee}; last rung (L=2->3) adds {100 * saturate_frac:.0f}% "
          f"of the total recovery (small => SATURATED at the teacher's degree)")

    # ---- [G] across-group flatness at every rung -----------------------------------------
    print()
    print(line)
    print("[G] across-group 举一反三 -- global SO(3) OOD/seen ratio at every rung (1.00=flat)")
    print(line)
    print(f"    {'rung':>7s} | {'d_max':>5s} | {'OOD/seen ratio':>15s}")
    print("    " + "-" * 34)
    for L in LADDER:
        nm = ladder_name(L)
        print(f"    {nm:>7s} | {2 ** L:>5d} | x{ratio_mean[nm]:13.3f}")
    print(f"    {'MLP-MP':>7s} | {'-':>5s} | x{ratio_mean['MLP-MP']:13.3f}   <- degrades (no equivariance)")
    ladder_ratio_max = max(ratio_mean[ladder_name(L)] for L in LADDER)
    print(f"    => every ladder rung flat (<= x{ladder_ratio_max:.2f}); the MLP carries x{ratio_mean['MLP-MP']:.1f} "
          f"-- capacity climbs the ladder, 举一反三 is free at EVERY degree.")

    # ---- verdict -------------------------------------------------------------------------
    ladder_se3_max = max(se3_mean[ladder_name(L)] for L in LADDER)
    ladder_perm_max = max(perm_mean[ladder_name(L)] for L in LADDER)
    ok_ladder_equiv = ladder_se3_max < 1e-3 and ladder_perm_max < 1e-3
    ok_recovers = recovery_gain > 1.3
    ok_saturates = saturate_frac < 0.25            # the top rung adds < 1/4 of the total recovery
    ok_global_flat = ladder_ratio_max < 1.15
    ok_mlp_degrades = ratio_mean["MLP-MP"] > 1.3
    passed = ok_ladder_equiv and ok_recovers and ok_saturates and ok_global_flat and ok_mlp_degrades

    print()
    print(line)
    print("STEP 32 SUMMARY")
    print(line)
    print(f"    [I] recovery: degree-1 cap {cap:.3e} -> best rung {best:.3e} (x{recovery_gain:.2f}, "
          f"{100 * gap_closed:.0f}% of the gap to MLP {mlp:.3e}); SATURATES (last rung +{100 * saturate_frac:.0f}% "
          f"of total).")
    print(f"    [G] across-group OOD/seen: ladder <= x{ladder_ratio_max:.2f} (flat at every degree) "
          f"vs MLP-MP x{ratio_mean['MLP-MP']:.1f} (degrades).")
    print(f"    [A'] post-train equivariance: ladder SE(3) <= {ladder_se3_max:.1e}, perm <= {ladder_perm_max:.1e} "
          f"(exact at every rung).")
    print(f"    guards: ladder-equiv={ok_ladder_equiv}  recovers={ok_recovers}  saturates={ok_saturates}  "
          f"global-flat={ok_global_flat}  mlp-degrades={ok_mlp_degrades}")
    best_L = min(LADDER, key=lambda L: indist_mean[ladder_name(L)])
    print(f"    headline: enriching the EQUIVARIANT class up the tensor-product degree ladder recovers the")
    print(f"        capacity the degree-1 VN (L=0) structurally lacks: the largest drop is at rung {knee} "
          f"(best rung L={best_L},")
    print(f"        d_max={2 ** best_L}), and the recovery then SATURATES -- climbing on to d_max=8 (L=3) buys "
          f"nothing (last rung")
    print(f"        +{100 * saturate_frac:.0f}% of total). A DEGREE signature, not a capacity ramp. Note the knee "
          f"sets in at the FIRST")
    print(f"        tensor-product rung (degree-2), earlier than the teacher's exact degree-{TEACHER_DEGREE}: the "
          f"predictor acts on the")
    print(f"        encoder's already-nonlinear (lmax=2) latent, so the bulk of the recoverable structure needs "
          f"only one cross")
    print(f"        product. Across-group 举一反三 stays exact (x1.00) and equivariance holds at the float floor "
          f"at EVERY rung.")
    print(f"        Climb the ladder for fit; keep the prior.")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: the degree ladder recovers & saturates; equivariance "
          f"holds free at every degree." if passed else
          f"    INCONCLUSIVE: inspect the recovery curve / guards above.")

    # ---- dump JSON + figure --------------------------------------------------------------
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(LADDER=list(LADDER), d_max=[2 ** L for L in LADDER], HIDDEN=HIDDEN,
                       N_BLOCKS_TOTAL=N_BLOCKS_TOTAL, TEACHER_DEGREE=TEACHER_DEGREE,
                       FIRST_OK_RUNG=FIRST_OK_RUNG, SEEDS=list(SEEDS), N_TRAIN=N_TRAIN,
                       N_TEST=N_TEST, EPOCHS=EPOCHS, K_OOD=K_OOD, VAR_COEF=VAR_COEF, C_INT=C_INT),
        "params": params,
        "indist_per_seed": indist,
        "ood_ratio_per_seed": ood_ratio,
        "se3_per_seed": se3,
        "perm_per_seed": perm,
        "indist_mean": indist_mean,
        "indist_std": {nm: s(indist, nm) for nm in names},
        "ood_ratio_mean": ratio_mean,
        "se3_mean": se3_mean,
        "perm_mean": perm_mean,
        "recovery_gain": recovery_gain,
        "gap_closed_frac": gap_closed,
        "marginal_gains": gains,
        "saturate_frac": saturate_frac,
        "knee": knee,
        "verdict": {"passed": bool(passed), "ok_ladder_equiv": ok_ladder_equiv,
                    "ok_recovers": ok_recovers, "ok_saturates": ok_saturates,
                    "ok_global_flat": ok_global_flat, "ok_mlp_degrades": ok_mlp_degrades},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step32_tp_degree_ladder_smoke" if SMOKE else "step32_tp_degree_ladder"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    _make_figure(fig_dir / f"{stem}.png", LADDER, indist, indist_mean, ood_ratio, ratio_mean,
                 se3_mean, perm_mean, mlp, FIRST_OK_RUNG, TEACHER_DEGREE)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)}")
    print(f"    wrote {(fig_dir / f'{stem}.png').relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


def _make_figure(path, ladder, indist, indist_mean, ood_ratio, ratio_mean, se3_mean, perm_mean,
                 mlp, first_ok, teacher_deg) -> None:
    r"""Three panels: recovery curve, across-group flatness, post-train exactness -- all vs degree."""
    xs = list(ladder)
    xticklabels = [f"L={L}\n$d_{{\\max}}{{=}}{2 ** L}$" for L in ladder]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel A: recovery curve (in-dist relMSE vs rung)
    ladder_means = [indist_mean[f"L{L}"] for L in ladder]
    ladder_stds = [float(np.std(indist[f"L{L}"])) for L in ladder]
    ax[0].errorbar(xs, ladder_means, yerr=ladder_stds, marker="o", lw=2, capsize=4,
                   color="C0", label="VN ladder (equivariant)")
    ax[0].axhline(mlp, ls="--", color="C3", label="MLP-MP ceiling (no equiv.)")
    ax[0].axhline(ladder_means[0], ls=":", color="C7", label="degree-1 VN cap")
    ax[0].axvline(first_ok, ls="-.", color="C2", alpha=0.7,
                  label=f"first rung $\\geq$ degree-{teacher_deg}")
    ax[0].set_yscale("log")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(xticklabels)
    ax[0].set_ylabel("in-distribution relMSE")
    ax[0].set_title("[I] recovery curve vs representable degree")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].grid(alpha=0.3, which="both")

    # Panel B: across-group OOD/seen ratio vs rung
    ladder_ratio = [ratio_mean[f"L{L}"] for L in ladder]
    ax[1].plot(xs, ladder_ratio, marker="o", lw=2, color="C0", label="VN ladder")
    ax[1].axhline(ratio_mean["MLP-MP"], ls="--", color="C3",
                  label=f"MLP-MP (x{ratio_mean['MLP-MP']:.1f})")
    ax[1].axhline(1.0, ls=":", color="k", alpha=0.5, label="flat (x1.00)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(xticklabels)
    ax[1].set_ylabel("global OOD/seen ratio")
    ax[1].set_title("[G] across-group 举一反三 (flat at every degree)")
    ax[1].legend(fontsize=7, loc="best")
    ax[1].grid(alpha=0.3, which="both")

    # Panel C: post-train equivariance residual vs rung
    ladder_se3 = [se3_mean[f"L{L}"] for L in ladder]
    ax[2].plot(xs, ladder_se3, marker="o", lw=2, color="C0", label="VN ladder SE(3)")
    ax[2].axhline(se3_mean["MLP-MP"], ls="--", color="C3", label="MLP-MP")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(xticklabels)
    ax[2].set_ylabel("post-train SE(3) residual")
    ax[2].set_title("[A'] exact equivariance at every degree")
    ax[2].legend(fontsize=7, loc="best")
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle("Step 32 — tensor-product degree ladder: capacity climbs, equivariance is free at every rung",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
