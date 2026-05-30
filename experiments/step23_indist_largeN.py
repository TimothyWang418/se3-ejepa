r"""Step 23: does the in-distribution gap widen with the symmetry break at **large data**?

Why this step exists -- a genuine tension between Step 16 and Step 22
--------------------------------------------------------------------
Two earlier steps measured the *same* quantity -- the in-wedge (in-distribution) gap between the
non-equivariant MLP and the equivariant VN as the symmetry break $g$ grows -- and disagreed:

  * **Step 16** (sweep $g$ at a *single, large* $N{=}1200$) reported the in-distribution gap
    **widening** with $g$: as the teacher's fixed-lab-$z$ anisotropy grows, the VN -- whose map
    obeys $F(Rx,Ra)=\rho(R)F(x,a)$ *exactly* and therefore **cannot represent** a fixed-axis term --
    falls further behind the unconstrained MLP *on the training wedge itself*.
  * **Step 22** (sweep $g\times N$ with $N\le512$) reported the in-distribution gap **flat-to-
    narrowing** with $g$. Its written-down explanation was a conjecture: *"the Step-16 widening is a
    higher-data ($N{=}1200$) effect; at $N\le512$ both nets are still data-limited and the lab-frame
    term, being a fixed-axis (orientation-free) target, is easy for both, so the VN's in-wedge floor
    does not yet blow up with $g$."*

That conjecture was never tested -- Step 22 simply did not run past $N{=}512$. This step does. It
re-uses Step 22's verified per-cell primitive ``train_eval_cell`` and sweeps the **same** $(g,N)$
design *into the large-$N$ regime* ($N\in\{512,1024,2048\}$, past Step 16's $1200$), reading the
in-distribution (``seen``) wedge relMSE for both models. The single question:

    Does the in-distribution VN$-$MLP gap's dependence on $g$ grow with $N$ -- i.e. is Step 16's
    widening a large-data phenomenon that Step 22 simply could not see at $N\le512$?

The optimisation protocol -- and why it differs from Steps 21/22 on purpose
---------------------------------------------------------------------------
Steps 21/22 fixed the *total gradient-update budget* across $N$ (a compute-matched **frontier**:
more data, same compute). That is exactly wrong for *this* question. Step 16's mechanism is about the
**converged** in-distribution capacity gap: the MLP can fit the fixed-axis term *only if it is given
enough data **and** enough optimisation to actually learn it*. Under a fixed total-update budget the
124k-param MLP is starved at large $N$ (e.g. $N{=}2048$ gets $\sim$8x fewer epochs than $N{=}512$),
so its ``seen`` error would be an **undertraining artifact**, not a representational floor -- it would
falsely flip the gap in the VN's favour and *manufacture* a "reconciliation" that is really just a
compute story. So Step 23 fixes the number of **epochs** (passes over the data) instead, matching the
converged regime Step 22 reached at its $N{=}512$ point ($\approx$150 epochs) at *every* $N$. With more
data per epoch, large $N$ then receives *more* total updates, not fewer -- so the MLP is at least as
converged at $N{=}2048$ as at $N{=}512$, and the measured gap reflects **capacity, not budget**.

A hard guard enforces this honesty: if the MLP fails to converge in-distribution (``mlp_seen`` at the
exact teacher, largest $N$, is not well below the trivial relMSE$=1$ baseline), the run is declared
INCONCLUSIVE rather than reporting the gap. The scientific finding (*reconciled* vs *refuted*) is
reported either way and does **not** gate the exit code, so the experiment cannot be tuned to confirm
its own hypothesis.

Two definitions, to be precise about what "gap" means here
----------------------------------------------------------
At each $(g,N)$ cell, both models trained on the thin $z$-wedge of the misspecified teacher
$\mathrm{Dyn}_g(x,a)_i=\mathrm{Dyn}_0(x,a)_i-g\,\langle e_z,\tilde x_i\rangle\,e_z$ (exactly SO(3) at
$g{=}0$):

  * ``gap(g,N) = vn_seen - mlp_seen`` -- the **in-distribution** gap (seed mean $\pm$ std). Positive
    means the higher-capacity MLP fits the *training wedge* better (the Bitter-Lesson half).
  * ``widen(N) = gap(g_hi, N) - gap(g_lo, N)`` -- how much the in-distribution gap **grows from the
    exact teacher to the most-broken one** at a fixed data size. Step 16 says $>0$ at large $N$;
    Step 22 says $\approx 0$ at $N\le512$. The reconciliation hypothesis is $\mathrm{widen}(N)$
    **increasing in $N$**.

The across-group (``ood``) metric stays a data-proof VN win (Steps 21/22); we re-confirm the $g{=}0$
wall persists at the largest $N$ as a bonus, but Step 23's subject is strictly the *in-distribution*
half, where the Bitter Lesson lives.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step23_indist_largeN.py
    # fast smoke: STEP23_SMOKE=1 .venv/bin/python experiments/step23_indist_largeN.py
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
sys.path.insert(0, str(HERE))   # for the Step 13/16/22 helpers we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402

from step13_se3_latent_jepa import build_eq_jepa, build_mlp_jepa  # noqa: E402
from step16_misspecification import collect_transitions_g, noneq_fraction  # noqa: E402
# Reuse Step 22's *verified* per-cell train+eval primitive VERBATIM: it trains one model on the
# z-wedge of Dyn_g and returns {seen (held-out wedge), ood (genuine full-SO(3)), n_params}.
from step22_symmetry_data_phase import train_eval_cell  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP23_SMOKE"))


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable; no torch state)
# --------------------------------------------------------------------------- #
def updates_for_epochs(N: int, epochs: int, *, bs_cap: int = 128) -> int:
    r"""Total gradient updates that ``train_eval_cell`` converts back to exactly ``epochs`` passes.

    ``train_eval_cell`` recovers ``epochs = round(updates / ceil(N/bs))`` with ``bs=min(bs_cap,N)``.
    Inverting: ``updates = epochs * ceil(N/bs)``. Fixing *epochs* (not total updates) makes the
    in-distribution comparison about **converged capacity**, not compute budget -- and gives large
    $N$ *more* total updates, so the high-capacity MLP is not starved at the right end of the sweep.
    """
    bs = min(bs_cap, N)
    return epochs * math.ceil(N / bs)


def widening_over_N(
    gap_lo: list[float], gap_hi: list[float]
) -> list[float]:
    r"""Per-$N$ widening ``gap_hi - gap_lo``: growth of the in-distribution gap from $g_{lo}$ to $g_{hi}$.

    The reconciliation hypothesis (Step 16's widening is a large-data effect) predicts this list is
    **increasing in $N$**: $\approx 0$ at small $N$ (Step 22's regime), clearly $>0$ at large $N$.
    """
    return [hi - lo for lo, hi in zip(gap_lo, gap_hi)]


def is_increasing(xs: list[float], *, tol: float = 1e-9) -> bool:
    r"""True iff ``xs`` is non-decreasing within ``tol`` (monotone-up check for the widening curve)."""
    return all(xs[i] <= xs[i + 1] + tol for i in range(len(xs) - 1))


# --------------------------------------------------------------------------- #
# inline figure: the in-distribution gap vs N, one line per g (fans out if reconciled)
# --------------------------------------------------------------------------- #
def save_gap_png(G: list[float], N_grid: list[int], grid: dict, out_png: Path) -> None:
    r"""Plot in-distribution gap $\mathrm{vn\_seen}-\mathrm{mlp\_seen}$ ($\pm$ seed std) vs $N$, per $g$.

    If Step 16's widening is a large-data effect, the lines **fan out** as $N$ grows: bunched near a
    small common value at $N{=}512$ (Step 22's regime) and spreading (larger $g$ = larger gap) by the
    largest $N$ (Step 16's regime). A flat fan within error bars would refute the reconciliation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"    [plot skipped: {exc}]")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for g in G:
        gaps = [grid[f"{g}"][f"{N}"]["gap_mean"] for N in N_grid]
        errs = [grid[f"{g}"][f"{N}"]["gap_std"] for N in N_grid]
        ax.errorbar(N_grid, gaps, yerr=errs, marker="o", capsize=3, label=f"g={g:g}")
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_grid, [str(n) for n in N_grid])
    ax.set_xlabel("training-set size  N  (wedge transitions)")
    ax.set_ylabel("in-distribution gap   vn_seen - mlp_seen   (>0: MLP fits wedge better)")
    ax.set_title("Step 23: does the in-distribution gap widen with the symmetry break at large N?")
    ax.legend(title="symmetry break")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"    figure -> {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 80
    # Same knobs as Step 22 (g=0 is exactly SO(3)); N pushed PAST 512 into the Step-16 regime.
    G_VALUES = [0.0, 0.8] if SMOKE else [0.0, 0.4, 0.8]
    N_GRID = [256, 512] if SMOKE else [512, 1024, 2048]
    SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]   # hardened 2026-05-30: 5 seeds so the no-widening std is a credible error bar
    EPOCHS = 30 if SMOKE else 150   # FIXED epochs (convergence), not fixed total updates -- see docstring
    N_TEST = 80 if SMOKE else 400
    VAR_COEF = 0.1

    print(line)
    print("[Step 23] IN-DISTRIBUTION gap vs symmetry break at LARGE N: testing the Step-22 conjecture")
    print(line)
    print("    teacher = exactly-SO(3) Step-13 dynamics + g * fixed lab-z anisotropy (g=0 is exact)")
    print(f"    sweep g={G_VALUES}  x  N={N_GRID} (past Step-22's 512, into Step-16's 1200+ regime)")
    print(f"    {len(SEEDS)} seed(s); FIXED {EPOCHS} epochs/run (convergence, NOT fixed updates); eval N_test={N_TEST}")
    print("    question: does (vn_seen - mlp_seen) grow with g MORE at large N? (reconcile Step 16 vs 22)")
    print()

    builders = {"VN": build_eq_jepa, "MLP": build_mlp_jepa}
    grid: dict[str, dict[str, dict[str, float]]] = {f"{g}": {} for g in G_VALUES}
    noneq: dict[str, float] = {}
    vn_params = mlp_params = 0

    for g in G_VALUES:
        So, Ao, _ = collect_transitions_g(N_TEST, seed=1234, g=g, full_so3=True)
        noneq[f"{g}"] = noneq_fraction(g, So, Ao)
        for N in N_GRID:
            updates = updates_for_epochs(N, EPOCHS)
            # paired per-seed values: same data seed feeds BOTH models, so gap is paired -> std meaningful
            vn_seen_s, mlp_seen_s, vn_ood_s, mlp_ood_s, gap_s = [], [], [], [], []
            for seed in SEEDS:
                rv = train_eval_cell(build_eq_jepa, N, g, seed=seed, updates=updates, n_test=N_TEST, var_coef=VAR_COEF)
                rm = train_eval_cell(build_mlp_jepa, N, g, seed=seed, updates=updates, n_test=N_TEST, var_coef=VAR_COEF)
                vn_seen_s.append(rv["seen"]); vn_ood_s.append(rv["ood"])
                mlp_seen_s.append(rm["seen"]); mlp_ood_s.append(rm["ood"])
                gap_s.append(rv["seen"] - rm["seen"])
                vn_params, mlp_params = rv["n_params"], rm["n_params"]
            grid[f"{g}"][f"{N}"] = {
                "vn_seen": float(np.mean(vn_seen_s)), "vn_ood": float(np.mean(vn_ood_s)),
                "mlp_seen": float(np.mean(mlp_seen_s)), "mlp_ood": float(np.mean(mlp_ood_s)),
                "gap_mean": float(np.mean(gap_s)), "gap_std": float(np.std(gap_s)),
                "epochs": EPOCHS, "updates": updates,
            }
            c = grid[f"{g}"][f"{N}"]
            print(f"    g={g:4.2f} N={N:>5} ({updates:>4}u) | VN seen={c['vn_seen']:6.3f} | "
                  f"MLP seen={c['mlp_seen']:6.3f} | gap={c['gap_mean']:+6.3f}+-{c['gap_std']:.3f} "
                  f"{'(MLP better)' if c['gap_mean'] > 0 else '(VN better)'} | "
                  f"VN ood={c['vn_ood']:5.3f} MLP ood={c['mlp_ood']:6.3f}")

    # ----- the widening analysis (the whole point) ------------------------- #
    g_lo, g_hi = G_VALUES[0], G_VALUES[-1]
    N_lo, N_hi = N_GRID[0], N_GRID[-1]

    def col(g: float, key: str) -> list[float]:
        return [grid[f"{g}"][f"{N}"][key] for N in N_GRID]

    gap_lo_curve = col(g_lo, "gap_mean")          # in-dist gap vs N at the EXACT teacher (g=0)
    gap_hi_curve = col(g_hi, "gap_mean")          # in-dist gap vs N at the MOST-BROKEN teacher
    widen = widening_over_N(gap_lo_curve, gap_hi_curve)
    # pooled seed std at the two extreme-g endpoints (for a "is the widening bigger than noise" read)
    std_pool_hiN = math.sqrt(grid[f"{g_lo}"][f"{N_hi}"]["gap_std"] ** 2 + grid[f"{g_hi}"][f"{N_hi}"]["gap_std"] ** 2)

    print()
    print(line)
    print("WIDENING ANALYSIS  (in-distribution gap's dependence on g, as a function of N)")
    print(line)
    h_lo, h_hi = "gap(g=" + format(g_lo, "g") + ")", "gap(g=" + format(g_hi, "g") + ")"
    print(f"    {'N':>6} | {h_lo:>14} | {h_hi:>14} | {'widen = hi-lo':>14}")
    print("    " + "-" * 58)
    for i, N in enumerate(N_GRID):
        print(f"    {N:>6} | {gap_lo_curve[i]:>+14.3f} | {gap_hi_curve[i]:>+14.3f} | {widen[i]:>+14.3f}")

    widen_increasing = is_increasing(widen)
    widen_lo, widen_hi = widen[0], widen[-1]
    # "reconciled": the widening genuinely emerges with data -- grows from small-N to large-N AND ends
    # positive AND exceeds the pooled seed noise at the largest N (so it is not a fluctuation).
    reconciled = (widen_hi > widen_lo + 0.02) and (widen_hi > 0.02) and (widen_hi > std_pool_hiN)

    vn_wins_indist_g0 = [grid[f"{g_lo}"][f"{N}"]["vn_seen"] < grid[f"{g_lo}"][f"{N}"]["mlp_seen"] for N in N_GRID]
    any_vn_indist_g0 = any(vn_wins_indist_g0)

    # ----- guards: ROBUST facts only (NOT the hypothesis); MLP-convergence gates honesty ---------- #
    mlp_seen_g0_hiN = grid[f"{g_lo}"][f"{N_hi}"]["mlp_seen"]
    ok_mlp_converges = mlp_seen_g0_hiN < 1.0       # MLP actually fits in-wedge (else gap is an artifact)
    ok_vn_fits = grid[f"{g_lo}"][f"{N_hi}"]["vn_seen"] < 0.9
    ok_vn_smaller = vn_params < mlp_params
    ok_ood_wall = grid[f"{g_lo}"][f"{N_hi}"]["mlp_ood"] > 1.3   # g=0 across-group wall data-proof at large N
    noneq_seq = [noneq[f"{g}"] for g in G_VALUES]
    ok_knob = all(noneq_seq[i] <= noneq_seq[i + 1] + 1e-6 for i in range(len(noneq_seq) - 1))
    passed = ok_mlp_converges and ok_vn_fits and ok_vn_smaller and ok_ood_wall and ok_knob

    # ----- summary --------------------------------------------------------- #
    print()
    print(line)
    print("STEP 23 SUMMARY  (resolving the Step 16 vs Step 22 in-distribution tension)")
    print(line)
    if not ok_mlp_converges:
        print(f"    [!] MLP did NOT converge in-distribution (mlp_seen@g=0,N={N_hi} = {mlp_seen_g0_hiN:.3f} >= 1.0):")
        print("        the in-distribution gap is an UNDERTRAINING artifact, not a capacity floor. INCONCLUSIVE.")
        print("        (raise EPOCHS and re-run before drawing any reconciliation conclusion.)")
    print(f"    in-distribution gap (vn_seen - mlp_seen) widening from g={g_lo:g} to g={g_hi:g}:")
    print(f"      at N={N_lo}: widen={widen_lo:+.3f}   ->   at N={N_hi}: widen={widen_hi:+.3f}   "
          f"(monotone-up: {widen_increasing}; pooled seed std @N={N_hi}: {std_pool_hiN:.3f})")
    if ok_mlp_converges and reconciled:
        print("    => RECONCILED: the in-distribution gap's growth with g is a LARGE-DATA effect.")
        print(f"       At N={N_lo} (Step 22's regime) the gap barely moves with g (widen~{widen_lo:+.3f}); by")
        print(f"       N={N_hi} (past Step 16's N=1200) the VN -- which structurally cannot fit the fixed-lab")
        print("       term -- falls clearly further behind the higher-capacity MLP as g grows (Step 16's finding),")
        print("       and the widening exceeds seed noise. Step 16 and Step 22 sampled opposite ends of one N-axis.")
    elif ok_mlp_converges and not reconciled:
        print("    => NOT RECONCILED: with the MLP converged, the widening does NOT clearly emerge with data.")
        print(f"       Step 22's written conjecture (Step 16's widening is purely a large-N effect) is NOT")
        print(f"       supported here (widen {widen_lo:+.3f} -> {widen_hi:+.3f}, vs seed std {std_pool_hiN:.3f}). Honest")
        print("       reading: the single-slice difference has another cause -- soften step22's docstring conjecture.")
    print(f"    does the equivariant prior EVER win in-distribution at g=0?  {any_vn_indist_g0}  "
          f"(per-N {vn_wins_indist_g0}) -- capacity owns the training distribution either way.")
    print(f"    bonus: g=0 across-group MLP wall still DATA-PROOF at N={N_hi} "
          f"(MLP ood={grid[f'{g_lo}'][f'{N_hi}']['mlp_ood']:.3f} > 1.3) -- extends Steps 21/22.")
    print(f"    params: VN={vn_params}  MLP={mlp_params}  ({mlp_params / max(vn_params,1):.1f}x VN).")
    print(f"    guards: mlp-converges={ok_mlp_converges}  vn-fits={ok_vn_fits}  vn-smaller={ok_vn_smaller}  "
          f"ood-wall={ok_ood_wall}  knob-ok={ok_knob}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'} (robust facts); finding "
          f"= {'RECONCILED' if (ok_mlp_converges and reconciled) else ('NOT RECONCILED' if ok_mlp_converges else 'INCONCLUSIVE-undertrained')} "
          "(reported, not gated).")

    # ----- dump JSON + figure ---------------------------------------------- #
    out = {
        "G_values": G_VALUES, "N_grid": N_GRID, "seeds": SEEDS, "epochs": EPOCHS, "n_test": N_TEST,
        "noneq_fraction": noneq, "grid": grid,
        "params": {"VN": vn_params, "MLP": mlp_params},
        "indist_gap_mean": {f"{g_lo}": gap_lo_curve, f"{g_hi}": gap_hi_curve},
        "widening_over_N": widen, "widen_increasing": widen_increasing,
        "widen_lo_N": widen_lo, "widen_hi_N": widen_hi, "pooled_seed_std_hiN": std_pool_hiN,
        "reconciled": bool(ok_mlp_converges and reconciled),
        "mlp_converged": ok_mlp_converges, "mlp_seen_g0_hiN": mlp_seen_g0_hiN,
        "vn_wins_indist_g0_per_N": vn_wins_indist_g0, "any_vn_indist_g0": any_vn_indist_g0,
        "guards": {
            "mlp_converges": ok_mlp_converges, "vn_fits": ok_vn_fits, "vn_smaller": ok_vn_smaller,
            "ood_wall_dataproof": ok_ood_wall, "knob_ok": ok_knob, "passed": passed,
        },
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = "step23_indist_largeN" + ("_smoke" if SMOKE else "") + ".json"
    (fig_dir / fname).write_text(json.dumps(out, indent=2))
    print(f"\n    wrote {fig_dir / fname}")
    save_gap_png(G_VALUES, N_GRID, grid, fig_dir / ("step23_indist_largeN" + ("_smoke" if SMOKE else "") + ".png"))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
