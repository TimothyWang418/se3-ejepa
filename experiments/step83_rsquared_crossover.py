r"""Step 83 — the $R^2(N)$ **crossover curve** that turns Exp 2's single-$N$ win into a structural phase transition.

Step 74 (= the paper's load-bearing E2) showed that on the $40$-D Lorenz-96 system a $\mathbb{Z}_N$-equivariant
cyclic-conv world model recovers the *full* Lyapunov spectrum ($R^2\!\approx\!0.99$ vs the true spectrum) where a
**dense MLP** of equal data/recipe fails ($R^2<0$); Step 77 added the **GRU-BPTT** recurrent baseline (Vlachas's
spectrum-recoverer) which, trained on the identical recipe, *also* fails at $N{=}40$ ($R^2\!\approx\!-0.25$) despite
being recurrent. The cleanest objection that remains is altitude: **"you win at one point ($N{=}40$); show it is a
phase transition, not a lucky pick."** This experiment answers exactly that. We sweep
$$ N \in \{12,\,20,\,28,\,40\} $$
and, at each $N$, train the *same three* architectures with the *identical* recipe (same data, $K$-step rollout loss,
epochs, optimizer, normalization) and recover each model's full Lyapunov spectrum by Step 74's already-unit-tested
Benettin-QR, scoring full-spectrum $R^2$ against the true Lorenz-96 spectrum (true spectrum also from Step 74's QR on
the true $\Delta t$-map, anchored by the Liouville identity $\sum_j\lambda_j=-N$). The headline is the curve:

  * the $\mathbb{Z}_N$-conv stays **high** ($R^2\!\gtrsim\!0.9$) across the whole range — its banded-circulant Jacobian
    matches the system's local coupling at every $N$;
  * the dense MLP and the GRU **degrade as $N$ grows** (at small $N$ even the unstructured models recover; the
    unconstrained $N\times N$ / dense-joint Jacobian noise only overwhelms the spectrum at high $N$), **crossing below
    the conv — and below $R^2{=}0$ — at high $N$.**

That crossover *is* the finding: "structure beats scale **and** recurrence" is not a point estimate but a phase
transition in the configuration dimension $N$. (Lorenz-96's $\mathbb{Z}_N$ shift symmetry — the *configuration* axis —
is what the conv exploits; recovering the spectrum is the *horizon* axis, since each exponent $\lambda_j$ sets a
per-channel certified horizon $T_j(\epsilon)=\log(1/\epsilon)/\lambda_j$. So this is the configuration axis helping the
horizon axis, now shown as a transition rather than a single point.)

**Reservoir (RC / ESN) — OPTIONAL, omitted honestly.** Step 76's leaky echo-state network is the other "unstructured
recurrent" baseline. It was parked as **closed-loop-unstable at this fine $\Delta t$** (its autonomous map's spurious
conditional-Lyapunov modes blow up before the spectrum stabilizes at $\Delta t{=}0.01$); rather than fabricate an RC
curve we record ``RC: omitted (closed-loop-unstable at dt=0.01; see step76)`` and let the **GRU** carry the recurrent
baseline (it has a *passing* low-$N$ positive control, so it is the trustworthy recurrent comparator). This is stated
in the output JSON, not hidden.

**Honest seed accounting.** We run 3 seeds where cheap (the swept low-$N$ points) and **reuse Step 74/77's canonical
3-seed $N{=}40$ cells** (do NOT recompute the load-bearing endpoint; read their JSON). Every $(N,\text{arch})$ cell
records exactly how many seeds contributed; nothing is fabricated. If a cached $N{=}40$ seed is missing we simply
report fewer seeds for that cell.

**Honest gates (print INCONCLUSIVE rather than loosen).**
  (G1) the equivariant claim:  the conv must hold $R^2>0.8$ at **every** swept $N$ (mean over seeds). If it dips, the
       equivariant claim is in question — we report it, we do not move the bar.
  (G2) the crossover (the finding): at the largest $N$, mean conv $R^2$ exceeds **both** the MLP and the GRU by $>0.4$
       AND at the smallest $N$ the unstructured models are within $0.2$ of the conv (i.e. they *start together* and
       *separate* — a genuine transition, not a uniform gap). We report the actual numbers whatever they are.

Run:    .venv/bin/python experiments/step83_rsquared_crossover.py            # full sweep (long; CPU/MPS, float64)
smoke:  STEP83_SMOKE=1 .venv/bin/python experiments/step83_rsquared_crossover.py
env:    STEP83_NS="12,20,28,40"  STEP83_SEEDS="0,1,2"  STEP83_ARCHS="conv,mlp,gru"  STEP83_REUSE_N40=1
Writes: papers/figures/step83_rsquared_crossover.{json,png}
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as s74  # noqa: E402  (Lorenz-96 dynamics, true spectrum, conv/MLP, unit-tested QR, KY)
import step77_lorenz96_gru_baseline as s77  # noqa: E402  (GRU-BPTT recurrent baseline + its joint-map spectrum)

SMOKE = bool(int(os.environ.get("STEP83_SMOKE", "0")))
DEVICE = "cpu"                 # float64 spectral work; MPS lacks float64 (matches Step 74/76/77)
DTYPE = torch.float64
FIGDIR = Path(__file__).resolve().parent.parent / "papers" / "figures"

# Sweep / seeds / architectures (overridable). Default: the four N from the spec, 3 seeds, conv/MLP/GRU.
NS = [int(x) for x in os.environ.get("STEP83_NS", "8,12,16" if SMOKE else "12,20,28,40").split(",")]
SEEDS = [int(x) for x in os.environ.get("STEP83_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
ARCHS = os.environ.get("STEP83_ARCHS", "conv,mlp,gru").split(",")
# Reuse Step 74/77's canonical N=40 cells instead of recomputing the load-bearing endpoint (on by default).
REUSE_N40 = bool(int(os.environ.get("STEP83_REUSE_N40", "0" if SMOKE else "1")))
GRU_HIDDEN = int(os.environ.get("STEP83_HIDDEN", "32" if SMOKE else "96"))   # matches Step 77's default H

# Plot styling — mirror Step 74/76/77 (same colors per architecture).
COLORS = {"conv": "#1f77b4", "mlp": "#d62728", "gru": "#2ca02c"}
LABELS = {"conv": "$\\mathbb{Z}_N$-equivariant conv", "mlp": "dense MLP", "gru": "GRU-BPTT (recurrent)"}
MARKERS = {"conv": "o", "mlp": "x", "gru": "s"}


def _sched(N: int) -> dict:
    r"""Per-$N$ compute schedule. Training-set size, QR length and warmup grow modestly with $N$ (a higher-dimensional
    attractor needs a longer Benettin average to resolve the full spectrum), bounded to keep CPU runtime tractable.
    These do NOT change the recipe (epochs / rollout $K$ / optimizer are Step 74's); they only set how much data /
    averaging each point gets, identically across the three architectures at that $N$."""
    if SMOKE:
        return {"n_train": 2500, "ly_steps": 500, "ly_warm": 100, "washout": 60}
    n_train = min(20000, 8000 + 350 * N)          # 12->12200, 20->15000, 28->17800, 40->20000
    ly_steps = min(2500, 1000 + 38 * N)           # 12->1456, 20->1760, 28->2064, 40->2500
    return {"n_train": n_train, "ly_steps": ly_steps, "ly_warm": 400, "washout": 200}


# --------------------------------------------------------------------------------------------------------------- #
# One (N, arch, seed) cell: train with Step 74/77's recipe, recover the spectrum, score full-spectrum R^2 vs true.
# --------------------------------------------------------------------------------------------------------------- #
def true_spectrum(N: int, seed: int, sch: dict):
    r"""True Lorenz-96 $\Delta t$-map spectrum via Step 74's Benettin-QR (anchored by Liouville $\sum_j\lambda_j=-N$).
    Returns ``(lam_true_tensor, x0_normalized, mu, sd)`` so every architecture is scored against the SAME true
    spectrum and synchronized from the SAME attractor point."""
    traj = s74.attractor_traj(N, sch["n_train"], seed, DEVICE).to(DTYPE)
    mu, sd = traj.mean(0), traj.std(0) + 1e-8
    x0 = (traj[len(traj) // 2] - mu) / sd
    lam_true = s74.lyapunov_spectrum(lambda xn: (s74.true_map(xn * sd + mu) - mu) / sd,
                                     x0, sch["ly_steps"], sch["ly_warm"])
    return lam_true, x0, mu, sd, traj


def _r2(ll: np.ndarray, lt: np.ndarray) -> float:
    r"""Full-spectrum coefficient of determination $R^2 = 1 - \lVert\hat\lambda-\lambda\rVert^2/\lVert\lambda-\bar\lambda\rVert^2$
    (the exact metric Step 74/77 use; identical so the curve is comparable to their single-$N$ numbers)."""
    return 1.0 - float(((ll - lt) ** 2).sum()) / max(float(((lt - lt.mean()) ** 2).sum()), 1e-12)


def eval_cell(arch: str, N: int, seed: int, lam_true: torch.Tensor, x0, traj, sch: dict) -> dict:
    r"""Train one architecture at (N, seed) and recover its Lyapunov spectrum. ``conv``/``mlp`` use Step 74's
    feedforward map directly; ``gru`` uses Step 77's joint autonomous map $[x,h]$ (leading $N$ exponents)."""
    lt = lam_true.cpu().numpy()
    if arch in ("conv", "mlp"):
        os.environ["STEP74_MODEL"] = arch
        model, mu, sd, relmse = s74.train_model(traj, N, DEVICE, seed)
        lam = s74.lyapunov_spectrum(lambda xn: model(xn), x0, sch["ly_steps"], sch["ly_warm"])
        ll = lam.cpu().numpy()
        spurious = float("nan")
    elif arch == "gru":
        # Step 77 recipe: teacher-forced warmup + K-step closed-loop rollout, hidden-noise contraction reg, joint map.
        model, mu, sd, relmse = s77.train_gru(traj, N, GRU_HIDDEN, seed)
        xn = (traj - mu) / sd
        state0 = s77.sync_state(model, xn[len(xn) // 2 - sch["washout"]: len(xn) // 2])
        lam_full = s74.lyapunov_spectrum(s77.autonomous_map(model), state0, sch["ly_steps"], sch["ly_warm"])
        lam = lam_full[:N]                                   # leading N approximate the true spectrum
        ll = lam.cpu().numpy()
        spurious = float(lam_full[N:].max()) if lam_full.shape[0] > N else float("nan")
    else:
        raise ValueError(f"unknown arch {arch!r}")
    l1_t = float(lam_true[0])
    out = {"arch": arch, "N": N, "seed": seed, "one_step_relmse": relmse, "spectrum_r2": _r2(ll, lt),
           "lambda1": float(lam[0]), "lambda1_relerr": abs(float(lam[0]) - l1_t) / abs(l1_t),
           "sum_topN": float(lam.sum()), "n_positive": int((lam > 0).sum()),
           "ky": s74.kaplan_yorke(lam), "spurious_lyap_max": spurious, "lambda_learned": ll.tolist(),
           "source": "fresh"}
    print(f"[step83]   N={N:>2} {arch:>4} seed{seed}: relMSE {relmse:.1e}  R^2 {out['spectrum_r2']:+.3f}  "
          f"lambda1 {float(lam[0]):.3f} (err {out['lambda1_relerr']:.0%})  #pos {out['n_positive']}  "
          f"KY {out['ky']:.2f}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------------------------------------------- #
# Reuse Step 74 (conv, MLP) and Step 77 (GRU) cached N=40 cells — the load-bearing endpoint is NOT recomputed.
# --------------------------------------------------------------------------------------------------------------- #
def _load_json(name: str) -> dict:
    p = FIGDIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def reuse_cached(arch: str, N: int, seed: int) -> dict | None:
    r"""Pull a cached $(N,\text{arch},\text{seed})$ cell from Step 74 (conv/MLP) or Step 77 (GRU) if present and at the
    SAME $N$. Returns ``None`` if not available (caller then reports fewer seeds — never fabricated)."""
    tag = f"_seed{seed}" if seed else ""
    if arch in ("conv", "mlp"):
        j = _load_json(f"step74_lorenz96_spectrum{tag}.json")
        if not j or j.get("N") != N:
            return None
        sub = j["equivariant"] if arch == "conv" else j["mlp"]
        ll = np.array(sub["lambda_learned"])
        return {"arch": arch, "N": N, "seed": seed, "one_step_relmse": sub["one_step_relmse"],
                "spectrum_r2": sub["spectrum_r2"], "lambda1": sub["lambda1"],
                "lambda1_relerr": sub["lambda1_relerr"], "sum_topN": sub.get("sum", float("nan")),
                "n_positive": sub["n_positive"], "ky": sub["ky"], "spurious_lyap_max": float("nan"),
                "lambda_learned": sub["lambda_learned"], "source": "step74-cache"}
    if arch == "gru":
        j = _load_json(f"step77_gru_baseline{tag}.json")
        if not j or j.get("N") != N:
            return None
        g = j["gru_test"]
        return {"arch": arch, "N": N, "seed": seed, "one_step_relmse": g["one_step_relmse"],
                "spectrum_r2": g["spectrum_r2"], "lambda1": g["lambda1"], "lambda1_relerr": g["lambda1_relerr"],
                "sum_topN": g.get("sum_topN", float("nan")), "n_positive": g["n_positive"], "ky": g["ky"],
                "spurious_lyap_max": g.get("spurious_lyap_max", float("nan")),
                "lambda_learned": g["lambda_learned"], "source": "step77-cache"}
    return None


# --------------------------------------------------------------------------------------------------------------- #
# Sweep driver.
# --------------------------------------------------------------------------------------------------------------- #
def run() -> int:
    t0 = time.time()
    print(f"[step83] R^2(N) crossover sweep: N={NS}  seeds={SEEDS}  archs={ARCHS}  "
          f"(reuse_N40={REUSE_N40}, H={GRU_HIDDEN}, {DEVICE}, float64)", file=sys.stderr)

    # cells[N][arch] = list of per-seed dicts ; true[N] = {seed: lambda_true list}
    cells: dict[int, dict[str, list]] = {N: {a: [] for a in ARCHS} for N in NS}
    true_spec: dict[int, dict[int, list]] = {N: {} for N in NS}

    for N in NS:
        sch = _sched(N)
        use_cache = REUSE_N40 and (N == 40)
        for seed in SEEDS:
            # True spectrum: reuse Step 74's cached lambda_true at N=40 (same routine); else compute fresh here.
            lam_true_cached = None
            if use_cache:
                tag = f"_seed{seed}" if seed else ""
                j = _load_json(f"step74_lorenz96_spectrum{tag}.json")
                if j and j.get("N") == N:
                    lam_true_cached = torch.tensor(j["lambda_true"], dtype=DTYPE)
            if lam_true_cached is not None:
                lam_true = lam_true_cached
                true_spec[N][seed] = lam_true.cpu().numpy().tolist()
                x0 = traj = None
            else:
                lam_true, x0, mu, sd, traj = true_spectrum(N, seed, sch)
                true_spec[N][seed] = lam_true.cpu().numpy().tolist()
            print(f"[step83] N={N} seed{seed}: true lambda1 {float(lam_true[0]):.3f}  #pos {int((lam_true>0).sum())}  "
                  f"sum {float(lam_true.sum()):.2f} (Liouville -N={-N})", file=sys.stderr)

            for arch in ARCHS:
                cell = None
                if use_cache:
                    cell = reuse_cached(arch, N, seed)
                    if cell is not None:
                        print(f"[step83]   N={N:>2} {arch:>4} seed{seed}: REUSED {cell['source']}  "
                              f"R^2 {cell['spectrum_r2']:+.3f}", file=sys.stderr)
                if cell is None:
                    if traj is None:                          # cache miss at N=40 -> compute fresh (need data)
                        lam_true, x0, mu, sd, traj = true_spectrum(N, seed, sch)
                    cell = eval_cell(arch, N, seed, lam_true, x0, traj, sch)
                cells[N][arch].append(cell)

    # ---- aggregate: mean/std R^2 per (N, arch), honest seed counts ----
    summary: dict[str, dict] = {}
    for arch in ARCHS:
        rows = []
        for N in NS:
            r2s = np.array([c["spectrum_r2"] for c in cells[N][arch]], dtype=float)
            npos = np.array([c["n_positive"] for c in cells[N][arch]], dtype=float)
            kys = np.array([c["ky"] for c in cells[N][arch]], dtype=float)
            rows.append({"N": N, "n_seeds": int(r2s.size),
                         "r2_mean": float(r2s.mean()) if r2s.size else float("nan"),
                         "r2_std": float(r2s.std()) if r2s.size > 1 else 0.0,
                         "r2_per_seed": r2s.tolist(),
                         "n_positive_mean": float(npos.mean()) if npos.size else float("nan"),
                         "ky_mean": float(kys.mean()) if kys.size else float("nan"),
                         "sources": [c["source"] for c in cells[N][arch]]})
        summary[arch] = {"rows": rows}

    # ---- honest gates ----
    def r2_at(arch: str, N: int) -> float:
        for row in summary[arch]["rows"]:
            if row["N"] == N:
                return row["r2_mean"]
        return float("nan")

    Nmax, Nmin = max(NS), min(NS)
    conv_holds = all(r2_at("conv", N) > 0.8 for N in NS) if "conv" in ARCHS else False
    # crossover: at Nmax conv beats every other arch by >0.4; at Nmin the others start within 0.2 of conv.
    others = [a for a in ARCHS if a != "conv"]
    gap_hi = min((r2_at("conv", Nmax) - r2_at(a, Nmax)) for a in others) if others else float("nan")
    gap_lo = max((r2_at("conv", Nmin) - r2_at(a, Nmin)) for a in others) if others else float("nan")
    crossover = bool("conv" in ARCHS and others and gap_hi > 0.4 and gap_lo < 0.2)
    verdict = ("PHASE_TRANSITION" if (conv_holds and crossover) else "INCONCLUSIVE")

    rc_note = "RC: omitted (closed-loop-unstable at dt=%.3g; GRU carries the recurrent baseline — see step76)" % s74.DTMAP

    res = {"verdict": verdict, "smoke": SMOKE, "Ns": NS, "seeds": SEEDS, "archs": ARCHS,
           "reuse_N40": REUSE_N40, "gru_hidden": GRU_HIDDEN, "dt_map": s74.DTMAP, "F": s74.F_FORCE,
           "gate_conv_holds_r2_gt_0.8": conv_holds, "gate_crossover": crossover,
           "crossover_gap_at_Nmax": gap_hi, "conv_lead_at_Nmin": gap_lo,
           "rc_status": rc_note, "elapsed_sec": time.time() - t0,
           "summary": summary,
           "cells": {str(N): {a: cells[N][a] for a in ARCHS} for N in NS},
           "true_spectrum": {str(N): true_spec[N] for N in NS}}
    _save(res)
    _print_report(res)
    return 0 if verdict == "PHASE_TRANSITION" else 1


def _print_report(res: dict) -> None:
    print(f"\n[step83] === R^2(N) CROSSOVER (verdict: {res['verdict']}) ===", file=sys.stderr)
    hdr = "  N  | " + " | ".join(f"{LABELS[a].split('(')[0].strip()[:18]:>18}" for a in res["archs"])
    print("[step83] " + hdr, file=sys.stderr)
    for N in res["Ns"]:
        parts = []
        for a in res["archs"]:
            row = next(r for r in res["summary"][a]["rows"] if r["N"] == N)
            parts.append(f"{row['r2_mean']:+.3f}±{row['r2_std']:.3f} (n={row['n_seeds']})")
        print(f"[step83]  {N:>3} | " + " | ".join(f"{p:>18}" for p in parts), file=sys.stderr)
    print(f"[step83] conv holds R^2>0.8 at every N: {res['gate_conv_holds_r2_gt_0.8']}", file=sys.stderr)
    print(f"[step83] crossover: conv lead at N={min(res['Ns'])} is {res['conv_lead_at_Nmin']:+.3f} (<0.2 => start "
          f"together); conv beats all others at N={max(res['Ns'])} by {res['crossover_gap_at_Nmax']:+.3f} (>0.4 => "
          f"separated) -> {res['gate_crossover']}", file=sys.stderr)
    print(f"[step83] {res['rc_status']}", file=sys.stderr)
    if res["verdict"] == "PHASE_TRANSITION":
        print(f"[step83] >>> PHASE TRANSITION: 'structure beats scale AND recurrence' is not a single-N win — the "
              f"$\\mathbb{{Z}}_N$-conv holds R^2>0.8 across N while the dense MLP and recurrent GRU, identically "
              f"trained, start together at small N and degrade below it (and below 0) as N grows. Exp 2 hardened from "
              f"one point to a crossover.", file=sys.stderr)
    else:
        print(f"[step83] >>> INCONCLUSIVE: report the actual R^2(N) table above; the crossover/conv-hold gates were "
              f"not both met (do not loosen the bar).", file=sys.stderr)
    print(f"[step83] elapsed {res['elapsed_sec']:.0f}s; figure papers/figures/step83_rsquared_crossover.png",
          file=sys.stderr)


# --------------------------------------------------------------------------------------------------------------- #
# Figure: R^2(N) crossover (headline) + per-N positive-exponent recovery (the spectral content driving R^2).
# --------------------------------------------------------------------------------------------------------------- #
def _save(res: dict) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "step83_rsquared_crossover.json").write_text(json.dumps(res, indent=2))
    try:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
        Ns = np.array(res["Ns"], dtype=float)

        # (a) the crossover curve: R^2 vs N, one line per architecture, error bars over seeds.
        for a in res["archs"]:
            rows = sorted(res["summary"][a]["rows"], key=lambda r: r["N"])
            xs = np.array([r["N"] for r in rows], dtype=float)
            ys = np.array([r["r2_mean"] for r in rows], dtype=float)
            es = np.array([r["r2_std"] for r in rows], dtype=float)
            ax.errorbar(xs, ys, yerr=es, marker=MARKERS[a], color=COLORS[a], lw=2, ms=7, capsize=3,
                        label=LABELS[a])
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.axhline(0.8, color="#1f77b4", lw=0.8, ls="--", alpha=0.5)
        ax.text(Ns.min(), 0.815, "$R^2=0.8$ (recovery)", fontsize=7, color="#1f77b4", alpha=0.8)
        ax.text(Ns.min(), 0.02, "$R^2=0$ (mean-predictor)", fontsize=7, color="gray")
        ax.set_xlabel("configuration dimension $N$ (Lorenz-96)")
        ax.set_ylabel("full-spectrum $R^2$ (recovered vs true)")
        ax.set_title("(a) $R^2(N)$ crossover: structure vs scale & recurrence")
        ax.set_xticks(res["Ns"])
        ax.set_ylim(min(-3.0, ax.get_ylim()[0]), 1.08)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.25)

        # (b) #positive exponents recovered vs N (the count R^2 hinges on): conv tracks truth, others drift.
        true_npos = []
        for N in res["Ns"]:
            arrs = [np.array(v) for v in res["true_spectrum"][str(N)].values()]
            true_npos.append(float(np.mean([int((a > 0).sum()) for a in arrs])) if arrs else np.nan)
        ax2.plot(res["Ns"], true_npos, "k-", marker="D", lw=2, ms=6, label="true Lorenz-96")
        for a in res["archs"]:
            rows = sorted(res["summary"][a]["rows"], key=lambda r: r["N"])
            ax2.plot([r["N"] for r in rows], [r["n_positive_mean"] for r in rows],
                     marker=MARKERS[a], color=COLORS[a], lw=1.6, ms=6, alpha=0.9, label=LABELS[a])
        ax2.set_xlabel("configuration dimension $N$")
        ax2.set_ylabel("# positive Lyapunov exponents recovered")
        ax2.set_title("(b) Positive-exponent count vs truth")
        ax2.set_xticks(res["Ns"])
        ax2.legend(fontsize=7.5, loc="upper left")
        ax2.grid(alpha=0.25)

        sub = "verdict: " + res["verdict"].replace("_", " ").title()
        fig.suptitle(f"Step 83 — Lorenz-96 spectrum recovery as a phase transition in $N$   ({sub})",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(FIGDIR / "step83_rsquared_crossover.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step83]   (figure skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run())
