r"""Step 96 — the deployed monitor replicates the REMAINING taxonomy cells: both ABSTAIN sub-cases + one more
replication cell. With E15 (cheetah: optimistic ×2 + calibrated ×1, walker: regime caveat) this completes the
full published scope map IN DEPLOYMENT — every cell's behaviour predicted from the step89 artifact a-priori.

Cells and pre-registered predictions (gates frozen before any full run; the only prior contact with these cells is
the 5-episode finger-spin-2 uniformity diagnostic at k=24, disclosed):

- **finger-spin-2 / finger-spin-3 — STABLE-ABSTAIN** ($\lambda_1<0$, bench: 15–19/20 starts censored at 300 steps;
  the certificate issued NO horizon — nothing diverges to price). Deployment prediction: monitoring is FREE — the
  nominal forecast never goes stale. **G3** (per cell, both must pass): belief-invalid fraction at $k{=}24$
  $\le5\%$ (vs $\sim50\%$ at HALF that cadence on cheetah), AND frozen-actuator fault recall $\ge95\%$ at the
  pre-registered $k_{\mathrm{op}}{=}8$ with median delay $\le8$ — zero-false-alarm monitoring at arbitrary cadence
  with detection intact.
- **hopper-hop-1 — BIAS-ABSTAIN** ($\lambda_1=-0.105<0$ yet bench median crossing $5.5$, $0$ censored: divergence
  is bias, not amplification — the certificate ABSTAINED, correctly refusing to price a clock that no Lyapunov
  number governs). Deployment prediction: the in-situ clock is fast DESPITE contraction. **G4**: in-situ crossing
  median $\in[\mathrm{bench}/1.5,\ \mathrm{bench}\times1.5]=[3.7,8.25]$ — deployment replicates the very clock the
  certificate declined to price (a Lyapunov price here would be $\infty$; the abstain is the correct verdict).
  Fault detection at $k_{\mathrm{op}}{=}4$ reported descriptively. Hopper's prior actor may be regime-heterogeneous
  (it can fall); the bias-abstain prediction is regime-robust (fast crossing everywhere) — per-episode spread
  reported.
- **finger-spin-1 — REPLICATION** (weakly expansive $\lambda_1{=}0.099$ yet bench-calibrated $r^{\mathrm{bench}}
  {=}0.95$, $T_1^{\mathrm{pub}}{=}16.3$). **G5** (E15's G1a form): $|r^{\mathrm{insitu}}-0.95|\le0.25$.

Machinery: step94's episode loop reused verbatim via module-global retargeting (same monitor, same $\theta{=}0.2$,
same float64 slices); certificates loaded from the published step89 JSON only.

Run (smoke): STEP96_SMOKE=1 ...; full: .venv/bin/python experiments/step96_taxonomy_monitor.py
Writes: papers/figures/step96_taxonomy_monitor.json
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89  # noqa: E402
import step94_budgeted_monitor as s94  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMOKE = bool(int(os.environ.get("STEP96_SMOKE", "0")))
THETA = 0.2


def retarget(task: str):
    r"""Point step94's module globals at ``task`` (its episode loop reads them at call time)."""
    s94.TASK = task
    s94.DOMAIN, s94.TNAME = task.split("-", 1)
    s94.OBS_DIM, s94.ACT_DIM = s94._REG[task]


def load(task: str, seed: int):
    retarget(task)
    return s89.load_tdmpc2_slices(ROOT / f"models/tdmpc2/{task}-{seed}.pt",
                                  obs_dim=s94.OBS_DIM, action_dim=s94.ACT_DIM)


def cell_stats(sl, n_ep: int, seed: int, k_probe: int = 24):
    r"""No-fault: invalid fraction + crossing distribution at ``k_probe``; returns (inv_mean, crossings)."""
    fr, crossings = [], []
    for e in range(n_ep):
        _, inv, cr = s94.run_episode(sl, k_probe, seed=1000 + 7 * e + seed, fault_t=None, collect_crossings=True)
        fr.append(inv)
        crossings += cr
    return float(np.mean(fr)), np.array(crossings, dtype=float)


def fault_stats(sl, n_ep: int, seed: int, k_op: int):
    rec, delays = 0, []
    for e in range(n_ep):
        tf = 150 + (e * 13) % 100
        flags, _, _ = s94.run_episode(sl, k_op, seed=2000 + 7 * e + seed, fault_t=tf)
        post = [f for f in flags if f >= tf]
        if post:
            rec += 1
            delays.append(post[0] - tf)
    return rec / n_ep, (float(np.median(delays)) if delays else None)


def main() -> int:
    n_ep = int(os.environ.get("STEP96_EPISODES", "4" if SMOKE else "20"))
    out = {"theta": THETA, "n_ep": n_ep, "cells": {}}

    # --- stable-abstain cells: finger-spin-2, finger-spin-3 (G3) ---
    g3 = []
    for seed in [2, 3]:
        sl = load("finger-spin", seed)
        pub = s94.published_cert(seed)
        inv, cr = cell_stats(sl, n_ep, seed)
        recall, delay = fault_stats(sl, n_ep, seed, k_op=8)
        passed = inv <= 0.05 and recall >= 0.95 and delay is not None and delay <= 8
        g3.append(passed)
        out["cells"][f"finger-spin-{seed}"] = {
            "type": "stable-abstain", "lambda1_pub": pub["lambda1_pub"], "bench_censored": pub["bench_censored"],
            "invalid_at_k24": inv, "censored_frac": float((cr > 24).mean()),
            "fault_recall_at_k8": recall, "median_delay": delay, "G3": bool(passed)}
        print(f"[step96] finger-spin-{seed} (stable-abstain): invalid@k24={inv:.3f} censored={out['cells'][f'finger-spin-{seed}']['censored_frac']:.2f} "
              f"recall@k8={recall:.2f} delay={delay} | G3={passed}", file=sys.stderr)

    # --- bias-abstain cell: hopper-hop-1 (G4) ---
    sl = load("hopper-hop", 1)
    pub = s94.published_cert(1)
    bench_med = pub["bench_median"]
    inv, cr = cell_stats(sl, n_ep, 1)
    med = float(np.median(cr))
    q25, q75 = float(np.percentile(cr, 25)), float(np.percentile(cr, 75))
    g4 = bench_med / 1.5 <= med <= bench_med * 1.5
    recall, delay = fault_stats(sl, n_ep, 1, k_op=4)
    out["cells"]["hopper-hop-1"] = {
        "type": "bias-abstain", "lambda1_pub": pub["lambda1_pub"], "bench_median": bench_med,
        "insitu_median": med, "q25": q25, "q75": q75, "invalid_at_k24": inv,
        "fault_recall_at_k4": recall, "median_delay": delay, "G4": bool(g4)}
    print(f"[step96] hopper-hop-1 (bias-abstain): insitu med={med:.1f} (q25={q25:.0f} q75={q75:.0f}) vs bench {bench_med} "
          f"| recall@k4={recall:.2f} delay={delay} | G4={g4}", file=sys.stderr)

    # --- replication cell: finger-spin-1 (G5) ---
    sl = load("finger-spin", 1)
    pub = s94.published_cert(1)
    inv, cr = cell_stats(sl, n_ep, 1)
    med = float(np.median(cr))
    r_insitu = med / pub["T1_pub"]
    g5 = abs(r_insitu - pub["ratio_bench"]) <= 0.25
    out["cells"]["finger-spin-1"] = {
        "type": "replication", "T1_pub": pub["T1_pub"], "ratio_bench": pub["ratio_bench"],
        "insitu_median": med, "censored_frac": float((cr > 24).mean()),
        "ratio_insitu": r_insitu, "G5": bool(g5)}
    print(f"[step96] finger-spin-1 (replication): insitu med={med:.1f} ratio={r_insitu:.2f} vs bench "
          f"{pub['ratio_bench']:.2f} (censored {out['cells']['finger-spin-1']['censored_frac']:.2f}) | G5={g5}",
          file=sys.stderr)

    out["verdict"] = {"G3_stable_abstain": bool(all(g3)), "G4_bias_abstain": bool(g4), "G5_replication": bool(g5)}
    v = out["verdict"]
    print(f"[step96] G3 {'PASS' if v['G3_stable_abstain'] else 'INCONCLUSIVE'} (2/2 needed); "
          f"G4 {'PASS' if v['G4_bias_abstain'] else 'INCONCLUSIVE'}; G5 {'PASS' if v['G5_replication'] else 'INCONCLUSIVE'}.",
          file=sys.stderr)
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step96_taxonomy_monitor{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(out, indent=2))
    return 0 if all(v.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
