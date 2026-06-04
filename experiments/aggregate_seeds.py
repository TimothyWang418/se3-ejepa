r"""Run the paper2 experiments at seeds {0,1,2}, commit per-seed JSONs, and print true aggregates.

The paper quotes multi-seed ranges for Steps 50/51/52/53/57/58, but each experiment writes a single-seed JSON
(overwritten per run), so the ranges were not reproducible from committed artifacts (red-team blocker). This driver
makes the "3 seeds" claim real and reproducible: it runs each experiment at SEED=0,1,2, saves a per-seed JSON
``<step>_seeds.json`` = {seeds:[...], runs:[json0,json1,json2]}, and prints min/mean/max for the scalar metrics
(and the Step-51 derived ratios) so the paper can quote the true aggregates.

Run:  .venv/bin/python experiments/aggregate_seeds.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers" / "figures"
PY = str(ROOT / ".venv" / "bin" / "python")
SEEDS = [0, 1, 2]
STEPS = [
    "step50_noether_hinge",
    "step51_structure_vs_scale",
    "step52_horizon_resolution",
    "step53_approximate_symmetry",
    "step57_embodied_hinge",
    "step58_3d_containment",
    "step59_pusht_certificate",
    "step60_augmentation",
    "step61_closed_loop_certificate",
]


def env_key(step: str) -> str:
    n = re.match(r"step(\d+)_", step).group(1)
    return f"STEP{n}_SEED"


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


def main() -> None:
    for step in STEPS:
        key = env_key(step)
        runs = []
        for s in SEEDS:
            env = {**os.environ, key: str(s)}
            r = subprocess.run([PY, f"experiments/{step}.py"], cwd=ROOT, env=env,
                               capture_output=True, text=True)
            jf = FIG / f"{step}.json"
            runs.append(json.loads(jf.read_text()))
            print(f"  {step} seed {s}: exit {r.returncode}", file=sys.stderr)
        (FIG / f"{step}_seeds.json").write_text(
            json.dumps({"seeds": SEEDS, "runs": runs}, indent=2, default=float), encoding="utf-8")

        print(f"\n=== {step} (seeds {SEEDS}) ===")
        keys = sorted({k for run in runs for k, v in run.items() if isinstance(v, (int, float))})
        for k in keys:
            a = agg([run.get(k) for run in runs])
            if a:
                print(f"    {k:28s} min {a['min']:.4g}  mean {a['mean']:.4g}  max {a['max']:.4g}")
        # Step-51 derived ratios (the red-team blocker): per-seed dicts mlp_far/mlp_in vs eq_far
        if step == "step51_structure_vs_scale":
            ws = sorted(int(w) for w in runs[0]["mlp_far"].keys())
            small, big = str(min(ws)), str(max(ws))
            floor_pen = [run["mlp_far"][big] / run["eq_far"] for run in runs]          # big-MLP penalty vs equiv floor
            floor_pen_small = [run["mlp_far"][small] / run["eq_far"] for run in runs]   # small-MLP penalty
            inwedge_gain = [run["mlp_in"][small] / run["mlp_in"][big] for run in runs]  # scale buys interpolation
            flat = [run["eq_flatness_ratio"] for run in runs]
            print(f"    [derived] eq flatness ratio: {[round(x,2) for x in flat]}")
            print(f"    [derived] big-MLP(w{big}) out-of-wedge / equiv floor: {[round(x,1) for x in floor_pen]}")
            print(f"    [derived] small-MLP(w{small}) / equiv floor: {[round(x,0) for x in floor_pen_small]}")
            print(f"    [derived] in-wedge gain w{small}->w{big}: {[round(x,0) for x in inwedge_gain]}")
        if step == "step50_noether_hinge":
            ratio = [run["slowest_mode_equivariant"] / run["slowest_mode_invariant"] for run in runs]
            print(f"    [derived] slow-mode ratio (equiv/inv): {[round(x,1) for x in ratio]}")
    print("\nwrote *_seeds.json for", ", ".join(STEPS))


if __name__ == "__main__":
    main()
