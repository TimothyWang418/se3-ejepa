r"""Step 89c — measured-column thickening of the 84-cell audit map: n_starts 20 -> 100.

Spec (frozen before any cell ran): `docs/specs/2026-06-11-step89c-measured-thicken-seed.md`.
- ONLY the measured side is recomputed (same ``measure()``, same eps grid, same mid-episode start
  convention, same checkpoints); ratios re-formed against the PUBLISHED ``T1_steps`` of the
  step89/step89b artifacts. No gate, no tuning — regime cuts stay as published; fractions are
  folded honestly whatever they say. n=20 canonical artifacts are not touched.
- Incremental + resumable: JSON rewritten after every cell; errored cells retried on restart.

Run (box): .venv/bin/python experiments/step89c_measured_thicken.py      [STEP89C_LIMIT=N smoke]
Writes: papers/figures/step89c_measured_n100.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "in1d"):
    np.in1d = np.isin                        # NumPy 2.x removed the alias dm_control still uses

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89                      # noqa: E402
import step89b_audit_expansion as s89b                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers/figures/step89c_measured_n100.json"
N_STARTS = 100
EPS_LIST = [0.05, 0.1, 0.2]
LIMIT = int(os.environ.get("STEP89C_LIMIT", "0"))


def published_cells() -> dict:
    """All 84 published cells: {cell: {eps: T1_steps_or_None}} from the canonical artifacts."""
    cells = {}
    for name in ("step89_pretrained_audit.json", "step89b_audit_expansion.json"):
        for k, v in json.loads((ROOT / "papers/figures" / name).read_text()).items():
            if k.startswith("_") or "error" in v:
                continue
            cells[k] = {str(r["eps"]): r["T1_steps"] for r in v["cert_rows"]}
    return cells


def main() -> int:
    from dm_control import suite
    torch.manual_seed(0)
    pub = published_cells()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    n_new = 0
    for cell in sorted(pub):
        if cell in results and "error" not in results[cell]:
            continue
        if LIMIT and n_new >= LIMIT:
            break
        key, seed = cell.rsplit("-", 1)
        seed = int(seed)
        try:
            if key not in s89.TASKS:                      # register dims for expansion tasks
                domain, task = s89b.map_task(key)
                env = suite.load(domain, task, task_kwargs={"random": 0})
                ts = env.reset()
                s89.TASKS[key] = (domain, task, int(s89._flat_obs(ts).numel()),
                                  int(env.action_spec().shape[0]))
                env.close()
            ck = s89b.ensure_ckpt(key, seed)
            raw = torch.load(ck, map_location="cpu", weights_only=True)
            sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
            obs_dim, act_dim = s89.TASKS[key][2], s89.TASKS[key][3]
            slices = s89.load_tdmpc2_slices({"model": s89b._maybe_translate_old_format(sd)},
                                            action_dim=act_dim, obs_dim=obs_dim)
            meas = s89.measure(key, slices, EPS_LIST, n_starts=N_STARTS, seed=seed)
            row = {"measured_n100": meas, "ratio_n100": {}}
            for e in EPS_LIST:
                t1 = pub[cell][str(e)]
                row["ratio_n100"][str(e)] = (meas[str(e)]["median"] / t1) if t1 else None
            results[cell] = row
            n_new += 1
            m2 = meas["0.2"]
            r2 = row["ratio_n100"]["0.2"]
            print(f"[step89c] {cell}: med@0.2={m2['median']:.0f} cen={m2['n_censored']} "
                  f"ratio={'%.2f' % r2 if r2 else '—'} [{n_new} new]", file=sys.stderr)
        except Exception as e:
            results[cell] = {"error": repr(e)[:300]}
            print(f"[step89c] {cell}: ERROR {repr(e)[:120]}", file=sys.stderr)
        OUT.write_text(json.dumps(results, indent=1))
    ok = sum(1 for k, v in results.items() if "error" not in v)
    err = sum(1 for k, v in results.items() if "error" in v)
    print(f"[step89c] DONE: {ok} cells ok, {err} errors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
