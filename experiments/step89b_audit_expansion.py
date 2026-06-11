r"""Step 89b — E13 audit EXPANSION: the entire public single-task TD-MPC2 zoo (15 -> up to 84 cells).

Spec (frozen before any cell ran): `docs/specs/2026-06-11-step89b-audit-expansion-seed.md`.
- UNIVERSE = the 34 unaudited dmcontrol tasks enumerated on the official HF repo (2026-06-11), seeds 1-3.
- Inclusion rule, pre-registered and SELF-EXECUTING: a task enters iff stock dm_control ``suite.load`` accepts its
  (domain, task) mapping (first token = domain, ``cup``->``ball_in_cup``; rest joined by underscores). TD-MPC2's
  custom variants are excluded BY THE RULE, not by choice — the measured cross-check column (load-bearing per E16)
  requires the true environment. The excluded list is written into the artifact.
- Protocol: verbatim step89 — same ``certify``/``measure``, same mid-episode z0 convention, same eps grid.
  Dims derived at runtime (action_dim from the env spec, obs_dim from the flattened reset observation;
  ``strict=True`` state-dict load implicitly asserts the encoder dimension). Checkpoints downloaded on demand
  from the pinned official URLs (never vendored).
- Incremental + resumable: the JSON is rewritten after EVERY cell; existing keys are skipped on restart.

Run (box): .venv/bin/python experiments/step89b_audit_expansion.py     [STEP89B_LIMIT=N for smoke]
Writes: papers/figures/step89b_audit_expansion.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers/figures/step89b_audit_expansion.json"
HF = "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol"
LIMIT = int(os.environ.get("STEP89B_LIMIT", "0"))          # 0 = no limit; N = stop after N new cells (smoke)
EPS_LIST = [0.05, 0.1, 0.2]

UNIVERSE = [
    "cartpole-balance", "cartpole-balance-sparse", "cartpole-swingup", "cartpole-swingup-sparse",
    "cheetah-jump", "cheetah-run-back", "cheetah-run-backwards", "cheetah-run-front",
    "cup-catch", "cup-spin", "dog-run", "dog-stand", "dog-trot", "dog-walk",
    "finger-turn-easy", "finger-turn-hard", "fish-swim", "hopper-hop-backwards", "hopper-stand",
    "humanoid-run", "humanoid-stand", "humanoid-walk", "pendulum-spin", "pendulum-swingup",
    "quadruped-run", "quadruped-walk", "reacher-easy", "reacher-hard",
    "reacher-three-easy", "reacher-three-hard",
    "walker-run", "walker-run-backwards", "walker-stand", "walker-walk-backwards",
]
SPECIAL_DOMAIN = {"cup": "ball_in_cup"}


def map_task(key: str):
    toks = key.split("-")
    return SPECIAL_DOMAIN.get(toks[0], toks[0]), "_".join(toks[1:])


def ensure_ckpt(key: str, seed: int) -> Path:
    ck = s89.CKPT_DIR / f"{key}-{seed}.pt"
    if not ck.exists():
        ck.parent.mkdir(parents=True, exist_ok=True)
        print(f"[step89b] downloading {key}-{seed}.pt ...", file=sys.stderr)
        subprocess.run(["curl", "-sL", "-o", str(ck), f"{HF}/{key}-{seed}.pt"], check=True)
        assert ck.stat().st_size > 1_000_000, f"download failed: {ck}"
    return ck


def main() -> int:
    from dm_control import suite
    torch.manual_seed(0)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    excluded, included = [], []
    n_new = 0
    for key in UNIVERSE:
        domain, task = map_task(key)
        try:
            env = suite.load(domain, task, task_kwargs={"random": 0})
        except Exception as e:
            excluded.append(key)
            continue
        ts = env.reset()
        obs_dim = int(s89._flat_obs(ts).numel())
        act_dim = int(env.action_spec().shape[0])
        env.close()
        s89.TASKS[key] = (domain, task, obs_dim, act_dim)
        included.append(key)
        for seed in (1, 2, 3):
            cell = f"{key}-{seed}"
            if cell in results:
                continue
            if LIMIT and n_new >= LIMIT:
                break
            try:
                ck = ensure_ckpt(key, seed)
                print(f"[step89b] {cell}: loading slices (obs={obs_dim}, act={act_dim}) ...", file=sys.stderr)
                slices = s89.load_tdmpc2_slices(ck, action_dim=act_dim, obs_dim=obs_dim)
                zs = s89.rollout_true(key, slices, 60, seed)
                z0 = zs[len(zs) // 2]
                cert = s89.certify(slices, z0, EPS_LIST, seed=seed)
                meas = s89.measure(key, slices, EPS_LIST, seed=seed)
                row = {"lambda1": cert["lambda1"], "lambda1_ci": cert["lambda1_ci"],
                       "n_structural_band": cert["n_structural_band"], "cert_rows": cert["rows"],
                       "measured": meas, "obs_dim": obs_dim, "action_dim": act_dim}
                for r in cert["rows"]:
                    m = meas[str(r["eps"])]
                    r["measured_median"] = m["median"]
                    r["ratio_measured_over_certified"] = (m["median"] / r["T1_steps"]) if r["T1_steps"] else None
                results[cell] = row
                n_new += 1
                top = [r for r in cert["rows"] if r["eps"] == 0.2][0]
                print(f"[step89b] {cell}: lam1={cert['lambda1']:.4f} "
                      f"T1@0.2={'%.1f' % top['T1_steps'] if top['T1_steps'] else 'ABSTAIN'} "
                      f"med={top['measured_median']:.0f} "
                      f"ratio={'%.2f' % top['ratio_measured_over_certified'] if top['ratio_measured_over_certified'] else '—'} "
                      f"[{n_new} new]", file=sys.stderr)
            except Exception as e:
                results[cell] = {"error": repr(e)[:300]}
                print(f"[step89b] {cell}: ERROR {repr(e)[:120]}", file=sys.stderr)
            results["_meta"] = {"universe": UNIVERSE, "excluded_by_rule": excluded, "included": included,
                                "spec": "docs/specs/2026-06-11-step89b-audit-expansion-seed.md"}
            OUT.write_text(json.dumps(results, indent=2))
        if LIMIT and n_new >= LIMIT:
            break
    results["_meta"] = {"universe": UNIVERSE, "excluded_by_rule": excluded, "included": included,
                        "spec": "docs/specs/2026-06-11-step89b-audit-expansion-seed.md"}
    OUT.write_text(json.dumps(results, indent=2))
    n_cells = sum(1 for k, v in results.items() if not k.startswith("_") and "error" not in v)
    n_err = sum(1 for k, v in results.items() if not k.startswith("_") and "error" in v)
    print(f"[step89b] DONE: {n_cells} cells ok, {n_err} errors, excluded by rule: {excluded}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
