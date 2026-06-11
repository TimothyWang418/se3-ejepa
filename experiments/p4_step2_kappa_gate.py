r"""P4-step2 — κ validation gate: twin-trajectory $\hat\lambda_1(\kappa)$ of the TRUE PushT env.

Spec: docs/specs/2026-06-10-p4-step2-kappa-gate-seed.md (protocol + GATE-A/B pre-registered).
No learned model is involved; this certifies the *knob*, not a model.

Run (smoke):  P4_SMOKE=1 .venv/bin/python experiments/p4_step2_kappa_gate.py
Run (full):   .venv/bin/python experiments/p4_step2_kappa_gate.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_step1_pipeline import make_env, weak_action  # noqa: E402

SMOKE = bool(int(os.environ.get("P4_SMOKE", "0")))
SEED = 0
KAPPAS = [0.0, 1.0] if SMOKE else [0.0, 0.5, 0.8, 0.95, 1.0]
N_EPISODES = 2 if SMOKE else 10
EP_LEN = 30 if SMOKE else 100
# NOTE (post-run review): with WINDOW=60 and EP_LEN=100, the t=55 capture is never collected
# (55+60>100) — the 2026-06-10 run's as-run n is 30/κ, not 40 (record entry corrected; the fit
# filter dropped 0). Fixing this grid (e.g. {10,25,40} or EP_LEN=120) = registered protocol v1.1
# for any rerun; the gate verdict stands on n=30.
CAPTURE_AT = [10, 20] if SMOKE else [10, 25, 40, 55]
WINDOW = 15 if SMOKE else 60
DELTA0 = 1e-4
N_BOOT = 200 if SMOKE else 1000
OUT_JSON = ROOT / "papers" / "figures" / ("p4_step2_kappa_gate_smoke.json" if SMOKE else "p4_step2_kappa_gate.json")

# diagonal normalizer for the config-space metric (positions /512, angle /pi) — registered
POS_S, ANG_S = 512.0, np.pi


# --------------------------------------------------------------------- extended state (10-d)
def get_ext(env) -> np.ndarray:
    e = env.unwrapped
    return np.array(
        [*e.agent.position, *e.block.position, e.block.angle,
         *e.agent.velocity, *e.block.velocity, e.block.angular_velocity],
        dtype=np.float64,
    )


def set_ext(env, s: np.ndarray) -> None:
    r"""Mirror _set_state (env.py:514-527) + the block momentum it omits. No settle step —
    twins must start from EXACTLY this state; the common rollout does the stepping."""
    e = env.unwrapped
    e.agent.position = tuple(s[0:2])
    e.block.position = tuple(s[2:4])
    e.block.angle = float(s[4])
    e.agent.velocity = tuple(s[5:7])
    e.block.velocity = tuple(s[7:9])
    e.block.angular_velocity = float(s[9])
    e.space.reindex_shapes_for_body(e.block)
    e.space.reindex_shapes_for_body(e.agent)


def config_dist(a: np.ndarray, b: np.ndarray) -> float:
    dpos = np.linalg.norm(a[[0, 1, 2, 3]] - b[[0, 1, 2, 3]]) / POS_S
    dth = abs(a[4] - b[4])
    dth = min(dth, 2 * np.pi - dth) / ANG_S
    return float(dpos + dth)


# --------------------------------------------------------------------- collection + twins
def collect_pairs(kappa: float, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""(extended state, next-WINDOW-action) pairs along WeakPolicy episodes at this κ."""
    env = make_env(kappa if kappa > 0 else None)
    rng = np.random.default_rng(seed)
    pairs = []
    for ep in range(N_EPISODES):
        env.reset(seed=seed * 50_000 + ep)
        ext_log, act_log = [get_ext(env)], []
        for _t in range(EP_LEN):
            a = weak_action(env, rng)
            env.step(a)
            act_log.append(a)
            ext_log.append(get_ext(env))
        for t in CAPTURE_AT:
            if t + WINDOW <= EP_LEN:
                pairs.append((ext_log[t], np.stack(act_log[t : t + WINDOW])))
    env.close()
    return pairs


def run_traj(env, s0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    env.reset(seed=123)  # fixed scaffold reset (goal irrelevant); then exact state injection
    set_ext(env, s0)
    out = [get_ext(env)]
    for a in actions:
        env.step(a)
        out.append(get_ext(env))
    return np.stack(out)


def pair_exponent(env, s0: np.ndarray, actions: np.ndarray) -> tuple[float, np.ndarray]:
    sA = s0.copy()
    sB = s0.copy()
    sB[4] += DELTA0  # block-angle perturbation (registered)
    tA, tB = run_traj(env, sA, actions), run_traj(env, sB, actions)
    d = np.array([config_dist(a, b) for a, b in zip(tA, tB)])
    # fit ln d on the registered window
    mask = (d > 1e-7) & (d < 0.05)
    mask[0] = d[0] > 0  # always include t=0 if defined
    t_idx = np.nonzero(mask)[0]
    if len(t_idx) < 3:
        return float("nan"), d
    slope = np.polyfit(t_idx.astype(float), np.log(d[t_idx]), 1)[0]
    return float(slope), d


def selfcheck_determinism(env, pairs) -> float:
    r"""G0: δ0=0 twins must coincide exactly."""
    s0, acts = pairs[0]
    tA = run_traj(env, s0, acts)
    tB = run_traj(env, s0, acts)
    return float(np.abs(tA - tB).max())


# --------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    art: dict = {"smoke": SMOKE, "seed": SEED, "kappas": KAPPAS, "delta0": DELTA0,
                 "window": WINDOW, "protocol": "twin-trajectory, block-angle perturbation, "
                 "config-space metric (pos/512 + angle/pi), per-control-step exponents"}
    rng = np.random.default_rng(SEED)
    results = {}
    for kappa in KAPPAS:
        pairs = collect_pairs(kappa, SEED)
        env = make_env(kappa if kappa > 0 else None)
        g0 = selfcheck_determinism(env, pairs)
        assert g0 == 0.0, f"G0 determinism FAIL at kappa={kappa}: max twin gap {g0:.3e}"
        slopes = []
        for s0, acts in pairs:
            lam, _d = pair_exponent(env, s0, acts)
            if np.isfinite(lam):
                slopes.append(lam)
        env.close()
        slopes = np.array(slopes)
        boot = np.array([
            rng.choice(slopes, size=len(slopes), replace=True).mean() for _ in range(N_BOOT)
        ])
        results[str(kappa)] = {
            "n_pairs": int(len(slopes)),
            "lambda_mean": float(slopes.mean()),
            "lambda_median": float(np.median(slopes)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        }
        r = results[str(kappa)]
        print(f"  kappa={kappa}: lambda1 = {r['lambda_mean']:+.4f} /step  "
              f"CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  median {r['lambda_median']:+.4f}  "
              f"(n={r['n_pairs']})")

    means = [results[str(k)]["lambda_mean"] for k in KAPPAS]
    # GATE-A: monotone (Spearman rho over the kappa grid)
    from scipy.stats import spearmanr

    rho = float(spearmanr(KAPPAS, means).statistic) if len(KAPPAS) > 2 else (1.0 if means[-1] > means[0] else -1.0)
    gate_a = rho >= 0.8
    # GATE-B: endpoint CIs disjoint
    lo_k, hi_k = results[str(KAPPAS[0])], results[str(KAPPAS[-1])]
    gate_b = lo_k["ci_hi"] < hi_k["ci_lo"]
    art["results"] = results
    art["gate_a_monotone"] = {"spearman_rho": rho, "pass": bool(gate_a)}
    art["gate_b_resolution"] = {"pass": bool(gate_b)}
    art["verdict"] = "PASS" if (gate_a and gate_b) else ("FALLBACK-2PT" if gate_b else "FAIL")
    art["wall_sec"] = round(time.time() - t0, 1)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(art, indent=1))
    print(f"GATE-A monotone: rho={rho:+.2f} -> {'PASS' if gate_a else 'FAIL'}")
    print(f"GATE-B resolution (endpoint CIs disjoint): {'PASS' if gate_b else 'FAIL'}")
    print(f"VERDICT: {art['verdict']}   ({art['wall_sec']}s)  -> {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
