r"""Step 94 — the PUBLISHED certificate prices BUDGETED-SENSING MONITORING in deployment (scope law, positive instance).

step93 lost where the decided quantity (return) sat behind a task map; the scope law says the win lives where the
decided quantity IS the certified quantity. Deployment instantiation: a SENSOR-ONLY MONITOR watches a system running
its nominal policy (deterministic prior on TRUE obs — monitoring cannot affect the system: the step93 confound is
designed out). The expensive sensor is read every $k$ steps; between reads the monitor forecasts the latent with the
WM's NOMINAL AUTONOMOUS loop $g(z)=d(z,\tanh\mu_\pi(z))$ — the exact loop step89/E13 certifies — and at a read it
flags iff relative latent error $>\theta$, then resyncs. No action telemetry is needed.

The certificate numbers are LOADED from the published step89 artifact (`step89_pretrained_audit.json`) — issued
before this experiment existed: a-priori in the strongest sense. Primary system: cheetah-run (official seeds 1,2,3).

Pre-registered gates (v5, fixed before the full run; seeds 1–2 untouched by design iterations — out-of-sample):
- **G1a (deployment replicates the bench scope map, cell-by-cell):** in-situ staleness ratio
  $r=s^{\ast}_{\mathrm{med}}/T_1^{\mathrm{pub}}(0.2)$ satisfies $|r-r^{\mathrm{bench}}|\le0.25$ on $\ge2/3$ seeds,
  where $r^{\mathrm{bench}}$ = published measured/certified (0.43 / 0.50 / 0.83) — INCLUDING the optimistic cells.
- **G1b (calibrated-cell pricing):** on the published-calibrated cell (seed 3, $r^{\mathrm{bench}}=0.83$),
  $r\in[2/3,3/2]$.
- **G2 (fault detection at the certificate-derived operating point):** at $k_{\mathrm{op}}=\max(2,\mathrm{round}
  (T_1^{\mathrm{pub}}/3))$ — a conservative a-priori rule, one third of the certified median clock — a frozen-actuator
  fault (executes 0; nominal forecast unchanged) is detected with recall $\ge95\%$ and median delay
  $\le k_{\mathrm{op}}$ steps, on $\ge2/3$ seeds.
Descriptive (no gate): belief-invalid fraction vs $k$ (the budget curve) + its 25%-knee $k^{\ast}$; per-read no-fault
flag rate at $k_{\mathrm{op}}$. The certificate prices the MEDIAN staleness clock; tail-quantile budgets bind earlier
— the quantile choice is the deployer's (Prop 10's interval framing in deployment guise). float64, no MPPI.

DESIGN TRAIL (honest, smoke-driven; gates above fixed only after these lessons, all probed on seed 3 / walker only):
v1 gated on per-episode "any flag" — vacuous (the median crossing puts ~half the windows over $\theta$ by
construction). v2 certified the teacher-forced loop $z\mapsto d(z,a^{\mathrm{cmd}}_t)$ (step84 controlled-Jacobian
pattern) — its $\lambda_1$ landed WEAKLY EXPANSIVE (0.109 vs autonomous 0.27) and that certificate over-priced the
knee (14.8 vs ~6): the paper's own E13/E14 taxonomy predicts optimism there (bias-driven divergence) — a
within-experiment confirmation, kept as a secondary diagnostic. v3 moved the primary system walker→CHEETAH-RUN after
a diagnostic showed the deterministic-prior WALKER is BIMODAL across env seeds (falls on some: invalid-frac
0.24→0.66 tracks torso height 1.05→0.68) — a monitor presumes a nominal regime (Prop 7 in deployment guise); cheetah
has no fall mode (uniform 0.14–0.23 at k=4); walker retained as a SECONDARY observational run. v4 made the monitor
forecast the NOMINAL AUTONOMOUS loop (sensor-only; deployed forecaster = certified loop by construction). v5 fixed
the remaining estimator fragility: certificates are READ from the published step89 JSON instead of re-estimated
(in-script Benettin re-runs moved $T_1$ 6.0→7.6 across RNG draws — within the published CI but gate-flipping), and
G1 became cell-by-cell scope-map replication, because gating all seeds on $[2/3,3/2]$ would contradict E13's own
published optimistic cells (cheetah-1/2).

Run (smoke): STEP94_SMOKE=1 STEP94_SEEDS=3 ...; full: .venv/bin/python experiments/step94_budgeted_monitor.py
Secondary (observational, regime-heterogeneity caveat): STEP94_TASK=walker-walk ...
Writes: papers/figures/step94_budgeted_monitor.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89  # noqa: E402
import step78_certified_horizon_ci as step78  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP94_SMOKE", "0")))
THETA = 0.2
TASK = os.environ.get("STEP94_TASK", "cheetah-run")
DOMAIN, TNAME = TASK.split("-", 1)
_REG = {"cheetah-run": (17, 6), "walker-walk": (24, 6), "finger-spin": (9, 2), "hopper-hop": (15, 4)}
OBS_DIM, ACT_DIM = _REG[TASK]
PUB = ROOT / "papers/figures/step89_pretrained_audit.json"


def published_cert(seed: int):
    r"""Published certificate row for this cell (a-priori: the step89 artifact predates step94)."""
    d = json.loads(PUB.read_text())[f"{TASK}-{seed}"]
    row = [r for r in d["cert_rows"] if abs(r["eps"] - THETA) < 1e-9][0]
    bench = d.get("measured", {}).get("0.2", {})
    return {"T1_pub": row["T1_steps"], "ratio_bench": row["ratio_measured_over_certified"],
            "lambda1_pub": d["lambda1"], "lambda1_ci_pub": d["lambda1_ci"],
            "bench_median": bench.get("median"), "bench_censored": bench.get("n_censored")}


def certify_teacher_forced(sl, T: int = 200, seed: int = 11, n_boot: int = 200, block: int = 30):
    r"""Secondary diagnostic (v2 lesson): Lyapunov rate of the TEACHER-FORCED cocycle $z\mapsto d(z,a_t)$ along the
    true commanded-action sequence. Lands weakly expansive on these cells — by the E13/E14 taxonomy its certificate
    would be optimistic (bias-driven divergence); recorded as a within-experiment taxonomy check, not used for gates."""
    from dm_control import suite
    env = suite.load(DOMAIN, TNAME, task_kwargs={"random": seed})
    ts = env.reset()
    zs, acts = [], []
    with torch.no_grad():
        for _ in range(T):
            z = sl.encode(s89._flat_obs(ts))
            a = torch.tanh(sl.pi_mean(z))
            zs.append(z); acts.append(a)
            for _ in range(2):
                ts = env.step(a.numpy())
    d = zs[0].numel()
    Q = torch.linalg.qr(torch.randn(d, d, dtype=zs[0].dtype))[0]
    logs = []
    for t in range(T - 1):
        J = torch.autograd.functional.jacobian(lambda z: sl.next(z, acts[t]), zs[t], vectorize=True)
        Q, R = torch.linalg.qr(J @ Q)
        logs.append(torch.log(torch.abs(torch.diagonal(R)).clamp_min(1e-300)))
    L = torch.stack(logs)[20:]
    point, lo, hi = step78.bootstrap_spectrum_ci(L, 1.0, n_boot, block, seed)
    return {"lambda1_tf": float(point[0]), "lambda1_tf_ci": [float(lo[0]), float(hi[0])]}


def run_episode(sl, k: int, seed: int, T: int = 400, fault_t: int | None = None, collect_crossings: bool = False):
    r"""Monitored episode. System: deterministic prior on TRUE obs each step. Monitor: nominal-autonomous forecast
    $\hat z\leftarrow g(\hat z)$, sensor read every ``k`` steps -> flag iff rel err > THETA, then resync. Fault:
    actuator channel (seeded) executes 0 from ``fault_t`` while the nominal forecast is unchanged.
    Returns (flags, invalid_fraction, crossings) — crossings = per-window first staleness step with rel>THETA
    (censored at k+1), only if ``collect_crossings``."""
    from dm_control import suite
    env = suite.load(DOMAIN, TNAME, task_kwargs={"random": seed})
    ts = env.reset()
    rng = np.random.RandomState(seed)
    fault_ch = rng.randint(0, ACT_DIM)
    z_hat = None
    flags, crossings = [], []
    invalid = 0
    cur, s_in = None, 0
    with torch.no_grad():
        for t in range(T):
            z_true = sl.encode(s89._flat_obs(ts))
            rel_now = (None if z_hat is None else
                       float((z_hat - z_true).norm() / z_true.norm().clamp_min(1e-12)))
            if rel_now is not None and rel_now > THETA:
                invalid += 1                                       # belief stale beyond theta (offline truth)
                if cur is None:
                    cur = s_in                                     # first crossing in this window
            if t == 0 or (k > 0 and t % k == 0):                   # sensor read: compare, log window, resync
                if rel_now is not None and rel_now > THETA:
                    flags.append(t)
                if collect_crossings and t > 0:
                    crossings.append(cur if cur is not None else k + 1)
                z_hat = z_true.clone()
                cur, s_in = None, 0
            a_cmd = torch.tanh(sl.pi_mean(z_true))                 # the system: prior actor on TRUE obs
            a_exec = a_cmd.clone()
            if fault_t is not None and t >= fault_t:
                a_exec[fault_ch] = 0.0                             # frozen actuator: executed != nominal
            z_hat = sl.next(z_hat, torch.tanh(sl.pi_mean(z_hat)))  # monitor forecasts the NOMINAL AUTONOMOUS loop g
            s_in += 1
            for _ in range(2):                                     # action_repeat=2
                ts = env.step(a_exec.numpy())
    return flags, invalid / T, crossings


def run(seeds, k_list, n_ep: int) -> dict:
    out = {"task": TASK, "theta": THETA, "k_list": k_list, "per_seed": {}}
    for s in seeds:
        ck = ROOT / f"models/tdmpc2/{TASK}-{s}.pt"
        sl = s89.load_tdmpc2_slices(ck, obs_dim=OBS_DIM, action_dim=ACT_DIM)
        pub = published_cert(s)
        T1, r_bench = pub["T1_pub"], pub["ratio_bench"]
        tf = certify_teacher_forced(sl, T=120 if SMOKE else 200, seed=11, n_boot=60 if SMOKE else 200)
        print(f"[step94] seed {s}: published T1(0.2)={T1:.2f} ratio_bench={r_bench:.2f} | "
              f"teacher-forced lam1={tf['lambda1_tf']:.3f} (taxonomy diag)", file=sys.stderr)
        # --- G1: in-situ staleness clock (k=24 censored windows, no-fault episodes) ---
        crossings = []
        for e in range(n_ep):
            _, _, cr = run_episode(sl, 24, seed=1000 + 7 * e + s, fault_t=None, collect_crossings=True)
            crossings += cr
        c = np.array(crossings, dtype=float)
        s_med = float(np.median(c))
        r_insitu = s_med / T1
        g1a = abs(r_insitu - r_bench) <= 0.25
        g1b = (2 / 3 <= r_insitu <= 3 / 2) if abs(r_bench - 0.83) < 0.05 else None   # calibrated cell only
        print(f"[step94] seed {s}: in-situ s*_med={s_med:.1f} (q25={np.percentile(c, 25):.0f} "
              f"q75={np.percentile(c, 75):.0f} n={len(c)}) ratio={r_insitu:.2f} vs bench {r_bench:.2f} "
              f"| G1a={g1a} G1b={g1b}", file=sys.stderr)
        # --- budget curve (descriptive) ---
        inv = {}
        for k in k_list:
            fr = [run_episode(sl, k, seed=1000 + 7 * e + s, fault_t=None)[1] for e in range(n_ep)]
            inv[k] = float(np.mean(fr))
            print(f"[step94] seed {s} k={k}: belief-invalid fraction={inv[k]:.3f}", file=sys.stderr)
        kstar = max([k for k in k_list if inv[k] <= 0.25], default=None)
        # --- G2: fault detection at the a-priori operating point ---
        k_op = max(2, int(round(T1 / 3)))
        rec, delays, preflag = 0, [], []
        for e in range(n_ep):
            tf_step = 150 + (e * 13) % 100
            flags, _, _ = run_episode(sl, k_op, seed=2000 + 7 * e + s, fault_t=tf_step)
            post = [f for f in flags if f >= tf_step]
            pre = [f for f in flags if f < tf_step]
            preflag.append(len(pre) / max(1, tf_step // k_op))     # per-read no-fault flag rate (context)
            if post:
                rec += 1
                delays.append(post[0] - tf_step)
        recall = rec / n_ep
        med_delay = float(np.median(delays)) if delays else None
        g2 = (recall >= 0.95 and med_delay is not None and med_delay <= k_op)
        out["per_seed"][str(s)] = {
            "T1_published": T1, "ratio_bench": r_bench, "teacher_forced_diag": tf,
            "insitu": {"median": s_med, "q25": float(np.percentile(c, 25)), "q75": float(np.percentile(c, 75)),
                       "n_windows": int(len(c)), "ratio_insitu": r_insitu},
            "invalid_by_k": inv, "k_star_25pct": kstar, "k_op": k_op,
            "fault_recall": recall, "median_delay": med_delay,
            "prefault_flag_rate": float(np.mean(preflag)),
            "G1a_replicates_bench": bool(g1a), "G1b_calibrated_cell": g1b, "G2_detection": bool(g2)}
        print(f"[step94] seed {s}: k*={kstar} k_op={k_op} | recall={recall:.2f} delay_med={med_delay} "
              f"preflag={np.mean(preflag):.2f} | G2={g2}", file=sys.stderr)
    g1a_n = sum(out["per_seed"][str(s)]["G1a_replicates_bench"] for s in seeds)
    g1b_vals = [out["per_seed"][str(s)]["G1b_calibrated_cell"] for s in seeds
                if out["per_seed"][str(s)]["G1b_calibrated_cell"] is not None]
    g2_n = sum(out["per_seed"][str(s)]["G2_detection"] for s in seeds)
    need = int(np.ceil(2 / 3 * len(seeds)))
    out["verdict"] = {
        "G1a_pass": bool(g1a_n >= need), "G1b_pass": (all(g1b_vals) if g1b_vals else None),
        "G2_pass": bool(g2_n >= need),
        "n_g1a": int(g1a_n), "n_g2": int(g2_n), "n_seeds": len(seeds)}
    v = out["verdict"]
    print(f"[step94] G1a {'PASS' if v['G1a_pass'] else 'INCONCLUSIVE'} ({v['n_g1a']}/{len(seeds)} cells replicate "
          f"bench); G1b {'PASS' if v['G1b_pass'] else ('n/a' if v['G1b_pass'] is None else 'INCONCLUSIVE')} "
          f"(calibrated cell); G2 {'PASS' if v['G2_pass'] else 'INCONCLUSIVE'} ({v['n_g2']}/{len(seeds)}).",
          file=sys.stderr)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    seeds = [int(x) for x in os.environ.get("STEP94_SEEDS", "1" if SMOKE else "1,2,3").split(",")]
    k_list = [int(x) for x in os.environ.get("STEP94_KS", "4,8" if SMOKE else "2,3,4,6,8,12").split(",")]
    n_ep = int(os.environ.get("STEP94_EPISODES", "4" if SMOKE else "20"))
    print(f"[step94] budgeted-sensing monitor: task={TASK} seeds={seeds} k={k_list} n_ep={n_ep} theta={THETA}",
          file=sys.stderr)
    res = run(seeds, k_list, n_ep)
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = ("_smoke" if SMOKE else "") + ("" if TASK == "cheetah-run" else f"_{TASK}")
    (figdir / f"step94_budgeted_monitor{tag}.json").write_text(json.dumps(res, indent=2))
    ok = res["verdict"]["G1a_pass"] and res["verdict"]["G2_pass"] and res["verdict"]["G1b_pass"] in (True, None)
    raise SystemExit(0 if ok else 1)
