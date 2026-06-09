r"""Step 86 — the certified horizon as a SAFETY / catastrophe-avoidance bound (direction ①).

Design (seed ①). Flip §5.19's objective from return-optimality — where the certificate's $\sim2\times$ conservatism
*hurt* — to **catastrophe avoidance**, where a sound conservative horizon is exactly the guarantee you want. On the
controlled Lorenz-96 (reuse `step79`: exact $\mathbb{Z}_N$-equivariant action-conditioned WM + the equivariant planner +
the certificate), the agent stabilizes the UNSTABLE uniform fixed point $x_i{=}F$ (keep $\lVert x-F\mathbf1\rVert$
small — do not escape into chaos). It does **receding-horizon control from a state ESTIMATE** that it re-observes
(resets to the true state) only every $\tau$ steps; between observations the estimate is propagated by the WM. Planning
depth stays MODEST (gradients through a chaotic rollout explode with depth — `step79.plan_control`'s own caveat); the
**lever is the re-observation interval $\tau$**, not the planning depth:
- **cert-aware ($\tau=T_1$):** within the certified horizon the estimate stays $\epsilon$-accurate, so the control
  computed from it is $\approx$ the control from truth → the true system stays bounded.
- **horizon-blind ($\tau>T_1$):** the estimate drifts past where the WM is trustworthy (the same $\lambda_1$ the
  certificate reads); control is computed for a *phantom* state and applied to the real one, which — near the unstable
  fixed point — is driven out of the safe region → a **catastrophe (escape)**.

So $T_1$ is the **largest safe re-observation cadence**: $\tau<T_1$ is safe but wasteful (more observations); $\tau>T_1$
is catastrophic. Because escape is irreversible, the certificate's *conservative side* is mandatory — the property that
hurt on return is the guarantee here. Metric = **escape (catastrophe) rate vs $\tau$** over near-$F$ starts × seeds (NOT
return). Honest gate G1 (never loosened): escape-rate at $\tau{=}T_1$ is $\approx0$ (cert-aware safe) AND escape-rate
**rises for $\tau>T_1$** (horizon-binding — the ablation), on $\ge2/3$ seeds. INCONCLUSIVE if cert-aware itself escapes
($T_1$ optimistic) or larger $\tau$ does not escape more (not horizon-binding). float64 CPU (plan is float64).

Reuse, do NOT modify: `step79` (`rk4_controlled`, `make_equivariant_wm`/`train_wm`, `collect_data_mixed`, `certificate`,
`certified_T1_steps`, equivariant `plan_control`).

Run (smoke): STEP86_SMOKE=1 .venv/bin/python experiments/step86_certified_safety.py
Knobs: STEP86_N (24), STEP86_SEEDS (0,1,2), STEP86_EPS (0.3), STEP86_SIGMA (2.0), STEP86_HPLAN (6).
Writes: papers/figures/step86_certified_safety.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step79_certified_control as step79  # noqa: E402

DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP86_SMOKE", "0")))
F_FORCE = step79.F_FORCE


def safety_run(model, mu, sd, x0, tau: int, n_steps: int, sigma: float, H_plan: int = 6,
               u_max: float = 4.0, n_iter: int = 30, lr: float = 0.2, seed: int = 0) -> dict:
    r"""Receding-horizon stabilization toward $x_i{=}F$ from a state ESTIMATE re-observed every ``tau`` steps. Each step:
    plan ``H_plan`` controls from the current estimate (:func:`step79.plan_control`, modest depth), apply the FIRST to the
    TRUE dynamics (:func:`step79.rk4_controlled`); then if $t\bmod\tau=0$ reset the estimate to truth (re-observe), else
    propagate the estimate one step by the WM under the applied control. A **catastrophe** is escape from the safe region,
    $\max_t\lVert x_t-F\mathbf1\rVert>5\sigma\sqrt N$. Returns ``escaped``, ``max_dev``, ``steps``, ``reobs`` (count)."""
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    N = x0.shape[-1]
    target = F_FORCE * torch.ones(N, dtype=DTYPE)
    thr = 5.0 * sigma * (N ** 0.5)
    x_true = x0.to(DTYPE).clone()
    x_est = x_true.clone()                                              # observed at t=0
    max_dev = float((x_true - target).norm())
    escaped = False
    reobs = 1
    for t in range(1, n_steps + 1):
        u = step79.plan_control(model, x_est, mu, sd, H_plan, u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)[0]
        x_true = step79.rk4_controlled(x_true, u).detach()             # apply first control to the TRUE system
        dev = float((x_true - target).norm())
        max_dev = max(max_dev, dev)
        if dev > thr:
            escaped = True
            break
        if t % tau == 0:
            x_est = x_true.clone()                                     # RE-OBSERVE: reset estimate to truth
            reobs += 1
        else:
            with torch.no_grad():                                      # propagate estimate by the WM (drift)
                x_est = (model((x_est - mu) / sd, u) * sd + mu)
    return {"escaped": bool(escaped), "max_dev": max_dev, "steps": int(t), "reobs": int(reobs), "tau": int(tau)}


def escape_rate(model, mu, sd, N: int, tau: int, n_starts: int, sigma: float, n_steps: int,
                seed: int, H_plan: int = 6, u_max: float = 4.0, n_iter: int = 30, lr: float = 0.2) -> float:
    r"""Fraction of near-$F$ starts ($x_0=F\mathbf1+\sigma\xi$) that escape under re-observation interval ``tau``."""
    g = torch.Generator().manual_seed(seed + 4321)
    esc = 0
    for _ in range(n_starts):
        x0 = F_FORCE + sigma * torch.randn(N, generator=g, dtype=DTYPE)
        esc += int(safety_run(model, mu, sd, x0, tau, n_steps, sigma, H_plan=H_plan,
                              u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)["escaped"])
    return esc / max(1, n_starts)


def run(seeds, N: int, eps: float, sigma: float, H_plan: int) -> dict:
    r"""Per seed: train the equivariant controlled WM, read $T_1$ (map steps), and sweep the re-observation interval
    $\tau\in\{T_1/2, T_1, 2T_1, 4T_1\}$, scoring escape (catastrophe) rate. Gate G1: escape@$T_1\le$ tol (cert-aware
    safe) AND escape rises for $\tau>T_1$ (horizon-binding), on $\ge2/3$ seeds."""
    n_traj = 1500 if SMOKE else 6000
    K = 5
    epochs = 8 if SMOKE else 60
    n_starts = 6 if SMOKE else 16
    cert_kw = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
                   n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)
    n_iter = 20 if SMOKE else 30
    per_seed = {}
    for s in seeds:
        print(f"[step86] seed {s}: training Z_N-equivariant controlled WM at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        data = step79.collect_data_mixed(N, n_traj, K, s, near_frac=0.5, sigma=sigma)
        model, mu, sd, relmse = step79.train_wm("conv", N, data, s, epochs=epochs, K=K)
        cert = step79.certificate(model, mu, sd, N, eps=eps, seed=s, **cert_kw)
        T1 = step79.certified_T1_steps(cert)
        taus = [max(1, T1 // 2), T1, 2 * T1, 4 * T1]
        labels = ["T1/2", "T1", "2T1", "4T1"]
        n_steps = int(min(round(4.5 * T1), 120 if SMOKE else 360))
        rates = {}
        for lab, tau in zip(labels, taus):
            rates[lab] = escape_rate(model, mu, sd, N, tau, n_starts, sigma, n_steps, s, H_plan=H_plan, n_iter=n_iter)
        cert_safe = rates["T1"] <= (0.2 if SMOKE else 0.15)
        binding = (rates["2T1"] > rates["T1"] + 0.15) and (rates["4T1"] >= rates["2T1"] - 1e-9)
        per_seed[s] = {"T1": T1, "lambda1": cert["lambda1"], "relmse": relmse, "n_steps": n_steps,
                       "taus": dict(zip(labels, taus)), "escape_rates": rates,
                       "cert_safe": bool(cert_safe), "horizon_binding": bool(binding)}
        print(f"[step86] seed {s}: T1={T1} n_steps={n_steps} | escape {[f'{l}:{rates[l]:.2f}' for l in labels]} | "
              f"cert_safe={cert_safe} binding={binding}", file=sys.stderr)

    safe = [per_seed[s]["cert_safe"] for s in seeds]
    bind = [per_seed[s]["horizon_binding"] for s in seeds]
    need = int(np.ceil(2 / 3 * len(seeds)))
    g1_pass = bool(sum(safe) >= need and sum(bind) >= need)
    verdict = {"G1_pass": g1_pass, "n_cert_safe": int(sum(safe)), "n_horizon_binding": int(sum(bind)),
               "n_seeds": len(seeds), "eps": eps, "N": N, "sigma": sigma, "H_plan": H_plan, "smoke": SMOKE}
    print(f"[step86] {'G1 PASS' if g1_pass else 'G1 INCONCLUSIVE'}: cert-aware (tau=T1) safe on {sum(safe)}/{len(seeds)}, "
          f"horizon-binding (escape rises past T1) on {sum(bind)}/{len(seeds)}. "
          f"{'Conservatism is a safety feature; T1 is the largest safe cadence.' if g1_pass else 'Not cleanly horizon-binding — report honestly.'}",
          file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


if __name__ == "__main__":
    torch.manual_seed(0)
    N = int(os.environ.get("STEP86_N", "10" if SMOKE else "24"))
    eps = float(os.environ.get("STEP86_EPS", "0.3"))
    sigma = float(os.environ.get("STEP86_SIGMA", "2.0"))
    H_plan = int(os.environ.get("STEP86_HPLAN", "6"))
    seeds = [int(x) for x in os.environ.get("STEP86_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    print(f"[step86] safety bound: N={N} eps={eps} sigma={sigma} H_plan={H_plan} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
    res = run(seeds, N, eps, sigma, H_plan)
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"step86_certified_safety{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0 if res["verdict"]["G1_pass"] else 1)
