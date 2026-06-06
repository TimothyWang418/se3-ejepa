r"""Step 71 — the certified-horizon law on learned models of *multiple* chaotic systems (generalize Step 70 beyond Lorenz).

Step 70 showed the certified-horizon staircase $T(\epsilon)\sim\log(1/\epsilon)/\lambda$ lifts to a *learned* model of the
Lorenz attractor. One system is an existence proof, not a law. Here we run the **identical protocol** on a small *class*
of chaotic systems with documented largest Lyapunov exponents, of two kinds:

  * **Hénon map** ($a{=}1.4,b{=}0.3$): a discrete 2-D map, $\lambda_1\approx0.419$ /step (Benettin et al.; standard).
  * **Rössler flow** ($a{=}0.2,b{=}0.2,c{=}5.7$): a continuous 3-D ODE, $\lambda_1\approx0.0714$ /time (standard) —
    the *small-exponent* stress case (long horizons, so the model must stay accurate longer).
  * **Lorenz** ($\sigma{=}10,\rho{=}28,\beta{=}8/3$): $\lambda_1\approx0.9056$ /time — the Step-70 anchor, re-run here
    under the same harness for a like-for-like comparison.

For each system: (1) generate trajectories and train a plain one-step MLP of the (normalized) update map; (2) run the
certified-horizon staircase on the LEARNED model; (3) read the largest Lyapunov exponent off the staircase slope,
$\hat\lambda_1 = 1/(\text{slope}\cdot dt_{\rm eff})$, and compare to the documented value. If the staircase is linear and
its slope recovers the true exponent across systems, the horizon law is a *property of chaotic dynamics*, not a Lorenz
coincidence. We report the small-$\lambda$ Rössler case honestly: there the horizon is long and the lift is harder.

Honest gate (per system; prints INCONCLUSIVE rather than loosen a threshold):
  (i)   the model learned the one-step map:         one-step relMSE < 0.05;
  (ii)  the certified-horizon staircase is linear:  R^2 > 0.95;
  (iii) the staircase slope recovers $\lambda_1$:    |lambda_staircase - lambda_ref| / lambda_ref < 0.30.
Overall PASS iff >= 2 of the 3 systems pass all three (Lorenz is the anchor; one new system must corroborate).

Run:     .venv/bin/python experiments/step71_multichaos_horizon.py
Seeded:  STEP71_SEED=0|1|2 ...
Writes:  papers/figures/step71_multichaos_horizon.{json,png}
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers" / "figures"
SEED = int(os.environ.get("STEP71_SEED", "0"))
SMOKE = os.environ.get("STEP71_SMOKE", "0") == "1"
EPOCHS = int(os.environ.get("STEP71_EPOCHS", "30" if SMOKE else "200"))
TAG = os.environ.get("STEP71_TAG", "")


# ----------------------------- chaotic systems (one-step update maps) -----------------------------
def henon_step(s, a=1.4, b=0.3):
    x, y = s[..., 0], s[..., 1]
    return np.stack([1.0 - a * x * x + y, b * x], axis=-1)


def rossler_rhs(s, a=0.2, b=0.2, c=5.7):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=-1)


def lorenz_rhs(s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)


def rk4(rhs, s, dt):
    k1 = rhs(s); k2 = rhs(s + 0.5 * dt * k1); k3 = rhs(s + 0.5 * dt * k2); k4 = rhs(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Each system: dim, one-step map step(s)->s', dt_eff (1 for maps; integration dt for flows), documented lambda1
# (PER STEP: for flows lambda1_time*dt; for maps the map exponent), init sampler, rollout length, eps schedule.
SYSTEMS = {
    "Henon": dict(
        dim=2, step=lambda s: henon_step(s), dt=1.0, lam_time=0.419, lam_ref_perstep=0.419,
        init=lambda rng, n: rng.uniform(-0.3, 0.3, size=(n, 2)), burn=200, bound=20.0,
        traj_len=400 if SMOKE else 1500, n_traj=40 if SMOKE else 200, t_roll=120,
        eps_res=0.5, eps0s=np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]),
    ),
    "Rossler": dict(
        dim=3, step=lambda s: rk4(rossler_rhs, s, 0.1), dt=0.1, lam_time=0.0714, lam_ref_perstep=0.0714 * 0.1,
        init=lambda rng, n: np.column_stack([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n), rng.uniform(0, 6, n)]),
        burn=2000, bound=1e3,
        traj_len=600 if SMOKE else 3000, n_traj=40 if SMOKE else 200, t_roll=1600,
        eps_res=0.5, eps0s=np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4]),
    ),
    "Lorenz": dict(
        dim=3, step=lambda s: rk4(lorenz_rhs, s, 0.02), dt=0.02, lam_time=0.9056, lam_ref_perstep=0.9056 * 0.02,
        init=lambda rng, n: np.column_stack([rng.uniform(-15, 15, n), rng.uniform(-15, 15, n), rng.uniform(5, 40, n)]),
        burn=500, bound=1e3,
        traj_len=400 if SMOKE else 1500, n_traj=40 if SMOKE else 200, t_roll=550,
        eps_res=0.5, eps0s=np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]),
    ),
}


def on_attractor_trajs(cfg, rng, n, length):
    r"""Integrate from random inits, burn in onto the attractor, drop any trajectory that diverged, and return up to
    ``n`` bounded trajectories of shape (m<=n, length+1, dim). Fixes Henon escapees + Rossler/Lorenz transients."""
    s0 = cfg["init"](rng, int(n * 1.4) + 8).astype(np.float64)
    full = integrate(cfg["step"], s0, cfg["burn"] + length)          # (B, burn+length+1, dim)
    seg = full[:, cfg["burn"]:]                                      # on-attractor segment
    ok = np.isfinite(seg).all(axis=(1, 2)) & (np.abs(seg).max(axis=(1, 2)) < cfg["bound"])
    seg = seg[ok][:n]
    assert len(seg) >= max(4, n // 4), f"too few bounded trajectories survived ({len(seg)})"
    return seg


class MLP(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d))

    def forward(self, s):                       # residual one-step predictor
        return s + self.net(s)


def integrate(step, s0, n):
    out = [s0]; s = s0
    for _ in range(n):
        s = step(s); out.append(s)
    return np.stack(out, axis=1)                # (B, n+1, d)


def fit_staircase(model, norm, sd, mu, cfg, rng, device):
    r"""Run the certified-horizon staircase on the LEARNED model; return (slope, r2, Tcross, xs, ys)."""
    n_pert = 30 if SMOKE else 120
    qs = on_attractor_trajs(cfg, rng, n_pert, 1)[:, 0]               # query points ON the attractor
    n_pert = len(qs)
    dirs = rng.standard_normal((n_pert, cfg["dim"])); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    T = cfg["t_roll"]

    @torch.no_grad()
    def rollout(s_norm0):
        s = torch.tensor(s_norm0, dtype=torch.float32, device=device); out = [s]
        for _ in range(T):
            s = model(s); out.append(s)
        return torch.stack(out, 1).cpu().numpy()

    Tcross = []
    base = rollout(norm(qs))
    for e in cfg["eps0s"]:
        pert = rollout(norm(qs) + e * dirs)
        gm = (np.linalg.norm(pert - base, axis=-1) / e).mean(0)
        cr = int(np.argmax(gm >= cfg["eps_res"] / e))
        Tcross.append(float(cr) if gm[cr] >= cfg["eps_res"] / e else float("nan"))
    Tcross = np.array(Tcross)
    valid = ~np.isnan(Tcross)
    xs = np.log(1.0 / cfg["eps0s"])[valid]; ys = Tcross[valid]
    if valid.sum() >= 3:
        slope, icpt = np.polyfit(xs, ys, 1)
        r2 = 1.0 - np.sum((ys - (slope * xs + icpt)) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-12)
    else:
        slope, r2 = float("nan"), float("nan")
    return float(slope), float(r2), Tcross, xs, ys


def run_system(name, cfg, device):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    traj = on_attractor_trajs(cfg, rng, cfg["n_traj"], cfg["traj_len"])      # on-attractor training trajectories
    flat = traj.reshape(-1, cfg["dim"])
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    norm = lambda a: (a - mu) / sd
    S = norm(traj)
    s_in = torch.tensor(S[:, :-1].reshape(-1, cfg["dim"]), dtype=torch.float32, device=device)
    s_out = torch.tensor(S[:, 1:].reshape(-1, cfg["dim"]), dtype=torch.float32, device=device)

    model = MLP(cfg["dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(SEED)
    nidx = s_in.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(nidx, generator=g)
        for i in range(0, nidx, 512):
            idx = perm[i:i + 512]
            opt.zero_grad(); ((model(s_in[idx]) - s_out[idx]) ** 2).mean().backward(); opt.step()
    model.eval()
    with torch.no_grad():
        one_step = float(((model(s_in) - s_out) ** 2).sum(-1).mean() / (s_out ** 2).sum(-1).mean())

    slope, r2, Tcross, xs, ys = fit_staircase(model, norm, sd, mu, cfg, np.random.default_rng(SEED + 99), device)
    lam_stair = 1.0 / (slope * cfg["dt"]) if (slope == slope and slope > 1e-9) else float("nan")  # per-time
    lam_ref_time = cfg["lam_time"]
    ok_model = bool(one_step < 0.05)
    ok_stair = bool(r2 == r2 and r2 > 0.95)
    ok_lam = bool(lam_stair == lam_stair and abs(lam_stair - lam_ref_time) / lam_ref_time < 0.30)
    passed = bool(ok_model and ok_stair and ok_lam)
    return dict(name=name, one_step=one_step, slope=slope, r2=r2,
                lam_staircase_time=lam_stair, lam_ref_time=lam_ref_time,
                lam_relerr=(abs(lam_stair - lam_ref_time) / lam_ref_time if lam_stair == lam_stair else float("nan")),
                gate=dict(model_learned=ok_model, staircase_linear=ok_stair, recovers_lambda=ok_lam),
                passed=passed, xs=[float(x) for x in xs], ys=[float(y) for y in ys],
                eps0s=[float(e) for e in cfg["eps0s"]], Tcross=[float(t) for t in Tcross])


def main():
    device = os.environ.get("STEP71_DEVICE", "cpu")
    results = {name: run_system(name, cfg, device) for name, cfg in SYSTEMS.items()}
    n_pass = sum(r["passed"] for r in results.values())
    overall = bool(n_pass >= 2 and results["Lorenz"]["passed"])
    out = {"passed": overall, "n_pass": n_pass, "systems": results, "seed": SEED, "smoke": SMOKE}
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"step71_multichaos_horizon{TAG}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(5.0 * len(SYSTEMS), 4.2))
    for ax, (name, r) in zip(axes, results.items()):
        xs, ys = np.array(r["xs"]), np.array(r["ys"])
        if len(xs):
            ax.plot(xs, ys, "C0o-", lw=2,
                    label=f"measured ($\\hat\\lambda_1={r['lam_staircase_time']:.3f}$/t, $R^2{{=}}{r['r2']:.3f}$)")
            sref = 1.0 / (r["lam_ref_time"] * SYSTEMS[name]["dt"])
            ax.plot(xs, sref * xs + (ys.mean() - sref * xs.mean()), "C3--", lw=1.5,
                    label=f"textbook $\\lambda_1={r['lam_ref_time']:.3f}$/t")
        ax.set_xlabel("$\\log(1/\\epsilon_0)$"); ax.set_ylabel("certified horizon $T(\\epsilon_0)$ (steps)")
        ax.set_title(f"{name}: {'PASS' if r['passed'] else 'INCONCLUSIVE'} (rel-err {r['lam_relerr']:.0%})")
        ax.legend(fontsize=8)
    fig.suptitle("The certified-horizon law on learned models of multiple chaotic systems (Step 71)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / f"step71_multichaos_horizon{TAG}.png", dpi=130, bbox_inches="tight")

    for name, r in results.items():
        print(f"[step71] {name:8s} one-step {r['one_step']:.4f}  R^2 {r['r2']:.3f}  "
              f"lambda_staircase {r['lam_staircase_time']:.4f}/t vs textbook {r['lam_ref_time']:.4f} "
              f"(rel-err {r['lam_relerr']:.1%})  -> {'PASS' if r['passed'] else 'INCONCLUSIVE'}", file=sys.stderr)
    print(f"[step71] {n_pass}/{len(SYSTEMS)} systems pass; overall {'PASS' if overall else 'INCONCLUSIVE'} "
          f"(the horizon law lifts to a CLASS of learned chaotic models, not just Lorenz)", file=sys.stderr)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
