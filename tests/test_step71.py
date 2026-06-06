r"""Guard for Step 71 / the certified-horizon law across a CLASS of chaotic systems. Fast (no model training): it
validates the *staircase protocol* on the GROUND-TRUTH maps/flows, so the multi-system claim does not depend on any one
trained network. For the Hénon map and the Rössler flow (the two systems Step 71 adds beyond Lorenz), the certified
horizon $T(\epsilon)$ is linear in $\log(1/\epsilon)$ and its slope recovers the documented top Lyapunov exponent.
(Lorenz's true-integrator staircase is already guarded by test_step70.)

Run:  .venv/bin/python tests/test_step71.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import step71_multichaos_horizon as s71  # noqa: E402


def true_staircase(name, n_pert=80, seed=0):
    r"""Run the certified-horizon staircase on the TRUE system (no learned model) in normalized coordinates;
    return (lambda_staircase_per_time, r2)."""
    cfg = s71.SYSTEMS[name]
    rng = np.random.default_rng(seed)
    ref = s71.on_attractor_trajs(cfg, rng, 64, 400).reshape(-1, cfg["dim"])
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    norm = lambda a: (a - mu) / sd
    qs = s71.on_attractor_trajs(cfg, rng, n_pert, 1)[:, 0]
    m = len(qs)
    dirs = rng.standard_normal((m, cfg["dim"])); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    T = cfg["t_roll"]

    base = norm(s71.integrate(cfg["step"], qs, T))                       # (m, T+1, d), normalized
    Tcross = []
    for e in cfg["eps0s"]:
        pert = norm(s71.integrate(cfg["step"], (norm(qs) + e * dirs) * sd + mu, T))
        gm = (np.linalg.norm(pert - base, axis=-1) / e).mean(0)
        cr = int(np.argmax(gm >= cfg["eps_res"] / e))
        Tcross.append(float(cr) if gm[cr] >= cfg["eps_res"] / e else float("nan"))
    Tcross = np.array(Tcross); valid = ~np.isnan(Tcross)
    xs = np.log(1.0 / cfg["eps0s"])[valid]; ys = Tcross[valid]
    assert valid.sum() >= 3, f"{name}: too few resolutions crossed ({valid.sum()})"
    slope, icpt = np.polyfit(xs, ys, 1)
    r2 = 1.0 - np.sum((ys - (slope * xs + icpt)) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-12)
    lam = 1.0 / (slope * cfg["dt"])
    return lam, r2


def test_staircase_recovers_lyapunov_on_true_systems() -> None:
    r"""On the true Hénon map and Rössler flow, the horizon staircase is linear and its slope recovers $\lambda_1$."""
    for name, tol in [("Henon", 0.30), ("Rossler", 0.30)]:
        lam, r2 = true_staircase(name)
        ref = s71.SYSTEMS[name]["lam_time"]
        relerr = abs(lam - ref) / ref
        assert r2 > 0.95, f"{name}: staircase not linear on the true system (R^2={r2:.3f})"
        assert relerr < tol, f"{name}: slope does not recover lambda_1 ({lam:.4f}/t vs {ref}/t, rel-err {relerr:.1%})"
        print(f"PASS: {name} true-system staircase linear (R^2 {r2:.3f}); slope recovers lambda_1 "
              f"{lam:.4f}/t vs textbook {ref}/t (rel-err {relerr:.1%}).")
    print("=> the certified-horizon protocol recovers the top exponent across a CLASS of chaotic systems "
          "(map + flow), independent of any learned model.")


if __name__ == "__main__":
    test_staircase_recovers_lyapunov_on_true_systems()
