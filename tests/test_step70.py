r"""Guard for Step 70 / the certified-horizon law on REAL chaotic dynamics (Lorenz). Two checks, both fast (no model
training needed — they validate the *protocol's* math on the ground-truth integrator):

  1. Lorenz carries the $\mathbb Z_2$ symmetry $g\cdot(x,y,z)=(-x,-y,z)$: the vector field is equivariant,
     $f(g\cdot s)=g\cdot f(s)$ (project rule: every symmetry we rely on gets an equivariance unit test).
  2. The horizon staircase $T(\epsilon)\sim\log(1/\epsilon)/\lambda$ is a clean log-law on the TRUE integrator, and its
     slope recovers the textbook Lorenz exponent $\lambda_1\approx0.9056$/t. This is the load-bearing protocol Step 70
     then re-runs on the *learned* model; if it failed on the true integrator the whole staircase claim would be void.

Run:  .venv/bin/python tests/test_step70.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import step70_lorenz_horizon as s70  # noqa: E402

LORENZ_LAMBDA1 = 0.9056          # documented largest Lyapunov exponent of the standard Lorenz attractor (per time)


def test_lorenz_z2_equivariance() -> None:
    r"""$f(g\cdot s)=g\cdot f(s)$ for $g=\mathrm{diag}(-1,-1,1)$ — the symmetry the certificate's config axis uses."""
    rng = np.random.default_rng(0)
    s = rng.uniform(-20, 20, size=(64, 3))
    g = np.array([-1.0, -1.0, 1.0])
    lhs = s70.lorenz_rhs(s * g)                 # f(g . s)
    rhs = s70.lorenz_rhs(s) * g                 # g . f(s)
    err = np.abs(lhs - rhs).max()
    assert err < 1e-9, f"Lorenz Z2 equivariance broken: max|f(g.s)-g.f(s)|={err:.2e}"
    print(f"PASS: Lorenz vector field is exactly Z2-equivariant (max err {err:.2e}).")


def test_horizon_staircase_recovers_lyapunov_on_true_integrator() -> None:
    r"""On the GROUND-TRUTH Lorenz integrator the certified horizon $T(\epsilon)$ is linear in $\log(1/\epsilon)$ and the
    slope recovers $1/(\lambda_1 dt)$ — validating the staircase protocol independent of any learned model."""
    rng = np.random.default_rng(1)
    dt = s70.DT
    n_pert, T_roll = 64, 600
    eps_res = 0.5
    eps_sweep = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])

    qs = rng.uniform(-15, 15, size=(n_pert, 3)); qs[:, 2] = rng.uniform(5, 40, size=n_pert)
    # normalize in the same spirit as Step 70 (use a long reference trajectory for the scale)
    ref = s70.integrate(qs, 800).reshape(-1, 3)
    mu, sd = ref.mean(0), ref.std(0)
    norm = lambda a: (a - mu) / sd
    dirs = rng.standard_normal((n_pert, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    Tcross = []
    for e in eps_sweep:
        base = norm(s70.integrate(qs, T_roll))
        pert = norm(s70.integrate((norm(qs) + e * dirs) * sd + mu, T_roll))
        gm = (np.linalg.norm(pert - base, axis=-1) / e).mean(0)
        cr = int(np.argmax(gm >= eps_res / e))
        Tcross.append(float(cr) if gm[cr] >= eps_res / e else float("nan"))
    Tcross = np.array(Tcross)
    valid = ~np.isnan(Tcross)
    assert valid.sum() >= 5, f"too few resolutions crossed within horizon ({valid.sum()})"

    xs = np.log(1.0 / eps_sweep)[valid]; ys = Tcross[valid]
    slope, icpt = np.polyfit(xs, ys, 1)
    pred = slope * xs + icpt
    r2 = 1.0 - np.sum((ys - pred) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-12)
    lam_staircase = 1.0 / (slope * dt)          # per time-unit

    assert r2 > 0.95, f"horizon staircase not linear on the true integrator: R^2={r2:.3f}"
    relerr = abs(lam_staircase - LORENZ_LAMBDA1) / LORENZ_LAMBDA1
    assert relerr < 0.25, (f"staircase slope does not recover Lorenz lambda1: "
                           f"lam_staircase={lam_staircase:.3f}/t vs {LORENZ_LAMBDA1}/t (rel-err {relerr:.1%})")
    print(f"PASS: horizon staircase on the TRUE Lorenz integrator is linear (R^2 {r2:.3f}) and its slope recovers "
          f"lambda1 = {lam_staircase:.3f}/t vs textbook {LORENZ_LAMBDA1}/t (rel-err {relerr:.1%}).")
    print("=> the T~log(1/eps)/lambda protocol is sound; Step 70 shows the LEARNED model inherits it.")


if __name__ == "__main__":
    test_lorenz_z2_equivariance()
    test_horizon_staircase_recovers_lyapunov_on_true_integrator()
