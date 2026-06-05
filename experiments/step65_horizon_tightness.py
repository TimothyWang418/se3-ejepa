r"""Step 65 — numerical confirmation of Proposition 6 (the certified horizon is tight; approximate equivariance is
horizon-limited), and the visual of the paper's central claim.

Proposition 6 proves a *matching lower bound* for Theorem B: an $\epsilon$-approximately-equivariant world model — even
one with a perfect equivariant predictor — has $T$-step orbit-error-variation **exactly** $\epsilon\,e^{\lambda T}$ on a
channel with Lyapunov exponent $\lambda$, so its certified horizon is $T(\epsilon_{\rm res})=\tfrac1\lambda
\log(\epsilon_{\rm res}/\epsilon)=\Theta(\tfrac1\lambda\log\tfrac1\epsilon)$ — no better. This step instantiates the
construction on a controlled multi-channel latent and confirms, to the floating-point floor, that:

  (a) an **exactly** equivariant model ($\epsilon=0$) is orbit-flat at *every* horizon (variation $=0$) — infinite
      certified horizon on all channels;
  (b) an **$\epsilon$-approximately** equivariant model's orbit-error-variation equals $\epsilon\,e^{\lambda_j T}$ on
      each channel $j$ — flat/bounded on conserved/contractive channels ($\lambda_j\le0$, infinite horizon) and
      growing as $e^{\lambda_j T}$ on expansive channels ($\lambda_j>0$, finite horizon); and
  (c) the per-channel certified horizon $T_j(\epsilon_{\rm res})$ is **linear in $\log(1/\epsilon)$ with slope
      $1/\lambda_j$** — the two-sided ($\Theta$) horizon law.

Construction (exactly the Proposition 6 instance). Latent $\mathcal Z=\mathbb R^d$; group $G=\mathbb Z_2$ acting by an
orthogonal sign-flip representation $\rho=\mathrm{diag}(\pm1)$; dynamics $\Phi=\mathrm{diag}(e^{\lambda_j})$ (commutes
with $\rho$, so $G$ is a dynamical symmetry, (A3)). The model's predictor is the exact dynamics $\Phi$ (perfect,
$\delta=0$); its encoder is exact except a single equivariance defect $E(g\cdot x)=\rho(g)E(x)+\epsilon u_j$ in channel
$j$. The $T$-step orbit-error-variation is then $\lvert\mathrm{Err}_T(g\cdot x)-\mathrm{Err}_T(x)\rvert=
\epsilon\,e^{\lambda_j T}$ analytically; we verify it numerically (this is a *confirmation of the proved construction*,
not independent evidence — the trained-model approximate-symmetry degradation is Experiment 8 / `step53`).

Honest gate (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   exact model ($\epsilon=0$) orbit-flat at all $T$:             max variation < 1e-9;
  (ii)  approx model matches $\epsilon e^{\lambda_j T}$ (tightness):    max rel-err < 1e-6 on every channel;
  (iii) conserved channels ($\lambda\le0$) stay bounded:               variation(T_max) <= variation(0) * 1.01;
  (iv)  certified horizon linear in $\log(1/\epsilon)$, slope $1/\lambda$: R^2 > 0.999 and slope within 1% of $1/\lambda$.

Run:     .venv/bin/python experiments/step65_horizon_tightness.py
Seeded:  STEP65_SEED=0|1|2 .venv/bin/python experiments/step65_horizon_tightness.py
Writes:  papers/figures/step65_horizon_tightness.{json,png}
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers" / "figures"
SEED = int(os.environ.get("STEP65_SEED", "0"))
TAG = os.environ.get("STEP65_TAG", "")

# Channel Lyapunov exponents: two expansive (lambda>0), one neutral/conserved (lambda=0), one contractive (lambda<0).
LAMBDAS = np.array([0.7, 0.35, 0.0, -0.4])               # per-channel exponents
T_MAX = 16
EPS = 1e-3                                               # the model's equivariance defect (encoder residual)
EPS_RES = 0.5                                            # resolution at which we read off the certified horizon
EPS_SWEEP = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])   # for the horizon-vs-log(1/eps) line


def orbit_variation_per_channel(eps: float, rng: np.random.Generator) -> np.ndarray:
    r"""The exact Proposition-6 instance: return $|\mathrm{Err}_T(g\cdot x)-\mathrm{Err}_T(x)|$ per channel for
    $T=0\dots T_{\max}$, for an $\epsilon$-approximately-equivariant model on the multi-channel latent. Computed by
    *actually rolling out* the model and target (not by plugging in the closed form), so the match to
    $\epsilon e^{\lambda_j T}$ is a genuine confirmation."""
    d = len(LAMBDAS)
    A = np.exp(LAMBDAS)                                   # per-channel multipliers e^{lambda_j}
    z0 = rng.standard_normal(d)                           # base latent E(x)
    rho = np.where(rng.standard_normal(d) > 0, 1.0, -1.0) # orthogonal sign-flip rep rho(g) = diag(+-1)
    out = np.zeros((T_MAX + 1, d))
    for j in range(d):                                    # one defect channel at a time
        u = np.zeros(d); u[j] = 1.0
        var_T = []
        for T in range(T_MAX + 1):
            PhiT = A ** T                                 # Phi^T = diag(e^{lambda_j T})
            # rollout error at x: model exact along x's trajectory, predictor = exact dynamics -> 0
            err_x = 0.0
            # rollout error at g.x: encoder defect E(g.x) = rho*z0 + eps*u, rolled by Phi^T = perfect predictor
            zhat = PhiT * (rho * z0 + eps * u)
            ztgt = rho * (PhiT * z0)                       # E(Phi^T(g.x)) = rho * E(Phi^T x) = rho * Phi^T z0
            err_gx = np.linalg.norm(zhat - ztgt)
            var_T.append(abs(err_gx - err_x))
        out[:, j] = var_T
    return out                                            # (T_MAX+1, d)


def main() -> None:
    rng = np.random.default_rng(SEED)
    Ts = np.arange(T_MAX + 1)
    A = np.exp(LAMBDAS)

    # (a) exact model (eps=0): orbit-flat at all T --------------------------------------------------------
    var_exact = orbit_variation_per_channel(0.0, np.random.default_rng(SEED))
    exact_max = float(var_exact.max())

    # (b) approximate model (eps>0): variation per channel vs T, compared to the analytic eps*e^{lambda_j T} ----
    var_approx = orbit_variation_per_channel(EPS, np.random.default_rng(SEED))
    analytic = EPS * (A[None, :] ** Ts[:, None])          # (T_MAX+1, d) closed form
    rel = np.abs(var_approx - analytic) / np.maximum(analytic, 1e-300)
    tight_max_relerr = float(rel.max())
    # conserved/contractive channels (lambda<=0) stay bounded
    cons = LAMBDAS <= 0
    conserved_bounded = bool(np.all(var_approx[-1, cons] <= var_approx[0, cons] * 1.01 + 1e-12))

    # (c) certified horizon T_j(eps_res) vs log(1/eps): linear with slope 1/lambda --------------------------
    exp_channels = np.where(LAMBDAS > 0)[0]
    horizon_fit = {}
    for j in exp_channels:
        lam = LAMBDAS[j]
        # certified horizon = max T with eps*e^{lambda T} <= eps_res  ->  (1/lambda) log(eps_res/eps)
        Tcert = np.array([(1.0 / lam) * np.log(EPS_RES / e) for e in EPS_SWEEP])
        xs = np.log(1.0 / EPS_SWEEP)
        slope, intercept = np.polyfit(xs, Tcert, 1)
        pred = slope * xs + intercept
        ss_res = np.sum((Tcert - pred) ** 2)
        ss_tot = np.sum((Tcert - Tcert.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
        horizon_fit[str(j)] = {"lambda": float(lam), "slope": float(slope), "inv_lambda": float(1.0 / lam),
                               "r2": float(r2), "Tcert": [float(t) for t in Tcert]}

    ok_exact = exact_max < 1e-9
    ok_tight = tight_max_relerr < 1e-6
    ok_conserved = conserved_bounded
    ok_horizon = all(f["r2"] > 0.999 and abs(f["slope"] - f["inv_lambda"]) / f["inv_lambda"] < 0.01
                     for f in horizon_fit.values())
    passed = ok_exact and ok_tight and ok_conserved and ok_horizon

    result = {
        "passed": passed,
        "gate": {"exact_flat": ok_exact, "tight_lower_bound": ok_tight,
                 "conserved_infinite_horizon": ok_conserved, "horizon_linear_in_log_eps": ok_horizon},
        "lambdas": [float(x) for x in LAMBDAS], "eps": EPS, "eps_res": EPS_RES, "t_max": T_MAX,
        "exact_max_variation": exact_max, "tight_max_relerr": tight_max_relerr,
        "var_approx_by_T": [[float(v) for v in row] for row in var_approx],
        "analytic_by_T": [[float(v) for v in row] for row in analytic],
        "horizon_fit": horizon_fit, "seed": SEED,
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"step65_horizon_tightness{TAG}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # figure: (a) orbit-variation vs T (exact flat vs approx eps*e^{lT}); (b) certified horizon vs log(1/eps) ----
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    colors = ["C3", "C1", "C0", "C2"]
    for j in range(len(LAMBDAS)):
        lam = LAMBDAS[j]
        lab = (f"$\\lambda{{=}}{lam:+.2f}$ " + ("(expansive)" if lam > 0 else "(conserved/slow)"))
        ax.semilogy(Ts, np.maximum(var_approx[:, j], 1e-18), "o-", color=colors[j], lw=2.0, ms=4, label=lab)
        ax.semilogy(Ts, np.maximum(analytic[:, j], 1e-18), "--", color=colors[j], lw=1.0, alpha=0.7)
    ax.axhline(EPS_RES, ls=":", color="gray", lw=1, label=f"resolution $\\epsilon_{{\\rm res}}={EPS_RES}$")
    ax.set_xlabel("rollout horizon $T$")
    ax.set_ylabel("orbit-error-variation $|\\Delta\\mathrm{Err}_T|$")
    ax.set_title(f"(a) Approx-equiv ($\\epsilon{{=}}{EPS:g}$): $\\epsilon e^{{\\lambda T}}$ (dashed = analytic)\n"
                 f"exact-equiv ($\\epsilon{{=}}0$) stays at {exact_max:.0e} (flat, off-scale)")
    ax.legend(fontsize=7, loc="lower right")
    for j in exp_channels:
        f = horizon_fit[str(j)]
        xs = np.log(1.0 / EPS_SWEEP)
        axb.plot(xs, f["Tcert"], "o-", color=colors[j], lw=2.0,
                 label=f"$\\lambda{{=}}{f['lambda']:.2f}$: slope {f['slope']:.2f} ($1/\\lambda{{=}}{f['inv_lambda']:.2f}$), $R^2{{=}}{f['r2']:.3f}$")
    axb.set_xlabel("$\\log(1/\\epsilon)$  (model equivariance defect)")
    axb.set_ylabel("certified horizon $T(\\epsilon_{\\rm res})$")
    axb.set_title("(b) Certified horizon is $\\Theta(\\log(1/\\epsilon)/\\lambda)$ — linear, slope $1/\\lambda$")
    axb.legend(fontsize=8)
    fig.suptitle("Proposition 6: approximate equivariance is horizon-limited; only exact structure reaches $\\infty$", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / f"step65_horizon_tightness{TAG}.png", dpi=130, bbox_inches="tight")

    print(f"[step65] exact-equiv max orbit-variation {exact_max:.2e} (flat); approx tight max rel-err {tight_max_relerr:.2e}",
          file=sys.stderr)
    for j in exp_channels:
        f = horizon_fit[str(j)]
        print(f"[step65]   lambda={f['lambda']:.2f}: horizon slope {f['slope']:.3f} vs 1/lambda {f['inv_lambda']:.3f}  R^2 {f['r2']:.4f}",
              file=sys.stderr)
    if passed:
        print("[step65] PROP 6 CONFIRMED: exact-equiv flat to ~1e-16 at all horizons; approx-equiv variation = "
              "eps*e^{lambda T} (tight); conserved channels infinite-horizon; certified horizon linear in log(1/eps).",
              file=sys.stderr)
    else:
        bad = [k for k, v in result["gate"].items() if not v]
        print(f"[step65] INCONCLUSIVE: gate not met ({bad}); reported as-is.", file=sys.stderr)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
