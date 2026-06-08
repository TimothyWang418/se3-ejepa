r"""Step 76 — the RESERVOIR (ESN) baseline for the high-$N$ spectrum-recovery claim (defends Step 74 / Exp 18).

Step 74 showed a $\mathbb{Z}_N$-equivariant cyclic-conv recovers the full $40$-D Lorenz-96 Lyapunov spectrum
($R^2\approx0.99$) where a **dense feedforward MLP** of equal data fails ($R^2<0$). A reviewer's sharpest attack:

  "The spectrum-from-data literature recovers chaotic Lyapunov spectra with **reservoir computing / echo-state
   networks**, not feedforward MLPs (Pathak et al. 2017/2018; Vlachas et al. 2020; and Kobayashi et al. 2024 did
   *exactly* Lorenz-96 at $K{=}8$). Your unstructured baseline is the wrong one — a properly-tuned ESN, also without
   symmetry, would recover the $40$-D spectrum too, and 'structure beats scale' collapses."

This experiment runs that baseline honestly: a standard leaky echo-state network — the exact tool that line uses —
under the **same data** (and *less* compute: the readout is a ridge solve, not SGD). Does it recover the $40$-D
spectrum **without** the $\mathbb{Z}_N$ prior?

**Mechanism prediction.** The ESN's autonomous map has a *dense* $D_r\times D_r$ Jacobian (no banded-circulant
structure). Step 74's thesis — spurious off-diagonal Jacobian mass wrecks the high-$N$ spectrum — therefore predicts
the ESN should *also* struggle at $N{=}40$ even though it works at low $N$. That is the falsifiable thing we test.

**Positive control (implementation correctness — the linchpin).** Kobayashi et al. recover the Lorenz-96 spectrum at
low $N$ with a reservoir, so our ESN MUST too; otherwise the implementation is wrong/under-tuned and nothing here is
trustworthy. We run the ESN at $N{=}N_{\mathrm{ctrl}}$ (default 12, the regime where even the dense MLP succeeds) as a
positive control before the $N{=}40$ test.

**ESN.** Reservoir $r\in\mathbb R^{D_r}$, leaky update $r_{t+1}=(1-\alpha)r_t+\alpha\tanh(Wr_t+W_{\mathrm{in}}u_t+b)$,
linear readout $\hat u=W_{\mathrm{out}}^\top r$ trained by ridge. In *autonomous* (closed-loop) mode the input is the
model's own prediction, giving the autonomous map $M(r)=(1-\alpha)r+\alpha\tanh(W_c r+b)$ with
$W_c=W+W_{\mathrm{in}}W_{\mathrm{out}}^\top$, whose **analytic** Jacobian
$J(r)=(1-\alpha)I+\alpha\,\mathrm{diag}(\mathrm{sech}^2 z)\,W_c$ ($z=W_c r+b$) drives a thin-frame Benettin–QR for the
top-$k$ Lyapunov exponents in reservoir space (the leading $N$ approximate the true spectrum; the rest are the
reservoir's spurious contracting modes — cf. Hart 2024's conditional-Lyapunov caveat). Exponents are in the same
$1/\Delta t_{\mathrm{map}}$ units as Step 74, so they compare directly to its true/conv/MLP spectra.

**Hyperparameters are selected by validation autonomous-forecast NRMSE — NOT by spectrum recovery (that would be
circular).** Compute is in the ESN's favour (a ridge solve vs the MLP's 80 SGD epochs), so a failure cannot be blamed
on an unfair compute budget.

Honest gate (prints INCONCLUSIVE / which branch, never loosens a threshold):
  (PC)  positive control: ESN at N_ctrl recovers (R^2 > 0.80). If it FAILS -> ABORT ("ESN impl not trustworthy").
  Then the headline is whichever is TRUE at N=40:
   (A) STRUCTURE-STILL-WINS : ESN R^2 < 0.50  AND  (conv R^2 - ESN R^2) > 0.40   -> claim holds, hardened.
   (B) SOFTEN               : ESN R^2 > 0.80                                     -> an unstructured *recurrent* model
                                                                                    also recovers; narrow the claim.
   (else) INCONCLUSIVE (the ESN lands in the middle).

Run:    .venv/bin/python experiments/step76_lorenz96_reservoir_baseline.py
smoke:  STEP76_SMOKE=1 .venv/bin/python experiments/step76_lorenz96_reservoir_baseline.py
seeded: STEP76_SEED=0|1|2     (loads papers/figures/step74_lorenz96_spectrum{_seedS}.json for conv/MLP)
Writes: papers/figures/step76_reservoir_baseline{_seedS}.{json,png}    device: cpu, float64 (MPS lacks float64)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as s74  # noqa: E402  (shares Lorenz-96 dynamics, true spectrum, QR, kaplan_yorke)

SMOKE = bool(int(os.environ.get("STEP76_SMOKE", "0")))
SEED = int(os.environ.get("STEP76_SEED", "0"))
DEVICE = "cpu"                       # float64 QR; MPS has no float64
DTYPE = torch.float64
NCTRL = int(os.environ.get("STEP76_NCTRL", "12"))     # positive-control dimension (unstructured-recovers regime)


# --------------------------------------------------------------------------------------------------------------- #
# Echo-state network: random sparse reservoir, ridge readout, analytic autonomous Jacobian.
# --------------------------------------------------------------------------------------------------------------- #
def build_esn(N: int, Dr: int, rho: float, sigma_in: float, alpha: float, density: float, seed: int) -> dict:
    r"""Random leaky ESN. $W$ sparse, rescaled to spectral radius $\rho$; $W_{\rm in}\sim U[-\sigma_{\rm in},\sigma_{\rm in}]$."""
    g = torch.Generator().manual_seed(1000 * seed + 17)
    W = torch.zeros(Dr, Dr, dtype=DTYPE)
    mask = torch.rand(Dr, Dr, generator=g) < density
    W[mask] = (torch.rand(int(mask.sum()), generator=g, dtype=DTYPE) - 0.5) * 2.0
    # rescale to spectral radius rho (largest |eigenvalue|)
    with torch.no_grad():
        ev = torch.linalg.eigvals(W).abs().max().item()
    if ev > 0:
        W = W * (rho / ev)
    W_in = (torch.rand(Dr, N, generator=g, dtype=DTYPE) - 0.5) * 2.0 * sigma_in
    b = (torch.rand(Dr, generator=g, dtype=DTYPE) - 0.5) * 2.0 * 0.1
    return {"W": W, "W_in": W_in, "b": b, "alpha": alpha, "Dr": Dr, "N": N}


def drive(esn: dict, u: torch.Tensor) -> torch.Tensor:
    r"""Teacher-forced drive on inputs ``u`` (T, N). Returns reservoir states ``r_1..r_T`` (T, Dr); ``r_t`` depends on
    ``u_{t-1}`` (so state ``r_t`` is paired with target ``u_t`` for next-step prediction)."""
    W, W_in, b, a = esn["W"], esn["W_in"], esn["b"], esn["alpha"]
    r = torch.zeros(esn["Dr"], dtype=DTYPE)
    out = []
    for t in range(u.shape[0]):
        r = (1 - a) * r + a * torch.tanh(W @ r + W_in @ u[t] + b)
        out.append(r)
    return torch.stack(out, 0)


def train_readout(esn: dict, u: torch.Tensor, washout: int, beta: float) -> torch.Tensor:
    r"""Ridge: with states ``S=r_1..r_{T-1}`` (predict next), targets ``Y=u_2..u_T``. ``W_out`` solves
    $\min\|SW-Y\|^2+\beta\|W\|^2$, i.e. $W=(S^\top S+\beta I)^{-1}S^\top Y$, shape (Dr, N)."""
    states = drive(esn, u[:-1])                      # r_1..r_{T-1}, each depends on u_0..u_{T-2}
    S = states[washout:]
    Y = u[1 + washout:]                              # align: state r_t predicts u_{t+1}
    Dr = esn["Dr"]
    A = S.T @ S + beta * torch.eye(Dr, dtype=DTYPE)
    Wout = torch.linalg.solve(A, S.T @ Y)            # (Dr, N)
    return Wout


def autonomous_Wc(esn: dict, Wout: torch.Tensor) -> torch.Tensor:
    r"""Closed-loop coupling $W_c=W+W_{\rm in}W_{\rm out}^\top$ (input replaced by readout $\hat u=W_{\rm out}^\top r$)."""
    return esn["W"] + esn["W_in"] @ Wout.T


def auto_step(esn: dict, Wc: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    a, b = esn["alpha"], esn["b"]
    return (1 - a) * r + a * torch.tanh(Wc @ r + b)


def auto_jacobian(esn: dict, Wc: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""Analytic Jacobian $J(r)=(1-\alpha)I+\alpha\,\mathrm{diag}(\mathrm{sech}^2 z)\,W_c$, $z=W_c r+b$."""
    a, b = esn["alpha"], esn["b"]
    z = Wc @ r + b
    d = a * (1.0 - torch.tanh(z) ** 2)               # alpha * sech^2 z
    J = d[:, None] * Wc
    J.diagonal().add_(1 - a)
    return J


def sync_state(esn: dict, u: torch.Tensor, washout: int) -> torch.Tensor:
    r"""Drive teacher-forced through ``u`` (washout, N) to synchronize the reservoir, return final state."""
    return drive(esn, u)[-1]


def auto_forecast_nrmse(esn: dict, Wc: torch.Tensor, Wout: torch.Tensor, r0: torch.Tensor,
                        u_true: torch.Tensor) -> float:
    r"""Closed-loop H-step forecast NRMSE vs ``u_true`` (H, N); large/``inf`` if it blows up (selection penalty)."""
    r = r0.clone()
    preds = []
    for _ in range(u_true.shape[0]):
        r = auto_step(esn, Wc, r)
        preds.append(Wout.T @ r)
    P = torch.stack(preds, 0)
    if not torch.isfinite(P).all():
        return float("inf")
    num = ((P - u_true) ** 2).mean().sqrt().item()
    den = u_true.std().item() + 1e-12
    return num / den


def esn_lyapunov_topk(esn: dict, Wc: torch.Tensor, r0: torch.Tensor, k: int, n_steps: int, warmup: int,
                      dt_map: float, seed: int) -> torch.Tensor:
    r"""Thin-frame Benettin–QR: evolve a $D_r\times k$ orthonormal frame by the analytic Jacobian, accumulate
    $\log|\mathrm{diag}(R)|$, return the top-$k$ exponents (sorted desc) in $1/\Delta t_{\rm map}$ units."""
    Dr = esn["Dr"]
    a = esn["alpha"]
    g = torch.Generator().manual_seed(7 * seed + 3)
    Q, _ = torch.linalg.qr(torch.randn(Dr, k, generator=g, dtype=DTYPE))
    acc = torch.zeros(k, dtype=DTYPE)
    cnt = 0
    r = r0.clone()
    for t in range(n_steps + warmup):
        z = Wc @ r + esn["b"]
        d = a * (1.0 - torch.tanh(z) ** 2)
        JQ = (1 - a) * Q + d[:, None] * (Wc @ Q)         # J @ Q without forming the dense J
        r = (1 - a) * r + a * torch.tanh(z)
        Q, R = torch.linalg.qr(JQ)
        if t >= warmup:
            acc = acc + torch.log(torch.abs(torch.diagonal(R)).clamp_min(1e-300))
            cnt += 1
    return torch.sort(acc / (cnt * dt_map), descending=True).values


# --------------------------------------------------------------------------------------------------------------- #
# Fit + select (by forecast NRMSE) + recover spectrum, on one Lorenz-96 system.
# --------------------------------------------------------------------------------------------------------------- #
def fit_select_recover(N: int, seed: int, lam_true: torch.Tensor, tag: str) -> dict:
    n_train = 3000 if SMOKE else 20000
    Dr = 200 if SMOKE else 800
    washout = 100 if SMOKE else 300
    ly_steps = 400 if SMOKE else 2000
    ly_warm = 80 if SMOKE else 300
    H_val = 30 if SMOKE else 60
    k = min(N + 12, Dr)

    traj = s74.attractor_traj(N, n_train, seed, DEVICE).to(DTYPE)
    mu, sd = traj.mean(0), traj.std(0) + 1e-8
    un = (traj - mu) / sd                                       # normalized inputs (Lyapunov exps are coord-invariant)
    n_fit = un.shape[0] - H_val - washout - 2
    u_fit = un[:n_fit]
    u_sync = un[n_fit: n_fit + washout]                        # sync segment for the autonomous start
    u_val = un[n_fit + washout: n_fit + washout + H_val]       # held-out forecast target

    grid = ([(0.7, 1.0, 1.0)] if SMOKE else
            [(rho, sin, al) for rho in (0.4, 0.7, 0.95) for sin in (0.5, 1.0) for al in (1.0,)])
    beta = 1e-6
    density = 0.02 if not SMOKE else 0.05

    best = None
    for (rho, sin, al) in grid:
        esn = build_esn(N, Dr, rho, sin, al, density, seed)
        Wout = train_readout(esn, u_fit, washout, beta)
        Wc = autonomous_Wc(esn, Wout)
        r0 = sync_state(esn, u_sync, washout)
        nrmse = auto_forecast_nrmse(esn, Wc, Wout, r0, u_val)
        # one-step (teacher-forced) relMSE for reporting
        with torch.no_grad():
            S = drive(esn, un[:-1])
            P = S @ Wout
            relmse = (((P - un[1:]) ** 2).sum(-1) / (un[1:] ** 2).sum(-1).clamp_min(1e-12)).mean().item()
        cand = {"rho": rho, "sigma_in": sin, "alpha": al, "nrmse": nrmse, "relmse": relmse,
                "esn": esn, "Wout": Wout, "Wc": Wc, "r0": r0}
        print(f"[step76]   {tag} N={N} rho={rho} sin={sin} a={al}: fcast-NRMSE {nrmse:.3f}  1step-relMSE {relmse:.1e}",
              file=sys.stderr)
        if best is None or nrmse < best["nrmse"]:
            best = cand

    lam = esn_lyapunov_topk(best["esn"], best["Wc"], best["r0"], k, ly_steps, ly_warm, s74.DTMAP, seed)
    lamN = lam[:N]                                             # leading N exponents approximate the true N-spectrum
    lt = lam_true.cpu().numpy()
    ll = lamN.cpu().numpy()
    l1_t = float(lam_true[0])
    r2 = 1.0 - float(((ll - lt) ** 2).sum()) / max(float(((lt - lt.mean()) ** 2).sum()), 1e-12)
    out = {"tag": tag, "N": N, "Dr": Dr, "rho": best["rho"], "sigma_in": best["sigma_in"], "alpha": best["alpha"],
           "forecast_nrmse": best["nrmse"], "one_step_relmse": best["relmse"],
           "lambda1": float(lamN[0]), "lambda1_relerr": abs(float(lamN[0]) - l1_t) / abs(l1_t),
           "sum_topN": float(lamN.sum()), "n_positive": int((lamN > 0).sum()),
           "ky": s74.kaplan_yorke(lamN), "spectrum_r2": r2,
           "cond_lyap_max": float(lam[N:].max()) if lam.shape[0] > N else float("nan"),
           "lambda_learned": ll.tolist()}
    print(f"[step76]   {tag} N={N} BEST rho={best['rho']} sin={best['sigma_in']}: spectrum R^2 {r2:.3f}  "
          f"lambda1 {float(lamN[0]):.3f} (err {out['lambda1_relerr']:.0%})  #pos {out['n_positive']}  "
          f"KY {out['ky']:.2f}  cond-lyap-max {out['cond_lyap_max']:.3f}", file=sys.stderr)
    return out


def _load_step74(seed: int) -> dict:
    tag = f"_seed{seed}" if seed else ""
    p = Path(__file__).resolve().parent.parent / "papers" / "figures" / f"step74_lorenz96_spectrum{tag}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def run(seed: int) -> int:
    N = int(os.environ.get("STEP76_N", "10" if SMOKE else "40"))
    print(f"[step76] reservoir/ESN baseline (seed {seed}, {DEVICE}, float64); positive control N={NCTRL}, test N={N}",
          file=sys.stderr)

    # ---- positive control: ESN must recover the spectrum at low N (else implementation is untrustworthy) ----
    traj_c = s74.attractor_traj(NCTRL, 1500 if not SMOKE else 600, seed, DEVICE).to(DTYPE)
    mu_c, sd_c = traj_c.mean(0), traj_c.std(0) + 1e-8
    x0_c = (traj_c[len(traj_c) // 2] - mu_c) / sd_c
    lam_true_c = s74.lyapunov_spectrum(lambda xn: (s74.true_map(xn * sd_c + mu_c) - mu_c) / sd_c, x0_c,
                                       600 if not SMOKE else 200, 100 if not SMOKE else 50)
    ctrl = fit_select_recover(NCTRL, seed, lam_true_c, tag="ctrl")
    pc_pass = bool(ctrl["spectrum_r2"] > 0.80)

    # ---- the test: ESN at N=40, same data as step74; compare to step74's conv/MLP ----
    s74j = _load_step74(seed)
    if s74j and s74j.get("N") == N:                           # only reuse the canonical conv/MLP when N matches
        lam_true = torch.tensor(s74j["lambda_true"], dtype=DTYPE)
        conv_r2, mlp_r2 = s74j["equivariant"]["spectrum_r2"], s74j["mlp"]["spectrum_r2"]
    else:                                                     # fallback: compute the true N spectrum fresh
        traj = s74.attractor_traj(N, 20000 if not SMOKE else 3000, seed, DEVICE).to(DTYPE)
        mu, sd = traj.mean(0), traj.std(0) + 1e-8
        x0 = (traj[len(traj) // 2] - mu) / sd
        lam_true = s74.lyapunov_spectrum(lambda xn: (s74.true_map(xn * sd + mu) - mu) / sd, x0,
                                         2500 if not SMOKE else 600, 400 if not SMOKE else 100)
        conv_r2, mlp_r2 = float("nan"), float("nan")
    esn = fit_select_recover(N, seed, lam_true, tag="test")

    # ---- honest branch ----
    esn_r2 = esn["spectrum_r2"]
    have_ref = bool(s74j and s74j.get("N") == N)
    structure_wins = bool(pc_pass and esn_r2 < 0.50 and have_ref and (conv_r2 - esn_r2) > 0.40)
    soften = bool(pc_pass and esn_r2 > 0.80)
    if not pc_pass:
        branch = "ABORT_PC_FAIL"
    elif structure_wins:
        branch = "STRUCTURE_STILL_WINS"
    elif soften:
        branch = "SOFTEN"
    else:
        branch = "INCONCLUSIVE"

    res = {"seed": seed, "smoke": SMOKE, "N": N, "N_ctrl": NCTRL, "branch": branch,
           "positive_control_pass": pc_pass, "positive_control": ctrl, "esn_test": esn,
           "conv_r2": conv_r2, "mlp_r2": mlp_r2, "esn_r2": esn_r2,
           "conv_minus_esn": (conv_r2 - esn_r2) if s74j else None,
           "lambda_true": lam_true.cpu().numpy().tolist()}
    _save(res)

    print(f"\n[step76] === RESULT (seed {seed}) ===", file=sys.stderr)
    print(f"[step76] positive control N={NCTRL}: ESN R^2 {ctrl['spectrum_r2']:.3f}  -> {'PASS' if pc_pass else 'FAIL (impl untrustworthy)'}",
          file=sys.stderr)
    print(f"[step76] test N={N}: true vs models  |  Z_N-conv R^2 {conv_r2:.3f}  dense-MLP R^2 {mlp_r2:.3f}  "
          f"ESN R^2 {esn_r2:.3f}  (cond-lyap-max {esn['cond_lyap_max']:.3f})", file=sys.stderr)
    if branch == "STRUCTURE_STILL_WINS":
        print(f"[step76] >>> STRUCTURE STILL WINS: even a tuned ESN (the literature's spectrum-recoverer, valid at "
              f"N={NCTRL}) fails at N={N} without symmetry (R^2 {esn_r2:.3f}); the Z_N prior closes a gap NO "
              f"unstructured model of equal data does. The structure-beats-scale claim is hardened.", file=sys.stderr)
    elif branch == "SOFTEN":
        print(f"[step76] >>> SOFTEN: an unstructured *recurrent* ESN ALSO recovers the N={N} spectrum (R^2 {esn_r2:.3f}). "
              f"Narrow the claim to 'a dense FEEDFORWARD model of equal data fails; recurrence/structure closes it'.",
              file=sys.stderr)
    elif branch == "INCONCLUSIVE":
        print(f"[step76] >>> INCONCLUSIVE: ESN R^2 {esn_r2:.3f} is between the gates; report as a partial-recovery "
              f"middle case (neither a clean win nor a clean tie).", file=sys.stderr)
    else:
        print(f"[step76] >>> ABORT: positive control failed — fix the ESN before trusting the N={N} number.",
              file=sys.stderr)
    return 0 if branch in ("STRUCTURE_STILL_WINS", "SOFTEN") else 1


def _save(res: dict) -> None:
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"_seed{res['seed']}" if res["seed"] else ""
    (figdir / f"step76_reservoir_baseline{tag}.json").write_text(json.dumps(res, indent=2))
    try:
        lt = np.array(res["lambda_true"])
        le = np.array(res["esn_test"]["lambda_learned"])
        s74j = _load_step74(res["seed"])
        fig, ax = plt.subplots(1, 1, figsize=(5.6, 5.0))
        lo = float(min(lt.min(), le.min())); hi = float(max(lt.max(), le.max()))
        if s74j:
            lc = np.array(s74j["equivariant"]["lambda_learned"]); lm = np.array(s74j["mlp"]["lambda_learned"])
            lo = float(min(lo, lc.min(), lm.min())); hi = float(max(hi, lc.max(), lm.max()))
            ax.scatter(lt, lc, s=22, color="#1f77b4", zorder=3, label=f"$\\mathbb{{Z}}_N$-conv ($R^2={res['conv_r2']:.2f}$)")
            ax.scatter(lt, lm, s=26, color="#d62728", marker="x", zorder=2, label=f"dense MLP ($R^2={res['mlp_r2']:.2f}$)")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
        ax.scatter(lt, le, s=30, facecolors="none", edgecolors="#2ca02c", zorder=4,
                   label=f"ESN/reservoir ($R^2={res['esn_r2']:.2f}$)")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("true Lyapunov exponent $\\lambda_j$")
        ax.set_ylabel("learned-model exponent $\\hat\\lambda_j$")
        ax.set_title(f"{res['N']}-D Lorenz-96: ESN baseline vs structure\n(positive control N={res['N_ctrl']}: "
                     f"ESN $R^2={res['positive_control']['spectrum_r2']:.2f}$)")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(figdir / f"step76_reservoir_baseline{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step76]   (figure skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run(SEED))
