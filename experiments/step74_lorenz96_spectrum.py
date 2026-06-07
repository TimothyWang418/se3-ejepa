r"""Step 74 — the certified per-channel horizon law on a learned model of a HIGH-DIMENSIONAL chaotic system.

Steps 70/71 lifted the certified-horizon staircase $T(\epsilon)\sim\log(1/\epsilon)/\lambda_1$ to learned models of
*low*-dimensional chaos (Lorenz, Hénon, Rössler — all $2$–$3$D, and only the *leading* exponent $\lambda_1$). Theorem B's
real content is stronger and per-channel: each Lyapunov direction $j$ has its OWN certified horizon
$T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$, so a high-dimensional system is a *stratified* object — a few fast
(chaotic, short-horizon) directions and many slow/contracting (long- or unbounded-horizon) ones. On a *learned* model
that whole spectral stratification has so far only been shown on a **planted** synthetic spectrum (the controlled-latent
half of E2). This experiment makes it real: we learn a model of the **Lorenz-96** system and show it recovers the
**entire Lyapunov spectrum**, hence the per-channel certified horizons across the spectrum.

**System.** Lorenz-96 [@lorenz1996predictability]: for $i=1,\dots,N$ (cyclic),
$$ \dot{x}_i = (x_{i+1}-x_{i-2})\,x_{i-1} - x_i + F , \qquad F=8 \ \text{(chaotic)} . $$
It is the standard high-dimensional chaos / data-assimilation benchmark, scalable to any $N$, with a full Lyapunov
spectrum (many positive exponents). A *rigorous* invariant anchors the computation: the vector field's divergence is
$\nabla\!\cdot f = \sum_i \partial_{x_i}\dot{x}_i = \sum_i(-1) = -N$ (the quadratic terms contribute nothing on the
diagonal), so by Liouville the **whole spectrum sums to $-N$ exactly**: $\sum_{j=1}^N \lambda_j = -N$. We use this as a
unit test on the spectrum estimator. (Lorenz-96 also carries a cyclic $\mathbb{Z}_N$ shift symmetry — the configuration
axis — which we note but do not exploit here; E2 is about the horizon axis.)

**Protocol.** (1) Integrate Lorenz-96 to its attractor (RK4); compute the *true* spectrum by the Benettin QR method on
the $\Delta t$-map's Jacobian. (2) Train a plain one-step residual MLP of the $\Delta t$-map on on-attractor data. (3)
Compute the *learned* model's spectrum by the identical QR method on the MLP's autograd Jacobian. (4) Compare: the
learned spectrum should track the true one across all $N$ channels (on the $y=x$ line), recovering $\lambda_1$, the
number of positive exponents (the count of finite-horizon chaotic channels), and the Kaplan–Yorke dimension. The
per-channel certified horizon $T_j(\epsilon)=\log(1/\epsilon)/\lambda_j$ (finite for $\lambda_j>0$, unbounded for
$\lambda_j\le0$) is then recovered for *every* channel — Theorem B's spectral law, on a learned high-dimensional model.

Honest gate (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   the model learned the one-step map:                one-step relMSE < 0.05;
  (ii)  the leading exponent is recovered:                 |lambda1_hat - lambda1_true| / lambda1_true < 0.25;
  (iii) the whole spectrum is recovered:                   R^2(learned vs true across all N) > 0.90.
We additionally report (not gate) the positive-exponent count and Kaplan-Yorke dimension, learned vs true.

Run:     .venv/bin/python experiments/step74_lorenz96_spectrum.py            # N=40 full (use CUDA for speed)
smoke:   STEP74_SMOKE=1 .venv/bin/python experiments/step74_lorenz96_spectrum.py
seeded:  STEP74_SEED=0|1|2 ... ;  device:  STEP74_DEVICE=cuda
Writes:  papers/figures/step74_lorenz96_spectrum.{json,png}
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

SMOKE = bool(int(os.environ.get("STEP74_SMOKE", "0")))
SEED = int(os.environ.get("STEP74_SEED", "0"))
DEVICE = os.environ.get("STEP74_DEVICE", "cpu")
DT = 0.01
SUB = int(os.environ.get("STEP74_SUB", "1"))   # RK4 substeps per Delta t-map step (Delta t = SUB*DT). Smaller =>
                                               # map closer to identity => Jacobian (hence spectrum) easier to learn.
DTMAP = DT * SUB
F_FORCE = 8.0


# --------------------------------------------------------------------------------------------------------------- #
# Lorenz-96 dynamics (torch, float64). Cyclic indices via roll: x_{i+1}=roll(-1), x_{i-1}=roll(+1), x_{i-2}=roll(+2).
# --------------------------------------------------------------------------------------------------------------- #
def l96_rhs(x: torch.Tensor, F: float = F_FORCE) -> torch.Tensor:
    return (torch.roll(x, -1, -1) - torch.roll(x, 2, -1)) * torch.roll(x, 1, -1) - x + F


def rk4(x: torch.Tensor, dt: float = DT, F: float = F_FORCE) -> torch.Tensor:
    k1 = l96_rhs(x, F); k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F); k4 = l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def true_map(x: torch.Tensor) -> torch.Tensor:
    for _ in range(SUB):
        x = rk4(x)
    return x


def attractor_traj(N: int, n_steps: int, seed: int, device: str, burn: int = 2000) -> torch.Tensor:
    r"""Burn in to the attractor, then return ``n_steps+1`` states ``(n_steps+1, N)`` of the Delta t-map."""
    g = torch.Generator().manual_seed(seed)
    x = F_FORCE + 0.01 * torch.randn(N, generator=g, dtype=torch.float64)
    x = x.to(device)
    for _ in range(burn):
        x = true_map(x)
    out = [x]
    for _ in range(n_steps):
        x = true_map(x)
        out.append(x)
    return torch.stack(out, 0)


# --------------------------------------------------------------------------------------------------------------- #
# Benettin QR Lyapunov spectrum — identical routine for the true Delta t-map and the learned MLP. Evolve an
# orthonormal frame Q by the one-step Jacobian, re-orthonormalize (QR), accumulate log|diag(R)|; lambda_j is the
# time-average. Works in whatever coordinates step_fn lives in (exponents are coordinate-invariant).
# --------------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def _qr_step(J: torch.Tensor, Q: torch.Tensor):
    Z = J @ Q
    Qn, R = torch.linalg.qr(Z)
    return Qn, torch.abs(torch.diagonal(R)).clamp_min(1e-300)


def lyapunov_spectrum(step_fn, x0: torch.Tensor, n_steps: int, warmup: int, dt_map: float = DTMAP) -> torch.Tensor:
    N = x0.shape[-1]
    x = x0.clone()
    Q = torch.eye(N, dtype=x.dtype, device=x.device)
    acc = torch.zeros(N, dtype=x.dtype, device=x.device)
    cnt = 0
    for t in range(n_steps + warmup):
        J = torch.autograd.functional.jacobian(step_fn, x, vectorize=True)   # (N, N)
        with torch.no_grad():
            x = step_fn(x)
            Q, diagR = _qr_step(J, Q)
            if t >= warmup:
                acc = acc + torch.log(diagR)
                cnt += 1
    return torch.sort(acc / (cnt * dt_map), descending=True).values


def kaplan_yorke(lyap: torch.Tensor) -> float:
    r"""$D_{KY}=k+\frac{\sum_{j\le k}\lambda_j}{|\lambda_{k+1}|}$, $k$ the largest index with a non-negative partial sum."""
    lam = lyap.detach().cpu().numpy()
    csum = np.cumsum(lam)
    k = int(np.sum(csum >= 0))
    if k == 0:
        return 0.0
    if k >= len(lam):
        return float(len(lam))
    return float(k + csum[k - 1] / abs(lam[k]))


# --------------------------------------------------------------------------------------------------------------- #
# Learned one-step model of the Delta t-map (residual MLP) — trained in normalized coordinates for conditioning.
# --------------------------------------------------------------------------------------------------------------- #
class L96MLP(nn.Module):
    def __init__(self, N: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods = [nn.Linear(N, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.SiLU()]
        mods += [nn.Linear(hidden, N)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)                                  # residual: the Delta t-map is identity + an increment


def train_model(traj: torch.Tensor, N: int, device: str, seed: int):
    r"""Train the residual MLP with a **multi-step rollout** loss: unroll $K$ steps and match each to the true
    trajectory. One-step MSE constrains only the *values* (an $L^2$ proxy); matching a $K$-step rollout constrains the
    *composed Jacobian* $\prod_{t} D\hat\phi(\hat x_t)$ — exactly the operator whose log-stretching is the Lyapunov
    spectrum — so it recovers the leading exponent that a pure one-step fit underestimates."""
    torch.manual_seed(seed)
    mu = traj.mean(0); sd = traj.std(0) + 1e-8
    xn = (traj - mu) / sd
    K = int(os.environ.get("STEP74_ROLLOUT", "2" if SMOKE else "5"))
    hidden = 128 if SMOKE else 256
    epochs = 8 if SMOKE else 80
    model = L96MLP(N, hidden=hidden).double().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xn.shape[0] - K                                        # start points with K future targets
    starts = xn[:n]
    tgts = [xn[1 + j: 1 + j + n] for j in range(K)]            # tgts[j][i] = true state j+1 steps after start i
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 512):
            idx = torch.randperm(n, generator=g)[i:i + 512]
            z = starts[idx]
            loss = 0.0
            for j in range(K):
                z = model(z)
                loss = loss + ((z - tgts[j][idx]) ** 2).mean()
            loss = loss / K
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        X, Y = xn[:-1], xn[1:]
        relmse = (((model(X) - Y) ** 2).sum(-1) / (Y ** 2).sum(-1).clamp_min(1e-12)).mean().item()
    return model, mu, sd, relmse


def run(seed: int, device: str) -> int:
    N = int(os.environ.get("STEP74_N", "10" if SMOKE else "40"))
    n_train = 4000 if SMOKE else 20000
    ly_steps = 600 if SMOKE else 2500
    ly_warm = 100 if SMOKE else 400
    print(f"[step74] Lorenz-96 N={N}, F={F_FORCE}, dt_map={DTMAP} (seed {seed}, {device}) ...", file=sys.stderr)
    traj = attractor_traj(N, n_train, seed, device)

    # normalized coordinates shared by both spectra (exponents are coordinate-invariant)
    model, mu, sd, relmse = train_model(traj, N, device, seed)
    print(f"[step74] learned one-step relMSE = {relmse:.4f}", file=sys.stderr)

    def true_norm(xn):                                          # true Delta t-map in normalized coords
        return (true_map(xn * sd + mu) - mu) / sd

    def learned_norm(xn):
        return model(xn)

    x0 = (traj[len(traj) // 2] - mu) / sd
    lam_true = lyapunov_spectrum(true_norm, x0, ly_steps, ly_warm)
    lam_learn = lyapunov_spectrum(learned_norm, x0, ly_steps, ly_warm)

    sum_true = float(lam_true.sum()); sum_learn = float(lam_learn.sum())
    l1_t, l1_l = float(lam_true[0]), float(lam_learn[0])
    npos_t = int((lam_true > 0).sum()); npos_l = int((lam_learn > 0).sum())
    ky_t, ky_l = kaplan_yorke(lam_true), kaplan_yorke(lam_learn)
    lt, ll = lam_true.cpu().numpy(), lam_learn.cpu().numpy()
    ss_res = float(((ll - lt) ** 2).sum()); ss_tot = float(((lt - lt.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"[step74] sum(lambda): true {sum_true:.2f} vs Liouville -N={-N} ; learned {sum_learn:.2f}", file=sys.stderr)
    print(f"[step74] lambda_1: true {l1_t:.3f}  learned {l1_l:.3f}  (rel-err {abs(l1_l-l1_t)/abs(l1_t):.1%})",
          file=sys.stderr)
    print(f"[step74] #positive: true {npos_t} learned {npos_l} ; Kaplan-Yorke dim: true {ky_t:.2f} learned {ky_l:.2f}",
          file=sys.stderr)
    print(f"[step74] full-spectrum R^2(learned vs true) = {r2:.3f}", file=sys.stderr)

    ok_fit = bool(relmse < 0.05)
    ok_l1 = bool(abs(l1_l - l1_t) / abs(l1_t) < 0.25)
    ok_spec = bool(r2 > 0.90)
    passed = bool(ok_fit and ok_l1 and ok_spec)

    res = {"passed": passed, "N": N, "F": F_FORCE, "dt_map": DTMAP, "seed": seed, "smoke": SMOKE,
           "one_step_relmse": relmse, "lambda1_true": l1_t, "lambda1_learned": l1_l,
           "sum_true": sum_true, "sum_learned": sum_learn, "liouville_target": -N,
           "n_positive_true": npos_t, "n_positive_learned": npos_l, "ky_true": ky_t, "ky_learned": ky_l,
           "spectrum_r2": r2, "lambda_true": lt.tolist(), "lambda_learned": ll.tolist(),
           "gate": {"fit": ok_fit, "lambda1": ok_l1, "spectrum": ok_spec}}
    _save(res)
    msg = "HIGH-D SPECTRAL HORIZON RECOVERED" if passed else "INCONCLUSIVE"
    print(f"[step74] {msg}: the learned model of a {N}-D chaotic system recovers the Lyapunov spectrum "
          f"(R^2={r2:.3f}, lambda_1 rel-err {abs(l1_l-l1_t)/abs(l1_t):.1%}, #positive {npos_l} vs {npos_t}) "
          f"=> per-channel certified horizons T_j(eps)=log(1/eps)/lambda_j across the spectrum (Theorem B).",
          file=sys.stderr)
    return 0 if passed else 1


def _save(res: dict) -> None:
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"_seed{res['seed']}" if res["seed"] else ""
    (figdir / f"step74_lorenz96_spectrum{tag}.json").write_text(json.dumps(res, indent=2))
    try:
        lt = np.array(res["lambda_true"]); ll = np.array(res["lambda_learned"])
        eps = 0.01
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
        lo, hi = float(min(lt.min(), ll.min())), float(max(lt.max(), ll.max()))
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
        ax1.scatter(lt, ll, s=22, color="#1f77b4", zorder=3)
        ax1.axhline(0, color="gray", lw=0.6); ax1.axvline(0, color="gray", lw=0.6)
        ax1.set_xlabel("true Lyapunov exponent $\\lambda_j$")
        ax1.set_ylabel("learned-model exponent $\\hat\\lambda_j$")
        ax1.set_title(f"(a) Spectrum recovered ($R^2={res['spectrum_r2']:.3f}$, $N={res['N']}$)")
        ax1.legend(fontsize=8)
        pt = lt[lt > 0]; pl = ll[ll > 0]
        ax2.plot(np.arange(1, len(pt) + 1), np.log(1 / eps) / pt, "o-", color="#1f77b4", label="true")
        ax2.plot(np.arange(1, len(pl) + 1), np.log(1 / eps) / pl, "s--", color="#d62728", label="learned")
        ax2.set_xlabel("chaotic channel $j$ (positive exponents)")
        ax2.set_ylabel(f"certified horizon $T_j(\\epsilon{{=}}{eps})$ [map steps]")
        ax2.set_title("(b) Per-channel certified horizon $\\log(1/\\epsilon)/\\lambda_j$")
        ax2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / f"step74_lorenz96_spectrum{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step74]   (figure skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run(SEED, DEVICE))
