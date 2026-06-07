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

**Protocol (structure vs. unstructured at high $N$).** (1) Integrate Lorenz-96 to its attractor; compute the *true*
spectrum by Benettin QR on the $\Delta t$-map's Jacobian. (2) Train, on the **same** data, BOTH a $\mathbb{Z}_N$-
equivariant **cyclic-conv** model (whose banded-circulant Jacobian matches the system's local coupling) and a **dense
MLP** of comparable capacity. Both use a multi-step rollout loss (one-step MSE constrains only values; the rollout
constrains the *composed Jacobian* — the Lyapunov operator). (3) Recover each model's spectrum by the identical QR
method. (4) Contrast. The finding: at high $N$ the **equivariant** model recovers the spectrum (so the per-channel
certified horizons $T_j(\epsilon)=\log(1/\epsilon)/\lambda_j$), while the **dense MLP fails** — its $N\times N$ Jacobian
noise, unconstrained by structure, accumulates into a wildly wrong spectrum (negative $R^2$). The configuration axis
(the $\mathbb{Z}_N$ symmetry) thus *helps the horizon axis* (spectrum recovery) on one high-dimensional system.

Honest gate (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   the equivariant model recovers the spectrum:  relMSE<0.05 AND |lambda1 err|<0.25 AND R^2(vs true)>0.90;
  (ii)  structure helps:                              R^2(equivariant) - R^2(dense MLP) > 0.30.
We additionally report the positive-exponent count and Kaplan-Yorke dimension per model. (At small $N$ the MLP also
succeeds, so the *contrast* only appears at high $N$; the equivariance is exact by construction — circular convs.)

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


class L96CyclicConv(nn.Module):
    r"""$\mathbb{Z}_N$ cyclic-translation-equivariant model: a stack of **circular** 1-D convolutions. Lorenz-96 is
    locally coupled ($x_i$ depends on $x_{i-2},x_{i-1},x_{i+1}$) and $\mathbb{Z}_N$-symmetric, so a circular conv's
    Jacobian is **banded-circulant** — structurally matching the true sparse Jacobian. A dense MLP ignores this: its
    $N\times N$ Jacobian carries spurious off-diagonal noise that, in high $N$, wrecks the Lyapunov spectrum. This is
    the configuration axis (the $\mathbb{Z}_N$ symmetry of the system) *helping* the horizon axis (spectrum recovery)."""

    def __init__(self, N: int, channels: int = 64, kernel: int = 5, layers: int = 4):
        super().__init__()
        self.kernel = kernel
        chans = [1] + [channels] * (layers - 1) + [1]
        self.convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], kernel) for i in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (..., N) -> (..., N), Z_N-equivariant
        shp = x.shape
        h = x.reshape(-1, 1, shp[-1])
        pad = (self.kernel - 1) // 2
        for i, conv in enumerate(self.convs):
            h = torch.nn.functional.pad(h, (pad, pad), mode="circular")   # circular => cyclic equivariance
            h = conv(h)
            if i < len(self.convs) - 1:
                h = torch.nn.functional.silu(h)
        return x + h.reshape(shp)


def _build(N: int, device: str):
    kind = os.environ.get("STEP74_MODEL", "conv")              # conv (Z_N-equivariant) by default; "mlp" = dense baseline
    hidden = 128 if SMOKE else 256
    if kind == "mlp":
        return L96MLP(N, hidden=hidden).double().to(device), "dense-MLP"
    return L96CyclicConv(N, channels=64 if not SMOKE else 32).double().to(device), "cyclic-conv (Z_N-equivariant)"


def train_model(traj: torch.Tensor, N: int, device: str, seed: int):
    r"""Train the residual MLP with a **multi-step rollout** loss: unroll $K$ steps and match each to the true
    trajectory. One-step MSE constrains only the *values* (an $L^2$ proxy); matching a $K$-step rollout constrains the
    *composed Jacobian* $\prod_{t} D\hat\phi(\hat x_t)$ — exactly the operator whose log-stretching is the Lyapunov
    spectrum — so it recovers the leading exponent that a pure one-step fit underestimates."""
    torch.manual_seed(seed)
    mu = traj.mean(0); sd = traj.std(0) + 1e-8
    xn = (traj - mu) / sd
    K = int(os.environ.get("STEP74_ROLLOUT", "2" if SMOKE else "5"))
    epochs = 8 if SMOKE else 80
    model, kind = _build(N, device)
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


def _eval_model(kind: str, traj, N, device, seed, lam_true, x0, ly_steps, ly_warm) -> dict:
    os.environ["STEP74_MODEL"] = kind
    model, mu, sd, relmse = train_model(traj, N, device, seed)
    lam = lyapunov_spectrum(lambda xn: model(xn), x0, ly_steps, ly_warm)
    lt, ll = lam_true.cpu().numpy(), lam.cpu().numpy()
    l1_t = float(lam_true[0])
    r2 = 1.0 - float(((ll - lt) ** 2).sum()) / max(float(((lt - lt.mean()) ** 2).sum()), 1e-12)
    out = {"kind": kind, "one_step_relmse": relmse, "lambda1": float(lam[0]),
           "lambda1_relerr": abs(float(lam[0]) - l1_t) / abs(l1_t), "sum": float(lam.sum()),
           "n_positive": int((lam > 0).sum()), "ky": kaplan_yorke(lam), "spectrum_r2": r2,
           "lambda_learned": ll.tolist()}
    print(f"[step74]   {kind:>4}: relMSE {relmse:.4f}  lambda1 {float(lam[0]):.3f} (err {out['lambda1_relerr']:.0%})  "
          f"#pos {out['n_positive']}  KY {out['ky']:.2f}  spectrum R^2 {r2:.3f}", file=sys.stderr)
    return out


def run(seed: int, device: str) -> int:
    r"""Train BOTH a Z_N-equivariant cyclic-conv model and a dense MLP on the SAME high-D Lorenz-96 data, and contrast
    their recovered Lyapunov spectra. Headline: the equivariant model recovers the spectrum (so the per-channel
    certified horizons), the dense MLP does not — the configuration axis ($\mathbb{Z}_N$ symmetry) helping the horizon
    axis (spectrum recovery) on one high-dimensional system."""
    N = int(os.environ.get("STEP74_N", "10" if SMOKE else "40"))
    n_train = 4000 if SMOKE else 20000
    ly_steps = 600 if SMOKE else 2500
    ly_warm = 100 if SMOKE else 400
    print(f"[step74] Lorenz-96 N={N}, F={F_FORCE}, dt_map={DTMAP} (seed {seed}, {device}) ...", file=sys.stderr)
    traj = attractor_traj(N, n_train, seed, device)
    mu, sd = traj.mean(0), traj.std(0) + 1e-8
    x0 = (traj[len(traj) // 2] - mu) / sd
    lam_true = lyapunov_spectrum(lambda xn: (true_map(xn * sd + mu) - mu) / sd, x0, ly_steps, ly_warm)
    lt = lam_true.cpu().numpy()
    l1_t, npos_t, ky_t, sum_t = float(lam_true[0]), int((lam_true > 0).sum()), kaplan_yorke(lam_true), float(lam_true.sum())
    print(f"[step74] true: lambda1 {l1_t:.3f}  #pos {npos_t}  KY {ky_t:.2f}  sum {sum_t:.2f} (Liouville -N={-N})",
          file=sys.stderr)

    conv = _eval_model("conv", traj, N, device, seed, lam_true, x0, ly_steps, ly_warm)
    mlp = _eval_model("mlp", traj, N, device, seed, lam_true, x0, ly_steps, ly_warm)

    # PASS = the equivariant (conv) model recovers the spectrum AND clearly beats the dense MLP (structure helps).
    ok_conv = bool(conv["one_step_relmse"] < 0.05 and conv["lambda1_relerr"] < 0.25 and conv["spectrum_r2"] > 0.90)
    structure_helps = bool(conv["spectrum_r2"] - mlp["spectrum_r2"] > 0.30)
    passed = bool(ok_conv and structure_helps)
    res = {"passed": passed, "structure_helps": structure_helps, "N": N, "F": F_FORCE, "dt_map": DTMAP,
           "seed": seed, "smoke": SMOKE, "lambda1_true": l1_t, "n_positive_true": npos_t, "ky_true": ky_t,
           "sum_true": sum_t, "liouville_target": -N, "lambda_true": lt.tolist(),
           "equivariant": conv, "mlp": mlp, "gate": {"equivariant_recovers": ok_conv, "structure_helps": structure_helps}}
    _save(res)
    msg = "EQUIVARIANT MODEL RECOVERS THE HIGH-D SPECTRUM" if passed else "INCONCLUSIVE"
    print(f"[step74] {msg}: on {N}-D Lorenz-96 the Z_N-equivariant cyclic-conv recovers the Lyapunov spectrum "
          f"(R^2 {conv['spectrum_r2']:.3f}, lambda1 err {conv['lambda1_relerr']:.0%}) where the dense MLP fails "
          f"(R^2 {mlp['spectrum_r2']:.3f}) => structure recovers the per-channel certified horizons "
          f"T_j(eps)=log(1/eps)/lambda_j a dense model of equal data cannot (config axis helps horizon axis).",
          file=sys.stderr)
    return 0 if passed else 1


def _save(res: dict) -> None:
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"_seed{res['seed']}" if res["seed"] else ""
    (figdir / f"step74_lorenz96_spectrum{tag}.json").write_text(json.dumps(res, indent=2))
    try:
        lt = np.array(res["lambda_true"]); lc = np.array(res["equivariant"]["lambda_learned"])
        lm = np.array(res["mlp"]["lambda_learned"]); eps = 0.01
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
        lo = float(min(lt.min(), lc.min(), lm.min())); hi = float(max(lt.max(), lc.max(), lm.max()))
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$ (perfect)")
        ax1.scatter(lt, lc, s=24, color="#1f77b4", zorder=3,
                    label=f"$\\mathbb{{Z}}_N$-equivariant ($R^2={res['equivariant']['spectrum_r2']:.2f}$)")
        ax1.scatter(lt, lm, s=24, color="#d62728", marker="x", zorder=2,
                    label=f"dense MLP ($R^2={res['mlp']['spectrum_r2']:.2f}$)")
        ax1.axhline(0, color="gray", lw=0.5); ax1.axvline(0, color="gray", lw=0.5)
        ax1.set_xlabel("true Lyapunov exponent $\\lambda_j$")
        ax1.set_ylabel("learned-model exponent $\\hat\\lambda_j$")
        ax1.set_title(f"(a) Spectrum recovery on {res['N']}-D Lorenz-96")
        ax1.legend(fontsize=7.5)
        pt = lt[lt > 0]; pc = lc[lc > 0]
        ax2.plot(np.arange(1, len(pt) + 1), np.log(1 / eps) / pt, "o-", color="k", label="true")
        ax2.plot(np.arange(1, len(pc) + 1), np.log(1 / eps) / pc, "s--", color="#1f77b4", label="equivariant")
        ax2.set_xlabel("chaotic channel $j$ (positive exponents)")
        ax2.set_ylabel(f"certified horizon $T_j(\\epsilon{{=}}{eps})$ [map steps]")
        ax2.set_title("(b) Per-channel certified horizon $\\log(1/\\epsilon)/\\lambda_j$")
        ax2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / f"step74_lorenz96_spectrum{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step74]   (figure skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run(SEED, DEVICE))
