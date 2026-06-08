r"""Step 79 — certified-control co-demonstration on controlled Lorenz-96.

Anchor experiment for the two-axis co-demonstration (design: docs/specs/2026-06-07-step5-certified-control-codemo.md;
plan: docs/plans/2026-06-07-step79-certified-control.md).

Phase 0: the controlled dynamics + Z_N-equivariance.
Phase 1: an **action-conditioned world model** $\hat\phi:(x_t,u_t)\mapsto x_{t+1}$ on controlled Lorenz-96, in two
flavours — a $\mathbb{Z}_N$-equivariant **cyclic-conv** (treating $[x,u]$ as 2 input channels through circular convs,
so the learned $\Delta t$-map is exactly cyclic-equivariant, mirroring the true field) and a dense-MLP **baseline**
(not equivariant) — plus on-attractor data collection under random controls and a multi-step rollout trainer. The
control input $u$ enters as a planted forcing channel; rolling the model under the data's controls constrains the
*controlled* composed Jacobian, the operator whose certified per-channel horizons Phase 2+ will read off.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# step74 supplies the (uncontrolled) Lorenz-96 attractor sampler we reuse for on-attractor initial states. step78
# supplies the certified-horizon machinery (Benettin-QR log|R| series + block-bootstrap spectrum CI + horizon
# interval). Import both; do NOT modify them. (step74's module-level constants — DT, F_FORCE — agree with ours.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as step74  # noqa: E402
import step78_certified_horizon_ci as step78  # noqa: E402

DTYPE = torch.float64
F_FORCE = 8.0
DT = 0.01


def l96_controlled_rhs(x: torch.Tensor, u: torch.Tensor, F: float = F_FORCE) -> torch.Tensor:
    r"""Controlled Lorenz-96 field $\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F+u_i$. Jointly $\mathbb{Z}_N$-equivariant in
    $(x,u)$: a cyclic shift of both inputs shifts the output (roll commutes with the local coupling and with $u$)."""
    return (torch.roll(x, -1, -1) - torch.roll(x, 2, -1)) * torch.roll(x, 1, -1) - x + F + u


def rk4_controlled(x: torch.Tensor, u: torch.Tensor, dt: float = DT, F: float = F_FORCE) -> torch.Tensor:
    r"""One RK4 step of the controlled $\Delta t$-map with a zero-order-hold control $u$ (held constant over the step)."""
    k1 = l96_controlled_rhs(x, u, F); k2 = l96_controlled_rhs(x + 0.5 * dt * k1, u, F)
    k3 = l96_controlled_rhs(x + 0.5 * dt * k2, u, F); k4 = l96_controlled_rhs(x + dt * k3, u, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------------------------------------------- #
# Phase 1 — action-conditioned world models. We learn a one-step map of the controlled Delta t-map,
# hat_phi(x_t, u_t) ~= x_{t+1}, in normalized state coordinates. Two flavours that differ ONLY in whether the
# Z_N (cyclic-shift) symmetry of the true controlled field is built in:
#   * L96ControlledConv -- stacks circular Conv1d over the 2-channel input [x, u]; exactly Z_N-equivariant.
#   * L96ControlledMLP  -- dense baseline over the concatenation [x, u]; NOT equivariant.
# Mirrors step74's L96CyclicConv / L96MLP, extended to take a control channel u alongside the state x.
# --------------------------------------------------------------------------------------------------------------- #
class L96ControlledConv(nn.Module):
    r"""$\mathbb{Z}_N$-equivariant action-conditioned world model: a stack of **circular** 1-D convolutions over the
    **2-channel** input $[x,u]$ (state and control stacked as channels). Because the true controlled field
    $\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F+u_i$ is jointly cyclic-equivariant in $(x,u)$ and locally coupled, a
    circular conv on $[x,u]$ has a **banded-circulant** Jacobian structurally matching it; the learned $\Delta t$-map
    $\hat\phi(x,u)=x+\mathrm{conv}([x,u])$ is **exactly** $\mathbb{Z}_N$-equivariant by construction (a cyclic shift of
    both inputs shifts the output). Mirrors :class:`step74.L96CyclicConv` with an extra input channel for $u$."""

    def __init__(self, N: int, channels: int = 64, kernel: int = 5, layers: int = 4):
        super().__init__()
        self.kernel = kernel
        chans = [2] + [channels] * (layers - 1) + [1]          # 2 input channels [x, u]; last conv back to 1 channel
        self.convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], kernel) for i in range(layers))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""``x, u: (..., N) -> (..., N)``, $\mathbb{Z}_N$-equivariant. Stacks $[x,u]$ as 2 channels, runs circular
        convs (SiLU between layers), residual on the state ``x``. Handles arbitrary leading dims via reshape."""
        shp = x.shape
        xf = x.reshape(-1, shp[-1]); uf = u.reshape(-1, shp[-1])
        h = torch.stack([xf, uf], dim=1)                       # (B, 2, N): channel 0 = state, channel 1 = control
        pad = (self.kernel - 1) // 2
        for i, conv in enumerate(self.convs):
            h = torch.nn.functional.pad(h, (pad, pad), mode="circular")   # circular => cyclic (Z_N) equivariance
            h = conv(h)
            if i < len(self.convs) - 1:
                h = torch.nn.functional.silu(h)
        return x + h.reshape(shp)                              # residual on x: Delta t-map = identity + increment


class L96ControlledMLP(nn.Module):
    r"""Dense action-conditioned baseline (NOT equivariant): concatenate $[x,u]\in\mathbb{R}^{2N}$ and pass through a
    residual MLP, $\hat\phi(x,u)=x+\mathrm{MLP}([x,u])$. Its dense $N\times 2N$ first layer mixes all sites, so a cyclic
    shift of $(x,u)$ does **not** commute with it. Mirrors :class:`step74.L96MLP` with the control concatenated in."""

    def __init__(self, N: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        mods = [nn.Linear(2 * N, hidden), nn.SiLU()]           # 2N inputs: [x, u] concatenated along the last dim
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.SiLU()]
        mods += [nn.Linear(hidden, N)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:   # (..., N), (..., N) -> (..., N)
        h = torch.cat([x, u], dim=-1)                          # (..., 2N)
        return x + self.net(h)                                 # residual on x


def make_equivariant_wm(N: int) -> L96ControlledConv:
    r"""Factory for the $\mathbb{Z}_N$-equivariant action-conditioned world model (used by later phases)."""
    return L96ControlledConv(N)


def collect_data(N: int, n_traj: int, K: int, seed: int, u_max: float = 1.0):
    r"""Collect ``n_traj`` length-``K`` segments of the TRUE controlled dynamics, started from on-attractor states and
    driven by random zero-order-hold controls $u\sim\mathcal U[-u_{\max},u_{\max}]^N$.

    From each of ``n_traj`` distinct on-attractor initial states (sampled along a single long :func:`step74.attractor_traj`
    of the **uncontrolled** field), we roll :func:`rk4_controlled` for ``K`` steps under fresh random per-step controls.
    Normalization statistics ``mu, sd`` are computed over **all visited states** (the starts and every rolled target).

    Returns ``(starts, controls, targets, mu, sd)``, all ``float64``:
      * ``starts:   (n_traj, N)``   -- initial states, **normalized** by ``(x - mu) / sd``.
      * ``controls: (n_traj, K, N)``-- the applied controls, **raw** (already bounded in $[-u_{\max},u_{\max}]$).
      * ``targets:  (n_traj, K, N)``-- ``targets[i, k]`` is the state $k{+}1$ steps after ``starts[i]``, **normalized**.
    """
    g = torch.Generator().manual_seed(seed)
    # On-attractor initial states: one long burned-in trajectory of the uncontrolled field, subsampled to n_traj points.
    # Ask for enough states to choose n_traj distinct inits even when n_traj is large.
    traj = step74.attractor_traj(N, max(n_traj * 2, n_traj + 1), seed, device="cpu")   # (>=n_traj+1, N), float64
    idx = torch.randperm(traj.shape[0], generator=g)[:n_traj]
    x0 = traj[idx].clone().to(DTYPE)                            # (n_traj, N) raw on-attractor states

    # Random controls u ~ U[-u_max, u_max], one per (segment, step). Shape (n_traj, K, N).
    controls = (2.0 * u_max) * torch.rand(n_traj, K, N, generator=g, dtype=DTYPE) - u_max

    # Roll the TRUE controlled dynamics; collect raw targets x_{t+1..t+K}.
    targets_raw = torch.empty(n_traj, K, N, dtype=DTYPE)
    x = x0.clone()
    for k in range(K):
        x = rk4_controlled(x, controls[:, k, :])
        targets_raw[:, k, :] = x

    # Normalization stats over ALL visited states (starts + every rolled target).
    visited = torch.cat([x0.unsqueeze(1), targets_raw], dim=1).reshape(-1, N)   # (n_traj*(K+1), N)
    mu = visited.mean(0)
    sd = visited.std(0) + 1e-8

    starts = (x0 - mu) / sd
    targets = (targets_raw - mu) / sd                          # controls stay RAW (already bounded)
    return starts, controls, targets, mu, sd


def train_wm(kind: str, N: int, data, seed: int, epochs: int = 60, K: int = 5):
    r"""Train an action-conditioned world model with a $K$-step **rollout** loss, feeding the DATA's controls at each
    step: $z_0=\,$``starts``, $z_{k+1}=\hat\phi(z_k,u_k)$, matching $z_{k+1}$ to ``targets[:,k]``. One-step MSE constrains
    only the values; the rollout constrains the *composed controlled Jacobian* $\prod_k D_x\hat\phi(z_k,u_k)$ — the
    operator whose per-channel log-stretching gives the certified horizons. Mirrors :func:`step74.train_model`.

    ``kind in {"conv","mlp"}`` selects the equivariant conv or the dense-MLP baseline. ``data`` is the tuple returned by
    :func:`collect_data` (``starts, controls, targets, mu, sd``); its ``controls/targets`` second axis must have length
    ``>= K``. Adam, lr ``1e-3``. Returns ``(model, mu, sd, one_step_relmse)`` (relMSE over the 1-step prediction)."""
    torch.manual_seed(seed)
    starts, controls, targets, mu, sd = data
    assert kind in ("conv", "mlp"), f"unknown kind {kind!r} (expected 'conv' or 'mlp')"
    assert controls.shape[1] >= K and targets.shape[1] >= K, "data horizon shorter than rollout K"
    starts = starts.to(DTYPE); controls = controls.to(DTYPE); targets = targets.to(DTYPE)

    model = (L96ControlledConv(N) if kind == "conv" else L96ControlledMLP(N)).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = starts.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 512):
            idx = torch.randperm(n, generator=g)[i:i + 512]
            z = starts[idx]
            loss = 0.0
            for k in range(K):
                z = model(z, controls[idx, k, :])             # feed the DATA's control at step k
                loss = loss + ((z - targets[idx, k, :]) ** 2).mean()
            loss = loss / K
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():                                      # one-step relMSE: predict targets[:,0] from starts, u_0
        pred = model(starts, controls[:, 0, :])
        y = targets[:, 0, :]
        one_step_relmse = (((pred - y) ** 2).sum(-1) / (y ** 2).sum(-1).clamp_min(1e-12)).mean().item()
    return model, mu, sd, one_step_relmse


# --------------------------------------------------------------------------------------------------------------- #
# Phase 2 — read a CERTIFIED HORIZON T_j(eps) with a bootstrap CI off the trained action-conditioned world model.
# The WM is action-conditioned: hat_phi(x, u). The certificate is a property of the predictor's *autonomous* (free-
# running) dynamics, so we read it from the **zero-control map** g(x) = hat_phi(x, u=0) -- the learned Delta t-map the
# model would iterate if no control were applied. We run Benettin-QR on g's autograd Jacobian to get the predictor's
# Lyapunov spectrum, block-bootstrap a per-channel CI (reusing step78), and turn lambda_1's CI into a certified-horizon
# interval T_1(eps) in [T_lo, T_hi] = [log(1/eps)/lambda_hi, log(1/eps)/lambda_lo]. (Lyapunov exponents are coordinate-
# invariant, so doing this in the WM's normalized frame is exact -- no need to map back to physical units.)
# --------------------------------------------------------------------------------------------------------------- #
def certificate(model, mu, sd, N: int, eps: float = 0.01, n_steps: int = 2000, warmup: int = 400,
                n_boot: int = 1000, block: int = 50, seed: int = 0) -> dict:
    r"""Certified horizon $T_j(\epsilon)$ with a block-bootstrap CI, read off a trained action-conditioned world model.

    The predictor $\hat\phi(x,u)$ is action-conditioned; the certificate is a property of its **autonomous map**
    $g(x_n)=\hat\phi(x_n,u{=}0)$ (the free-running $\Delta t$-map under zero control), in the WM's **normalized**
    coordinates $x_n=(x-\mu)/\sigma$. We run Benettin–QR on $g$'s autograd Jacobian at an on-attractor operating point
    to estimate the predictor's Lyapunov spectrum, block-bootstrap a per-channel CI (reusing :mod:`step78`), and convert
    $\lambda_1$'s CI into the certified-horizon interval $T_1(\epsilon)\in[\log(1/\epsilon)/\lambda_{\rm hi},
    \log(1/\epsilon)/\lambda_{\rm lo}]$. Because Lyapunov exponents are coordinate-invariant, working in the normalized
    frame is exact. If the $\lambda_1$ CI straddles/$\le 0$ (non-chaotic predictor), the certificate **abstains**: the
    $T$ fields are ``None`` (see :func:`step78.horizon_interval`).

    Args:
        model: trained WM with signature ``model(x, u) -> x_next`` in normalized coords (e.g. from :func:`train_wm`).
        mu, sd: ``(N,)`` normalization stats from :func:`collect_data` (used only to place ``x0`` on the attractor).
        N: state dimension.  eps: resolution at which the horizon is reported.
        n_steps, warmup: post-warmup QR steps and warmup steps for the Benettin frame.
        n_boot, block, seed: moving-block-bootstrap settings for the per-channel spectrum CI.

    Returns:
        dict with keys ``lambda`` (sorted-descending point spectrum, len-N list), ``lambda_lo``/``lambda_hi`` (per-channel
        CI lists), ``lambda1`` (float, fastest exponent), ``lambda1_ci`` (``[lo, hi]``), ``T1``/``T1_lo``/``T1_hi``
        (floats, or ``None`` on abstain), and ``eps``.
    """
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)

    # Autonomous map in NORMALIZED coords: the WM predicting one step under zero control. autograd flows through it so
    # step78.qr_logR_series can take the Jacobian D_x g at each step.
    def g(xn: torch.Tensor) -> torch.Tensor:
        return model(xn, torch.zeros_like(xn))

    # On-attractor operating point: burn-in trajectory of the TRUE uncontrolled field, normalized to the WM's frame, and
    # take a mid-trajectory state (well past burn-in, representative of the attractor).
    traj = step74.attractor_traj(N, n_steps // 4, seed, "cpu").double()    # (n_steps//4 + 1, N) raw on-attractor states
    x0 = ((traj[len(traj) // 2] - mu) / sd).detach()                       # normalized mid-trajectory operating point

    logR = step78.qr_logR_series(g, x0, n_steps, warmup)                   # (n_steps, N) per-step log|diag(R)|
    point, lo, hi = step78.bootstrap_spectrum_ci(logR, step74.DTMAP, n_boot, block, seed)   # sorted desc, each len N

    # Binding channel = fastest positive exponent = index 0 after the descending sort (if it is positive). Its certified
    # horizon interval comes from the lambda1 CI; the point horizon from the point lambda1.
    iv = step78.horizon_interval(lo[0], hi[0], eps) if point[0] > 0 else None
    T1 = (float(np.log(1.0 / eps) / point[0]) if point[0] > 0 else None)
    T1_lo = (float(iv[0]) if iv is not None else None)
    T1_hi = (float(iv[1]) if iv is not None else None)

    return {"lambda": point.tolist(), "lambda_lo": lo.tolist(), "lambda_hi": hi.tolist(),
            "lambda1": float(point[0]), "lambda1_ci": [float(lo[0]), float(hi[0])],
            "T1": T1, "T1_lo": T1_lo, "T1_hi": T1_hi, "eps": eps}


def _smoke_phase2() -> None:
    r"""Train a small equivariant conv WM and read its certified horizon $T_1(\epsilon)$ with a bootstrap CI off the
    autonomous ($u{=}0$) map. Not a pytest test (it trains). Confirms the learned predictor is chaotic ($\lambda_1>0$)
    and yields a finite, positive certified horizon $T_1$."""
    torch.manual_seed(0)
    N = 12
    data = collect_data(N=N, n_traj=2000, K=3, seed=0)
    model, mu, sd, relmse = train_wm("conv", N, data, seed=0, epochs=20, K=3)
    cert = certificate(model, mu, sd, N, eps=0.01, n_steps=800, warmup=150, n_boot=300, block=40)
    l1, (l1_lo, l1_hi) = cert["lambda1"], cert["lambda1_ci"]
    print(f"[step79 phase2 smoke] N={N} conv WM one-step relMSE {relmse:.2e}", file=sys.stderr)
    print(f"[step79 phase2 smoke] lambda1 = {l1:.4f}  CI[{l1_lo:.4f}, {l1_hi:.4f}]  (eps={cert['eps']})",
          file=sys.stderr)
    print(f"[step79 phase2 smoke] T1 = {cert['T1']}  [T1_lo={cert['T1_lo']}, T1_hi={cert['T1_hi']}] (map steps)",
          file=sys.stderr)
    assert l1 > 0, f"learned WM autonomous map not chaotic: lambda1 = {l1:.4f} <= 0"
    assert cert["T1"] is not None and np.isfinite(cert["T1"]) and cert["T1"] > 0, \
        f"certified horizon T1 not finite-positive: {cert['T1']}"
    print("[step79 phase2 smoke] PASS: trained WM yields a chaotic certificate (lambda1>0) with a finite positive "
          "certified horizon T1 and a bootstrap CI.", file=sys.stderr)


def _smoke() -> None:
    r"""Tiny end-to-end check that the equivariant conv WM actually *learns* the controlled one-step map. Not a pytest
    test (it trains — too slow for CI). Asserts the conv one-step relMSE is below ``1e-2``."""
    torch.manual_seed(0)
    N = 10
    data = collect_data(N=N, n_traj=400, K=3, seed=0)
    _, _, _, conv_relmse = train_wm("conv", N, data, seed=0, epochs=8, K=3)
    _, _, _, mlp_relmse = train_wm("mlp", N, data, seed=0, epochs=8, K=3)
    print(f"[step79 smoke] N={N} n_traj=400 K=3 epochs=8 -> conv one-step relMSE {conv_relmse:.2e}  "
          f"mlp one-step relMSE {mlp_relmse:.2e}", file=sys.stderr)
    assert conv_relmse < 1e-2, f"conv one-step relMSE {conv_relmse:.2e} not < 1e-2"
    print("[step79 smoke] PASS: equivariant conv WM learns the controlled one-step map (relMSE < 1e-2).",
          file=sys.stderr)


if __name__ == "__main__":
    _smoke()
