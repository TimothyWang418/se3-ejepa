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

# step74 supplies the (uncontrolled) Lorenz-96 attractor sampler we reuse for on-attractor initial states. Import it;
# do NOT modify it. (Its module-level constants — DT, F_FORCE — agree with ours.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as step74  # noqa: E402

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
