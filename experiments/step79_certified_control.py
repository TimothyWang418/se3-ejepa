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

    # Normalization stats over ALL visited states (starts + every rolled target). We pool over BOTH the batch AND the
    # N sites to a single SCALAR mean/std (broadcast back to length N), instead of per-site stats. Two reasons:
    #   (i) the controlled Lorenz-96 attractor is statistically Z_N-symmetric (every site is dynamically equivalent), so
    #       the per-site empirical mean/std differ ONLY by finite-sample noise -- a scalar is the symmetric estimator;
    #   (ii) a scalar (equivalently, a roll-invariant constant vector mu*1, sd*1) is EXACTLY Z_N-invariant, S(mu*1)=mu*1.
    #       This is what makes the normalized rollout commute with the cyclic shift: (roll(x,s)-mu)/sd = roll((x-mu)/sd,s).
    #       Per-site mu/sd would inject an O(1) symmetry break into the normalized frame (mu,sd not roll-invariant), which
    #       would degrade the planner's orbit-equivariance from machine precision to ~1e-1 (see Phase 4 orbit_flatness).
    visited = torch.cat([x0.unsqueeze(1), targets_raw], dim=1).reshape(-1, N)   # (n_traj*(K+1), N)
    mu = visited.mean().repeat(N)                              # scalar mean, broadcast to (N,): Z_N-invariant
    sd = (visited.std() + 1e-8).repeat(N)                      # scalar std,  broadcast to (N,): Z_N-invariant

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


# --------------------------------------------------------------------------------------------------------------- #
# Phase 3 — an EXACTLY Z_N-orbit-equivariant PLANNER (model-predictive control). Given the trained action-conditioned
# WM and a raw initial state x0, return a control sequence u_{0:H} that drives the predicted trajectory toward the
# UNSTABLE uniform fixed point x_i = F (chaos suppression). We work entirely in the WM's NORMALIZED coordinates.
#
# Exact-equivariance construction (a deterministic GRADIENT planner from the Z_N-invariant zero control). This mirrors
# the *principle* behind step73's equivariant CEM (cf. tests/test_step73_planner_equivariance.py, link iii): an
# optimizer that is equivariant-by-construction, started from a symmetric initialization, stays exactly equivariant
# when the model is exactly equivariant and the cost is invariant. step73 achieved this for SO(2) via isotropic
# covariance + scene-covariant (rotated-noise) CEM; for the discrete cyclic group Z_N the *cleanest* equivariant
# optimizer is plain projected gradient descent from u=0 — no stochastic noise to rotate, so nothing can break the
# symmetry. Concretely, writing the shift S = roll(., s):
#   * the WM is exactly equivariant:        phi_hat(S x, S u) = S phi_hat(x, u)              (L96ControlledConv)
#   * the target x_target = (F - mu)/sd is a CONSTANT vector, so S x_target = x_target;
#   * hence the rollout cost J(u; x0) = sum_t || x_hat_t - x_target ||^2 is JOINTLY Z_N-invariant:
#         J(S u; S x0) = J(u; x0)                                                            (sum over sites)
#   * therefore the gradient is equivariant:  grad_u J(S u; S x0) = S grad_u J(u; x0)        (S is a permutation
#         == orthogonal, and it commutes with the per-site clamp and the elementwise GD update);
#   * starting from u^(0) = 0 (which satisfies S.0 = 0), by induction every projected-GD iterate commutes with S, so
#         plan(S x0) = S plan(x0)   EXACTLY (to float round-off).
# We use plain fixed-step GD + a per-coordinate clamp |u| <= u_max (the box bound is permutation-equivariant under Z_N,
# unlike SO(2) where a box would break rotation symmetry and a DISK bound is required) so the iteration is a pure
# elementwise/linear map that provably commutes with the shift; no Adam state, no momentum, no stochasticity.
# Note (honest): gradients through a chaotic rollout amplify with H (the same lambda_1 the Phase-2 certificate reads),
# so this is kept to a MODEST horizon — that "horizon matters" tension is itself part of the co-demonstration story.
# --------------------------------------------------------------------------------------------------------------- #
def plan_control(model, x0, mu, sd, H: int, u_max: float = 1.0, n_iter: int = 40, lr: float = 0.1,
                 seed: int = 0) -> torch.Tensor:
    r"""Plan a chaos-suppressing control sequence $u_{0:H}$ for the action-conditioned WM, **exactly**
    $\mathbb{Z}_N$-orbit-equivariant: shifting the initial condition shifts the plan, ``plan(roll(x0, s)) ==
    roll(plan(x0), s)`` to float round-off.

    A deterministic projected-gradient planner started from the $\mathbb{Z}_N$-invariant zero control. The predicted
    rollout in the WM's **normalized** frame, $z_0=(x_0-\mu)/\sigma$, $z_{t+1}=\hat\phi(z_t,u_t)$, is scored against the
    unstable uniform fixed point $x_i=F$ — i.e. $x_{\rm target}=(F-\mu)/\sigma$ (a **constant** vector) — by the
    $\mathbb{Z}_N$-**invariant** cost $J=\sum_{t=1}^{H}\lVert z_t-x_{\rm target}\rVert^2$. Because $\hat\phi$ is exactly
    equivariant and $J$ is invariant, $\nabla_u J(Su;Sx_0)=S\,\nabla_u J(u;x_0)$ for any cyclic shift $S$; starting at
    $u\equiv 0$ (which is shift-invariant) every clamped GD step commutes with $S$, so the returned plan is exactly
    orbit-equivariant by construction. The control is clamped to $\lvert u\rvert\le u_{\max}$ each step (a per-site box,
    which **is** $\mathbb{Z}_N$-equivariant).

    Args:
        model: trained (or untrained — equivariant by construction) WM, ``model(z, u) -> z_next`` in normalized coords.
        x0: ``(N,)`` **RAW** (un-normalized) initial state; normalized internally by ``(x0 - mu) / sd``.
        mu, sd: ``(N,)`` normalization stats (from :func:`collect_data`).
        H: planning horizon (number of control steps; returns ``(H, N)``). Keep modest — see the chaotic-gradient note.
        u_max: per-coordinate control bound (box, $\mathbb{Z}_N$-equivariant). n_iter, lr: GD iterations and step size.
        seed: accepted for API symmetry / determinism; the planner is deterministic (init $u=0$), so it only pins the
            RNG state and does not affect the result.

    Returns:
        ``(H, N)`` ``float64`` control sequence, clamped to $[-u_{\max}, u_{\max}]$, exactly $\mathbb{Z}_N$-equivariant.

    Shapes: ``x0: (N,)`` raw -> ``u: (H, N)`` (normalized-frame rollout internally; controls live in raw forcing units).
    """
    torch.manual_seed(seed)                                    # determinism only; planner init is the fixed u=0
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    N = x0.shape[-1]
    z0 = ((x0.to(DTYPE) - mu) / sd).detach()                   # normalized initial state (no grad through x0)
    x_target = ((F_FORCE - mu) / sd).detach()                  # constant target = uniform fixed point x_i = F

    u = torch.zeros(H, N, dtype=DTYPE, requires_grad=True)     # Z_N-invariant init: S . 0 = 0
    for _ in range(n_iter):
        z = z0.clone()
        cost = z.new_zeros(())
        for t in range(H):
            z = model(z, u[t])                                 # one WM step under the planned control u_t
            cost = cost + ((z - x_target) ** 2).sum()          # Z_N-invariant per-step cost (sum over sites)
        (grad,) = torch.autograd.grad(cost, u)                 # grad_u J; equivariant because phi_hat & J are
        with torch.no_grad():
            u -= lr * grad                                     # fixed-step GD: a pure elementwise/linear update
            u.clamp_(-u_max, u_max)                            # per-site box bound (Z_N-equivariant)
    return u.detach()


def rollout_cost(model, x0, mu, sd, u: torch.Tensor) -> float:
    r"""Open-loop predicted chaos-suppression cost of a control sequence ``u`` under the WM, in normalized coords:
    $\sum_{t=1}^{H}\lVert z_t-x_{\rm target}\rVert^2$ with $z_0=(x_0-\mu)/\sigma$, $z_{t+1}=\hat\phi(z_t,u_t)$ and the
    constant target $x_{\rm target}=(F-\mu)/\sigma$. Used to compare a plan against the zero control (smoke)."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    with torch.no_grad():
        z = ((x0.to(DTYPE) - mu) / sd)
        x_target = (F_FORCE - mu) / sd
        cost = 0.0
        for t in range(u.shape[0]):
            z = model(z, u[t].to(DTYPE))
            cost += float(((z - x_target) ** 2).sum())
    return cost


# --------------------------------------------------------------------------------------------------------------- #
# Phase 4 — a CLOSED-LOOP MPC runner on the TRUE controlled dynamics, and the measurement that the equivariant
# planner gives ORBIT-FLAT control (the *configuration* axis of the co-demonstration). At each MPC step we re-plan
# from the current raw state (receding horizon), apply ONLY the first control u_0 to the TRUE rk4_controlled map, and
# record the realized state. Because BOTH the planner and the true dynamics are exactly Z_N-equivariant, a cyclic
# shift of the initial condition produces an exactly shifted closed-loop trajectory and an exactly shifted applied-
# control sequence -- so the realized cost (a shift-invariant sum over sites) is INVARIANT along the orbit: the
# closed-loop control is "orbit-flat" to machine precision (cost ratio -> 1.000, control mismatch ~ float round-off).
# Concretely, writing the shift S = roll(., s):
#   * plan(S x0) = S plan(x0)            (plan_control, exactly Z_N-orbit-equivariant)  => u_0(S x0) = S u_0(x0)
#   * rk4_controlled(S x, S u) = S rk4_controlled(x, u)   (true field is jointly Z_N-equivariant in (x, u))
#   * by induction every realized state x_t(S x0) = S x_t(x0), and every applied control u^{(t)}(S x0) = S u^{(t)}(x0);
#   * the per-state target is F * 1 (a CONSTANT, S-invariant vector), so ||x_t - F 1|| is invariant under S
#         => cost(S x0) = cost(x0) EXACTLY (to float round-off).
# This exactness is independent of u_max and of whether the WM is trained: it is a structural symmetry statement.
# u_max only sets how hard the planner can push (control AUTHORITY) -- we default it to 4.0 (not 1.0) so the closed
# loop can MEANINGFULLY suppress the chaos (visible cost drop vs no control), which the later decision contrast needs.
# --------------------------------------------------------------------------------------------------------------- #
def closed_loop(model, x0, mu, sd, H: int, n_steps: int, u_max: float = 4.0, n_iter: int = 30, lr: float = 0.1,
                seed: int = 0) -> dict:
    r"""Closed-loop model-predictive control on the TRUE controlled Lorenz-96 dynamics, driving the state toward the
    unstable uniform fixed point $x_i=F$ (chaos suppression). Receding-horizon MPC: starting from the **raw** state
    ``x0``, for each of ``n_steps`` we re-plan a length-``H`` control sequence with :func:`plan_control` from the
    *current* state, apply ONLY the first control $u_0$ to the TRUE map :func:`rk4_controlled` (zero-order hold), advance
    the realized state, and record it.

    Because both :func:`plan_control` and :func:`rk4_controlled` are exactly $\mathbb{Z}_N$-equivariant, the realized
    closed-loop trajectory is exactly orbit-equivariant in ``x0`` (see :func:`orbit_flatness`).

    Args:
        model: action-conditioned WM, ``model(z, u) -> z_next`` in normalized coords (equivariant by construction).
        x0: ``(N,)`` **RAW** initial state.  mu, sd: ``(N,)`` normalization stats (passed through to the planner).
        H: per-step planning horizon.  n_steps: number of closed-loop (apply-first-control) MPC steps.
        u_max, n_iter, lr, seed: forwarded to :func:`plan_control` at every MPC step.

    Returns:
        dict with ``"traj"`` ``(n_steps+1, N)`` realized RAW states (row 0 = ``x0``), ``"controls"`` ``(n_steps, N)`` the
        first-controls actually applied, and ``"cost"`` (float) = time-averaged $\lVert x_t - F\mathbf 1\rVert$ over the
        realized trajectory (RAW coords, target $=F$ uniform, averaged over all ``n_steps+1`` realized states).
    """
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    N = x0.shape[-1]
    x_cur = x0.to(DTYPE).clone()
    target = F_FORCE * torch.ones(N, dtype=DTYPE)              # uniform fixed point x_i = F, in RAW coords

    traj = torch.empty(n_steps + 1, N, dtype=DTYPE)
    controls = torch.empty(n_steps, N, dtype=DTYPE)
    traj[0] = x_cur
    for t in range(n_steps):
        u = plan_control(model, x_cur, mu, sd, H, u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)   # (H, N)
        u0 = u[0]                                              # apply ONLY the first control (receding horizon)
        x_cur = rk4_controlled(x_cur, u0).detach()            # advance the TRUE controlled dynamics one step
        traj[t + 1] = x_cur
        controls[t] = u0

    # Time-averaged distance to the uniform target over the realized trajectory (all n_steps+1 states), RAW coords.
    cost = float((traj - target).norm(dim=-1).mean())
    return {"traj": traj, "controls": controls, "cost": cost}


def orbit_flatness(model, x0, mu, sd, H: int, n_steps: int, s: int, u_max: float = 4.0, n_iter: int = 30,
                   lr: float = 0.1, seed: int = 0) -> dict:
    r"""Measure that the closed-loop control is **orbit-flat**: run :func:`closed_loop` from ``x0`` and from the cyclic
    shift ``roll(x0, s)``, and quantify how invariant the realized cost is along the $\mathbb{Z}_N$ orbit.

    For an exactly $\mathbb{Z}_N$-equivariant planner on the exactly equivariant true dynamics, the rolled run is the
    cyclic shift of the original run: applied controls satisfy $u^{(t)}(\mathrm{roll}(x_0,s))=\mathrm{roll}(u^{(t)}(x_0),
    s)$ and, since the cost is a sum over sites against the shift-invariant target $F\mathbf 1$, the realized cost is
    invariant ($\text{ratio}\to 1.000$). The ``control_mismatch`` is then $\sim$ machine precision.

    Returns:
        dict with ``"ratio"`` = ``cost(roll x0) / cost(x0)``, ``"control_mismatch"`` = $\max_t \lVert u^{(t)}(\mathrm{
        roll}(x_0,s)) - \mathrm{roll}(u^{(t)}(x_0), s)\rVert$, and ``"cost_x0"`` / ``"cost_rolled"`` (the two raw costs).
    """
    out0 = closed_loop(model, x0, mu, sd, H, n_steps, u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)
    out_s = closed_loop(model, torch.roll(x0.to(DTYPE), s), mu, sd, H, n_steps,
                        u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)

    # Per-step applied-control mismatch: the rolled run's controls vs the (shifted) original run's controls.
    rolled_ctrl = torch.roll(out0["controls"], s, dims=-1)    # (n_steps, N) -> roll each applied u_0 by s sites
    control_mismatch = float((out_s["controls"] - rolled_ctrl).norm(dim=-1).max())

    cost_x0 = out0["cost"]; cost_rolled = out_s["cost"]
    ratio = cost_rolled / cost_x0
    return {"ratio": ratio, "control_mismatch": control_mismatch, "cost_x0": cost_x0, "cost_rolled": cost_rolled}


def _smoke_phase4() -> None:
    r"""Train a small equivariant conv WM, pick an on-attractor state, and run the CLOSED-LOOP MPC on the TRUE
    controlled dynamics. Report the controlled realized cost vs the uncontrolled (zero-control) realized cost over the
    same horizon, and the suppression ratio (controlled / uncontrolled). Not a pytest test (it trains)."""
    torch.manual_seed(0)
    N = 12
    data = collect_data(N=N, n_traj=3000, K=3, seed=0)
    model, mu, sd, relmse = train_wm("conv", N, data, seed=0, epochs=25, K=3)
    # On-attractor initial state (raw): a mid-trajectory point of the true uncontrolled field.
    traj = step74.attractor_traj(N, 400, 0, "cpu").double()
    x0 = traj[len(traj) // 2].clone()

    H, n_steps = 8, 40
    out = closed_loop(model, x0, mu, sd, H=H, n_steps=n_steps, u_max=4.0)
    cost_ctrl = out["cost"]

    # Uncontrolled baseline: roll the SAME true dynamics from x0 with zero control over the same horizon.
    target = F_FORCE * torch.ones(N, dtype=DTYPE)
    x = x0.to(DTYPE).clone()
    unc = torch.empty(n_steps + 1, N, dtype=DTYPE); unc[0] = x
    for t in range(n_steps):
        x = rk4_controlled(x, torch.zeros(N, dtype=DTYPE))
        unc[t + 1] = x
    cost_unc = float((unc - target).norm(dim=-1).mean())

    # Sanity that the planner is FAITHFUL (the modest suppression is a horizon/geometry limit, not a transfer failure):
    # compare the WM's OWN predicted open-loop cost under the plan vs zero control from x0.
    u_plan = plan_control(model, x0, mu, sd, H=H, u_max=4.0)
    pred_zero = rollout_cost(model, x0, mu, sd, torch.zeros(H, N, dtype=DTYPE))
    pred_plan = rollout_cost(model, x0, mu, sd, u_plan)

    supp = cost_ctrl / cost_unc
    print(f"[step79 phase4 smoke] N={N} conv WM one-step relMSE {relmse:.2e}", file=sys.stderr)
    print(f"[step79 phase4 smoke] H={H} n_steps={n_steps} u_max=4.0  realized cost (RAW ||x - F*1||, time-avg):",
          file=sys.stderr)
    print(f"[step79 phase4 smoke]   controlled (closed-loop MPC) {cost_ctrl:.4f}  vs  uncontrolled (u=0) {cost_unc:.4f}",
          file=sys.stderr)
    print(f"[step79 phase4 smoke]   WM-predicted open-loop (one plan from x0): zero {pred_zero:.3f} -> planned "
          f"{pred_plan:.3f} ({100.0 * (1.0 - pred_plan / pred_zero):.1f}% predicted reduction)", file=sys.stderr)
    print(f"[step79 phase4 smoke]   suppression ratio (controlled/uncontrolled) = {supp:.3f}  "
          f"({100.0 * (1.0 - supp):.1f}% lower)" if supp < 1 else
          f"[step79 phase4 smoke]   suppression ratio (controlled/uncontrolled) = {supp:.3f}  "
          f"(WARN: controlled is HIGHER)", file=sys.stderr)
    # Orbit-flatness side-check (the configuration axis): control mismatch + cost ratio along the Z_N orbit.
    of = orbit_flatness(model, x0, mu, sd, H=H, n_steps=n_steps, s=3, u_max=4.0)
    print(f"[step79 phase4 smoke]   orbit-flatness: control_mismatch {of['control_mismatch']:.2e}  "
          f"cost ratio {of['ratio']:.6f}", file=sys.stderr)


def _smoke_phase3() -> None:
    r"""Train a small equivariant conv WM, pick an on-attractor state, and confirm the planner's control REDUCES the
    open-loop predicted chaos-suppression cost vs the zero control. Not a pytest test (it trains)."""
    torch.manual_seed(0)
    N = 12
    data = collect_data(N=N, n_traj=2000, K=3, seed=0)
    model, mu, sd, relmse = train_wm("conv", N, data, seed=0, epochs=20, K=3)
    # On-attractor initial state (raw): a mid-trajectory point of the true uncontrolled field.
    traj = step74.attractor_traj(N, 400, 0, "cpu").double()
    x0 = traj[len(traj) // 2].clone()

    H = 6
    u_zero = torch.zeros(H, N, dtype=DTYPE)
    u_plan = plan_control(model, x0, mu, sd, H=H, u_max=1.0, n_iter=40, lr=0.1, seed=0)
    cost_zero = rollout_cost(model, x0, mu, sd, u_zero)
    cost_plan = rollout_cost(model, x0, mu, sd, u_plan)
    print(f"[step79 phase3 smoke] N={N} conv WM one-step relMSE {relmse:.2e}", file=sys.stderr)
    print(f"[step79 phase3 smoke] H={H} predicted cost: zero-control {cost_zero:.4f} -> planned {cost_plan:.4f} "
          f"({100.0 * (1.0 - cost_plan / cost_zero):.1f}% reduction)", file=sys.stderr)
    assert cost_plan < cost_zero, \
        f"planner did not reduce predicted cost: planned {cost_plan:.4f} >= zero-control {cost_zero:.4f}"
    print("[step79 phase3 smoke] PASS: the equivariant planner reduces the open-loop predicted chaos-suppression "
          "cost vs the zero control.", file=sys.stderr)


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
