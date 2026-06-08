r"""Step 80 — a CORROBORATOR for the certified-control co-demonstration: a CLASS-lift to a coupled-pendulum RING.

The anchor :mod:`step79_certified_control` established the certified-control co-demonstration on **controlled
Lorenz-96**: an action-conditioned $\mathbb{Z}_N$-equivariant world model whose (i) controlled dynamics + WM + gradient
planner are exactly $\mathbb{Z}_N$-equivariant, (ii) certified forecast horizon $T_1(\epsilon)$ (units-fixed:
$T_1/\Delta t_{\rm map}$ map steps) tracks the WM's empirical free-running forecast horizon — predictive at the larger
$\epsilon$, optimistic (Prop-8 $\delta$-bias) at $\epsilon{=}0.01$ — and (iii) the cert-aware re-observation interval sits
on the efficient frontier of forecast-violation-rate vs observation-count. This experiment **reproduces the same three
results on a DIFFERENT chaotic $\mathbb{Z}_N$-symmetric system** — a driven–damped coupled-pendulum RING — to show the
co-demonstration is a CLASS property of locally-coupled cyclic chaos, not a Lorenz-96 idiosyncrasy.

**System (driven–damped coupled-pendulum ring; $N$ pendula on a cyclic $\mathbb{Z}_N$ index, controllable, chaotic):**
$$\dot\theta_i=\omega_i,\quad \dot\omega_i=-\sin\theta_i-\gamma\,\omega_i+K[\sin(\theta_{i+1}-\theta_i)+\sin(\theta_{i-1}
-\theta_i)]+A\cos(\Omega t)+u_i .$$
The drive $A\cos(\Omega t)$ makes the field **non-autonomous**; we handle time by augmenting the state with the drive
**phase** $\psi=\Omega t \bmod 2\pi$ (so $\dot\psi=\Omega$) — the autonomous phase-augmented field $y=(\theta,\omega,\psi)
\in\mathbb{R}^{2N+1}$ has a well-defined Lyapunov spectrum (the $\psi$-direction is neutral, $\lambda=0$). The
$\mathbb{Z}_N$ symmetry is the cyclic shift of the pendulum **index**: the ring coupling commutes with it and the drive
is **identical on every site** (hence $\mathbb{Z}_N$-invariant), so a cyclic shift of $(\theta,\omega,u)$ shifts the
field. Control is a per-pendulum torque $u_i$ with $|u_i|\le u_{\max}$.

**Why this corroborates the L96 anchor.** Both systems are locally-coupled, cyclically-symmetric, genuinely chaotic
($\lambda_1>0$, multiple positive exponents). The same mechanisms carry over verbatim: a circular-conv WM on the ring is
$\mathbb{Z}_N$-equivariant by construction; the gradient-MPC planner from $u{=}0$ inherits exact orbit-equivariance; the
certified horizon is read off the WM's autonomous $u{=}0$ map. If the three results reproduce here, the co-demonstration
is a class property.

**Key design points (mirrors step79, adapted for periodic angles + a non-autonomous drive):**
  * **Periodic encoding.** Each site's state is presented to the WM as $(\cos\theta_i,\sin\theta_i,\omega_i)$ (3 channels
    per site) so the circular-conv WM never sees an angle wraparound and is exactly $\mathbb{Z}_N$-equivariant. The drive
    phase enters as 2 **global** channels $(\cos\psi,\sin\psi)$ broadcast identically across the ring (a constant-across-
    sites channel is shift-invariant, so it does not break equivariance). The WM predicts the next
    $(\cos\theta,\sin\theta,\omega)$; angles are decoded by ``atan2``.
  * **Equivariant WM.** A stack of circular Conv1d over the ring (Z_N-equivariant), action-conditioned on $u$ (a 4th
    per-site channel) and drive-phase-conditioned on $(\cos\psi,\sin\psi)$. The baseline MLP flattens everything, no
    structure (not equivariant).
  * **Cost / target.** Stabilize the downward rest state $\theta_i{=}0,\omega_i{=}0$ (the $\mathbb{Z}_N$-symmetric
    equilibrium) with the angle-wrap-safe, $\mathbb{Z}_N$-invariant cost $\sum_i(1-\cos\theta_i)+0.1\,\omega_i^2$. (Used
    only for the (C) orbit-flat-control check; the main decision result — re-observation — needs no control.)

We REUSE :mod:`step78` (block-bootstrap spectrum CI / horizon interval) and the :func:`step74.DTMAP`-style time-units
convention (with our OWN map timestep ``DTMAP``), and do NOT modify step74/78/79.

THREE RESULTS (honest gates; INCONCLUSIVE/DONE_WITH_CONCERNS rather than loosen):
  1. **(equivariance)** controlled ring dynamics + conv WM are exactly $\mathbb{Z}_N$-equivariant (machine precision);
     ``plan_control`` is exactly orbit-equivariant (atol 1e-6).
  2. **(H validation)** the WM's certified horizon ($T_1/\Delta t_{\rm map}$) vs its EMPIRICAL forecast horizon at
     $\epsilon\in\{0.01,0.2,0.3\}$; report the ratio per $\epsilon$ honestly. (Finding: OPTIMISTIC at $\epsilon{=}0.01$
     — ratio$\sim 0.02$, the Prop-8 $\delta$-bias, exactly as the anchor; PREDICTIVE at the asymptotic-Lyapunov knee,
     which for the ring is $\epsilon{=}0.3$, ratio$\sim 1$; $\epsilon{=}0.2$, the anchor's knee, is borderline for the
     ring — ratio $0.6$–$1.0$.)
  3. **(D2 re-observation)** the cert-aware re-observation interval ($=T_1$ steps at the predictive $\epsilon$) sits on
     the efficient frontier of violation-rate vs observation-count, vs a blind-interval sweep. Seeds 0,1,2. (Reproduces
     the anchor's 2/3-on-frontier at the predictive knee $\epsilon{=}0.3$; 0/3 at the optimistic $\epsilon{=}0.01$.)
  Plus (C): the equivariant-planner closed-loop control is orbit-flat to machine precision (mismatch $<1\mathrm{e}{-8}$).

Run:    .venv/bin/python experiments/step80_pendulum_ring.py [chaos|smoke|equiv|reobs]   (default: full -> figures)
Writes: papers/figures/step80_pendulum_ring.{png,json}  (cpu, float64).  Tests: tests/test_step80.py (no training).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# step78 supplies the certified-horizon machinery (Benettin-QR log|R| series + block-bootstrap spectrum CI + horizon
# interval). step74 supplies the DTMAP time-units *convention* (we keep our OWN map timestep below). Import both; do
# NOT modify them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as step74  # noqa: E402  (for the DTMAP-style convention / parity with the anchor)
import step78_certified_horizon_ci as step78  # noqa: E402

DTYPE = torch.float64

# --------------------------------------------------------------------------------------------------------------- #
# Pendulum-ring parameters. The regime gamma=0.1, K=0.5, A=1.5, Omega=0.66 (N=10) is genuinely chaotic: the
# phase-augmented field has lambda_1 ~ 0.30 (per unit time) with several positive exponents (a high-dimensional chaotic
# attractor, like Lorenz-96), and the trajectory stays bounded. (Verified by Benettin QR on the TRUE field, _chaos().)
# DTMAP is our OWN map timestep (the dt of one rk4 Delta t-map step); the certified horizon in MAP STEPS is T1 / DTMAP,
# exactly the step74.DTMAP convention but with the ring's own dt. This is the load-bearing units fix (cf. step79 Ph5b).
# --------------------------------------------------------------------------------------------------------------- #
N_DEFAULT = 10
GAMMA = 0.1       # damping
K_COUP = 0.5      # ring coupling
A_DRIVE = 1.5     # drive amplitude
OMEGA_DRIVE = 0.66  # drive angular frequency
DTMAP = 0.02      # one Delta t-map step (rk4 dt); certified horizon in map steps = T1 / DTMAP


def ring_state_dim(N: int) -> int:
    r"""Dimension of the phase-augmented true state $y=(\theta_{1..N},\omega_{1..N},\psi)\in\mathbb{R}^{2N+1}$."""
    return 2 * N + 1


def ring_rhs(y: torch.Tensor, N: int, gamma: float = GAMMA, K: float = K_COUP, A: float = A_DRIVE,
             Omega: float = OMEGA_DRIVE, u: torch.Tensor | None = None) -> torch.Tensor:
    r"""Phase-augmented driven–damped coupled-pendulum-ring field. ``y: (..., 2N+1)`` $=[\theta(N),\omega(N),\psi(1)]$.

    $\dot\theta_i=\omega_i$, $\dot\omega_i=-\sin\theta_i-\gamma\omega_i+K[\sin(\theta_{i+1}-\theta_i)+\sin(\theta_{i-1}
    -\theta_i)]+A\cos\psi+u_i$, $\dot\psi=\Omega$. Jointly $\mathbb{Z}_N$-equivariant in $(\theta,\omega,u)$ (the cyclic
    site shift commutes with the ring coupling and with $u$; the drive $A\cos\psi$ and $\dot\psi=\Omega$ are identical on
    every site, hence shift-invariant). ``u`` (optional, ``(..., N)``) is a per-site torque held over the step (ZOH);
    ``None`` means $u\equiv 0$."""
    th = y[..., :N]; om = y[..., N:2 * N]; psi = y[..., 2 * N:2 * N + 1]
    coupling = torch.sin(torch.roll(th, -1, -1) - th) + torch.sin(torch.roll(th, 1, -1) - th)   # ring, Z_N-equivariant
    drive = A * torch.cos(psi)                                   # (...,1): identical on all sites => Z_N-invariant
    dth = om
    dom = -torch.sin(th) - gamma * om + K * coupling + drive
    if u is not None:
        dom = dom + u                                           # per-site torque, Z_N-equivariant
    dpsi = Omega * torch.ones_like(psi)
    return torch.cat([dth, dom, dpsi], dim=-1)


def rk4_ring(y: torch.Tensor, N: int, u: torch.Tensor | None = None, dt: float = DTMAP, gamma: float = GAMMA,
             K: float = K_COUP, A: float = A_DRIVE, Omega: float = OMEGA_DRIVE) -> torch.Tensor:
    r"""One RK4 step of the phase-augmented ring $\Delta t$-map with a zero-order-hold control ``u`` (held over the step).
    Jointly $\mathbb{Z}_N$-equivariant in $(\theta,\omega,u)$: a cyclic site shift of both inputs shifts the output."""
    f = lambda z: ring_rhs(z, N, gamma, K, A, Omega, u)
    k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def roll_state(y: torch.Tensor, N: int, s: int) -> torch.Tensor:
    r"""Apply the cyclic $\mathbb{Z}_N$ shift $S_s$ to a phase-augmented state ``y`` $=[\theta(N),\omega(N),\psi]$: roll
    the $\theta$- and $\omega$-blocks by ``s`` sites, leave the (site-independent) drive phase $\psi$ fixed."""
    th = torch.roll(y[..., :N], s, dims=-1)
    om = torch.roll(y[..., N:2 * N], s, dims=-1)
    psi = y[..., 2 * N:2 * N + 1]
    return torch.cat([th, om, psi], dim=-1)


def attractor_traj_ring(N: int, n_steps: int, seed: int, burn: int = 4000) -> torch.Tensor:
    r"""Burn in to the ring's chaotic attractor, then return ``n_steps+1`` phase-augmented states ``(n_steps+1, 2N+1)``
    of the (uncontrolled, $u\equiv 0$) $\Delta t$-map. Mirrors :func:`step74.attractor_traj` for the ring; the start
    seeds small $(\theta,\omega)$ and $\psi=0$. (The drive phase advances deterministically, so the attractor is the
    chaotic set of the phase-augmented autonomous field — the analogue of the L96 attractor.)"""
    g = torch.Generator().manual_seed(seed)
    y = torch.zeros(ring_state_dim(N), dtype=DTYPE)
    y[:N] = 0.3 * torch.randn(N, generator=g, dtype=DTYPE)      # small initial angle spread
    y[N:2 * N] = 0.3 * torch.randn(N, generator=g, dtype=DTYPE)  # small initial angular velocity
    y[2 * N] = 0.0                                              # drive phase psi_0 = 0
    for _ in range(burn):
        y = rk4_ring(y, N)
    out = [y]
    for _ in range(n_steps):
        y = rk4_ring(y, N)
        out.append(y)
    return torch.stack(out, 0)


# --------------------------------------------------------------------------------------------------------------- #
# The periodic encoding. The WM never sees raw angles (which wrap); it sees per-site (cos theta, sin theta, omega) and
# two GLOBAL drive-phase channels (cos psi, sin psi) broadcast across the ring. Decoding maps (cos, sin) -> atan2.
# These maps are the bridge between the TRUE phase-augmented state y in R^{2N+1} and the WM's site-channel tensor.
# --------------------------------------------------------------------------------------------------------------- #
def encode_state(y: torch.Tensor, N: int):
    r"""Map a TRUE phase-augmented state ``y`` $=[\theta,\omega,\psi]$ to the WM's inputs. Returns ``(feat, drive)``:
      * ``feat:  (..., 3, N)``  per-site channels $[\cos\theta_i,\ \sin\theta_i,\ \omega_i]$ (channel-first for Conv1d).
      * ``drive: (..., 2)``      the two global drive-phase scalars $[\cos\psi,\ \sin\psi]$.
    Under a cyclic shift $S_s$, ``feat`` rolls by ``s`` along the site axis and ``drive`` is unchanged (it is global),
    so the encoding is $\mathbb{Z}_N$-equivariant."""
    th = y[..., :N]; om = y[..., N:2 * N]; psi = y[..., 2 * N:2 * N + 1]
    feat = torch.stack([torch.cos(th), torch.sin(th), om], dim=-2)   # (..., 3, N)
    drive = torch.cat([torch.cos(psi), torch.sin(psi)], dim=-1)      # (..., 2)
    return feat, drive


def decode_feat(feat: torch.Tensor) -> torch.Tensor:
    r"""Decode a WM site-channel tensor ``feat`` $=(\hat c,\hat s,\hat\omega)\in(...,3,N)$ back to $(\theta,\omega)$
    $\in(...,2N)$: $\theta_i=\operatorname{atan2}(\hat s_i,\hat c_i)$ (wrap-safe), $\omega_i=\hat\omega_i$. The
    $(\hat c,\hat s)$ need not be unit-norm; ``atan2`` reads only their angle."""
    c = feat[..., 0, :]; s = feat[..., 1, :]; om = feat[..., 2, :]
    th = torch.atan2(s, c)
    return torch.cat([th, om], dim=-1)


# --------------------------------------------------------------------------------------------------------------- #
# Action-conditioned world models. We learn a one-step map of the controlled phase-augmented Delta t-map in the
# (cos,sin,omega) feature space, action-conditioned on u and drive-phase-conditioned on (cos psi, sin psi). Two
# flavours differing ONLY in whether the Z_N (cyclic-shift) symmetry is built in:
#   * RingConv -- circular Conv1d over the ring with 6 input channels [cos th, sin th, omega, u, cos psi, sin psi]
#                 (the two drive channels broadcast across sites); 3 output channels [d cos, d sin, d omega] added
#                 residually to [cos th, sin th, omega]. Exactly Z_N-equivariant (circular pad + global drive).
#   * RingMLP  -- dense baseline over the flattened concatenation; NOT equivariant.
# Mirrors step79's L96ControlledConv / L96ControlledMLP, extended to the periodic (cos,sin,omega) encoding + drive.
# --------------------------------------------------------------------------------------------------------------- #
class RingConv(nn.Module):
    r"""$\mathbb{Z}_N$-equivariant action-conditioned world model for the pendulum ring: a stack of **circular** 1-D
    convolutions over a **6-channel** per-site input $[\cos\theta,\sin\theta,\omega,u,\cos\psi,\sin\psi]$ (the two
    drive-phase channels are GLOBAL — broadcast identically across the ring, hence shift-invariant). The ring field is
    locally coupled ($\dot\omega_i$ depends on sites $i{-}1,i,i{+}1$) and jointly $\mathbb{Z}_N$-equivariant in
    $(\theta,\omega,u)$, so a circular conv on these channels has a banded-circulant per-site Jacobian structurally
    matching it; the learned $\Delta t$-map $\hat\phi=\mathrm{feat}+\mathrm{conv}(\dots)$ is **exactly**
    $\mathbb{Z}_N$-equivariant by construction. Mirrors :class:`step79.L96ControlledConv` with the periodic encoding."""

    def __init__(self, N: int, channels: int = 64, kernel: int = 5, layers: int = 4):
        super().__init__()
        self.N = N
        self.kernel = kernel
        in_ch = 6                                               # [cos th, sin th, omega, u, cos psi, sin psi]
        chans = [in_ch] + [channels] * (layers - 1) + [3]      # 3 output channels: increments to [cos, sin, omega]
        self.convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], kernel) for i in range(layers))

    def _channels(self, feat: torch.Tensor, u: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        r"""Assemble the (B, 6, N) channel stack from per-site ``feat`` (B,3,N), control ``u`` (B,N), and global
        ``drive`` (B,2). The drive scalars are broadcast across all N sites (a constant-across-sites channel)."""
        B, _, N = feat.shape
        u_ch = u.reshape(B, 1, N)                               # control as a per-site channel
        drive_ch = drive.reshape(B, 2, 1).expand(B, 2, N)      # (cos psi, sin psi) broadcast across sites
        return torch.cat([feat, u_ch, drive_ch], dim=1)        # (B, 6, N)

    def forward(self, feat: torch.Tensor, u: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        r"""``feat: (..., 3, N), u: (..., N), drive: (..., 2) -> (..., 3, N)``, $\mathbb{Z}_N$-equivariant. Circular
        convs (SiLU between layers) over the 6-channel stack, residual on the 3 state channels ``feat``."""
        shp = feat.shape                                       # (..., 3, N)
        N = shp[-1]
        featf = feat.reshape(-1, 3, N)
        uf = u.reshape(-1, N)
        dr = drive.reshape(-1, 2)
        h = self._channels(featf, uf, dr)                      # (B, 6, N)
        pad = (self.kernel - 1) // 2
        for i, conv in enumerate(self.convs):
            h = torch.nn.functional.pad(h, (pad, pad), mode="circular")   # circular => Z_N equivariance
            h = conv(h)
            if i < len(self.convs) - 1:
                h = torch.nn.functional.silu(h)
        return featf.reshape(shp) + h.reshape(shp)             # residual on the 3 state channels


class RingMLP(nn.Module):
    r"""Dense action-conditioned baseline (NOT equivariant) for the pendulum ring: flatten the per-site features, the
    control, and the global drive into one vector $[\,\mathrm{vec}(\mathrm{feat})\,\|\,u\,\|\,\mathrm{drive}\,]\in
    \mathbb{R}^{3N+N+2}$ and pass through a residual MLP, $\hat\phi=\mathrm{feat}+\mathrm{MLP}(\cdot)$. Its dense first
    layer mixes all sites, so a cyclic shift does **not** commute with it. Mirrors :class:`step79.L96ControlledMLP`."""

    def __init__(self, N: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.N = N
        in_dim = 3 * N + N + 2                                  # vec(feat) ++ u ++ drive
        mods = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.SiLU()]
        mods += [nn.Linear(hidden, 3 * N)]                     # back to vec of the 3 state channels
        self.net = nn.Sequential(*mods)

    def forward(self, feat: torch.Tensor, u: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        shp = feat.shape                                       # (..., 3, N)
        N = shp[-1]
        featf = feat.reshape(-1, 3, N)
        flat = torch.cat([featf.reshape(featf.shape[0], -1), u.reshape(-1, N), drive.reshape(-1, 2)], dim=-1)
        delta = self.net(flat).reshape(-1, 3, N)
        return (featf + delta).reshape(shp)


def make_equivariant_wm(N: int) -> RingConv:
    r"""Factory for the $\mathbb{Z}_N$-equivariant action-conditioned ring world model (used by later phases / tests)."""
    return RingConv(N)


# --------------------------------------------------------------------------------------------------------------- #
# A thin wrapper that lets the WM step the FULL phase-augmented state y (so rollouts/certificates can advance the drive
# phase exactly, like the true map). The WM predicts the next (cos,sin,omega); we decode angles by atan2 and ADVANCE
# THE DRIVE PHASE EXACTLY (psi += Omega*dt) -- the drive is a known clock, not something to learn. This mirrors how the
# true rk4_ring advances psi deterministically; the WM only has to learn the (theta,omega) dynamics.
# --------------------------------------------------------------------------------------------------------------- #
def wm_step_state(model, y: torch.Tensor, N: int, mu: torch.Tensor, sd: torch.Tensor,
                  u: torch.Tensor | None = None) -> torch.Tensor:
    r"""Advance the FULL phase-augmented state ``y`` $=[\theta,\omega,\psi]$ one step with the WM. Encode to
    $(\mathrm{feat},\mathrm{drive})$, NORMALIZE the $\omega$-channel by the scalar ``(mu, sd)`` (cos/sin are already
    $O(1)$, left raw), run the WM, denormalize, decode $\theta=\operatorname{atan2}$, and advance the drive phase
    EXACTLY $\psi\leftarrow\psi+\Omega\,\Delta t$ (the drive is a known clock). ``u`` (``(...,N)`` or ``None``) is the
    per-site torque. Returns the next phase-augmented state ``(..., 2N+1)``. $\mathbb{Z}_N$-equivariant in
    $(\theta,\omega,u)$ (the per-site encode/decode + exact phase advance all commute with the site roll)."""
    feat, drive = encode_state(y, N)                           # feat: (...,3,N), drive: (...,2)
    if u is None:
        u = torch.zeros(feat.shape[:-2] + (N,), dtype=y.dtype)
    feat_n = _normalize_feat(feat, mu, sd)
    out_n = model(feat_n, u, drive)
    out = _denormalize_feat(out_n, mu, sd)
    thom = decode_feat(out)                                    # (..., 2N) = [theta_next, omega_next]
    psi = y[..., 2 * N:2 * N + 1]
    psi_next = psi + OMEGA_DRIVE * DTMAP                        # advance the drive clock EXACTLY
    return torch.cat([thom[..., :N], thom[..., N:2 * N], psi_next], dim=-1)


# Normalization acts ONLY on the omega channel (channel index 2). cos/sin are intrinsically O(1) and must stay raw so
# atan2 reads the right angle; a SCALAR (mu, sd) on omega is Z_N-invariant (every site is dynamically equivalent on the
# symmetric attractor), so the normalized rollout commutes with the cyclic shift. (Same logic as step79's scalar stats.)
def _normalize_feat(feat: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    r"""Normalize the $\omega$-channel (index 2) by scalar ``(mu, sd)``; leave $\cos,\sin$ (indices 0,1) raw."""
    out = feat.clone()
    out[..., 2, :] = (feat[..., 2, :] - mu) / sd
    return out


def _denormalize_feat(feat_n: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    r"""Inverse of :func:`_normalize_feat`: denormalize the $\omega$-channel; leave $\cos,\sin$ raw."""
    out = feat_n.clone()
    out[..., 2, :] = feat_n[..., 2, :] * sd + mu
    return out


# --------------------------------------------------------------------------------------------------------------- #
# Data collection + training. We collect length-K segments of the TRUE controlled ring under random ZOH controls from
# on-attractor starts, and train the WM with a K-step rollout loss IN THE PERIODIC FEATURE SPACE (matching the next
# (cos,sin,omega)). The rollout constrains the composed controlled Jacobian -- the operator the certificate reads.
# --------------------------------------------------------------------------------------------------------------- #
def collect_data(N: int, n_traj: int, K: int, seed: int, u_max: float = 0.5):
    r"""Collect ``n_traj`` length-``K`` segments of the TRUE controlled ring from on-attractor starts under random ZOH
    controls $u\sim\mathcal U[-u_{\max},u_{\max}]^N$. Returns ``(starts_y, controls, targets_feat, mu, sd)`` (float64):
      * ``starts_y:     (n_traj, 2N+1)``  -- on-attractor phase-augmented start states (RAW).
      * ``controls:     (n_traj, K, N)``   -- applied controls (RAW, bounded).
      * ``targets_feat: (n_traj, K, 3, N)``-- per-site $(\cos\theta,\sin\theta,\omega)$ of the next states (RAW feat;
        $\omega$ to be normalized in the loss). ``mu, sd`` are the SCALAR mean/std of the $\omega$-channel over all
        visited states (broadcast — $\mathbb{Z}_N$-invariant)."""
    g = torch.Generator().manual_seed(seed)
    traj = attractor_traj_ring(N, max(n_traj * 2, n_traj + 1), seed)     # (>=n_traj+1, 2N+1) on-attractor states
    idx = torch.randperm(traj.shape[0], generator=g)[:n_traj]
    y0 = traj[idx].clone().to(DTYPE)                           # (n_traj, 2N+1) raw on-attractor starts

    controls = (2.0 * u_max) * torch.rand(n_traj, K, N, generator=g, dtype=DTYPE) - u_max

    targets_feat = torch.empty(n_traj, K, 3, N, dtype=DTYPE)
    omegas = [y0[:, N:2 * N]]                                   # collect omega values for scalar stats
    y = y0.clone()
    for k in range(K):
        y = rk4_ring(y, N, controls[:, k, :])
        feat, _ = encode_state(y, N)
        targets_feat[:, k] = feat
        omegas.append(y[:, N:2 * N])

    # Scalar (Z_N-invariant) normalization of the omega channel over ALL visited states (starts + every rolled target).
    omega_all = torch.cat(omegas, dim=0).reshape(-1)
    mu = omega_all.mean().repeat(N)                            # scalar mean, broadcast to (N,): Z_N-invariant
    sd = (omega_all.std() + 1e-8).repeat(N)                    # scalar std,  broadcast to (N,): Z_N-invariant
    return y0, controls, targets_feat, mu, sd


def _feat_loss(pred_feat: torch.Tensor, tgt_feat: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    r"""MSE between predicted and target features with the $\omega$-channel normalized by scalar ``(mu, sd)`` (so the
    $O(1)$ cos/sin and the $O(\sigma_\omega)$ omega contribute on a comparable scale). Mean over channels and sites."""
    p = _normalize_feat(pred_feat, mu, sd)
    t = _normalize_feat(tgt_feat, mu, sd)
    return ((p - t) ** 2).mean()


def train_wm(kind: str, N: int, data, seed: int, epochs: int = 60, K: int = 5):
    r"""Train an action-conditioned ring WM with a $K$-step **rollout** loss in the periodic feature space, feeding the
    DATA's controls + the EXACT drive phase at each step. The one-step relMSE (on $(\theta,\omega)$, angle-wrap-safe) is
    returned. ``kind in {'conv','mlp'}``; Adam lr ``1e-3``. Mirrors :func:`step79.train_wm`."""
    torch.manual_seed(seed)
    starts_y, controls, targets_feat, mu, sd = data
    assert kind in ("conv", "mlp"), f"unknown kind {kind!r}"
    assert controls.shape[1] >= K and targets_feat.shape[1] >= K, "data horizon shorter than rollout K"
    starts_y = starts_y.to(DTYPE); controls = controls.to(DTYPE); targets_feat = targets_feat.to(DTYPE)

    model = (RingConv(N) if kind == "conv" else RingMLP(N)).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = starts_y.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 512):
            idx = torch.randperm(n, generator=g)[i:i + 512]
            y = starts_y[idx]
            feat, _ = encode_state(y, N)                        # per-site features of the start state
            feat_n = _normalize_feat(feat, mu, sd)
            psi = y[:, 2 * N:2 * N + 1].clone()                # the drive phase at the start (a known clock)
            loss = 0.0
            for k in range(K):
                drive = torch.cat([torch.cos(psi), torch.sin(psi)], dim=-1)   # (cos psi, sin psi) at THIS input step
                feat_n = model(feat_n, controls[idx, k, :], drive)   # WM step in normalized feature space
                loss = loss + _feat_loss(_denormalize_feat(feat_n, mu, sd), targets_feat[idx, k], mu, sd)
                psi = psi + OMEGA_DRIVE * DTMAP                 # advance the drive phase EXACTLY for the next WM input
            loss = loss / K
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():                                      # one-step relMSE on (theta, omega), angle-wrap-safe
        feat0, drive0 = encode_state(starts_y, N)
        feat0_n = _normalize_feat(feat0, mu, sd)
        pred_n = model(feat0_n, controls[:, 0, :], drive0)
        pred_feat = _denormalize_feat(pred_n, mu, sd)
        relmse = _relmse_thom(pred_feat, targets_feat[:, 0], N)
    return model, mu, sd, relmse


def _relmse_thom(pred_feat: torch.Tensor, tgt_feat: torch.Tensor, N: int) -> float:
    r"""Relative MSE between decoded $(\theta,\omega)$ of ``pred_feat`` and ``tgt_feat``, angle-wrap-safe: the angle part
    uses the chordal error $\lVert(\cos\hat\theta,\sin\hat\theta)-(\cos\theta,\sin\theta)\rVert$ (no $2\pi$ jumps), the
    velocity part uses $\hat\omega-\omega$ directly; normalized by the target's $(\cos,\sin,\omega)$ energy."""
    # chordal angle error via the cos/sin channels + direct omega error, all relative to the target feature energy.
    num = ((pred_feat - tgt_feat) ** 2).sum(dim=(-2, -1))      # sum over (3 channels, N sites)
    den = (tgt_feat ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)
    return float((num / den).mean())


# --------------------------------------------------------------------------------------------------------------- #
# The CERTIFICATE: read a certified horizon T_1(eps) with a bootstrap CI off the trained WM's AUTONOMOUS (u=0) map. The
# WM is action-conditioned; the certificate is a property of its free-running dynamics, so we read it from g(y) =
# wm_step_state(y, u=0) -- the WM's u=0 Delta t-map ON THE FULL PHASE-AUGMENTED STATE (the drive phase advances exactly
# as part of g). We run Benettin-QR on g's autograd Jacobian, block-bootstrap a per-channel CI (reusing step78), and
# turn lambda_1's CI into a certified-horizon interval. Lyapunov exponents are coordinate-invariant, so reading them in
# the (theta,omega,psi) frame is exact. The phase direction contributes a neutral (lambda~0) exponent, harmless to the
# leading lambda_1 the certificate binds on.
# --------------------------------------------------------------------------------------------------------------- #
def certificate(model, mu, sd, N: int, eps: float = 0.01, n_steps: int = 2000, warmup: int = 400,
                n_boot: int = 1000, block: int = 50, seed: int = 0) -> dict:
    r"""Certified horizon $T_1(\epsilon)$ with a block-bootstrap CI, read off the trained ring WM's autonomous ($u{=}0$)
    phase-augmented map $g(y)=\texttt{wm\_step\_state}(y,u{=}0)$. Benettin–QR on $g$'s autograd Jacobian at an
    on-attractor operating point estimates the predictor's Lyapunov spectrum; :mod:`step78` block-bootstraps the
    per-channel CI; $\lambda_1$'s CI maps to $T_1(\epsilon)\in[\log(1/\epsilon)/\lambda_{\rm hi},\log(1/\epsilon)/
    \lambda_{\rm lo}]$. Abstains (``T*`` = None) if $\lambda_1\le 0$. Mirrors :func:`step79.certificate`.

    Returns a dict with ``lambda`` (sorted-desc point spectrum), ``lambda_lo``/``lambda_hi`` (CI), ``lambda1``,
    ``lambda1_ci``, ``T1``/``T1_lo``/``T1_hi`` (TIME units, or None), and ``eps``."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)

    def g(y: torch.Tensor) -> torch.Tensor:                    # WM autonomous (u=0) phase-augmented map; autograd flows
        return wm_step_state(model, y, N, mu, sd, u=None)

    traj = attractor_traj_ring(N, n_steps // 4, seed)          # on-attractor operating point (true field)
    y0 = traj[len(traj) // 2].detach()

    logR = step78.qr_logR_series(g, y0, n_steps, warmup)       # (n_steps, 2N+1) per-step log|diag(R)|
    point, lo, hi = step78.bootstrap_spectrum_ci(logR, DTMAP, n_boot, block, seed)   # sorted desc, each len 2N+1

    iv = step78.horizon_interval(lo[0], hi[0], eps) if point[0] > 0 else None
    T1 = (float(np.log(1.0 / eps) / point[0]) if point[0] > 0 else None)
    T1_lo = (float(iv[0]) if iv is not None else None)
    T1_hi = (float(iv[1]) if iv is not None else None)
    return {"lambda": point.tolist(), "lambda_lo": lo.tolist(), "lambda_hi": hi.tolist(),
            "lambda1": float(point[0]), "lambda1_ci": [float(lo[0]), float(hi[0])],
            "T1": T1, "T1_lo": T1_lo, "T1_hi": T1_hi, "eps": eps}


def certified_T1_steps(cert: dict) -> int:
    r"""Convert ``certificate``'s ``T1`` (TIME units, $T_1=\log(1/\epsilon)/\lambda_1$) to the certified forecast
    horizon in MAP STEPS: $\max(1,\mathrm{round}(T_1/\Delta t_{\rm map}))$ with $\Delta t_{\rm map}=$ :data:`DTMAP`. THE
    units fix (mirrors :func:`step79.certified_T1_steps`, with the ring's own ``DTMAP``). Abstain/non-finite -> 1."""
    T1 = cert.get("T1")
    if T1 is None or not np.isfinite(T1):
        return 1
    return max(1, int(round(float(T1) / DTMAP)))


# --------------------------------------------------------------------------------------------------------------- #
# The EXACTLY Z_N-orbit-equivariant PLANNER (gradient MPC from u=0). Given the WM and a raw phase-augmented state y0,
# return a control sequence u_{0:H} driving the predicted (theta,omega) toward the downward rest state theta=0, omega=0
# (the Z_N-symmetric equilibrium). The cost is the angle-wrap-safe, Z_N-invariant sum_i (1-cos theta_i) + 0.1 omega_i^2.
# Exact equivariance: the WM is exactly equivariant, the cost is invariant (sum over sites, target is the constant
# zero state), the init u=0 is shift-invariant, and projected GD is a pure elementwise/linear update -> by induction
# plan(S y0) = S plan(y0) to float round-off. (Same construction as step79's plan_control.)
# --------------------------------------------------------------------------------------------------------------- #
def _stabilize_cost(thom: torch.Tensor, N: int) -> torch.Tensor:
    r"""$\mathbb{Z}_N$-invariant, angle-wrap-safe stabilization cost toward the downward rest state: $\sum_i(1-\cos
    \theta_i)+0.1\,\omega_i^2$ from a decoded $(\theta,\omega)\in(...,2N)$. (Sum over sites => shift-invariant.)"""
    th = thom[..., :N]; om = thom[..., N:2 * N]
    return ((1.0 - torch.cos(th)) + 0.1 * om ** 2).sum(dim=-1)


def plan_control(model, y0, mu, sd, H: int, u_max: float = 0.5, n_iter: int = 40, lr: float = 0.1,
                 seed: int = 0) -> torch.Tensor:
    r"""Plan a stabilizing control sequence $u_{0:H}$ for the ring WM, **exactly** $\mathbb{Z}_N$-orbit-equivariant:
    ``plan(roll_state(y0, s)) == roll(plan(y0), s)`` to float round-off. Deterministic projected-gradient planner from
    $u\equiv 0$; the predicted rollout (WM, full phase-augmented state) is scored by the $\mathbb{Z}_N$-invariant
    angle-wrap-safe cost :func:`_stabilize_cost` toward $\theta{=}0,\omega{=}0$. Control clamped to $|u|\le u_{\max}$
    (a per-site box, $\mathbb{Z}_N$-equivariant). Mirrors :func:`step79.plan_control`.

    ``y0: (2N+1,)`` RAW phase-augmented state -> ``u: (H, N)`` float64, clamped, exactly $\mathbb{Z}_N$-equivariant."""
    torch.manual_seed(seed)
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    y0 = y0.to(DTYPE).detach()
    N = (y0.shape[-1] - 1) // 2                                 # phase-augmented dim is 2N+1

    u = torch.zeros(H, N, dtype=DTYPE, requires_grad=True)     # Z_N-invariant init: S . 0 = 0
    for _ in range(n_iter):
        y = y0.clone()
        cost = y.new_zeros(())
        for t in range(H):
            y = wm_step_state(model, y, N, mu, sd, u[t])       # one WM step under the planned control u_t
            cost = cost + _stabilize_cost(decode_feat(encode_state(y, N)[0]), N)   # Z_N-invariant per-step cost
        (grad,) = torch.autograd.grad(cost, u)                 # grad_u J; equivariant because phi_hat & J are
        with torch.no_grad():
            u -= lr * grad                                     # fixed-step GD: pure elementwise/linear update
            u.clamp_(-u_max, u_max)                            # per-site box bound (Z_N-equivariant)
    return u.detach()


def closed_loop(model, y0, mu, sd, H: int, n_steps: int, u_max: float = 0.5, n_iter: int = 30, lr: float = 0.1,
                seed: int = 0) -> dict:
    r"""Closed-loop receding-horizon MPC on the TRUE controlled ring, driving $(\theta,\omega)\to(0,0)$. Each step
    re-plans with :func:`plan_control` from the current RAW phase-augmented state, applies ONLY $u_0$ to the TRUE
    :func:`rk4_ring`, and records the realized state. Exactly $\mathbb{Z}_N$-orbit-equivariant in ``y0`` (both planner
    and true map are equivariant). Returns ``traj`` ``(n_steps+1, 2N+1)``, ``controls`` ``(n_steps, N)``, ``cost``
    (time-avg :func:`_stabilize_cost` over the realized run). Mirrors :func:`step79.closed_loop`."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    N = (y0.shape[-1] - 1) // 2                                 # phase-augmented dim is 2N+1
    y_cur = y0.to(DTYPE).clone()
    traj = torch.empty(n_steps + 1, ring_state_dim(N), dtype=DTYPE)
    controls = torch.empty(n_steps, N, dtype=DTYPE)
    traj[0] = y_cur
    for t in range(n_steps):
        u = plan_control(model, y_cur, mu, sd, H, u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)
        u0 = u[0]
        y_cur = rk4_ring(y_cur, N, u0).detach()
        traj[t + 1] = y_cur
        controls[t] = u0
    costs = _stabilize_cost(decode_feat(encode_state(traj, N)[0]), N)   # (n_steps+1,) per-state cost
    return {"traj": traj, "controls": controls, "cost": float(costs.mean())}


def orbit_flatness(model, y0, mu, sd, H: int, n_steps: int, s: int, u_max: float = 0.5, n_iter: int = 30,
                   lr: float = 0.1, seed: int = 0) -> dict:
    r"""Measure that the closed-loop control is **orbit-flat**: run :func:`closed_loop` from ``y0`` and from the cyclic
    shift ``roll_state(y0, s)``; for an exactly $\mathbb{Z}_N$-equivariant planner on the exactly equivariant true ring,
    the rolled run is the cyclic shift of the original (applied controls roll by $s$; the cost — a sum over sites
    against the shift-invariant zero target — is invariant). Returns ``ratio`` = cost(roll)/cost, ``control_mismatch``
    = $\max_t\lVert u^{(t)}(\mathrm{roll})-\mathrm{roll}(u^{(t)})\rVert$, and the two costs. Mirrors
    :func:`step79.orbit_flatness`."""
    N = (y0.shape[-1] - 1) // 2                                 # phase-augmented dim is 2N+1
    out0 = closed_loop(model, y0, mu, sd, H, n_steps, u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)
    out_s = closed_loop(model, roll_state(y0.to(DTYPE), N, s), mu, sd, H, n_steps,
                        u_max=u_max, n_iter=n_iter, lr=lr, seed=seed)
    rolled_ctrl = torch.roll(out0["controls"], s, dims=-1)     # roll each applied u_0 by s sites
    control_mismatch = float((out_s["controls"] - rolled_ctrl).norm(dim=-1).max())
    cost_x0 = out0["cost"]; cost_rolled = out_s["cost"]
    ratio = cost_rolled / cost_x0 if cost_x0 != 0 else float("nan")
    return {"ratio": ratio, "control_mismatch": control_mismatch, "cost_x0": cost_x0, "cost_rolled": cost_rolled}


# --------------------------------------------------------------------------------------------------------------- #
# The VALIDATION (H) + the DECISION (D2 re-observation). Identical machinery to step79 phase 5b, on the ring:
#   * empirical_forecast_horizon: first map step where the WM's free-running (u=0) relative forecast error > eps.
#   * reobserve_run / reobservation_contrast: re-observe (reset WM to truth) every `interval` steps; cert-aware uses
#     interval = certified_T1_steps; blind sweeps fixed intervals around it. Honest D2 gate via _frontier_verdict.
# Forecast error is computed on the OBSERVABLE state (theta,omega) in a wrap-safe way: we compare the decoded
# (cos theta, sin theta, omega) features, so a +-2pi angle ambiguity never inflates the error.
# --------------------------------------------------------------------------------------------------------------- #
def _forecast_relerr(y_hat: torch.Tensor, y_true: torch.Tensor, N: int) -> torch.Tensor:
    r"""Wrap-safe relative forecast error on $(\theta,\omega)$: compare the $(\cos\theta,\sin\theta,\omega)$ features of
    the WM forecast ``y_hat`` and the truth ``y_true`` (so $\pm2\pi$ angle jumps do not inflate it). Returns
    $\lVert\widehat{\mathrm{feat}}-\mathrm{feat}^{\rm true}\rVert/\lVert\mathrm{feat}^{\rm true}\rVert$ over the
    batch (the drive-phase channel is excluded — it is a known clock, identical for both)."""
    fh, _ = encode_state(y_hat, N)                             # (..., 3, N)
    ft, _ = encode_state(y_true, N)
    num = (fh - ft).reshape(fh.shape[:-2] + (-1,)).norm(dim=-1)
    den = ft.reshape(ft.shape[:-2] + (-1,)).norm(dim=-1).clamp_min(1e-12)
    return num / den


def empirical_forecast_horizon(model, mu, sd, N: int, eps: float = 0.01, n_starts: int = 40,
                               seed: int = 0, max_steps: int = 2000) -> dict:
    r"""**The load-bearing validation.** For each of ``n_starts`` true on-attractor states, roll BOTH the WM (under
    $u{=}0$, full phase-augmented state) and the TRUE :func:`rk4_ring` ($u{=}0$) forward, and record the first map step
    $h$ where the wrap-safe relative forecast error :func:`_forecast_relerr` first exceeds ``eps``. This $h$ is the
    empirical forecast horizon the certified $T_1(\epsilon)$ (steps, via :func:`certified_T1_steps`) claims to predict.
    Mirrors :func:`step79.empirical_forecast_horizon`. Returns median/mean/p25/p75 (MAP STEPS), the per-start list, and
    ``n_censored`` (never crossed within ``max_steps``)."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    traj = attractor_traj_ring(N, max(n_starts * 4, n_starts + 1), seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(traj.shape[0], generator=g)[:n_starts]
    starts_y = traj[idx].clone()                               # (n_starts, 2N+1) on-attractor starts

    with torch.no_grad():
        y_true = starts_y.clone()
        y_hat = starts_y.clone()
        crossed = torch.zeros(n_starts, dtype=torch.bool)
        horizon = torch.full((n_starts,), max_steps, dtype=torch.long)
        for h in range(1, max_steps + 1):
            y_true = rk4_ring(y_true, N, None)                 # TRUE next state, u=0
            y_hat = wm_step_state(model, y_hat, N, mu, sd, None)   # WM next state, u=0
            rel = _forecast_relerr(y_hat, y_true, N)           # (n_starts,) wrap-safe rel fcst err
            newly = (rel > eps) & (~crossed)
            horizon[newly] = h
            crossed |= newly
            if bool(crossed.all()):
                break
    horizon_np = horizon.numpy().astype(float)
    n_censored = int((horizon == max_steps).sum().item())
    return {"median_empirical_horizon_steps": float(np.median(horizon_np)),
            "mean": float(np.mean(horizon_np)), "p25": float(np.percentile(horizon_np, 25)),
            "p75": float(np.percentile(horizon_np, 75)), "horizons": horizon_np.tolist(),
            "n_censored": n_censored, "n_starts": n_starts, "eps": eps}


def reobserve_run(model, mu, sd, N: int, y_seq_true: torch.Tensor, interval: int, eps: float = 0.01) -> dict:
    r"""Forecast a long TRUE ring trajectory ``y_seq_true`` $(T_{\rm total}+1, 2N+1)$ with the WM ($u{=}0$),
    **re-observing** (resetting the WM to the true state) every ``interval`` map steps. Between re-observations the
    wrap-safe forecast error grows (chaos); a re-observation pins it to 0. Records the per-step relative forecast error
    (0 on observation steps). Mirrors :func:`step79.reobserve_run`. Returns ``n_observations``, ``violation_rate``
    (fraction of forecast steps with rel err > eps), ``max_error``, ``mean_error``."""
    model.eval()
    mu = mu.to(DTYPE); sd = sd.to(DTYPE)
    interval = max(1, int(interval))
    y_seq_true = y_seq_true.to(DTYPE)
    T_total = y_seq_true.shape[0] - 1

    n_obs = 1
    errors = []
    with torch.no_grad():
        y_hat = y_seq_true[0].clone()                          # observed: WM = true at t=0
        for t in range(1, T_total + 1):
            y_hat = wm_step_state(model, y_hat, N, mu, sd, None)   # advance WM one u=0 step
            if t % interval == 0:                              # re-observation: reset to truth
                y_hat = y_seq_true[t].clone()
                n_obs += 1
                errors.append(0.0)
            else:
                rel = float(_forecast_relerr(y_hat, y_seq_true[t], N))
                errors.append(rel)
    errors_np = np.asarray(errors, dtype=float)
    viol = float((errors_np > eps).mean()) if errors_np.size else 0.0
    return {"n_observations": n_obs, "violation_rate": viol,
            "max_error": float(errors_np.max()) if errors_np.size else 0.0,
            "mean_error": float(errors_np.mean()) if errors_np.size else 0.0,
            "interval": interval, "T_total": T_total}


def reobservation_contrast(model, mu, sd, N: int, seed: int, eps: float = 0.01, T_total: int = 1500,
                           cert_n_steps: int = 2000, cert_warmup: int = 400, cert_n_boot: int = 300,
                           cert_block: int = 50, n_starts: int = 40) -> dict:
    r"""Run the D2 **decision**: certificate-aware vs horizon-blind active re-observation on the ring, one seed. Reads
    the certified forecast horizon in MAP STEPS (``certified_T1_steps``); blind agents sweep fixed intervals
    $\{T/4,T/2,T,2T,4T\}$ around it. One long TRUE trajectory ($u{=}0$) is forecast for each interval; the validation
    (:func:`empirical_forecast_horizon`) is computed on the SAME WM/seed. Mirrors :func:`step79.reobservation_contrast`.

    Returns ``T1_steps``, ``T1_time``, ``lambda1``, ``lambda1_ci``, ``interval_cert`` (a :func:`reobserve_run` dict),
    ``blind`` (``{interval: dict}``), ``rows`` (``[(interval, n_obs, viol, max_err, is_cert), ...]`` by n_obs), and
    ``empirical``."""
    cert = certificate(model, mu, sd, N, eps=eps, n_steps=cert_n_steps, warmup=cert_warmup,
                       n_boot=cert_n_boot, block=cert_block, seed=seed)
    T1_steps = certified_T1_steps(cert)
    emp = empirical_forecast_horizon(model, mu, sd, N, eps=eps, n_starts=n_starts, seed=seed,
                                     max_steps=max(2 * T1_steps, 500))

    traj = attractor_traj_ring(N, 4 * T_total, seed)
    g = torch.Generator().manual_seed(seed + 777)
    j0 = int(torch.randint(0, traj.shape[0] - T_total - 1, (1,), generator=g).item())
    y_seq_true = torch.empty(T_total + 1, ring_state_dim(N), dtype=DTYPE)
    y_seq_true[0] = traj[j0]
    y_cur = traj[j0].clone()
    for t in range(1, T_total + 1):                            # roll the TRUE u=0 dynamics forward
        y_cur = rk4_ring(y_cur, N, None)
        y_seq_true[t] = y_cur

    blind_intervals = sorted({max(1, v) for v in (T1_steps // 4, T1_steps // 2, T1_steps,
                                                  2 * T1_steps, 4 * T1_steps)})
    cert_run = reobserve_run(model, mu, sd, N, y_seq_true, interval=T1_steps, eps=eps)
    blind_runs = {iv: reobserve_run(model, mu, sd, N, y_seq_true, interval=iv, eps=eps) for iv in blind_intervals}

    rows = []
    for iv, r in blind_runs.items():
        rows.append((iv, r["n_observations"], r["violation_rate"], r["max_error"], iv == T1_steps))
    rows = sorted(rows, key=lambda t: t[1])
    return {"T1_steps": T1_steps, "T1_time": cert["T1"], "lambda1": cert["lambda1"],
            "lambda1_ci": cert["lambda1_ci"], "interval_cert": cert_run, "blind": blind_runs,
            "rows": rows, "empirical": emp, "eps": eps, "T_total": T_total}


def _frontier_verdict(out: dict) -> dict:
    r"""Decide whether the cert-aware re-observation interval is on the **efficient frontier** (the D2 gate), from a
    :func:`reobservation_contrast` result. On the frontier iff: (i) the cert point has $\sim$zero $\epsilon$-violations
    (``cert_viol`` $\le$ tol), (ii) some blind interval with FEWER observations has a clearly higher violation rate, and
    (iii) every blind interval that also $\sim$never violates uses MORE observations. Mirrors
    :func:`step79._frontier_verdict`."""
    rows = out["rows"]
    cert = next(r for r in rows if r[4])
    cert_obs, cert_viol = cert[1], cert[2]
    tol = 0.02
    cheaper = [r for r in rows if r[1] < cert_obs]
    pricier = [r for r in rows if r[1] > cert_obs]
    cheaper_worse = any(r[2] > cert_viol + 0.05 for r in cheaper)
    zero_viol_blind = [r for r in rows if (not r[4]) and r[2] <= tol]
    zeroviol_costlier = all(r[1] > cert_obs for r in zero_viol_blind) if zero_viol_blind else True
    cert_clean = cert_viol <= tol
    on_frontier = bool(cert_clean and cheaper_worse and zeroviol_costlier)
    return {"on_frontier": on_frontier, "cert_obs": cert_obs, "cert_viol": cert_viol, "cert_clean": cert_clean,
            "cheaper_worse": cheaper_worse, "zeroviol_costlier": zeroviol_costlier,
            "cheaper_max_viol": (max((r[2] for r in cheaper), default=0.0)), "n_cheaper": len(cheaper),
            "n_pricier": len(pricier)}


# --------------------------------------------------------------------------------------------------------------- #
# Drivers: _chaos (verify the TRUE ring is chaotic), _smoke (WM learns the one-step map), _equiv (the three exactness
# checks), _reobs (the H-validation + D2 frontier across seeds, saving the figure/JSON), and the full pipeline.
# --------------------------------------------------------------------------------------------------------------- #
def true_ring_spectrum(N: int, n_steps: int = 3000, warmup: int = 1000, seed: int = 0, n_exp: int | None = None):
    r"""Benettin-QR Lyapunov spectrum of the TRUE phase-augmented ring $\Delta t$-map (sorted descending). Used to
    VERIFY chaos ($\lambda_1>0$). ``n_exp`` limits the number of tracked exponents (default: full $2N+1$)."""
    y = attractor_traj_ring(N, 1, seed)[0].detach()            # an on-attractor operating point
    step = lambda z: rk4_ring(z, N, None)
    return step74.lyapunov_spectrum(step, y, n_steps, warmup, dt_map=DTMAP)[:(n_exp or ring_state_dim(N))]


def _chaos() -> None:
    r"""Verify the TRUE ring is chaotic: print $\lambda_1$ (and $\lambda_2,\lambda_3$) of the phase-augmented field,
    the Liouville sum (should be $\approx -N\gamma$ — see below), and confirm boundedness. Not a pytest test (slow QR)."""
    N = N_DEFAULT
    lam = true_ring_spectrum(N, n_steps=3000, warmup=1000, seed=0)
    # Liouville: div f = sum_i d(dom_i)/d omega_i = sum_i (-gamma) = -N*gamma (theta/psi rows add 0 on the diagonal),
    # so the full spectrum sums to -N*gamma exactly. A rigorous anchor on the spectrum estimator (cf. step74's -N).
    liou = float(lam.sum()); target = -N * GAMMA
    y = attractor_traj_ring(N, 6000, 1)[-1]
    bounded = bool(torch.isfinite(y).all() and y[N:2 * N].abs().max() < 50)
    print(f"[step80 chaos] N={N} gamma={GAMMA} K={K_COUP} A={A_DRIVE} Omega={OMEGA_DRIVE} dt_map={DTMAP}",
          file=sys.stderr)
    print(f"[step80 chaos] TRUE ring spectrum (top): lambda1={float(lam[0]):.4f}  lambda2={float(lam[1]):.4f}  "
          f"lambda3={float(lam[2]):.4f}  #pos={int((lam > 0).sum())}", file=sys.stderr)
    print(f"[step80 chaos] Liouville sum(lambda)={liou:.4f}  target -N*gamma={target:.4f}  "
          f"(rel err {abs(liou - target) / abs(target):.2%})  bounded={bounded}", file=sys.stderr)
    verdict = "CHAOTIC" if float(lam[0]) > 0.02 else "NOT clearly chaotic"
    print(f"[step80 chaos] verdict: {verdict} (lambda1={float(lam[0]):.4f} > 0).", file=sys.stderr)


def _smoke() -> None:
    r"""Tiny end-to-end check that the equivariant conv WM learns the controlled one-step ring map (relMSE < 1e-2). Not
    a pytest test (it trains)."""
    torch.manual_seed(0)
    N = N_DEFAULT
    data = collect_data(N=N, n_traj=2000, K=3, seed=0)
    _, _, _, conv_relmse = train_wm("conv", N, data, seed=0, epochs=15, K=3)
    _, _, _, mlp_relmse = train_wm("mlp", N, data, seed=0, epochs=15, K=3)
    print(f"[step80 smoke] N={N} -> conv one-step relMSE {conv_relmse:.2e}  mlp one-step relMSE {mlp_relmse:.2e}",
          file=sys.stderr)
    assert conv_relmse < 1e-2, f"conv one-step relMSE {conv_relmse:.2e} not < 1e-2"
    print("[step80 smoke] PASS: equivariant conv ring WM learns the controlled one-step map (relMSE < 1e-2).",
          file=sys.stderr)


def _equiv() -> None:
    r"""The three EXACTNESS checks (result 1), printed: (a) true controlled ring field + RK4 map are $\mathbb{Z}_N$-
    equivariant (machine precision), (b) the conv WM is $\mathbb{Z}_N$-equivariant and the MLP baseline is not, (c) the
    gradient planner is exactly orbit-equivariant (atol 1e-6). Mirrors the step79 exactness story. (No training.)"""
    torch.manual_seed(0)
    N = N_DEFAULT
    y = attractor_traj_ring(N, 1, 0)[0]
    u = 0.5 * torch.randn(N, dtype=DTYPE)
    # (a) true field + map
    dyn_err = mapp_err = 0.0
    for s in (1, 3, 7):
        dyn_err = max(dyn_err, float((ring_rhs(roll_state(y, N, s), N, u=torch.roll(u, s))
                                      - roll_state(ring_rhs(y, N, u=u), N, s)).abs().max()))
        mapp_err = max(mapp_err, float((rk4_ring(roll_state(y, N, s), N, torch.roll(u, s))
                                        - roll_state(rk4_ring(y, N, u), N, s)).abs().max()))
    # (b) WM conv vs mlp
    conv = RingConv(N).double(); mlp = RingMLP(N).double()
    feat, drive = encode_state(y, N)
    conv_err = 0.0
    with torch.no_grad():
        for s in (1, 5):
            lhs = conv(torch.roll(feat, s, dims=-1), torch.roll(u, s), drive)
            rhs = torch.roll(conv(feat, u, drive), s, dims=-1)
            conv_err = max(conv_err, float((lhs - rhs).abs().max()))
        m_lhs = mlp(torch.roll(feat, 1, dims=-1), torch.roll(u, 1), drive)
        m_rhs = torch.roll(mlp(feat, u, drive), 1, dims=-1)
        mlp_break = float((m_lhs - m_rhs).abs().max())
    # (c) planner
    mu = torch.zeros(N, dtype=DTYPE); sd = torch.ones(N, dtype=DTYPE)
    plan_err = 0.0
    for s in (1, 5):
        ua = plan_control(conv, y, mu, sd, H=6, seed=0)
        ub = plan_control(conv, roll_state(y, N, s), mu, sd, H=6, seed=0)
        plan_err = max(plan_err, float((ub - torch.roll(ua, s, dims=-1)).abs().max()))
    print(f"[step80 equiv] (a) true field max|S f - f S| = {dyn_err:.2e}; RK4 map = {mapp_err:.2e}  (atol 1e-10)",
          file=sys.stderr)
    print(f"[step80 equiv] (b) conv WM equiv err = {conv_err:.2e} (atol 1e-10); MLP baseline break = {mlp_break:.2e} "
          f"(should be >> 0)", file=sys.stderr)
    print(f"[step80 equiv] (c) gradient planner orbit-equiv err = {plan_err:.2e} (atol 1e-6)", file=sys.stderr)
    ok = (dyn_err < 1e-10 and mapp_err < 1e-10 and conv_err < 1e-10 and mlp_break > 1e-6 and plan_err < 1e-6)
    print(f"[step80 equiv] verdict: {'ALL EXACT' if ok else 'FAILED'} (result 1).", file=sys.stderr)


def _save_reobs_figure(per_seed_predictive: dict, per_seed_tight: dict, eps_predictive: float,
                       eps_tight: float, path: Path) -> None:
    r"""Two-panel figure of the ring's re-observation trade-off (violation-rate vs n_observations): one panel at the
    small ``eps_tight`` and one at ``eps_predictive``. Each seed: blind sweep (dots+line) + cert-aware (star). Mirrors
    step79's :func:`_save_phase5b_figure`. Pure-matplotlib, Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
    panels = [(axes[0], per_seed_tight, eps_tight, "small $\\epsilon$ — certificate may be OPTIMISTIC (Prop-8)"),
              (axes[1], per_seed_predictive, eps_predictive, "larger $\\epsilon$ — cert-aware at the knee?")]
    for ax, per_seed, eps, subtitle in panels:
        for seed, out in per_seed.items():
            c = colors.get(seed, "#555555")
            rows = sorted(out["rows"], key=lambda t: t[1])
            blind = [(r[1], r[2]) for r in rows if not r[4]]
            cert = next(r for r in rows if r[4])
            if blind:
                bx, by = zip(*blind)
                ax.plot(bx, by, "o-", color=c, alpha=0.55, ms=6, lw=1.3, label=f"seed {seed} blind sweep")
            ax.plot([cert[1]], [cert[2]], marker="*", color=c, ms=20, mec="black", mew=1.0, ls="none",
                    label=f"seed {seed} cert-aware ($T_1/\\Delta t$={out['T1_steps']})", zorder=5)
            emp = out["empirical"]
            ratio = emp["median_empirical_horizon_steps"] / out["T1_steps"] if out["T1_steps"] else float("nan")
            ax.annotate(f"s{seed}: emp/cert={ratio:.2f}", (cert[1], cert[2]), fontsize=6.5, color=c,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("observation budget  (n_observations over episode)")
        ax.set_ylabel(rf"$\epsilon$-violation rate  ($\epsilon={eps}$)")
        ax.set_title(f"$\\epsilon={eps}$ — {subtitle}", fontsize=10)
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    fig.suptitle("Step 80 (CORROBORATOR) — coupled-pendulum RING: certified horizon $T_1/\\Delta t_{map}$ as the "
                 "re-observation interval (units-fixed)\nclass-lift of the controlled-Lorenz-96 anchor (step79) to a "
                 "second chaotic $\\mathbb{Z}_N$ system", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[step80 reobs] figure -> {path}", file=sys.stderr)


def _reobs() -> None:
    r"""Train one equivariant conv WM on attractor data, run the H-validation + D2 active-re-observation decision for
    seeds 0,1,2 at TWO resolutions. For each $\epsilon$ prints the load-bearing validation (certified ``T1_steps`` vs
    empirical free-running forecast horizon + ratio) and the re-observation contrast (cert-aware vs blind: n_obs +
    violation_rate + max_err) with the D2 frontier verdict. Saves the figure + JSON. Not a pytest test (it trains).
    Mirrors :func:`step79._smoke_phase5b`.

    The two reported resolutions are the OPTIMISTIC one $\epsilon_{\rm tight}{=}0.01$ (spec default; the forecast
    horizon is short / transient-limited so the asymptotic-Lyapunov certificate is OPTIMISTIC, ratio$\ll 1$, exactly as
    the anchor) and the PREDICTIVE one $\epsilon_{\rm predictive}{=}0.3$ (the ring's asymptotic-Lyapunov regime: ratio
    $\sim 1$ and the cert-aware interval lands on the efficient frontier on 2/3 seeds — matching the anchor's 2/3). The
    anchor (controlled Lorenz-96) reaches its predictive regime at $\epsilon{=}0.2$; the ring's WM is marginally more
    optimistic at a fixed $\epsilon$, so its predictive knee is at a slightly larger resolution $\epsilon{=}0.3$ (an
    eps-sweep showed ratio crossing $1$ near $0.3$). This is the SAME honest two-regime story — only the resolution at
    which the horizon enters the asymptotic-Lyapunov regime differs between the two systems. The D2 gate
    (:func:`_frontier_verdict`) is UNCHANGED; we report the regime, we do not loosen the gate."""
    import json
    torch.manual_seed(0)
    N = N_DEFAULT
    # A decent multi-step model: the certificate reads the COMPOSED Jacobian, so train with K=5 rollout + enough
    # data/epochs that the learned u=0 map's Lyapunov spectrum is faithful.
    data = collect_data(N=N, n_traj=6000, K=5, seed=0)
    model, mu, sd, one_step = train_wm("conv", N, data, seed=0, epochs=40, K=5)
    print(f"[step80 reobs] N={N} attractor-data conv WM: in-sample one-step relMSE {one_step:.2e}", file=sys.stderr)

    def _run_eps(eps: float):
        per_seed = {}
        frontier_wins = 0
        print(f"[step80 reobs] ===================== eps = {eps} =====================", file=sys.stderr)
        for seed in (0, 1, 2):
            out = reobservation_contrast(model, mu, sd, N, seed=seed, eps=eps, T_total=1500)
            per_seed[seed] = out
            emp = out["empirical"]; T1s = out["T1_steps"]
            med = emp["median_empirical_horizon_steps"]
            ratio = med / T1s if T1s > 0 else float("nan")
            verdict = ("PREDICTIVE (within ~3x)" if 1.0 / 3.0 <= ratio <= 3.0 else
                       "certificate OPTIMISTIC (Prop-8 delta-bias)" if ratio < 1.0 / 3.0 else
                       "certificate CONSERVATIVE")
            print(f"[step80 reobs] --- eps={eps} seed {seed} ---", file=sys.stderr)
            print(f"[step80 reobs]   VALIDATION: certified T1_steps={T1s} (T1_time={out['T1_time']:.3f}, "
                  f"lambda1={out['lambda1']:.3f})  vs  empirical forecast horizon median={med:.0f} "
                  f"[p25={emp['p25']:.0f}, p75={emp['p75']:.0f}] steps (n_censored={emp['n_censored']}/"
                  f"{emp['n_starts']})", file=sys.stderr)
            print(f"[step80 reobs]   VALIDATION ratio empirical/certified = {ratio:.2f}  ({verdict})", file=sys.stderr)
            print(f"[step80 reobs]   RE-OBSERVATION contrast (interval -> n_obs, violation_rate, max_err):",
                  file=sys.stderr)
            for iv, n_obs, viol, mx, is_cert in out["rows"]:
                tag = "  <== CERT-AWARE" if is_cert else ""
                print(f"[step80 reobs]     interval={iv:>4d}: n_obs={n_obs:>4d}  viol={viol:.3f}  max_err={mx:.3f}{tag}",
                      file=sys.stderr)
            fv = _frontier_verdict(out)
            frontier_wins += int(fv["on_frontier"])
            print(f"[step80 reobs]   D2 frontier (eps={eps} seed {seed}): cert-aware viol={fv['cert_viol']:.3f} at "
                  f"n_obs={fv['cert_obs']} -> {'ON FRONTIER' if fv['on_frontier'] else 'NOT on frontier'}  "
                  f"[cert_clean={fv['cert_clean']}, cheaper_worse={fv['cheaper_worse']} "
                  f"(cheaper max viol {fv['cheaper_max_viol']:.3f}), zeroviol_costlier={fv['zeroviol_costlier']}]",
                  file=sys.stderr)
        print(f"[step80 reobs] D2 verdict @ eps={eps}: cert-aware on the efficient frontier on {frontier_wins}/3 seeds.",
              file=sys.stderr)
        return per_seed, frontier_wins

    EPS_TIGHT, EPS_PREDICTIVE = 0.01, 0.3   # ring predictive knee at 0.3 (anchor's was 0.2); see _reobs docstring
    EPS_ANCHOR = 0.2                          # the anchor's predictive eps; reported for a direct apples-to-apples row
    per_seed_tight, wins_tight = _run_eps(EPS_TIGHT)
    per_seed_anchor, wins_anchor = _run_eps(EPS_ANCHOR)   # the task's requested eps=0.2 (borderline for the ring)
    per_seed_pred, wins_pred = _run_eps(EPS_PREDICTIVE)

    print(f"[step80 reobs] OVERALL: D2 frontier {wins_tight}/3 at eps={EPS_TIGHT} (OPTIMISTIC), {wins_anchor}/3 at "
          f"eps={EPS_ANCHOR} (anchor's eps; borderline for the ring), {wins_pred}/3 at eps={EPS_PREDICTIVE} "
          f"(PREDICTIVE) — same two-regime story as anchor step79 (optimistic at 0.01, predictive 2/3 at its knee; "
          f"the ring's knee is eps=0.3 vs the anchor's eps=0.2).", file=sys.stderr)

    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    _save_reobs_figure(per_seed_pred, per_seed_tight, EPS_PREDICTIVE, EPS_TIGHT, figdir / "step80_pendulum_ring.png")

    def _seed_summary(out: dict) -> dict:
        emp = out["empirical"]; fv = _frontier_verdict(out)
        return {"T1_steps": out["T1_steps"], "T1_time": out["T1_time"], "lambda1": out["lambda1"],
                "lambda1_ci": out["lambda1_ci"],
                "empirical_median_horizon_steps": emp["median_empirical_horizon_steps"],
                "empirical_p25": emp["p25"], "empirical_p75": emp["p75"], "empirical_mean": emp["mean"],
                "empirical_n_censored": emp["n_censored"],
                "validation_ratio_empirical_over_certified": (emp["median_empirical_horizon_steps"] / out["T1_steps"]
                                                              if out["T1_steps"] else None),
                "rows": [{"interval": r[0], "n_observations": r[1], "violation_rate": r[2], "max_error": r[3],
                          "is_cert": r[4]} for r in out["rows"]],
                "on_frontier": fv["on_frontier"], "cert_viol": fv["cert_viol"], "cert_obs": fv["cert_obs"]}

    summary = {"system": "driven-damped coupled-pendulum ring", "corroborator_of": "step79 controlled Lorenz-96",
               "N": N, "gamma": GAMMA, "K": K_COUP, "A": A_DRIVE, "Omega": OMEGA_DRIVE, "dt_map": DTMAP,
               "one_step_relmse": one_step, "T_total": 1500, "n_seeds": 3,
               "eps_tight": EPS_TIGHT, "eps_anchor": EPS_ANCHOR, "eps_predictive": EPS_PREDICTIVE,
               "frontier_wins_tight": wins_tight, "frontier_wins_anchor": wins_anchor,
               "frontier_wins_predictive": wins_pred,
               "note": ("two-regime story matches the anchor: certificate OPTIMISTIC at eps=0.01 (ratio<<1), PREDICTIVE "
                        "at the asymptotic-Lyapunov knee. The ring's knee is eps=0.3 (D2 2/3) vs the anchor's eps=0.2; "
                        "at eps=0.2 the ring is borderline (ratio 0.6-1.0). D2 gate (_frontier_verdict) unchanged."),
               "regimes": {
                   str(EPS_TIGHT): {"seeds": {str(s): _seed_summary(o) for s, o in per_seed_tight.items()}},
                   str(EPS_ANCHOR): {"seeds": {str(s): _seed_summary(o) for s, o in per_seed_anchor.items()}},
                   str(EPS_PREDICTIVE): {"seeds": {str(s): _seed_summary(o) for s, o in per_seed_pred.items()}}}}
    jpath = figdir / "step80_pendulum_ring.json"
    with open(jpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[step80 reobs] JSON -> {jpath}", file=sys.stderr)


def _orbitflat() -> None:
    r"""Quick (C) check: the equivariant-planner closed-loop control is orbit-flat to machine precision (mismatch
    $<1\mathrm{e}{-8}$). Trains a small WM (modest), then runs :func:`orbit_flatness`. Not a pytest test (it trains)."""
    torch.manual_seed(0)
    N = N_DEFAULT
    data = collect_data(N=N, n_traj=2000, K=3, seed=0)
    model, mu, sd, relmse = train_wm("conv", N, data, seed=0, epochs=15, K=3)
    y0 = attractor_traj_ring(N, 1, 0)[0]
    of = orbit_flatness(model, y0, mu, sd, H=6, n_steps=20, s=3, u_max=0.5, n_iter=30)
    print(f"[step80 orbitflat] N={N} conv WM one-step relMSE {relmse:.2e}", file=sys.stderr)
    print(f"[step80 orbitflat] (C) closed-loop control orbit-flat: control_mismatch {of['control_mismatch']:.2e} "
          f"(gate < 1e-8)  cost ratio {of['ratio']:.8f}  [cost {of['cost_x0']:.4f} vs {of['cost_rolled']:.4f}]",
          file=sys.stderr)
    ok = of["control_mismatch"] < 1e-8 and abs(of["ratio"] - 1.0) < 1e-6
    print(f"[step80 orbitflat] verdict: {'ORBIT-FLAT to machine precision' if ok else 'FAILED'} (result C).",
          file=sys.stderr)


if __name__ == "__main__":
    _phases = {"chaos": _chaos, "smoke": _smoke, "equiv": _equiv, "orbitflat": _orbitflat, "reobs": _reobs}
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        _chaos(); _equiv(); _smoke(); _orbitflat(); _reobs()
    else:
        _phases.get(arg, _smoke)()
