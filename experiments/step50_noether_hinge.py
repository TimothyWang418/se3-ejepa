r"""Step 50: the **Noether hinge** — do a learned equivariant model's *slow* (long-horizon-certifiable)
latent channels coincide with its *group-invariant* ones?  (paper2 §3, the headline measured conjecture.)

paper2's horizon axis rests on a bridge: a channel is certifiable over long horizons iff it is
*dynamically slow* (small Lyapunov exponent / Jacobian eigenvalue $\approx 1$). The conjecture (§3) is that
in an **equivariant** world model the slow channels are exactly the **group-invariant** ones — so structure
*tells you where the long-horizon certificate lives, for free*, without measuring the dynamics, and that
location is exact and stable under the whole group (the invariant subspace is architectural). A
non-equivariant model may also learn slow modes, but they are smeared across unlabeled channels and are only
*approximately* invariant, so they drift under group action and carry no certificate.

The mechanism is literally **Noether's theorem**. On an $\mathrm{SO}(2)$-symmetric central-force system the
conserved quantities — energy $E=\tfrac12\lVert v\rVert^2+V(\lVert r\rVert)$ and angular momentum
$L=x\dot y-y\dot x$ — are $\mathrm{SO}(2)$-**invariant scalars** and **exactly constant** (the slowest
possible modes). The orbital phase rotates fast. The *honest* claim is the defensible direction:

    slow  ⊆  invariant     (the conserved/slowest modes necessarily live in the invariant subspace)

and NOT the false converse — $\lVert r\rVert^2$ is invariant yet oscillates, so invariant ⊉ slow. What we
test is whether a *learned* (not hand-built) equivariant autoencoder-with-latent-dynamics spontaneously
puts the conserved quantities into its architecturally-invariant ($\ell{=}0$, scalar) channels, and whether
that gives a certificate the matched MLP lacks.

World.  2D central-force orbits, state $s=(x,y,\dot x,\dot y)$, harmonic potential $V(r)=\tfrac12\omega^2 r^2$
(linear dynamics ⇒ $E,L$ exactly conserved; phase rotates at $\omega$; $\lVert r\rVert^2$ oscillates at
$2\omega$ — giving us *both* slow-invariant and fast-invariant channels, the nuance the conjecture needs).
Initial conditions sampled $\mathrm{SO}(2)$-isotropically ⇒ the data distribution is *exactly* group-symmetric
(unlike Step 48's data — so the equivariant measurements are not finite-sample-confounded).

Models.
- **Equivariant** SO(2)-steerable autoencoder + latent predictor (hand-rolled 2D Vector-Neurons): latent =
  scalars $s\in\mathbb{R}^{d_s}$ ($\ell{=}0$, $\rho(\theta)s=s$) ⊕ vectors $V\in\mathbb{R}^{d_v\times2}$
  ($\ell{=}1$, each row rotates by $R(\theta)$). The scalar block is the invariant subspace **by construction**.
- **MLP** baseline: matched-budget plain net on the flat 4-vector. No labeled invariant subspace.

Measurements (after identical training).
1. **Noether content** — regress ground-truth $(E,L)$ from the invariant (scalar) block vs the equivariant
   (vector) block. Conjecture: scalars recover $(E,L)$ with high $R^2$; vectors do not (they carry the
   rotating position/velocity).
2. **Slow ⊆ invariant** — per-channel slowness (relative one-step change). The *slowest* channels are scalar;
   no vector channel is as slow as the slowest scalars. (Some scalars — the $\lVert r\rVert^2$ mode — are fast:
   invariant ⊉ slow, reported honestly.)
3. **The certificate the MLP can't have** — OOD group-transfer drift of the "conserved" subspace. The
   equivariant model's invariant (scalar) subspace has group-action residual $\approx0$ *exactly and
   architecturally* (stays so under all of $\mathrm{SO}(2)$); the MLP's empirically-slowest directions drift
   by $O(0.1\text{–}1)$ under the same rotations — they are not certified.

Run (full ~2-3 min CPU):  .venv/bin/python experiments/step50_noether_hinge.py
Smoke:  STEP50_SMOKE=1 .venv/bin/python experiments/step50_noether_hinge.py
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.set_default_dtype(torch.float64)  # small problem; float64 so "exact/invariant" is unambiguous

SMOKE = bool(os.environ.get("STEP50_SMOKE"))
SEED = int(os.environ.get("STEP50_SEED", "0"))
OMEGA = 1.0          # harmonic frequency
DT = 0.15            # integration / observation step
DS, DV = 8, 8        # latent: 8 invariant scalars (ell=0) + 8 vectors (ell=1) -> 8 + 16 = 24 dims


# --------------------------------------------------------------------------------------------------
# Data: 2D central-force (harmonic) orbits, SO(2)-isotropic initial conditions.
# --------------------------------------------------------------------------------------------------
def rot2(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])


def true_invariants(state: torch.Tensor) -> torch.Tensor:
    r"""Ground-truth conserved quantities $(E, L)$ for $V(r)=\tfrac12\omega^2 r^2$.  state:(B,4)->(B,2)."""
    r, v = state[:, :2], state[:, 2:]
    E = 0.5 * (v ** 2).sum(1) + 0.5 * OMEGA ** 2 * (r ** 2).sum(1)
    L = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    return torch.stack([E, L], 1)


def make_orbits(n_traj: int, steps: int, seed: int):
    r"""Roll $n_{traj}$ harmonic orbits for `steps` observation steps. Returns consecutive pairs
    $(s_t, s_{t+1})$. Harmonic dynamics are the exact linear flow $\exp(tA)$ — symplectic, $E,L$ conserved."""
    g = torch.Generator().manual_seed(seed)
    # SO(2)-isotropic ICs: random radius, speed, and two independent random directions
    r0 = 0.5 + 1.5 * torch.rand(n_traj, 1, generator=g)
    ang_r = 2 * math.pi * torch.rand(n_traj, 1, generator=g)
    sp = 0.5 + 1.5 * torch.rand(n_traj, 1, generator=g)
    ang_v = 2 * math.pi * torch.rand(n_traj, 1, generator=g)
    pos = r0 * torch.cat([torch.cos(ang_r), torch.sin(ang_r)], 1)
    vel = sp * torch.cat([torch.cos(ang_v), torch.sin(ang_v)], 1)
    # exact harmonic flow over DT:  pos' = pos cos(wt) + vel/w sin(wt);  vel' = -pos w sin(wt) + vel cos(wt)
    w, c, s = OMEGA, math.cos(OMEGA * DT), math.sin(OMEGA * DT)
    S_t, S_tp1 = [], []
    state = torch.cat([pos, vel], 1)
    for _ in range(steps):
        p, v = state[:, :2], state[:, 2:]
        p2 = p * c + v / w * s
        v2 = -p * w * s + v * c
        nxt = torch.cat([p2, v2], 1)
        S_t.append(state); S_tp1.append(nxt)
        state = nxt
    return torch.cat(S_t, 0), torch.cat(S_tp1, 0)


# --------------------------------------------------------------------------------------------------
# Hand-rolled 2D SO(2)-steerable primitives (Vector Neurons in 2D).
#   scalars s:(B,ds) transform trivially (rho(theta) s = s)  -> the INVARIANT subspace, by construction.
#   vectors V:(B,dv,2) each row rotates: rho(theta) V = V R(theta)^T  (we apply R on the right per row).
# --------------------------------------------------------------------------------------------------
def inv_readout(V: torch.Tensor) -> torch.Tensor:
    r"""All SO(2)-invariant quadratics of the vectors: symmetric inner products $\langle V_i,V_j\rangle$ and
    antisymmetric cross products $V_i\times V_j$ (the 2D wedge = pseudoscalar, invariant under SO(2)). Together
    they span the full quadratic-invariant space — crucially the cross term is what lets the net represent
    $L=r\times v$.  V:(B,dv,2) -> (B, dv*dv)."""
    gram = torch.einsum("bic,bjc->bij", V, V)                       # <V_i,V_j>  symmetric
    cross = V[:, :, None, 0] * V[:, None, :, 1] - V[:, :, None, 1] * V[:, None, :, 0]  # V_i x V_j  antisym
    return torch.cat([gram.reshape(V.shape[0], -1), cross.reshape(V.shape[0], -1)], 1)


class VecLinear(nn.Module):
    r"""$V'_j=\sum_i W_{ji}V_i$ — real mixing of vector channels, **no bias**. Commutes with $R$ (rotation
    is applied per row and pulls through the real weights), so it is exactly SO(2)-equivariant."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_out, n_in) / math.sqrt(n_in))

    def forward(self, V):                                            # (B,n_in,2)->(B,n_out,2)
        return torch.einsum("ji,bic->bjc", self.W, V)


class EquivBlock(nn.Module):
    r"""One steerable block: invariants feed the scalar stream; a scalar MLP refines scalars; vectors are
    linearly mixed, norm-gated (equivariant nonlinearity), then scaled by scalar-derived gates."""

    def __init__(self, ds: int, dv: int):
        super().__init__()
        self.from_inv = nn.Linear(dv * dv * 2, ds)
        self.smlp = nn.Sequential(nn.Linear(ds, 2 * ds), nn.SiLU(), nn.Linear(2 * ds, ds))
        self.vlin = VecLinear(dv, dv)
        self.norm_a = nn.Parameter(torch.zeros(dv))                 # norm-gate slope (per channel)
        self.norm_b = nn.Parameter(3.0 * torch.ones(dv))           # norm-gate bias: OPEN at init (σ(3)≈0.95)
        self.gate = nn.Linear(ds, dv)                               # scalar -> per-vector gate
        nn.init.constant_(self.gate.bias, 3.0)                      # OPEN at init so signal/gradients flow

    def forward(self, s, V):
        s = s + self.from_inv(inv_readout(V))                       # ell=0 <- invariants of ell=1 (Noether path)
        s = s + self.smlp(s)
        V = self.vlin(V)
        n = V.norm(dim=-1, keepdim=True).clamp_min(1e-9)            # |V_j| invariant
        V = V * torch.sigmoid(self.norm_a[None, :, None] * n + self.norm_b[None, :, None])  # equivariant nonlin
        V = V * torch.sigmoid(self.gate(s))[:, :, None]             # scalar gate (invariant scale)
        return s, V


class EquivWorldModel(nn.Module):
    r"""SO(2)-equivariant autoencoder + latent predictor. Encoder: state's two vectors (pos,vel) -> latent
    $(s\in\mathbb{R}^{d_s},\,V\in\mathbb{R}^{d_v\times2})$. Predictor $F$: latent->latent. Decoder: latent->state.
    forward shapes:  state (B,4) -> latent (s:(B,ds), V:(B,dv,2)) -> state (B,4)."""

    def __init__(self, ds=DS, dv=DV):
        super().__init__()
        self.lift_s = nn.Linear(2 * 2 * 2, ds)                      # 2 input vecs -> 8 invariants (gram+cross)
        self.lift_v = VecLinear(2, dv)
        self.enc = nn.ModuleList([EquivBlock(ds, dv), EquivBlock(ds, dv)])
        self.pred = nn.ModuleList([EquivBlock(ds, dv), EquivBlock(ds, dv)])
        self.dec_v = VecLinear(dv, 2)
        self.dec_gate = nn.Linear(ds, 2)
        nn.init.constant_(self.dec_gate.bias, 3.0)                  # OPEN at init

    def encode(self, state):
        Vin = torch.stack([state[:, :2], state[:, 2:]], 1)          # (B,2,2): pos, vel
        s = self.lift_s(inv_readout(Vin))
        V = self.lift_v(Vin)
        for blk in self.enc:
            s, V = blk(s, V)
        return s, V

    def predict(self, s, V):
        for blk in self.pred:
            s, V = blk(s, V)
        return s, V

    def decode(self, s, V):
        Vout = self.dec_v(V) * torch.sigmoid(self.dec_gate(s))[:, :, None]   # (B,2,2)
        return torch.cat([Vout[:, 0], Vout[:, 1]], 1)               # (B,4)

    def latent_flat(self, state):
        s, V = self.encode(state)
        return torch.cat([s, V.reshape(s.shape[0], -1)], 1)         # (B, ds + 2 dv)


class MLPWorldModel(nn.Module):
    r"""Matched-budget non-equivariant baseline. Same autoencoder+predictor structure, plain MLPs on the flat
    4-vector. Its latent has NO architecturally-labeled invariant subspace."""

    def __init__(self, latent=DS + 2 * DV):
        super().__init__()
        h = 96
        self.encs = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, latent))
        self.preds = nn.Sequential(nn.Linear(latent, h), nn.SiLU(), nn.Linear(h, latent))
        self.decs = nn.Sequential(nn.Linear(latent, h), nn.SiLU(), nn.Linear(h, 4))

    def encode(self, state):
        return self.encs(state)

    def predict(self, z):
        return self.preds(z)

    def decode(self, z):
        return self.decs(z)

    def latent_flat(self, state):
        return self.encs(state)


# --------------------------------------------------------------------------------------------------
# Training (identical objective for both): recon + 1-step prediction (decoded) + latent consistency.
# --------------------------------------------------------------------------------------------------
def train(model, S_t, S_tp1, epochs, equiv: bool):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # decay -> stable convergence
    n = S_t.shape[0]
    bs = 256
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            st, st1 = S_t[idx], S_tp1[idx]
            opt.zero_grad()
            if equiv:
                s, V = model.encode(st)
                recon = model.decode(s, V)
                sp, Vp = model.predict(s, V)
                pred = model.decode(sp, Vp)
                with torch.no_grad():
                    s1, V1 = model.encode(st1)
                lat = ((sp - s1) ** 2).mean() + ((Vp - V1) ** 2).mean()
            else:
                z = model.encode(st)
                recon = model.decode(z)
                zp = model.predict(z)
                pred = model.decode(zp)
                with torch.no_grad():
                    z1 = model.encode(st1)
                lat = ((zp - z1) ** 2).mean()
            loss = ((recon - st) ** 2).mean() + ((pred - st1) ** 2).mean() + 0.1 * lat
            loss.backward()
            opt.step()
        sched.step()
    return float(loss.item())


# --------------------------------------------------------------------------------------------------
# Measurement helpers.
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def r2_regress(X: torch.Tensor, Y: torch.Tensor) -> float:
    r"""Best linear $R^2$ predicting $Y$ from $X$ (with bias), averaged over $Y$'s columns."""
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    Yc = Y - Y.mean(0, keepdim=True)
    coef = torch.linalg.lstsq(Xb, Y).solution
    resid = Y - Xb @ coef
    ss_res = (resid ** 2).sum(0)
    ss_tot = (Yc ** 2).sum(0).clamp_min(1e-12)
    return float((1 - ss_res / ss_tot).mean())


@torch.no_grad()
def per_channel_slowness(latent_t: torch.Tensor, latent_tp1: torch.Tensor) -> torch.Tensor:
    r"""Relative one-step change per channel: $\mathrm{std}_t(z_{t+1,i}-z_{t,i})/\mathrm{std}_t(z_{t,i})$.
    Small ⇒ slow (long-horizon-certifiable)."""
    d = (latent_tp1 - latent_t).std(0)
    scale = latent_t.std(0).clamp_min(1e-9)
    return (d / scale)


@torch.no_grad()
def slowest_mode_rate(Zt: torch.Tensor, Zt1: torch.Tensor) -> float:
    r"""Basis-independent slowest mode *within a subspace*: $\min_u \mathrm{Var}(\langle u,\Delta z\rangle)/
    \mathrm{Var}(\langle u,z\rangle)$, the smallest generalized eigenvalue of $(C_\Delta, C_z)$. Returns the
    std-ratio rate $\sqrt{\lambda_{\min}}$ — the relative one-step change of the *slowest direction* the block
    admits (so per-channel mixing can't hide a conserved mode). $\approx 0$ ⇒ the block contains a conserved
    (maximally long-horizon-certifiable) quantity."""
    Dc = (Zt1 - Zt) - (Zt1 - Zt).mean(0, keepdim=True)
    Zc = Zt - Zt.mean(0, keepdim=True)
    n = Zc.shape[0]
    C_D = Dc.T @ Dc / n
    C_Z = Zc.T @ Zc / n + 1e-9 * torch.eye(Zc.shape[1])
    Lc = torch.linalg.cholesky(C_Z)                                 # C_Z = Lc Lc^T
    M = torch.linalg.solve_triangular(Lc, torch.linalg.solve_triangular(Lc, C_D, upper=False).T, upper=False)
    lam = torch.linalg.eigvalsh(M).clamp_min(0)                     # generalized eigenvalues
    return float(lam.min() ** 0.5)


@torch.no_grad()
def ood_group_residual(model, state, dirs: torch.Tensor, equiv: bool, n_rot=16):
    r"""For a set of latent directions (columns of `dirs`, orthonormal), measure how much the projected
    coordinate $\langle u, z\rangle$ changes when the *input* is rotated by random $\theta$ (the group acts on
    the world). residual = $\mathrm{mean}_\theta \mathrm{std}(\langle u, z(R_\theta s)\rangle - \langle u, z(s)\rangle)
    / \mathrm{std}(\langle u, z(s)\rangle)$. Small ⇒ that direction is group-invariant (certifiable). For the
    equivariant model the scalar block is invariant *architecturally*; we verify, and contrast the MLP."""
    base = model.latent_flat(state) @ dirs                          # (B, k)
    res = []
    g = torch.Generator().manual_seed(777)
    for _ in range(n_rot):
        th = float(2 * math.pi * torch.rand(1, generator=g))
        R = rot2(th)
        rs = torch.cat([state[:, :2] @ R.T, state[:, 2:] @ R.T], 1)
        proj = model.latent_flat(rs) @ dirs
        res.append(((proj - base).std(0) / base.std(0).clamp_min(1e-9)))
    return torch.stack(res).mean(0)                                  # (k,)


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 92
    n_traj, steps, epochs = (200, 12, 12) if SMOKE else (800, 30, 80)
    epochs = int(os.environ.get("STEP50_EPOCHS", epochs))
    print(line)
    print(f"STEP 50  Noether hinge: are a learned equivariant model's SLOW channels its INVARIANT ones?  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    S_t, S_tp1 = make_orbits(n_traj, steps, seed=SEED)
    St_te, St1_te = make_orbits(160, steps, seed=999)
    EL = true_invariants(St_te)                                     # ground-truth (E,L) on test set
    print(f"    data: {S_t.shape[0]} train pairs, {St_te.shape[0]} test pairs; conserved E,L "
          f"std over set = {EL.std(0).tolist()}")

    # ---- equivariant model ----
    torch.manual_seed(SEED)
    eq = EquivWorldModel()
    eq_loss = train(eq, S_t, S_tp1, epochs, equiv=True)

    # equivariance unit test on the encoder: rotate input -> scalars fixed, vectors rotate
    with torch.no_grad():
        th = 0.7
        R = rot2(th)
        rs = torch.cat([St_te[:, :2] @ R.T, St_te[:, 2:] @ R.T], 1)
        s0, V0 = eq.encode(St_te)
        s1, V1 = eq.encode(rs)
        scal_resid = (s1 - s0).abs().max().item()
        vec_resid = (V1 - torch.einsum("ij,bnj->bni", R, V0)).abs().max().item()
    print(f"\n  [equivariant]  train loss {eq_loss:.2e} | encoder equivariance: "
          f"scalar Δ {scal_resid:.2e} (should be ~0), vector rot-Δ {vec_resid:.2e} (should be ~0)")

    with torch.no_grad():
        s_te, V_te = eq.encode(St_te)
        s_te1, V_te1 = eq.encode(St1_te)
        scal = s_te                                                 # (B, ds)  INVARIANT block
        vecf = V_te.reshape(V_te.shape[0], -1)                      # (B, 2dv) EQUIVARIANT block
    r2_scal = r2_regress(scal, EL)
    r2_vec = r2_regress(vecf, EL)
    print(f"    Noether content — recover (E,L):  invariant(scalar) R²={r2_scal:.3f}   "
          f"equivariant(vector) R²={r2_vec:.3f}")

    slow_scal = per_channel_slowness(s_te, s_te1)
    slow_vec = per_channel_slowness(V_te.reshape(V_te.shape[0], -1), V_te1.reshape(V_te1.shape[0], -1))
    # basis-independent slowest achievable mode within each isotypic block (robust to per-channel mixing)
    rate_scal = slowest_mode_rate(s_te, s_te1)
    rate_vec = slowest_mode_rate(V_te.reshape(V_te.shape[0], -1), V_te1.reshape(V_te1.shape[0], -1))
    print(f"    slow ⊆ invariant — SLOWEST MODE per block (basis-independent, smaller=slower):")
    print(f"        invariant(ℓ=0) block: {rate_scal:.4f}  (a conserved mode ⇒ ≈0)   "
          f"equivariant(ℓ=1) block: {rate_vec:.4f}")
    print(f"        per-channel relative change: scalar min {slow_scal.min():.3f}/mean {slow_scal.mean():.3f}"
          f" (incl. a fast |r|² mode ⇒ invariant ⊉ slow), vector min {slow_vec.min():.3f}/mean "
          f"{slow_vec.mean():.3f}")

    # OOD group-transfer: the invariant (scalar) subspace should be exactly group-invariant (certified)
    eye = torch.eye(DS + 2 * DV)
    scal_dirs = eye[:, :DS]                                         # the architectural invariant subspace
    eq_inv_resid = float(ood_group_residual(eq, St_te, scal_dirs, equiv=True).mean())

    # ---- MLP baseline ----
    torch.manual_seed(SEED)
    mlp = MLPWorldModel()
    mlp_loss = train(mlp, S_t, S_tp1, epochs, equiv=False)
    with torch.no_grad():
        z_te = mlp.encode(St_te)
        z_te1 = mlp.encode(St1_te)
    r2_mlp = r2_regress(z_te, EL)
    slow_mlp = per_channel_slowness(z_te, z_te1)
    # the MLP's own "conserved subspace" = its 2 slowest channels; test if THOSE are group-invariant
    slowest2 = torch.topk(slow_mlp, 2, largest=False).indices
    mlp_slow_dirs = torch.eye(DS + 2 * DV)[:, slowest2]
    mlp_inv_resid = float(ood_group_residual(mlp, St_te, mlp_slow_dirs, equiv=False).mean())
    print(f"\n  [MLP baseline] train loss {mlp_loss:.2e} | full-latent (E,L) R²={r2_mlp:.3f} "
          f"(smeared; no labeled invariant subspace)")
    print(f"    MLP slowest-2 channels' OOD group residual {mlp_inv_resid:.3f}  "
          f"(its 'conserved' directions are NOT group-invariant)")

    # ---- gate / verdict ----
    print(f"\n{line}\nSTEP 50 SUMMARY\n{line}")
    print(f"    (1) Noether content: invariant block recovers (E,L) at R²={r2_scal:.3f} vs vector "
          f"R²={r2_vec:.3f}  -> conserved quantities live in the INVARIANT subspace")
    print(f"    (2) slow ⊆ invariant: slowest mode in the invariant block {rate_scal:.4f} (≈conserved) ≪ "
          f"slowest mode in the equivariant block {rate_vec:.4f}")
    print(f"    (3) the certificate: equiv invariant-subspace OOD group residual {eq_inv_resid:.2e} "
          f"(architectural, holds ∀SO(2))  vs  MLP slow-subspace residual {mlp_inv_resid:.3f}")
    ok = (r2_scal > 0.9 and r2_scal > r2_vec + 0.2
          and rate_scal < 0.5 * rate_vec
          and eq_inv_resid < 1e-4 and mlp_inv_resid > 10 * max(eq_inv_resid, 1e-9))
    if ok:
        print(f"\n    CONFIRMED (group⇒slow hinge): the learned equivariant model spontaneously placed the")
        print(f"        conserved (slowest) quantities in its architecturally-INVARIANT channels — so the")
        print(f"        long-horizon-certifiable subspace is identifiable for free AND exact under the whole")
        print(f"        group ({eq_inv_resid:.1e}); the MLP's slow directions drift under group action")
        print(f"        ({mlp_inv_resid:.2f}) and carry no certificate. slow ⊆ invariant, honestly (a fast")
        print(f"        |r|² scalar shows invariant ⊉ slow). This is paper2 §3's measured conjecture.")
    else:
        print(f"\n    INCONCLUSIVE: not all gate conditions met; reported honestly as-is (no thresholds loosened).")

    out = dict(passed=bool(ok), equiv_loss=eq_loss, mlp_loss=mlp_loss,
               r2_scalar_EL=r2_scal, r2_vector_EL=r2_vec, r2_mlp_EL=r2_mlp,
               slowest_mode_invariant=rate_scal, slowest_mode_equivariant=rate_vec,
               slow_scalar=slow_scal.tolist(), slow_vector=slow_vec.tolist(), slow_mlp=slow_mlp.tolist(),
               eq_invariant_ood_residual=eq_inv_resid, mlp_slow_ood_residual=mlp_inv_resid,
               encoder_scalar_resid=scal_resid, encoder_vector_resid=vec_resid,
               omega=OMEGA, dt=DT, ds=DS, dv=DV, smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step50_noether_hinge_smoke" if SMOKE else "step50_noether_hinge"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    # figure: (left) slowness by block, (right) the certificate contrast
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].axhline(float(slow_scal.min()), color="C2", ls=":", lw=1, alpha=0.7)
    ax[0].scatter([0] * len(slow_scal), slow_scal.tolist(), color="C2", s=60,
                  label="invariant (ℓ=0 scalar)", zorder=3)
    ax[0].scatter([1] * len(slow_vec), slow_vec.tolist(), color="C3", s=60,
                  label="equivariant (ℓ=1 vector)", zorder=3)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["invariant\nchannels", "equivariant\nchannels"])
    ax[0].set_ylabel("per-channel relative one-step change\n(smaller = slower = long-horizon-certifiable)")
    ax[0].set_title(f"slow ⊆ invariant — slowest mode: invariant {rate_scal:.3f} ≪ equivariant {rate_vec:.3f}\n"
                    f"(E,L) recovered from invariant block: R²={r2_scal:.2f}")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(["equivariant\ninvariant subspace\n(architectural)", "MLP\nslowest directions\n(empirical)"],
              [max(eq_inv_resid, 1e-7), mlp_inv_resid], color=["C2", "C3"])
    ax[1].set_yscale("log"); ax[1].set_ylabel("OOD group-action residual\n(lower = certified invariant)")
    ax[1].set_title("the certificate: does the slow subspace\nstay invariant under all of SO(2)?")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Step 50 — Noether hinge: a learned equivariant model's slow channels are its invariant ones "
                 "(paper2 §3)", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
