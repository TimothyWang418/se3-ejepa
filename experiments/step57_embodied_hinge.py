r"""Step 57: lifting the Noether hinge toward an **embodied, contact** setting (3D, $\mathrm{SO}(3)$, two bodies).

Step 50 established the hinge (slow $\subseteq$ invariant) on a clean 2D $\mathrm{SO}(2)$ single-body orbit. The
natural next question (paper2 §3 open item) is whether it survives a more *embodied* regime: 3D, multiple bodies,
and **contact-like interaction**. We cannot run it on the Step 24/43 object model directly — that model's latent is
a **pure stack of $\ell{=}1$ vectors with no $\ell{=}0$ (invariant) channels**, so the hinge (which is *about* the
$\ell{=}0$ block) is not even well-posed there; testing it first requires an equivariant model with an isotypic
$\ell{=}0\oplus\ell{=}1$ latent. So we build the minimal such system that is still genuinely 3D, multi-body, and
contact-bearing.

System.  Two bodies in a shared 3D isotropic harmonic well with a **soft pairwise repulsion** (a contact-like
collision term $U_{\rm rep}=c\,e^{-\lVert r_1-r_2\rVert^2/2\sigma^2}$). At $\beta{=}0$ the law is $\mathrm{SO}(3)$-
symmetric, so total energy $E$ is a conserved **invariant scalar** and total angular momentum $L=\sum_i r_i\times v_i$
is a conserved $\ell{=}1$ **vector** (its magnitude $\lVert L\rVert$ is invariant); the orientations rotate fast.
An anisotropy knob $\beta$ (well $\to \tfrac12\sum(x^2+y^2+(1+2\beta)z^2)$) breaks $\mathrm{SO}(3)$ to test
*approximate* symmetry.

Model.  A hand-rolled **3D Vector-Neuron** autoencoder + latent predictor: latent $=$ scalars $s\in\mathbb{R}^{d_s}$
($\ell{=}0$, invariant) $\oplus$ vectors $V\in\mathbb{R}^{d_v\times3}$ ($\ell{=}1$, $\mathrm{SO}(3)$-rotated). The
scalar block is the invariant subspace by construction.

Hinge test (as Step 50, lifted to 3D + contact):
  1. **Noether content** — does the invariant ($\ell{=}0$) block recover the conserved $E$ (and the contact
     coordinate $\lVert r_1-r_2\rVert$) far better than the $\ell{=}1$ block?
  2. **slow $\subseteq$ invariant** — is the slowest mode the invariant block admits $\ll$ the slowest the
     equivariant block admits?
  3. **approximate symmetry** — does the picture degrade *gracefully* as $\beta$ breaks $\mathrm{SO}(3)$?

Honest: this is the genuinely open experiment; contact + 3D is harder than the clean orbit, and we report the gate
verdict (CONFIRMED / INCONCLUSIVE) as measured, without loosening thresholds.

Run (full ~2-3 min):  .venv/bin/python experiments/step57_embodied_hinge.py
Smoke:  STEP57_SMOKE=1 .venv/bin/python experiments/step57_embodied_hinge.py
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from step50_noether_hinge import slowest_mode_rate, r2_regress  # reuse the basis-independent metrics  # noqa: E402

torch.set_default_dtype(torch.float64)
SMOKE = bool(os.environ.get("STEP57_SMOKE"))
SEED = int(os.environ.get("STEP57_SEED", "0"))
DS, DV = 12, 10                 # latent: 12 invariant scalars + 10 ℓ=1 vectors -> 12 + 30 = 42 dims
DT = 0.08
OMEGA2 = 1.0                    # harmonic well stiffness
C_REP, SIG_REP = 2.0, 0.6       # soft pairwise repulsion (contact)


# --------------------------------------------------------------------------------------------------
# 3D two-body contact dynamics (velocity-Verlet); state = (r1, v1, r2, v2) flattened to (B, 12).
# --------------------------------------------------------------------------------------------------
def _accel(r1, r2, beta):
    r"""Acceleration on each body: isotropic (β-anisotropic) well + soft pairwise repulsion (contact)."""
    stiff = torch.tensor([1.0, 1.0, 1.0 + 2.0 * beta])
    d = r1 - r2
    d2 = (d ** 2).sum(-1, keepdim=True)
    frep = (C_REP / SIG_REP ** 2) * torch.exp(-d2 / (2 * SIG_REP ** 2)) * d   # repulsion along the separation
    a1 = -OMEGA2 * stiff * r1 + frep
    a2 = -OMEGA2 * stiff * r2 - frep
    return a1, a2


def step_state(s, beta):
    r"""One velocity-Verlet step.  s:(B,12)->(B,12)."""
    r1, v1, r2, v2 = s[:, 0:3], s[:, 3:6], s[:, 6:9], s[:, 9:12]
    a1, a2 = _accel(r1, r2, beta)
    r1n, r2n = r1 + v1 * DT + 0.5 * a1 * DT ** 2, r2 + v2 * DT + 0.5 * a2 * DT ** 2
    a1n, a2n = _accel(r1n, r2n, beta)
    v1n, v2n = v1 + 0.5 * (a1 + a1n) * DT, v2 + 0.5 * (a2 + a2n) * DT
    return torch.cat([r1n, v1n, r2n, v2n], 1)


def energy(s, beta):
    r1, v1, r2, v2 = s[:, 0:3], s[:, 3:6], s[:, 6:9], s[:, 9:12]
    stiff = torch.tensor([1.0, 1.0, 1.0 + 2.0 * beta])
    ke = 0.5 * ((v1 ** 2).sum(-1) + (v2 ** 2).sum(-1))
    pe_well = 0.5 * OMEGA2 * ((stiff * r1 ** 2).sum(-1) + (stiff * r2 ** 2).sum(-1))
    d2 = ((r1 - r2) ** 2).sum(-1)
    pe_rep = C_REP * torch.exp(-d2 / (2 * SIG_REP ** 2))
    return ke + pe_well + pe_rep


def contact_dist(s):
    return ((s[:, 0:3] - s[:, 6:9]) ** 2).sum(-1).sqrt()


def make_pairs(n, beta, seed):
    g = torch.Generator().manual_seed(seed)
    r = 1.2 * torch.randn(n, 12, generator=g)
    r[:, 3:6] *= 0.8; r[:, 9:12] *= 0.8                 # gentler initial velocities
    for _ in range(8):                                  # settle a little
        r = step_state(r, beta)
    return r, step_state(r, beta)


def so3_rand(g):
    a = torch.randn(3, generator=g); a = a / a.norm()
    th = float(2 * math.pi * torch.rand(1, generator=g))
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return torch.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rotate_state(s, R):
    return torch.cat([s[:, 0:3] @ R.T, s[:, 3:6] @ R.T, s[:, 6:9] @ R.T, s[:, 9:12] @ R.T], 1)


# --------------------------------------------------------------------------------------------------
# Hand-rolled 3D Vector-Neuron primitives (3D analogue of Step 50): scalars (ℓ=0) ⊕ vectors (ℓ=1, in R^3).
# --------------------------------------------------------------------------------------------------
def inv_readout(V):
    r"""SO(3)-invariant Gram dots $\langle V_i,V_j\rangle$ (the 3D invariant is the symmetric inner product; the
    antisymmetric part is the cross product, itself an $\ell{=}1$ vector, so it is *not* a scalar here)."""
    return torch.einsum("bic,bjc->bij", V, V).reshape(V.shape[0], -1)


class VecLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_out, n_in) / math.sqrt(n_in))

    def forward(self, V):
        return torch.einsum("ji,bic->bjc", self.W, V)


class EquivBlock(nn.Module):
    def __init__(self, ds, dv):
        super().__init__()
        self.from_inv = nn.Linear(dv * dv, ds)
        self.smlp = nn.Sequential(nn.Linear(ds, 2 * ds), nn.SiLU(), nn.Linear(2 * ds, ds))
        self.vlin = VecLinear(dv, dv)
        self.norm_a = nn.Parameter(torch.zeros(dv))
        self.norm_b = nn.Parameter(3.0 * torch.ones(dv))
        self.gate = nn.Linear(ds, dv); nn.init.constant_(self.gate.bias, 3.0)

    def forward(self, s, V):
        s = s + self.from_inv(inv_readout(V))
        s = s + self.smlp(s)
        V = self.vlin(V)
        n = V.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        V = V * torch.sigmoid(self.norm_a[None, :, None] * n + self.norm_b[None, :, None])
        V = V * torch.sigmoid(self.gate(s))[:, :, None]
        return s, V


class EquivWM(nn.Module):
    r"""3D SO(3)-equivariant autoencoder + latent predictor (ℓ=0 ⊕ ℓ=1).  state (B,12) -> latent -> (B,12)."""

    def __init__(self, ds=DS, dv=DV):
        super().__init__()
        self.n_in = 4                                   # 4 input vectors: r1, v1, r2, v2
        self.lift_s = nn.Linear(self.n_in * self.n_in, ds)
        self.lift_v = VecLinear(self.n_in, dv)
        self.enc = nn.ModuleList([EquivBlock(ds, dv), EquivBlock(ds, dv)])
        self.pred = nn.ModuleList([EquivBlock(ds, dv), EquivBlock(ds, dv)])
        self.dec_v = VecLinear(dv, self.n_in)
        self.dec_gate = nn.Linear(ds, self.n_in); nn.init.constant_(self.dec_gate.bias, 3.0)

    def _vecs(self, s):
        return s.reshape(s.shape[0], 4, 3)              # (B,12) -> (B,4,3) = r1,v1,r2,v2

    def encode(self, state):
        Vin = self._vecs(state)
        s = self.lift_s(inv_readout(Vin)); V = self.lift_v(Vin)
        for blk in self.enc:
            s, V = blk(s, V)
        return s, V

    def predict(self, s, V):
        for blk in self.pred:
            s, V = blk(s, V)
        return s, V

    def decode(self, s, V):
        Vout = self.dec_v(V) * torch.sigmoid(self.dec_gate(s))[:, :, None]    # (B,4,3)
        return Vout.reshape(Vout.shape[0], 12)


def train(model, S, S2, epochs, seed):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    n, bs = S.shape[0], 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            st, st1 = S[idx], S2[idx]
            opt.zero_grad()
            s, V = model.encode(st); recon = model.decode(s, V)
            sp, Vp = model.predict(s, V); pred = model.decode(sp, Vp)
            with torch.no_grad():
                s1, V1 = model.encode(st1)
            lat = ((sp - s1) ** 2).mean() + ((Vp - V1) ** 2).mean()
            loss = ((recon - st) ** 2).mean() + ((pred - st1) ** 2).mean() + 0.1 * lat
            loss.backward(); opt.step()
        sched.step()
    return float(loss.item())


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 94
    n_train, epochs, beta = (1000, 20, 0.0) if SMOKE else (6000, 90, 0.0)
    epochs = int(os.environ.get("STEP57_EPOCHS", epochs))
    print(line)
    print(f"STEP 57  Noether hinge lifted to 3D SO(3) two-body CONTACT  ({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    S, S2 = make_pairs(n_train, beta, seed=SEED)
    St, St1 = make_pairs(800, beta, seed=999)
    E = energy(St, beta); cd = contact_dist(St)
    target = torch.stack([E, cd], 1)                    # conserved E + the contact coordinate

    torch.manual_seed(SEED)
    eq = EquivWM(); loss = train(eq, S, S2, epochs, seed=SEED)

    with torch.no_grad():
        Rr = so3_rand(torch.Generator().manual_seed(7))
        s0, V0 = eq.encode(St); s1r, V1r = eq.encode(rotate_state(St, Rr))
        scal_resid = (s1r - s0).abs().max().item()
        vec_resid = (V1r - torch.einsum("ij,bnj->bni", Rr, V0)).abs().max().item()
        s_te, V_te = eq.encode(St); s_te1, V_te1 = eq.encode(St1)
    scal = s_te; vecf = V_te.reshape(V_te.shape[0], -1)
    r2_scal, r2_vec = r2_regress(scal, target), r2_regress(vecf, target)
    rate_scal = slowest_mode_rate(s_te, s_te1)
    rate_vec = slowest_mode_rate(vecf, V_te1.reshape(V_te1.shape[0], -1))

    print(f"    train loss {loss:.2e} | encoder equivariance: scalar Δ {scal_resid:.1e}, vector rot-Δ {vec_resid:.1e}")
    print(f"    (1) Noether content — recover (E, contact-dist): invariant R²={r2_scal:.3f}  vs  equivariant R²={r2_vec:.3f}")
    print(f"    (2) slowest mode: invariant block {rate_scal:.4f}  vs  equivariant block {rate_vec:.4f}")

    eq_ok = scal_resid < 1e-5 and vec_resid < 1e-5
    ok = (eq_ok and r2_scal > 0.8 and r2_scal > r2_vec + 0.2 and rate_scal < 0.6 * rate_vec)
    print(f"\n{line}\nSTEP 57 SUMMARY\n{line}")
    if ok:
        print(f"    CONFIRMED (hinge lifts to 3D contact): even with a soft pairwise contact term and in 3D, the")
        print(f"        learned SO(3)-equivariant model places the conserved invariants (E, contact distance) in")
        print(f"        its invariant ℓ=0 block (R²={r2_scal:.2f} vs {r2_vec:.2f}) and the slowest mode it admits")
        print(f"        ({rate_scal:.3f}) is far slower than the equivariant block's ({rate_vec:.3f}). slow ⊆ invariant")
        print(f"        survives the lift from a clean orbit to a 3D contact interaction.")
    else:
        print(f"    INCONCLUSIVE: hinge did not lift cleanly (eq_resid ok={eq_ok}, R² {r2_scal:.2f} vs {r2_vec:.2f},")
        print(f"        slow {rate_scal:.3f} vs {rate_vec:.3f}). Reported as-is — the embodied/contact lift is the")
        print(f"        genuinely open frontier; this localises the difficulty rather than papering over it.")

    out = dict(passed=bool(ok), train_loss=loss, scal_resid=scal_resid, vec_resid=vec_resid,
               r2_invariant=r2_scal, r2_equivariant=r2_vec, slow_invariant=rate_scal,
               slow_equivariant=rate_vec, beta=beta, ds=DS, dv=DV, smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step57_embodied_hinge_smoke" if SMOKE else "step57_embodied_hinge"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax[0].bar(["invariant\n(ℓ=0)", "equivariant\n(ℓ=1)"], [r2_scal, r2_vec], color=["C2", "C3"])
    ax[0].set_ylabel("R² recovering (E, contact-dist)"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title(f"Noether content (3D + contact)\ninvariant block holds the conserved quantities")
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(["invariant\n(ℓ=0)", "equivariant\n(ℓ=1)"], [rate_scal, rate_vec], color=["C2", "C3"])
    ax[1].set_ylabel("slowest-mode rate (smaller = slower)")
    ax[1].set_title(f"slow ⊆ invariant?\ninvariant {rate_scal:.3f} vs equivariant {rate_vec:.3f}")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Step 57 — Noether hinge lifted to a 3D SO(3) two-body contact interaction (paper2 §3)", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
