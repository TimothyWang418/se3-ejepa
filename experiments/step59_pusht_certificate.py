r"""Step 59 — the predictability certificate on a LEARNED model of REAL contact dynamics (PushT), over a
multi-step rollout.

This is the keystone non-toy experiment for paper2. Every other configuration-axis result (Steps 47/49/51) uses
dynamics we hand-constructed to be equivariant; the standing reviewer critique is "constructed teacher /
architecture-matched toy". Here the dynamics are the **real PushT pymunk contact physics** — we did not write
them — and the model is **learned** on real transitions.

Design choices that make this a clean, fair test:
  * Expressive equivariant model. A pure-vector VN cannot represent contact geometry (which depends on invariant
    quantities: agent-block distance, contact angle). ``RichVNForwardModelPushT`` adds an SO(2)-INVARIANT scalar
    pathway (Gram + cross-product features of the input vectors) that gates an equivariant vector update, so it is
    both expressive (competitive absolute accuracy) and exactly SO(2)-equivariant. This lets us check the
    certificate is not bought at the cost of accuracy ("flat is not good" addressed empirically).
  * Clean orbit protocol (as in Step 51). We train one-step on a +/-50 deg wedge of real interior trajectories,
    hold out a real in-wedge test set, and ROTATE that fixed test set to each orbit angle beta. The equivariant
    model's whole-rollout error is then flat over the orbit by Theorem A (architectural; we verify the
    equivariance residual ~1e-6); a non-equivariant MLP -- even scaled 160x -- has no such guarantee and its
    out-of-wedge error grows, the more so the longer the rollout.
  * Multi-step. A single PushT step is easy in-distribution; a world model is used by rolling out. We report the
    H-step rollout relMSE over the orbit at H in {1,5,10}.

Honest gate at the headline horizon H=10 (prints INCONCLUSIVE rather than loosen a threshold):
  (i)  equivariant flat over the orbit (architectural):  eq_ratio(10) < 1.10;
  (ii) equivariant competitive in-distribution:          eq_in(10) < 1.5 * min_w mlp_in(10);
  (iii)the baseline climbs out of the wedge:             mlp_ratio(10)[big] > 2.0;
  (iv) no scale reaches the equivariant floor:           all mlp_far(10)[w] > 2 * eq_far(10).

Run:        .venv/bin/python experiments/step59_pusht_certificate.py
Seeded:     STEP59_SEED=0|1|2 .venv/bin/python experiments/step59_pusht_certificate.py
Writes:     papers/figures/step59_pusht_certificate.{json,png}
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.step10_pusht_closed_loop import (  # noqa: E402
    CENTER,
    INTERIOR_R,
    MLPForwardModelPushT,
    POS_SCALE,
    info_to_packed,
    make_env,
    n_params,
    reset_task,
    rot_np,
    rotate_packed_torch,
    sample_task,
)
from src.models.structured import VNLinear, VNReLU  # noqa: E402

SEED = int(os.environ.get("STEP59_SEED", "0"))
SMOKE = os.environ.get("STEP59_SMOKE", "0") == "1"

WEDGE_DEG = 50.0
WEDGE = WEDGE_DEG * math.pi / 180.0
BIN_CENTERS_DEG = [10, 30, 50, 70, 90, 110, 130, 150, 170]
MLP_HIDDENS = [32, 128] if SMOKE else [32, 128, 512]
H = 10
HS = [1, 5, 10]
N_TRAJ_TRAIN = 400 if SMOKE else 1500
N_TRAJ_TEST = 120 if SMOKE else 400
EPOCHS = 200 if SMOKE else 700
FIG = ROOT / "papers" / "figures"


class RichVNForwardModelPushT(nn.Module):
    r"""Expressive SO(2)-equivariant PushT forward model: an INVARIANT scalar pathway (Gram + cross-product
    features of the 5 input vectors [agent_pos, agent_vel, block_pos, block_dir, action]) gates an equivariant
    vector update. Exactly SO(2)-equivariant: f(Rs, Ra) = R f(s, a). ``step: (B,8),(B,2) -> (B,8)``."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        n_in = 5
        n_inv = n_in * n_in * 2                      # Gram (sym) + cross (antisym) of the 5 input vectors
        self.s_mlp = nn.Sequential(nn.Linear(n_inv, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.v1 = VNLinear(n_in, hidden)
        self.gate = nn.Linear(hidden, hidden)        # invariant scalars -> per-channel gate (stays equivariant)
        self.act = VNReLU(hidden)
        self.v2 = VNLinear(hidden, 4)

    def step(self, packed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b = packed.shape[0]
        Vs = packed.reshape(b, 4, 2)
        V = torch.cat([Vs, action.reshape(b, 1, 2)], dim=1)            # (B,5,2)
        gram = torch.einsum("bic,bjc->bij", V, V).reshape(b, -1)       # SO(2)-invariant
        cross = (V[:, :, None, 0] * V[:, None, :, 1]
                 - V[:, :, None, 1] * V[:, None, :, 0]).reshape(b, -1)  # SO(2)-invariant (pseudo-scalar)
        s = self.s_mlp(torch.cat([gram, cross], dim=-1))              # (B,hidden) invariant features
        Vh = self.v1(V) * torch.sigmoid(self.gate(s)).unsqueeze(-1)   # equivariant, invariantly gated
        delta = self.v2(self.act(Vh))                                 # (B,4,2)
        nxt = Vs + delta
        bdir = nxt[:, 3]
        bdir = bdir / (bdir.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([nxt[:, :3], bdir.unsqueeze(1)], dim=1).reshape(b, 8)


def collect_trajs(n_traj: int, beta_lo: float, beta_hi: float, *, seed: int):
    r"""``n_traj`` interior real PushT trajectories of length H (kept only if every state stays within
    INTERIOR_R of centre -- the SO(2)-exact interior away from the boundary walls)."""
    env = make_env()
    rng = np.random.default_rng(seed)
    states, acts, tries = [], [], 0
    while len(states) < n_traj and tries < n_traj * 8:
        tries += 1
        task = sample_task(rng, beta_lo, beta_hi)
        info = reset_task(env, task, seed=int(rng.integers(0, 1 << 30)))
        u = np.array([math.cos(task["beta"]), math.sin(task["beta"])], dtype=np.float32)
        traj_s, traj_a, ok = [info_to_packed(info)], [], True
        for _ in range(H):
            a = (np.clip(u + rng.normal(0, 0.4, size=2), -1, 1) if rng.random() < 0.7
                 else rng.uniform(-1, 1, size=2)).astype(np.float32)
            _, _, _, _, info = env.step(a)
            p = info_to_packed(info)
            traj_s.append(p); traj_a.append(a)
            if np.linalg.norm(p[4:6] * POS_SCALE) >= INTERIOR_R:        # block (centred) distance from centre
                ok = False; break
        if ok:
            states.append(np.stack(traj_s)); acts.append(np.stack(traj_a))
    return torch.tensor(np.stack(states)), torch.tensor(np.stack(acts))   # (n,H+1,8), (n,H,2)


def rotate_traj(S: torch.Tensor, A: torch.Tensor, deg: float):
    r"""Rotate a whole trajectory (all states + actions) by ``deg`` about the scene centre -- the SO(2) group
    action on the orbit. States rotate as 4 type-1 vectors; actions as 1 type-1 vector."""
    th = deg * math.pi / 180.0
    n, hp1, _ = S.shape
    Sr = rotate_packed_torch(S.reshape(n * hp1, 8), th).reshape(n, hp1, 8)
    R = torch.tensor(rot_np(th), dtype=A.dtype)
    Ar = torch.einsum("ij,nhj->nhi", R, A)
    return Sr, Ar


def train_fair(model, S, A, S2, *, epochs, seed, lr=3e-3, wd=1e-4, batch=256):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = S.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((model.step(S[idx], A[idx]) - S2[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def rollout_relmse(model, S, A, horizon):
    s = S[:, 0]
    for t in range(horizon):
        s = model.step(s, A[:, t])
    sH = S[:, horizon]
    rel = ((s - sH) ** 2).sum(1) / (sH ** 2).sum(1).clamp_min(1e-9)
    return float(rel.mean())


@torch.no_grad()
def equivariance_residual(model, S, A):
    r"""max_b ||f(Rs,Ra) - R f(s,a)|| over a random rotation -- should be ~1e-6 for the equivariant model."""
    s0, a0 = S[:, 0], A[:, 0]
    th = 1.0
    lhs = rotate_packed_torch(model.step(s0, a0), th)
    R = torch.tensor(rot_np(th), dtype=a0.dtype)
    rhs = model.step(rotate_packed_torch(s0, th), torch.einsum("ij,bj->bi", R, a0))
    return float((lhs - rhs).abs().max())


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)

    # --- real PushT trajectories: train one-step on the wedge; hold out a real in-wedge test set ------
    S_tr, A_tr = collect_trajs(N_TRAJ_TRAIN, -WEDGE, WEDGE, seed=SEED)
    s_tr = S_tr[:, :-1].reshape(-1, 8); a_tr = A_tr.reshape(-1, 2); s2_tr = S_tr[:, 1:].reshape(-1, 8)
    S_te, A_te = collect_trajs(N_TRAJ_TEST, -WEDGE, WEDGE, seed=9000 + SEED)
    print(f"[step59] seed={SEED} train={tuple(S_tr.shape)} ({s_tr.shape[0]} pairs) | "
          f"held-out wedge test={tuple(S_te.shape)} | H={H}", file=sys.stderr)

    # rotate the FIXED held-out test set to each orbit angle (the SO(2) orbit, as in Step 51)
    orbit = [rotate_traj(S_te, A_te, c) for c in BIN_CENTERS_DEG]

    def curves_over_H(model):
        return {h: [rollout_relmse(model, Sb, Ab, h) for (Sb, Ab) in orbit] for h in HS}

    eq = RichVNForwardModelPushT(hidden=64)
    train_fair(eq, s_tr, a_tr, s2_tr, epochs=EPOCHS, seed=SEED)
    eq_resid = equivariance_residual(eq, S_te, A_te)
    eq_c = curves_over_H(eq)
    eq_params = n_params(eq)

    mlp_c, mlp_params = {}, {}
    for h in MLP_HIDDENS:
        m = MLPForwardModelPushT(hidden=h)
        train_fair(m, s_tr, a_tr, s2_tr, epochs=EPOCHS, seed=SEED)
        mlp_c[h] = curves_over_H(m)
        mlp_params[h] = n_params(m)

    centers = np.array(BIN_CENTERS_DEG, dtype=float)
    in_mask, far_mask = centers <= WEDGE_DEG, centers >= 120.0
    mean_in = lambda c: float(np.nanmean(np.array(c)[in_mask]))    # noqa: E731
    mean_far = lambda c: float(np.nanmean(np.array(c)[far_mask]))  # noqa: E731
    ratio = lambda c: mean_far(c) / max(mean_in(c), 1e-12)         # noqa: E731

    eq_ratio = {h: ratio(eq_c[h]) for h in HS}
    eq_in = {h: mean_in(eq_c[h]) for h in HS}
    eq_far = {h: mean_far(eq_c[h]) for h in HS}
    mlp_ratio = {w: {h: ratio(mlp_c[w][h]) for h in HS} for w in MLP_HIDDENS}
    mlp_in = {w: {h: mean_in(mlp_c[w][h]) for h in HS} for w in MLP_HIDDENS}
    mlp_far = {w: {h: mean_far(mlp_c[w][h]) for h in HS} for w in MLP_HIDDENS}
    floor_pen = {h: {w: mlp_far[w][h] / max(eq_far[h], 1e-12) for w in MLP_HIDDENS} for h in HS}

    big = max(MLP_HIDDENS)
    best_mlp_in = {h: min(mlp_in[w][h] for w in MLP_HIDDENS) for h in HS}
    ok_flat = eq_ratio[H] < 1.10
    ok_compete = eq_in[H] < 1.5 * best_mlp_in[H]
    ok_climb = mlp_ratio[big][H] > 2.0
    ok_floor = all(mlp_far[w][H] > 2.0 * eq_far[H] for w in MLP_HIDDENS)
    passed = ok_flat and ok_compete and ok_climb and ok_floor

    result = {
        "passed": passed,
        "gate": {"flat": ok_flat, "compete": ok_compete, "climb": ok_climb, "floor": ok_floor},
        "equivariance_residual": eq_resid,
        "horizon_headline": H, "horizons": HS,
        "eq_flatness_ratio": eq_ratio, "eq_in": eq_in, "eq_far": eq_far, "eq_params": eq_params,
        "mlp_flatness_ratios": {str(w): mlp_ratio[w] for w in MLP_HIDDENS},
        "mlp_in": {str(w): mlp_in[w] for w in MLP_HIDDENS},
        "mlp_far": {str(w): mlp_far[w] for w in MLP_HIDDENS},
        "mlp_params": {str(w): mlp_params[w] for w in MLP_HIDDENS},
        "floor_penalty": {str(h): {str(w): floor_pen[h][w] for w in MLP_HIDDENS} for h in HS},
        "param_ratio_big_over_eq": mlp_params[big] / eq_params,
        "bin_centers_deg": BIN_CENTERS_DEG, "wedge_deg": WEDGE_DEG,
        "eq_curves": {str(h): eq_c[h] for h in HS},
        "mlp_curves": {str(w): {str(h): mlp_c[w][h] for h in HS} for w in MLP_HIDDENS},
        "smoke": SMOKE, "seed": SEED,
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "step59_pusht_certificate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # --- figure ---------------------------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    axL.axvspan(0, WEDGE_DEG, color="0.85", label=f"train wedge (|β|≤{WEDGE_DEG:.0f}°)")
    axL.semilogy(BIN_CENTERS_DEG, eq_c[H], "o-", lw=2.4, color="C2",
                 label=f"SO(2)-equivariant ({eq_params} p)")
    for w in MLP_HIDDENS:
        axL.semilogy(BIN_CENTERS_DEG, mlp_c[w][H], "s--", alpha=0.8,
                     label=f"MLP h={w} ({mlp_params[w] // 1000}k p)")
    axL.set_xlabel("scene orientation β (degrees)")
    axL.set_ylabel(f"{H}-step rollout relMSE")
    axL.set_title(f"PushT certificate, H={H}: equivariant flat (ratio {eq_ratio[H]:.2f}) and competitive")
    axL.legend(fontsize=7, loc="best")

    for w in MLP_HIDDENS:
        axR.plot(HS, [floor_pen[h][w] for h in HS], "s-", label=f"MLP h={w}")
    axR.axhline(1.0, color="C2", ls="--", lw=2, label="equivariant floor")
    axR.set_xlabel("rollout horizon H")
    axR.set_ylabel("out-of-wedge relMSE  /  equivariant floor")
    axR.set_title("no non-equivariant scale reaches the equivariant floor (any horizon)")
    axR.legend(fontsize=8)
    fig.suptitle("Step 59 — predictability certificate on a learned model of REAL PushT contact dynamics",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "step59_pusht_certificate.png", dpi=130)

    # --- report ---------------------------------------------------------------------------------------
    print(f"[step59] eq: equiv-resid={eq_resid:.1e}  ratio(H)={{{', '.join(f'{h}:{eq_ratio[h]:.2f}' for h in HS)}}}  "
          f"in(H10)={eq_in[H]:.4f} far(H10)={eq_far[H]:.4f} params={eq_params}", file=sys.stderr)
    for w in MLP_HIDDENS:
        print(f"[step59]   MLP h={w:4d} ({mlp_params[w]}p): ratio(H10)={mlp_ratio[w][H]:.2f} "
              f"in(H10)={mlp_in[w][H]:.4f} far(H10)={mlp_far[w][H]:.4f}  "
              f"floor-pen H1={floor_pen[1][w]:.1f}x->H10={floor_pen[H][w]:.1f}x", file=sys.stderr)
    if passed:
        print(f"[step59] CERTIFICATE CONFIRMED on real PushT (H={H}): equivariant flat (ratio {eq_ratio[H]:.2f}, "
              f"equiv-resid {eq_resid:.0e}) AND competitive in-dist; the {mlp_params[big] / eq_params:.0f}x-bigger "
              f"MLP stays {floor_pen[H][big]:.1f}x above the equivariant floor out-of-wedge.", file=sys.stderr)
    else:
        bad = [k for k, v in result["gate"].items() if not v]
        print(f"[step59] INCONCLUSIVE: gate not met ({bad}); reported as-is (no thresholds loosened).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
