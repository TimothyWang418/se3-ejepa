r"""Step 101 — the S₂-equivariant TD-MPC2 latent (core module for the online bridge).

Spec: `docs/specs/2026-06-13-step101-online-tdmpc2-bridge-seed.md`. This file is the *science* artifact —
the equivariant encoder/dynamics/reward/policy over TD-MPC2's 512-d SimNorm latent — decoupled from the
heavy training stack (dm_control/tensordict/hydra), so it is unit-testable with torch alone. The thin
wiring into the official trainer (subclass WorldModel, swap these in, add a `model=eq|dense` cfg flag)
is `external/tdmpc2/.../common/eq_world_model.py`, applied at deploy time.

S₂ = walker leg-exchange (verified EXACT, step100 G-SYM 4.75e-16). Layout (single-task walker):
  obs 24:  torso T = [0,1,14,15,16,17]; right leg R = [2..8)+[18..21); left leg L = [8..14)+[21..24)
  action 6: aR = [0:3], aL = [3:6];  rho_a swaps them
  latent 512 = 64 SimNorm groups of 8:  INV = 32 groups (256-d) | R = 16 groups (128-d) | L = 16 (128-d)
              rho_z swaps the R and L blocks, identity on INV.  SimNorm is group-wise so it commutes with
              this whole-group block permutation. Equivariance is exact by weight-tying (no e3nn).

Run (G-EQ self-test, torch only):  .venv/bin/python experiments/step101_eq_world_model.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step89_pretrained_wm_audit import NormedLinear, SimNorm, mlp  # TD-MPC2 layer replicas  # noqa: E402

# ---- dimensions (TD-MPC2 walker single-task defaults) ----
OBS, ACT, LAT, SIMG = 24, 6, 512, 8
ENC_DIM, MLP_DIM = 256, 512
N_GROUP = LAT // SIMG                       # 64 SimNorm groups
INV_G, LEG_G = 32, 16                       # 32 invariant + 16 R + 16 L  (32+16+16 = 64)
INV_D, LEG_D = INV_G * SIMG, LEG_G * SIMG   # 256, 128, 128

# obs/action index layout (from step100)
T_IDX = [0, 1, 14, 15, 16, 17]
R_IDX = list(range(2, 8)) + list(range(18, 21))
L_IDX = list(range(8, 14)) + list(range(21, 24))
ACT_R, ACT_L = [0, 1, 2], [3, 4, 5]


def rho_obs(o):
    out = o.clone()
    out[..., R_IDX], out[..., L_IDX] = o[..., L_IDX], o[..., R_IDX]
    return out


def rho_act(a):
    out = a.clone()
    out[..., ACT_R], out[..., ACT_L] = a[..., ACT_L], a[..., ACT_R]
    return out


def rho_z(z):
    """Swap the R-block and L-block of the 512-d latent; identity on the invariant block."""
    inv, r, l = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
    return torch.cat([inv, l, r], dim=-1)


def _simnorm(x):
    return SimNorm(SIMG)(x)  # group-wise softmax over groups of 8 (commutes with whole-group permutation)


class EqEncoder(nn.Module):
    """obs(24) -> latent(512) in (inv|R|L) layout, S₂-equivariant by tied leg subnet."""

    def __init__(s):
        super().__init__()
        s.leg = mlp(len(R_IDX) + len(T_IDX), [ENC_DIM], LEG_D)        # (leg, torso) -> leg-latent pre-SimNorm
        s.inv = mlp(len(T_IDX) + LEG_D, [ENC_DIM], INV_D)            # (torso, pooled legs) -> inv pre-SimNorm

    def forward(s, o):
        t, r, l = o[..., T_IDX], o[..., R_IDX], o[..., L_IDX]
        zr = s.leg(torch.cat([r, t], -1))
        zl = s.leg(torch.cat([l, t], -1))
        zi = s.inv(torch.cat([t, zr + zl], -1))                      # symmetric pool -> invariant
        return _simnorm(torch.cat([zi, zr, zl], -1))


class EqDynamics(nn.Module):
    """(z 512, a 6) -> next z 512, S₂-equivariant (tied per-leg + symmetric-pooled invariant)."""

    def __init__(s):
        super().__init__()
        s.leg = mlp(LEG_D + INV_D + LEG_D + 3 + 3, [MLP_DIM], LEG_D)  # (z_leg, z_inv, pool, a_leg, a_pool)
        s.inv = mlp(INV_D + LEG_D + 3, [MLP_DIM], INV_D)             # (z_inv, pool, a_pool)

    def forward(s, z, a):
        zi, zr, zl = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
        ar, al = a[..., ACT_R], a[..., ACT_L]
        pool, apool = zr + zl, ar + al
        zr2 = s.leg(torch.cat([zr, zi, pool, ar, apool], -1))
        zl2 = s.leg(torch.cat([zl, zi, pool, al, apool], -1))
        zi2 = s.inv(torch.cat([zi, pool, apool], -1))
        return _simnorm(torch.cat([zi2, zr2, zl2], -1))


class EqReward(nn.Module):
    """(z 512, a 6) -> reward logits(num_bins), S₂-INVARIANT (reads symmetric pools only)."""

    def __init__(s, num_bins=101):
        super().__init__()
        s.net = mlp(INV_D + LEG_D + 3, [MLP_DIM], num_bins)          # (z_inv, z_pool, a_pool)

    def forward(s, z, a):
        zi, zr, zl = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
        ar, al = a[..., ACT_R], a[..., ACT_L]
        return s.net(torch.cat([zi, zr + zl, ar + al], -1))


class EqPi(nn.Module):
    """z 512 -> 12 = [meanR,meanL,logstdR,logstdL], S₂-EQUIVARIANT (tied leg head)."""

    def __init__(s):
        super().__init__()
        s.head = mlp(LEG_D + INV_D, [MLP_DIM], 6)                    # (z_leg, z_inv) -> (mean3, logstd3)

    def forward(s, z):
        zi, zr, zl = z[..., :INV_D], z[..., INV_D:INV_D + LEG_D], z[..., INV_D + LEG_D:]
        hr, hl = s.head(torch.cat([zr, zi], -1)), s.head(torch.cat([zl, zi], -1))
        mr, sr = hr[..., :3], hr[..., 3:]
        ml, sl = hl[..., :3], hl[..., 3:]
        return torch.cat([mr, ml, sr, sl], -1)                       # TD-MPC2 chunks into mean[0:6], logstd[6:12]


def g_eq_test(seed=0, tol=1e-5):
    """Pre-registered G-EQ: every module's equivariance/invariance defect must be <= tol (weight-tied)."""
    torch.manual_seed(seed)
    enc, dyn, rew, pi = EqEncoder(), EqDynamics(), EqReward(), EqPi()
    o = torch.randn(64, OBS)
    a = torch.randn(64, ACT)
    z = enc(o)
    rz = lambda x: x.norm() + 1e-9
    d_enc = (enc(rho_obs(o)) - rho_z(z)).norm() / rz(z)
    d_dyn = (dyn(rho_z(z), rho_act(a)) - rho_z(dyn(z, a))).norm() / rz(dyn(z, a))
    d_rew = (rew(rho_z(z), rho_act(a)) - rew(z, a)).norm() / rz(rew(z, a))           # invariant
    # pi equivariance: swapping legs must swap (meanR<->meanL, logstdR<->logstdL)
    p = pi(z)
    mr, ml, sr, sl = p[..., 0:3], p[..., 3:6], p[..., 6:9], p[..., 9:12]
    p_swap_expected = torch.cat([ml, mr, sl, sr], -1)
    d_pi = (pi(rho_z(z)) - p_swap_expected).norm() / rz(p)
    res = {"enc": float(d_enc), "dyn": float(d_dyn), "reward_inv": float(d_rew), "pi": float(d_pi)}
    ok = all(v <= tol for v in res.values())
    print(f"[step101 G-EQ] {res}  -> {'PASS' if ok else 'FAIL'} (tol {tol})", file=sys.stderr)
    # SimNorm simplex sanity: each group of 8 sums to 1
    grp = z.view(64, N_GROUP, SIMG).sum(-1)
    print(f"[step101] SimNorm group-sum range [{float(grp.min()):.4f},{float(grp.max()):.4f}] (expect ~1)",
          file=sys.stderr)
    return res, ok


if __name__ == "__main__":
    _, ok = g_eq_test()
    raise SystemExit(0 if ok else 1)
