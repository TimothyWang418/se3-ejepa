r"""Step 100b — train the three arms (eq / dense / dense+aug) and audit their certificates.

Spec: `docs/specs/2026-06-12-step100-walker-s2-equivariant-seed.md` (frozen). G-SYM verdict: the true
walker dynamics are EXACTLY S2-equivariant (median defect 4.75e-16, `step100_gsym_probe.json`), so the
exact-symmetry reading applies — no approximate-symmetry caveat is needed for the world side.

Arms (identical data, identical recipe, param-matched):
  eq        — S2-equivariant by weight tying: obs splits into torso block t (6) and leg blocks
              r, l (9 each); encoder z = (z_inv, z_R, z_L) with a SHARED leg subnet; dynamics with a
              shared per-leg map + a symmetric-pooled invariant map; equivariance is exact by
              construction: rho_z (z_inv, z_R, z_L) = (z_inv, z_L, z_R).
  dense     — unconstrained MLPs at matched latent dim and ~matched parameter count.
  dense+aug — the dense architecture, each batch augmented with the swapped copy (rho obs, rho act).

Loop closure for the audit: a distilled policy head pi_hat(z) (per-arm, same recipe; equivariant by
tying for the eq arm), giving the autonomous loop g(z) = d(z, pi_hat(z)) — audited with the same
Benettin-QR + block-bootstrap + measured-crossing protocol as step89 (100 mid-episode starts).

Run:  STEP100_SEEDS=0,1,2 STEP100_ARMS=eq,dense,aug .venv/bin/python experiments/step100b_train_audit.py
Writes papers/figures/step100_walker_s2_results.json   (incremental per (arm, seed); resumable)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step100_walker_s2 as s100  # noqa: E402  (rho_*, make_env, load_policy)
import step89_pretrained_wm_audit as s89  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers/figures/step100_walker_s2_results.json"
DATA = ROOT / "data/step100_walker_transitions.npz"

SEEDS = [int(x) for x in os.environ.get("STEP100_SEEDS", "0").split(",")]
ARMS = os.environ.get("STEP100_ARMS", "eq,dense,aug").split(",")
SMOKE = bool(int(os.environ.get("STEP100_SMOKE", "0")))
K_ROLL = 5
LAT_LEG = int(os.environ.get("STEP100_LAT", "16"))
LAT_INV = LAT_LEG                      # z = (z_inv, z_R, z_L); 16 -> 48-d, 32 -> 96-d
HID = int(os.environ.get("STEP100_HID", "64"))
EPOCHS = 2 if SMOKE else int(os.environ.get("STEP100_EPOCHS", "400"))
EPS_LIST = [0.05, 0.1, 0.2]

# obs index blocks (see step100 header): torso t = orient[0:2] + height[14] + rootvel[15:18]
T_IDX = [0, 1, 14, 15, 16, 17]
R_IDX = list(range(2, 8)) + list(range(18, 21))
L_IDX = list(range(8, 14)) + list(range(21, 24))


def split_obs(o):                  # o: (B, 24) -> t (B,6), r (B,9), l (B,9)
    return o[:, T_IDX], o[:, R_IDX], o[:, L_IDX]


def mlp(i, o, h, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class EqWM(nn.Module):
    """S2-equivariant world model: shared leg subnets + symmetric pooling everywhere."""

    def __init__(s):
        super().__init__()
        s.enc_leg = mlp(9 + 6, LAT_LEG, HID)            # (leg, torso) -> z_leg
        s.enc_inv = mlp(6 + LAT_LEG, LAT_INV, HID)      # (torso, pooled legs) -> z_inv
        s.dyn_leg = mlp(LAT_LEG + LAT_INV + LAT_LEG + 3 + 3, LAT_LEG, HID + HID // 2)   # (z_i, z_inv, pool, a_i, a_pool)
        s.dyn_inv = mlp(LAT_INV + LAT_LEG + 3, LAT_INV, HID + HID // 2)                 # (z_inv, pool, a_pool)
        s.dec_leg = mlp(LAT_LEG + LAT_INV, 9, HID)
        s.dec_inv = mlp(LAT_INV + LAT_LEG, 6, HID)

    def encode(s, o):
        t, r, l = split_obs(o)
        zr = s.enc_leg(torch.cat([r, t], -1))
        zl = s.enc_leg(torch.cat([l, t], -1))
        zi = s.enc_inv(torch.cat([t, zr + zl], -1))
        return torch.cat([zi, zr, zl], -1)

    def dynamics(s, z, a):
        zi, zr, zl = z[:, :LAT_INV], z[:, LAT_INV:LAT_INV + LAT_LEG], z[:, LAT_INV + LAT_LEG:]
        ar, al = a[:, :3], a[:, 3:]
        pool, apool = zr + zl, ar + al
        zr2 = s.dyn_leg(torch.cat([zr, zi, pool, ar, apool], -1))
        zl2 = s.dyn_leg(torch.cat([zl, zi, pool, al, apool], -1))
        zi2 = s.dyn_inv(torch.cat([zi, pool, apool], -1))
        return torch.cat([zi2, zr2, zl2], -1)

    def decode(s, z):
        zi, zr, zl = z[:, :LAT_INV], z[:, LAT_INV:LAT_INV + LAT_LEG], z[:, LAT_INV + LAT_LEG:]
        t = s.dec_inv(torch.cat([zi, zr + zl], -1))
        r = s.dec_leg(torch.cat([zr, zi], -1))
        l = s.dec_leg(torch.cat([zl, zi], -1))
        o = torch.zeros(z.shape[0], 24, dtype=z.dtype, device=z.device)
        o[:, T_IDX], o[:, R_IDX], o[:, L_IDX] = t, r, l
        return o


class DenseWM(nn.Module):
    def __init__(s):
        super().__init__()
        D = LAT_INV + 2 * LAT_LEG
        s.enc = mlp(24, D, HID * 5 // 4)
        s.dyn = mlp(D + 6, D, HID * 2 + HID // 8)
        s.dec = mlp(D, 24, HID * 5 // 4)

    def encode(s, o):
        return s.enc(o)

    def dynamics(s, z, a):
        return s.dyn(torch.cat([z, a], -1))

    def decode(s, z):
        return s.dec(z)


def rho_z(z):
    zi, zr, zl = z[:, :LAT_INV], z[:, LAT_INV:LAT_INV + LAT_LEG], z[:, LAT_INV + LAT_LEG:]
    return torch.cat([zi, zl, zr], -1)


def equivariance_defect(model, obs, act):
    with torch.no_grad():
        z = model.encode(obs)
        a1 = model.dynamics(rho_z(z), torch.as_tensor(s100.rho_act(act.numpy())))
        a2 = rho_z(model.dynamics(z, act))
        zr = model.encode(torch.as_tensor(s100.rho_obs(obs.numpy())))
        enc_def = (zr - rho_z(z)).norm() / (z.norm() + 1e-12)
        dyn_def = (a1 - a2).norm() / (a2.norm() + 1e-12)
    return float(enc_def), float(dyn_def)


def train_arm(arm, seed, obs, act, nxt, mu, sd):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EqWM() if arm == "eq" else DenseWM()
    model = model.double()
    n = obs.shape[0]
    ep_len = 1000
    n_ep = n // ep_len
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    O = torch.as_tensor((obs - mu) / sd)
    A = torch.as_tensor(act, dtype=torch.float64)
    for epoch in range(EPOCHS):
        # K-step rollout loss on random windows
        idx_ep = np.random.randint(0, n_ep, 256)
        idx_t = np.random.randint(0, ep_len - K_ROLL - 1, 256)
        base = idx_ep * ep_len + idx_t
        loss = 0.0
        o0 = O[base]
        if arm == "aug":
            swap = np.random.rand(256) < 0.5
            o0 = o0.clone()
            o0[swap] = torch.as_tensor(s100.rho_obs(o0[swap].numpy()))
        z = model.encode(o0)
        for k in range(K_ROLL):
            a = A[base + k]
            tgt = O[base + k + 1]
            if arm == "aug":
                a = a.clone(); tgt = tgt.clone()
                a[swap] = torch.as_tensor(s100.rho_act(a[swap].numpy()))
                tgt[swap] = torch.as_tensor(s100.rho_obs(tgt[swap].numpy()))
            z = model.dynamics(z, a)
            loss = loss + ((model.decode(z) - tgt) ** 2).mean()
        recon = ((model.decode(model.encode(o0)) - o0) ** 2).mean()
        loss = loss / K_ROLL + recon
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 9:
            print(f"[step100b] {arm} seed {seed} epoch {epoch+1}/{EPOCHS} loss {loss.item():.5f}",
                  file=sys.stderr)
    # distilled policy head on latents (eq arm: tied leg heads)
    if arm == "eq":
        head_leg = mlp(LAT_LEG + LAT_INV, 3, HID * 3 // 4).double()
        head_params = list(head_leg.parameters())

        def pi_hat(z):
            zi, zr, zl = z[:, :LAT_INV], z[:, LAT_INV:LAT_INV + LAT_LEG], z[:, LAT_INV + LAT_LEG:]
            return torch.cat([head_leg(torch.cat([zr, zi], -1)), head_leg(torch.cat([zl, zi], -1))], -1)
    else:
        head = mlp(LAT_INV + 2 * LAT_LEG, 6, HID).double()
        head_params = list(head.parameters())

        def pi_hat(z):
            return head(z)
    opt2 = torch.optim.Adam(head_params, lr=1e-3)
    with torch.no_grad():
        Z = model.encode(O)
    for epoch in range(300 if not SMOKE else 10):
        b = np.random.randint(0, n, 512)
        l = ((pi_hat(Z[b]) - A[b]) ** 2).mean()
        opt2.zero_grad(); l.backward(); opt2.step()
    return model, pi_hat


def audit_arm(model, pi_hat, seed, mu, sd):
    """Benettin-QR on g(z)=dyn(z, pi_hat(z)) + measured crossings from true rollouts (step89 protocol)."""
    D = LAT_INV + 2 * LAT_LEG

    def g(z):
        return model.dynamics(z, torch.tanh(pi_hat(z)))

    env, pol = s100.make_env(), s100.load_policy()
    ts = env.reset()
    obs0 = []
    for t in range(420):
        a = pol(s89._flat_obs(ts).numpy())
        ts = env.step(a)
        obs0.append(s89._flat_obs(ts).numpy())
    z0 = model.encode(torch.as_tensor(((np.array(obs0)[210] - mu) / sd)[None]))
    # Benettin on the leading k directions
    k, steps, warm = 12, 240, 40
    torch.manual_seed(1000 + seed)
    Q = torch.linalg.qr(torch.randn(D, k, dtype=torch.float64))[0]
    z = z0.clone()
    sums = torch.zeros(k, dtype=torch.float64)
    sums_hist = []
    for t in range(steps + warm):
        J = torch.autograd.functional.jacobian(lambda zz: g(zz[None])[0], z[0], vectorize=True)
        Q = J @ Q
        Q, R = torch.linalg.qr(Q)
        diag = R.diagonal().abs().clamp_min(1e-300)
        if t >= warm:
            sums += diag.log()
            sums_hist.append(diag.log().numpy())
        z = g(z)
        if not torch.isfinite(z).all():
            return {"error": "diverged latent in Benettin"}
    lams = (sums / steps).numpy()
    H = np.array(sums_hist)
    B = 20
    blocks = np.array_split(H[:, 0], B)
    bs = [np.mean(np.concatenate([blocks[i] for i in np.random.default_rng(7).choice(B, B)]))
          for _ in range(200)]
    lo, hi = np.percentile([np.mean(np.random.default_rng(i).choice(
        [b.mean() for b in blocks], B)) for i in range(500)], [2.5, 97.5])
    lam1 = float(lams[0])
    # measured crossings: 100 mid-episode starts, free-run vs encoded truth
    starts = np.linspace(0, len(obs0) - 130, 100).astype(int)
    obs_n = (np.array(obs0) - mu) / sd
    Z_true = model.encode(torch.as_tensor(obs_n))
    scale = float(Z_true.std())
    meas = {str(e): [] for e in EPS_LIST}
    with torch.no_grad():
        for s0 in starts:
            z = Z_true[s0:s0 + 1]
            crossed = {str(e): None for e in EPS_LIST}
            for h in range(1, 120):
                z = g(z)
                if s0 + h >= len(obs0):
                    break
                err = float((z - Z_true[s0 + h]).norm()) / (float(Z_true[s0 + h].norm()) + 1e-12)
                for e in EPS_LIST:
                    if crossed[str(e)] is None and err > e:
                        crossed[str(e)] = h
            for e in EPS_LIST:
                meas[str(e)].append(crossed[str(e)] if crossed[str(e)] else 120)
    rows = []
    for e in EPS_LIST:
        med = float(np.median(meas[str(e)]))
        t1 = float(np.log(1 / e) / lam1) if lam1 > 0 and lo > 0 else None
        rows.append({"eps": e, "T1": t1, "measured_median": med,
                     "ratio": (med / t1) if t1 else None,
                     "censored": int(sum(1 for m in meas[str(e)] if m >= 120))})
    return {"lambda1": lam1, "lambda1_ci": [float(lo), float(hi)], "band": [float(x) for x in lams[:6]],
            "rows": rows, "latent_scale": scale}


def main():
    d = np.load(DATA)
    obs, act, nxt = d["obs"].astype(np.float64), d["act"].astype(np.float64), d["nxt"].astype(np.float64)
    # rho-invariant normalization: symmetrize stats so normalization commutes with the group action
    mu_raw, sd_raw = obs.mean(0), obs.std(0) + 1e-6
    mu = (mu_raw + s100.rho_obs(mu_raw)) / 2
    sd = (sd_raw + s100.rho_obs(sd_raw)) / 2
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            if key in results and "error" not in results[key]:
                continue
            model, pi_hat = train_arm(arm, seed, obs, act, nxt, mu, sd)
            ed, dd = equivariance_defect(model, torch.as_tensor(((obs[:256] - mu) / sd)),
                                         torch.as_tensor(act[:256]))
            res = audit_arm(model, pi_hat, seed, mu, sd)
            res["enc_defect"], res["dyn_defect"] = ed, dd
            results[key] = res
            r02 = [r for r in res.get("rows", []) if r["eps"] == 0.2]
            print(f"[step100b] {key}: enc_def={ed:.2e} dyn_def={dd:.2e} lam1={res.get('lambda1'):.4f} "
                  f"ratio@0.2={'%.2f' % r02[0]['ratio'] if r02 and r02[0]['ratio'] else '—'}",
                  file=sys.stderr)
            OUT.write_text(json.dumps(results, indent=1))
    print("[step100b] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
