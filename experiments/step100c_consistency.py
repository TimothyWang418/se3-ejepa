r"""Step 100c — the bridge, retried with a TD-MPC2-style latent objective (the real fix).

Diagnosis of step100b's INCONCLUSIVE-BY-LEVEL-DOMINANCE (`paper2_record` 2026-06-13): the reconstruction
objective forces the latent to keep the fast chaotic modes of the walker (true closed-loop lambda1=+0.82),
so a small scratch model either under-fits (lambda~0) or has one-step latent error >20% — measured median
pins at 1 and the growth-vs-level comparison never opens. The official TD-MPC2 walker LATENT instead has
lambda1=0.25-0.30 (horizon 5-6) precisely because its **latent-consistency + reward** objective compresses
the dynamics onto a control-relevant slow manifold. This file swaps the objective accordingly:

  loss = consistency( dynamics(z_k, a_k),  stopgrad enc_target(o_{k+1}) )    # the latent-shaping force
       + w_r * reward_head(z_k) -> r_k                                        # anti-collapse anchor (TD-MPC2)
       + w_d * decode(z_0) -> o_0                                            # small reconstruction safety

with an EMA target encoder. Architectures (eq / dense / aug) and the audit are reused verbatim from
step100b, so the ONLY change vs the sealed negative result is the training objective — a controlled test
of the diagnosis. Hypothesis (pre-registered here, before reading any arm result): a consistency latent
reaches lambda1 in [0.1, 0.4] with measured horizon > 1, opening the structure-vs-dense calibration test;
if it does not, that is the recorded ceiling and the bridge needs full online TD-MPC2 (GPU, deferred).

Run: STEP100_SEEDS=0 STEP100_ARMS=eq,dense .venv/bin/python experiments/step100c_consistency.py
Writes papers/figures/step100c_consistency_results.json  (incremental per (arm,seed); resumable)
"""
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step100b_train_audit as s100b  # noqa: E402  (EqWM, DenseWM, rho_z, audit_arm, mlp, constants)
import step100_walker_s2 as s100  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers/figures/step100c_consistency_results.json"
DATA = ROOT / "data/step100_walker_transitions.npz"

SEEDS = [int(x) for x in os.environ.get("STEP100_SEEDS", "0").split(",")]
ARMS = os.environ.get("STEP100_ARMS", "eq,dense,aug").split(",")
SMOKE = bool(int(os.environ.get("STEP100_SMOKE", "0")))
EPOCHS = 200 if SMOKE else int(os.environ.get("STEP100_EPOCHS", "8000"))
K_ROLL = s100b.K_ROLL
LL, LI = s100b.LAT_LEG, s100b.LAT_INV
W_REW = float(os.environ.get("STEP100_WREW", "1.0"))
W_DEC = float(os.environ.get("STEP100_WDEC", "0.1"))
EMA = float(os.environ.get("STEP100_EMA", "0.99"))


def reward_head(arm):
    """TD-MPC2-style reward predictor on the latent (eq arm: S2-invariant via symmetric leg pooling)."""
    if arm == "eq":
        net = s100b.mlp(LI + LL, 1, 64).double()          # input: (z_inv, z_R + z_L) -> invariant

        def f(z):
            zi, zr, zl = z[:, :LI], z[:, LI:LI + LL], z[:, LI + LL:]
            return net(torch.cat([zi, zr + zl], -1))
        return f, list(net.parameters())
    net = s100b.mlp(LI + 2 * LL, 1, 64).double()

    def f(z):
        return net(z)
    return f, list(net.parameters())


def train_consistency(arm, seed, obs, act, nxt, rew, mu, sd):
    torch.manual_seed(seed); np.random.seed(seed)
    model = (s100b.EqWM() if arm == "eq" else s100b.DenseWM()).double()
    target = deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    rew_fn, rew_params = reward_head(arm)
    opt = torch.optim.Adam(list(model.parameters()) + rew_params, lr=1e-3)
    O = torch.as_tensor((obs - mu) / sd)
    A = torch.as_tensor(act, dtype=torch.float64)
    R = torch.as_tensor(rew, dtype=torch.float64).view(-1, 1)
    n, ep_len = obs.shape[0], 1000
    n_ep = n // ep_len
    for epoch in range(EPOCHS):
        idx = np.random.randint(0, n_ep, 256) * ep_len + np.random.randint(0, ep_len - K_ROLL - 1, 256)
        idx = torch.as_tensor(idx)
        o0 = O[idx].clone()
        if arm == "aug":
            swap = torch.as_tensor(np.random.rand(256) < 0.5)
            o0[swap] = torch.as_tensor(s100.rho_obs(o0[swap].numpy()))
        z = model.encode(o0)
        cons = rew_l = 0.0
        for k in range(K_ROLL):
            a = A[idx + k].clone()
            o_next = O[idx + k + 1].clone()
            r_k = R[idx + k].clone()
            if arm == "aug":
                a[swap] = torch.as_tensor(s100.rho_act(a[swap].numpy()))
                o_next[swap] = torch.as_tensor(s100.rho_obs(o_next[swap].numpy()))
            rew_l = rew_l + ((rew_fn(z) - r_k) ** 2).mean()
            z = model.dynamics(z, a)
            with torch.no_grad():
                z_tgt = target.encode(o_next)              # stop-grad EMA target latent
            cons = cons + ((z - z_tgt) ** 2).mean()
        dec = ((model.decode(model.encode(o0)) - o0) ** 2).mean()
        loss = cons / K_ROLL + W_REW * rew_l / K_ROLL + W_DEC * dec
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                              # EMA target update
            for pt, pm in zip(target.parameters(), model.parameters()):
                pt.mul_(EMA).add_(pm, alpha=1 - EMA)
        if epoch % 20 == 19:
            print(f"[step100c] {arm} s{seed} ep{epoch+1}/{EPOCHS} cons {float(cons/K_ROLL):.5f} "
                  f"rew {float(rew_l/K_ROLL):.5f} dec {float(dec):.4f}", file=sys.stderr)
    # distilled policy head + audit reused verbatim from step100b
    if arm == "eq":
        head_leg = s100b.mlp(LL + LI, 3, 48).double()

        def pi_hat(z):
            zi, zr, zl = z[:, :LI], z[:, LI:LI + LL], z[:, LI + LL:]
            return torch.cat([head_leg(torch.cat([zr, zi], -1)), head_leg(torch.cat([zl, zi], -1))], -1)
        hp = list(head_leg.parameters())
    else:
        head = s100b.mlp(LI + 2 * LL, 6, 64).double()
        pi_hat = lambda z: head(z)  # noqa: E731
        hp = list(head.parameters())
    opt2 = torch.optim.Adam(hp, lr=1e-3)
    with torch.no_grad():
        Z = model.encode(O)
    for _ in range(300 if not SMOKE else 10):
        b = np.random.randint(0, n, 512)
        l = ((pi_hat(Z[b]) - A[b]) ** 2).mean()
        opt2.zero_grad(); l.backward(); opt2.step()
    return model, pi_hat


def main():
    d = np.load(DATA)
    if "rew" not in d:
        raise SystemExit("data has no reward — re-run step100 collector (MODE=collect) first")
    obs, act, nxt, rew = (d["obs"].astype(np.float64), d["act"].astype(np.float64),
                          d["nxt"].astype(np.float64), d["rew"].astype(np.float64))
    mu_raw, sd_raw = obs.mean(0), obs.std(0) + 1e-6
    mu = (mu_raw + s100.rho_obs(mu_raw)) / 2                # rho-invariant normalization (commutes with the group)
    sd = (sd_raw + s100.rho_obs(sd_raw)) / 2
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            if key in results and "error" not in results[key]:
                continue
            model, pi_hat = train_consistency(arm, seed, obs, act, nxt, rew, mu, sd)
            ed, dd = s100b.equivariance_defect(model, torch.as_tensor(((obs[:256] - mu) / sd)),
                                               torch.as_tensor(act[:256]))
            res = s100b.audit_arm(model, pi_hat, seed, mu, sd)
            res["enc_defect"], res["dyn_defect"] = ed, dd
            results[key] = res
            r02 = [r for r in res.get("rows", []) if r["eps"] == 0.2]
            print(f"[step100c] {key}: enc_def={ed:.1e} lam1={res.get('lambda1'):+.4f} "
                  f"CI[{res['lambda1_ci'][0]:+.3f},{res['lambda1_ci'][1]:+.3f}] "
                  f"med@0.2={r02[0]['measured_median'] if r02 else '?'} "
                  f"ratio={('%.2f'%r02[0]['ratio']) if r02 and r02[0]['ratio'] else '—'}", file=sys.stderr)
            OUT.write_text(json.dumps(results, indent=1))
    print("[step100c] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
