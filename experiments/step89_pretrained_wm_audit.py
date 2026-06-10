r"""Step 89 — certify a PUBLIC PRETRAINED world model: TD-MPC2 (Hansen et al., MIT), training-free.

Design: docs/specs/2026-06-09-step89-pretrained-wm-audit-design.md. The certificate is a READ-OUT: rebuild the
checkpoint's `_encoder/_dynamics/_pi` slices as faithful module replicas (layer math copied line-for-line from
tdmpc2/common/layers.py — SimNorm = per-group softmax over groups of 8; NormedLinear = Mish(LayerNorm(Linear));
mlp = NormedLinear hidden + NormedLinear(act)/Linear tail), wrap the autonomous policy-prior loop
$g(z)=d(z,\tanh(\mu_\pi(z)))$, and run the UNCHANGED step78 certified-horizon machinery on its autograd Jacobians.
The measured side rolls the TRUE dm_control env under the same prior and open-loops the latent for the divergence
horizon. Checkpoints (verified URLs): huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/{task}-{seed}.pt,
downloaded to models/tdmpc2/ (gitignored). Pre-registered risks (spec §2): (a) ~64 structural SimNorm zero-directions
in the 512-d spectrum (report, don't hide); (b) certificate covers the policy-prior loop, not MPPI; (c) λ1<=0 on a
stable gait is a FINDING (certified stability, cross-validated), not a failure.

Run (one task quick):  STEP89_TASKS=acrobot-swingup STEP89_SEEDS=1 .venv/bin/python experiments/step89_pretrained_wm_audit.py
Run (full):            .venv/bin/python experiments/step89_pretrained_wm_audit.py
Writes: papers/figures/step89_pretrained_audit.json (+ .png)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step78_certified_horizon_ci as step78  # noqa: E402

DTYPE = torch.float64
ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "models" / "tdmpc2"

# Task registry: dm_control (domain, task), obs_dim (flattened, their wrapper order), action_dim.
TASKS = {"acrobot-swingup": ("acrobot", "swingup", 6, 1),
         "walker-walk": ("walker", "walk", 24, 6)}


# ----------------------------- faithful layer replicas (tdmpc2/common/layers.py) ------------------------------- #
class SimNorm(nn.Module):
    r"""Simplicial normalization (Lavoie et al. 2022), verbatim from TD-MPC2: view (..., L) as (..., L/dim, dim),
    softmax over the last axis. The latent lives on $\prod_{L/\dim}\Delta^{\dim-1}$ — compact, $C^\infty$."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    r"""Linear + LayerNorm + activation (Mish default) — verbatim semantics: ``act(ln(linear(x)))``. state_dict keys:
    ``weight, bias, ln.weight, ln.bias`` (matches the checkpoint layout)."""

    def __init__(self, *args, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = nn.Mish(inplace=False) if act is None else act

    def forward(self, x):
        return self.act(self.ln(super().forward(x)))


def mlp(in_dim: int, mlp_dims, out_dim: int, act=None) -> nn.Sequential:
    r"""TD-MPC2 ``layers.mlp`` replica: NormedLinear(Mish) hidden layers; tail = NormedLinear(act) if ``act`` else a
    plain Linear. Key layout: ``{i}.weight/{i}.bias(+ {i}.ln.*)``."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    mods = []
    for i in range(len(dims) - 2):
        mods.append(NormedLinear(dims[i], dims[i + 1]))
    mods.append(NormedLinear(dims[-2], dims[-1], act=act) if act is not None else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mods)


# ----------------------------------------- checkpoint slice loader --------------------------------------------- #
class TDMPC2Slices:
    r"""The three slices the certificate needs: ``encode(obs)->z`` (SimNorm simplex), ``next(z,a)->z'`` (deterministic
    latent dynamics), ``pi_mean(z)->mu`` (policy-prior mean, pre-tanh). All float64, frozen, eval-mode."""

    def __init__(self, encoder, dynamics, pi, latent_dim, action_dim):
        self.encoder, self.dynamics, self.pi, self.L, self.A = encoder, dynamics, pi, latent_dim, action_dim

    def encode(self, obs):
        return self.encoder(obs.to(DTYPE))

    def next(self, z, a):
        return self.dynamics(torch.cat([z, a], dim=-1))

    def pi_mean(self, z):
        return self.pi(z).chunk(2, dim=-1)[0]


def load_tdmpc2_slices(ckpt, latent_dim=512, action_dim=1, obs_dim=6, enc_dim=256, mlp_dim=512,
                       simnorm_dim=8) -> TDMPC2Slices:
    r"""Build faithful slice modules and load the matching key subsets from a TD-MPC2 checkpoint (path or already-
    loaded dict ``{'model': state_dict}``). Q-ensembles/targets are ignored. strict=True per slice: a key/shape
    mismatch raises with the available prefixes printed (defensive against upstream drift)."""
    if not isinstance(ckpt, dict):
        ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    enc = mlp(obs_dim, [enc_dim], latent_dim, act=SimNorm(simnorm_dim))
    dyn = mlp(latent_dim + action_dim, [mlp_dim, mlp_dim], latent_dim, act=SimNorm(simnorm_dim))
    pi = mlp(latent_dim, [mlp_dim, mlp_dim], 2 * action_dim, act=None)
    def sub(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    try:
        enc.load_state_dict(sub("_encoder.state."), strict=True)
        dyn.load_state_dict(sub("_dynamics."), strict=True)
        pi.load_state_dict(sub("_pi."), strict=True)
    except Exception:
        prefixes = sorted({k.split(".")[0] for k in sd.keys()})
        print(f"[step89] LOAD MISMATCH — available top-level prefixes: {prefixes}", file=sys.stderr)
        raise
    for m in (enc, dyn, pi):
        m.double().eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return TDMPC2Slices(enc, dyn, pi, latent_dim, action_dim)


def make_autonomous(slices: TDMPC2Slices):
    r"""The autonomous policy-prior closed loop $g(z)=d\big(z,\tanh(\mu_\pi(z))\big)$ — smooth, deterministic, the map
    the certificate reads (spec risk (b): this is the prior loop, NOT the MPPI planner; stated, not hidden)."""
    def g(z):
        return slices.next(z, torch.tanh(slices.pi_mean(z)))
    return g


# -------------------------------------------- certified side --------------------------------------------------- #
def certify(slices, z0, eps_list, n_steps=400, warmup=80, n_boot=200, block=40, seed=0) -> dict:
    r"""UNCHANGED step78 machinery on $g$'s autograd Jacobians, dt = 1 MODEL STEP: Benettin-QR log|diag R| series →
    block-bootstrap spectrum CI → per-eps certified horizon $T_1(\epsilon)=\log(1/\epsilon)/\lambda_1$ in model steps
    (abstain if the $\lambda_1$ CI straddles 0 — then the certificate claims a long/unbounded horizon, risk (c))."""
    g = make_autonomous(slices)
    logR = step78.qr_logR_series(g, z0.to(DTYPE), n_steps, warmup)
    point, lo, hi = step78.bootstrap_spectrum_ci(logR, 1.0, n_boot, block, seed)
    lam1, lam1_lo, lam1_hi = float(point[0]), float(lo[0]), float(hi[0])
    rows = []
    for eps in eps_list:
        if lam1 > 0:
            T1 = float(np.log(1.0 / eps) / lam1)
            iv = step78.horizon_interval(lam1_lo, lam1_hi, eps) if lam1_lo > 0 else None
        else:
            T1, iv = None, None                                    # contracting/neutral loop -> long-horizon claim
        rows.append({"eps": eps, "T1_steps": T1,
                     "T1_lo": (float(iv[0]) if iv else None), "T1_hi": (float(iv[1]) if iv else None)})
    spec = point.tolist()
    n_structural = int(sum(1 for v in spec if v < -10.0))          # SimNorm zero-direction band (expected ~64)
    return {"lambda1": lam1, "lambda1_ci": [lam1_lo, lam1_hi], "rows": rows,
            "spectrum_head": spec[:8], "spectrum_tail": spec[-4:], "n_structural_band": n_structural,
            "n_steps": n_steps}


# -------------------------------------------- measured side ---------------------------------------------------- #
def _flat_obs(time_step) -> torch.Tensor:
    return torch.cat([torch.as_tensor(np.asarray(v).ravel(), dtype=torch.float64)
                      for v in time_step.observation.values()])


def rollout_true(task_key: str, slices: TDMPC2Slices, T: int, seed: int):
    r"""One TRUE dm_control episode under the policy prior (their wrapper semantics: flattened obs dict, action_repeat
    = 2). Returns the sequence of TRUE-encoded latents $z_t=\mathrm{enc}(o_t)$, length $T{+}1$, model-step indexed."""
    from dm_control import suite
    domain, task, _, _ = TASKS[task_key]
    env = suite.load(domain, task, task_kwargs={"random": seed})
    ts = env.reset()
    zs = []
    with torch.no_grad():
        for _ in range(T + 1):
            z = slices.encode(_flat_obs(ts))
            zs.append(z)
            a = torch.tanh(slices.pi_mean(z)).numpy()
            for _ in range(2):                                     # action_repeat=2 (their dmcontrol wrapper)
                ts = env.step(a)
    return torch.stack(zs, 0)                                      # (T+1, L)


def measure(task_key: str, slices: TDMPC2Slices, eps_list, n_starts=20, max_h=300, T=420, seed=0) -> dict:
    r"""Measured divergence horizon: from ``n_starts`` points along a TRUE episode, open-loop the latent under $g$ and
    record the first model-step where $\lVert\hat z_h - z^{\rm true}_h\rVert/\lVert z^{\rm true}_h\rVert>\epsilon$.
    Median per eps (censored at ``max_h`` — censoring count reported, the long-horizon reading of risk (c))."""
    zs = rollout_true(task_key, slices, T, seed)
    g = make_autonomous(slices)
    starts = np.linspace(0, zs.shape[0] - max_h - 1, n_starts).astype(int)
    horizons = {eps: [] for eps in eps_list}
    with torch.no_grad():
        for s0 in starts:
            zhat = zs[s0].clone()
            crossed = {eps: None for eps in eps_list}
            for h in range(1, max_h + 1):
                zhat = g(zhat)
                rel = float((zhat - zs[s0 + h]).norm() / zs[s0 + h].norm().clamp_min(1e-12))
                for eps in eps_list:
                    if crossed[eps] is None and rel > eps:
                        crossed[eps] = h
                if all(v is not None for v in crossed.values()):
                    break
            for eps in eps_list:
                horizons[eps].append(crossed[eps] if crossed[eps] is not None else max_h)
    out = {}
    for eps in eps_list:
        hs = np.asarray(horizons[eps], float)
        out[str(eps)] = {"median": float(np.median(hs)), "p25": float(np.percentile(hs, 25)),
                         "p75": float(np.percentile(hs, 75)), "n_censored": int((hs >= max_h).sum())}
    return out


# ------------------------------------------------- driver ------------------------------------------------------ #
def run(tasks, seeds, eps_list) -> dict:
    results = {}
    for task_key in tasks:
        domain, task, obs_dim, action_dim = TASKS[task_key]
        for seed in seeds:
            ck = CKPT_DIR / f"{task_key}-{seed}.pt"
            if not ck.exists():
                print(f"[step89] missing checkpoint {ck} — skip", file=sys.stderr)
                continue
            print(f"[step89] {task_key} seed {seed}: loading slices ...", file=sys.stderr)
            slices = load_tdmpc2_slices(ck, action_dim=action_dim, obs_dim=obs_dim)
            # operating point: a true-encoded latent mid-episode (on the model's own data manifold)
            zs = rollout_true(task_key, slices, 60, seed)
            z0 = zs[len(zs) // 2]
            print(f"[step89] {task_key} seed {seed}: certifying (512-d QR) ...", file=sys.stderr)
            cert = certify(slices, z0, eps_list, seed=seed)
            print(f"[step89] {task_key} seed {seed}: lambda1={cert['lambda1']:.4f} "
                  f"CI={['%.4f' % v for v in cert['lambda1_ci']]} structural_band={cert['n_structural_band']}",
                  file=sys.stderr)
            print(f"[step89] {task_key} seed {seed}: measuring divergence horizon ...", file=sys.stderr)
            meas = measure(task_key, slices, eps_list, seed=seed)
            row = {"lambda1": cert["lambda1"], "lambda1_ci": cert["lambda1_ci"],
                   "n_structural_band": cert["n_structural_band"], "cert_rows": cert["rows"], "measured": meas}
            for r in cert["rows"]:
                eps = r["eps"]
                m = meas[str(eps)]
                ratio = (m["median"] / r["T1_steps"]) if r["T1_steps"] else None
                r["measured_median"] = m["median"]
                r["ratio_measured_over_certified"] = ratio
                print(f"[step89]   eps={eps}: certified={('%.0f' % r['T1_steps']) if r['T1_steps'] else 'ABSTAIN(lam1<=0)'} "
                      f"measured_med={m['median']:.0f} (censored {m['n_censored']}/20) "
                      f"ratio={('%.2f' % ratio) if ratio else '—'}", file=sys.stderr)
            results[f"{task_key}-{seed}"] = row
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    tasks = os.environ.get("STEP89_TASKS", "acrobot-swingup,walker-walk").split(",")
    seeds = [int(x) for x in os.environ.get("STEP89_SEEDS", "1,2,3").split(",")]
    eps_list = [float(x) for x in os.environ.get("STEP89_EPS", "0.05,0.1,0.2").split(",")]
    print(f"[step89] pretrained-WM audit: tasks={tasks} seeds={seeds} eps={eps_list}", file=sys.stderr)
    res = run(tasks, seeds, eps_list)
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "step89_pretrained_audit.json").write_text(json.dumps(res, indent=2))
    print(f"[step89] wrote papers/figures/step89_pretrained_audit.json ({len(res)} cells)", file=sys.stderr)
    raise SystemExit(0)
