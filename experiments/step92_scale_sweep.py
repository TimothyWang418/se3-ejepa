r"""Step 92 — the certified horizon vs MODEL SCALE, on the official TD-MPC2 MULTITASK checkpoints (walker-walk).

The title thesis as a measured curve: same task (walker-walk, index from each ckpt's own metadata), same read-out,
five official model scales (mt30: 1M/5M/19M/48M/317M; bonus mt80 1M/5M = data-diversity axis). Slice loader is
SHAPE-DRIVEN (all dims derived from the state dict itself), which automatically absorbs the two verified upstream
quirks (mt30-19M latent=512; task_dim 96-vs-64 rule). Task conditioning verified against upstream code: encoder
input [obs_pad, e], dynamics [z, e, a], pi [z, e]; e = `_task_emb.weight[idx]` (max_norm already baked in).
Certificate: leading-k JVP Benettin (step91's, identical across scales — full-basis QR is O(L^3) and L spans
128..1376); measured side: step89.measure() UNCHANGED via the slice interface. Reuses step89 layer replicas verbatim.

Run (one cell):  STEP92_CELLS=mt30-1M .venv/bin/python experiments/step92_scale_sweep.py
Full sweep:      .venv/bin/python experiments/step92_scale_sweep.py
Writes: papers/figures/step92_scale_sweep.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89  # noqa: E402  (SimNorm/mlp replicas, measure, TASKS)
from step91_lewm_audit import benettin_jvp  # noqa: E402  (leading-k JVP Benettin + bootstrap CI)

ROOT = Path(__file__).resolve().parent.parent
DTYPE = torch.float64
CKPT_DIR = ROOT / "models" / "tdmpc2_mt"
PARAMS = {"1M": 1e6, "5M": 5e6, "19M": 1.9e7, "48M": 4.8e7, "317M": 3.17e8}


class MTSlices:
    r"""Task-conditioned slices exposing the SAME interface as step89.TDMPC2Slices (encode/next/pi_mean), with the
    task embedding $e$ frozen as a constant — so step89's measure()/rollout_true() work unchanged."""

    def __init__(self, enc, dyn, pi, e, obs_pad, act_pad, latent):
        self.enc, self.dyn, self.pi, self.e = enc, dyn, pi, e
        self.obs_pad, self.A, self.L = obs_pad, act_pad, latent

    def encode(self, obs):
        o = obs.to(DTYPE)
        if o.shape[-1] < self.obs_pad:                              # mt80: zero-pad walker's 24-d obs to 39
            o = torch.cat([o, o.new_zeros(*o.shape[:-1], self.obs_pad - o.shape[-1])], -1)
        return self.enc(torch.cat([o, self.e.expand(*o.shape[:-1], -1)], -1))

    def next(self, z, a):
        return self.dyn(torch.cat([z, self.e.expand(*z.shape[:-1], -1), a], -1))

    def pi_mean(self, z):
        return self.pi(torch.cat([z, self.e.expand(*z.shape[:-1], -1)], -1)).chunk(2, dim=-1)[0]


def load_mt_slices(path, task="walker-walk") -> MTSlices:
    r"""Shape-driven loader: every dimension is read off the checkpoint's own tensors; strict=True per slice."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:                                               # metadata embeds omegaconf containers; the source is
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)   # the OFFICIAL nicklashansen HF repo (trusted)
    sd, md = ckpt["model"], ckpt.get("metadata", {})
    tasks = list(md.get("tasks", []))
    idx = tasks.index(task) if task in tasks else 1                 # walker-walk verified index 1 in mt30/mt80
    emb = sd["_task_emb.weight"]
    task_dim = emb.shape[1]
    n_enc = len({k.split(".")[2] for k in sd if k.startswith("_encoder.state.")})
    enc_in = sd["_encoder.state.0.weight"].shape[1]
    enc_dim = sd["_encoder.state.0.weight"].shape[0]
    latent = [v for k, v in sd.items() if k.startswith("_encoder.state.") and k.endswith(".weight")][-1].shape[0]
    dyn_in = sd["_dynamics.0.weight"].shape[1]
    mlp_dim = sd["_dynamics.0.weight"].shape[0]
    act_pad = dyn_in - latent - task_dim
    obs_pad = enc_in - task_dim
    enc = s89.mlp(enc_in, [enc_dim] * (n_enc - 1), latent, act=s89.SimNorm(8))
    dyn = s89.mlp(dyn_in, [mlp_dim, mlp_dim], latent, act=s89.SimNorm(8))
    pi = s89.mlp(latent + task_dim, [mlp_dim, mlp_dim], 2 * act_pad)
    def sub(p):
        return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    enc.load_state_dict(sub("_encoder.state."), strict=True)
    dyn.load_state_dict(sub("_dynamics."), strict=True)
    pi.load_state_dict(sub("_pi."), strict=True)
    for m in (enc, dyn, pi):
        m.double().eval()
        for p in m.parameters():
            p.requires_grad_(False)
    e = emb[idx].double()
    print(f"[step92] {Path(path).name}: L={latent} enc_in={enc_in} ({n_enc} enc layers, {enc_dim}w) "
          f"mlp={mlp_dim} task_dim={task_dim} obs_pad={obs_pad} act_pad={act_pad} task_idx={idx}", file=sys.stderr)
    return MTSlices(enc, dyn, pi, e, obs_pad, act_pad, latent)


def run_cell(name: str, eps_list, k=16, qr_steps=300, qr_warm=60, n_starts=12) -> dict:
    sl = load_mt_slices(CKPT_DIR / f"{name}.pt")
    # on-attractor latent: encode a mid-episode true obs under the policy prior
    zs = s89.rollout_true("walker-walk", sl, T=60, seed=11)
    z0 = zs[40]
    if sl.L >= 1000:                                               # 317M: trim QR budget (O(width^2) per JVP)
        qr_steps, qr_warm, k = 160, 30, 12
    g = s89.make_autonomous(sl)
    print(f"[step92] {name}: leading-{k} Benettin ({qr_steps}+{qr_warm}) on L={sl.L} ...", file=sys.stderr)
    lam, (lo, hi) = benettin_jvp(g, z0, k, qr_steps, qr_warm)
    l1 = float(lam[0])
    meas = s89.measure("walker-walk", sl, eps_list, n_starts=n_starts, max_h=300, T=380, seed=7)
    rows = []
    for e in eps_list:
        T1 = float(np.log(1.0 / e) / l1) if (l1 > 0 and lo > 0) else None     # abstain if CI straddles 0
        med = meas[str(e)]["median"]
        rows.append({"eps": e, "T1_steps": T1, "measured_median": med,
                     "ratio": (med / T1 if T1 else None), "n_censored": meas[str(e)]["n_censored"]})
        print(f"[step92] {name} eps={e}: certified={'ABSTAIN' if T1 is None else f'{T1:.0f}'} "
              f"measured={med:.0f} ratio={'—' if T1 is None else f'{med/T1:.2f}'}", file=sys.stderr)
    return {"cell": name, "params": PARAMS[name.split('-')[1]], "latent_dim": sl.L,
            "lambda1": l1, "lambda1_ci": [lo, hi], "leading_band": [float(x) for x in lam],
            "rows": rows}


if __name__ == "__main__":
    torch.manual_seed(0)
    eps_list = [0.05, 0.1, 0.2]
    cells = os.environ.get("STEP92_CELLS", "mt30-1M,mt30-5M,mt30-19M,mt30-48M,mt30-317M,mt80-1M,mt80-5M").split(",")
    out = {"task": "walker-walk", "eps_list": eps_list, "cells": []}
    for c in cells:
        if not (CKPT_DIR / f"{c}.pt").exists():
            print(f"[step92] missing {c}.pt — skip", file=sys.stderr)
            continue
        out["cells"].append(run_cell(c, eps_list))
        (ROOT / "papers/figures/step92_scale_sweep.json").write_text(json.dumps(out, indent=2))  # checkpoint-as-you-go
    print("[step92] wrote step92_scale_sweep.json", file=sys.stderr)
