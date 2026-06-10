r"""wm-audit — certify a pretrained world model's trustworthy horizon in one command (training-free).

Reads the leading Lyapunov band of a model's latent loop via forward-mode Benettin-QR, issues a certified horizon
$T_1(\epsilon)=\log(1/\epsilon)/\lambda_1$ per resolution — or ABSTAINS when the $\lambda_1$ CI straddles zero
(contracting/neutral loop: divergence there is bias, not amplification — outside Lyapunov jurisdiction).

Supported families (official checkpoints, loaded into faithful slices / the authors' own code):
  tdmpc2      single-task DMC checkpoints   (huggingface.co/nicklashansen/tdmpc2 .../dmcontrol/{task}-{seed}.pt)
  tdmpc2-mt   multitask mt30/mt80 ladder    (.../multitask/{suite}-{size}.pt)
  lewm        LeWM PushT et al.             (huggingface.co/quentinll/lewm-*)

Examples:
  python scripts/wm_audit.py tdmpc2    models/tdmpc2/walker-walk-1.pt --task walker-walk
  python scripts/wm_audit.py tdmpc2-mt models/tdmpc2_mt/mt30-317M.pt  --task walker-walk
  python scripts/wm_audit.py lewm      models/lewm/pusht-weights.pt   --config models/lewm/pusht-config.json

Python API for ANY deterministic differentiable latent map:
  from scripts.wm_audit import audit_map
  report = audit_map(g, z0, eps_list=[0.05, 0.1, 0.2])     # g: R^d -> R^d (torch, float64)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))


def audit_map(g, z0: torch.Tensor, eps_list=(0.05, 0.1, 0.2), k: int = 12, n_steps: int = 300,
              warmup: int = 60) -> dict:
    r"""Certify any deterministic $C^1$ latent map ``g`` from state ``z0``: leading-$k$ Lyapunov band (JVP Benettin)
    + block-bootstrap CI on $\lambda_1$ + per-$\epsilon$ certified horizons, with the abstain rule built in."""
    from step91_lewm_audit import benettin_jvp
    lam, (lo, hi) = benettin_jvp(g, z0.double(), k, n_steps, warmup)
    l1 = float(lam[0])
    rows = []
    for e in eps_list:
        T1 = float(np.log(1.0 / e) / l1) if (l1 > 0 and lo > 0) else None
        rows.append({"eps": e, "T1_steps": T1})
    verdict = ("EXPANSIVE — certified horizons issued" if (l1 > 0 and lo > 0) else
               "ABSTAIN — lambda1 CI straddles/below 0 (contracting or neutral loop); residual divergence is "
               "bias-driven and outside a Lyapunov certificate's jurisdiction")
    return {"lambda1": l1, "lambda1_ci": [lo, hi], "leading_band": [float(x) for x in lam],
            "horizons": rows, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description="Certify a pretrained world model's trustworthy horizon (training-free).")
    ap.add_argument("family", choices=["tdmpc2", "tdmpc2-mt", "lewm"])
    ap.add_argument("ckpt")
    ap.add_argument("--task", default="walker-walk")
    ap.add_argument("--config", default=None, help="lewm: path to the HF config.json")
    ap.add_argument("--eps", default="0.05,0.1,0.2")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args()
    eps = [float(x) for x in a.eps.split(",")]

    if a.family == "tdmpc2":
        import step89_pretrained_wm_audit as s89
        obs_dim, act_dim = {t: (v[2], v[3]) for t, v in s89.TASKS.items()}[a.task]
        sl = s89.load_tdmpc2_slices(a.ckpt, obs_dim=obs_dim, action_dim=act_dim)
        zs = s89.rollout_true(a.task, sl, T=60, seed=11)
        g, z0 = s89.make_autonomous(sl), zs[40]
    elif a.family == "tdmpc2-mt":
        import step89_pretrained_wm_audit as s89
        import step92_scale_sweep as s92
        sl = s92.load_mt_slices(a.ckpt, task=a.task)
        zs = s89.rollout_true(a.task, sl, T=60, seed=11)
        g, z0 = s89.make_autonomous(sl), zs[40]
    else:
        import step91_lewm_audit as s91
        if a.config:
            cfgp = Path(a.config)
            assert cfgp.exists()
        model = s91.load_lewm()
        fr = s91.env_rollout(seed=0, n_model_steps=6)
        z0 = s91.encode_seq(model, fr, True)[:s91.HS].reshape(-1)
        g = s91.make_autonomous(model)

    rep = audit_map(g, z0, eps, k=a.k, n_steps=a.steps)
    print(json.dumps(rep, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
