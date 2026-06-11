r"""wma-step2 — imprint across architectures: ATM + echo-excess audit of official TD-MPC2.

Spec (pre-registered, v1.2): docs/specs/2026-06-11-wma-step2-tdmpc2-imprint-seed.md
Instrument: src/audit/atm.py (atm_audit + echo_excess/imprint_ratio/derangement, certified
in tests/test_wma_step2.py three-arm discriminator).
Loader: step89's faithful TD-MPC2 slice replicas (strict load, official checkpoints).

Question: wma-step1 found pure echo on LeWM ($\eta = 0.474$, $\rho \approx 1$). TD-MPC2's
dynamics is also action-conditioned, but trained with latent-consistency regression onto
the EMA-encoded TRUE next latent + reward/value grounding. Is the echo generic or
objective-specific?

Protocol: 3 tasks x seed-1 (walker-walk / cheetah-run / finger-spin), TWO behavior lanes
per task (step1c distribution lesson institutionalized): lane-pi = tanh(mu_pi(z) + 0.3 xi),
lane-U = U[-1,1]^A. 1 model step = action_repeat 2 (step89 convention). 10 episodes x 500
model steps = 5000 transitions per cell; episode-level 80/20 split. Real audit = full
registered protocol (widths {64,256,1024} x probe seeds {0,1,2}); imprint audit = width 256
x 3 seeds (registered). Cross-episode derangement seed 2026.

Gates: G-s2a-eta (eta < 0.2 on >=2/3 tasks, lane-pi, VACUOUS-BY-CEILING marking when
D~_TT < 0.2) / G-s2a-sym (L_sym < 2.0 on >=2/3 tasks) / G-s2b (D~_TT(a') > 0.9 sanity) /
G-s2c (one-step rel-err median < 1; step89 spectral column cross-referenced).

Run (smoke): WMA2_SMOKE=1 .venv/bin/python experiments/wma_step2_tdmpc2_atm.py
Writes: papers/figures/wma_step2_tdmpc2_atm{_smoke}.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from step89_pretrained_wm_audit import TASKS, _flat_obs, load_tdmpc2_slices  # noqa: E402

from src.audit.atm import atm_audit, derangement, echo_excess, imprint_ratio  # noqa: E402

SMOKE = bool(int(os.environ.get("WMA2_SMOKE", "0")))
DTYPE = torch.float64
TASK_KEYS = ["walker-walk"] if SMOKE else ["walker-walk", "cheetah-run", "finger-spin"]
SEED_CKPT = 1
EP_SEED0 = 2000
SPLIT_SEED = 0
DERANGE_SEED = 2026
NOISE_STD = 0.3
ACTION_REPEAT = 2
LEWM_REF = {"eta": 0.474, "rho": 1.03, "L_sym": 6.90}          # wma-step1/1c reference row


def collect(task_key: str, lane: str, slices, n_eps: int, n_steps: int):
    """Roll dm_control episodes under the lane's behavior; return per-episode (Z, A_t)."""
    from dm_control import suite
    domain, task, _, adim = TASKS[task_key]
    episodes = []
    for ep in range(n_eps):
        seed = EP_SEED0 + ep
        env = suite.load(domain, task, task_kwargs={"random": seed})
        ts = env.reset()
        g = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs, acts = [_flat_obs(ts)], []
        with torch.no_grad():
            for _ in range(n_steps):
                if lane == "pi":
                    z = slices.encode(obs[-1])
                    a = torch.tanh(slices.pi_mean(z) +
                                   NOISE_STD * torch.randn(adim, generator=g, dtype=DTYPE))
                    a_np = a.numpy()
                else:                                            # lane-U
                    a_np = rng.uniform(-1.0, 1.0, size=adim)
                for _ in range(ACTION_REPEAT):
                    ts = env.step(a_np)
                obs.append(_flat_obs(ts))
                acts.append(torch.as_tensor(np.asarray(a_np, dtype=np.float64)))
        env.close()
        with torch.no_grad():
            Z = slices.encode(torch.stack(obs, 0))               # (n_steps+1, L)
        episodes.append((Z, torch.stack(acts, 0)))
    return episodes


def audit_cell(task_key: str, lane: str, slices, n_eps: int, n_steps: int, audit_kw: dict,
               audit_kw_perm: dict) -> dict:
    episodes = collect(task_key, lane, slices, n_eps, n_steps)
    Zt = torch.cat([Z[:-1] for Z, _ in episodes])
    Znext = torch.cat([Z[1:] for Z, _ in episodes])
    Atgt = torch.cat([A for _, A in episodes]).float()
    ep_ids = np.concatenate([np.full(A.shape[0], i) for i, (_, A) in enumerate(episodes)])
    N = Zt.shape[0]

    with torch.no_grad():
        Zpred = torch.cat([slices.next(Z[:-1], A) for Z, A in episodes])
    one_step = float(((Zpred - Znext).norm(dim=-1) /
                      Znext.norm(dim=-1).clamp_min(1e-12)).median())

    eps_u = np.unique(ep_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(eps_u)
    n_test = max(1, int(round(0.2 * eps_u.shape[0])))
    mask = np.isin(ep_ids, eps_u[:n_test].tolist())
    tr, te = torch.tensor(np.where(~mask)[0]), torch.tensor(np.where(mask)[0])

    res_real = atm_audit(Zt, Znext, Zpred, Atgt, train_idx=tr, test_idx=te, **audit_kw)

    perm, cross = derangement(N, seed=DERANGE_SEED, ep_ids=ep_ids)
    Aperm = Atgt[perm].double()
    with torch.no_grad():
        Zpred_p = slices.next(Zt, Aperm)
    res_perm = atm_audit(Zt, Znext, Zpred_p, Aperm.float(),
                         train_idx=tr, test_idx=te, **audit_kw_perm)

    eta = echo_excess(res_real.D_norm)
    rho = imprint_ratio(res_real.D_norm[("P", "P")], res_perm.D_norm[("P", "P")])
    dtt = res_real.D_norm[("T", "T")]
    cell = {
        "real": res_real.to_dict(), "perm": res_perm.to_dict(),
        "eta": eta, "rho_imp": rho, "L_sym": res_real.readouts["L_sym"],
        "Dnorm_TT_ceiling": dtt,
        "eta_vacuous_by_ceiling": bool(dtt < 0.2),
        "sanity_Dnorm_TT_perm": res_perm.D_norm[("T", "T")],
        "G_s2b_pass": bool(res_perm.D_norm[("T", "T")] > 0.9),
        "one_step_rel_err_median": one_step,
        "G_s2c_pass": bool(one_step < 1.0),
        "cross_episode_fraction": cross, "n_transitions": int(N),
    }
    print(f"[wma2] {task_key}/{lane}: Dn_TT={dtt:.3f} Dn_PP={res_real.D_norm[('P','P')]:.3f} "
          f"eta={eta:.3f}{' (VACUOUS)' if cell['eta_vacuous_by_ceiling'] else ''} "
          f"rho={'-' if rho is None else f'{rho:.2f}'} L_sym={cell['L_sym']:.2f} "
          f"1step={one_step:.3f} G2b={cell['G_s2b_pass']}", file=sys.stderr)
    return cell


def main() -> int:
    t0 = time.time()
    n_eps, n_steps = (2, 60) if SMOKE else (10, 500)
    audit_kw = (dict(widths=(256,), main_width=256, probe_seeds=(0,), max_steps=400)
                if SMOKE else {})
    audit_kw_perm = (dict(widths=(256,), main_width=256, probe_seeds=(0,), max_steps=400)
                     if SMOKE else dict(widths=(256,), main_width=256, probe_seeds=(0, 1, 2)))

    # step89 spectral column for the report card (read-only cross-reference)
    spectral = {}
    p89 = ROOT / "papers/figures/step89_pretrained_audit.json"
    if p89.exists():
        rec = json.loads(p89.read_text())
        for tk in TASK_KEYS:                                     # flat "{task}-{seed}" schema
            v = rec.get(f"{tk}-{SEED_CKPT}")
            if isinstance(v, dict) and "lambda1" in v:
                spectral[tk] = {"lambda1": v["lambda1"], "lambda1_ci": v.get("lambda1_ci")}

    cells = {}
    for task_key in TASK_KEYS:
        domain, task, obs_dim, adim = TASKS[task_key]
        ck = ROOT / "models/tdmpc2" / f"{task_key}-{SEED_CKPT}.pt"
        print(f"[wma2] loading {task_key} seed {SEED_CKPT} ...", file=sys.stderr)
        slices = load_tdmpc2_slices(ck, action_dim=adim, obs_dim=obs_dim)
        for lane in ("pi", "U"):
            cells[f"{task_key}/{lane}"] = audit_cell(
                task_key, lane, slices, n_eps, n_steps, audit_kw, audit_kw_perm)

    # ---- step-level gates (lane-pi across tasks) ----
    pi_cells = {k: c for k, c in cells.items() if k.endswith("/pi")}
    eta_evaluable = {k: c for k, c in pi_cells.items() if not c["eta_vacuous_by_ceiling"]}
    eta_pass = [k for k, c in eta_evaluable.items() if c["eta"] < 0.2]
    sym_pass = [k for k, c in pi_cells.items() if c["L_sym"] < 2.0]
    gates = {
        # Verdict per spec text: vacuous cells leave the denominator ("不计入通过");
        # tri-state keeps the shrunken-denominator case honest (v1.2; code-bug fix disclosed
        # in the record — the first run's max(2,..) floor contradicted the registered text).
        "G_s2a_eta": {"evaluable_cells": list(eta_evaluable),
                      "vacuous_cells": [k for k in pi_cells if k not in eta_evaluable],
                      "pass_cells": eta_pass,
                      "verdict": ("VACUOUS-BY-CEILING (all cells)" if not eta_evaluable else
                                  (f"PASS-ON-EVALUABLE ({len(eta_pass)}/{len(eta_evaluable)}; "
                                   f"{len(pi_cells) - len(eta_evaluable)}/{len(pi_cells)} vacuous)"
                                   if 3 * len(eta_pass) >= 2 * len(eta_evaluable) else "FAIL"))},
        "G_s2a_sym": {"pass_cells": sym_pass,
                      "verdict": "PASS" if len(sym_pass) >= 2 else "FAIL"},
        "G_s2b": {"verdict": "PASS" if all(c["G_s2b_pass"] for c in cells.values()) else "FAIL"},
        "G_s2c": {"verdict": "PASS" if all(c["G_s2c_pass"] for c in cells.values()) else "FAIL"},
    }
    out = {"model": f"tdmpc2 official single-task ckpts, seed {SEED_CKPT} (step89 loader)",
           "spec": "docs/specs/2026-06-11-wma-step2-tdmpc2-imprint-seed.md (v1.2)",
           "lewm_step1_reference": LEWM_REF,
           "cells": cells, "gates": gates, "spectral_crossref_step89": spectral,
           "wall_seconds": round(time.time() - t0, 1), "smoke": SMOKE}
    tag = "_smoke" if SMOKE else ""
    (ROOT / "papers/figures" / f"wma_step2_tdmpc2_atm{tag}.json").write_text(
        json.dumps(out, indent=2))
    print(f"[wma2] gates: eta={gates['G_s2a_eta']['verdict']} sym={gates['G_s2a_sym']['verdict']} "
          f"s2b={gates['G_s2b']['verdict']} s2c={gates['G_s2c']['verdict']} "
          f"wall={out['wall_seconds']}s -> wma_step2_tdmpc2_atm{tag}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
