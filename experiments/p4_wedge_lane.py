r"""P4 wedge lane — C1b-prelim + C3-wedge: does the certificate transport across the orbit?

Data (verified 2026-06-11, 100% in-constraint): wedge_corpus (200 eps, initial block angle
$\theta_0 \in [0°, 90°)$), wedge_ho_in (60, in-wedge), wedge_ho_out (60, $\theta_0 \in
[90°, 360°)$). Training sees ONLY the wedge; out-of-wedge initial poses are group-OOD.

Arms (recipes with the most 200-ep run history; new-candidate aux0.5_v0.3 deliberately NOT
used here — one untested thing at a time): eq = v1.6 winner (e0.99, v0.2); plain = ep40
winner (v0.04, ep40, width-matched). n = 6 runs/arm, stability floor std ≥ 0.7 (v1.6).

Per stable pair (audits CPU f64, window 16 / k 3 / burn-in 4 / h_max 8):
- audit + dual-boundary cells (CANONICAL src/audit/gates — never copied) on ho_in AND ho_out;
- eq only, descriptive: ORBIT-TRANSPORT readout — rotate ho_out frames AND action vectors by
  the nearest-$C_{16}$ inverse of $\theta_0$ (residual ≤ 11.25°, documented), bringing initial
  poses (approximately) into the wedge; $\hat\delta(\text{transported})$ vs
  $\hat\delta(\text{in})$ tests Lemma-2 orbit-constancy through the actual pixel pipeline.

REGISTERED GATES (declared before any result):
- **G-W1 (C3-wedge guarantee):** faithful one-sided (h_cert ≤ h_meas, ALL ε cells, canonical
  semantics) on **ho_out**, in ≥ 90% of stable eq runs. INCONCLUSIVE-BY-STABILITY if < 4
  stable eq runs.
- **G-W2 (C1b-prelim, sign gate):** median over stable runs of the degradation ratio
  $\hat\delta(\text{out})/\hat\delta(\text{in})$ is strictly smaller for eq than for plain.
  Effect size + bootstrap CIs reported descriptively; no margin is claimed at this tier.

Run: nohup .venv/bin/python -u experiments/p4_wedge_lane.py  (~2 h, MPS lane 2 + CPU audits)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_cpu_batch_0612 import tensorize  # noqa: E402
from experiments.p4_spine_stage1a import measured_err  # noqa: E402
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, pick_ladder, rotate_images,
    to_transitions,
)
from experiments.p4_v16_stageA_sweep import run_one, state_targets  # noqa: E402
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.audit.gates import dual_boundary_cells, faithful_guar, violations  # noqa: E402

T0 = time.time()
FLOOR = 0.7
N_RUNS = 6
C16_STEP = 2 * np.pi / 16
OUT = ROOT / "papers" / "figures" / "p4_wedge_lane.json"

EQ_CFG = dict(ema_decay=0.99, var_coef=0.2, lr_scale=1.0)
PLAIN_CFG = dict(ema_decay=0.99, var_coef=0.04, lr_scale=1.0, epochs=40)


def load_set(tag: str) -> dict:
    z = np.load(DATA_DIR / f"{tag}.npz")
    return {k: z[k] for k in ("frames", "states", "actions")}


def transport_to_wedge(ho: dict) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Rotate frames + action vectors by the nearest-$C_{16}$ inverse of $\theta_0$ per episode.

    Group action on the observation: scene rotation about the (circular-masked) arena centre;
    relative-displacement actions co-rotate, $a \mapsto R(-\theta_g)a$. Returns (fb_d, ach_d)
    in the audit convention (chunk-boundary frames f64, chunked actions f64).
    """
    th0 = torch.from_numpy(ho["states"][:, 0, 4]).double() % (2 * np.pi)
    thg = torch.round(th0 / C16_STEP) * C16_STEP                  # nearest C16 element
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    e, t1 = f.shape[0], f.shape[1]
    flat = circ_mask(f.reshape(-1, 3, RES, RES))
    ang = (-thg).float().repeat_interleave(t1)                    # per-frame angle
    flat = circ_mask(rotate_images(flat, ang))                    # rotate, re-mask edges
    fb = flat.reshape(e, t1, 3, RES, RES)[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"]).double()                  # (E, T, 2)
    c, s = torch.cos(-thg), torch.sin(-thg)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (E,2,2)
    a_rot = torch.einsum("eij,etj->eti", rot, a)
    n_ch = a.shape[1] // CHUNK
    ach = a_rot[:, : n_ch * CHUNK].reshape(e, n_ch, CHUNK * 2)
    return fb, ach


def main() -> int:
    art: dict = {"gates": "G-W1 (faithful one-sided on ho_out, >=90% stable eq) + "
                          "G-W2 (sign: median out/in delta ratio, eq < plain)",
                 "cfgs": {"eq": str(EQ_CFG), "plain": str(PLAIN_CFG)}, "cells": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[setup] wedge corpora ...")
    corpus = load_set("wedge_corpus")
    obs, act, nxt = to_transitions(corpus, 200)
    aux_t = state_targets(corpus)
    ho_in, ho_out = load_set("wedge_ho_in"), load_set("wedge_ho_out")
    fb_in, ach_in = tensorize(ho_in)
    fb_out, ach_out = tensorize(ho_out)
    fb_tr, ach_tr = transport_to_wedge(ho_out)
    fb_small = fb_in[:20].reshape(-1, 3, RES, RES).float()
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    builders = {"eq": build_eq, "plain": (lambda: build_plain(w_match))}
    cfgs = {"eq": EQ_CFG, "plain": PLAIN_CFG}

    for arm in ("eq", "plain"):
        print(f"[{arm}] {N_RUNS} runs ...")
        for r in range(N_RUNS):
            try:
                cell, pr, m, tgt = run_one(builders[arm], cfgs[arm], r, obs, act, nxt,
                                           fb_in, ach_in, fb_small, ho_in, aux_t)
                if cell["stable"]:
                    for tag, fb, ach in (("in", fb_in, ach_in), ("out", fb_out, ach_out)):
                        rep = audit_gap(pr, fb, ach, window=16, k=3, burn_in=4, h_max=8)
                        meas = measured_err(pr, fb, ach, 8)
                        dm = rep["delta"]["mean"]
                        cells = dual_boundary_cells(meas["q90"],
                                                    rep["certified_curve"]["err_q90"], dm)
                        cell[tag] = {"delta": round(dm, 3), "cells": cells,
                                     "guar": faithful_guar(cells),
                                     "viol": len(violations(cells))}
                    if arm == "eq":  # orbit-transport readout (descriptive)
                        rep_t = audit_gap(pr, fb_tr, ach_tr, window=16, k=3, burn_in=4, h_max=8)
                        cell["transported_delta"] = round(rep_t["delta"]["mean"], 3)
                    cell["out_in_ratio"] = round(cell["out"]["delta"] / cell["in"]["delta"], 3)
                    torch.save({"model": m.state_dict(), "target_encoder": tgt.state_dict()},
                               DATA_DIR / f"ckpt9_wedge_{arm}_r{r}.pt")
            except Exception as exc:  # noqa: BLE001
                cell = {"error": str(exc)[:200], "stable": False}
            art["cells"][f"{arm}_r{r}"] = cell
            save()
            print(f"  {arm} r{r}: std {cell.get('std', float('nan')):.3f} "
                  f"in-δ {cell.get('in', {}).get('delta')} out-δ {cell.get('out', {}).get('delta')} "
                  f"guar(out) {cell.get('out', {}).get('guar')} "
                  f"transp {cell.get('transported_delta', '—')}")

    # verdicts (registered)
    verd: dict = {}
    stable = {arm: [c for k, c in art["cells"].items()
                    if k.startswith(arm) and c.get("stable") and "out" in c]
              for arm in ("eq", "plain")}
    eq_s = stable["eq"]
    if len(eq_s) < 4:
        verd["G_W1"] = f"INCONCLUSIVE-BY-STABILITY (stable eq {len(eq_s)}/6)"
    else:
        n_pass = sum(c["out"]["guar"] for c in eq_s)
        verd["G_W1"] = {"verdict": "PASS" if n_pass / len(eq_s) >= 0.9 else "FAIL",
                        "passing": f"{n_pass}/{len(eq_s)}"}
    ratios = {arm: sorted(c["out_in_ratio"] for c in cs) for arm, cs in stable.items()}
    if all(len(v) >= 3 for v in ratios.values()):
        med = {arm: float(np.median(v)) for arm, v in ratios.items()}
        rng = np.random.default_rng(0)
        boots = {arm: [float(np.median(rng.choice(v, len(v)))) for _ in range(2000)]
                 for arm, v in ratios.items()}
        verd["G_W2"] = {"verdict": "PASS" if med["eq"] < med["plain"] else "FAIL",
                        "median_out_in": med,
                        "ci90": {a: [round(float(np.percentile(b, 5)), 3),
                                     round(float(np.percentile(b, 95)), 3)]
                                 for a, b in boots.items()},
                        "ratios": ratios}
    else:
        verd["G_W2"] = f"INCONCLUSIVE-BY-STABILITY (stable: { {a: len(v) for a, v in ratios.items()} })"
    if eq_s:
        verd["orbit_transport_descriptive"] = {
            "transported_vs_in": [(c["transported_delta"], c["in"]["delta"]) for c in eq_s]}
    art["verdicts"] = verd
    save()
    print("VERDICTS:", json.dumps(verd, indent=1)[:600])
    print(f"WEDGE LANE DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
