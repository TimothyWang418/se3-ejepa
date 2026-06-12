r"""Integrated campaign D2 — planner v2 sweep (Component A; spec 2026-06-12, Amendment A1 era).

Per banked pair (9): motion-selected windows from FRESH held-out (seed 3 — unseen by training
seed-0 corpus and by the seed-1 audit/D1 sets), rows = {H* (certificate-chosen a priori from
the pair's STANDING seed-1 audit), fixed H ∈ {1,2,4,8}, zero-action, random-action}.

Window protocol (planner v2 as registered): T_far = 8 chunks; window (e, t) qualifies iff
$\lVert z(e, t{+}8) - z(e, t) \rVert > 4\hat\delta$ (motion selection — kills quasi-static
vacuity); W = 30 windows/pair (fixed rng). Row execution: subgoals $z(t{+}kH)$, CEM plans each
segment (budgets FROZEN at Stage-1b defaults via import), closed-loop re-encode at segment
boundaries; reach = final true latent within $\epsilon_{\mathrm{reach}} = 2\hat\delta$ (D1) of
the final target. NO reset between segments (one reset per window).

Run: nohup .venv/bin/python -u experiments/p4_campaign_d2.py > /tmp/p4_d2.log 2>&1 &  (~1.5 h)
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.p4_spine_stage1a import measured_err  # noqa: E402  (audit convention import)
from experiments.p4_spine_stage1b import FIXED_GOAL, cem_plan  # noqa: E402  (frozen budgets ride along)
from experiments.p4_spine_stage2_kappa08 import Pair  # noqa: E402
from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, circ_mask, collect_weakpolicy, make_env,
)
from src.audit.gap_mode import audit_gap  # noqa: E402
from src.audit.gates import boundary_from_curve  # noqa: E402

T0 = time.time()
T_FAR = 8
N_WIN = 30
FIXED_ROWS = (1, 2, 4, 8)
OUT = ROOT / "papers" / "figures" / "p4_campaign_d2.json"
D1 = json.loads((ROOT / "papers/figures/p4_campaign_d1.json").read_text())


class DeployCand:
    r"""Deployable pair (EMA target encoder + f64 predictor) from a candidate ckpt."""

    def __init__(self, r: int) -> None:
        ck = torch.load(DATA_DIR / f"ckpt8_cand_champ_r{r}.pt",
                        map_location="cpu", weights_only=True)
        m = build_eq()
        m.load_state_dict(ck["model"])
        m.encoder.load_state_dict(ck["target_encoder"])
        self.enc = m.encoder.eval()
        self.pred = copy.deepcopy(m.predictor).double().eval()
        for p in self.pred.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_frames(self, frames_u8: np.ndarray) -> torch.Tensor:
        f = torch.from_numpy(frames_u8).float().div_(255.0)
        lead = f.shape[:-3]
        f = circ_mask(f.reshape(-1, RES, RES, 3).permute(0, 3, 1, 2))
        return self.enc(f).cpu().double().reshape(*lead, -1)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        z = z0
        for h in range(acts.shape[1]):
            z = self.pred(z, acts[:, h])
        return z


def run_window(dp: DeployCand, env, state7: np.ndarray, zb_ep: torch.Tensor, t: int,
               row, rng: torch.Generator) -> bool:
    r"""One window, one row. Returns reach success (final true latent within eps_reach)."""
    z_final_target = zb_ep[t + T_FAR].unsqueeze(0)
    env.reset(options={"goal_state": FIXED_GOAL, "state": state7})
    if row == "zero":
        for _ in range(T_FAR * CHUNK):
            env.step(np.zeros(2, dtype=np.float32))
    elif row == "rand":
        for _ in range(T_FAR * CHUNK):
            env.step(np.random.default_rng(int(torch.randint(1 << 30, (1,), generator=rng))).uniform(-1, 1, 2).astype(np.float32))
    else:
        h_row = int(row)
        z_cur = dp.encode_frames(np.asarray(env.render())[None])[0].unsqueeze(0)
        k = 0
        while k < T_FAR:
            h_seg = min(h_row, T_FAR - k)
            z_sub = zb_ep[t + k + h_seg].unsqueeze(0)
            plan = cem_plan(dp, z_cur, z_sub, h_seg, rng)
            for h in range(h_seg):
                for c in range(CHUNK):
                    env.step(plan[h, 2 * c : 2 * c + 2].float().numpy())
            k += h_seg
            z_cur = dp.encode_frames(np.asarray(env.render())[None])[0].unsqueeze(0)
        z_reached = z_cur
        return bool((z_reached - z_final_target).norm().item() <= D1["pairs"][f"r{dp.r}"]["eps_reach"])
    z_reached = dp.encode_frames(np.asarray(env.render())[None])[0].unsqueeze(0)
    return bool((z_reached - z_final_target).norm().item() <= D1["pairs"][f"r{dp.r}"]["eps_reach"])


def main() -> int:
    art: dict = {"design": "planner v2 (motion windows, frozen CEM, fresh seed-3 ho)",
                 "pairs": {}}

    def save():
        art["elapsed_min"] = round((time.time() - T0) / 60, 1)
        OUT.write_text(json.dumps(art, indent=1))

    print("[D2] fresh held-out (seed 3) ...")
    ho = collect_weakpolicy(60, seed=3)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb_d = f[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach_d = a[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()
    # seed-1 standing-audit tensors (H* must be chosen a priori, not from seed-3)
    ho1 = collect_weakpolicy(60, seed=1)
    f1 = torch.from_numpy(ho1["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f1 = circ_mask(f1.reshape(-1, 3, RES, RES)).reshape(f1.shape)
    fb1 = f1[:, ::CHUNK].double()
    a1 = torch.from_numpy(ho1["actions"])
    ach1 = a1[:, : n_ch * CHUNK].reshape(60, n_ch, CHUNK * 2).double()

    env = make_env(None)
    for rkey, binding in D1["pairs"].items():
        r = int(rkey[1:])
        dp = DeployCand(r)
        dp.r = r
        # H* a priori: standing seed-1 audit -> certified q90 curve -> boundary at eps_reach
        rep = audit_gap(Pair(dp.enc, dp.pred), fb1, ach1, window=16, k=3, burn_in=4, h_max=8)
        h_star_raw = boundary_from_curve(rep["certified_curve"]["err_q90"], binding["eps_reach"])
        h_star = max(1, h_star_raw)   # 0 -> clamp to 1 for executability; raw value recorded
        zb = dp.encode_frames(ho["frames"][:, ::CHUNK])           # (60, n_ch+1, D)
        delta = binding["delta"]
        cands = [(e, t) for e in range(60) for t in range(n_ch + 1 - T_FAR)
                 if (zb[e, t + T_FAR] - zb[e, t]).norm().item() > 4 * delta]
        rng = torch.Generator().manual_seed(r)
        sel = [cands[i] for i in torch.randperm(len(cands), generator=rng)[:N_WIN].tolist()]
        rows: dict = {}
        for row in (("hstar",) + FIXED_ROWS + ("zero", "rand")):
            h_eff = h_star if row == "hstar" else row
            ok = 0
            for (e, t) in sel:
                state7 = ho["states"][e, t * CHUNK, :7].astype(np.float64)
                ok += run_window(dp, env, state7, zb[e], t, h_eff if row != "hstar" else h_star,
                                 rng)
            rows[str(row)] = {"reach": round(ok / len(sel), 3), "n": len(sel)}
            print(f"  r{r} H*={h_star} row {row}: reach {rows[str(row)]['reach']}")
            art["pairs"][rkey] = {"h_star": h_star, "h_star_raw": int(h_star_raw),
                                  "n_motion_cands": len(cands), "rows": rows}
            save()
    env.close()
    print(f"D2 DONE ({(time.time() - T0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
