r"""P4 #10 — subgoal predictor $G$ on human demos (κ=0), in the deployable latent space.

Spec: docs/specs/2026-06-10-p4-spine-ep43-seed.md §1 (demo source amended 2026-06-11: human
demos; the oracle gate is retired). Design decisions registered here:

- **Source** = diffusion-policy `pusht_cchi_v7_replay.zarr` (the canonical 206 human episodes;
  `state` is the full 5-d pose [agent_xy, block_xy, block_θ]). The lerobot/pusht parquet exposes
  only the 2-d agent state — unusable for re-rendering; recorded as the reason for the zarr
  source. New dev dependency: `zarr` (+numcodecs<0.16), venv-only.
- **State re-rendering, not frame reuse**: demo frames are re-rendered in OUR env via
  `reset(options={'goal_state': FIXED_GOAL, 'state': s5})` at chunk boundaries (every 5 steps,
  both datasets at 10 Hz) — kills rendering-convention mismatch AND replay drift in one move;
  $G$ is an autonomous flow, so demo ACTIONS are not needed at all. `FIXED_GOAL` pins the goal
  zone pixels across all rendered frames (the DP env's goal is fixed; ours randomizes per reset).
- **Latent space** = ckpt3 TARGET encoders (the deployable pair, #9), circular-masked frames.
- **$G$ per base, architecture matching the base's philosophy** (registered conflation note: the
  eq-vs-plain $\hat\delta_G$ comparison mixes base-space and G-architecture; Stage-2 readouts are
  descriptive anyway): plain-G = residual MLP $[2D\to512\to512\to D]$; eq-G = `GroupConv1x1`
  stack on $2F$ stacked regular fields → residual on $z_m$, with an exact-equivariance assert
  (repo rule). Inputs $K{=}2$ past boundary latents; target = next boundary latent.
- Split: episodes 180 train / 26 val; Adam 1e-3, 60 epochs, batch 256, MSE; readouts: val MSE,
  $\hat\delta_G$ (val one-step norm), vs the trivial persistence baseline $z_{m+1} := z_m$.

Run: .venv/bin/python experiments/p4_step10_g_trainer.py   (~5 min)
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

from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, DATA_DIR, RES, build_eq, build_plain, circ_mask, make_env, pick_ladder,
)
from src.models.cn_regular import GroupConv1x1  # noqa: E402

ZARR = ROOT / "data" / "p4_demos" / "pusht" / "pusht_cchi_v7_replay.zarr"
FIXED_GOAL = np.array([256.0, 256.0, 256.0, 256.0, np.pi / 4, 0.0, 0.0])  # recorded constant
N_TRAIN_EP, SEED = 180, 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "papers" / "figures" / "p4_step10_g.json"


def load_demo_states() -> list[np.ndarray]:
    import zarr

    z = zarr.open(str(ZARR), mode="r")
    states = z["data"]["state"][:]                      # (25650, 5)
    ends = z["meta"]["episode_ends"][:]                 # cumulative row ends
    eps, prev = [], 0
    for e in ends:
        eps.append(states[prev:e])
        prev = e
    return eps


def render_boundaries(eps: list[np.ndarray]) -> list[np.ndarray]:
    r"""Per episode: frames at chunk boundaries (T_b, 96, 96, 3) uint8, OUR rendering."""
    env = make_env(None)
    out = []
    t0 = time.time()
    for i, s in enumerate(eps):
        rows = s[::CHUNK]
        frames = []
        for s5 in rows:
            env.reset(options={"goal_state": FIXED_GOAL, "state": s5.astype(np.float64)})
            frames.append(env.render())
        out.append(np.stack(frames).astype(np.uint8))
        if i % 50 == 0:
            print(f"      episode {i}/{len(eps)}  ({time.time()-t0:.0f}s)")
    env.close()
    return out


@torch.no_grad()
def encode_eps(target_enc, ep_frames: list[np.ndarray]) -> list[torch.Tensor]:
    target_enc = target_enc.eval().to(DEVICE)
    zs = []
    for fr in ep_frames:
        f = torch.from_numpy(fr).float().div_(255.0).permute(0, 3, 1, 2)
        zs.append(target_enc(circ_mask(f).to(DEVICE)).cpu().double())
    target_enc.cpu()
    return zs


class PlainG(torch.nn.Module):
    def __init__(self, d: int = 128, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * d, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d),
        )

    def forward(self, z_prev, z_cur):
        return z_cur + self.net(torch.cat([z_prev, z_cur], dim=-1))


class EqG(torch.nn.Module):
    r"""$C_N$-equivariant autonomous subgoal flow: GroupConv stack on $2F$ stacked regular
    fields, residual on $z_m$ — P4.C's group×invariant decomposition by construction."""

    def __init__(self, d: int = 128, n_rot: int = 16, hidden_fields: int = 32):
        super().__init__()
        assert d % n_rot == 0
        self.f, self.n = d // n_rot, n_rot
        self.net = torch.nn.Sequential(
            GroupConv1x1(2 * self.f, hidden_fields, n_rot), torch.nn.ReLU(),
            GroupConv1x1(hidden_fields, hidden_fields, n_rot), torch.nn.ReLU(),
            GroupConv1x1(hidden_fields, self.f, n_rot),
        )

    def forward(self, z_prev, z_cur):
        B = z_cur.shape[0]
        x = torch.cat([z_prev.view(B, self.f, self.n), z_cur.view(B, self.f, self.n)], dim=1)
        return z_cur + self.net(x).reshape(B, -1)


def eqg_equivariance_assert(g: EqG) -> float:
    torch.manual_seed(0)
    g = g.double().eval()
    B, d = 4, g.f * g.n
    zp, zc = torch.randn(B, d, dtype=torch.float64), torch.randn(B, d, dtype=torch.float64)
    worst = 0.0
    with torch.no_grad():
        base = g(zp, zc)
        for s in range(g.n):
            roll = lambda t: torch.roll(t.view(B, g.f, g.n), shifts=s, dims=-1).reshape(B, d)  # noqa: E731
            worst = max(worst, float((g(roll(zp), roll(zc)) - roll(base)).abs().max()))
    assert worst < 1e-10, f"EqG equivariance broken: {worst:.2e}"
    return worst


def windows(zs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zp, zc, zn = [], [], []
    for z in zs:
        if z.shape[0] < 3:
            continue
        zp.append(z[:-2]); zc.append(z[1:-1]); zn.append(z[2:])
    return torch.cat(zp).float(), torch.cat(zc).float(), torch.cat(zn).float()


def train_g(gmod, tr, va, epochs: int = 60, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    gmod = gmod.float().to(DEVICE).train()
    opt = torch.optim.Adam(gmod.parameters(), lr=1e-3)
    zp, zc, zn = (t.to(DEVICE) for t in tr)
    vp, vc, vn = (t.to(DEVICE) for t in va)
    n = zp.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            loss = torch.nn.functional.mse_loss(gmod(zp[idx], zc[idx]), zn[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    gmod.eval()
    with torch.no_grad():
        val_pred = gmod(vp, vc)
        out = {
            "val_mse": float(torch.nn.functional.mse_loss(val_pred, vn)),
            "delta_G": float((val_pred - vn).norm(dim=-1).mean()),
            "delta_persist": float((vc - vn).norm(dim=-1).mean()),
            "n_train": int(n), "n_val": int(vp.shape[0]),
        }
    gmod.cpu()
    return out


def main() -> int:
    t0 = time.time()
    art: dict = {"source": "diffusion-policy pusht_cchi_v7_replay.zarr (206 eps)",
                 "fixed_goal": FIXED_GOAL.tolist(), "chunk": CHUNK, "device": DEVICE,
                 "latent_space": "ckpt3 TARGET encoders (deployable pair)"}

    print("[1/4] demo states -> OUR-env re-rendered boundary frames ...")
    eps = load_demo_states()
    art["episodes"] = len(eps)
    frames = render_boundaries(eps)

    print("[2/4] load ckpt3 target encoders ...")
    w_match = pick_ladder(sum(p.numel() for p in build_eq().parameters()))["plain_match"]
    encs = {}
    for name, builder in (("eq", build_eq), ("plain_match", lambda: build_plain(w_match))):
        ck = torch.load(DATA_DIR / f"ckpt3_{name}_f200.pt", map_location="cpu", weights_only=True)
        m = builder()
        m.load_state_dict(ck["model"])
        m.encoder.load_state_dict(ck["target_encoder"])
        encs[name] = m.encoder

    print("[3/4] encode + train G per base ...")
    art["g"] = {}
    for name, enc in encs.items():
        zs = encode_eps(enc, frames)
        tr, va = windows(zs[:N_TRAIN_EP]), windows(zs[N_TRAIN_EP:])
        gmod = EqG() if name == "eq" else PlainG()
        if name == "eq":
            art.setdefault("eqg_equivariance_max_err", eqg_equivariance_assert(EqG()))
        rep = train_g(gmod, tr, va, seed=SEED)
        torch.save(gmod.state_dict(), DATA_DIR / f"gckpt_{name}.pt")
        art["g"][name] = rep
        print(f"      {name}: val_mse {rep['val_mse']:.4f}  delta_G {rep['delta_G']:.3f}  "
              f"persistence {rep['delta_persist']:.3f}")

    art["wall_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"[4/4] wrote {OUT.name} ({art['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
