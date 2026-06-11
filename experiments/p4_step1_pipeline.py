r"""P4-step1 part 2 — pixel-PushT pipeline: collection → oracle demos → base grid → probes.

Spec: docs/specs/2026-06-10-p4-step1-pipeline-bases-seed.md (gates G0b/G0c/G1/G3).
Registered claims live in the proposal (protocol v1.1); **this script evaluates NO claim gate** —
it builds the machinery and runs the seed-0 sizing pass of E-P4.1.

Registered build decisions (recorded in the artifact):
- Episode length 100 control steps @ 10 Hz; render 96×96; pixels in [0,1] float32 for training.
- Encoder corpus = WeakPolicy episodes (G0b); fractions are nested episode prefixes
  $\{2,10,20,60,200\}$. Probe budgets fixed: 40 held-out train / 20 held-out eval episodes.
- Oracle demos (G0c) = true-env CEM via ``sim.reset(options={'goal_state': g, 'state': s})``
  (verified API): K=64 candidates × 3 iters, elites 8, plan horizon 4, replan every step,
  ≤200 steps; success = env's ``terminated`` (pose match: pos<20 px ∧ angle<π/9).
- Bases: eq = SteerableEncoder($C_{16}$, latent 128) + CNRegularPredictor (the exact-equivariant
  pair, tests/test_p4_step1.py); plain ladder = ConvEncoder widths chosen at runtime to bracket
  eq's parameter count $\{\sim0.5\times, \sim1\times, \sim2\times\}$ + LatentPredictor; aug =
  plain-1× with ×4 rotation pre-expansion (identity + 3 uniform angles; image rotated CCW about
  centre, displacement action rotated by the same $R(+\theta)$ — sign pinned by the calibration
  test) and epochs ÷4 ⇒ **equal optimizer steps** vs plain-1×.
- Training: train_jepa defaults (EMA-target, Muon/AdamW, variance witness), epochs 20, batch 64,
  device MPS-if-available (f32); per-cell histories recorded (G1 = latent_std above 0.1 floor and
  decreasing pred_loss).
- Probes (G3): frozen encoder; heads $[D\to256\to128\to k]$; targets — block $(x,y)/512$,
  $(\cos\theta,\sin\theta)$; effect probe $(z_t,z_{t+1})\to\Delta\theta$ (wrapped). $R^2$ per
  target on the eval split; $\theta$-$R^2$ = joint $R^2$ over the (cos, sin) pair.
- Wall-clock honesty: collection is regenerated from seeds (env+policy deterministic); oracle
  demos are cached to ``data/p4_step1/`` (expensive); JSON artifact to ``papers/figures/``.

Run (smoke):  P4_SMOKE=1 .venv/bin/python experiments/p4_step1_pipeline.py
Run (full):   .venv/bin/python experiments/p4_step1_pipeline.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.cn_regular import CNRegularPredictor  # noqa: E402
from src.models.eqjepa import ConvEncoder, EqJEPA, SteerableEncoder  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

SMOKE = bool(int(os.environ.get("P4_SMOKE", "0")))
SKIP_ORACLE = bool(int(os.environ.get("P4_SKIP_ORACLE", "0")))  # G0c verdict stands; solver fix queued
SEED = 0
RES = 96
CHUNK = 5  # protocol v1.2: transitions are 5-env-step action chunks (1 f-chunk = 5 env steps)
EP_LEN = 8 if SMOKE else 100
N_COLLECT = 4 if SMOKE else 260  # 200 corpus + 40 probe-train + 20 probe-eval
FRACTIONS = [2, 3] if SMOKE else [2, 10, 20, 60, 200]
N_PROBE_TRAIN = 1 if SMOKE else 40
N_PROBE_EVAL = 1 if SMOKE else 20
EPOCHS = 2 if SMOKE else 20
ORACLE_ATTEMPTS = 2 if SMOKE else 20  # G0c gate sample (full demo generation is a later overnight)
ORACLE_MAX_STEPS = 30 if SMOKE else 200
DEVICE = "mps" if torch.backends.mps.is_available() and not SMOKE else "cpu"
DATA_DIR = ROOT / "data" / "p4_step1"
OUT_JSON = ROOT / "papers" / "figures" / ("p4_step1_smoke.json" if SMOKE else "p4_step1_v12.json")
# (v1.1's artifact p4_step1.json is committed history — never overwritten)


# ----------------------------------------------------------------------------- env + collection
def make_env(kappa: float | None = None):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    kwargs = dict(resolution=RES)
    if kappa is not None and kappa > 0:
        kwargs["damping"] = float(kappa)
    return gym.make("swm/PushT-v1", **kwargs)


def weak_action(env, rng: np.random.Generator, dist_constraint: float = 100.0) -> np.ndarray:
    r"""Verbatim replica of WeakPolicy's sampling (expert_policy.py:66-81) for a SINGLE env.

    The vendored class is VecEnv-shaped (``actions[i] = action`` breaks on a single env's
    ``action_space.shape == (2,)``); the distribution is reproduced exactly: uniform displacement,
    scaled to canvas, clipped to a ``dist_constraint`` box around the block, mapped back to the
    relative action space.
    """
    e = env.unwrapped
    action = rng.uniform(-1, 1, size=2) * e.action_scale + np.array(e.agent.position)
    block = np.array((e.block.position.x, e.block.position.y))
    action = np.clip(action, block - dist_constraint, block + dist_constraint)
    action = (action - np.array(e.agent.position)) / e.action_scale
    return np.clip(action, -1, 1).astype(np.float32)


def collect_weakpolicy(n_episodes: int, seed: int, kappa: float | None = None) -> dict:
    r"""G0b: WeakPolicy episodes → frames uint8 (E, T+1, 96, 96, 3), states (E, T+1, 7), actions (E, T, 2)."""
    env = make_env(kappa)
    rng = np.random.default_rng(seed)
    frames, states, actions = [], [], []
    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed * 100_000 + ep)
        f = [env.render()]
        s = [obs["state"]]
        a = []
        for _ in range(EP_LEN):
            act = weak_action(env, rng)
            obs, _r, _term, _trunc, _info = env.step(act)
            f.append(env.render())
            s.append(obs["state"])
            a.append(act)
        frames.append(np.stack(f))
        states.append(np.stack(s))
        actions.append(np.stack(a))
    env.close()
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "states": np.stack(states).astype(np.float64),
        "actions": np.stack(actions).astype(np.float32),
    }


# ----------------------------------------------------------------------------- oracle CEM (G0c)
def oracle_cem_episode(env, sim, seed: int, k: int = 64, iters: int = 3, elite: int = 8, horizon: int = 4):
    r"""True-env MPC: plan on ``sim`` clones via reset(options={'goal_state','state'}), execute on ``env``."""
    obs, _ = env.reset(seed=seed)
    goal = np.asarray(env.unwrapped.goal_state, dtype=np.float64)
    rng = np.random.default_rng(seed)
    traj_a, traj_s = [], [obs["state"]]
    for _t in range(ORACLE_MAX_STEPS):
        s = np.asarray(obs["state"], dtype=np.float64)
        mu = np.zeros((horizon, 2), dtype=np.float64)
        sig = np.ones((horizon, 2), dtype=np.float64) * 0.6
        for _i in range(iters):
            cand = np.clip(rng.normal(mu, sig, size=(k, horizon, 2)), -1, 1)
            costs = np.empty(k)
            for j in range(k):
                sim.reset(options={"goal_state": goal, "state": s})
                dist = None
                for h in range(horizon):
                    o2, r, term, _tr, _ = sim.step(cand[j, h].astype(np.float32))
                    dist = -float(r)  # reward = -||goal - state||
                    if term:
                        dist -= 1000.0  # success bonus
                        break
                costs[j] = dist
            order = np.argsort(costs)[:elite]
            mu, sig = cand[order].mean(0), cand[order].std(0) + 1e-3
        act = mu[0].astype(np.float32)
        obs, _r, term, _tr, _ = env.step(act)
        traj_a.append(act)
        traj_s.append(obs["state"])
        if term:
            return True, np.stack(traj_s), np.stack(traj_a)
    return False, np.stack(traj_s), np.stack(traj_a)


def gate_g0c(seed: int, kappa: float | None = None) -> dict:
    env, sim = make_env(kappa), make_env(kappa)
    succ = 0
    t0 = time.time()
    for i in range(ORACLE_ATTEMPTS):
        ok, _s, _a = oracle_cem_episode(env, sim, seed=seed * 7_000 + i)
        succ += int(ok)
    env.close(), sim.close()
    rate = succ / ORACLE_ATTEMPTS
    return {"attempts": ORACLE_ATTEMPTS, "successes": succ, "rate": rate, "sec": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- transitions + aug
_CIRC = None


def circ_mask(x: torch.Tensor) -> torch.Tensor:
    r"""Fixed circular mask applied to ALL frames everywhere (v1.2: kills the rotation
    corner-wedge cue at the source; applied uniformly to augmented and unaugmented data, train
    and probe — metric fairness)."""
    global _CIRC
    if _CIRC is None or _CIRC.shape[-1] != x.shape[-1]:
        r = x.shape[-1] / 2.0
        yy, xx = torch.meshgrid(torch.arange(x.shape[-2]), torch.arange(x.shape[-1]), indexing="ij")
        _CIRC = (((yy - r + 0.5) ** 2 + (xx - r + 0.5) ** 2) <= r**2).to(torch.float32)[None, None]
    return x * _CIRC.to(x.dtype)


def to_transitions(data: dict, n_episodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""v1.2: 5-step action-chunk transitions on chunk-boundary frames, circular-masked."""
    f = torch.from_numpy(data["frames"][:n_episodes]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    a = torch.from_numpy(data["actions"][:n_episodes])  # (E, T, 2)
    T = a.shape[1]
    n_ch = T // CHUNK
    obs = f[:, 0 : n_ch * CHUNK : CHUNK].reshape(-1, 3, RES, RES)
    nxt = f[:, CHUNK : n_ch * CHUNK + 1 : CHUNK].reshape(-1, 3, RES, RES)
    act = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).reshape(-1, CHUNK * 2)
    return obs, act, nxt


def rotate_images(x: torch.Tensor, theta) -> torch.Tensor:
    r"""CCW rotation about the image centre; ``theta`` is a float or a per-sample tensor (B,)."""
    if not torch.is_tensor(theta):
        theta = torch.full((x.shape[0],), float(theta), dtype=x.dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(c)
    # grid_sample samples input at grid locations: the affine matrix is the INVERSE rotation.
    mat = torch.stack([torch.stack([c, s, zero], -1), torch.stack([-s, c, zero], -1)], 1)  # (B,2,3)
    grid = F.affine_grid(mat, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def augment_x4(obs, act, nxt, seed: int):
    r"""v1.2: identity + 3 copies with PER-SAMPLE uniform angles (the per-copy drawing of v1.1
    produced a 4-discrete-angle dataset and a corner-cue cluster collapse — step1b/c, confirmed);
    every chunked sub-action rotates by the sample's own $R(+\theta)$ (sign pinned by the
    calibration test). Combined with the universal circular mask, the corner cue is gone."""
    g = torch.Generator().manual_seed(seed)
    B = obs.shape[0]
    n_sub = act.shape[1] // 2
    outs_o, outs_a, outs_n = [obs], [act], [nxt]
    for _ in range(3):
        th = torch.rand(B, generator=g) * 2 * math.pi
        c, s = torch.cos(th), torch.sin(th)
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], 1)  # (B,2,2)
        outs_o.append(circ_mask(rotate_images(obs, th)))
        outs_n.append(circ_mask(rotate_images(nxt, th)))
        a_rot = torch.einsum("bij,bcj->bci", R.to(act.dtype), act.view(B, n_sub, 2))
        outs_a.append(a_rot.reshape(B, n_sub * 2))
    return torch.cat(outs_o), torch.cat(outs_a), torch.cat(outs_n)


# ----------------------------------------------------------------------------- bases
def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_eq() -> EqJEPA:
    n_rot, width = (4, 4) if SMOKE else (16, 8)
    enc = SteerableEncoder(in_channels=3, latent_dim=128, n_rot=n_rot, width=width)
    pred = CNRegularPredictor(latent_dim=128, action_dim=2 * CHUNK, n_rot=n_rot,
                              hidden_fields=8 if SMOKE else 32)
    return EqJEPA(latent_dim=128, action_dim=2 * CHUNK, encoder=enc, predictor=pred)


def build_plain(width: int) -> EqJEPA:
    return EqJEPA(latent_dim=128, action_dim=2 * CHUNK, encoder=ConvEncoder(3, 128, width=width))


def pick_ladder(target: int) -> dict[str, int]:
    r"""Choose ConvEncoder widths bracketing the eq param count {~0.5x, ~1x, ~2x}.

    Widths are multiples of 8 (the vendored ConvEncoder uses GroupNorm(8, width)).
    """
    counts = {w: count_params(build_plain(w)) for w in (8, 16, 24, 32, 48, 64)}
    best = min(counts, key=lambda w: abs(counts[w] - target))
    half = min(counts, key=lambda w: abs(counts[w] - target / 2))
    dbl = min(counts, key=lambda w: abs(counts[w] - target * 2))
    return {"plain_half": half, "plain_match": best, "plain_double": dbl}


# ----------------------------------------------------------------------------- probes (G3)
class Probe(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.net(x)


def r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    sse = ((pred - tgt) ** 2).sum().item()
    sst = ((tgt - tgt.mean(0)) ** 2).sum().item()
    return 1.0 - sse / max(sst, 1e-12)


def fit_probe(z_tr, y_tr, z_ev, y_ev, epochs=100 if not SMOKE else 5) -> float:
    probe = Probe(z_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(z_tr.shape[0])
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            loss = F.mse_loss(probe(z_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return r2(probe(z_ev), y_ev)


def probe_targets(states: np.ndarray):
    r"""states (E, T+1, 7) → per-frame targets: block (x,y)/512, (cosθ, sinθ); per-step Δθ wrapped."""
    s = torch.from_numpy(states).float()
    xy = s[..., 2:4] / 512.0
    th = s[..., 4]
    cs = torch.stack([th.cos(), th.sin()], dim=-1)
    dth = th[:, 1:] - th[:, :-1]
    dth = torch.atan2(dth.sin(), dth.cos())  # wrap to (-pi, pi]
    return xy, cs, dth


@torch.no_grad()
def encode_all(model: EqJEPA, frames: np.ndarray, device: str) -> torch.Tensor:
    f = torch.from_numpy(frames).float().div_(255.0).permute(0, 1, 4, 2, 3)
    E, T = f.shape[0], f.shape[1]
    flat = circ_mask(f.reshape(-1, 3, RES, RES))
    z = model.encode(flat.to(device)).cpu()
    return z.reshape(E, T, -1)


def base_device(name: str, art: dict) -> str:
    r"""Per-base device probe: e2cnn's basis expansion is unverified on MPS — try a tiny
    forward+backward once; on failure fall back to CPU (recorded in the artifact)."""
    if DEVICE != "mps":
        return DEVICE
    cache = art.setdefault("device_probe", {})
    key = "eq" if name == "eq" else "plain"
    if key not in cache:
        try:
            m = build_eq() if key == "eq" else build_plain(8)
            m.to("mps")
            z = m.encoder(torch.randn(2, 3, RES, RES, device="mps"))
            z.sum().backward()
            cache[key] = "mps"
        except Exception as exc:  # noqa: BLE001 — any MPS failure ⇒ honest CPU fallback
            cache[key] = "cpu"
            cache[f"{key}_mps_error"] = str(exc)[:200]
    return cache[key]


def run_probes(model: EqJEPA, probe_tr: dict, probe_ev: dict, device: str = "cpu") -> dict:
    r"""v1.2: probes live on chunk boundaries (Δθ becomes per-chunk, ~3.5° median motion)."""
    probe_tr = {k: v[:, ::CHUNK] for k, v in probe_tr.items() if k in ("frames", "states")}
    probe_ev = {k: v[:, ::CHUNK] for k, v in probe_ev.items() if k in ("frames", "states")}
    model.eval().to(device)
    z_tr, z_ev = encode_all(model, probe_tr["frames"], device), encode_all(model, probe_ev["frames"], device)
    out = {}
    (xy_tr, cs_tr, dth_tr), (xy_ev, cs_ev, dth_ev) = probe_targets(probe_tr["states"]), probe_targets(probe_ev["states"])
    flat = lambda t: t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t.reshape(-1, 1)  # noqa: E731
    out["xy_r2"] = fit_probe(z_tr.reshape(-1, z_tr.shape[-1]), flat(xy_tr), z_ev.reshape(-1, z_ev.shape[-1]), flat(xy_ev))
    out["theta_r2"] = fit_probe(z_tr.reshape(-1, z_tr.shape[-1]), flat(cs_tr), z_ev.reshape(-1, z_ev.shape[-1]), flat(cs_ev))
    zz_tr = torch.cat([z_tr[:, :-1], z_tr[:, 1:]], dim=-1).reshape(-1, 2 * z_tr.shape[-1])
    zz_ev = torch.cat([z_ev[:, :-1], z_ev[:, 1:]], dim=-1).reshape(-1, 2 * z_ev.shape[-1])
    out["dtheta_r2"] = fit_probe(zz_tr, flat(dth_tr), zz_ev, flat(dth_ev))
    model.cpu()
    return out


# ----------------------------------------------------------------------------- main
def main() -> int:
    torch.manual_seed(SEED)
    t0 = time.time()
    art: dict = {"smoke": SMOKE, "seed": SEED, "res": RES, "ep_len": EP_LEN, "device": DEVICE,
                 "fractions": FRACTIONS, "epochs": EPOCHS,
                 "protocol": "v1.2 (stride-5 chunks, circular mask, per-sample aug angles)",
                 "chunk": CHUNK}

    print(f"[1/5] G0b collection: {N_COLLECT} WeakPolicy episodes ...")
    data = collect_weakpolicy(N_COLLECT, seed=SEED)
    corpus_n = N_COLLECT - N_PROBE_TRAIN - N_PROBE_EVAL
    corpus = {k: v[:corpus_n] for k, v in data.items()}
    probe_tr = {k: v[corpus_n : corpus_n + N_PROBE_TRAIN] for k, v in data.items()}
    probe_ev = {k: v[corpus_n + N_PROBE_TRAIN :] for k, v in data.items()}
    art["g0b"] = {"episodes": int(N_COLLECT), "corpus": int(corpus_n),
                  "frames_shape": list(data["frames"].shape)}
    print(f"      corpus {corpus_n} eps, probe {N_PROBE_TRAIN}+{N_PROBE_EVAL} eps  ({time.time()-t0:.0f}s)")

    if SKIP_ORACLE:
        art["g0c"] = {"skipped": True, "note": "verdict FAIL of 2026-06-10 stands; solver fix queued"}
        print("[2/5] G0c oracle gate SKIPPED (P4_SKIP_ORACLE=1; standing verdict unchanged)")
    else:
        print(f"[2/5] G0c oracle-CEM gate: {ORACLE_ATTEMPTS} attempts ...")
        art["g0c"] = gate_g0c(SEED)
        art["g0c"]["verdict"] = "PASS" if art["g0c"]["rate"] >= 0.8 else "FAIL"
        print(f"      success {art['g0c']['successes']}/{art['g0c']['attempts']} -> {art['g0c']['verdict']} "
              f"({art['g0c']['sec']}s)")
    # NOTE: a G0c FAIL is recorded honestly but does NOT abort the run — the training grid does not
    # consume the oracle demos (they feed G in a later step); the gate verdict stands in the record.

    print("[3/5] bases + param ladder ...")
    eq = build_eq()
    eq_n = count_params(eq)
    ladder = pick_ladder(eq_n) if not SMOKE else {"plain_match": 8}
    bases: dict[str, EqJEPA] = {"eq": eq}
    for name, w in ladder.items():
        bases[name] = build_plain(w)
    art["params"] = {"eq": eq_n, **{n: count_params(m) for n, m in bases.items() if n != "eq"}}
    print(f"      params: {art['params']}")

    print(f"[4/5] G1 training grid: {len(bases)+1} bases x {len(FRACTIONS)} fractions on {DEVICE} ...")
    art["grid"] = {}
    for frac in FRACTIONS:
        obs, act, nxt = to_transitions(corpus, frac)
        for name in list(bases) + ["aug"]:
            key = f"{name}@{frac}"
            torch.manual_seed(SEED)
            if name == "aug":
                model = build_plain(ladder["plain_match"])
                o, a, n = augment_x4(obs, act, nxt, seed=SEED)
                ep = max(1, EPOCHS // 4)  # equal optimizer steps vs plain_match
            else:
                model = build_eq() if name == "eq" else build_plain(ladder[name])
                o, a, n = obs, act, nxt
                ep = EPOCHS
            dev = base_device(name, art)
            hist = train_jepa(model, o, a, n, epochs=ep, batch_size=64, device=dev,
                              seed=SEED, verbose=False)
            cell = {"pred_loss": hist["pred_loss"][-1], "latent_std": hist.get("latent_std", float("nan")),
                    "transitions": int(o.shape[0]), "epochs": ep, "device": dev}
            print(f"      {key}: pred_loss {cell['pred_loss']:.4f}, latent_std {cell['latent_std']:.3f} [{dev}]")
            cell["probes"] = run_probes(model, probe_tr, probe_ev, device=dev)
            print(f"        probes: {cell['probes']}")
            # checkpoint for downstream steps (audit/G-training consume trained bases; added in
            # post-launch review — the original overnight run predates this and relies on
            # seed-determinism for re-materialization)
            ck = DATA_DIR / f"ckpt_{name}_f{frac}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck)
            cell["ckpt"] = str(ck.relative_to(ROOT))
            art["grid"][key] = cell
            del model

    print("[5/5] artifact ...")
    g1_ok = all(c["latent_std"] > 0.1 for c in art["grid"].values() if not math.isnan(c["latent_std"]))
    art["g1_latent_std_ok"] = bool(g1_ok)
    art["wall_sec"] = round(time.time() - t0, 1)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(art, indent=1))
    print(f"      wrote {OUT_JSON}  (total {art['wall_sec']}s)  G1 latent_std ok: {g1_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
