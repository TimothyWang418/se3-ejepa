r"""Step 91 — LeWM audit: E13's SECOND public pretrained world-model family (design: docs/specs/2026-06-10-step91-lewm-audit-seed.md).

ZERO-REPLICA path: the official `stable-worldmodel` package is installed, so the checkpoint loads into the AUTHORS'
OWN code via hydra-instantiate + strict=True (G0a passed before this file was written; transformers<5 pinned for the
classic ViT naming the checkpoint uses). The audit state is the 3-frame delay embedding $s=(z_{t-2},z_{t-1},z_t)\in
\mathbb{R}^{576}$; the autonomous loop holds the action chunk FIXED at $a^\*{=}0\in[-1,1]^{10}$ (LeWM has no policy
prior — pre-registered scope: this audits the fixed-action free-running loop, not the CEM planner):
$$ g(s) = (z_{t-1},\, z_t,\, \mathrm{predproj}(\mathrm{Pred}([z_{t-2},z_{t-1},z_t], E(a^\*)))[:, -1]). $$
Leading-$k$ Benettin via forward-mode JVP (a full 576-d Jacobian of a 6-layer transformer is wasteful; the audit needs
$\lambda_1$ + the leading band). Pre-registered readings (seed spec): expansive => certified $T_1(\epsilon)$ vs
measured divergence (encoder along the TRUE env rollout under the same chunks); contracting => abstain, classify
stable-vs-bias-driven by censoring. Pixel-scale ambiguity ([0,1] vs [0,255]) resolved EMPIRICALLY by a pre-registered
one-step sanity check (both reported). float64 CPU.

Run (smoke): STEP91_SMOKE=1 .venv/bin/python experiments/step91_lewm_audit.py
Writes: papers/figures/step91_lewm_audit.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP91_SMOKE", "0")))
CHUNK = 5            # env steps per model step (frameskip/action_block)
ADIM = 2             # raw env action dim
ZD = 192             # latent dim
HS = 3               # history window (num_frames)


def load_lewm():
    from hydra.utils import instantiate
    cfg = json.load(open(ROOT / "models/lewm/pusht-config.json"))
    model = instantiate(cfg)
    sd = torch.load(ROOT / "models/lewm/pusht-weights.pt", map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)                       # G0a: bit-faithful into the authors' own code
    model.eval().double()
    model.action_encoder.float()      # authors' Embedder.forward hard-casts input to float32; keep its weights f32 and
    for p in model.parameters():      # cast its (constant) output to f64 via zero_act_emb() — it is not in the Jacobian.
        p.requires_grad_(False)
    return model


def zero_act_emb(model):
    r"""Constant action embedding for the fixed zero chunk, computed in the authors' float32 Embedder and cast to f64."""
    with torch.no_grad():
        return model.action_encoder(torch.zeros(1, HS, CHUNK * ADIM)).double()


def env_rollout(seed: int, n_model_steps: int):
    r"""Roll the TRUE PushT env under the fixed zero action for ``n_model_steps`` chunks of CHUNK env steps; return the
    frame at每 chunk boundary as a float tensor (T+HS, C, H, W) in [0, 255] (scaling applied later), starting with the
    HS warmup frames needed for the first window."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/* envs)
    env = gym.make("swm/PushT-v1")
    obs, info = env.reset(seed=seed)
    frames = []

    def grab(o):
        px = o["pixels"] if isinstance(o, dict) and "pixels" in o else env.render()
        a = np.asarray(px)
        if a.ndim == 3 and a.shape[-1] == 3:
            a = a.transpose(2, 0, 1)                              # HWC -> CHW
        return torch.tensor(a.copy(), dtype=DTYPE)

    frames.append(grab(obs))
    a0 = np.zeros(ADIM, dtype=np.float32)
    for _ in range(n_model_steps + HS - 1):
        for _ in range(CHUNK):
            obs, *_ = env.step(a0)
        frames.append(grab(obs))
    env.close()
    return torch.stack(frames, 0)                                 # (n+HS, C, H, W), values 0..255


def encode_seq(model, frames: torch.Tensor, scale01: bool) -> torch.Tensor:
    px = frames / 255.0 if scale01 else frames
    with torch.no_grad():
        out = model.encoder(px, interpolate_pos_encoding=True)
        z = model.projector(out.last_hidden_state[:, 0])
    return z                                                      # (T, 192)


def make_autonomous(model):
    r"""$g:\mathbb{R}^{576}\to\mathbb{R}^{576}$, fixed zero action chunk (pre-registered $a^\*$)."""
    act_emb = zero_act_emb(model)                              # (1, HS, 192) — constant, f64

    def g(s: torch.Tensor) -> torch.Tensor:
        z = s.view(HS, ZD).unsqueeze(0)                           # (1, HS, 192)
        nxt = model.predict(z, act_emb)[:, -1]                    # (1, 192)
        return torch.cat([s[ZD:], nxt.squeeze(0)])                # shift ⊕ append

    return g


def benettin_jvp(g, s0: torch.Tensor, k: int, n_steps: int, warmup: int):
    r"""Leading-$k$ Lyapunov exponents (per MODEL step) of ``g`` from ``s0`` via forward-mode JVPs + QR."""
    from torch.func import jvp
    from torch.nn.attention import SDPBackend, sdpa_kernel
    d = s0.numel()
    torch.manual_seed(0)
    Q = torch.linalg.qr(torch.randn(d, k, dtype=DTYPE))[0]
    s = s0.clone()
    acc = torch.zeros(k, dtype=DTYPE)
    logs = []
    for t in range(n_steps + warmup):
        cols = []
        s_next = None
        with sdpa_kernel(SDPBackend.MATH):                       # flash-SDPA lacks forward-AD on CPU; math kernel has it
            for j in range(k):
                out, jv = jvp(g, (s,), (Q[:, j].contiguous(),))
                cols.append(jv)
                s_next = out
        Z = torch.stack(cols, 1)                                  # (d, k)
        Q, R = torch.linalg.qr(Z)
        diag = torch.abs(torch.diagonal(R)).clamp_min(1e-300)
        if t >= warmup:
            acc += torch.log(diag)
            logs.append(torch.log(diag))
        s = s_next.detach()
    lam = (acc / len(logs)).sort(descending=True).values
    L = torch.stack(logs)                                         # (n_steps, k)
    # moving-block bootstrap CI on lambda1's per-step log-stretch series
    g_ = torch.Generator().manual_seed(0)
    n, block = L.shape[0], max(10, n_steps // 12)
    tops = []
    for _ in range(200 if not SMOKE else 60):
        idx = torch.randint(0, n - block + 1, ((n // block) + 1,), generator=g_)
        sample = torch.cat([L[i:i + block] for i in idx])[:n]
        tops.append(float(sample.mean(0).max()))
    lo, hi = float(np.quantile(tops, 0.05)), float(np.quantile(tops, 0.95))
    return lam, (lo, hi)


def main() -> int:
    eps_list = [0.05, 0.1, 0.2, 0.5]   # 0.5 = exploratory column (bias here is large; labeled as such)
    n_starts = 3 if SMOKE else 8
    H_meas = 40 if SMOKE else 120
    k = 8 if SMOKE else 12
    qr_steps = 80 if SMOKE else 240
    qr_warm = 20 if SMOKE else 40

    print("[step91] loading official LeWM (PushT) ...", file=sys.stderr)
    model = load_lewm()

    # ---- G0b: pixel-scale sanity (pre-registered; both reported) ----
    fr = env_rollout(seed=0, n_model_steps=6)
    errs = {}
    for scale01 in (True, False):
        z = encode_seq(model, fr, scale01)
        win = z[:HS].unsqueeze(0)
        act_emb = zero_act_emb(model)
        with torch.no_grad():
            pred = model.predict(win, act_emb)[0, -1]
        errs[scale01] = float((pred - z[HS]).norm() / z[HS].norm().clamp_min(1e-12))
    scale01 = errs[True] <= errs[False]
    print(f"[step91] G0b pixel-scale: one-step rel-err [0,1]={errs[True]:.3f} vs [0,255]={errs[False]:.3f} "
          f"=> using {'[0,1]' if scale01 else '[0,255]'}", file=sys.stderr)

    # ---- certificate (autonomous fixed-action loop) ----
    z0 = encode_seq(model, fr, scale01)[:HS]
    s0 = z0.reshape(-1)
    print(f"[step91] certifying: leading-{k} Benettin (JVP), {qr_steps}+{qr_warm} steps on 576-d ...", file=sys.stderr)
    lam, (lo, hi) = benettin_jvp(make_autonomous(model), s0, k, qr_steps, qr_warm)
    l1 = float(lam[0])
    print(f"[step91] lambda1={l1:.4f} CI[{lo:.4f},{hi:.4f}]  leading band: {[round(float(x),3) for x in lam[:6]]}",
          file=sys.stderr)

    # ---- measured divergence (encoder along TRUE rollouts, same fixed chunks) ----
    rows = []
    horizons_all = {e: [] for e in eps_list}
    cens = {e: 0 for e in eps_list}
    for st in range(n_starts):
        fr = env_rollout(seed=100 + st, n_model_steps=H_meas)
        z_true = encode_seq(model, fr, scale01)                    # (H+HS, 192)
        act_emb = zero_act_emb(model)
        zwin = list(z_true[:HS])
        errs_t = []
        with torch.no_grad():
            for t in range(H_meas):
                nxt = model.predict(torch.stack(zwin, 0).unsqueeze(0), act_emb)[0, -1]
                zwin = zwin[1:] + [nxt]
                tgt = z_true[HS + t]
                errs_t.append(float((nxt - tgt).norm() / tgt.norm().clamp_min(1e-12)))
        for e in eps_list:
            h = next((i + 1 for i, v in enumerate(errs_t) if v > e), None)
            if h is None:
                cens[e] += 1
                h = H_meas
            horizons_all[e].append(h)

    for e in eps_list:
        med = float(np.median(horizons_all[e]))
        T1 = (float(np.log(1.0 / e) / l1) if l1 > 0 else None)
        ratio = (med / T1) if T1 else None
        rows.append({"eps": e, "T1_steps": T1, "measured_median": med, "n_censored": cens[e],
                     "ratio_measured_over_certified": ratio})
        print(f"[step91]   eps={e}: certified={'ABSTAIN(lam1<=0)' if T1 is None else f'{T1:.0f}'} "
              f"measured_med={med:.0f} (censored {cens[e]}/{n_starts}) "
              f"ratio={'—' if ratio is None else f'{ratio:.2f}'}", file=sys.stderr)

    out = {"model": "lewm-pusht (official ckpt, authors' code, strict load)", "lambda1": l1, "lambda1_ci": [lo, hi],
           "leading_band": [float(x) for x in lam], "pixel_scale01": bool(scale01), "one_step_errs": errs,
           "cert_rows": rows, "n_starts": n_starts, "H_meas": H_meas, "k": k,
           "scope": "fixed-action (a*=0) autonomous loop; LeWM has no policy prior (pre-registered)",
           "smoke": SMOKE}
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"step91_lewm_audit{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[step91] wrote step91_lewm_audit{tag}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
