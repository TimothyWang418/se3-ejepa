r"""Step 98 — the certificate reads **V-JEPA 2-AC** (Meta's 1B-encoder action-conditioned world model), training-free.

THIRD architecture family for the audit (after TD-MPC2 slices and LeWM): the official `vjepa2_ac_vit_giant`
(ViT-g/16 encoder, 1012M params + ~300M action-conditioned predictor; MIT; post-trained on DROID; loaded via
torch.hub from the authors' own code). The audited loop is EXACTLY the authors' energy-landscape rollout
(`notebooks/energy_landscape_example.ipynb`): per-frame token blocks $z\in\mathbb R^{256\times1408}$
($d=362{,}752$... 256 tokens $\times$ 1408 dims), one AR step
$$ g(z) = \mathrm{LN}\big(\mathrm{Pred}(z, a^{\ast}, s)\big)[:, -256{:}], $$
with the action held at the FIXED zero end-effector delta $a^{\ast}=0$ (V-JEPA 2-AC has no policy prior — the same
pre-registered fixed-action scope as LeWM/step91) and the pose $s$ advanced by the authors' `compute_new_pose`
(identity under $a^{\ast}=0$). Their planning "energy" is the L1 distance between predicted and encoded tokens —
the SAME quantity our monitor thresholds: the certificate prices the growth rate of V-JEPA 2-AC's own energy.

Pre-registered procedure (the CLASSIFICATION is registered, not the outcome — we do not know this cell's regime):
1. Leading-$k$ Benettin (torch.func.jvp, $k{=}6$, fp32 CUDA — fp64 is ~64x slower on consumer Ampere; precision
   disclosed, two independent $Q$ seeds must agree on sign and within 30% on $|\lambda_1|$ else INCONCLUSIVE).
2. Start states: the repo-shipped REAL Franka pair (`franka_example_traj.npz`, Meta's lab) and DROID episode frames.
3. Verdict by the published taxonomy: $\lambda_1$ CI $>0$ => expansive (issue $T_1(\epsilon)$); CI straddles/below
   $0$ => ABSTAIN, sub-classified stable-vs-bias by the measured one-step error and crossing censoring on real
   DROID frames (step99 supplies the measured side; this file emits the certificate).

Upstream goof, disclosed: vjepa2 main pins `VJEPA_BASE_URL = "http://localhost:8300"` (the public URL is commented
out); we monkeypatch the module constant to `https://dl.fbaipublicfiles.com/vjepa2` before loading.

Run (box, GPU):  ~/se3-ejepa/.venv/bin/python experiments/step98_vjepa2_audit.py
Writes: papers/figures/step98_vjepa2_audit.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SMOKE = bool(int(os.environ.get("STEP98_SMOKE", "0")))
HUB_DIR = Path.home() / ".cache/torch/hub/facebookresearch_vjepa2_main"
NPZ = Path(os.environ.get("STEP98_NPZ", str(Path.home() / "vjepa2_scout/franka_example_traj.npz")))
TOKENS_PER_FRAME = 256
EPS_LIST = [0.05, 0.1, 0.2, 0.5]


def load_model():
    r"""Hub load with the localhost-URL goof monkeypatched (upstream main, 2026-06)."""
    import torch.hub
    hub_dir = torch.hub.get_dir()
    sys.path.insert(0, str(Path(hub_dir) / "facebookresearch_vjepa2_main"))
    try:
        import src.hub.backbones as bb
        bb.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
    except Exception as e:
        print(f"[step98] backbones premonkeypatch skipped ({e})", file=sys.stderr)
    enc, pred = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", skip_validation=True)
    # forward-AD path: the authors wrap SDPA in a bare (deprecated) `torch.backends.cuda.sdp_kernel()` context that
    # re-enables efficient attention regardless of global flags; their module ALSO ships an explicit
    # softmax-attention branch gated on `use_sdpa` — flip it off on the predictor (authors' own math path,
    # forward-AD-safe by construction; encoder stays on fast SDPA, it runs under no_grad only).
    n_flip = 0
    for m in pred.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = False
            n_flip += 1
    print(f"[step98] use_sdpa=False on {n_flip} predictor attention modules", file=sys.stderr)
    return enc.cuda().eval().half(), pred.cuda().eval().float()


def encode_frame(enc, frame_hw3_uint8: np.ndarray) -> torch.Tensor:
    r"""Authors' per-frame recipe: transform -> repeat the frame into a 2-slot tubelet clip -> encoder -> (256,1408).
    Normalization follows the repo transform (resize/crop to 256, ImageNet-ish norm inside make_transforms)."""
    from app.vjepa_droid.transforms import make_transforms  # hub-repo import (notebook's own path)
    if not hasattr(encode_frame, "_tf"):
        encode_frame._tf = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                                           random_resize_scale=(1., 1.), reprob=0., auto_augment=False,
                                           motion_shift=False, crop_size=256)
    clip = encode_frame._tf(frame_hw3_uint8[None])                  # (T=1,H,W,C) -> (C,1,256,256)
    c = clip.unsqueeze(0).cuda().half()                             # (1,C,1,256,256)
    c = c.repeat(1, 1, 2, 1, 1)                                     # tubelet_size=2: repeat the frame
    with torch.no_grad():
        h = encode_frame._enc(c)                                    # (1, 256, 1408)
    return h[0].float()


def make_loop(pred, s0: torch.Tensor):
    r"""$g$ on flattened frame tokens (fp32), zero-delta action, pose frozen (= compute_new_pose under zero delta)."""
    import torch.nn.functional as F
    a = torch.zeros(1, 1, 7, device="cuda", dtype=torch.float32)
    s = s0.view(1, 1, 7).cuda().float()

    def g(zflat: torch.Tensor) -> torch.Tensor:
        z = zflat.view(1, TOKENS_PER_FRAME, 1408)
        out = pred(z, a, s)[:, -TOKENS_PER_FRAME:]
        out = F.layer_norm(out, (out.size(-1),))
        return out.reshape(-1)

    return g


def _math_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    r"""Explicit softmax attention — forward-AD-safe by construction. Needed because (i) the authors' bare
    `torch.backends.cuda.sdp_kernel()` context re-enables efficient attention over any global flag, and (ii) their
    explicit-attention fallback is bypassed whenever an attn_mask is passed (`if attn_mask is not None or use_sdpa`).
    dropout_p ignored (eval; Dropout-as-identity)."""
    sc = scale if scale is not None else q.size(-1) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * sc
    if is_causal:
        L, S = q.size(-2), k.size(-2)
        attn = attn.masked_fill(~torch.ones(L, S, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn + attn_mask
    return attn.softmax(-1) @ v


def benettin_jvp_gpu(g, z0: torch.Tensor, k: int, n_steps: int, warmup: int, q_seed: int):
    from torch.func import jvp
    import torch.nn.functional as TF
    TF.scaled_dot_product_attention = _math_sdpa                    # global; all encoding is done before this point
    d = z0.numel()
    gen = torch.Generator(device="cpu").manual_seed(q_seed)
    Q = torch.linalg.qr(torch.randn(d, k, generator=gen).float())[0].cuda()
    z = z0.clone().cuda()
    logs = []
    for t in range(n_steps + warmup):
        cols, z_next = [], None
        for j in range(k):
            out, jv = jvp(g, (z,), (Q[:, j].contiguous(),))
            cols.append(jv)
            z_next = out
        Z = torch.stack(cols, 1)
        Q, R = torch.linalg.qr(Z)
        diag = torch.abs(torch.diagonal(R)).clamp_min(1e-30)
        if t >= warmup:
            logs.append(torch.log(diag))
        z = z_next.detach()
    L = torch.stack(logs).cpu()
    lam = L.mean(0).sort(descending=True).values
    n, block = L.shape[0], max(8, n_steps // 10)
    gen2 = torch.Generator().manual_seed(0)
    tops = []
    for _ in range(200):
        idx = torch.randint(0, n - block + 1, ((n // block) + 1,), generator=gen2)
        sample = torch.cat([L[i:i + block] for i in idx])[:n]
        tops.append(float(sample.mean(0).max()))
    return [float(x) for x in lam], (float(np.quantile(tops, 0.05)), float(np.quantile(tops, 0.95)))


def main() -> int:
    t0 = time.time()
    print("[step98] loading V-JEPA 2-AC (ViT-g) ...", file=sys.stderr)
    enc, pred = load_model()
    encode_frame._enc = enc
    d = np.load(NPZ)
    frames, states = d["observations"][0], d["states"][0]           # (2,256,256,3), (2,7)
    print(f"[step98] loaded in {time.time()-t0:.0f}s | franka pair {frames.shape}", file=sys.stderr)

    z0 = encode_frame(enc, frames[0])
    import torch.nn.functional as F
    z0 = F.layer_norm(z0, (z0.size(-1),)).reshape(-1)               # normalize_reps convention
    # one-step sanity: predict frame1 tokens from frame0 under the TRUE pose diff action
    sys.path.insert(0, str(HUB_DIR / "notebooks"))
    try:
        from utils.mpc_utils import poses_to_diff  # type: ignore
        a_true = torch.tensor(poses_to_diff(states[0], states[1]), dtype=torch.float32).view(1, 1, 7).cuda()
    except Exception:
        diff = states[1] - states[0]
        a_true = torch.tensor(np.concatenate([diff[:3], np.zeros(4)]), dtype=torch.float32).view(1, 1, 7).cuda()
        print("[step98] poses_to_diff import failed; using raw xyz diff (disclosed)", file=sys.stderr)
    s0 = torch.tensor(states[0], dtype=torch.float32)
    z1 = F.layer_norm(encode_frame(enc, frames[1]), (1408,)).reshape(-1)
    with torch.no_grad():
        zp = pred(z0.view(1, TOKENS_PER_FRAME, 1408).cuda(), a_true, s0.view(1, 1, 7).cuda())[:, -TOKENS_PER_FRAME:]
        zp = F.layer_norm(zp, (zp.size(-1),)).reshape(-1).cpu()
    one_step = float((zp - z1.cpu()).norm() / z1.norm().clamp_min(1e-12))
    print(f"[step98] rel err to the npz GOAL frame: {one_step:.3f} (the pair is (current, goal) for the energy "
          f"demo, NOT consecutive frames — consecutive-frame one-step bias is measured on DROID in step99)",
          file=sys.stderr)

    g = make_loop(pred, s0)
    k = 4 if SMOKE else 6
    n_steps = 30 if SMOKE else 120
    warmup = 8 if SMOKE else 25
    runs = {}
    for q_seed in ([0] if SMOKE else [0, 1]):
        t1 = time.time()
        lam, (lo, hi) = benettin_jvp_gpu(g, z0, k, n_steps, warmup, q_seed)
        runs[q_seed] = {"lambda1": lam[0], "ci": [lo, hi], "band": lam}
        print(f"[step98] Q-seed {q_seed}: lambda1={lam[0]:.4f} CI[{lo:.4f},{hi:.4f}] "
              f"band={[round(x,3) for x in lam[:4]]} ({time.time()-t1:.0f}s)", file=sys.stderr)
    l1s = [runs[s]["lambda1"] for s in runs]
    stable = (len(l1s) == 1) or (np.sign(l1s[0]) == np.sign(l1s[1])
                                 and abs(l1s[0] - l1s[1]) / max(abs(l1s[0]), abs(l1s[1]), 1e-9) < 0.30)
    l1 = float(np.mean(l1s))
    lo = min(runs[s]["ci"][0] for s in runs)
    hi = max(runs[s]["ci"][1] for s in runs)
    expansive = l1 > 0 and lo > 0
    rows = [{"eps": e, "T1_steps": (float(np.log(1 / e) / l1) if expansive else None)} for e in EPS_LIST]
    verdict = ("EXPANSIVE — certified horizons issued" if expansive else
               "ABSTAIN — lambda1 CI straddles/below 0; sub-classification (stable vs bias-driven) from the "
               "measured side on real DROID frames (step99)")
    out = {"model": "vjepa2-ac-vit-giant (official hub ckpt, authors' code; fp32 predictor loop on CUDA)",
           "d_state": int(z0.numel()), "k": k, "n_steps": n_steps,
           "rel_err_to_goal_frame_NOT_one_step": one_step,
           "runs": {str(s): runs[s] for s in runs}, "q_seed_stable": bool(stable),
           "lambda1": l1, "lambda1_ci_envelope": [lo, hi], "cert_rows": rows, "verdict": verdict,
           "scope": "fixed zero-delta-action AR frame-token loop (authors' energy-landscape rollout); "
                    "fp32 CUDA precision disclosed; upstream localhost-URL goof monkeypatched",
           "smoke": SMOKE}
    figdir = ROOT / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step98_vjepa2_audit{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(out, indent=2))
    print(f"[step98] verdict: {verdict} | q-seed stable={stable}", file=sys.stderr)
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
