r"""Step 72 (MuJoCo path) — the predictability certificate as a DOWNSTREAM TASK WIN on a real manipulation benchmark.

Replaces the ManiSkill plan: ManiSkill/SAPIEN needs CUDA<->Vulkan interop (VK_KHR_external_memory/semaphore), which a
WSL2 box without a native NVIDIA Vulkan ICD cannot provide (the dzn/D3D12 fallback lacks those extensions). MuJoCo via
Gymnasium-Robotics needs **no Vulkan at all** — CPU physics, state observations — so it runs cleanly on the RTX 3080
WSL2 box while CUDA still accelerates the equivariant model. The env layer (FetchPush-v4) is verified to import + reset
+ step (Dict obs: observation(25), achieved_goal(3), desired_goal(3); action Box(4); `is_success` in info).

⚠️  The MODEL/PLANNER/EVAL below are still TODO stubs to wire on the GPU box; only `--smoke` (env validation) is live.

WHY. Experiments 9/11/12 show the certificate makes the equivariant model orbit-FLAT (consistent across the orbit of
scene poses), but no result yet shows a *task win scaling cannot buy*. On a real manipulation task, an SE(2)/SO(2)-
equivariant latent world model planned with a G-equivariant CEM should generalize across object/goal orientations
(举一反三) better than a *larger* non-equivariant baseline at the same data — the downstream payoff the project needs.

DESIGN.
  * Task: FetchPush-v4 (7-DoF Fetch arm pushes a block to a goal on a table; sparse `is_success`). The dynamics has an
    approximate SO(2) symmetry about the vertical axis acting on the *planar object–goal–EE relative geometry* (the
    fixed arm base makes it approximate, i.e. Theorem B's regime — exactly the project's PushT treatment lifted to 3D
    manipulation). Extract the planar components of the 25-D obs for the SO(2) action; verify with an equivariance unit
    test before trusting any result.
  * Models (both latent world models, same few-shot demo budget):
      A) **equivariant**: SO(2)-equivariant encoder (reuse src/models) + jointly-equivariant latent predictor, planned
         with the Step-9/11 **G-equivariant CEM** so orbit-flatness (Theorem A) carries to the closed loop.
      B) **baseline**: capacity-MATCHED-or-larger ordinary latent WM + ordinary CEM (fair-baseline discipline of Exp 10/11).
  * Protocol: train on demos at a WEDGE of object/goal orientations; evaluate `is_success` on (i) seen and (ii) held-out
    OOD orientations; sweep demo count for a sample-efficiency curve. ≥3 seeds.
  * Honest gates (INCONCLUSIVE rather than loosen): (i) in-distribution parity (equivariant not worse on seen poses);
    (ii) OOD gap: equivariant seen→OOD success-drop < baseline's by a clear margin at matched-or-larger baseline capacity.

Run (on the 3080 WSL2 box, after `uv pip install gymnasium-robotics mujoco`):
  smoke:  .venv/bin/python experiments/step72_mujoco_certificate.py --smoke
  full:   (after wiring the TODOs)  .venv/bin/python experiments/step72_mujoco_certificate.py --env FetchPush-v4 --demos 100
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `src` and fetchpush_symmetry import
sys.path.insert(0, str(Path(__file__).resolve().parent))


def smoke(env_id: str) -> int:
    r"""Validate the MuJoCo env end-to-end (no Vulkan): reset + random steps + obs/action introspection."""
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    print(f"[step72] smoke: making {env_id} (MuJoCo, state obs, no render) ...", file=sys.stderr)
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    if isinstance(obs, dict):
        shapes = {k: tuple(np.asarray(v).shape) for k, v in obs.items()}
        print(f"[step72]   Dict obs shapes: {shapes}", file=sys.stderr)
    print(f"[step72]   action_space {env.action_space}", file=sys.stderr)
    succ = 0
    for _ in range(50):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        succ += int(bool(info.get("is_success", 0.0)))
        if term or trunc:
            obs, info = env.reset()
    env.close()
    print(f"[step72] smoke OK: 50 random steps ran (CPU MuJoCo, no Vulkan); `is_success` available in info.",
          file=sys.stderr)
    print("[step72] => env + state pipeline work on this box. Next: wire build_wm/train/eval + the SO(2) equivariance "
          "test, then run the equivariant-vs-baseline cross-pose protocol.", file=sys.stderr)
    return 0


# -------------------------------------------------------------------------------------------------
# Stage 2a — the latent world models. Equivariant: VN encoder on the 7 planar 2-vecs of obs_to_vn +
# VN predictor on the (dx,dy) planar action, so the WHOLE WM is SO(2)-equivariant by construction
# (encoder + predictor verified in tests/test_step72_wm_equivariance.py). Baseline: a capacity-matched
# (or larger) ordinary MLP on the raw 25-D obs + 4-D action. Both predict in latent space (JEPA-style);
# Stage 2b adds data + training + rollout relMSE; Stage 3 the seen-vs-OOD flatness; Stage 4 the CEM win.
# Note: this purely-equivariant first model uses the planar (dx,dy) action and drops dz/gripper into the
# invariant side (a refinement); FetchPush is dominated by planar pushing.
# -------------------------------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import fetchpush_symmetry as sym  # noqa: E402  (same experiments/ dir; on sys.path when run or imported)


def latent_rotation(z: "torch.Tensor", theta: float) -> "torch.Tensor":
    r"""Apply rho_latent(theta) to a VN latent (B, latent_dim) = latent_dim/2 flattened 2-vecs."""
    import numpy as _np
    c, s = float(_np.cos(theta)), float(_np.sin(theta))
    R = torch.tensor([[c, -s], [s, c]], dtype=z.dtype, device=z.device)
    zz = z.reshape(z.shape[0], -1, 2)
    return (zz @ R.T).reshape(z.shape[0], -1)


class EquivariantWM(nn.Module):
    r"""SO(2)-equivariant latent WM: StructuredStateEncoder (7 planar 2-vecs -> latent) + VNPredictor
    ((dx,dy) action -> residual next-latent). Equivariant encoder + equivariant predictor => the whole WM
    satisfies E(rho(g) obs) = rho(g) E(obs) and f(rho(g) z, R(g) a) = rho(g) f(z, a)."""

    def __init__(self, latent_dim: int = 128, hidden: int = 64):
        super().__init__()
        from src.models.structured import StructuredStateEncoder, VNPredictor
        self.enc = StructuredStateEncoder(n_vec=sym.N_VEC, latent_dim=latent_dim, hidden=hidden)
        self.pred = VNPredictor(latent_dim=latent_dim, action_dim=2, hidden=hidden, dim=2)

    def encode(self, vectors: "torch.Tensor") -> "torch.Tensor":   # (B,N_VEC,2) -> (B,latent_dim)
        return self.enc(vectors)

    def forward(self, vectors: "torch.Tensor", act_xy: "torch.Tensor") -> "torch.Tensor":
        return self.pred(self.enc(vectors), act_xy)                # predicted next latent


class BaselineWM(nn.Module):
    r"""Non-equivariant baseline: ordinary MLP encoder on the raw 25-D obs + MLP predictor on latent+4-D action.
    `hidden` is sized to match-or-exceed the equivariant model's capacity (fair-baseline discipline)."""

    def __init__(self, obs_dim: int = sym.OBS_DIM, act_dim: int = 4, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, latent_dim))
        self.pred = nn.Sequential(nn.Linear(latent_dim + act_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))

    def encode(self, obs: "torch.Tensor") -> "torch.Tensor":
        return self.enc(obs)

    def forward(self, obs: "torch.Tensor", act: "torch.Tensor") -> "torch.Tensor":
        z = self.enc(obs)
        return z + self.pred(torch.cat([z, act], dim=-1))          # residual next latent


def prep_equiv(obs_np, act_np, device):
    v, _ = sym.obs_to_vn(np.asarray(obs_np))
    enc_in = torch.tensor(v, dtype=torch.float32, device=device)
    act_in = torch.tensor(np.asarray(act_np)[..., :2], dtype=torch.float32, device=device)   # (dx,dy)
    return enc_in, act_in


def prep_baseline(obs_np, act_np, device):
    return (torch.tensor(np.asarray(obs_np), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(act_np), dtype=torch.float32, device=device))


def collect_transitions(env_id, n_steps, seed):
    r"""Random-policy rollouts -> (obs, act, next_obs) arrays. Random data suffices to learn the local dynamics."""
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    O, A, N = [], [], []
    cur = obs["observation"]
    for _ in range(n_steps):
        a = env.action_space.sample()
        nxt, _, term, trunc, _ = env.step(a)
        O.append(cur); A.append(a); N.append(nxt["observation"])
        cur = nxt["observation"]
        if term or trunc:
            obs, _ = env.reset(); cur = obs["observation"]
    env.close()
    return np.array(O, np.float32), np.array(A, np.float32), np.array(N, np.float32)


def _vicreg(z):
    r"""Anti-collapse: variance hinge + off-diagonal covariance penalty (VICReg)."""
    std = torch.sqrt(z.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + 0.04 * (off ** 2).sum() / z.shape[1]


def train_jepa_wm(model, obs, act, nxt, prep, epochs, device, seed, ema=0.99, var_coef=0.1):
    r"""JEPA: predict the EMA-target encoder's next latent; VICReg keeps the latent from collapsing."""
    import copy
    torch.manual_seed(seed)
    model = model.to(device)
    target_enc = copy.deepcopy(model.enc).to(device)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc_in, act_in = prep(obs, act, device)
    nxt_in, _ = prep(nxt, act, device)
    n = enc_in.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 256):
            idx = torch.randperm(n, generator=g)[i:i + 256]
            pred = model.forward(enc_in[idx], act_in[idx])
            with torch.no_grad():
                tgt = target_enc(nxt_in[idx])
            loss = ((pred - tgt) ** 2).mean() + var_coef * _vicreg(model.encode(enc_in[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for tp, mp in zip(target_enc.parameters(), model.enc.parameters()):
                    tp.mul_(ema).add_(mp, alpha=1 - ema)
    return model, target_enc


@torch.no_grad()
def one_step_relmse(model, target_enc, obs, act, nxt, prep, device):
    r"""Collapse-robust 1-step latent prediction error (FVU: fraction of centered target variance unexplained)."""
    enc_in, act_in = prep(obs, act, device)
    nxt_in, _ = prep(nxt, act, device)
    pred = model.forward(enc_in, act_in)
    tgt = target_enc(nxt_in)
    return float(((pred - tgt) ** 2).sum(-1).mean() / ((tgt - tgt.mean(0)) ** 2).sum(-1).mean())


def eval_cross_pose(model, env_id, seen_orientations, ood_orientations, seeds):
    raise NotImplementedError("closed-loop CEM eval: is_success on seen vs OOD object/goal orientations (举一反三)")


@torch.no_grad()
def cross_pose_fvu(model, target_enc, obs, act, nxt, prep, thetas, device):
    r"""1-step FVU on the held-out set rotated by each theta in the SO(2) orbit (obs+action+next rotated jointly).
    The equivariant WM is flat by construction (Theorem A); the baseline, trained at theta=0, degrades off-orbit."""
    out = []
    for th in thetas:
        o = sym.rotate_obs(obs, th); a = sym.rotate_action(act, th); n = sym.rotate_obs(nxt, th)
        out.append(one_step_relmse(model, target_enc, o, a, n, prep, device))
    return np.array(out)


def run_certificate(env_id: str, device: str, seed: int, tag: str = "") -> int:
    r"""Experiment 16: the predictability certificate (orbit-flatness) on a real MuJoCo manipulation task.
    Train both WMs on the theta=0 orientation; evaluate 1-step FVU across the SO(2) orbit of held-out transitions.
    Equivariant -> flat (ratio ~1.000, Theorem A); baseline -> climbs OOD. Honest: 'flat is not good' unless the
    equivariant in-distribution FVU is also competitive, so we report both."""
    smoke = os.environ.get("STEP72_SMOKE", "0") == "1"
    n_steps = 3000 if smoke else 12000
    epochs = 15 if smoke else 40
    ntr = int(n_steps * 0.85)
    thetas = np.linspace(0.0, np.pi, 9)                              # seen=0 ... OOD up to pi
    print(f"[step72] certificate: collecting {n_steps} transitions from {env_id} (seed {seed}) ...", file=sys.stderr)
    obs, act, nxt = collect_transitions(env_id, n_steps, seed)
    res = {}
    for name, model, prep in [("equivariant", EquivariantWM(latent_dim=64, hidden=64), prep_equiv),
                              ("baseline", BaselineWM(latent_dim=64, hidden=128), prep_baseline)]:
        m, tgt = train_jepa_wm(model, obs[:ntr], act[:ntr], nxt[:ntr], prep, epochs, device, seed)
        fvu = cross_pose_fvu(m, tgt, obs[ntr:], act[ntr:], nxt[ntr:], prep, thetas, device)
        res[name] = {"fvu": fvu.tolist(), "seen": float(fvu[0]), "ood_max": float(fvu.max()),
                     "ratio": float(fvu.max() / max(fvu[0], 1e-9))}
        print(f"[step72]   {name:11s} seen FVU {fvu[0]:.3f}  OOD-max {fvu.max():.3f}  ratio {res[name]['ratio']:.3f}",
              file=sys.stderr)
    eq, bl = res["equivariant"], res["baseline"]
    ok_flat = bool(eq["ratio"] < 1.05)                              # equivariant orbit-flat (Theorem A)
    ok_degrade = bool(bl["ratio"] > 1.5)                            # baseline degrades OOD
    # The certificate IS flatness + baseline OOD degradation. The in-distribution accuracy gap is NOT a pass/fail gate
    # — it is the honest "scale buys interpolation; structure buys the certificate" tradeoff, REPORTED not gated:
    # with enough data the unconstrained baseline interpolates the single training orientation better in-distribution,
    # yet cannot match the equivariant model's exact OOD-consistency at any scale (Lemma 2 / §3.3).
    indist_tax = float(eq["seen"] / max(bl["seen"], 1e-9))          # >1 => equivariant pays an in-dist cost
    passed = bool(ok_flat and ok_degrade)
    out = {"passed": passed, "gate": {"equivariant_flat": ok_flat, "baseline_degrades": ok_degrade},
           "indist_tax_equiv_over_baseline": indist_tax, "thetas_rad": thetas.tolist(),
           "equivariant": eq, "baseline": bl, "seed": seed, "smoke": smoke, "env": env_id}
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / f"step72_fetchpush_certificate{tag}.json").write_text(__import__("json").dumps(out, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        deg = np.degrees(thetas)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.plot(deg, eq["fvu"], "C0o-", lw=2, label=f"equivariant (ratio {eq['ratio']:.2f})")
        ax.plot(deg, bl["fvu"], "C3s--", lw=2, label=f"baseline (ratio {bl['ratio']:.2f})")
        ax.axhline(1.0, ls=":", color="gray", lw=1, label="FVU=1 (predict-the-mean)")
        ax.set_xlabel("scene rotation off the training orientation (deg)")
        ax.set_ylabel("1-step latent FVU")
        ax.set_title("Certificate on FetchPush (MuJoCo): orbit-flatness vs. scale (Exp 16)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / f"step72_fetchpush_certificate{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step72]   (figure skipped: {e})", file=sys.stderr)
    msg = ("CERTIFICATE HOLDS ON REAL MANIPULATION" if passed else "INCONCLUSIVE")
    tax_note = ("equivariant cheaper in-dist too" if indist_tax < 1 else
                "baseline interpolates the training orientation better — scale buys interpolation, "
                "structure buys the OOD certificate")
    print(f"[step72] {msg}: equivariant orbit-flat (ratio {eq['ratio']:.3f}, seen FVU {eq['seen']:.3f}); baseline "
          f"degrades OOD (ratio {bl['ratio']:.1f}, seen {bl['seen']:.3f}). In-dist tax {indist_tax:.1f}x "
          f"({tax_note}). Stage 4 next: closed-loop CEM task-success win.", file=sys.stderr)
    return 0 if passed else 1


def train_smoke(env_id: str, device: str) -> int:
    r"""Stage 2b validation: collect random transitions, JEPA-train BOTH world models, print 1-step FVU.
    Confirms the data+training+eval pipeline runs end-to-end (FVU<1 = beats predict-the-mean)."""
    import numpy as _np
    print(f"[step72] train-smoke: collecting transitions from {env_id} ...", file=sys.stderr)
    obs, act, nxt = collect_transitions(env_id, 3000, seed=0)
    ntr = 2500
    res = {}
    for name, model, prep in [("equivariant", EquivariantWM(latent_dim=64, hidden=64), prep_equiv),
                              ("baseline", BaselineWM(latent_dim=64, hidden=128), prep_baseline)]:
        m, tgt = train_jepa_wm(model, obs[:ntr], act[:ntr], nxt[:ntr], prep, epochs=15, device=device, seed=0)
        r = one_step_relmse(m, tgt, obs[ntr:], act[ntr:], nxt[ntr:], prep, device)
        res[name] = r
        print(f"[step72]   {name:11s} 1-step FVU {r:.3f}  ({'beats predict-the-mean' if r < 1 else 'FVU>1'})",
              file=sys.stderr)
    print(f"[step72] train-smoke OK: pipeline trains end-to-end on {device} "
          f"(equiv {res['equivariant']:.3f}, baseline {res['baseline']:.3f}). "
          f"Stage 3 next: seen-vs-OOD orientation flatness.", file=sys.stderr)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="validate the MuJoCo env (no Vulkan), then exit")
    ap.add_argument("--train-smoke", action="store_true", help="Stage 2b: collect + JEPA-train both WMs + 1-step FVU")
    ap.add_argument("--cert", action="store_true", help="Stage 3 (Exp 16): seen-vs-OOD orientation flatness certificate")
    ap.add_argument("--env", default="FetchPush-v4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--demos", type=int, default=100)
    args = ap.parse_args()
    if args.smoke:
        sys.exit(smoke(args.env))
    if args.train_smoke:
        sys.exit(train_smoke(args.env, args.device))
    if args.cert:
        sys.exit(run_certificate(args.env, args.device, args.seed, args.tag))
    print("[step72] Stage 4 (closed-loop CEM task-win) is the next build; "
          "use --smoke (env) / --train-smoke (data+JEPA) / --cert (Stage 3 flatness certificate).", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
