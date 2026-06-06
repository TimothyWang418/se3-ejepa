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
import sys


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


def train_wm(model, demos):
    raise NotImplementedError("Stage 2b: train the latent WM (JEPA) on the rollout dataset")


def eval_cross_pose(model, env_id, seen_orientations, ood_orientations, seeds):
    raise NotImplementedError("closed-loop CEM eval: is_success on seen vs OOD object/goal orientations (举一反三)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="validate the MuJoCo env (no Vulkan), then exit")
    ap.add_argument("--env", default="FetchPush-v4")
    ap.add_argument("--demos", type=int, default=100)
    args = ap.parse_args()
    if args.smoke:
        sys.exit(smoke(args.env))
    print("[step72] full pipeline is a TODO scaffold — wire build_wm/train_wm/eval_cross_pose on the 3080 first; "
          "run with --smoke to validate the environment.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
