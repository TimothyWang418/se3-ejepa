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
# TODO (wire + debug on the 3080): equivariant-vs-baseline certificate-as-task-win. Reuse the repo's
# SO(2)-equivariant encoder/predictor (src/models) and the G-equivariant CEM (Step 9/11). Add an
# SO(2)-equivariance unit test on the planar-obs action BEFORE trusting results (project rule).
# -------------------------------------------------------------------------------------------------
def build_wm(equivariant: bool, obs_dim: int, act_dim: int):
    raise NotImplementedError("wire the SO(2)-equivariant / baseline latent world model (src/models) on the GPU box")


def train_wm(model, demos):
    raise NotImplementedError("train the latent WM on the few-shot demo set")


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
