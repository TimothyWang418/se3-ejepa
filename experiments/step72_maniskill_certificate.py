r"""Step 72 — the predictability certificate as a DOWNSTREAM TASK WIN on a real 3D manipulation benchmark (ManiSkill).

⚠️  SCAFFOLD — NOT YET RUN. This file was authored on a CUDA-less Mac; it CANNOT be tested here (ManiSkill/SAPIEN are
CUDA-only). It encodes the experiment *design* + a `--smoke` env validator; the model/planner/eval functions are
TODO stubs to wire and debug on the RTX 3080 box (see docs/remote_3080.md, scripts/setup_cuda_box.sh). Treat the first
GPU run as a debugging pass, not a result.

WHY THIS EXPERIMENT. Every embodied result so far (Experiments 9/11/12) shows the *configuration axis* — the certificate
makes the equivariant model orbit-FLAT (consistent across the orbit of scene poses) — and a closed-loop *flatness* on
PushT. What is still missing, and what would lift the project from "synthesis-level" to "flagship," is a **downstream
task win that scaling cannot buy**: on a real 3D manipulation benchmark, the equivariant latent world model should
generalize across object poses/orientations (举一反三) better than a *larger* non-equivariant baseline at the same data.

DESIGN.
  * Task: a ManiSkill task with a genuine spatial symmetry — e.g. PickCube-v1 / PushCube-v1, where rotating the scene
    about gravity (an SO(2) ⊂ SE(3) action) maps an optimal trajectory to the rotated optimal trajectory. (PickCube is
    the smoke default; PushCube is the contact-rich analogue of the project's PushT line.)
  * Models (both latent world models, trained on the same few-shot demo set):
      A) **equivariant**: SE(3)/SO(2)-equivariant encoder (reuse src/models equivariant stack) + jointly-equivariant
         latent predictor, planned with a **G-equivariant CEM** (reuse the Step-9/11 planner) so Theorem A's
         orbit-flatness carries to the closed loop.
      B) **baseline**: a capacity-MATCHED-or-larger ordinary (non-equivariant) latent WM + ordinary CEM. Fair-baseline
         discipline from Experiments 10/11: scale up the baseline until in-distribution parity, then test OOD poses.
  * Protocol: train on demos at a WEDGE of initial orientations; evaluate task success on (i) seen orientations and
    (ii) held-out OOD orientations (the orbit). Sweep demo count for a sample-efficiency curve.
  * The certificate-level claim: the equivariant model's success is ~flat seen→OOD (orbit-flatness becomes task
    competence), while the scaled baseline degrades OOD — a downstream win that data/scale do not close (Lemma 2 / §3.3).
  * Honest gates (report INCONCLUSIVE rather than loosen): (i) in-distribution parity (equivariant not worse than
    baseline on seen poses, within noise); (ii) OOD gap: equivariant seen→OOD success drop < baseline's by a clear
    margin, at matched-or-larger baseline capacity; (iii) ≥3 seeds.

Run (on the 3080, after scripts/setup_cuda_box.sh):
  smoke (validate env + wiring):  .venv/bin/python experiments/step72_maniskill_certificate.py --smoke
  full (after wiring the TODOs):   .venv/bin/python experiments/step72_maniskill_certificate.py --env PushCube-v1 --demos 100
"""

import argparse
import os
import sys

os.environ.setdefault("SAPIEN_HEADLESS", "1")


def smoke(env_id: str) -> int:
    r"""Validate the ManiSkill env + GPU render path end-to-end (the part most likely to need WSL2/Vulkan debugging).
    Does a reset + a few random steps + prints obs/action shapes. No models — just proves the sim runs headless."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the envs)
    import numpy as np

    print(f"[step72] smoke: making {env_id} (state obs, headless) ...", file=sys.stderr)
    env = gym.make(env_id, num_envs=1, obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="none")
    obs, info = env.reset(seed=0)
    print(f"[step72]   obs type {type(obs).__name__}; action_space {env.action_space}", file=sys.stderr)
    ret = 0.0
    for t in range(20):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        ret += float(np.asarray(rew).mean())
    env.close()
    print(f"[step72] smoke OK: 20 random steps ran headless; mean return {ret:.3f}.", file=sys.stderr)
    print("[step72] => env + GPU render path work. Next: wire build_wm/train/plan/eval below and run the full protocol.",
          file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------------
# TODO (wire + debug on the 3080): the full equivariant-vs-baseline certificate-as-task-win pipeline.
# Reuse the repo's equivariant encoder/predictor (src/models) and the G-equivariant CEM (Step 9/11).
# --------------------------------------------------------------------------------------------------
def build_wm(equivariant: bool, obs_dim: int, act_dim: int):
    raise NotImplementedError("wire the equivariant / baseline latent world model (src/models) on the GPU box")


def train_wm(model, demos):
    raise NotImplementedError("train the latent WM on the few-shot demo set")


def eval_cross_pose(model, env_id: str, seen_orientations, ood_orientations, seeds):
    raise NotImplementedError("closed-loop CEM eval: success rate on seen vs OOD scene orientations (举一反三)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="validate the ManiSkill env + GPU render path, then exit")
    ap.add_argument("--env", default="PickCube-v1")
    ap.add_argument("--demos", type=int, default=100)
    args = ap.parse_args()
    if args.smoke:
        sys.exit(smoke(args.env))
    print("[step72] full pipeline is a TODO scaffold — wire build_wm/train_wm/eval_cross_pose on the 3080 first; "
          "run with --smoke to validate the environment.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
