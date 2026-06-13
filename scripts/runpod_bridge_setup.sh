#!/usr/bin/env bash
# RunPod setup for the step101 bridge (online TD-MPC2, eq vs dense, walker-walk).
# Target: a RunPod "PyTorch 2.x / CUDA 12.x" template instance. Run ONCE per instance, on /root (NOT
# the /workspace MooseFS volume — it breaks pip/rsync temp-renames; see B300 post-mortem).
#
#   bash runpod_bridge_setup.sh            # installs deps + verifies headless MuJoCo + smoke-imports
#
# Deps mirror external/tdmpc2/docker/environment.yaml. MUJOCO_GL=egl gives headless rendering with the
# instance's CUDA driver (no Vulkan/SAPIEN needed — TD-MPC2 uses MuJoCo, not SAPIEN, so the cloud-render
# trap that bit paper3's 3D lane does not apply here).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
echo "[setup] apt deps (headless GL for MuJoCo)..."
apt-get -y update >/dev/null
apt-get install -y --no-install-recommends \
  build-essential git ffmpeg unzip wget curl \
  libglew-dev libgl1-mesa-glx libosmesa6-dev libegl1 patchelf >/dev/null

echo "[setup] python deps (tdmpc2 environment.yaml)..."
pip install -q --upgrade pip
# torch is preinstalled on the PyTorch template; do NOT reinstall it. tensordict/torchrl must match it.
pip install -q \
  "dm-control==1.0.16" "gymnasium==0.29.1" "mujoco==3.1.2" "hydra-core==1.3.2" "omegaconf==2.3.0" \
  "numpy==1.24.4" "tensordict==0.6.2" "torchrl==0.6.0" "kornia==0.7.2" "termcolor==2.4.0" \
  "tqdm==4.66.4" "pandas==2.0.3" "imageio==2.34.1" "imageio-ffmpeg==0.4.9" "h5py==3.11.0" \
  "hydra-submitit-launcher==1.2.0" 2>&1 | tail -2
# NOTE: if tensordict/torchrl error against the instance's torch, pin to the nightly pair from
# environment.yaml (tensordict-nightly==2025.1.1 torchrl-nightly==2025.1.1) on a torch 2.5 image.

echo "[setup] headless MuJoCo smoke..."
export MUJOCO_GL=egl
python - <<'PY'
import os; os.environ["MUJOCO_GL"]="egl"
from dm_control import suite
env = suite.load("walker", "walk")
ts = env.reset()
import numpy as np
obs = np.concatenate([v.ravel() for v in ts.observation.values()])
print(f"[setup] walker-walk OK, obs dim {obs.shape[0]} (expect 24)")
import torch, tensordict  # noqa
print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()} tensordict OK")
PY
echo "[setup] DONE — instance ready. Set MUJOCO_GL=egl in the run env."
