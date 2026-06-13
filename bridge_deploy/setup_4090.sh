#!/bin/bash
# RunPod setup for the bridge on a NON-Blackwell GPU (4090/3090/A100/L40) — the CLEAN path.
# Goal: the official tdmpc2 stack (torch 2.5.1 + tensordict/torchrl 0.6) where compile=true works.
# Use /root (not /workspace network volume). Run once per instance.
#
# EASIEST: launch the pod with Custom Image  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#          (torch is then already 2.5.1; this script skips the torch reinstall in that case).
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get -y update >/dev/null 2>&1
apt-get install -y --no-install-recommends libegl1 libgl1 libglew-dev libosmesa6 libgl1-mesa-dri ffmpeg git >/dev/null 2>&1
echo APT-DONE

# torch 2.5.1 (skip if the image already has it; harmless to pin). cu124 works on Ada/Ampere/Hopper.
python -c "import torch; assert torch.__version__.startswith('2.5'), torch.__version__" 2>/dev/null \
  || pip install --break-system-packages "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124

# the matched STABLE trio + tdmpc2 deps (NOT the 0.13 that the 5090/torch2.8 forced)
pip install --break-system-packages \
  "tensordict==0.6.0" "torchrl==0.6.0" \
  "numpy<2" "dm-control==1.0.16" "mujoco==3.1.2" "gymnasium==0.29.1" \
  "hydra-core==1.3.2" "hydra-submitit-launcher==1.2.0" "omegaconf==2.3.0" \
  termcolor tqdm pandas 2>&1 | tail -3
echo PIP-DONE

cd /root && rm -rf tdmpc2 && git clone -q --depth 1 https://github.com/nicklashansen/tdmpc2.git && echo CLONE-DONE
# then: copy bridge_deploy/eq_world_model.py into tdmpc2/tdmpc2/common/, apply the model=eq patch
# (PATCHES.md steps 1-3), and run with compile=true (it works on this stack).
MUJOCO_GL=egl python -c "import torch,tensordict,torchrl;from dm_control import suite;suite.load('walker','walk').reset();print('STACK OK torch',torch.__version__,'td',tensordict.__version__,'trl',torchrl.__version__)"
