# tdmpc2 patches for the step101 bridge (apply after cloning official tdmpc2)

Tested locally 2026-06-13: eq WorldModel constructs through full tdmpc2 machinery (5.2M params, Q-ensemble
TensorDict init intact) and G-EQ passes at MACHINE PRECISION (enc/dyn/reward/pi defects all 0.0) through
the real tdmpc2 layers. The training smoke belongs on Linux+CUDA (RunPod), not Mac (no CUDA/EGL).

## 1. Drop in the eq module
cp bridge_deploy/eq_world_model.py  <tdmpc2>/tdmpc2/common/eq_world_model.py

## 2. tdmpc2/tdmpc2/tdmpc2.py — model=eq flag (in TDMPC2.__init__, replace the WorldModel build):
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if getattr(cfg, 'model', 'dense') == 'eq':
        from common.eq_world_model import make_eq_world_model
        self.model = make_eq_world_model(WorldModel)(cfg).to(self.device)
    else:
        self.model = WorldModel(cfg).to(self.device)

## 3. tdmpc2/tdmpc2/config.yaml — add at end:
    model: dense

## 4. (CPU/MPS smoke only — NOT needed on CUDA) de-CUDA patches:
   train.py:        comment out `assert torch.cuda.is_available()`
   common/buffer.py: device/storage_device/mem_get_info -> guard on torch.cuda.is_available()
   (On RunPod CUDA these are unnecessary — leave stock.)

## 5. Dep pins discovered (Mac; on the RunPod pytorch/cuda image use environment.yaml):
   numpy<2  (dm_control 1.0.16 uses np.array(copy=False) — breaks on numpy 2.x)
   gymnasium==0.29.1, tensordict==0.6.2, torchrl==0.6.0, dm-control==1.0.16, mujoco==3.1.2, hydra-core==1.3.2
   Env GL: Linux headless -> MUJOCO_GL=egl ; Mac -> unset (state obs needs no renderer).

## 6. Launch (per instance, one arm×seed):
   MUJOCO_GL=egl python train.py task=walker-walk model=<eq|dense> seed=<s> steps=1_000_000 \
     exp_name=bridge_<arm>_s<s> eval_episodes=5 enable_wandb=false save_csv=true save_agent=true save_video=false

## 7. CRITICAL (2026-06-13, RTX 5090 / torch 2.8 / torchrl-tensordict 0.13):
   - Blackwell (5090) forces torch 2.8 -> torchrl/tensordict 0.13 (tdmpc2 was written for ~0.6).
     numpy<2 + --break-system-packages + hydra-submitit-launcher needed (PEP668 system python).
   - torch.compile x tensordict 0.13 crashes the update: "'TensorDict' has no attribute '_batch_size'"
     (dynamo tracing artifact). FIX: run with **compile=false** (eager). Trains end-to-end, ~40 steps/s
     on a 5090 eager (compiled would be ~2-3x faster but is broken on this stack).
   - On a 4090/3090 (Ada/Ampere) one could instead use torch 2.5.1 + the official pinned nightly
     tensordict/torchrl and keep compile=true (faster). 5090 = eager-only here.
