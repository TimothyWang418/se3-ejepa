# P4 3D lane bootstrap — WSL2 GPU box (EMP Tier 3.1)

*Target: user's Windows machine, Linux via WSL2, RTX 3080 10GB. Goal: ManiSkill3 +
VN-DGCNN SE(3)-JEPA on PegInsertionSide-v1 (W1 recon pick; LIBERO-spatial fallback).*

## ⚠️ SHARED-LANE PROTOCOL (2026-06-11 incident amendment — read before touching the box)

**Discovery (post-rsync, should have been pre):** this box is ALSO the paper2 research line's
CUDA lane — bash history shows step74/75/84 ran here (`.venv`, py3.11, torch cu130). The 3D-lane
rsync overlaid that checkout without checking first. Forensics found **zero damage** (ledgered
2026-06-11): research-line stashes `stash@{0,1}` intact; the "Step 77" dangling object is a
pre-amend duplicate of `eb1d7aa` which IS in main; untracked box-side artifact
`step85_phase1_frontier_seed10.json` untouched; `.venv` imports torch+CUDA fine. rsync had no
`--delete`, so nothing was removed. Rules from now on:

1. **Environments:** research line owns `.venv` (py3.11/cu130); 3D lane owns `.venv3d`
   (py3.12/cu124). Never install into the other's env.
2. **Git:** the box is a **pull-only leaf**. No commits or pushes from the box; 3D-lane artifacts
   rsync back to the Mac and are committed there. 3D-lane writes on the box live under
   `data/p4_3d/` + untracked logs only, so the research line's `git stash -u && git pull` flow
   keeps working unchanged.
3. **GPU scheduling:** before any >10-min GPU job, check `nvidia-smi` AND `~/GPU_LANE.lock`;
   write `owner pid ETA` into the lockfile, remove on completion. On contention the research
   line wins (prior tenant); the 3D lane falls back to the Mac.
4. **Namespaces:** all 3D-lane files use the `p4_3d_` prefix.

## WSL2 rendering boundary (measured 2026-06-11)

SAPIEN `RenderSystem` fails with `Failed to find a supported physical device "cuda:0"`: WSL
exposes Vulkan only via **dzn** (D3D12 wrapper — no NVIDIA UUID/CUDA-interop extensions) and
**llvmpipe** (CPU). Torch CUDA itself is fine (smoke 1 passed: 2.6.0+cu124, RTX 3080 visible).
Consequence: **render where Vulkan is native, train where CUDA is** — candidate split: ManiSkill
point-cloud collection on the Mac (SAPIEN 3 ships macOS arm64 wheels, MoltenVK) or llvmpipe
CPU-render on the box for small batches; VN-DGCNN training on the 3080 (pure tensor work, no
SAPIEN import in training scripts — keep the dependency split clean).

## Access plan (pick one, A recommended)

- **A. Tailscale (recommended, 10 min):** install Tailscale on the Windows host AND inside WSL
  (`curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`), plus on the Mac.
  Then `ssh <wsl-hostname>` from the Mac works from anywhere, NAT-proof. Enable sshd in WSL:
  `sudo apt install openssh-server && sudo service ssh start` (+ `sudo systemctl enable ssh` if
  systemd is on).
- B. LAN-only: WSL sshd + Windows portproxy
  (`netsh interface portproxy add v4tov4 listenport=2222 connectaddress=$(wsl hostname -I)`),
  Mac connects to `windows-ip:2222`. Breaks when WSL IP rotates; A is sturdier.
- C. No SSH: run a Claude session directly on that machine against a clone of this repo
  (territory note: 3D lane only; ledger merges via git).

## Bootstrap (one paste, inside WSL)

```bash
# 0. sanity: GPU visible in WSL
nvidia-smi || { echo "install the Windows NVIDIA driver with WSL support first"; exit 1; }

# 1. isolated env (do NOT touch the Mac repo's .venv pins)
sudo apt update && sudo apt install -y python3.11-venv git libvulkan1 vulkan-tools
git clone <repo-remote-or-rsync from Mac> ~/se3-ejepa && cd ~/se3-ejepa
python3.11 -m venv .venv3d && source .venv3d/bin/activate
pip install --upgrade pip

# 2. ManiSkill3 + torch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install mani_skill
# WSL Vulkan note: SAPIEN renders via Vulkan; on WSL2 install the NVIDIA Vulkan ICD —
# usually shipped with the Windows driver; verify:
vulkaninfo --summary | head -20 || echo "Vulkan missing: check /usr/share/vulkan/icd.d + driver"

# 3. smoke: env + demo download
python -m mani_skill.examples.demo_random_action -e PegInsertionSide-v1 --render-mode rgb_array -n 1
python -m mani_skill.utils.download_demo "PegInsertionSide-v1" -o ~/maniskill_demos

# 4. VN-DGCNN reference
git clone https://github.com/FlyingGiraffe/vnn ~/vnn
```

## First-session checklist (G0 gates, mirror of the PushT discipline)

1. G0a: env makes, point-cloud obs mode works (`obs_mode="pointcloud"`), demo replay loads;
   record obs/action spaces (need rotation dims in action for H1-original — controller
   `pd_ee_delta_pose`).
2. G0b: VN-DGCNN forward on a demo point cloud; **SO(3) equivariance unit test** (rotate cloud,
   check feature transform — repo rule, exact for VN linear layers).
3. G0c: chunked transition tensors from demos (the v1.2 stride convention transplanted).
4. Then: JEPA training smoke (the train_jepa recipe lessons transfer: #9 refresh flag N/A for
   VN — no e2cnn caches — but the EMA-pairing equality test gets a VN version BEFORE any audit).

## What transfers from the PushT lane (hard-won, do not relearn)

deployable pair = (EMA-target encoder, predictor); runs-not-seeds language (check CUDA
determinism flags); stability floor + content probes from day one; n=10 standard; health-vs-claim
two-stage separation; the EMP registers all of it.
