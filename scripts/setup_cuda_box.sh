#!/usr/bin/env bash
# Set up the se3-ejepa CUDA environment for the embodied benchmark on a WSL2 (Ubuntu) box backed by
# an RTX 3080. Idempotent. Run from the repo root inside WSL2:  bash scripts/setup_cuda_box.sh
# Override the CUDA wheel if your Windows driver needs it:       TORCH_CUDA=cu118 bash scripts/setup_cuda_box.sh
#
# The Windows NVIDIA driver provides CUDA + Vulkan to WSL2; you do NOT install an NVIDIA driver inside WSL.
# See docs/remote_3080.md for the MacBook -> WSL2 connection (Tailscale + SSH) and gotchas.
set -uo pipefail
TORCH_CUDA="${TORCH_CUDA:-cu121}"
GREEN='\033[0;32m'; RED='\033[0;31m'; YEL='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[ok]${NC}   $1"; }
warn() { echo -e "  ${YEL}[warn]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }

echo "== se3-ejepa CUDA setup (WSL2 / RTX 3080), torch wheel: ${TORCH_CUDA} =="

# 0. sanity: are we in WSL2?
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then ok "running under WSL2"; else
  warn "not detected as WSL2 — continuing anyway (native Linux is fine too)"; fi

# 1. system deps (Vulkan for SAPIEN/ManiSkill, ffmpeg for video, build tools)
echo "-- apt system deps (sudo) --"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git curl ffmpeg \
  vulkan-tools libvulkan1 libglvnd0 libgl1 libegl1 libosmesa6 python3-dev \
  && ok "system deps installed" || fail "apt install failed"

# 2. uv (the project's package manager)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 && ok "uv $(uv --version 2>/dev/null | awk '{print $2}')" || fail "uv not on PATH (add ~/.local/bin)"

# 3. venv + CUDA torch + ManiSkill
echo "-- python env --"
uv venv --python 3.11 .venv && source .venv/bin/activate || { fail "venv create failed"; exit 1; }
uv pip install --quiet "torch" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
  && ok "torch (${TORCH_CUDA}) installed" || fail "torch install failed (try a different TORCH_CUDA)"
uv pip install --quiet mani-skill \
  && ok "mani-skill installed" || warn "mani-skill install failed — check the ManiSkill docs for your CUDA"
# project deps (equivariant stack); tolerate partial failures so the GPU checks still run
uv pip install --quiet numpy scipy matplotlib e3nn 2>/dev/null && ok "core scientific stack" || warn "some project deps failed"

echo "-- self-checks --"
# 4a. CUDA visible to torch?
python - <<'PY'
import torch, sys
print(f"    torch {torch.__version__}; cuda_available={torch.cuda.is_available()}; "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
sys.exit(0 if torch.cuda.is_available() else 3)
PY
[ $? -eq 0 ] && ok "torch sees the GPU" || fail "torch.cuda.is_available()=False — update the Windows NVIDIA driver / fix TORCH_CUDA"

# 4b. Vulkan sees the NVIDIA GPU? (SAPIEN/ManiSkill render path)
if command -v vulkaninfo >/dev/null 2>&1; then
  if vulkaninfo --summary 2>/dev/null | grep -qiE "nvidia|geforce|rtx"; then ok "Vulkan sees the NVIDIA GPU"; else
    warn "vulkaninfo did not list an NVIDIA GPU — ManiSkill rendering may fail (update the Windows driver)"; fi
else warn "vulkaninfo missing"; fi

# 4c. ManiSkill env reset smoke (offscreen)
SAPIEN_HEADLESS=1 python - <<'PY' 2>/dev/null && echo -e "  \033[0;32m[ok]\033[0m   ManiSkill env reset works" || echo -e "  \033[0;33m[warn]\033[0m ManiSkill smoke failed — see docs/remote_3080.md gotchas"
import gymnasium as gym, mani_skill.envs  # noqa
env = gym.make("PickCube-v1", num_envs=1, obs_mode="state", render_mode="none")
obs, _ = env.reset(seed=0); env.step(env.action_space.sample()); env.close()
PY

echo "== done. If all green: 'source .venv/bin/activate' then run experiments/step72_maniskill_certificate.py --smoke =="
