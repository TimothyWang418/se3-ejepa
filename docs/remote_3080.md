# Running the embodied benchmark on a Windows + RTX 3080 box from your MacBook

The equivariant **models** train fine on Apple MPS, but the 3D-manipulation **simulators**
(ManiSkill / SAPIEN, RLBench) are CUDA-only and effectively Linux-only. On a Windows machine the
clean path is **WSL2 (Ubuntu)** — CUDA passes through, and ManiSkill runs once Vulkan is wired.
This guide gets you from MacBook → WSL2 on the 3080, then `bash scripts/setup_cuda_box.sh` does the rest.

## 0. One-time, on the Windows box

1. **Update the NVIDIA driver** (GeForce Game Ready or Studio, recent). The Windows driver is what
   provides CUDA *and* Vulkan inside WSL2 — you do **not** install an NVIDIA driver inside WSL.
2. **Install WSL2 + Ubuntu 22.04** (PowerShell as admin):
   ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --update
   ```
3. **Enable systemd** in WSL (needed for Tailscale's daemon). In WSL, edit `/etc/wsl.conf`:
   ```ini
   [boot]
   systemd=true
   ```
   then from PowerShell: `wsl --shutdown` and reopen Ubuntu.

## 1. Connect the MacBook to the WSL2 box

**Recommended: Tailscale inside WSL2** (works across any network, gives the WSL box its own stable IP):
```bash
# inside WSL2 Ubuntu
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up          # log in via the printed URL
tailscale ip -4            # note this IP, e.g. 100.x.y.z
```
Install Tailscale on the MacBook too (same account). Then from the Mac:
```bash
ssh <user>@100.x.y.z       # direct into WSL2
```
First enable sshd inside WSL2: `sudo apt install -y openssh-server && sudo service ssh start`
(and `sudo systemctl enable ssh` if systemd is on).

**Fallback (same LAN, no Tailscale):** SSH to *Windows* (enable the Windows OpenSSH Server), then run
`wsl` to drop into Ubuntu — or set up a Windows→WSL port-proxy for port 22.

## 2. Dev workflow from the Mac

- **VS Code Remote-SSH** (easiest): connect to `<user>@<tailscale-ip>`; edit on the Mac, everything
  executes in WSL2. Or terminal + `tmux` (so long runs survive disconnects).
- **Code sync via git:** the repo is on GitHub. In WSL2: `git clone <repo>`; pull/commit there; the Mac
  `git pull`s results back. (Or `mutagen`/`rsync` for live file sync.)

## 3. Set up the environment (one command)

```bash
cd se3-ejepa
bash scripts/setup_cuda_box.sh        # uv venv + CUDA torch + ManiSkill + headless self-checks
```
It installs system deps (vulkan-tools, ffmpeg, build tools), `uv`, a Python 3.11 `.venv`, CUDA-enabled
torch, and ManiSkill; then self-checks `torch.cuda.is_available()`, `vulkaninfo`, and a ManiSkill env
reset. Override the CUDA wheel if your driver needs it: `TORCH_CUDA=cu118 bash scripts/setup_cuda_box.sh`.

## 4. Run

- Verify GPU + sim: `bash scripts/setup_cuda_box.sh` prints a green/red summary at the end.
- The embodied-certificate experiment scaffold is `experiments/step72_maniskill_certificate.py`
  (see its docstring for the design: equivariant latent JEPA + equivariant CEM vs a scaled
  non-equivariant baseline, sample-efficiency + cross-pose success). Start with its `--smoke` mode to
  validate the env end-to-end before the real run.

## Common WSL2 gotchas

- **`torch.cuda.is_available()` is False** → Windows NVIDIA driver too old, or you installed a CPU/non-CUDA
  torch wheel. Re-run with the right `TORCH_CUDA`.
- **Vulkan / SAPIEN can't find the GPU** → `vulkaninfo --summary` should list the NVIDIA GPU. If not,
  update the Windows driver; ensure `libvulkan1` + `vulkan-tools` are installed in WSL; check
  `ls /usr/lib/wsl/lib/` for `libcuda.so` (provided by the Windows driver).
- **ManiSkill headless render hangs** → set `export SAPIEN_HEADLESS=1` and use the GPU render path; no
  `DISPLAY` needed for the offscreen/GPU pipeline.
- **WSL2 sees little RAM/VRAM** → the 3080's 10 GB VRAM is fine for single-env ManiSkill; lower the
  parallel-env count if you OOM.
