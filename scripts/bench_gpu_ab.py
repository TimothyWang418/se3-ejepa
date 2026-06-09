r"""GPU A/B microbenchmark: WSL2-CUDA vs native-Windows-CUDA on the SAME RTX 3080.

Motivation: "Linux/WSL2 cuts 3080 performance" — test it on OUR actual workload profile instead of arguing.
Three workloads, chosen to span the regimes that matter for this repo:

  A. small-kernel WM training  — L96-style cyclic-conv (64ch, k=5, 4 layers, N=40), batch 512, K=5 rollout loss,
     Adam, float32, 200 iters. MANY tiny kernel launches per step — exactly where WSL2 launch overhead would show.
     This is the profile of step74/79/85/88 training and of ② Stage B's imagination loop.
  B. batched no-grad rollout   — batch 4096, 50 forward steps (CEM/forecast-style). Medium kernels.
  C. big-matmul control        — 4096x4096 matmul x60. Compute-bound; WSL2 ≈ Windows expected. If A/B differ but
     C doesn't, the gap is launch-overhead, not the GPU.

Self-contained (torch only — the Windows side needs just `pip install torch --index-url .../cu121`).
Prints one JSON line: {"platform", "torch", "device", "A_iters_per_s", "B_steps_per_s", "C_tflops"}.

Run:  python scripts/bench_gpu_ab.py
"""
import json
import platform
import time

import torch
import torch.nn as nn


def make_conv(N: int = 40, ch: int = 64, k: int = 5, layers: int = 4) -> nn.Module:
    mods = []
    chans = [1] + [ch] * (layers - 1) + [1]
    convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], k) for i in range(layers))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = convs
            self.k = k

        def forward(self, x):                                  # (B, N) -> (B, N), circular pad each layer
            h = x.unsqueeze(1)
            pad = (self.k - 1) // 2
            for i, c in enumerate(self.convs):
                h = torch.nn.functional.pad(h, (pad, pad), mode="circular")
                h = c(h)
                if i < len(self.convs) - 1:
                    h = torch.nn.functional.silu(h)
            return x + h.squeeze(1)

    return Net()


def bench_A(device: str, iters: int = 200, B: int = 512, N: int = 40, K: int = 5) -> float:
    torch.manual_seed(0)
    net = make_conv(N).float().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(B, N, device=device)
    tgt = torch.randn(B, K, N, device=device)
    # warmup
    for _ in range(10):
        z = x
        loss = 0.0
        for j in range(K):
            z = net(z)
            loss = loss + ((z - tgt[:, j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(iters):
        z = x
        loss = 0.0
        for j in range(K):
            z = net(z)
            loss = loss + ((z - tgt[:, j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize() if device == "cuda" else None
    return iters / (time.perf_counter() - t0)


def bench_B(device: str, B: int = 4096, N: int = 40, steps: int = 50, reps: int = 20) -> float:
    torch.manual_seed(0)
    net = make_conv(N).float().to(device).eval()
    x = torch.randn(B, N, device=device)
    with torch.no_grad():
        for _ in range(2):                                     # warmup
            z = x
            for _ in range(steps):
                z = net(z)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(reps):
            z = x
            for _ in range(steps):
                z = net(z)
        torch.cuda.synchronize() if device == "cuda" else None
    return reps * steps / (time.perf_counter() - t0)


def bench_C(device: str, n: int = 4096, reps: int = 60) -> float:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(5):
        (a @ b)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(reps):
        (a @ b)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.perf_counter() - t0
    return reps * 2 * n ** 3 / dt / 1e12                       # TFLOP/s


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"platform": platform.platform(), "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "A_train_iters_per_s": round(bench_A(dev), 2),
           "B_rollout_steps_per_s": round(bench_B(dev), 1),
           "C_matmul_tflops": round(bench_C(dev), 2)}
    print(json.dumps(out))
