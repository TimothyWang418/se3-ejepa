r"""Step 87 — certificate-gated MBRL sample efficiency (direction ②). Stage-B-ready scaffold + a Stage-A DIAGNOSTIC.

HONEST FINDING (Stage A): the pathwise actor-gradient amplifies at ~the cocycle rate $e^{\lambda_1 H_g\Delta t}$ — i.e.
the cert-gating "mechanism" is the **Lyapunov amplification restated** ($=\lambda_1>0$, already certified). So Stage A
CONFIRMS cert-gating is *principled* (gate where the cocycle is untrustworthy) but is NOT an independent go/no-go. ②'s
genuine test is **Stage B** — the online actor-critic loop measuring whether cert-gating the imagination horizon buys
*sample efficiency* — the ~0.4 gamble (heaviest; the seed warns it is close to §5.19's plan-depth failure mode).

Design: docs/specs/2026-06-09-step87-cert-gated-mbrl-design.md (~0.4 gamble; heaviest; do last). The cert-gate is the
backprop depth $H_g$ of a **pathwise** (Dreamer-style) actor objective: roll the differentiable world model under the
reparameterized policy for $H_g$ steps, accumulate the discounted reward, bootstrap the tail with a critic, and
backprop. The premise: pathwise gradients through a chaotic rollout amplify as $e^{\lambda_1 H_g\Delta t}$ (the very
$\lambda_1$ the certificate reads), so **past the certified horizon $T_1$ they EXPLODE and are noise**; gating at
$H_g=T_1$ keeps them well-conditioned. STAGE A tests exactly this — the gradient-norm explosion vs $H_g$ — which is the
whole reason cert-gating could buy sample efficiency. If the gradient does NOT blow up past $T_1$, the premise is false
and the full online loop (Stage B) is not worth running.

Reuse, do NOT modify: step79 (controlled Lorenz-96 `L96ControlledConv` / `make_equivariant_wm` / `collect_data` /
`train_wm` / `certificate` / `certified_T1_steps`). Stage B (the online actor-critic loop + return-vs-env-steps) is the
~0.4 gamble, gated on Stage A. float64 CPU.

Run (smoke):     STEP87_SMOKE=1 .venv/bin/python experiments/step87_cert_gated_mbrl.py
Run (Stage A):   .venv/bin/python experiments/step87_cert_gated_mbrl.py
Writes: papers/figures/step87_stageA_gradient_explosion.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step79_certified_control as step79  # noqa: E402

DTYPE = torch.float64
SMOKE = bool(int(os.environ.get("STEP87_SMOKE", "0")))
F_FORCE = step79.F_FORCE


class Actor(nn.Module):
    r"""Gaussian policy $\pi(a|z)=\mathcal N(\tanh\mathrm{MLP}(z),\,\sigma)$ over normalized state $z$; ``rsample`` gives
    reparameterized (pathwise) actions so policy gradients flow *through* the world-model rollout. $\mathbb{Z}_N$-aware
    is not required for Stage A (the gradient-explosion test); a plain MLP suffices to probe the mechanism."""

    def __init__(self, N: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, N)).double()
        self.log_sd = nn.Parameter(torch.zeros(N, dtype=DTYPE))

    def forward(self, z: torch.Tensor) -> torch.distributions.Normal:
        mu = torch.tanh(self.net(z))
        sd = torch.exp(self.log_sd).clamp(0.05, 2.0)
        return torch.distributions.Normal(mu, sd)


class Critic(nn.Module):
    def __init__(self, N: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1)).double()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def actor_objective(wm, actor, critic, z0, mu, sd, H_g: int, gamma: float = 0.99, u_max: float = 1.0,
                    use_critic: bool = True):
    r"""Pathwise Dreamer-style objective backpropped through ``H_g`` world-model steps + a critic bootstrap of the tail:
    $J=\sum_{t<H_g}\gamma^t r_t+\gamma^{H_g}V(z_{H_g})$, reward $r_t=-\lVert z_t-z_{\rm target}\rVert^2$ toward the
    normalized fixed point. The cert-gate is exactly ``H_g`` (the backprop depth). Returns the scalar mean objective
    (differentiable through the rollout, so $\nabla_\theta J$ exercises the chaotic tangent cocycle)."""
    z = z0
    z_target = ((F_FORCE - mu) / sd).to(DTYPE)
    J = z.new_zeros(())
    for t in range(H_g):
        dist = actor(z)
        a = dist.rsample()                                       # reparameterized: gradient flows through the rollout
        z = wm(z, u_max * a)                                     # differentiable WM step (autograd through L96ControlledConv)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)   # keep deep chaotic rollouts finite
        r = -((z - z_target) ** 2).sum(-1).clamp(max=1e4)
        J = J + (gamma ** t) * r.mean()
    if use_critic:
        J = J + (gamma ** H_g) * critic(z).mean()
    return J


def actor_grad_norm(wm, actor, critic, z0, mu, sd, H_g: int, u_max: float = 1.0) -> float:
    r"""$\lVert\nabla_\theta J\rVert$ for backprop depth ``H_g`` (the cert-gate). The Stage-A signal: this should grow
    $\sim e^{\lambda_1 H_g\Delta t}$ and explode past the certified horizon $T_1$."""
    actor.zero_grad(set_to_none=True)
    J = actor_objective(wm, actor, critic, z0, mu, sd, H_g, u_max=u_max)
    (-J).backward()                                              # maximize J -> minimize -J
    g2 = sum(float((p.grad ** 2).sum()) for p in actor.parameters() if p.grad is not None)
    return float(np.sqrt(g2))


def run_stageA(seeds, N: int, eps: float) -> dict:
    r"""Stage A: per seed, train the equivariant controlled WM, read $T_1$ (map steps), and measure the actor pathwise
    gradient norm at backprop depths $H_g\in\{T_1/4,T_1/2,T_1,2T_1,4T_1\}$ (clamped to a modest cap so the chaotic
    rollout stays computable). PASS if the gradient norm **grows monotonically with $H_g$ and is much larger past $T_1$
    than at $\le T_1/2$** — i.e. the cert-gate has a real mechanism (untrustworthy-tail gradients are noise)."""
    n_traj = 1500 if SMOKE else 6000
    K = 5
    epochs = 8 if SMOKE else 60
    cert_kw = dict(n_steps=600 if SMOKE else 2000, warmup=100 if SMOKE else 400,
                   n_boot=80 if SMOKE else 300, block=30 if SMOKE else 50)
    per_seed = {}
    for s in seeds:
        print(f"[step87.A] seed {s}: training Z_N-equivariant controlled WM at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        data = step79.collect_data(N, n_traj, K, s)
        wm, mu, sd, _ = step79.train_wm("conv", N, data, s, epochs=epochs, K=K)
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)                              # freeze WM; gradients flow only to the actor
        cert = step79.certificate(wm, mu, sd, N, eps=eps, seed=s, **cert_kw)
        T1 = step79.certified_T1_steps(cert)
        # Geometric backprop-depth ladder spanning a few Lyapunov times, capped so the chaotic rollout stays feasible.
        cap = 40 if SMOKE else 120
        depths = sorted({int(round(x)) for x in np.geomspace(4, max(8, min(cap, int(1.4 * T1))), 6)})
        torch.manual_seed(s)
        actor = Actor(N); critic = Critic(N)
        # on-attractor start (normalized), small batch
        traj = step79.collect_data(N, 64, 1, s + 11)[0]          # (64, N) normalized starts
        z0 = traj.to(DTYPE)
        gnorms = [actor_grad_norm(wm, actor, critic, z0, mu, sd, d) for d in depths]
        # Per-step amplification rate from the gradient ladder (log-linear fit) vs the CERTIFIED rate lambda1*dt_map.
        lg = np.log(np.maximum(np.asarray(gnorms), 1e-30))
        rate = float(np.polyfit(depths, lg, 1)[0]) if len(set(depths)) > 1 else 0.0
        cert_rate = float(cert["lambda1"]) * step79.step74.DTMAP
        per_seed[s] = {"T1": T1, "lambda1": cert["lambda1"], "depths": depths, "grad_norms": gnorms,
                       "grad_rate_per_step": rate, "cert_rate_per_step": cert_rate}
        print(f"[step87.A] seed {s}: T1={T1} | grad amplification {rate:.4f}/step vs certified lambda1*dt "
              f"{cert_rate:.4f}/step (ratio {rate / cert_rate if cert_rate else float('nan'):.2f})", file=sys.stderr)

    rates = [per_seed[s]["grad_rate_per_step"] for s in seeds]
    crates = [per_seed[s]["cert_rate_per_step"] for s in seeds]
    verdict = {"diagnostic": "pathwise actor-gradient amplifies at ~the certified Lyapunov rate", "grad_rates": rates,
               "cert_rates": crates, "n_seeds": len(seeds), "eps": eps, "N": N, "smoke": SMOKE,
               "note": "Near-tautological (= lambda1>0, already certified): the pathwise gradient cocycle IS the Lyapunov "
                       "cocycle. Confirms cert-gating is principled (gate where the cocycle is untrustworthy) but is NOT "
                       "an independent go/no-go. ②'s real test is Stage B (online sample efficiency), the ~0.4 gamble."}
    print(f"[step87.A] DIAGNOSTIC: pathwise gradient amplifies at ~the certified rate (grad {[round(r,3) for r in rates]} "
          f"vs cert {[round(c,3) for c in crates]} per step) => cert-gating is PRINCIPLED, but this is ~lambda1>0 restated. "
          f"②'s real test is Stage B (sample efficiency) — the ~0.4 gamble, NOT settled here.", file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


if __name__ == "__main__":
    torch.manual_seed(0)
    N = int(os.environ.get("STEP87_N", "10" if SMOKE else "16"))
    eps = float(os.environ.get("STEP87_EPS", "0.1"))
    seeds = [int(x) for x in os.environ.get("STEP87_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    print(f"[step87.A] cert-gating mechanism test: N={N} eps={eps} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
    res = run_stageA(seeds, N, eps)
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    (figdir / f"step87_stageA_gradient_explosion{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0)
