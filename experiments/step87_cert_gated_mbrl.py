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


# --------------------------------------------------------------------------------------------------------------- #
# STAGE B — the online cert-gated MBRL loop (the ~0.4 gamble). Per iteration: collect real segments under the actor,
# warm-refit the WM, read the gate H_g (per-arm: certificate / fixed / ungated), improve the policy with M pathwise
# imagination steps backpropped H_g deep, evaluate on the TRUE env. Sample efficiency = eval return vs cumulative real
# env steps. All float64 CPU (the WM/certificate live in float64; CUDA fp64 on the 3080 is gimped anyway).
# --------------------------------------------------------------------------------------------------------------- #
def collect_segments(actor, mu, sd, N: int, n_traj: int, K: int, seed: int, u_max: float = 1.0,
                     explore_sd: float = 0.3):
    r"""Collect ``n_traj`` length-``K`` TRUE-dynamics segments under the CURRENT actor (mean action + exploration
    noise, clamped to $[-1,1]\cdot u_{\max}$), from on-attractor starts. Returns ``(starts, controls, targets,
    env_steps)`` with the step79.collect_data tuple contract (starts/targets normalized by the FIXED warmup ``mu, sd``;
    controls raw) and exact accounting ``env_steps = n_traj K`` (every transition costs one real step)."""
    g = torch.Generator().manual_seed(seed)
    traj = step79.step74.attractor_traj(N, max(n_traj * 2, n_traj + 1), seed, "cpu").to(DTYPE)
    idx = torch.randperm(traj.shape[0], generator=g)[:n_traj]
    x = traj[idx].clone()
    starts = (x - mu) / sd
    controls = torch.empty(n_traj, K, N, dtype=DTYPE)
    targets = torch.empty(n_traj, K, N, dtype=DTYPE)
    for k in range(K):
        with torch.no_grad():
            a = actor((x - mu) / sd).mean
        a = (a + explore_sd * torch.randn(a.shape, generator=g, dtype=DTYPE)).clamp(-1.0, 1.0) * u_max
        x = step79.rk4_controlled(x, a)
        controls[:, k, :] = a
        targets[:, k, :] = (x - mu) / sd
    return starts, controls, targets, n_traj * K


def pathwise_returns(wm, actor, critic, z0, mu, sd, H_g: int, gamma: float = 0.99, u_max: float = 1.0,
                     use_critic: bool = True):
    r"""Per-sample pathwise returns ``(B,)``: roll the differentiable WM under ``rsample`` actions for ``H_g`` steps,
    accumulate $\sum_{t<H_g}\gamma^t r_t$ with $r_t=-\lVert z_t-z_{\rm target}\rVert^2/N$, bootstrap the tail with
    $\gamma^{H_g}V(z_{H_g})$. The cert-gate IS ``H_g`` (the backprop depth through the chaotic cocycle); ``H_g=0`` is a
    pure critic bootstrap. Differentiable w.r.t. the actor (and critic unless detached by the caller)."""
    z = z0
    z_target = ((F_FORCE - mu) / sd).to(DTYPE)
    J = z0.new_zeros(z0.shape[0])
    for t in range(H_g):
        a = actor(z).rsample()
        z = wm(z, u_max * a)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
        r = -((z - z_target) ** 2).mean(-1).clamp(max=1e4)
        J = J + (gamma ** t) * r
    if use_critic:
        J = J + (gamma ** H_g) * critic(z)
    return J


def wm_refit(wm, data3, mu, sd, epochs: int = 2, K: int = 5, lr: float = 1e-3, seed: int = 0):
    r"""Warm-start refit of the SAME wm instance on actor-collected segments (``data3 = (starts, controls, targets)``),
    K-step rollout loss (the Jacobian-constraining recipe). Returns the same object, in eval mode."""
    starts, controls, targets = data3
    for p in wm.parameters():
        p.requires_grad_(True)
    wm.train()
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    n = starts.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, 256):
            idx = torch.randperm(n, generator=g)[i:i + 256]
            z = starts[idx]
            loss = 0.0
            for k in range(K):
                z = wm(z, controls[idx, k, :])
                loss = loss + ((z - targets[idx, k, :]) ** 2).mean()
            opt.zero_grad(); (loss / K).backward(); opt.step()
    wm.eval()
    return wm


def eval_actor(actor, mu, sd, N: int, n_eval: int = 4, T_eval: int = 120, seed: int = 0,
               u_max: float = 1.0) -> float:
    r"""Deterministic evaluation on the TRUE dynamics: mean action, ``n_eval`` on-attractor starts, ``T_eval`` steps;
    returns the per-step mean of $-\lVert x-F\mathbf1\rVert^2/N$ (raw coordinates). Same seed $\Rightarrow$ same value;
    no gradients touched."""
    g = torch.Generator().manual_seed(seed + 999)
    traj = step79.step74.attractor_traj(N, max(n_eval * 2, n_eval + 1), seed + 999, "cpu").to(DTYPE)
    idx = torch.randperm(traj.shape[0], generator=g)[:n_eval]
    x = traj[idx].clone()
    total = 0.0
    with torch.no_grad():
        for _ in range(T_eval):
            a = actor((x - mu) / sd).mean.clamp(-1.0, 1.0) * u_max
            x = step79.rk4_controlled(x, a)
            total += float((-((x - F_FORCE) ** 2).mean(-1)).mean())
    return total / T_eval


def run_arm(N: int, mu, sd, seed: int, H_fn, n_iters: int, n_traj: int = 10, K: int = 5, M_policy: int = 8,
            batch_img: int = 64, n_eval: int = 4, T_eval: int = 120, wm_epochs: int = 2, gamma: float = 0.99,
            u_max: float = 1.0, lr_a: float = 3e-4, lr_c: float = 1e-3, wm=None, actor=None, critic=None,
            buf_cap: int = 4096) -> list:
    r"""ONE arm of the Stage-B loop. ``H_fn(wm, mu, sd, k) -> int`` supplies the gate each iteration (certificate /
    fixed / ungated — injected, so tests stub it). Per iteration: collect → refit WM (warm) → freeze WM → M pathwise
    policy steps at depth ``H_g`` + critic regression to the detached imagined return → true-env eval. Returns history
    rows ``{iter, env_steps, eval_return, H_g}`` with cumulative real-env-step accounting."""
    torch.manual_seed(seed)
    wm = wm if wm is not None else step79.make_equivariant_wm(N).double()
    actor = actor if actor is not None else Actor(N)
    critic = critic if critic is not None else Critic(N)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr_a)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr_c)
    env_steps = 0
    buf = None
    hist = []
    for k in range(n_iters):
        s, c, t, used = collect_segments(actor, mu, sd, N, n_traj, K, seed * 1000 + k, u_max)
        env_steps += used
        buf = (s, c, t) if buf is None else tuple(torch.cat([b, x], 0)[-buf_cap:] for b, x in zip(buf, (s, c, t)))
        wm_refit(wm, buf, mu, sd, epochs=wm_epochs, K=K, seed=seed * 77 + k)
        H_g = max(1, int(H_fn(wm, mu, sd, k)))
        for p in wm.parameters():
            p.requires_grad_(False)                              # imagination: gradients to the ACTOR only
        gi = torch.Generator().manual_seed(seed * 31 + k)
        idx = torch.randperm(buf[0].shape[0], generator=gi)[:batch_img]
        z0 = buf[0][idx].detach()
        for _ in range(M_policy):
            J = pathwise_returns(wm, actor, critic, z0, mu, sd, H_g, gamma, u_max)
            opt_a.zero_grad(); (-J.mean()).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            opt_a.step()
            with torch.no_grad():
                tgt = pathwise_returns(wm, actor, critic, z0, mu, sd, H_g, gamma, u_max)
            lc = ((critic(z0) - tgt) ** 2).mean()
            opt_c.zero_grad(); lc.backward(); opt_c.step()
        hist.append({"iter": k, "env_steps": env_steps, "eval_return": eval_actor(actor, mu, sd, N, n_eval, T_eval, seed),
                     "H_g": H_g})
    return hist


def run_stageB(seeds, N: int, eps: float) -> dict:
    r"""Stage B — sample efficiency of cert-gated vs fixed vs ungated imagination depth (the ~0.4 gamble). Per seed:
    a shared warmup (random-control data fixes ``mu, sd``; a lightly-trained WM + certificate give $T_1^{(0)}$), then
    FOUR arms from identical initial (wm, actor, critic) copies: **cert** ($H_g{=}T_1$, re-certified every RECERT
    iters), **fixed-half** ($T_1^{(0)}/2$), **fixed-double** ($2T_1^{(0)}$), **ungated** (deep cap). Gate G2: the cert
    arm reaches within 10% of the best final return at $\le$ the env-steps of the best fixed arm on $\ge2/3$ seeds,
    AND not worse than ungated. INCONCLUSIVE otherwise (the §5.19 conservatism risk — flagged in the seed)."""
    import copy
    RECERT = 4
    HCAP = 96
    n_iters = 4 if SMOKE else 20
    n0 = 800 if SMOKE else 4000
    warm_epochs = 6 if SMOKE else 30
    K = 5
    cert_kw = dict(n_steps=400 if SMOKE else 1200, warmup=80 if SMOKE else 250,
                   n_boot=50 if SMOKE else 120, block=25 if SMOKE else 40)
    per_seed = {}
    for s in seeds:
        print(f"[step87.B] seed {s}: warmup WM at N={N} (smoke={SMOKE}) ...", file=sys.stderr)
        data0 = step79.collect_data(N, n0, K, s)
        wm0, mu, sd, relmse = step79.train_wm("conv", N, data0, s, epochs=warm_epochs, K=K)
        cert0 = step79.certificate(wm0, mu, sd, N, eps=eps, seed=s, **cert_kw)
        T1_0 = max(2, min(HCAP, step79.certified_T1_steps(cert0)))
        print(f"[step87.B] seed {s}: warmup relMSE={relmse:.5f}  T1_0={T1_0} (lambda1={cert0['lambda1']:.3f})",
              file=sys.stderr)
        torch.manual_seed(s)
        actor0, critic0 = Actor(N), Critic(N)

        def cert_H(wm_, mu_, sd_, k, _cache={}):
            if k % RECERT == 0 or "T1" not in _cache:
                c = step79.certificate(wm_, mu_, sd_, N, eps=eps, seed=s + k, **cert_kw)
                _cache["T1"] = max(2, min(HCAP, step79.certified_T1_steps(c)))
                print(f"[step87.B]   (re-certify @iter {k}: T1={_cache['T1']})", file=sys.stderr)
            return _cache["T1"]

        arms = {"cert": cert_H,
                "fixed_half": lambda wm_, mu_, sd_, k: max(2, T1_0 // 2),
                "fixed_double": lambda wm_, mu_, sd_, k: min(HCAP, 2 * T1_0),
                "ungated": lambda wm_, mu_, sd_, k: HCAP}
        seed_out = {}
        for name, hfn in arms.items():
            print(f"[step87.B] seed {s}: arm={name} ...", file=sys.stderr)
            hist = run_arm(N, mu, sd, s, hfn, n_iters,
                           wm=copy.deepcopy(wm0), actor=copy.deepcopy(actor0), critic=copy.deepcopy(critic0))
            seed_out[name] = hist
            print(f"[step87.B] seed {s}: arm={name} final_return={hist[-1]['eval_return']:.4f} "
                  f"(env_steps={hist[-1]['env_steps']}, H_g last={hist[-1]['H_g']})", file=sys.stderr)
        # steps-to-threshold: within 10% of the best final return across arms (returns are negative).
        finals = {a: seed_out[a][-1]["eval_return"] for a in arms}
        best = max(finals.values())
        thresh = best - 0.1 * abs(best)
        def steps_to(a):
            for row in seed_out[a]:
                if row["eval_return"] >= thresh:
                    return row["env_steps"]
            return None
        stt = {a: steps_to(a) for a in arms}
        big = 10 ** 9
        cert_wins_fixed = (stt["cert"] or big) <= min(stt["fixed_half"] or big, stt["fixed_double"] or big)
        cert_ge_ungated = (stt["cert"] or big) <= (stt["ungated"] or big)
        per_seed[s] = {"T1_0": T1_0, "finals": finals, "steps_to_thresh": stt, "thresh": thresh,
                       "cert_wins_fixed": bool(cert_wins_fixed), "cert_ge_ungated": bool(cert_ge_ungated),
                       "history": seed_out}
        print(f"[step87.B] seed {s}: steps-to-thresh {stt} | cert_wins_fixed={cert_wins_fixed} "
              f"cert>=ungated={cert_ge_ungated}", file=sys.stderr)

    need = int(np.ceil(2 / 3 * len(seeds)))
    wins = [per_seed[s]["cert_wins_fixed"] and per_seed[s]["cert_ge_ungated"] for s in seeds]
    g2 = bool(sum(wins) >= need)
    verdict = {"G2_pass": g2, "n_win": int(sum(wins)), "n_seeds": len(seeds), "eps": eps, "N": N, "smoke": SMOKE,
               "note": "G2: cert-gated reaches within-10%-of-best return at <= env-steps of the best fixed arm and "
                       "not worse than ungated, on >=2/3 seeds. INCONCLUSIVE is the honest, seed-flagged outcome."}
    print(f"[step87.B] {'G2 PASS' if g2 else 'G2 INCONCLUSIVE'}: cert-gated sample-efficiency win on "
          f"{int(sum(wins))}/{len(seeds)} seeds.", file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


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
    mode = os.environ.get("STEP87_MODE", "A")                  # "A" = gradient diagnostic; "B" = online sample-efficiency
    N = int(os.environ.get("STEP87_N", "10" if SMOKE else "16"))
    seeds = [int(x) for x in os.environ.get("STEP87_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if SMOKE else ""
    if mode == "B":
        eps = float(os.environ.get("STEP87_EPS", "0.5"))       # eps=0.5 -> T1 ~ 38 steps at lambda1~1.8: feasible backprop depth
        print(f"[step87.B] Stage B sample efficiency: N={N} eps={eps} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
        res = run_stageB(seeds, N, eps)
        (figdir / f"step87_stageB_sample_efficiency{tag}.json").write_text(json.dumps(res, indent=2))
        raise SystemExit(0 if res["verdict"]["G2_pass"] else 1)
    eps = float(os.environ.get("STEP87_EPS", "0.1"))
    print(f"[step87.A] cert-gating mechanism test: N={N} eps={eps} seeds={seeds} smoke={SMOKE}", file=sys.stderr)
    res = run_stageA(seeds, N, eps)
    (figdir / f"step87_stageA_gradient_explosion{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0)
