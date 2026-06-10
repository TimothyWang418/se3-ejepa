r"""Step 93 — the certificate PRICES REPLAN CADENCE on the real TD-MPC2 agent (walker-walk, official checkpoint).

THE closing-the-loop experiment: E13 read the model; step93 lets the certificate make a DECISION on it. The agent
replans with the full (replicated) MPPI every k agent-steps; BETWEEN replans it acts open-loop on the world model's
imagined latent via the policy prior — i.e. the executed inter-replan loop is EXACTLY the prior loop
$g(z)=d(z,\tanh\mu_\pi(z))$ that the step89 certificate certifies ($T_1(0.2)\approx5.4$–$6.4$ model steps, calibrated
0.94–1.02). Pre-registered prediction: return holds near the cadence-1 anchor while $k\lesssim T_1(0.2)$ and degrades
beyond — the certificate names the largest safe cadence a priori. Pre-registered skip-replan convention (absent from
the official code): at a replan, encode the TRUE obs, run MPPI, execute its FIRST action; for the next $k{-}1$ steps
advance $\hat z\leftarrow d(\hat z,a)$ and execute $a=\tanh(\mu_\pi(\hat z))$ (deterministic prior mean). cadence=1
reproduces the official agent (fidelity anchor; official walker-walk eval band 977–983, trainer eval_mode=True
convention). Planner faithfulness per recon: MPPI horizon 3, 512 samples (24 pi trajs), 64 elites, 6 iters, temp 0.5,
std [0.05,2], discount 0.99, two-hot symexp reward/Q decode (101 bins, ±10), Q='avg' of 2-of-5 freshly subsampled,
Gumbel elite sampling (stochastic even in eval), warm-started mean / reset std. float32 (their dtype); fidelity is
DISTRIBUTIONAL (4 RNG draws per replan — bit-exactness impossible, stated).

Run (anchor):  STEP93_CADENCES=1 STEP93_EPISODES=2 .venv/bin/python experiments/step93_replan_cadence.py
Full sweep:    .venv/bin/python experiments/step93_replan_cadence.py
Writes: papers/figures/step93_replan_cadence.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step89_pretrained_wm_audit as s89  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
F32 = torch.float32

# planner constants (recon-verified, dmcontrol single-task 5M defaults)
HORIZON, NSAMP, NELITE, NPI, NITER = 3, 512, 64, 24, 6
TEMP, MINSTD, MAXSTD, DISCOUNT = 0.5, 0.05, 2.0, 0.99
BINS = torch.linspace(-10.0, 10.0, 101)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def two_hot_inv(logits):
    return symexp((torch.softmax(logits, -1) * BINS.to(logits.dtype)).sum(-1))


class Heads:
    r"""Reward head + unstacked 5-member Q ensemble from the OLD flat checkpoint format (recon-validated mapping:
    `_Qs.params.{j}` -> layer j//4, param ['weight','bias','ln.weight','ln.bias'][j%4]). float32, eval (dropout absent
    by construction — identity at eval anyway)."""

    def __init__(self, sd, latent=512, adim=6, mlp_dim=512, nq=5):
        self.reward = s89.mlp(latent + adim, [mlp_dim, mlp_dim], 101)
        self.reward.load_state_dict({k[len("_reward."):]: v for k, v in sd.items() if k.startswith("_reward.")},
                                    strict=True)
        names = ["weight", "bias", "ln.weight", "ln.bias"]
        self.qs = []
        for m in range(nq):
            q = s89.mlp(latent + adim, [mlp_dim, mlp_dim], 101)
            qsd = {}
            for j in range(10):
                layer, pname = j // 4, names[j % 4]
                qsd[f"{layer}.{pname}"] = sd[f"_Qs.params.{j}"][m]
            q.load_state_dict(qsd, strict=True)
            self.qs.append(q)
        for mdl in [self.reward] + self.qs:
            mdl.float().eval()
            for p in mdl.parameters():
                p.requires_grad_(False)

    def r(self, z, a):
        return two_hot_inv(self.reward(torch.cat([z, a], -1)))

    def q_avg(self, z, a, gen):
        idx = torch.randperm(5, generator=gen)[:2]
        vals = [two_hot_inv(self.qs[i](torch.cat([z, a], -1))) for i in idx]
        return (vals[0] + vals[1]) / 2.0


class Agent:
    r"""Faithful-MPPI TD-MPC2 agent with a CADENCE knob. eval_mode semantics (trainer convention): no exec noise;
    Gumbel elite draw kept (stochastic by design)."""

    def __init__(self, sl, heads, seed=0):
        self.sl, self.h = sl, heads
        self.gen = torch.Generator().manual_seed(seed)
        self.prev_mean = torch.zeros(HORIZON, 6)

    def _pi_sample(self, z):
        mu, log_std = self.sl.pi(z).chunk(2, -1)
        log_std = -10 + 0.5 * (2 - (-10)) * (torch.tanh(log_std) + 1)   # their log_std squash (min -10, max 2)
        eps = torch.randn(mu.shape, generator=self.gen, dtype=mu.dtype)
        return torch.tanh(mu + torch.exp(log_std) * eps)

    def _value(self, z0, actions):
        z = z0.expand(actions.shape[1], -1).clone()
        G, disc = torch.zeros(actions.shape[1]), 1.0
        for t in range(HORIZON):
            G = G + disc * self.h.r(z, actions[t])
            z = self.sl.next(z, actions[t])
            disc *= DISCOUNT
        G = G + disc * self.h.q_avg(z, self._pi_sample(z), self.gen)
        return torch.nan_to_num(G, 0.0)

    def plan(self, z, t0=False):
        z = z.float()
        pi_acts = torch.empty(HORIZON, NPI, 6)
        zp = z.expand(NPI, -1).clone()
        for t in range(HORIZON):
            pi_acts[t] = self._pi_sample(zp)
            zp = self.sl.next(zp, pi_acts[t])
        mean = torch.zeros(HORIZON, 6)
        if not t0:
            mean[:-1] = self.prev_mean[1:]
        std = torch.full((HORIZON, 6), MAXSTD)
        acts = torch.empty(HORIZON, NSAMP, 6)
        acts[:, :NPI] = pi_acts
        for _ in range(NITER):
            noise = torch.randn(HORIZON, NSAMP - NPI, 6, generator=self.gen)
            acts[:, NPI:] = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            v = self._value(z, acts)
            elite_idx = torch.topk(v, NELITE).indices
            ev, ea = v[elite_idx], acts[:, elite_idx]
            score = torch.exp(TEMP * (ev - ev.max()))
            score = score / (score.sum() + 1e-9)
            mean = (score.view(1, -1, 1) * ea).sum(1) / (score.sum() + 1e-9)
            std = ((score.view(1, -1, 1) * (ea - mean.unsqueeze(1)) ** 2).sum(1) / (score.sum() + 1e-9)).sqrt()
            std = std.clamp(MINSTD, MAXSTD)
        # Gumbel sample one elite ∝ score (their math.gumbel_softmax_sample; stochastic even in eval)
        gumb = -torch.log(-torch.log(torch.rand(NELITE, generator=self.gen).clamp_min(1e-9)))
        idx = int(torch.argmax(torch.log(score.clamp_min(1e-12)) + gumb))
        self.prev_mean = mean
        return ea[:, idx][0].clamp(-1, 1)                                # first action of the sampled elite


def episode(sl, heads, cadence: int, seed: int, T: int = 500) -> float:
    r"""One walker-walk episode (500 agent steps, action_repeat=2, rewards summed — official conventions). Replan
    every ``cadence`` steps; between replans act with the prior mean on the IMAGINED latent (the certified loop)."""
    from dm_control import suite
    env = suite.load("walker", "walk", task_kwargs={"random": seed})
    ts = env.reset()
    agent = Agent(sl, heads, seed=seed)
    total, zhat = 0.0, None
    for t in range(T):
        if t % cadence == 0:
            z = sl.encoder(s89._flat_obs(ts).float())
            a = agent.plan(z, t0=(t == 0))
            zhat = sl.next(z, a)                                        # imagined latent after the executed action
        else:
            a = torch.tanh(sl.pi_mean(zhat).squeeze(0) if zhat.ndim > 1 else sl.pi_mean(zhat))
            zhat = sl.next(zhat, a)
        a_np = a.detach().numpy()
        for _ in range(2):                                              # action_repeat=2, both rewards summed
            ts = env.step(a_np)
            total += float(ts.reward or 0.0)
    return total


def main() -> int:
    cadences = [int(x) for x in os.environ.get("STEP93_CADENCES", "1,2,4,6,8,12,24").split(",")]
    n_ep = int(os.environ.get("STEP93_EPISODES", "3"))
    ck = ROOT / "models/tdmpc2/walker-walk-1.pt"
    raw = torch.load(ck, map_location="cpu", weights_only=True)
    sd = raw["model"] if "model" in raw else raw
    sl = s89.load_tdmpc2_slices(raw, obs_dim=24, action_dim=6)
    # planner runs float32 (their dtype); slices were float64 -> cast copies for planning
    for m in (sl.encoder, sl.dynamics, sl.pi):
        m.float()
    heads = Heads(sd)
    out = {"cadences": {}, "official_band": [977.4, 983.1], "T1_eps02_band": [5.4, 6.4],
           "convention": "replan every k steps; between replans act tanh(mu_pi) on the imagined latent (the certified prior loop)"}
    for k in cadences:
        rets = [episode(sl, heads, k, seed=100 + i) for i in range(n_ep)]
        out["cadences"][str(k)] = {"returns": rets, "mean": float(np.mean(rets))}
        print(f"[step93] cadence={k}: returns={[round(r,1) for r in rets]} mean={np.mean(rets):.1f}", file=sys.stderr)
        (ROOT / "papers/figures/step93_replan_cadence.json").write_text(json.dumps(out, indent=2))
    print("[step93] wrote step93_replan_cadence.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
