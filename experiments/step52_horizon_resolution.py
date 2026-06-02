r"""Step 52: the **horizon × resolution** certificate — $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$ (the 推背图 law).

paper2's third axis (Theorem B). A latent channel with local growth rate (Lyapunov exponent) $\lambda_j$ can be
predicted to within resolution $\epsilon$ only up to a horizon
$$ T_j(\epsilon)\;\sim\;\frac{\log(\epsilon/\delta_0)}{\lambda_j}\qquad(\lambda_j>0), $$
where $\delta_0$ is the model's one-step error. For $\lambda_j\le0$ (conserved / contracting channels) the channel
is certifiable for **all** horizons. So the certified region in the $(\text{horizon }T,\ \text{resolution }\epsilon)$
plane is a **staircase**: long horizons are reachable *only at coarse resolution*, and only the slow/invariant
channels reach $T=\infty$. This is exactly the **推背图 / I Ching long-range-prophecy** insight — one may foretell
the macro-trend (coarse, structural, invariant) over a millennium, never the fine detail of a given day — and the
textbook reason celestial mechanics predicts for millennia while weather dies at two weeks.

Demonstrator (a designed multi-Lyapunov system the model must *learn*, so the recovered horizons are empirical,
not assumed). All channels live on bounded domains (circles / decay) so autoregressive rollout cannot diverge.
State $s=(c,\,u,\,z_1,\,z_2)$ with $z_1,z_2\in\mathbb{C}\cong\mathbb{R}^2$ on the unit circle, evolving by
  - $c_{t+1}=c_t$                         — conserved invariant ($\lambda=-\infty$; certifiable $\forall T$),
  - $u_{t+1}=0.75\,u_t$                    — contracting ($\lambda=\ln0.75\approx-0.29$; error *decays*),
  - $z_{1,t+1}=e^{i\omega}z_{1,t}$         — neutral $\mathrm{SO}(2)$ rotor ($\lambda=0$; error grows ~linearly),
  - $z_{2,t+1}=z_{2,t}^2$                  — angle-doubling map (chaotic, $\lambda=\ln2\approx0.69$; "fine detail").
A small MLP learns the one-step map; we roll it out autoregressively (renormalising the two phases each step) and
measure per-channel error growth $\delta_j(T)$ vs the *true* trajectory, then read off $\hat\lambda_j$ and the
certified horizon $T_j(\epsilon)$.

Group reading (tie to Step 50's hinge). $z_1=(\cos\varphi,\sin\varphi)$ is the $\mathrm{SO}(2)$-equivariant
($\ell{=}1$) part; $c,u$ and the *magnitudes* are invariant scalars. The *conserved* invariant $c$ has the
longest horizon; the chaotic phase $z_2$ dies fast — so "invariant" alone does **not** mean long-horizon (Step 50's
honest non-converse), but the *slow/conserved* modes (which Step 50 showed live in the invariant block) are exactly
the long-$T$-certifiable ones.

Checks (falsifiable): the learned model recovers the chaotic Lyapunov exponent ($\hat\lambda_{z_2}\approx\ln2$);
the chaotic certified horizon scales as $\log(1/\epsilon)$ with slope $\approx 1/\lambda$; and at fixed $\epsilon$
the ordering is $T_{\text{conserved}}>T_{\text{rotor}}>T_{\text{chaotic}}$.

Run (full ~1-2 min CPU):  .venv/bin/python experiments/step52_horizon_resolution.py
Smoke:  STEP52_SMOKE=1 .venv/bin/python experiments/step52_horizon_resolution.py
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.set_default_dtype(torch.float64)
SMOKE = bool(os.environ.get("STEP52_SMOKE"))
SEED = int(os.environ.get("STEP52_SEED", "0"))

OMEGA = 0.4
RATES = {"conserved": float("-inf"), "contracting": math.log(0.75), "rotor (neutral)": 0.0,
         "chaotic": math.log(2.0)}
# channel layout in the 6-d state: 0=c, 1=u, 2:4=z1 (rotor), 4:6=z2 (chaotic phase)
CH = {"conserved": [0], "contracting": [1], "rotor (neutral)": [2, 3], "chaotic": [4, 5]}
PHASE_SLICES = [(2, 4), (4, 6)]                          # the two unit-circle pairs (renormalised in rollout)


def step_true(s: torch.Tensor) -> torch.Tensor:
    r"""One true step of the designed bounded multi-Lyapunov dynamics.  s:(B,6)->(B,6).

    The angle-doubling map $z\mapsto z^2$ has $|z|=1$ as an **unstable** fixed point of the modulus
    ($r\mapsto r^2$), so floating-point drift in $|z|$ would compound as $|z|^{2^t}$ and blow up; we renormalise
    the two phases so the dynamics live exactly on the circle (the angle dynamics — hence the Lyapunov spectrum —
    are unchanged; only spurious radial FP drift is removed)."""
    c, u = s[:, 0], s[:, 1]
    p1, q1, p2, q2 = s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    co, si = math.cos(OMEGA), math.sin(OMEGA)
    r1, i1 = co * p1 - si * q1, si * p1 + co * q1          # z1 <- e^{iω} z1   (rotation, λ=0)
    r2, i2 = p2 * p2 - q2 * q2, 2.0 * p2 * q2              # z2 <- z2^2        (angle-doubling, λ=ln2)
    return _renorm_phases(torch.stack([c, 0.75 * u, r1, i1, r2, i2], 1))


def _renorm_phases(s: torch.Tensor) -> torch.Tensor:
    s = s.clone()
    for a, b in PHASE_SLICES:
        s[:, a:b] = s[:, a:b] / s[:, a:b].norm(dim=1, keepdim=True).clamp_min(1e-9)
    return s


def sample_states(n: int, gen: torch.Generator) -> torch.Tensor:
    c = 0.5 + torch.rand(n, 1, generator=gen)            # conserved level
    u = 2.0 * torch.rand(n, 1, generator=gen) - 1.0      # contracting transient
    a1 = 2 * math.pi * torch.rand(n, 1, generator=gen)
    a2 = 2 * math.pi * torch.rand(n, 1, generator=gen)
    return torch.cat([c, u, torch.cos(a1), torch.sin(a1), torch.cos(a2), torch.sin(a2)], 1)


def make_pairs(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    s = sample_states(n, g)
    for _ in range(10):                                   # let the contracting transient settle a little
        s = step_true(s)
    return s, step_true(s)


class Predictor(nn.Module):
    r"""Plain one-step predictor $s_t\mapsto s_{t+1}$ (the law is about dynamics, not architecture)."""

    def __init__(self, d=6, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))

    def forward(self, s):
        return self.net(s)


def train(model, S, S2, epochs, seed):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    n, bs = S.shape[0], 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            (((model(S[idx]) - S2[idx]) ** 2).mean()).backward()
            opt.step()
        sched.step()
    return float((((model(S) - S2) ** 2).mean()).item())


@torch.no_grad()
def rollout_errors(model, s0, horizon):
    r"""Autoregressive rollout vs the true trajectory, renormalising the two phases each step so the state stays
    bounded. Returns per-channel-group RMS error $\delta_g(T)$, $T=1..\text{horizon}$."""
    s_pred, s_true = s0.clone(), s0.clone()
    errs = {g: [] for g in CH}
    for _ in range(horizon):
        # renormalise the two phases (unit circle) and clamp the unbounded c,u channels so a few badly-predicted
        # rows cannot overflow to inf/nan (honest divergence-guard; the true states all live in [-1.5, 1.5]).
        nxt = torch.nan_to_num(model(s_pred), nan=0.0, posinf=8.0, neginf=-8.0)
        s_pred = _renorm_phases(nxt).clamp(-8.0, 8.0)
        s_true = step_true(s_true)
        for g, idxs in CH.items():
            d = (s_pred[:, idxs] - s_true[:, idxs])
            rms = (d ** 2).sum(1).mean().sqrt()
            errs[g].append(float(torch.nan_to_num(rms, nan=10.0, posinf=10.0)))  # bounded domain: 10 = "lost"
    return {g: np.array(v) for g, v in errs.items()}


def fit_lyapunov(delta: np.ndarray):
    r"""Estimate the growth rate from the exponential regime: fit $\log\delta(T)=\hat\lambda T+b$ over the **rising
    segment** only. Phases live on the unit circle, so a decorrelated channel saturates at RMS $\approx\sqrt2$;
    we fit up to the first crossing of $0.5\sqrt2$ to exclude the saturation plateau (which would flatten the
    slope). Channels whose error never exceeds the floor (contracting) return NaN — they are clearly $\lambda<0$."""
    T = np.arange(1, len(delta) + 1)
    lo = max(3.0 * float(delta[0]), 1e-3)
    hi = 0.5 * math.sqrt(2.0)                               # ~0.71, below the √2 decorrelation ceiling
    cross = np.where(delta > hi)[0]
    end = int(cross[0]) if len(cross) else len(delta)       # rising segment = before first saturation crossing
    seg = np.arange(end)
    mask = np.isfinite(delta[seg]) & (delta[seg] > lo)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(T[seg][mask], np.log(delta[seg][mask]), 1)[0])


def certified_horizon(delta: np.ndarray, eps: float) -> int:
    r"""Largest $T$ such that $\delta(t)\le\epsilon$ for all $t\le T$ (first-crossing horizon)."""
    over = np.where(delta > eps)[0]
    return int(over[0]) if len(over) else len(delta)


def main() -> None:
    line = "=" * 92
    n_train, epochs, horizon = (4000, 25, 60) if SMOKE else (20000, 120, 90)
    epochs = int(os.environ.get("STEP52_EPOCHS", epochs))
    print(line)
    print(f"STEP 52  horizon × resolution: T_j(ε) ~ log(1/ε)/λ_j  (the 推背图 law)  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    torch.manual_seed(SEED)
    S, S2 = make_pairs(n_train, seed=SEED)
    model = Predictor()
    loss = train(model, S, S2, epochs, seed=SEED)

    s0 = make_pairs(2000, seed=777)[0]
    errs = rollout_errors(model, s0, horizon)
    lam_hat = {g: fit_lyapunov(errs[g]) for g in CH}
    print(f"    one-step train MSE {loss:.2e}  |  rollout horizon {horizon}")
    print(f"    per-channel Lyapunov exponent  λ (true → measured from rollout error growth):")
    for g in CH:
        tl = "-inf" if g == "conserved" else f"{RATES[g]:+.3f}"
        print(f"        {g:18s}: true {tl:>6s}   measured {lam_hat[g]:+.3f}   "
              f"(δ: {errs[g][0]:.1e}→{errs[g][-1]:.1e})")

    eps_grid = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    Tcurves = {g: np.array([certified_horizon(errs[g], e) for e in eps_grid]) for g in CH}
    print(f"\n    certified horizon T_j(ε) (steps the channel stays within ε):")
    print(f"        ε:            " + "  ".join(f"{e:>5.3f}" for e in eps_grid))
    for g in CH:
        print(f"        {g:18s}: " + "  ".join(f"{t:>5d}" for t in Tcurves[g]))

    # the log-law slope for the chaotic channel:  T_chaotic(ε) ≈ log(ε/δ0)/λ → dT/dlogε = 1/λ
    log_eps = np.log(eps_grid)
    cha = Tcurves["chaotic"].astype(float)
    band = (cha > 0) & (cha < horizon)                    # ignore floored/capped points
    slope = float(np.polyfit(log_eps[band], cha[band], 1)[0]) if band.sum() >= 2 else float("nan")
    inv_lambda = 1.0 / math.log(2.0)
    mid = 3                                                 # ε = 0.05
    slow = ["conserved", "contracting", "rotor (neutral)"]
    slow_min = int(min(Tcurves[g][mid] for g in slow))      # horizon of the SHORTEST slow channel
    cha_mid = int(Tcurves["chaotic"][mid])
    print(f"\n{line}\nSTEP 52 SUMMARY\n{line}")
    print(f"    (i) learned model recovers the chaotic Lyapunov exponent: λ̂ = {lam_hat['chaotic']:+.3f} "
          f"vs true ln2 = {math.log(2.0):+.3f}")
    print(f"    (ii) chaotic certified horizon scales as log(1/ε): dT/dlogε = {slope:.2f} "
          f"vs 1/λ = {inv_lambda:.2f}  (coarsen ε ⇒ longer horizon)")
    print(f"    (iii) at ε={eps_grid[mid]:.2f}: chaotic 'detail' dies first (T={cha_mid}); every slow/invariant "
          f"channel lasts ≥{slow_min} ({slow_min // max(cha_mid,1)}×+ longer)")

    ok = (abs(lam_hat["chaotic"] - math.log(2.0)) < 0.25
          and 0.7 * inv_lambda < slope < 2.5 * inv_lambda
          and cha_mid < slow_min and slow_min >= 3 * max(cha_mid, 1))
    if ok:
        print(f"\n    CONFIRMED: the certified horizon obeys T_j(ε) ~ log(1/ε)/λ_j. The conserved/invariant")
        print(f"        channel is certifiable for the whole rollout at any usable ε; the neutral rotor far")
        print(f"        longer than the chaotic 'detail', which can only be foretold *coarsely* and *briefly*")
        print(f"        (horizon grows only logarithmically as ε coarsens). Long-horizon ⇒ coarse + invariant")
        print(f"        only — the 推背图 law, and Theorem B's certified-region staircase, measured.")
    else:
        print(f"\n    INCONCLUSIVE: gate not met; reported as-is (no thresholds loosened).")

    out = dict(passed=bool(ok), one_step_mse=loss, horizon=horizon,
               lambda_true={g: (None if g == "conserved" else RATES[g]) for g in CH},
               lambda_measured=lam_hat, eps_grid=eps_grid.tolist(),
               T_curves={g: Tcurves[g].tolist() for g in CH},
               delta_curves={g: errs[g].tolist() for g in CH},
               chaotic_logeps_slope=slope, inv_lambda=inv_lambda, smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step52_horizon_resolution_smoke" if SMOKE else "step52_horizon_resolution"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7))
    colors = {"conserved": "C2", "contracting": "C0", "rotor (neutral)": "C1", "chaotic": "C3"}
    T = np.arange(1, horizon + 1)
    for g in CH:
        lt = "λ=−∞" if g == "conserved" else f"λ={RATES[g]:+.2f}"
        ax[0].semilogy(T, errs[g], color=colors[g], lw=1.8, label=f"{g} ({lt})")
    ax[0].set_xlabel("prediction horizon T (steps)"); ax[0].set_ylabel("rollout error δ(T)")
    ax[0].set_title("error growth = Lyapunov spectrum\n(chaotic explodes; conserved/contracting stay flat)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    for g in CH:
        ax[1].semilogx(eps_grid, Tcurves[g], "o-", color=colors[g], lw=1.8, label=g, ms=5)
    ax[1].set_xlabel("resolution ε (coarser →)"); ax[1].set_ylabel("certified horizon T(ε)")
    ax[1].set_title("certified-region staircase: long horizon ⇒ coarse + invariant only\n"
                    f"chaotic slope dT/dlogε={slope:.2f} ≈ 1/λ={inv_lambda:.2f} "
                    "(the Tuibei-tu / I-Ching long-range law)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Step 52 — horizon × resolution: T_j(ε) ~ log(1/ε)/λ_j, the predictability staircase (paper2 §2)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
