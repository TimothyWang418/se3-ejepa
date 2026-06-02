r"""Step 51: **structure vs scale** — "scale buys interpolation; structure buys a certificate." (paper2 §1/P3)

paper2's tagline, made empirical and decisive. The single most likely reviewer question is *"why not just
scale a non-equivariant model?"* This experiment answers it on the group-orbit axis.

Setup (reuses the Step 50 SO(2) central-force world).  We restrict the **training** distribution to a wedge
of position-angles $|\angle r_t|\le\theta_{\max}$, then test next-state prediction around the **full circle**,
binned by $|\angle r_t|$. The group $\mathrm{SO}(2)$ acts by rotating the whole state; the dynamics commute
with it. For an **exactly equivariant** model,
$$ \mathrm{err}(R_\theta s)=\lVert \hat F(R_\theta s)-\Phi(R_\theta s)\rVert=\lVert R_\theta(\hat F(s)-\Phi(s))\rVert=\mathrm{err}(s)\quad\forall\theta, $$
so its error is **flat over the entire orbit** — a *certificate* that holds even though it trained only on the
wedge. A non-equivariant MLP has no such identity: it interpolates inside the wedge (and **scale helps there**)
but its error climbs outside, and **scaling width/data shifts the curve down without flattening the
out-of-wedge walls** — scale cannot buy the certificate.

What we show.
  - error vs orbit angle: equivariant = flat (certificate, $\theta$-independent by construction); MLP ladder =
    low in-wedge, climbing out-of-wedge.
  - the scale law: out-of-wedge error vs MLP parameter count plateaus **above** the equivariant floor — the
    certificate is unreachable by scale.

Honesty.  The MLP failing on unseen orbit regions is "just OOD" — exactly the point: the MLP has *no
certificate* for unseen group-orbit regions, while the equivariant model *provably* generalizes over the whole
orbit from wedge-only data. That provable orbit-wide flatness from partial data is the contribution; no amount
of MLP scale reproduces it.

Run (full ~2-4 min CPU):  .venv/bin/python experiments/step51_structure_vs_scale.py
Smoke:  STEP51_SMOKE=1 .venv/bin/python experiments/step51_structure_vs_scale.py
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from step50_noether_hinge import DS, DV, EquivWorldModel, make_orbits, rot2, train  # noqa: E402

torch.set_default_dtype(torch.float64)
SMOKE = bool(os.environ.get("STEP51_SMOKE"))
SEED = int(os.environ.get("STEP51_SEED", "0"))
THETA_MAX = math.radians(50.0)          # training wedge: |angle of pos| <= 50 degrees (~28% of the circle)


class MLPWorldModelW(nn.Module):
    r"""Width-parametrised non-equivariant baseline (same encode/predict/decode interface as Step 50's MLP,
    so `train(..., equiv=False)` drives it). Scaling `width` is the 'scale' knob."""

    def __init__(self, width: int, latent: int = DS + 2 * DV):
        super().__init__()
        h = width
        self.encs = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, latent))
        self.preds = nn.Sequential(nn.Linear(latent, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                   nn.Linear(h, latent))
        self.decs = nn.Sequential(nn.Linear(latent, h), nn.SiLU(), nn.Linear(h, 4))

    def encode(self, s):
        return self.encs(s)

    def predict(self, z):
        return self.preds(z)

    def decode(self, z):
        return self.decs(z)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def pos_angle(state: torch.Tensor) -> torch.Tensor:
    r"""Position angle $\angle r_t=\mathrm{atan2}(y,x)\in(-\pi,\pi]$."""
    return torch.atan2(state[:, 1], state[:, 0])


@torch.no_grad()
def relmse_by_angle(model, St, St1, equiv: bool, bins):
    r"""Next-state prediction relMSE, bucketed by $|\angle r_t|$ into `bins` (radian edges)."""
    if equiv:
        s, V = model.encode(St)
        sp, Vp = model.predict(s, V)
        pred = model.decode(sp, Vp)
    else:
        pred = model.decode(model.predict(model.encode(St)))
    err = ((pred - St1) ** 2).sum(1)
    tgt = (St1 ** 2).sum(1)
    a = pos_angle(St).abs()
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (a >= lo) & (a < hi)
        out.append(float(err[m].sum() / tgt[m].sum().clamp_min(1e-12)) if m.any() else float("nan"))
    return np.array(out)


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 92
    n_traj, steps, epochs = (300, 10, 12) if SMOKE else (1200, 24, 70)
    widths = [24, 96, 384] if not SMOKE else [24, 96]
    epochs = int(os.environ.get("STEP51_EPOCHS", epochs))
    print(line)
    print(f"STEP 51  structure vs scale: scale buys interpolation, structure buys the orbit certificate  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    # full-circle pool, then keep only the training wedge |angle(pos_t)| <= THETA_MAX
    S_all, S1_all = make_orbits(n_traj, steps, seed=SEED)
    wedge = pos_angle(S_all).abs() <= THETA_MAX
    S_tr, S1_tr = S_all[wedge], S1_all[wedge]
    St_te, St1_te = make_orbits(400, steps, seed=999)        # test = full circle
    bins = torch.linspace(0, math.pi, 10)                     # 9 bins over |angle| in [0, 180°]
    deg_centers = [math.degrees(float((bins[i] + bins[i + 1]) / 2)) for i in range(len(bins) - 1)]
    print(f"    train pairs in wedge (|angle|≤{math.degrees(THETA_MAX):.0f}°): {S_tr.shape[0]} of "
          f"{S_all.shape[0]}  |  test = full circle ({St_te.shape[0]} pairs)")

    curves, labels, results = [], [], {}

    # ---- equivariant (small) ----
    torch.manual_seed(SEED)
    eq = EquivWorldModel()
    train(eq, S_tr, S1_tr, epochs, equiv=True)
    eq_curve = relmse_by_angle(eq, St_te, St1_te, equiv=True, bins=bins)
    curves.append(eq_curve); labels.append(f"equivariant ({n_params(eq)/1e3:.1f}k)")
    results["equivariant"] = dict(params=n_params(eq), curve=eq_curve.tolist())
    print(f"\n  [equivariant {n_params(eq)/1e3:.1f}k params] relMSE by |angle|: "
          + "  ".join(f"{d:.0f}°:{v:.2e}" for d, v in zip(deg_centers, eq_curve)))

    # ---- MLP scale ladder (all on the SAME wedge data) ----
    for w in widths:
        torch.manual_seed(SEED)
        mlp = MLPWorldModelW(w)
        train(mlp, S_tr, S1_tr, epochs, equiv=False)
        c = relmse_by_angle(mlp, St_te, St1_te, equiv=False, bins=bins)
        curves.append(c); labels.append(f"MLP w={w} ({n_params(mlp)/1e3:.0f}k)")
        results[f"mlp_w{w}"] = dict(params=n_params(mlp), curve=c.tolist())
        print(f"  [MLP width={w}, {n_params(mlp)/1e3:.0f}k params] relMSE by |angle|: "
              + "  ".join(f"{d:.0f}°:{v:.2e}" for d, v in zip(deg_centers, c)))

    # ---- certificate / flatness metrics ----
    in_wedge = np.array(deg_centers) <= math.degrees(THETA_MAX)
    far = np.array(deg_centers) >= 120.0                      # well outside the wedge
    def io_ratio(c):                                          # out-of-wedge / in-wedge error ratio
        return float(np.nanmean(c[far]) / max(np.nanmean(c[in_wedge]), 1e-12))
    def in_err(c):
        return float(np.nanmean(np.array(c)[in_wedge]))
    eq_ratio = io_ratio(eq_curve)
    mlp_ratios = {w: io_ratio(np.array(results[f"mlp_w{w}"]["curve"])) for w in widths}
    eq_far = float(np.nanmean(eq_curve[far]))
    mlp_far = {w: float(np.nanmean(np.array(results[f"mlp_w{w}"]["curve"])[far])) for w in widths}
    mlp_in = {w: in_err(results[f"mlp_w{w}"]["curve"]) for w in widths}
    big, small = max(widths), min(widths)

    print(f"\n{line}\nSTEP 51 SUMMARY\n{line}")
    print(f"    flatness (out-of-wedge / in-wedge relMSE ratio; 1.0 = perfectly flat = certificate):")
    print(f"        equivariant: {eq_ratio:.2f}   <- flat over the WHOLE orbit from wedge-only training")
    for w in widths:
        print(f"        MLP w={w:<4d}: {mlp_ratios[w]:.2f}   (climbs out-of-wedge; scale doesn't flatten)")
    print(f"    in-wedge relMSE (scale should buy interpolation here):  "
          + "  ".join(f"MLP-w{w} {mlp_in[w]:.2e}" for w in widths)
          + f"   (w{small}→w{big}: {mlp_in[small]/max(mlp_in[big],1e-12):.0f}× better)")
    print(f"    out-of-wedge (≥120°) relMSE vs scale:  equiv {eq_far:.2e}  |  "
          + "  ".join(f"MLP-w{w} {mlp_far[w]:.2e}" for w in widths))
    print(f"    biggest MLP (w={big}, {results[f'mlp_w{big}']['params']/1e3:.0f}k params) is "
          f"{mlp_far[big]/max(eq_far,1e-12):.0f}× worse than the equivariant model out-of-wedge")

    # gate (honest): (i) equivariant flat over the orbit = certificate; (ii) scale BUYS in-wedge interpolation
    # (bigger MLP lower in-wedge error); (iii) the *capable* MLP still climbs out-of-wedge; (iv) NO MLP reaches
    # the equivariant floor out-of-wedge — scale can't buy the certificate.
    ok = (eq_ratio < 1.5
          and mlp_in[big] < 0.7 * mlp_in[small]
          and mlp_ratios[big] > 2.5
          and all(mlp_far[w] > 5 * eq_far for w in widths))
    if ok:
        print(f"\n    CONFIRMED: the exactly-equivariant model is FLAT over the entire SO(2) orbit "
              f"(ratio {eq_ratio:.2f}) from")
        print(f"        wedge-only training — a certificate. The MLP ladder interpolates in-wedge but every")
        print(f"        width climbs out-of-wedge (ratios {', '.join(f'{mlp_ratios[w]:.1f}' for w in widths)}) "
              f"and a {results[f'mlp_w{big}']['params']/1e3:.0f}k-param MLP is still {mlp_far[big]/max(eq_far,1e-12):.0f}× worse")
        print(f"        out-of-wedge. Scale buys interpolation; structure buys the certificate.")
    else:
        print(f"\n    INCONCLUSIVE: gate not met; reported as-is (no thresholds loosened).")

    results.update(passed=bool(ok), eq_flatness_ratio=eq_ratio, mlp_flatness_ratios=mlp_ratios,
                   eq_far=eq_far, mlp_far=mlp_far, mlp_in=mlp_in, theta_max_deg=math.degrees(THETA_MAX),
                   deg_centers=deg_centers, smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step51_structure_vs_scale_smoke" if SMOKE else "step51_structure_vs_scale"
    (fig_dir / f"{stem}.json").write_text(json.dumps(results, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))
    cmap = ["C2", "C1", "C3", "C4", "C5"]
    for c, lab, col in zip(curves, labels, cmap):
        ax[0].semilogy(deg_centers, c, "o-", color=col, label=lab, lw=1.8, ms=4)
    ax[0].axvspan(0, math.degrees(THETA_MAX), color="C0", alpha=0.10, label="training wedge")
    ax[0].set_xlabel("orbit position |∠ r| (degrees)"); ax[0].set_ylabel("next-state prediction relMSE")
    ax[0].set_title("error over the SO(2) orbit\n(equivariant = flat certificate; MLPs climb out-of-wedge)")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3, which="both")
    ps = [results[f"mlp_w{w}"]["params"] for w in widths]
    ax[1].loglog(ps, [mlp_far[w] for w in widths], "o-", color="C3", label="MLP (scale ladder)", ms=7)
    ax[1].axhline(eq_far, color="C2", ls="--", lw=2, label=f"equivariant floor ({eq_far:.1e})")
    ax[1].set_xlabel("MLP parameters"); ax[1].set_ylabel("out-of-wedge (≥120°) relMSE")
    ax[1].set_title("scale law: out-of-wedge error plateaus\nABOVE the equivariant floor — no certificate")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Step 51 — scale buys interpolation; structure buys the orbit certificate (paper2 §1)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
