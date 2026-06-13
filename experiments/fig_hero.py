r"""Hero figure — the paper's one-look argument, three acts from committed result JSONs (no recompute):

  (a) FAITHFUL:  structure recovers the 40-D Lyapunov spectrum where a dense model's is garbage  (step74)
  (b) PRICED:    the faithful certificate meets a sensing budget an inflated one provably cannot (step85, Prop 9)
  (c) REAL:      the same read-out audits official TD-MPC2 checkpoints — calibrated where expansive,
                 correctly abstaining where contracting (step89, Prop 7)

Run:  .venv/bin/python experiments/fig_hero.py
Writes: papers/figures/hero_certified_world_models.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "papers" / "figures"

s74 = json.loads((FIG / "step74_lorenz96_spectrum.json").read_text())
s85 = json.loads((FIG / "step85_phase1_frontier.json").read_text())
for _f in sorted(FIG.glob("step85_phase1_frontier_seed*.json")):          # 2026-06-12: n=20 thickening
    s85["per_seed"].update(json.loads(_f.read_text())["per_seed"])
s89 = json.loads((FIG / "step89_pretrained_audit.json").read_text())

fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.6, 4.6))
fig.suptitle("Scale buys interpolation; structure buys a certified horizon — faithful (a), priced (b), real (c)",
             fontsize=12.5, y=1.02)

# ---------------- (a) FAITHFUL: spectrum recovery at N=40 ---------------- #
lt = np.array(s74["lambda_true"])
lc = np.array(s74["equivariant"]["lambda_learned"])
lm = np.array(s74["mlp"]["lambda_learned"])
lo, hi = float(min(lt.min(), lc.min(), lm.min())), float(max(lt.max(), lc.max(), lm.max()))
a.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$ (faithful)")
a.scatter(lt, lc, s=22, color="#1f77b4", zorder=3,
          label=f"$\\mathbb{{Z}}_N$-equivariant ($R^2={s74['equivariant']['spectrum_r2']:.2f}$)")
a.scatter(lt, lm, s=22, color="#d62728", marker="x", zorder=2,
          label=f"dense ($R^2={s74['mlp']['spectrum_r2']:.2f}$, $\\lambda_1$ inflated)")
a.axhline(0, color="gray", lw=0.5); a.axvline(0, color="gray", lw=0.5)
a.set_xlabel("true Lyapunov exponent $\\lambda_j$")
a.set_ylabel("learned-model exponent $\\hat\\lambda_j$")
a.set_title("(a) Structure recovers the 40-D spectrum")
a.legend(fontsize=8, loc="upper left")

# ---------------- (b) PRICED: the budget frontier + Prop 9 ---------------- #
seeds = sorted(s85["per_seed"].keys(), key=int)
grid = np.linspace(2, min(max(s85["per_seed"][s]["frontier"]["budgets"]) for s in seeds), 40)
def band(key, color, label, ls="-"):
    ys = np.stack([np.interp(grid, s85["per_seed"][s]["frontier"]["budgets"],
                             s85["per_seed"][s]["frontier"][key]) for s in seeds])
    b.plot(grid, ys.mean(0), color=color, ls=ls, lw=1.9, label=label)
    b.fill_between(grid, ys.min(0), ys.max(0), color=color, alpha=0.14)
band("conv_arm", "#1f77b4", "faithful certificate (a-priori)")
band("mlp_iso", "#d62728", "inflated certificate ($\\hat\\lambda_1\\!\\approx\\!3.4\\lambda_1$)")
band("adaptive", "#7f7f7f", "certificate-free adaptive", ls=":")
b.annotate("Prop. 9: a $c\\times$-inflated certificate\nneeds $c\\times$ the budget",
           xy=(0.42, 0.55), xycoords="axes fraction", fontsize=9, color="#d62728")
b.set_xlabel("sensing budget $B$ (re-observations)")
b.set_ylabel("aggregate $\\epsilon$-violation rate")
b.set_title("(b) Faithfulness is priced by a budget law")
b.legend(fontsize=8)

# ---------------- (c) REAL: TD-MPC2 audit ---------------- #
eps_list = [0.05, 0.1, 0.2]
walker = [f"walker-walk-{s}" for s in (1, 2, 3) if f"walker-walk-{s}" in s89]
cert_means, meas_means, ratios = [], [], []
for e in eps_list:
    cs, ms = [], []
    for k in walker:
        for r in s89[k]["cert_rows"]:
            if r["eps"] == e and r["T1_steps"]:
                cs.append(r["T1_steps"]); ms.append(r["measured_median"])
    cert_means.append(np.mean(cs)); meas_means.append(np.mean(ms))
    ratios.append(np.mean(ms) / np.mean(cs))
x = np.arange(len(eps_list)); w = 0.36
c.bar(x - w / 2, cert_means, w, color="#1f77b4", label="certified $T_1(\\epsilon)$ [model steps]")
c.bar(x + w / 2, meas_means, w, color="#ff7f0e", label="measured divergence (median)")
for i, r in enumerate(ratios):
    c.text(x[i], max(cert_means[i], meas_means[i]) + 0.3, f"ratio {r:.2f}", ha="center", fontsize=9,
           fontweight="bold" if abs(np.log(r)) < 0.1 else "normal")
c.set_xticks(x); c.set_xticklabels([f"$\\epsilon={e}$" for e in eps_list])
c.set_ylabel("horizon [model steps]")
acro_l1 = np.mean([s89[k]["lambda1"] for k in s89 if k.startswith("acrobot")])
c.annotate(f"TD-MPC2 walker-walk (3 official seeds):\ncalibrated at coarse $\\epsilon$ — ratio $\\to$ 1\n"
           f"acrobot-swingup: $\\lambda_1\\approx{acro_l1:.2f}<0$ →\ncertificate correctly ABSTAINS (Prop. 7)",
           xy=(0.03, 0.66), xycoords="axes fraction", fontsize=8.5)
c.set_title("(c) The same read-out audits a public pretrained WM")
c.legend(fontsize=8, loc="lower right")

fig.tight_layout()
out = FIG / "hero_certified_world_models.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
