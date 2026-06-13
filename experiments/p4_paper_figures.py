r"""Paper figure drafts F3-F7 from banked artifacts (skeleton inventory). Pure plotting.

Outputs papers/figures/fig_{name}.pdf — DRAFTS for layout iteration, captions carry the
registered caveats (rank correlation, off-tune rung, censoring language).

Run: .venv/bin/python -u experiments/p4_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "papers" / "figures"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def fig3_cliff() -> None:
    r"""P(fail) vs N reliability cliff (fail = δ̂ > 6); rank-correlation caveat in caption."""
    lown = json.loads((F / "p4_lown_figure.json").read_text())["cells"]
    cliff = json.loads((F / "p4_plain_cliff.json").read_text())["cells"]
    exit_ = json.loads((F / "p4_cliff_exit.json").read_text())["cells"]
    pts: dict = {"plain": {}, "eq": {}}
    # pooled refs at 2606 (wedge v2 + plain_ctrl): 8/10 (ledgered)
    pts["plain"][2606] = (8, 10)
    pts["plain"][4000] = (1, 6)          # v1 pool (ledgered)
    pts["eq"][2606] = (0, 4)
    for src in (lown, cliff, exit_):
        for k, v in src.items():
            if not v.get("in_delta"):
                continue
            arm = "plain" if "plain" in k or k.startswith("c3") else "eq"
            n_str = k.split("_c")[-1].split("_r")[0] if "_c" in k else k[1:].split("_r")[0]
            try:
                n = int(n_str)
            except ValueError:
                continue
            f_, t_ = pts[arm].get(n, (0, 0))
            pts[arm][n] = (f_ + (v["in_delta"] > 6), t_ + 1)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for arm, col, mk in (("plain", "tab:red", "o"), ("eq", "tab:blue", "s")):
        ns = sorted(pts[arm])
        p = [pts[arm][n][0] / pts[arm][n][1] for n in ns]
        lo = [max(0, pi - 1.0 / np.sqrt(pts[arm][n][1])) for pi, n in zip(p, ns)]
        hi = [min(1, pi + 1.0 / np.sqrt(pts[arm][n][1])) for pi, n in zip(p, ns)]
        ax.fill_between(ns, lo, hi, color=col, alpha=0.12)
        ax.plot(ns, p, marker=mk, color=col, label=f"{arm} (n per point 3-10)")
    ax.set_xlabel("training transitions N")
    ax.set_ylabel(r"P(fail) = P($\hat\delta$ > 6)")
    ax.set_title("Reliability cliff: equivariance shifts the safe threshold ≥2× left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(F / "fig_F3_cliff.pdf")
    print("F3 saved")


def fig4_theta() -> None:
    full = json.loads((F / "p4_theta_probe_full.json").read_text())
    fair = json.loads((F / "p4_theta_fairness.json").read_text())
    fig, ax = plt.subplots(figsize=(4.0, 3))
    groups = [("eq (cand)", full["cand"]["theta_r2"], fair.get("cand", [])),
              ("eq (champ)", full["champ"]["theta_r2"], []),
              ("plain", full["plainc"]["theta_r2"], fair.get("plainc", []))]
    for i, (name, lin, mlp) in enumerate(groups):
        x = np.full(len(lin), i - 0.12)
        ax.scatter(x, lin, s=18, color="tab:blue", label="linear probe" if i == 0 else None)
        if mlp:
            ax.scatter(np.full(len(mlp), i + 0.12), mlp, s=18, marker="^",
                       color="tab:orange", label="MLP probe" if i == 0 else None)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel(r"block-$\theta$ probe $R^2$")
    ax.set_title("The group coordinate is equivariant-only\n(40k transitions; deficit deepens under MLP)")
    ax.legend(frameon=False, loc="center right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(F / "fig_F4_theta.pdf")
    print("F4 saved")


def fig5_ood() -> None:
    e03 = json.loads((F / "p4_e03b_boundaries.json").read_text())["rows"]
    wedge = json.loads((F / "p4_wedge_v2.json").read_text())["cells"]
    labels, vals = [], []
    for sname, row in e03.items():
        labels.append(f"shape:{sname}")
        vals.append(np.mean([v["guar_faithful"] for v in row.values()]))
    eqw = [c["out"]["guar"] for k, c in wedge.items() if k.startswith("eq_w2") and c.get("stable") and "out" in c]
    labels.append("pose:wedge→out")
    vals.append(np.mean(eqw))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.barh(range(len(labels)), vals, color="tab:green", alpha=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("faithful one-sided rate (within audited horizon)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Certificate-as-guarantee under OOD evaluation")
    fig.tight_layout()
    fig.savefig(F / "fig_F5_ood.pdf")
    print("F5 saved")


def fig6_gonogo() -> None:
    d2 = json.loads((F / "p4_campaign_d2.json").read_text())["pairs"]
    fig, ax = plt.subplots(figsize=(4.4, 3))
    rows = ("hstar", "1", "2", "4", "8", "zero", "rand")
    for rkey, v in d2.items():
        ax.plot(range(len(rows)), [v["rows"][r]["reach"] for r in rows],
                marker="o", lw=0.8, alpha=0.6, label=rkey)
    ax.set_xticks(range(len(rows)), ["$H^*$", "H=1", "H=2", "H=4", "H=8", "zero", "rand"])
    ax.set_ylabel("reach rate (motion-selected windows)")
    ax.set_title("The certificate said NO-GO ($H^* \\leq 1$) — and nothing beats random\n"
                 "(9 pairs; truth-replay reaches 10/10)")
    ax.set_ylim(-0.02, 0.4)
    fig.tight_layout()
    fig.savefig(F / "fig_F6_gonogo.pdf")
    print("F6 saved")


def fig7_ladder() -> None:
    d = json.loads((F / "p4_res_ladder.json").read_text())["rungs"]
    fig, ax = plt.subplots(figsize=(3.8, 3))
    for res, col in (("96", "tab:blue"), ("144", "tab:green"), ("192", "tab:red")):
        cells = [c for c in d[res] if c.get("delta") and c.get("eps_task")]
        ratios = [c["eps_task"] / c["delta"] for c in cells]
        ax.scatter([int(res)] * len(ratios), ratios, s=22, color=col)
    ax.axhline(1.0, ls="--", color="k", lw=0.8)
    ax.text(100, 1.03, "GO criterion", fontsize=8)
    ax.set_xlabel("input resolution (px)")
    ax.set_ylabel(r"$\epsilon_{task}/\hat\delta$ (within-rung, unit-safe)")
    ax.set_title("Resolution moves AWAY from GO\n(96px-tuned recipe at all rungs — caveat registered)")
    fig.tight_layout()
    fig.savefig(F / "fig_F7_ladder.pdf")
    print("F7 saved")


if __name__ == "__main__":
    for fn in (fig3_cliff, fig4_theta, fig5_ood, fig6_gonogo, fig7_ladder):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{fn.__name__} FAILED: {exc}")
    print("FIGURES DONE")
