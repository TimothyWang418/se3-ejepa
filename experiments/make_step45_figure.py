r"""Step 45 figure: exact vs augmentation-approximate equivariance, in the closed loop.

One image, two panels, read straight from
``papers/figures/step45_augmented_mlp_closed_loop.json`` (a plotting step over an already-computed
3-seed run; no model is re-trained here). Three models, all through the *same* SE(3)-equivariant
paired closed-loop planner on a pure-rotation orbit:

* **(a) Closed-loop OOD/seen orientation ratio.** The downstream metric. The **exact VN** sits on the
  invariance line ($\times1.000$); full-$\mathrm{SO}(3)$ **augmentation** narrows the un-augmented
  MLP's $\times1.401$ gap to $\times1.071$ but its pooled CI still **excludes 1** (sign $p=0.02$) — it
  does *not* close the loop; the **un-augmented MLP** degrades hardest.
* **(b) Composed equivariance residual $\Delta_{\mathrm{eq}}$ (log).** The mechanism. Augmentation's
  residual ($\approx11$) is *no better* than the un-augmented MLP's ($\approx4.4$) and
  $\sim\!10^{6}\times$ the VN's weight-independent float floor ($\sim\!8\times10^{-6}$): full-SO(3)
  data never makes a plain-MLP latent transform as $\rho(R)$. Approximate-by-coverage is not
  exact-by-construction — and only exactness closes the loop (panel a).

Run:
    .venv/bin/python experiments/make_step45_figure.py
    # -> papers/figures/step45_augmented_vs_exact.png  (and .pdf)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "papers" / "figures"

# Palette: equivariant/exact = green (hero), augmentation = orange (approximate),
# non-equivariant baseline = purple (consistent with make_figures.py's green/purple).
C_VN = "#1b7837"    # dark green  = exact (architectural)
C_AUG = "#e08214"   # orange      = augmentation (approximate-by-coverage)
C_MLP = "#762a83"   # purple      = no prior
C_REF = "#888888"   # grey reference line
_ORDER = ["VN (exact)", "MLP+aug", "MLP (no aug)"]
_COLORS = {"VN (exact)": C_VN, "MLP+aug": C_AUG, "MLP (no aug)": C_MLP}
_PRETTY = {"VN (exact)": "VN\n(exact)", "MLP+aug": "MLP + aug\n(full SO(3))",
           "MLP (no aug)": "MLP\n(no prior)"}


def _yerr(agg: dict, key: str) -> np.ndarray:
    r"""Asymmetric error bars from each model's pooled bootstrap ratio CI about the per-seed mean."""
    los, his = [], []
    for k in _ORDER:
        mean = agg[k]["ratio"][0]
        _, lo, hi = agg[k]["pooled_ratio_ci"]
        los.append(max(0.0, mean - lo))
        his.append(max(0.0, hi - mean))
    return np.array([los, his])


def panel_ratio(ax: plt.Axes, agg: dict) -> None:
    r"""Closed-loop OOD/seen orientation ratio: VN on the invariance line, aug above it, MLP highest."""
    x = np.arange(len(_ORDER))
    means = [agg[k]["ratio"][0] for k in _ORDER]
    ax.bar(x, means, 0.62, color=[_COLORS[k] for k in _ORDER],
           yerr=_yerr(agg, "ratio"), capsize=5, ecolor="#333333")
    for xi, k in zip(x, _ORDER):
        m = agg[k]["ratio"][0]
        p = agg[k]["sign"][2]
        tag = f"×{m:.3f}" if k == "VN (exact)" else f"×{m:.3f}\n(sign p={p:.1g})"
        ax.text(xi, agg[k]["pooled_ratio_ci"][2] + 0.012, tag, ha="center", va="bottom", fontsize=8.5)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1, zorder=0)
    ax.text(len(x) - 0.5, 1.004, "invariant / flat (×1)", color=C_REF, fontsize=7,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([_PRETTY[k] for k in _ORDER], fontsize=8.5)
    ax.set_ylabel("closed-loop  OOD / seen  orientation-error ratio")
    ax.set_ylim(0.95, 1.50)
    ax.set_title("(a) Only the exact model closes the loop\n"
                 "(3 seeds, 288 paired tasks, pure-rotation orbit)", fontsize=10)


def panel_eq(ax: plt.Axes, agg: dict) -> None:
    r"""Composed equivariance residual (log): augmentation never approaches the VN's float floor."""
    x = np.arange(len(_ORDER))
    eqs = [agg[k]["eq"][0] for k in _ORDER]
    ax.bar(x, eqs, 0.62, color=[_COLORS[k] for k in _ORDER])
    for xi, v in zip(x, eqs):
        label = f"{v:.0e}" if v < 1e-2 else f"{v:.1f}"
        ax.text(xi, v * 1.4, label, ha="center", va="bottom", fontsize=8.5)
    ax.axhline(1e-4, color=C_REF, ls=":", lw=1, zorder=0)
    ax.text(len(x) - 0.5, 1.3e-4, "guard tol (1e-4)", color=C_REF, fontsize=7,
            ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_PRETTY[k] for k in _ORDER], fontsize=8.5)
    ax.set_ylabel(r"composed equivariance residual  $\Delta_{\mathrm{eq}}$  (log)")
    ax.set_ylim(1e-6, 1e2)
    ax.set_title("(b) Augmentation never approximates equivariance\n"
                 r"(VN floor weight-independent; aug $\sim\!10^{6}\times$ above)", fontsize=10)


def main() -> None:
    data = json.loads((FIG / "step45_augmented_mlp_closed_loop.json").read_text())
    agg = data["agg"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    panel_ratio(axes[0], agg)
    panel_eq(axes[1], agg)
    fig.suptitle(
        "Step 45 — exact vs augmentation in the closed loop: augmentation narrows the gap by "
        "coverage,\nbut only exact equivariance closes the loop (and never even approximates it here)",
        fontsize=11.5, y=1.03,
    )
    fig.tight_layout()
    out_png = FIG / "step45_augmented_vs_exact.png"
    out_pdf = FIG / "step45_augmented_vs_exact.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png.relative_to(ROOT)} and {out_pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
