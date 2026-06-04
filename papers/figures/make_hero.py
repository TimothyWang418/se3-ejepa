r"""Generate the hero concept figure (Figure 1): the predictability certificate across three axes.

A schematic — no data — that anchors the paper's thesis: *scale buys interpolation; structure buys a certificate*.
Left panel shows the certified region in the (configuration $\langle S\rangle$, horizon $T$) plane: structure
certifies the **entire** generated monoid (every composition $w$) up to a spectral horizon ceiling, whereas
scale/data certifies only a small interpolation tube near the training set. Right panel shows the horizon$\times$
resolution trade-off $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$: conserved/invariant (slow, $\lambda\le0$)
channels are certified to all horizons (eclipses, millennia); chaotic ($\lambda>0$) channels shrink with demanded
resolution (weather, ~two weeks).

Run:  .venv/bin/python papers/figures/make_hero.py   ->   papers/figures/hero_certified_region.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

FIG = Path(__file__).resolve().parent
GREEN, LGREEN, ORANGE, GREY = "#2a9d4a", "#bfe3c6", "#e76f51", "#6b6b6b"


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # ---- Panel A: configuration x horizon (at fixed resolution) -------------------------------------
    wmax, T_chaotic, T_slow, T_top = 8.0, 1.6, 7.2, 8.5
    # structure: the WHOLE configuration axis is certified, up to a per-channel spectral ceiling
    axL.fill_between([0, wmax], 0, T_slow, color=LGREEN, zorder=1)            # slow channels: tall
    axL.fill_between([0, wmax], 0, T_chaotic, color=GREEN, alpha=0.85, zorder=2)  # all channels: binding ceiling
    axL.axhline(T_chaotic, 0.04, 0.96, color=GREEN, lw=1.6, ls="--", zorder=3)
    axL.axhline(T_slow, 0.04, 0.96, color=GREEN, lw=1.4, ls=":", zorder=3)
    # scale/data: only a small interpolation tube near the training set (low composition)
    axL.fill_between([0, 1.4], 0, T_top, color=ORANGE, alpha=0.30, zorder=2)
    axL.fill_between([0, 1.4], 0, T_top, facecolor="none", edgecolor=ORANGE, lw=1.6, ls="-", zorder=4)

    axL.text(4.3, T_chaotic / 2, "certified:\nevery composition,\nall channels",
             ha="center", va="center", fontsize=8.5, color="white", weight="bold", zorder=5)
    axL.text(4.3, (T_chaotic + T_slow) / 2 + 0.1, "slow channels only\n(conserved / invariant)",
             ha="center", va="center", fontsize=8.0, color=GREEN, zorder=5)
    axL.text(0.7, T_top * 0.62, "scale\n(interpolation\ntube $\\sim\\epsilon/L$)",
             ha="center", va="center", fontsize=7.6, color=ORANGE, weight="bold", zorder=6, rotation=90)
    axL.annotate("", xy=(wmax - 0.15, T_chaotic + 0.45), xytext=(1.5, T_chaotic + 0.45),
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4), zorder=6)
    axL.text((1.5 + wmax) / 2, T_chaotic + 0.75, "$k$ generators $\\Rightarrow$ exponential $\\langle S\\rangle$ (Lemma 1)",
             ha="center", va="bottom", fontsize=7.8, color=GREY, zorder=6)
    axL.text(4.0, T_slow + 0.35, "horizon ceiling $=$ spectrum $\\{\\lambda_j\\}$ (Theorem B)",
             ha="center", fontsize=7.8, color=GREY)

    axL.set_xlim(0, wmax); axL.set_ylim(0, T_top)
    axL.set_xlabel("configuration  —  composition length $|w|$, $w\\in\\langle S\\rangle$")
    axL.set_ylabel("certified horizon $T$")
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_title("Structure certifies the whole orbit; scale only a tube", fontsize=10)

    # ---- Panel B: horizon x resolution (the spectral trade-off) -------------------------------------
    eps = np.linspace(0.06, 3.2, 200)            # x = log(1/epsilon): left coarse, right fine
    lam = 0.69
    T_chan = np.log(1.0 / np.exp(-eps)) / lam     # = eps/lam ; chaotic channel horizon ~ log(1/eps)/lambda
    axR.plot(eps, np.minimum(T_chan, 9.0) * 0 + 9.2, color=GREEN, lw=2.6, label="slow $\\lambda\\leq 0$ (conserved)")
    axR.plot(eps, np.clip(9.2 - eps * 2.6, 0.4, 9.2), color=ORANGE, lw=2.6, ls="--",
             label="chaotic $\\lambda>0$:  $T\\sim\\log(1/\\epsilon)/\\lambda$")
    axR.fill_between(eps, 0, np.clip(9.2 - eps * 2.6, 0.4, 9.2), color=ORANGE, alpha=0.10)
    axR.text(2.55, 9.55, "eclipses (millennia)", ha="center", fontsize=8.2, color=GREEN, weight="bold")
    axR.text(2.35, 2.1, "weather\n($\\sim$2 weeks)", ha="center", fontsize=8.2, color=ORANGE, weight="bold")
    axR.annotate("coarse", xy=(0.2, -0.9), annotation_clip=False, fontsize=7.5, color=GREY)
    axR.annotate("fine", xy=(2.95, -0.9), annotation_clip=False, fontsize=7.5, color=GREY)
    axR.set_xlim(0.06, 3.2); axR.set_ylim(0, 10.2)
    axR.set_xlabel("demanded resolution  —  $\\log(1/\\epsilon)$")
    axR.set_ylabel("certified horizon $T_j(\\epsilon)$")
    axR.set_xticks([]); axR.set_yticks([])
    axR.set_title("Ultra-long forecasts must be coarse", fontsize=10)
    axR.legend(loc="lower left", fontsize=7.8, framealpha=0.9)

    fig.suptitle("The predictability certificate: a provable region across configuration $\\times$ horizon $\\times$ resolution",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    out = FIG / "hero_certified_region.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
