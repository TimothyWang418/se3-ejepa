r"""Render the project's headline figures from the Step-21 / Step-22 JSON dumps.

Two artefacts, both written to ``papers/figures/``:

  * ``step21_frontier.png`` -- the **sample-efficiency frontier** under an exactly-SO(3) teacher:
    the VN's whole-group learning curve (which *equals* its in-wedge curve, a theorem) **descends**
    with data, while the non-equivariant MLP's whole-group error is a flat **wall** even though it
    fits the wedge better. (Step 21 / CLAUDE.md Open Question #1.)

  * ``where_the_bet_pays.png`` (+ ``.pdf``) -- the 3-panel **"where the geometric bet pays off"**
    figure: the frontier (left), then the Step-22 symmetry-break $\times$ data phase diagram in the
    two metrics -- across the group (middle: the prior wins the whole box, data-proof and
    break-robust) and in-distribution (right: capacity wins, worsening with the symmetry break $g$).

Pure read-and-plot; no training. Run *after* the Step-21 and Step-22 experiments have dumped JSON:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/make_bet_figures.py
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers" / "figures"

_BLUE, _RED, _GREY = "#1f77b4", "#d62728", "#555555"


def _load(name: str) -> dict:
    return json.loads((FIG / name).read_text())


# --------------------------------------------------------------------------- #
# panel 1: the Step-21 sample-efficiency frontier
# --------------------------------------------------------------------------- #
def plot_frontier(ax, d21: dict) -> None:
    r"""relMSE vs $N$: VN whole-group(=in-wedge) descending vs the MLP whole-group wall."""
    N = d21["N_grid"]
    vn, mlp = d21["curves"]["VN"], d21["curves"]["MLP"]
    vn_o = np.array(vn["ood"]); vn_os = np.array(vn.get("ood_std", [0] * len(N)))

    # VN: whole-group == in-wedge (one bold curve, the theorem), with a seed-std band
    ax.plot(N, vn_o, "o-", color=_BLUE, lw=2.4, ms=6, zorder=5,
            label="VN (equivariant): whole-group $=$ in-wedge")
    ax.fill_between(N, vn_o - vn_os, vn_o + vn_os, color=_BLUE, alpha=0.15, zorder=1)
    # MLP: fits the wedge (dashed, descends) but its whole-group error is a wall (solid, flat-high)
    ax.plot(N, mlp["seen"], "s--", color=_RED, lw=1.6, ms=5, alpha=0.75, zorder=4,
            label="MLP: in-wedge (fits)")
    ax.plot(N, mlp["ood"], "s-", color=_RED, lw=2.4, ms=6, zorder=4,
            label="MLP: whole-group (wall)")
    # the trivial no-change predictor (relMSE = 1): anything above this has not even learned "stay"
    ax.axhline(1.0, color=_GREY, ls=":", lw=1.2, zorder=0)
    ax.text(N[0], 1.03, "no-change predictor (relMSE $=1$)", color=_GREY, fontsize=7.5, va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("training-set size  $N$  (wedge transitions)")
    ax.set_ylabel("latent 1-step relMSE")
    ax.set_title("Sample-efficiency frontier (exact SO(3))", fontsize=10.5)
    ax.set_ylim(0.0, max(3.4, float(np.max(mlp["ood"])) * 1.08))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.6, loc="center left")
    p = d21["params"]
    ax.text(0.97, 0.96, f"VN {p['VN']:,} params\nMLP {p['MLP']:,} ({p['MLP']/p['VN']:.1f}$\\times$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


# --------------------------------------------------------------------------- #
# panels 2-3: the Step-22 (g x N) phase diagram, one metric each
# --------------------------------------------------------------------------- #
def _ratio_map(d22: dict, metric: str) -> np.ndarray:
    G, N = d22["G_values"], d22["N_grid"]
    M = np.empty((len(G), len(N)), np.float32)
    for gi, g in enumerate(G):
        for ni, n in enumerate(N):
            c = d22["grid"][f"{g}"][f"{n}"]
            M[gi, ni] = math.log10(max(c["mlp_" + metric], 1e-12) / max(c["vn_" + metric], 1e-12))
    return M


def plot_phase(ax, d22: dict, metric: str, title: str, fig, vmax: float = 1.0) -> None:
    r"""Heatmap of $\log_{10}(\mathrm{MLP}/\mathrm{VN})$ over $(N,g)$; red $=$ VN wins, blue $=$ MLP wins."""
    from matplotlib.colors import TwoSlopeNorm
    G, N = d22["G_values"], d22["N_grid"]
    M = _ratio_map(d22, metric)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", origin="upper")
    ax.set_xticks(range(len(N)), [str(n) for n in N])
    ax.set_yticks(range(len(G)), [f"{g:g}" for g in G])
    ax.set_xlabel("training-set size  $N$")
    ax.set_ylabel("symmetry break  $g$")
    ax.set_title(title, fontsize=10.5)
    for gi in range(len(G)):
        for ni in range(len(N)):
            c = d22["grid"][f"{G[gi]}"][f"{N[ni]}"]
            w = "VN" if c["vn_" + metric] < c["mlp_" + metric] else "MLP"
            ax.text(ni, gi, w, ha="center", va="center", fontsize=7.6,
                    color="black" if abs(M[gi, ni]) < 0.45 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$\log_{10}(\mathrm{MLP}/\mathrm{VN})$")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d21 = _load("step21_sample_efficiency_frontier.json")

    # --- standalone frontier ------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.6))
    plot_frontier(ax1, d21)
    fig1.tight_layout()
    fig1.savefig(FIG / "step21_frontier.png", dpi=150)
    print(f"    wrote {FIG / 'step21_frontier.png'}")

    # --- combined "where the bet pays" (frontier + 2 phase panels) ---------- #
    try:
        d22 = _load("step22_symmetry_data_phase.json")
    except FileNotFoundError:
        print("    [step22 JSON not found -- run experiments/step22_symmetry_data_phase.py first;"
              " skipping the combined figure]")
        return

    # shared symmetric colour scale across both phase panels (so reds/blues are comparable)
    vmax = max(float(np.abs(_ratio_map(d22, "ood")).max()),
               float(np.abs(_ratio_map(d22, "seen")).max()), 0.3)
    fig2, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))
    plot_frontier(axes[0], d21)
    n_cells = len(d22["G_values"]) * len(d22["N_grid"])
    n_vn = round(d22.get("ood_vn_win_fraction", 1.0) * n_cells)
    plot_phase(axes[1], d22, "ood", f"Across the group: prior wins {n_vn}/{n_cells}", fig2, vmax=vmax)
    plot_phase(axes[2], d22, "seen", "In-distribution: capacity wins (early, every $g$)", fig2, vmax=vmax)
    fig2.suptitle("Where the geometric bet pays off:  equivariance is a near-total, data-proof win across "
                  "the group, a wash-to-loss in-distribution", fontsize=12.5)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(FIG / "where_the_bet_pays.png", dpi=150)
    fig2.savefig(FIG / "where_the_bet_pays.pdf")
    print(f"    wrote {FIG / 'where_the_bet_pays.png'}")
    print(f"    wrote {FIG / 'where_the_bet_pays.pdf'}")


if __name__ == "__main__":
    main()
