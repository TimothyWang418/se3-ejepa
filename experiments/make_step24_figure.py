r"""Step 24 figure: the interpolation/extrapolation flip under object interaction.

One image, two panels, read straight from ``papers/figures/step24_object_interaction.json``
(no model is re-trained here -- this is a plotting step over an already-computed run):

* **(a) In-distribution fit.** With objects *interacting* (a relative-pose torque
  $\omega_i=\hat r_{ij}\times a_i$), the non-equivariant **MLP-MP** fits *best* (relMSE
  $0.067$): an ordinary MLP can form the bilinear cross-product the dynamics need. Both
  Vector-Neuron models sit higher because a vanilla VN (VN-Linear + VN-ReLU) is **degree-1
  homogeneous** and *cannot* represent that cross product -- a real architectural cap. Among
  the VN models, the equivariant **relative-pose message** still helps: VN-MP ($0.331$) beats
  the channel-blind VN-Set ($0.450$), $\times1.36$ -- the message is necessary even in-dist.
* **(b) Across-the-group 举一反三.** Rotate the *whole* scene by a random $\mathrm{SO}(3)$ off
  the training wedge (a symmetry of the interacting teacher, after the interaction collapses
  the scene group to the global diagonal $\mathrm{SE}(3)\rtimes S_O$). The two equivariant
  models are **exactly flat** ($\times1.00$); the MLP that *won* panel (a) degrades
  $\times17$ -- to worse than predicting no latent change. The better interpolator is the
  catastrophically worse extrapolator: capacity wins in-distribution, the prior wins across
  the group.

Run:
    .venv/bin/python experiments/make_step24_figure.py
    # -> papers/figures/step24_object_interaction.png  (and .pdf)
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

# Palette, consistent with make_figures.py: equivariant = green, baseline = purple.
C_VNMP = "#1b7837"   # dark green  = equivariant + message (hero)
C_VNSET = "#5aae61"  # light green = equivariant, channel-blind
C_MLP = "#762a83"    # purple      = non-equivariant baseline
C_REF = "#888888"    # grey reference line
_ORDER = ["VN-MP", "VN-Set", "MLP-MP"]
_COLORS = {"VN-MP": C_VNMP, "VN-Set": C_VNSET, "MLP-MP": C_MLP}
_PRETTY = {"VN-MP": "VN-MP\n(equiv+msg)", "VN-Set": "VN-Set\n(equiv, no msg)",
           "MLP-MP": "MLP-MP\n(msg, no equiv)"}


def panel_indist(ax: plt.Axes, data: dict) -> None:
    r"""In-distribution relMSE bars; annotate the channel gap and the vanilla-VN cap."""
    indist = data["indist"]
    vals = [indist[k] for k in _ORDER]
    x = np.arange(len(_ORDER))
    bars = ax.bar(x, vals, 0.62, color=[_COLORS[k] for k in _ORDER])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1, zorder=0)
    ax.text(len(x) - 0.5, 1.01, "no-change baseline (×1)", color=C_REF,
            fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([_PRETTY[k] for k in _ORDER], fontsize=8)
    ax.set_ylabel("in-distribution 1-step latent relMSE")
    ax.set_ylim(0, 1.08)
    cf = data["channel_factor"]
    ax.set_title(f"(a) Interpolation: the MLP fits best\n"
                 f"(VN cross-product cap; message gap ×{cf:.2f})", fontsize=10)
    # bracket the channel gap (VN-MP vs VN-Set)
    y = max(indist["VN-MP"], indist["VN-Set"]) + 0.10
    ax.plot([0, 0, 1, 1], [y, y + 0.03, y + 0.03, y], color="#333333", lw=1.0)
    ax.text(0.5, y + 0.045, f"message ×{cf:.2f}", ha="center", va="bottom", fontsize=8)


def panel_ood(ax: plt.Axes, data: dict) -> None:
    r"""Global-orientation OOD/seen ratio bars (log): the two VN models flat, the MLP blows up."""
    glob = data["global_ood"]
    ratios = [glob[k]["ratio"] for k in _ORDER]
    x = np.arange(len(_ORDER))
    ax.bar(x, ratios, 0.62, color=[_COLORS[k] for k in _ORDER])
    for xi, r in zip(x, ratios):
        ax.text(xi, r * 1.06, f"×{r:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1, zorder=0)
    ax.text(len(x) - 0.5, 1.02, "invariant (×1)", color=C_REF, fontsize=7, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_PRETTY[k] for k in _ORDER], fontsize=8)
    ax.set_ylabel("global-orientation  OOD / seen relMSE  (log)")
    mlp_r = glob["MLP-MP"]["ratio"]
    ax.set_ylim(0.8, mlp_r * 3.0)
    ax.set_title("(b) Extrapolation: only the prior is flat\n"
                 "(whole scene rotated off the training wedge)", fontsize=10)


def main() -> None:
    path = FIG / "step24_object_interaction.json"
    data = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    panel_indist(axes[0], data)
    panel_ood(axes[1], data)
    cint = data["config"]["C_INT"]
    fig.suptitle(
        rf"Step 24 — object interaction ($\kappa$={cint}): capacity wins in-distribution, "
        rf"equivariance wins across the (collapsed) group",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    out_png = FIG / "step24_object_interaction.png"
    out_pdf = FIG / "step24_object_interaction.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png.relative_to(ROOT)} and {out_pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
