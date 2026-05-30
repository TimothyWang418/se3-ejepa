r"""The "killer figure": one 3-panel summary of the project's central empirical claim.

Thesis in one image -- a hard SE(3)/SO(n) symmetry prior, wired into a JEPA latent world
model, makes generalisation *across the symmetry group* a structural property rather than a
data-hungry one. The three panels are the three error bars a sceptic asks for, in order:

* **(a) Flatness across the group.** Train on a single orientation wedge, test on the whole
  orbit. The equivariant model's OOD/seen relMSE factor is $\approx\!1$ in every setting
  (SO(2) synthetic & real, SO(2) latent, SO(3) 3D, full SE(3) with translation); the
  same-hypothesis-class baseline blows up by $\times13$ to $\times157$. (Numbers are the
  verified per-step runs tabulated in ``papers/equivariance_generalization_core.md`` §4.)
* **(b) Stable across training seeds (Step 17).** Five *independently trained* (VN, MLP)
  pairs, closed-loop pose control on real PushT. Plotted as seen-vs-unseen block-angle: the
  VN points sit on the $y=x$ diagonal (orientation-invariant), the MLP points sit above it
  (degrades) -- the contrast is a property of the architecture, not the lucky seed.
* **(c) Robust to misspecification (Step 16).** Break the SO(3) symmetry of the teacher with a
  fixed-lab-axis term and sweep it: the VN's OOD error climbs (the prior is *not* free once the
  world de-symmetrises) yet stays below the unconstrained MLP's at every level tested, even past
  50% symmetry-breaking. This *brackets* Sutton's Bitter-Lesson crossover honestly.

Panels (b) and (c) are read straight from the JSON the experiments dump; panel (a) uses the
measured OOD factors recorded in the papers. No model is re-trained here -- this is a plotting
step over already-computed, reproducible results.

Run:
    .venv/bin/python experiments/make_figures.py
    # -> papers/figures/killer_figure.png  (and .pdf)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "papers" / "figures"

# Palette: VN (equivariant) vs MLP (baseline), kept identical across panels.
C_VN = "#1b7837"   # green  = equivariant / "flat"
C_MLP = "#762a83"  # purple = baseline / "breaks"
C_REF = "#888888"  # grey reference lines (y=x, relMSE=1, invariant=1)


# --------------------------------------------------------------------------- #
# Panel (a): OOD/seen relMSE factor across the group  (verified per-step runs)
# --------------------------------------------------------------------------- #
# Source: papers/equivariance_generalization_core.md §4 (Result [B]) -- each row is a
# separate verified experiment; VN factor ~1 = exact across-group generalisation.
PANEL_A_LABELS = ["SO(2)\nsynth\n(St 8)", "SO(2)\nreal\n(St 10)", "SO(2)\nlatent\n(St 11)",
                  "SO(3)\n3D\n(St 13)", "SE(3)\n3D\n(St 15)"]
PANEL_A_VN = [1.17, 1.00, 1.00, 1.00, 1.00]      # OOD/seen factor, equivariant model
PANEL_A_MLP = [107.0, 16.2, 13.8, 17.2, 157.0]   # OOD/seen factor, baseline


def panel_a(ax: plt.Axes) -> None:
    r"""Grouped log-scale bars: VN $\approx\!1$ (flat) vs baseline $\times13$--$\times157$."""
    import numpy as np

    x = np.arange(len(PANEL_A_LABELS))
    w = 0.38
    ax.bar(x - w / 2, PANEL_A_VN, w, label="VN (equivariant)", color=C_VN)
    ax.bar(x + w / 2, PANEL_A_MLP, w, label="MLP (baseline)", color=C_MLP)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1, zorder=0)
    ax.text(len(x) - 0.5, 1.05, "invariant (×1)", color=C_REF, fontsize=7, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(PANEL_A_LABELS, fontsize=7.5)
    ax.set_ylabel("OOD / seen relMSE factor  (log)")
    ax.set_title("(a) Flat across the group\n(one wedge → whole orbit)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    # annotate the worst baseline blow-up
    ax.annotate("×157", xy=(4 + w / 2, 157), xytext=(4 + w / 2, 157 * 1.4),
                color=C_MLP, fontsize=8, ha="center")


# --------------------------------------------------------------------------- #
# Panel (b): multi-seed seen-vs-unseen closed-loop angle  (Step 17 JSON)
# --------------------------------------------------------------------------- #
def panel_b(ax: plt.Axes) -> None:
    r"""Per-seed scatter of (seen, unseen) block-angle; $y=x$ = orientation-invariant."""
    path = FIG / "step17_multiseed.json"
    if not path.exists():
        ax.text(0.5, 0.5, "step17_multiseed.json\nnot found", ha="center", va="center")
        ax.set_axis_off()
        return
    data = json.loads(path.read_text())
    rows = data["rows"]
    vn_seen = [r["vn_seen"] for r in rows]
    vn_uns = [r["vn_unseen"] for r in rows]
    mlp_seen = [r["mlp_seen"] for r in rows]
    mlp_uns = [r["mlp_unseen"] for r in rows]

    hi = max(vn_seen + vn_uns + mlp_seen + mlp_uns) * 1.12
    ax.plot([0, hi], [0, hi], ls="--", lw=1, color=C_REF, zorder=0)
    ax.text(hi * 0.97, hi * 0.9, "y = x\n(invariant)", color=C_REF, fontsize=7, ha="right", va="top")
    ax.scatter(vn_seen, vn_uns, s=55, color=C_VN, edgecolor="white", linewidth=0.6,
               label="VN (equivariant)", zorder=3)
    ax.scatter(mlp_seen, mlp_uns, s=55, color=C_MLP, marker="X", edgecolor="white",
               linewidth=0.6, label="MLP (baseline)", zorder=3)

    vn_d = data["vn_delta"]["mean"]
    mlp_d = data["mlp_delta"]["mean"]
    ax.text(0.04, 0.96,
            f"VN  Δ = {vn_d:+.1f}°  (on diagonal)\nMLP Δ = {mlp_d:+.1f}°  (above)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="white", ec=C_REF, alpha=0.85))
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("seen-orientation angle error (deg)")
    ax.set_ylabel("unseen-orientation angle error (deg)")
    ax.set_title("(b) Stable across 5 training seeds\n(Step 17, real PushT closed loop)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


# --------------------------------------------------------------------------- #
# Panel (c): OOD error vs symmetry-breaking  (Step 16 JSON)
# --------------------------------------------------------------------------- #
def panel_c(ax: plt.Axes) -> None:
    r"""OOD relMSE vs the non-equivariant fraction of the dynamics; the prior's robustness."""
    path = FIG / "step16_misspecification.json"
    if not path.exists():
        ax.text(0.5, 0.5, "step16_misspecification.json\nnot found", ha="center", va="center")
        ax.set_axis_off()
        return
    rows = json.loads(path.read_text())
    rows = sorted(rows, key=lambda r: r["noneq_frac"])
    nf = [r["noneq_frac"] for r in rows]
    vn = [r["vn_ood"] for r in rows]
    mlp = [r["mlp_ood"] for r in rows]

    ax.plot(nf, vn, "-o", color=C_VN, label="VN (equivariant) OOD", lw=1.8, ms=5)
    ax.plot(nf, mlp, "-X", color=C_MLP, label="MLP (baseline) OOD", lw=1.8, ms=6)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1, zorder=0)
    ax.text(nf[-1], 1.03, "relMSE = 1\n(worse than no-change)", color=C_REF, fontsize=7,
            ha="right", va="bottom")
    ax.axvline(1.0, color="#cccccc", ls=":", lw=1, zorder=0)  # noneq_frac=1 => broken part = equivariant part
    ax.text(1.0, max(mlp) * 0.98, "50%+ broken →",
            color="#999999", fontsize=7, ha="left", va="top", rotation=90)
    ax.set_xlabel("non-equivariant fraction of dynamics")
    ax.set_ylabel("OOD relMSE")
    ax.set_title("(c) Robust to misspecification\n(Step 16: prior costs, but still wins)", fontsize=10)
    ax.legend(fontsize=8, loc="center right")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.suptitle(
        "An exact SE(3)/SO(n)-equivariant JEPA world model generalises across the symmetry group: "
        "flat (a), seed-stable (b), and misspecification-robust (c)",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    out_png = FIG / "killer_figure.png"
    out_pdf = FIG / "killer_figure.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
