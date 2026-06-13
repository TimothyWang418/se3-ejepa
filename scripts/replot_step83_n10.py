"""Replot the step83 crossover figure from the n=10 merged artifact (same layout as the original)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers/figures"
d = json.loads((FIG / "step83_n10_merged.json").read_text())
COLORS = {"conv": "#1f77b4", "mlp": "#d62728", "gru": "#2ca02c"}
LABELS = {"conv": "$\\mathbb{Z}_N$-equivariant conv", "mlp": "dense MLP", "gru": "GRU-BPTT (recurrent)"}
MARKERS = {"conv": "o", "mlp": "x", "gru": "s"}
Ns = sorted(int(N) for N in d["cells"])
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
for a in ("conv", "mlp", "gru"):
    ys = [np.mean([r["spectrum_r2"] for r in d["cells"][str(N)][a]]) for N in Ns]
    es = [np.std([r["spectrum_r2"] for r in d["cells"][str(N)][a]]) for N in Ns]
    ax.errorbar(Ns, ys, yerr=es, marker=MARKERS[a], color=COLORS[a], lw=2, ms=7, capsize=3, label=LABELS[a])
ax.axhline(0.0, color="gray", lw=0.8, ls=":")
ax.axhline(0.8, color="#1f77b4", lw=0.8, ls="--", alpha=0.5)
ax.text(min(Ns), 0.815, "$R^2=0.8$ (recovery)", fontsize=7, color="#1f77b4", alpha=0.8)
ax.text(min(Ns), 0.02, "$R^2=0$ (mean-predictor)", fontsize=7, color="gray")
ax.set_xlabel("configuration dimension $N$ (Lorenz-96)")
ax.set_ylabel("full-spectrum $R^2$ (recovered vs true)")
ax.set_title("(a) $R^2(N)$ crossover: structure vs scale \\& recurrence ($n{=}10$ seeds/cell)")
ax.set_xticks(Ns); ax.set_ylim(-3.0, 1.08)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)
true_npos = []
for N in Ns:
    arrs = [np.array(v) for v in d["true_spectrum"][str(N)].values()]
    true_npos.append(float(np.mean([int((x > 0).sum()) for x in arrs])) if arrs else np.nan)
ax2.plot(Ns, true_npos, "k-", marker="D", lw=2, ms=6, label="true Lorenz-96")
for a in ("conv", "mlp", "gru"):
    npos = [np.mean([r["n_positive"] for r in d["cells"][str(N)][a]]) for N in Ns]
    ax2.plot(Ns, npos, marker=MARKERS[a], color=COLORS[a], lw=1.6, ms=6, alpha=0.9, label=LABELS[a])
ax2.set_xlabel("configuration dimension $N$")
ax2.set_ylabel("# positive Lyapunov exponents recovered")
ax2.set_title("(b) Positive-exponent count vs truth")
ax2.set_xticks(Ns); ax2.legend(fontsize=7.5, loc="upper left"); ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "step83_rsquared_crossover.png", dpi=180)
print("replotted at n=10")
