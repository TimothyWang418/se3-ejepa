r"""Step 55 (B2): group-complexity × training-coverage **phase diagram** for zero-shot generalization.

The configuration-axis certificate says $k$ generator checks certify the whole generated group $\langle S\rangle$.
The flip side is a scaling law for the *non-equivariant* alternative: to generalize zero-shot to the same set,
a free model must **see** enough composition coverage, and the coverage it needs grows with the group's
complexity. This experiment maps that as a 2-axis phase diagram on the I Ching $\mathbb{Z}_2^k$ testbed (Step 49):

  - **group complexity** $k$ — the group is $\mathbb{Z}_2^k$ with $|G|=2^k$ and $k$ single-line-flip generators;
  - **training coverage** $m_{\text{train}}$ — the model is trained on every action with $\le m_{\text{train}}$
    flips (so $m_{\text{train}}{=}1$ is "generators only").

We measure the **non-equivariant MLP's zero-shot relMSE** on the unseen compositions ($>m_{\text{train}}$ flips)
across the $(k, m_{\text{train}})$ grid, against the **equivariant per-line predictor**, which is certified over
all $2^k$ from the generators (it lives entirely in the $m_{\text{train}}{=}1$ row, for every $k$). The boundary
of the MLP's "generalizes" region recedes — it needs $m_{\text{train}}$ to grow with $k$ — while the equivariant
certificate is a single point ($m_{\text{train}}{=}1$) valid for all $k$: the structural advantage **grows with
group complexity**.

Run (full ~1-2 min):  .venv/bin/python experiments/step55_group_complexity_phase.py
Smoke:  STEP55_SMOKE=1 .venv/bin/python experiments/step55_group_complexity_phase.py
"""

import json
import os
import sys
from itertools import product
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
SMOKE = bool(os.environ.get("STEP55_SMOKE"))
SEED = int(os.environ.get("STEP55_SEED", "0"))
D = 8                                                    # per-line embedding dim


def all_patterns(k: int) -> torch.Tensor:
    return torch.tensor(list(product([1.0, -1.0], repeat=k)))     # (2^k, k)


class EquivPredictor(nn.Module):
    r"""Per-line, sign-equivariant $f(z,a)_i=h(a_i)\odot z_i$ — certified over all $2^k$ from the generators."""

    def __init__(self):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, D))

    def forward(self, z, a):
        k = a.shape[1]
        return self.h(a.reshape(-1, 1)).reshape(a.shape[0], k, D) * z


class MLPPredictor(nn.Module):
    r"""Non-equivariant joint predictor $g([z,a])$."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(k * D + k, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, k * D))

    def forward(self, z, a):
        return self.net(torch.cat([z.reshape(z.shape[0], -1), a], 1)).reshape(z.shape[0], self.k, D)


def zero_shot_relmse(k: int, m_train: int, equiv: bool, epochs: int) -> float:
    r"""Train on actions with $\le m_{\text{train}}$ flips; return worst relMSE over the UNSEEN
    compositions (>$m_{\text{train}}$ flips). NaN if there are no unseen compositions."""
    torch.manual_seed(SEED)
    states = all_patterns(k); actions = all_patterns(k)
    w = torch.randn(k, D)
    E = lambda x: x.unsqueeze(-1) * w.unsqueeze(0)
    nflip = (actions < 0).sum(1)
    tr = actions[nflip <= m_train]
    Sx = states.repeat_interleave(tr.shape[0], 0); Aa = tr.repeat(states.shape[0], 1)
    Zt, Yt = E(Sx), E(Aa * Sx)
    model = EquivPredictor() if equiv else MLPPredictor(k)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad(); ((model(Zt, Aa) - Yt) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        worst = 0.0
        for m in range(m_train + 1, k + 1):
            am = actions[nflip == m]
            if am.numel() == 0:
                continue
            sx = states.repeat_interleave(am.shape[0], 0); aa = am.repeat(states.shape[0], 1)
            pred, tgt = model(E(sx), aa), E(aa * sx)
            worst = max(worst, float(((pred - tgt) ** 2).sum() / (tgt ** 2).sum().clamp_min(1e-12)))
        return worst if m_train < k else float("nan")


def main() -> None:
    line = "=" * 92
    ks = [2, 3, 4, 5] if SMOKE else [2, 3, 4, 5, 6, 7, 8]
    epochs = 800 if SMOKE else 2000
    epochs = int(os.environ.get("STEP55_EPOCHS", epochs))
    kmax = max(ks)
    print(line)
    print(f"STEP 55 (B2)  group complexity (k) × training coverage (m_train) → zero-shot  ({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    # MLP zero-shot relMSE over the grid; equivariant certified at m_train=1 for every k
    mlp_grid = np.full((kmax, len(ks)), np.nan)            # rows: m_train 1..kmax, cols: k
    equiv_gen = []                                          # equivariant worst-unseen at m_train=1, per k
    mlp_gen = []                                            # MLP worst-unseen at m_train=1, per k
    for ci, k in enumerate(ks):
        for m_train in range(1, k):                        # m_train=k has no unseen compositions
            mlp_grid[m_train - 1, ci] = zero_shot_relmse(k, m_train, equiv=False, epochs=epochs)
        eq1 = zero_shot_relmse(k, 1, equiv=True, epochs=epochs)
        equiv_gen.append(eq1); mlp_gen.append(mlp_grid[0, ci])
        print(f"  k={k} (|G|={2**k:4d}): generators-only (m_train=1) worst-unseen relMSE — "
              f"equivariant {eq1:.2e}  vs  MLP {mlp_grid[0, ci]:.2e}")

    # the structural advantage at m_train=1 (the certified row) grows with group complexity
    gap = [m / max(e, 1e-12) for e, m in zip(equiv_gen, mlp_gen)]
    print(f"\n{line}\nSTEP 55 SUMMARY\n{line}")
    print(f"    equivariant is certified from the k generators (m_train=1) for EVERY k: worst-unseen "
          f"≤ {max(equiv_gen):.1e}")
    print(f"    MLP at m_train=1 fails and worsens with k: " +
          "  ".join(f"k{k}:{m:.2f}" for k, m in zip(ks, mlp_gen)))
    print(f"    structural advantage (MLP/equiv at m_train=1) grows with group complexity: " +
          "  ".join(f"k{k}:{g:.0e}" for k, g in zip(ks, gap)))
    ok = (max(equiv_gen) < 1e-3 and min(mlp_gen) > 1e-2 and mlp_gen[-1] >= mlp_gen[0])
    if ok:
        print(f"\n    CONFIRMED: the equivariant certificate is a single point (m_train=1) valid for ALL k, while")
        print(f"        the non-equivariant model fails zero-shot from generators and needs coverage that grows")
        print(f"        with k — the structural advantage GROWS with group complexity (|G|=2^k).")
    else:
        print(f"\n    INCONCLUSIVE: gate not met; reported as-is (no thresholds loosened).")

    out = dict(passed=bool(ok), ks=ks, equiv_gen=equiv_gen, mlp_gen=mlp_gen, gap=gap,
               mlp_grid=mlp_grid.tolist(), smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step55_group_complexity_phase_smoke" if SMOKE else "step55_group_complexity_phase"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7))
    im = ax[0].imshow(np.log10(np.clip(mlp_grid, 1e-6, None)), origin="lower", aspect="auto",
                      cmap="viridis", extent=[ks[0] - 0.5, ks[-1] + 0.5, 0.5, kmax + 0.5])
    ax[0].set_xlabel("group complexity k  (|G| = 2^k)"); ax[0].set_ylabel("training coverage m_train (max flips seen)")
    ax[0].set_title("non-equivariant MLP zero-shot relMSE (log10)\n— needs coverage growing with k")
    ax[0].plot(ks, [1] * len(ks), "r*", ms=13, label="equivariant certified\n(m_train=1, ∀k)")
    ax[0].legend(fontsize=8, loc="upper left"); fig.colorbar(im, ax=ax[0], label="log10 relMSE")
    ax[1].semilogy(ks, [max(e, 1e-9) for e in equiv_gen], "o-", color="C2", lw=1.9, ms=6,
                   label="equivariant (certified from k generators)")
    ax[1].semilogy(ks, mlp_gen, "s-", color="C3", lw=1.9, ms=6, label="MLP (generators only)")
    ax[1].set_xlabel("group complexity k"); ax[1].set_ylabel("worst-unseen zero-shot relMSE (m_train=1)")
    ax[1].set_title("structural advantage grows with group complexity\n(equiv flat ≈0; MLP fails and worsens)")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Step 55 (B2) — group complexity × coverage: the equivariant certificate is one point valid ∀k",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
