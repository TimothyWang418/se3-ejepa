r"""Step 49: the I Ching $\mathbb{Z}_2^6$ **exponential** certificate — $k$ generators certify $2^k$ states.

paper2's headline (config axis) is "verify equivariance on $k$ generators ⇒ certified over the *exponential*
generated set $\langle S\rangle$." Step 47 showed the certificate on the real embodied model but with only a
modest $O{=}2$ word-closure. Here is the clean, *exponential* demonstration, on a culturally-resonant group:
the 64 hexagrams of the I Ching $=\{\pm1\}^6=\mathbb{Z}_2^6$, with the **6 changing-lines** as the generating
set $S$ (each $g_i$ flips line $i$). The generated group is all $2^6=64$ compositions.

World / task (a clean JEPA dynamics).
- State $x\in\{\pm1\}^6$ (a hexagram). Fixed per-line embedding $w_i\in\mathbb{R}^d$ gives the latent
  $z=E(x)$, $z_i=x_i w_i$. Action $a\in\{\pm1\}^6$ flips the lines with $a_i=-1$; the next state is
  $y=a\odot x$, so $E(y)_i = a_i z_i$ (the group acts on the latent by the sign rep $\rho(a)$).
- We **train predictors only on the $6$ single-line generators (+ identity), $7$ of the $64$ actions**, then
  test on **all $64$** actions, bucketed by composition length = number of flipped lines.

Two models.
- **Equivariant** predictor $f_\theta(z,a)_i = h_\theta(a_i)\odot z_i$ — a per-line, sign-equivariant map
  (odd in $z_i$, so $f(\rho(b)z,a)=\rho(b)f(z,a)$). Because it factorises **per line**, learning each line's
  rule from the single-line generators *determines* every composition: it is **certified over all 64 from 6
  generator-checks**.
- **Non-equivariant** MLP $g_\theta([z,a])$ — joint over all lines. Trained on the same 7 actions, it has no
  compositional structure and must **see** multi-line compositions, so it degrades with composition length.

This is the BRo-JEPA mechanism on an *exponential* (not cyclic) group: the empirical face of Theorem A's
"$k$ generator checks certify the generated set", with genuine learning (the equivariant model learns the
per-line rules), and the honest contrast that the MLP cannot generalise from generators to compositions.

Run:  .venv/bin/python experiments/step49_iching_certificate.py
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

torch.set_default_dtype(torch.float64)   # tiny problem; use float64 so "exact" is unambiguous

K = 6                # six lines of a hexagram -> Z_2^6, 2^6 = 64 states
D = 8                # per-line embedding dim
SEED = int(os.environ.get("STEP49_SEED", "0"))


def all_patterns():
    r"""All $2^6=64$ sign vectors $\{\pm1\}^6$ (hexagrams / actions)."""
    return torch.tensor(list(product([1.0, -1.0], repeat=K)))   # (64, 6)


class EquivPredictor(nn.Module):
    r"""Per-line, sign-equivariant: $f(z,a)_i = h(a_i)\odot z_i$. Odd in $z_i$ ⇒ commutes with $\rho(b)$.
    Learning $h(\pm1)\in\mathbb{R}^d$ from the single-line generators determines all $2^6$ compositions."""

    def __init__(self):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, D))  # h(a_i) in R^d, per line

    def forward(self, z, a):                       # z:(B,K,D)  a:(B,K)
        scale = self.h(a.reshape(-1, 1)).reshape(a.shape[0], K, D)
        return scale * z                           # (B,K,D)


class MLPPredictor(nn.Module):
    r"""Non-equivariant joint predictor $g([z,a])$ — no per-line factorisation."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K * D + K, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, K * D),
        )

    def forward(self, z, a):
        out = self.net(torch.cat([z.reshape(z.shape[0], -1), a], dim=-1))
        return out.reshape(z.shape[0], K, D)


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 78
    states = all_patterns()                        # (64, 6)
    actions = all_patterns()                        # (64, 6)  the full Z_2^6 action group
    w = torch.randn(K, D)                           # fixed per-line embedding
    E = lambda x: x.unsqueeze(-1) * w.unsqueeze(0)   # (B,6)->(B,6,D)  z_i = x_i w_i

    # generating set S = the 6 single-line flips (+ identity) = 7 of the 64 actions
    nflip = (actions < 0).sum(1)                    # composition length of each action
    train_mask = nflip <= 1                          # identity (0) + 6 single-line generators (1)
    train_actions = actions[train_mask]             # (7, 6)
    print(line)
    print(f"STEP 49  I Ching Z_2^6 certificate: train on {train_actions.shape[0]} generators -> certify all 64")
    print(line)
    print(f"    states={states.shape[0]}  actions={actions.shape[0]}  train actions (|flips|<=1)={train_actions.shape[0]}")

    # build training pairs (every state x every training action)
    Sx = states.repeat_interleave(train_actions.shape[0], 0)        # (64*7, 6)
    Aa = train_actions.repeat(states.shape[0], 1)                    # (64*7, 6)
    Yt = E(Aa * Sx)                                                  # target next-latent (B,6,D)
    Zt = E(Sx)

    results = {}
    for name, Model in [("equivariant", EquivPredictor), ("mlp", MLPPredictor)]:
        torch.manual_seed(SEED)
        model = Model()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for ep in range(2000):
            opt.zero_grad()
            loss = ((model(Zt, Aa) - Yt) ** 2).mean()
            loss.backward(); opt.step()
        # evaluate on ALL 64 actions, bucketed by composition length
        with torch.no_grad():
            per_len = {}
            for m in range(K + 1):
                am = actions[nflip == m]
                if am.numel() == 0:
                    continue
                sx = states.repeat_interleave(am.shape[0], 0)
                aa = am.repeat(states.shape[0], 1)
                pred, tgt = model(E(sx), aa), E(aa * sx)
                rel = (((pred - tgt) ** 2).sum() / (tgt ** 2).sum().clamp_min(1e-12)).item()
                per_len[m] = rel
        results[name] = per_len
        seen = per_len[1]; unseen = max(per_len[m] for m in per_len if m >= 2)
        print(f"\n  {name}: relMSE by composition length (|flips|): " +
              "  ".join(f"{m}:{per_len[m]:.2e}" for m in sorted(per_len)))
        print(f"    seen-len(1) {seen:.2e} | worst unseen-len(>=2) {unseen:.2e}")

    # equivariance check of the trained equivariant predictor (sanity: commutes with rho(b))
    torch.manual_seed(SEED); eqm = EquivPredictor()
    opt = torch.optim.Adam(eqm.parameters(), lr=3e-3)
    for _ in range(2000):
        opt.zero_grad(); (((eqm(Zt, Aa) - Yt) ** 2).mean()).backward(); opt.step()
    with torch.no_grad():
        b = all_patterns()[torch.randint(0, 64, (16,))]            # random rho(b)
        z0 = E(states[:16]); a0 = actions[torch.randint(0, 64, (16,))]
        lhs = eqm(b.unsqueeze(-1) * z0, a0)
        rhs = b.unsqueeze(-1) * eqm(z0, a0)
        equiv_resid = (lhs - rhs).abs().max().item()

    eq_worst = max(results["equivariant"][m] for m in results["equivariant"] if m >= 2)
    mlp_worst = max(results["mlp"][m] for m in results["mlp"] if m >= 2)
    ok = eq_worst < 1e-3 and mlp_worst > 10 * max(eq_worst, 1e-12)
    print(f"\n{line}\nSTEP 49 SUMMARY\n{line}")
    print(f"    equivariant predictor equivariance residual (commutes with rho(b)): {equiv_resid:.2e}")
    print(f"    EQUIV worst unseen-composition relMSE {eq_worst:.2e}  vs  MLP {mlp_worst:.2e}")
    if ok:
        print(f"    CONFIRMED: trained on 6 single-line generators, the EQUIVARIANT model is certified over")
        print(f"        all 2^6=64 hexagram compositions (worst relMSE {eq_worst:.1e}); the non-equivariant")
        print(f"        MLP degrades on unseen compositions ({mlp_worst:.1e}) — k generators -> 2^k, clean.")
    else:
        print(f"    INCONCLUSIVE: gate not met (eq_worst {eq_worst:.1e}, mlp_worst {mlp_worst:.1e}); reported as-is.")

    out = {"results": {k: {str(m): v for m, v in d.items()} for k, d in results.items()},
           "equiv_residual": equiv_resid, "eq_worst_unseen": eq_worst, "mlp_worst_unseen": mlp_worst,
           "passed": bool(ok), "K": K, "D": D, "n_train_actions": int(train_actions.shape[0])}
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "step49_iching_certificate.json").write_text(json.dumps(out, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, c in [("equivariant", "C2"), ("mlp", "C3")]:
        ms = sorted(results[name]); ax.plot(ms, [results[name][m] for m in ms], "o-", color=c, label=name)
    ax.axvspan(-0.3, 1.3, color="C0", alpha=0.10, label="trained (|flips|≤1: 6 generators)")
    ax.set_yscale("log"); ax.set_xlabel("composition length = # flipped lines")
    ax.set_ylabel("relMSE (predict E(y))")
    ax.set_title("Step 49 — I Ching $\\mathbb{Z}_2^6$: 6 generators certify all 64 (equivariant) vs MLP")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fig_dir / "step49_iching_certificate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote papers/figures/step49_iching_certificate.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
