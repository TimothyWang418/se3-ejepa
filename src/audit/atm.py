r"""ATM — Action-Consistency Transfer Matrix (WMA audit instrument II).

Self-implementation of Chen, *ATM: Action-Consistency Transfer Matrix for Diagnosing and
Improving Latent World Models* (arXiv:2606.09028), written from the paper's equations;
the official repository is a README-only placeholder as of 2026-06-11.

Given a frozen latent world model (encoder $E_\theta$, predictor $F_\theta$) and offline
transitions $(o_t, a_t, o_{t+1})$, the caller supplies pre-computed latents and this module
performs the pure measurement:

- features (eq. 4):  $\xi^T_t = [z_t,\, z_{t+1},\, z_{t+1} - z_t]$ (true encoded transition)
  and $\xi^P_t = [z_t,\, \hat z_{t+1},\, \hat z_{t+1} - z_t]$ (model-predicted transition);
- probes (eq. 5):    $h(\xi) = W_2\,\mathrm{GELU}(\mathrm{LN}(W_1 \xi + b_1)) + b_2$,
  one *fresh* probe per (domain, width, seed), trained to regress $a_t$;
- matrix (eq. 6):    $D_{i,j} = \mathbb{E}\,\|h_i(\xi^j_t) - a_t\|_2^2$, rows = probe
  training domain, columns = evaluation domain, shared held-out test split for all cells;
- readouts (eqs. 8-9): $G_{T\to P} = (D_{T,P} - D_{T,T})/(D_{T,T} + \epsilon)$,
  $G_{P\to T} = (D_{P,T} - D_{P,P})/(D_{P,P} + \epsilon)$,
  $I_{\mathrm{diag}} = |D_{T,T} - D_{P,P}| / (\tfrac12 (D_{T,T} + D_{P,P}) + \epsilon)$,
  $L_{\mathrm{sym}} = I_{\mathrm{diag}} + |G_{T\to P} - G_{P\to T}|$.

Protocol additions over the paper (registered: docs/specs/2026-06-11-wma-step1-atm-selfimpl-seed.md):

1. **Capacity mini-scan** — the paper fixes a single, unstated probe width (its limitation 1:
   probe-capacity confound); we scan ``widths`` and report the main width plus sensitivity.
2. **Scale-free cells** — $\tilde D_{i,j} = D_{i,j} / \mathbb{E}\,\|a - \bar a\|_2^2$ with
   $\bar a$ the train-split mean ($1 - R^2$ form): predicting the mean action scores
   $\tilde D \approx 1$. Cross-model report cards use $\tilde D$ only (the paper's raw-score
   coefficient transfer across families failed at 68.54%, its Table 4).

The screening score $S_{\mathrm{ATM}}$ is intentionally NOT computed: its $\lambda$ weights
are fit against simulator success rates (chicken-and-egg vs "simulator-free"); we report
components only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

Cell = Tuple[str, str]


def build_transition_features(z_t: Tensor, z_next: Tensor) -> Tensor:
    r"""Eq. 4: $\xi = [z_t,\, z_{t+1},\, z_{t+1} - z_t]$ — start, end, explicit difference.

    Shapes: (N, d), (N, d) -> (N, 3d). Token/patch-level latents must be mean-pooled to a
    single vector per frame by the caller *before* this point (paper's convention).
    """
    if z_t.shape != z_next.shape:
        raise ValueError(f"shape mismatch: {tuple(z_t.shape)} vs {tuple(z_next.shape)}")
    return torch.cat([z_t, z_next, z_next - z_t], dim=-1)


class _Probe(nn.Module):
    r"""Eq. 5: $h(\xi) = W_2\,\mathrm{GELU}(\mathrm{LN}(W_1 \xi + b_1)) + b_2$."""

    def __init__(self, d_in: int, d_out: int, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, width)
        self.ln = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, d_out)

    def forward(self, xi: Tensor) -> Tensor:  # (N, 3d) -> (N, d_a)
        return self.fc2(self.act(self.ln(self.fc1(xi))))


def _eval_D(probe: nn.Module, feats: Tensor, actions: Tensor) -> float:
    r"""$D = \mathbb{E}\,\|h(\xi) - a\|_2^2$ — squared L2 summed over action dims, mean over samples."""
    probe.eval()
    with torch.no_grad():
        return float(((probe(feats) - actions) ** 2).sum(dim=-1).mean())


def _train_probe(
    feats_tr: Tensor,
    a_tr: Tensor,
    feats_val: Tensor,
    a_val: Tensor,
    *,
    width: int,
    seed: int,
    max_steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    patience: int,
) -> nn.Module:
    """Train one fresh probe with early stopping on val D; restore the best state."""
    torch.manual_seed(seed)
    probe = _Probe(feats_tr.shape[-1], a_tr.shape[-1], width)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = feats_tr.shape[0]
    best_val, best_state, stale = float("inf"), copy.deepcopy(probe.state_dict()), 0
    step = 0
    while step < max_steps:
        idx = torch.randperm(n, generator=gen)
        for lo in range(0, n, batch_size):
            b = idx[lo:lo + batch_size]
            probe.train()
            loss = ((probe(feats_tr[b]) - a_tr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                v = _eval_D(probe, feats_val, a_val)
                if v < best_val:
                    best_val, best_state, stale = v, copy.deepcopy(probe.state_dict()), 0
                else:
                    stale += 1
                if stale >= patience:
                    probe.load_state_dict(best_state)
                    return probe
            if step >= max_steps:
                break
    probe.load_state_dict(best_state)
    return probe


def compute_readouts(D: Dict[Cell, float], eps: float = 1e-8) -> Dict[str, float]:
    r"""Eqs. 8-9 readouts from the four cells (see module docstring for formulas)."""
    dtt, dtp, dpt, dpp = D[("T", "T")], D[("T", "P")], D[("P", "T")], D[("P", "P")]
    g_tp = (dtp - dtt) / (dtt + eps)
    g_pt = (dpt - dpp) / (dpp + eps)
    i_diag = abs(dtt - dpp) / (0.5 * (dtt + dpp) + eps)
    return {"G_TP": g_tp, "G_PT": g_pt, "I_diag": i_diag, "L_sym": i_diag + abs(g_tp - g_pt)}


@dataclass
class ATMResult:
    """Main numbers (median over probe seeds at ``main_width``) + full sensitivity grid."""

    D: Dict[Cell, float]
    D_norm: Dict[Cell, float]
    readouts: Dict[str, float]
    action_var: float
    main_width: int
    per_width: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        key = lambda c: f"D_{c[0]}{c[1]}"  # noqa: E731
        return {
            "D": {key(c): v for c, v in self.D.items()},
            "D_norm": {key(c): v for c, v in self.D_norm.items()},
            "readouts": dict(self.readouts),
            "action_var": self.action_var,
            "main_width": self.main_width,
            "per_width": {str(w): d for w, d in self.per_width.items()},
            "config": dict(self.config),
        }


def atm_audit(
    z_t: Tensor,
    z_next_true: Tensor,
    z_next_pred: Tensor,
    actions: Tensor,
    *,
    widths: Sequence[int] = (64, 256, 1024),
    main_width: int = 256,
    probe_seeds: Sequence[int] = (0, 1, 2),
    test_frac: float = 0.2,
    val_frac: float = 0.125,
    max_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    eval_every: int = 50,
    patience: int = 5,
    eps: float = 1e-8,
    split_seed: int = 0,
    train_idx: Optional[Tensor] = None,
    test_idx: Optional[Tensor] = None,
) -> ATMResult:
    r"""Run the full ATM measurement on pre-computed latents (model stays outside, frozen).

    Args:
        z_t: true encoded latents at time $t$, mean-pooled, shape (N, d).
        z_next_true: true encoded latents at $t{+}1$ (domain $T$ endpoint), shape (N, d).
        z_next_pred: model-predicted latents $\hat z_{t+1}$ (domain $P$ endpoint), shape (N, d).
        actions: regression targets $a_t$ (flattened action/chunk), shape (N, d_a).
        train_idx / test_idx: optional explicit split (e.g. episode-level, to prevent
            leakage); default is a random sample-level split with ``split_seed``.

    Returns:
        ATMResult — main cells = median over ``probe_seeds`` at ``main_width``; the
        capacity mini-scan over ``widths`` is kept in ``per_width`` for disclosure.
    """
    z_t, z_next_true = z_t.float(), z_next_true.float()
    z_next_pred, actions = z_next_pred.float(), actions.float()
    n = z_t.shape[0]
    if not (z_next_true.shape[0] == z_next_pred.shape[0] == actions.shape[0] == n):
        raise ValueError("z_t, z_next_true, z_next_pred, actions must share dim 0")

    feats = {"T": build_transition_features(z_t, z_next_true),
             "P": build_transition_features(z_t, z_next_pred)}

    if train_idx is None or test_idx is None:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
        n_test = max(1, int(round(test_frac * n)))
        test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_idx, test_idx = train_idx.long(), test_idx.long()
    n_val = max(1, int(round(val_frac * train_idx.shape[0])))
    val_idx, fit_idx = train_idx[:n_val], train_idx[n_val:]

    # scale-free baseline: predict the TRAIN-split mean action, evaluated on the test split
    a_bar = actions[train_idx].mean(dim=0)
    action_var = float(((actions[test_idx] - a_bar) ** 2).sum(dim=-1).mean())

    per_width: Dict[int, Dict[str, Dict[str, float]]] = {}
    cells_main: Dict[Cell, list] = {c: [] for c in
                                    [("T", "T"), ("T", "P"), ("P", "T"), ("P", "P")]}
    for width in widths:
        runs: Dict[str, Dict[str, float]] = {}
        for seed in probe_seeds:
            D_run: Dict[Cell, float] = {}
            for di, dom in enumerate(("T", "P")):
                probe = _train_probe(
                    feats[dom][fit_idx], actions[fit_idx],
                    feats[dom][val_idx], actions[val_idx],
                    width=width, seed=7919 * seed + 31 * width + di,
                    max_steps=max_steps, batch_size=batch_size, lr=lr,
                    eval_every=eval_every, patience=patience)
                for ev in ("T", "P"):                                # shared test split, all 4 cells
                    D_run[(dom, ev)] = _eval_D(probe, feats[ev][test_idx], actions[test_idx])
            runs[str(seed)] = {f"D_{i}{j}": v for (i, j), v in D_run.items()}
            if width == main_width:
                for c in cells_main:
                    cells_main[c].append(D_run[c])
        med = {k: median(r[k] for r in runs.values()) for k in next(iter(runs.values()))}
        per_width[width] = {"median": med, "per_seed": runs}

    if main_width not in widths:
        raise ValueError(f"main_width {main_width} not in widths {tuple(widths)}")
    D_main = {c: median(v) for c, v in cells_main.items()}
    return ATMResult(
        D=D_main,
        D_norm={c: v / action_var for c, v in D_main.items()},
        readouts=compute_readouts(D_main, eps=eps),
        action_var=action_var,
        main_width=main_width,
        per_width=per_width,
        config={"widths": list(widths), "probe_seeds": list(probe_seeds),
                "max_steps": max_steps, "batch_size": batch_size, "lr": lr,
                "eval_every": eval_every, "patience": patience, "eps": eps,
                "n": n, "n_train": int(train_idx.shape[0]), "n_test": int(test_idx.shape[0]),
                "split_seed": split_seed},
    )
