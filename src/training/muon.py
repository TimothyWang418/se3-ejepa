r"""Muon optimiser — *Momentum Orthogonalised by Newton–Schulz*.

Reference: Keller Jordan et al., "Muon: An optimizer for the hidden layers of
neural networks" (2024), https://kellerjordan.github.io/posts/muon/ .

Idea. For a 2D weight matrix $W$ with momentum buffer $M$, Muon replaces the raw
momentum step by its *orthogonal polar factor* $\mathrm{UV}^\top$ (where
$M = U\Sigma V^\top$), approximated cheaply by a quintic Newton–Schulz iteration
that needs only matmuls (no SVD). The update direction is thus the closest
orthogonal matrix to the momentum, which equalises the per-direction step size —
empirically a strong optimiser for dense hidden layers.

Scope in this project. We route parameters with ``ndim >= 2`` (Linear/Conv
weights, and the predictor MLP) to Muon, and 1D params (biases, norm scales, and
``e2cnn`` basis-expansion coefficients, which are stored as 1D vectors) to AdamW
— the standard Muon recipe. See :func:`build_muon_adamw`.

Note (relevant to the Symmetry-Compatible-Optimizer line of work): for *hard*
basis-constrained equivariant layers (``e2cnn``), equivariance is a property of
the kernel *subspace*, so **any** optimiser acting on the basis coefficients
keeps the expanded kernel equivariant. Muon's orthogonalisation here only ever
touches ordinary dense weights; the steerable encoder's exact equivariance is
therefore optimiser-agnostic — a fact we verify empirically after training.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    r"""Approximate the orthogonal polar factor of a 2D matrix via Newton–Schulz.

    Computes $\mathrm{UV}^\top$ for $G = U\Sigma V^\top$ using the quintic
    iteration $X \leftarrow aX + (bA + cA^2)X$ with $A = XX^\top$ and the
    standard coefficients $(a,b,c) = (3.4445, -4.7750, 2.0315)$, after scaling
    $G$ to unit spectral-ish norm. Operates in float32 (CPU/MPS-safe).

    ``G: (m, n) -> (m, n)``
    """
    assert G.ndim == 2, f"Newton-Schulz expects a 2D matrix, got {G.ndim}D"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:  # iterate on the thinner orientation for stability
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.t()
    return X.to(G.dtype)


class Muon(Optimizer):
    r"""Muon for parameters with ``ndim >= 2`` (others should use AdamW).

    Args:
        params: iterable of 2D+ parameters.
        lr: learning rate.
        momentum: heavy-ball momentum coefficient.
        nesterov: use Nesterov-style lookahead on the momentum.
        ns_steps: Newton–Schulz iteration count (5 is plenty).
        weight_decay: decoupled (AdamW-style) weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # flatten conv filters (out, in, kh, kw) -> (out, in*kh*kw)
                g2d = g.reshape(g.size(0), -1) if g.ndim > 2 else g

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g2d)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g2d)
                update = g2d.add(buf, alpha=mom) if nesterov else buf

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # rescale so the update RMS roughly matches an SGD step
                scale = max(1.0, g2d.size(0) / g2d.size(1)) ** 0.5

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.reshape_as(p), alpha=-lr * scale)

        return loss


def build_muon_adamw(
    model: torch.nn.Module,
    *,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
) -> tuple[Optimizer | None, Optimizer | None, dict[str, int]]:
    r"""Split a model's params: ``ndim >= 2`` -> Muon, ``ndim < 2`` -> AdamW.

    Returns ``(muon, adamw, counts)`` where ``counts`` records how many
    parameters went to each optimiser (useful to log the routing for the A/B).
    Either optimiser may be ``None`` if its group is empty.
    """
    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (muon_params if p.ndim >= 2 else adamw_params).append(p)

    muon = (
        Muon(muon_params, lr=muon_lr, momentum=momentum, weight_decay=weight_decay)
        if muon_params
        else None
    )
    adamw = (
        torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)
        if adamw_params
        else None
    )
    counts = {
        "muon_tensors": len(muon_params),
        "adamw_tensors": len(adamw_params),
        "muon_params": sum(p.numel() for p in muon_params),
        "adamw_params": sum(p.numel() for p in adamw_params),
    }
    return muon, adamw, counts
