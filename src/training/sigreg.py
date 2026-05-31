r"""SIGReg — Sketched Isotropic Gaussian Regularisation, and its **equivariant**
refinement *block-SIGReg* (Direction-1 experiment, task #83).

Background (LeJEPA; Balestriero & LeCun, arXiv:2511.08544)
---------------------------------------------------------
LeJEPA drives an embedding distribution toward the isotropic Gaussian
$\mathcal N(\mathbf 0,\mathbf I_d)$ — the task-agnostic optimum of their Thm 1 —
with a single regulariser. It *sketches* a 1-D normality test along random
directions $\mathbf a$ and averages:

$$\mathrm{SIGReg}_T\big(\mathbb A,\{z_n\}\big)
   = \frac1{|\mathbb A|}\sum_{\mathbf a\in\mathbb A}
     T\big(\{\mathbf a^\top z_n\}_{n=1}^N\big),$$

with the recommended test $T$ = **Epps–Pulley**: a weighted $L^2$ distance between
the empirical characteristic function $\hat\phi_X(t)=\tfrac1N\sum_j e^{it X_j}$ and
the standard-Gaussian CF $\phi(t)=e^{-t^2/2}$, weight $w(t)=e^{-t^2/\sigma^2}$.

Why an *equivariant* refinement is needed (this project's contribution, C1)
---------------------------------------------------------------------------
For a $G$-equivariant latent (law invariant under an orthogonal representation
$\rho:G\to O(n)$), **Proposition 1** (Schur block-isotropy, see
``papers/equivariant_lejepa.md`` §3) forces the covariance to be *block-isotropic*

$$\Sigma=\bigoplus_i \mathbf I_{d_i}\otimes B_i$$

over the real isotypic components $V_i^{\oplus m_i}$ ($d_i=\dim V_i$, multiplicity
$m_i$). LeJEPA's isotropic target $\Sigma=\sigma^2\mathbf I_n$ is the special case
$B_i=\sigma^2\mathbf I_{m_i}$ — it forces **the same scale on every irrep**, a
measure-zero slice of the $G$-invariant Gaussians and one with no physical reason
(a type-0 scalar and a type-1 vector need not carry equal variance). Worse, equal
scales make $\Sigma$'s spectrum degenerate, so its eigenspaces do **not** expose
the isotypic blocks and the residual gauge stays as large as $O(n)$.

**Block-SIGReg** instead Gaussianises *the shape within each isotypic block at the
block's own scale*, and constrains only the **total** variance budget
$\tfrac1n\operatorname{tr}\hat\Sigma\approx 1$ (LeJEPA's "fixed scalar covariance
budget", reallocated across irreps rather than forced equal). The per-block scales
$\sigma_i^2$ are then free; when they come out distinct, $\Sigma$'s spectrum
separates the blocks and the gauge drops from $O(n)$ to the orthogonal commutant
$\prod_i O(m_i)$ (finite — per-irrep sign flips — when multiplicity-free).

This module provides both objectives so an experiment can compare them on the same
equivariant encoder. Everything is differentiable and ``O(N · K · |A|)`` (linear in
the batch, as in LeJEPA).
"""

from __future__ import annotations

import torch

from ..geometry.irreps import IrrepBlock

__all__ = [
    "IrrepBlock",  # re-exported from src.geometry.irreps for callers' convenience
    "epps_pulley_1d",
    "sigreg_isotropic",
    "sigreg_block",
    "variance_budget_penalty",
    "block_scales",
]


def epps_pulley_1d(
    x: torch.Tensor,
    *,
    t_grid: torch.Tensor,
    weight_sigma: float = 1.4142135623730951,  # sqrt(2): w(t)=e^{-t^2/2}
) -> torch.Tensor:
    r"""Epps–Pulley normality statistic of a 1-D sample against $\mathcal N(0,1)$.

    $$ T_{\mathrm{EP}} = \int \big|\hat\phi_X(t)-\phi(t)\big|^2\, w(t)\,dt,
       \quad \hat\phi_X(t)=\tfrac1N\sum_j e^{itx_j},\ \phi(t)=e^{-t^2/2},\
       w(t)=e^{-t^2/\sigma^2}, $$

    discretised on ``t_grid`` (trapezoid-free Riemann sum — the integrand is smooth
    and Gaussian-tailed, so a modest symmetric grid is exact to far below the float
    floor that matters here). With $\phi$ real,
    $|\hat\phi-\phi|^2=(\overline{\cos}-\phi)^2+\overline{\sin}^2$ where
    $\overline{\cos}(t)=\tfrac1N\sum_j\cos(tx_j)$, $\overline{\sin}(t)=\tfrac1N\sum_j\sin(tx_j)$.
    The statistic is $0$ iff the empirical CF matches the standard Gaussian's and is
    differentiable in ``x`` (hence trainable). The sample is expected to be roughly
    standardised; callers standardise per block as appropriate.

    Args:
        x: 1-D sample, shape ``(N,)`` (or ``(..., N)`` — the last axis is the sample).
        t_grid: frequencies $t_k$, shape ``(K,)``, symmetric about 0 recommended.
        weight_sigma: $\sigma$ in $w(t)=e^{-t^2/\sigma^2}$ (default $\sqrt2$).

    Returns:
        Scalar (or ``(...)``-shaped) Epps–Pulley statistic, ``>= 0``.
    """
    # x: (..., N), t_grid: (K,) -> phases (..., K, N)
    tx = t_grid[..., :, None] * x[..., None, :]  # broadcast: (..., K, N)
    cos_mean = tx.cos().mean(dim=-1)  # (..., K)  Re empirical CF
    sin_mean = tx.sin().mean(dim=-1)  # (..., K)  Im empirical CF
    phi = torch.exp(-0.5 * t_grid**2)  # (K,) standard-normal CF (real)
    w = torch.exp(-(t_grid**2) / weight_sigma**2)  # (K,) weight
    integrand = ((cos_mean - phi) ** 2 + sin_mean**2) * w  # (..., K)
    dt = (t_grid[-1] - t_grid[0]) / (t_grid.numel() - 1)
    return integrand.sum(dim=-1) * dt


def _default_t_grid(device: torch.device, n_t: int = 33, t_max: float = 6.0) -> torch.Tensor:
    return torch.linspace(-t_max, t_max, n_t, device=device)


def sigreg_isotropic(
    z: torch.Tensor,
    *,
    n_proj: int = 64,
    generator: torch.Generator | None = None,
    t_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""**Vanilla** LeJEPA SIGReg: push the *whole* latent to $\mathcal N(\mathbf 0,\mathbf I_n)$.

    Sketches ``n_proj`` random **unit** directions $\mathbf a\in\mathbb R^n$, projects
    $p=\mathbf a^\top z$ (variance $\mathbf a^\top\Sigma\mathbf a$), and tests each $p$
    against $\mathcal N(0,1)$ with :func:`epps_pulley_1d`. Driving every projection to
    a *standard* Gaussian forces $\Sigma\to\mathbf I_n$ — i.e. **unit scale on every
    direction, hence equal scale across all irreps** (the behaviour Prop. 1 says is
    over-constrained for an equivariant latent).

    ``z: (N, n) -> ()`` (scalar regulariser).
    """
    n = z.shape[1]
    device = z.device
    if t_grid is None:
        t_grid = _default_t_grid(device)
    a = torch.randn(n_proj, n, device=device, generator=generator)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)  # unit directions
    proj = z @ a.T  # (N, n_proj): each column is a 1-D projection sample
    ep = epps_pulley_1d(proj.T, t_grid=t_grid)  # (n_proj,)
    return ep.mean()


def block_scales(z: torch.Tensor, blocks: list[IrrepBlock]) -> dict[str, float]:
    r"""Per-block mean *per-dimension* variance $\sigma_i^2$ (a witness, no grad).

    Reports $\sigma_i^2=\tfrac1{d_im_i}\operatorname{tr}\operatorname{Cov}(z|_{\text{block }i})$
    so the caller can see whether the scales come out distinct (block-SIGReg) or are
    forced equal (vanilla). ``z: (N, n) -> {name: var}``.
    """
    out: dict[str, float] = {}
    with torch.no_grad():
        zc = z - z.mean(dim=0, keepdim=True)
        for b in blocks:
            sub = b.slice(zc)  # (N, d*m)
            out[b.name] = float((sub**2).mean().item())  # mean per-dim variance
    return out


def sigreg_block(
    z: torch.Tensor,
    blocks: list[IrrepBlock],
    *,
    n_proj: int = 64,
    generator: torch.Generator | None = None,
    t_grid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""**Block-SIGReg**: Gaussianise each isotypic block *at its own (free) scale*.

    For each block $i$ we standardise the sub-latent by the block's own RMS scale
    $s_i=\sqrt{\tfrac1{d_im_i}\sum\operatorname{var}}$ (**detached**, so the test
    constrains *shape only*, not scale), sketch ``n_proj`` random unit directions
    *within the block*, and Epps–Pulley-test the standardised projections against
    $\mathcal N(0,1)$. Averaging over blocks gives a shape regulariser that is
    satisfied by *any* block-isotropic Gaussian, regardless of the per-block scales
    $\sigma_i^2$ — unlike :func:`sigreg_isotropic`, which would force them equal.

    Scale is *not* pinned here; pair with :func:`variance_budget_penalty` to fix the
    **total** budget (LeJEPA's scalar-covariance budget, now reallocatable across
    irreps). The per-block split is then set by the task, and comes out distinct when
    the irreps carry different amounts of signal — which is what separates $\Sigma$'s
    spectrum and pins the gauge to the commutant (see module docstring).

    ``z: (N, n), blocks -> ()`` (scalar regulariser, mean over blocks).
    """
    device = z.device
    if t_grid is None:
        t_grid = _default_t_grid(device)
    zc = z - z.mean(dim=0, keepdim=True)
    terms = []
    for b in blocks:
        sub = b.slice(zc)  # (N, d*m)
        scale = sub.detach().pow(2).mean().clamp_min(eps).sqrt()  # detached RMS, shape-only
        sub_std = sub / scale
        a = torch.randn(n_proj, b.width, device=device, generator=generator)
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
        proj = sub_std @ a.T  # (N, n_proj)
        terms.append(epps_pulley_1d(proj.T, t_grid=t_grid).mean())
    return torch.stack(terms).mean()


def variance_budget_penalty(z: torch.Tensor, *, target: float = 1.0) -> torch.Tensor:
    r"""Keep the **total** variance budget near ``target`` per dimension (anti-collapse).

    $\big(\tfrac1n\operatorname{tr}\hat\Sigma-\text{target}\big)^2$. Used with
    :func:`sigreg_block`: it fixes the global scale (preventing the shape-only
    block test from collapsing or exploding the latent) while leaving the *split*
    of variance across irreps free. ``z: (N, n) -> ()``.
    """
    zc = z - z.mean(dim=0, keepdim=True)
    per_dim_var = zc.pow(2).mean()  # (1/n) tr Sigma-hat (mean over dims and samples)
    return (per_dim_var - target) ** 2
