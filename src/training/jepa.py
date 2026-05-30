r"""JEPA training utilities: data collection, EMA-target training, FoV A/B eval.

This is the **Step 4** machinery. It trains an :class:`~src.models.eqjepa.EqJEPA`
with a canonical JEPA objective and then measures the property the project is
betting on — robustness of the *planning cost* to global rotations of the scene.

Training objective (V-JEPA / BYOL-style, to avoid representation collapse):

$$ \mathcal{L} = \big\lVert f_\phi\!\big(E_\theta(o_t), a_t\big) - \mathrm{sg}\,E_{\bar\theta}(o_{t+1}) \big\rVert_2^2
   \; + \; \lambda\,\tfrac{1}{D}\sum_d \mathrm{ReLU}\!\big(1 - \mathrm{std}_b\, z^{(d)}\big) $$

where $E_{\bar\theta}$ is an exponential-moving-average ("target") copy of the
online encoder with a stop-gradient, and the second term is a light VICReg
variance hinge that keeps the latent from collapsing on small data.

FoV A/B metric (the thesis test). The CEM planning cost is
$\mathcal{C} = \lVert E_\theta(o) - E_\theta(o_g)\rVert$. Rotating the *entire*
scene (state and goal together) by $g\in C_4$ should leave $\mathcal{C}$
unchanged for an SO(2)-equivariant encoder, because
$\lVert E(g o) - E(g o_g)\rVert = \lVert \rho(g)\,(E(o)-E(o_g)) \rVert = \lVert E(o)-E(o_g)\rVert$
($\rho(g)$ orthogonal). For an ordinary CNN there is no such guarantee. We report
the relative drift of $\mathcal{C}$ under $90^\circ$ rotations for both encoders.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from .muon import build_muon_adamw


@torch.no_grad()
def collect_transitions(
    world,
    n_transitions: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Roll out random actions and gather $(o_t, a_t, o_{t+1})$ transitions.

    Steps the vectorised envs directly with uniform-random actions, resetting
    any env that finishes and **dropping** the boundary-crossing transition.
    Images are returned channel-first in $[0,1]$.

    Returns ``(obs, act, nxt)`` with shapes ``(N, C, H, W)``, ``(N, A)``,
    ``(N, C, H, W)``.
    """
    rng = np.random.default_rng(seed)
    world.reset(seed=seed)
    act_space = world.envs.single_action_space
    low, high = act_space.low, act_space.high
    a_dim = act_space.shape[-1]
    n_envs = world.num_envs

    obs_buf, act_buf, nxt_buf = [], [], []
    while len(obs_buf) < n_transitions:
        pixels_before = np.asarray(world.infos["pixels"]).copy()  # (E,H,W,C) uint8
        a = rng.uniform(low, high, size=(n_envs, a_dim)).astype(np.float32)
        _, _, term, trunc, infos = world.envs.step(a, mask=None)
        world.infos = infos
        pixels_after = np.asarray(world.infos["pixels"]).copy()
        done = np.asarray(term) | np.asarray(trunc)

        # infos carry a leading time dim of 1: (E, T, H, W, C); take last frame.
        for i in range(n_envs):
            if not done[i]:
                obs_buf.append(pixels_before[i, -1])  # (H, W, C)
                act_buf.append(a[i])
                nxt_buf.append(pixels_after[i, -1])

        if done.any():
            _, world.infos = world.envs.reset(mask=done)

    def to_chw(frames: list) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames[:n_transitions]))  # (N,H,W,C) uint8
        return x.permute(0, 3, 1, 2).float().div_(255.0).contiguous()

    obs = to_chw(obs_buf)
    nxt = to_chw(nxt_buf)
    act = torch.from_numpy(np.stack(act_buf[:n_transitions])).float()
    return obs, act, nxt


def train_jepa(
    model,
    obs: torch.Tensor,
    act: torch.Tensor,
    nxt: torch.Tensor,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    ema_decay: float = 0.99,
    var_coef: float = 0.04,
    weight_decay: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 5,
    verbose: bool = True,
) -> dict:
    r"""Train ``model`` (an :class:`EqJEPA`) with EMA-target JEPA + Muon/AdamW.

    Returns a history dict with per-epoch ``pred_loss``, ``var_loss`` and the
    final batch latent standard deviation ``latent_std`` (a collapse witness:
    a healthy run keeps this well above 0).
    """
    torch.manual_seed(seed)
    model.to(device).train()

    # EMA "target" encoder: a stop-grad copy of the online encoder.
    # (An EMA of equivariant weights is still equivariant, so this is safe for
    # the steerable encoder.)
    target_enc = copy.deepcopy(model.encoder).to(device)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    target_enc.eval()

    muon, adamw, counts = build_muon_adamw(
        model, muon_lr=muon_lr, adamw_lr=adamw_lr, weight_decay=weight_decay
    )
    if verbose:
        print(f"    optim routing: {counts}")

    n = obs.shape[0]
    history = {"pred_loss": [], "var_loss": []}
    last_std = float("nan")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_pred, ep_var, nb = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            o = obs[idx].to(device)
            a = act[idx].to(device)
            o2 = nxt[idx].to(device)

            z0 = model.encoder(o)  # online (B, D)
            with torch.no_grad():
                z_tgt = target_enc(o2)  # EMA target next-latent (stop-grad)
            z_pred = model.predictor(z0, a)  # (B, D)

            pred_loss = F.mse_loss(z_pred, z_tgt)
            std = z0.std(dim=0)  # per-dim batch std
            var_loss = F.relu(1.0 - std).mean()
            loss = pred_loss + var_coef * var_loss

            if muon is not None:
                muon.zero_grad(set_to_none=True)
            if adamw is not None:
                adamw.zero_grad(set_to_none=True)
            loss.backward()
            if muon is not None:
                muon.step()
            if adamw is not None:
                adamw.step()

            # EMA update of the target encoder. Match by *name* (positional zip
            # is unsafe for e2cnn, which lazily rebuilds expanded-filter buffers).
            with torch.no_grad():
                online_p = dict(model.encoder.named_parameters())
                for name, pt in target_enc.named_parameters():
                    pt.mul_(ema_decay).add_(online_p[name], alpha=1.0 - ema_decay)
                online_b = dict(model.encoder.named_buffers())
                for name, bt in target_enc.named_buffers():
                    bs = online_b.get(name)
                    if bs is not None and bt.shape == bs.shape:  # BN running stats
                        bt.copy_(bs)

            ep_pred += pred_loss.item()
            ep_var += var_loss.item()
            last_std = std.mean().item()
            nb += 1

        history["pred_loss"].append(ep_pred / max(nb, 1))
        history["var_loss"].append(ep_var / max(nb, 1))
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"    epoch {epoch:3d}  pred={history['pred_loss'][-1]:.4f}  "
                f"var={history['var_loss'][-1]:.4f}  latent_std={last_std:.4f}"
            )

    model.eval()
    history["latent_std"] = last_std
    history["routing"] = counts
    return history


@torch.no_grad()
def fov_cost_drift(
    model,
    obs: torch.Tensor,
    *,
    device: str = "cpu",
    n_pairs: int = 256,
    seed: int = 0,
) -> dict:
    r"""Relative drift of the JEPA goal-matching cost under $90^\circ$ rotations.

    Builds ``n_pairs`` (state, goal) pairs from held-out frames, computes the
    base cost $\mathcal{C} = \lVert E(o) - E(o_g)\rVert$, then re-computes it with
    *both* frames rotated by $k\cdot 90^\circ$ (exact pixel permutation via
    :func:`torch.rot90`). Returns mean relative drift per rotation:

    $$ \mathrm{drift}(k) = \frac{\mathbb{E}\big|\,\mathcal{C}(R^k o, R^k o_g) - \mathcal{C}(o, o_g)\,\big|}{\mathbb{E}\,\mathcal{C}(o, o_g)}. $$

    An SO(2)-equivariant encoder should give ``drift ~ 0``; an ordinary CNN drifts.
    """
    model.to(device).eval()
    g = torch.Generator().manual_seed(seed)
    n = obs.shape[0]
    ia = torch.randint(0, n, (n_pairs,), generator=g)
    ib = torch.randint(0, n, (n_pairs,), generator=g)
    oa = obs[ia].to(device)
    ob = obs[ib].to(device)

    def cost(xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        return (model.encoder(xa) - model.encoder(xb)).norm(dim=-1)  # (n_pairs,)

    base = cost(oa, ob)
    base_mean = base.mean().clamp_min(1e-8)
    drift = {0: 0.0}
    for k in (1, 2, 3):
        ra = torch.rot90(oa, k, dims=(-2, -1))
        rb = torch.rot90(ob, k, dims=(-2, -1))
        rot = cost(ra, rb)
        drift[k * 90] = ((rot - base).abs().mean() / base_mean).item()
    return {"base_cost": base.mean().item(), "drift": drift}
