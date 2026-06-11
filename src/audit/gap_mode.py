r"""Gap-mode audit: $(\hat\delta, \hat\lambda_1, \hat\epsilon_{\max})$ of a trained latent world
model along the demonstration distribution (paper3 / P4-step3).

The certified subgoal-spacing curve consumes exactly three measured quantities
(spec: docs/specs/2026-06-10-p4-step3-gap-mode-seed.md):

$$ \widehat{\mathrm{Err}}(H) \;=\; \hat\delta \sum_{t=0}^{H-1} e^{\hat\lambda_1 t}
   \;\;(+\; m\,\hat\epsilon_{\max} \text{ for group-transported claims, Thm B}), $$

- $\hat\delta$ — one-step bias $\lVert f(E(o_t), a_t) - E(o_{t+1}) \rVert$ over held-out demo
  transitions (raw latent norm: the same metric the certificate consumes);
- $\hat\lambda_1$ — leading finite-time exponent of the **action-conditioned model loop**
  $z_{s+1} = f(z_s, a_s)$, Jacobians along the model's own rollout from encoded demo states with
  recorded actions (the rollout the planner actually uses), Benettin/QR with forward-mode JVP
  **through the predictor only** (the encoder never enters the loop Jacobian — the e2cnn
  forward-AD concern from the seed spec is moot by construction);
- $\hat\epsilon_{\max}$ — per-generator encoder equivariance residual
  $\lVert E(g\cdot x) - \rho(g) E(x) \rVert$, **eq bases only**: for plain/aug bases no canonical
  $\rho$ exists (paper2 Lemma 2) and the field is ``None`` — never a fake number.

Statistics: episode-level bootstrap for $\hat\delta$; window-level bootstrap for $\hat\lambda_1$
(the Prop 10 estimator's moving-block spirit; mixing assumptions inherited from Prop 7, assumed
not certified — stated, as everywhere in the line). float64 CPU throughout (audit convention).
"""

from __future__ import annotations

import copy

import numpy as np
import torch


# --------------------------------------------------------------------------- delta (one-step bias)
def _model_dtype(model) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    for b in model.buffers():
        return b.dtype
    return torch.float32


@torch.no_grad()
def one_step_bias(model, frames: torch.Tensor, actions: torch.Tensor) -> np.ndarray:
    r"""Per-transition residuals: frames (E, T+1, C, H, W) in [0,1], actions (E, T, A).

    Runs in the model's NATIVE dtype (e2cnn's R2Conv cannot be wholesale ``.double()``-ed — its
    expanded filter stays f32 ⇒ dtype mismatch; and δ̂ at $\sim10^{-1}$ scale is indifferent to
    f32 vs f64). Returns residual norms shaped (E, T).
    """
    dt = _model_dtype(model)
    E, T = actions.shape[0], actions.shape[1]
    z = model.encode(frames.to(dt).reshape(-1, *frames.shape[2:])).reshape(E, T + 1, -1)
    pred = model.predictor(z[:, :-1].reshape(E * T, -1), actions.to(dt).reshape(E * T, -1))
    res = (pred.reshape(E, T, -1) - z[:, 1:]).norm(dim=-1)
    return res.double().cpu().numpy()


# --------------------------------------------------------------------------- lambda1 (Benettin/JVP)
def window_exponent(pred_fn, z0: torch.Tensor, acts: torch.Tensor, k: int = 4,
                    burn_in: int = 10) -> float:
    r"""Leading finite-time exponent over one window: model rollout from ``z0`` (D,) under
    recorded ``acts`` (W, A); Benettin QR on a k-band propagated by JVP of $f(\cdot, a_s)$.

    ``burn_in`` steps propagate the band (and the state) WITHOUT accumulating — the standard
    Benettin transient discard: a random initial band needs $\sim 1/(\lambda_1-\lambda_2)$ steps to
    align with the leading direction, and counting those steps biases the average toward the
    sub-leading spectrum (caught by gate G-I on a planted system: $-0.05$ read as $-0.076$ without
    burn-in). The residual finite-window bias after burn-in is a *known instrument bias* of order
    $e^{-2(\lambda_1-\lambda_2) B}/W'$ — the gap-mode analogue of Prop 8's finite-$T$ truncation;
    G-I measures it at deployed settings and the tolerance is registered there.
    """
    D = z0.shape[0]
    W = acts.shape[0]
    assert W > burn_in + 2, f"window {W} too short for burn_in {burn_in}"
    q = torch.linalg.qr(torch.randn(D, k, dtype=z0.dtype, generator=torch.Generator().manual_seed(0)))[0]
    z = z0.clone()
    log_r1, n_acc = 0.0, 0
    for s in range(W):
        a = acts[s]
        f_s = lambda zz: pred_fn(zz.unsqueeze(0), a.unsqueeze(0)).squeeze(0)  # noqa: E731
        cols = []
        for j in range(k):
            _, jv = torch.func.jvp(f_s, (z,), (q[:, j],))
            cols.append(jv)
        m = torch.stack(cols, dim=1)  # (D, k) = J @ q
        q, r = torch.linalg.qr(m)
        if s >= burn_in:
            log_r1 += float(torch.log(torch.abs(r[0, 0])))
            n_acc += 1
        with torch.no_grad():
            z = f_s(z)
    return log_r1 / n_acc


def loop_exponents(model, frames: torch.Tensor, actions: torch.Tensor, *, window: int = 40,
                   k: int = 4, burn_in: int = 10, starts: tuple[int, ...] = (0, 30)) -> np.ndarray:
    r"""Per-(episode, start) window exponents along the demo distribution.

    The loop Jacobian chain runs in **float64 on a deep-copied predictor** (QR precision); the
    encoder contributes only the window's initial state, in its native dtype. A non-Module
    ``model.predictor`` (test stubs, planted systems) is used as-is and must accept f64.
    """
    dt = _model_dtype(model)
    pred = model.predictor
    if isinstance(pred, torch.nn.Module):
        pred64 = copy.deepcopy(pred).double()
        for p in pred64.parameters():  # inference-only: kills graph buildup through QR steps
            p.requires_grad_(False)
    else:
        pred64 = pred

    E, T = actions.shape[0], actions.shape[1]
    out = []
    with torch.no_grad():
        z_all = model.encode(frames[:, 0].to(dt))  # (E, D) — windows start from encoded demo states
    for e in range(E):
        for t0 in starts:
            if t0 + window > T:
                continue
            z0 = (model.encode(frames[e, t0].to(dt).unsqueeze(0)).squeeze(0) if t0 > 0 else z_all[e]).double()
            lam = window_exponent(pred64, z0, actions[e, t0 : t0 + window].double(), k=k, burn_in=burn_in)
            out.append(lam)
    return np.array(out)


# --------------------------------------------------------------------------- eps_max (eq only)
@torch.no_grad()
def equivariance_residual(model, frames: torch.Tensor, n_sample: int = 32) -> dict | None:
    r"""Per-generator $\lVert E(g x) - \rho(g) E(x) \rVert$ via e2cnn transforms; None if the
    encoder exposes no geometric interface (plain/aug — Lemma 2, no canonical $\rho$)."""
    enc = model.encoder
    if not hasattr(enc, "encode_geometric") or not hasattr(enc, "in_type"):
        return None
    from e2cnn import nn as enn

    x = frames.reshape(-1, *frames.shape[2:])[:n_sample].float()
    gx = enn.GeometricTensor(x, enc.in_type)
    base = enc.encode_geometric(x)
    per_g_mean, per_g_max = {}, {}
    for g in enc.r2.testing_elements:
        gi = int(g)
        if gi == 0:
            continue
        left = enc.encode_geometric(gx.transform(g).tensor)
        right = base.transform(g)
        norms = (left.tensor - right.tensor).norm(dim=1)
        per_g_mean[gi] = float(norms.mean())
        per_g_max[gi] = float(norms.max())
    n = len(per_g_mean) + 1
    grid = [gi for gi in per_g_mean if (gi * 4) % n == 0]  # C_4 subgroup (pixel-exact angles)
    return {
        # Thm B's eps_max is a SUP — "max" here is the empirical sup over samples & generators;
        # the mean is reported for context, never substituted into the m*eps_max term.
        "per_generator_mean": per_g_mean,
        "per_generator_max": per_g_max,
        "max": max(per_g_max.values()),
        "mean_over_generators": float(np.mean(list(per_g_mean.values()))),
        "grid_subgroup_max": max((per_g_max[g] for g in grid), default=None),
        "offgrid_max": max((v for g, v in per_g_max.items() if g not in grid), default=None),
    }


# --------------------------------------------------------------------------- bootstrap + assembly
def _boot_ci(values: np.ndarray, n_boot: int, seed: int, agg=np.mean) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([agg(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def audit_gap(model, frames: torch.Tensor, actions: torch.Tensor, *, window: int = 40, k: int = 4,
              burn_in: int = 10, n_boot: int = 1000, h_max: int = 60, seed: int = 0) -> dict:
    r"""The instrument. frames (E, T+1, C, H, W) float64 in [0,1]; actions (E, T, A) float64.

    Returns the measured triple + the certified curve $\widehat{\mathrm{Err}}(H)$ on
    $H \in \{1..h_{\max}\}$. CIs: episode-bootstrap ($\hat\delta$), window-bootstrap
    ($\hat\lambda_1$).
    """
    model = model.eval()  # NATIVE dtype for the encoder; f64 enters via the predictor copy
    actions = actions.double()

    res = one_step_bias(model, frames, actions)  # (E, T)
    ep_means = res.mean(axis=1)
    d_lo, d_hi = _boot_ci(ep_means, n_boot, seed)
    delta = {
        "mean": float(res.mean()), "q50": float(np.median(res)), "q90": float(np.quantile(res, 0.9)),
        "ci_lo": d_lo, "ci_hi": d_hi, "n_transitions": int(res.size),
    }

    lams = loop_exponents(model, frames, actions, window=window, k=k, burn_in=burn_in)
    l_lo, l_hi = _boot_ci(lams, n_boot, seed + 1)
    lam1 = {
        "mean": float(lams.mean()), "median": float(np.median(lams)),
        "ci_lo": l_lo, "ci_hi": l_hi, "n_windows": int(len(lams)), "window": window, "k_band": k,
        "burn_in": burn_in,
    }

    eps = equivariance_residual(model, frames)

    h = np.arange(1, h_max + 1)

    def geo(d: float, lam: float) -> np.ndarray:
        return d * (np.exp(lam * h) - 1) / (np.exp(lam) - 1) if abs(lam) > 1e-9 else d * h.astype(float)

    lam_m = lam1["mean"]
    return {
        "delta": delta, "lambda1": lam1, "eps_max": eps,
        # Two registered curves: mean-delta = typical-case predictor; q90-delta = the
        # certificate-flavoured bound (Thm B's delta is a sup; q90 is its honest empirical proxy —
        # which one a downstream gate consumes is frozen in that gate's spec, not here).
        "certified_curve": {
            "H": h.tolist(),
            "err_mean": geo(delta["mean"], lam_m).tolist(),
            "err_q90": geo(delta["q90"], lam_m).tolist(),
            "err_ci_lo": geo(delta["ci_lo"], l_lo).tolist(),
            "err_ci_hi": geo(delta["ci_hi"], l_hi).tolist(),
        },
    }
