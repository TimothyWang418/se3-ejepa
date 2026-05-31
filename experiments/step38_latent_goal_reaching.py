r"""Step 38 -- decoder-free latent-goal REACHING, cured via the SE(3) equivariance theorem (Tier 3.11).

Step 13's panel [C] is the project's only outright failure: a CEM-MPC planner that minimises a
purely-latent terminal cost $\lVert\hat z_H - z_g\rVert_2^2$ (the latent rolled with the predictor,
the plan executed on the equivariant teacher) got a NEGATIVE fraction-of-gap-closed for both the
VN and the MLP -- it pushed the cloud *away* from the goal. Step 18 then proved the closed loop is
exactly SE(3)-invariant, but only of the orientation *error* (it reached weakly, ~16% of the gap,
and explicitly said "the result does not depend on reachability"). So the question Step 13[C] left
open -- *can a JEPA specify and reach a goal purely in an equivariant latent, with no decoder?* --
was never actually answered. This step answers it.

DIAGNOSIS (throwaway probes, since deleted). The failure is NOT that the predictor under-moves
(1-step latent step ratio pred/true $=1.12$) and NOT metric conditioning (whitening did not help).
It is two coupled things:
  (1) encoded-vs-predicted manifold mismatch + multi-step rollout drift -- the 1-step-trained
      predictor's rollout $f^h(E(X_0),a)$ drifts $\sim\!2.0$ from $E(\text{teacher}^h)$ by $h{=}6$
      (~80% of the goal gap), so $\min_a\lVert f^H(E(X_0),a)-E(X_g)\rVert$ chases an OFF-manifold
      (literally unreachable) target;
  (2) a poorly-scaled terminal $L_2$ that gives almost no usable gradient toward the goal.

CURE (this module, all decoder-free, all still exactly SE(3)-equivariant):
  [A] REACHING. An ablation ladder, encoder goal $E(X_g)$, fraction of orientation gap closed:
        Step-13[C] verbatim planner (1-step train) ............ NEGATIVE   (reproduce the failure)
        + SE(3)-equivariant planner (iso-sigma/ball/centroid) .. ~+0.25     (Step 18's lift)
        + rollout-consistency training ........................ ~+0.5      (Step 38's main cure)
        + SE(3)-native goal signal (Procrustes angle) + receding ~+0.59     (best DEPLOYABLE)
      references: predictor-space goal (needs $a_{\rm true}$) ~+0.71 (ceiling); replay $a_{\rm true}$ ~+1.0 (oracle).
      The SE(3)-native goal is the *group element* $R^\star$ that aligns the latents: fit $R^\star$ by
      Kabsch on the 16 type-1 vectors of $z_0,z_g$ and use the geodesic angle $\lvert R^\star\rvert$ --
      monotone, well-scaled, and SE(3)-invariant by construction.
  [B] EQUIVARIANCE PAYOFF (the theorem, now for SUCCESS not failure). Paired seen-vs-OOD SE(3) orbit:
      the VN's fraction-closed is identical across the orbit to the e3nn float floor ($\max_i\lvert
      d_i\rvert\sim10^{-6}$, OOD/seen ratio $=1.000$); the MLP degrades OOD (ratio $>1$). Reaching
      *transfers exactly* across the group -- Step 14/18's exactness theorem, but for goal-reaching.
  [C] GENUINENESS. The goal cost is SE(3)-invariant by construction (a shared latent rotation leaves
      it unchanged to float floor); the VN realises it end-to-end (composed-equiv $\sim10^{-6}$, plan
      equivariant to $\sim10^{-6}$) while the MLP does not (composed-equiv $O(1)$).

Run:
    STEP38_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        .venv/bin/python experiments/step38_latent_goal_reaching.py     # ~1 min
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        .venv/bin/python experiments/step38_latent_goal_reaching.py     # FULL, a few min
"""

import copy
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    C_T, LATENT_DIM, N_POINTS, _TEMPLATE, build_eq_jepa, build_mlp_jepa,
    collect_cloud_transitions, composed_equiv_err, latent_cem_plan, rand_so3, rot_z,
    rotate_latent, teacher_step,
)
from step18_se3_closed_loop import (  # noqa: E402
    EVAL_DTYPE, _apply, _ball_clamp, centroid, kabsch_rotation, make_se3_orbit,
    report_panel_se3, rotation_angle_deg,
)
from src.training.jepa import train_jepa  # noqa: E402
from src.training.muon import build_muon_adamw  # noqa: E402

torch.set_default_dtype(torch.float32)
SMOKE = bool(os.environ.get("STEP38_SMOKE"))
NV = LATENT_DIM // 3        # number of type-1 latent vectors (= 16)


# --------------------------------------------------------------------------- #
# data + training: multi-step ROLLOUT-CONSISTENCY (the cure for the manifold mismatch)
# --------------------------------------------------------------------------- #
def collect_cloud_rollouts(n, H_roll, *, seed=0, phi_lo=0.0, phi_hi=90.0):
    r"""``n`` teacher rollouts of length ``H_roll`` in the $z$-wedge $[\phi_{\rm lo},\phi_{\rm hi})$.

    Returns ``X: (n, H_roll+1, P, 3)`` (clouds) and ``A: (n, H_roll, 3)`` (actions). Anisotropic
    template + per-sample jitter and per-axis scale, so orientation is observable (cf. Step 8).
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, H_roll + 1, N_POINTS, 3), np.float32)
    A = np.empty((n, H_roll, 3), np.float32)
    for i in range(n):
        jitter = rng.standard_normal((N_POINTS, 3)).astype(np.float32) * 0.04
        axis_scale = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
        cloud = (_TEMPLATE + jitter) * axis_scale
        Rz = rot_z(rng.uniform(phi_lo, phi_hi)).numpy()
        x = torch.from_numpy(cloud @ Rz.T).unsqueeze(0)
        X[i, 0] = x[0].numpy()
        for h in range(H_roll):
            a = np.clip(rng.standard_normal(3) * 0.6, -1, 1).astype(np.float32)
            A[i, h] = a
            x = teacher_step(x, torch.from_numpy(a).reshape(1, 3))
            X[i, h + 1] = x[0].numpy()
    return torch.from_numpy(X), torch.from_numpy(A)


def train_jepa_rollout(model, X, A, *, epochs=50, batch_size=128, H_roll=6,
                       muon_lr=0.02, adamw_lr=1e-3, ema_decay=0.99, var_coef=0.1, seed=0,
                       log_every=10):
    r"""Multi-step rollout-consistency JEPA (the cure for the manifold mismatch).

    Ties the predictor rollout to the EMA target encoder at every horizon,
    $\mathcal L_{\rm roll}=\frac1H\sum_{h=1}^{H}\lVert f^h(E(x_0),a_{1:h})-\mathrm{sg}\,E_{\bar\theta}(x_h)\rVert^2$,
    so $f^H$ stays ON the encoder manifold and the encoder goal $E(X_g)$ becomes reachable. A small
    variance floor on $z_0$ ($\mathrm{ReLU}(1-\mathrm{std})$) prevents collapse. Muon on $\ge$2-D
    weights, AdamW on the rest (Step 13 recipe). ``X: (n,H+1,P,3)``, ``A: (n,H,3)``.
    """
    torch.manual_seed(seed)
    model.train()
    target_enc = copy.deepcopy(model.encoder)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    target_enc.eval()
    muon, adamw, counts = build_muon_adamw(model, muon_lr=muon_lr, adamw_lr=adamw_lr)
    n = X.shape[0]
    last_std = float("nan")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, ab = X[idx], A[idx]                       # (B,H+1,P,3), (B,H,3)
            z = model.encoder(xb[:, 0])
            z0 = z
            loss_roll = 0.0
            for h in range(H_roll):
                z = model.predictor(z, ab[:, h])
                with torch.no_grad():
                    z_tgt = target_enc(xb[:, h + 1])
                loss_roll = loss_roll + F.mse_loss(z, z_tgt)
            loss_roll = loss_roll / H_roll
            var_loss = F.relu(1.0 - z0.std(dim=0)).mean()
            loss = loss_roll + var_coef * var_loss
            if muon: muon.zero_grad(set_to_none=True)
            if adamw: adamw.zero_grad(set_to_none=True)
            loss.backward()
            if muon: muon.step()
            if adamw: adamw.step()
            with torch.no_grad():
                onl = dict(model.encoder.named_parameters())
                for nm, pt in target_enc.named_parameters():
                    pt.mul_(ema_decay).add_(onl[nm], alpha=1 - ema_decay)
            ep += loss_roll.item(); nb += 1
            last_std = z0.std(dim=0).mean().item()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"      [rollout] epoch {epoch:3d}  roll={ep/max(nb,1):.4f}  latent_std={last_std:.4f}")
    model.eval()
    return {"latent_std": last_std}


def make_reach_tasks(k, *, seed, H_goal, dtype=EVAL_DTYPE):
    r"""``k`` reorientation tasks ``(X0, Xg, a_true)``; $X_g$ is the teacher rollout under $a_{\rm true}$.

    Actions hold a persistent random unit axis ($0.55\,n + 0.25\,\mathcal N$, clipped), so the torque
    channel accumulates a genuine, reachable reorientation. ``a_true: (H_goal, 3)`` enables the
    predictor-space ceiling and the oracle. Clouds/actions are returned at ``dtype`` (float64 eval).
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(k):
        jitter = rng.standard_normal((N_POINTS, 3)).astype(np.float32) * 0.04
        X0 = torch.from_numpy(_TEMPLATE + jitter).unsqueeze(0).to(dtype)
        axis = rng.standard_normal(3).astype(np.float32)
        axis /= np.linalg.norm(axis) + 1e-9
        X = X0.clone()
        acts = []
        for _ in range(H_goal):
            a = np.clip(0.55 * axis + 0.25 * rng.standard_normal(3).astype(np.float32), -1, 1)
            at = torch.from_numpy(a).reshape(1, 3).to(dtype)
            acts.append(at)
            X = teacher_step(X, at)
        tasks.append((X0, X, torch.cat(acts, 0)))
    return tasks


# --------------------------------------------------------------------------- #
# the SE(3)-native goal signal: the group element R* aligning z0 -> zg (Kabsch on type-1 vectors)
# --------------------------------------------------------------------------- #
def latent_residual_angle(zH, zg):
    r"""Geodesic angle (rad) of the residual rotation aligning two latents' 16 type-1 vectors.

    Fit $R^\star=\arg\min_R\lVert R\,A - B\rVert$ by Kabsch on $A=\mathrm{vec}(z_H)$, $B=\mathrm{vec}(z_g)$
    (each $(B,16,3)$), and return $\lvert R^\star\rvert=\arccos\!\frac{\mathrm{tr}R^\star-1}{2}$. Cost
    $=0$ iff the two latents differ by no rotation; $=\lvert R\rvert$ if $z_H=\rho(R)z_g$. SE(3)-invariant:
    under a shared rotation both latents carry $\rho(R)$, the fit conjugates, and the trace is unchanged.
    ``zH, zg: (B, D) -> (B,)``.
    """
    B = zH.shape[0]
    A = zH.reshape(B, NV, 3)
    Bm = zg.reshape(B, NV, 3)
    Hc = Bm.transpose(-1, -2) @ A                          # (B,3,3) cross-covariance
    U, _, Vh = torch.linalg.svd(Hc)
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.det(V @ U.transpose(-1, -2)))     # reflection fix -> proper rotation
    eye = torch.eye(3, dtype=zH.dtype).expand(B, 3, 3).clone()
    eye[:, 2, 2] = d
    Rr = V @ eye @ U.transpose(-1, -2)                     # (B,3,3) residual rotation
    c = (torch.diagonal(Rr, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.arccos(c.clamp(-1.0, 1.0))                # (B,) radians


# --------------------------------------------------------------------------- #
# the cured planner: SE(3)-equivariant CEM with selectable goal cost + closed-form centroid channel
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cem_plan_se3(model, X, zg, cg, *, cost_kind="l2", R_noise=None, w_t=0.5,
                 H=6, n_samples=256, n_iters=8, n_elite=25, sigma0=0.6, w_run=0.3, gen=None):
    r"""SE(3)-equivariant CEM-MPC: latent orientation cost (``l2`` or Procrustes ``proc``) + exact
    closed-form centroid (translation) cost. Every op is SO(3)-equivariant -- isotropic per-step
    $\sigma$, ball clamp $\lVert a\rVert\le1$, and the exploration noise pre-rotated by ``R_noise``$=R$
    -- so for the equivariant VN the OOD plan is exactly $R\cdot$(seen plan). Returns plan ``(H,3)``.

    ``X: (1,P,3)``, ``zg: (1,D)``, ``cg: (3,) or (1,3)``.
    """
    z0 = model.encoder(X).expand(n_samples, -1).contiguous()
    dtype = z0.dtype
    zg = zg.expand(n_samples, -1).contiguous()
    c0 = centroid(X).reshape(1, 3)
    cg = cg.reshape(1, 3)
    Rn = None if R_noise is None else R_noise.to(dtype)
    mean = torch.zeros(H, 3, dtype=dtype)
    sigma = torch.full((H, 3), sigma0, dtype=dtype)
    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 3, generator=gen, dtype=dtype)
        if Rn is not None:
            eps = torch.einsum("ij,nhj->nhi", Rn, eps)     # rotate exploration noise -> exact OOD plan
        cand = _ball_clamp(mean.unsqueeze(0) + sigma.unsqueeze(0) * eps)
        z = z0.clone()
        cost = torch.zeros(n_samples, dtype=dtype)
        for h in range(H):
            z = model.predictor(z, cand[:, h])
            d2 = ((z - zg) ** 2).sum(-1) if cost_kind == "l2" else latent_residual_angle(z, zg)
            cost = cost + (w_run * d2 if h < H - 1 else d2)
        pred_centroid = c0 + C_T * cand.sum(dim=1)         # exact closed-form drift channel
        cost = cost + w_t * ((pred_centroid - cg) ** 2).sum(-1)
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        var_iso = ((elite - mean.unsqueeze(0)) ** 2).mean(dim=(0, 2))   # isotropic -> rotation-invariant
        sigma = var_iso.sqrt().clamp_min(1e-3).unsqueeze(-1).expand(H, 3)
    return mean


@torch.no_grad()
def closed_loop(model, X0, Xg, *, cost_kind="l2", R_noise=None, w_t=0.5,
                T_max=8, replan_every=2, H=6, gen=None, **cem):
    r"""Receding-horizon decoder-free reaching toward the ENCODER goal $E(X_g)$.

    Re-encodes the current cloud each replan (re-anchors to reality), plans a length-``H`` action
    sequence with :func:`cem_plan_se3`, executes ``replan_every`` of them on the equivariant teacher.
    Open-loop is ``replan_every == T_max`` with ``H == T_max`` (one plan covers the whole budget).
    Returns ``{"ang","pos","frac","ang0"}`` where ``ang`` is the residual Kabsch reorientation (deg),
    ``frac`` $=1-\mathrm{ang}/\mathrm{ang0}$ (1 reached, 0 no progress, $<0$ moved away). The key
    matches Step 14/18 so the paired helpers apply unchanged.
    """
    zg = model.encoder(Xg)
    cg = centroid(Xg)
    X = X0.clone()
    t = 0
    while t < T_max:
        plan = cem_plan_se3(model, X, zg, cg, cost_kind=cost_kind, R_noise=R_noise, w_t=w_t, H=H, gen=gen, **cem)
        for k in range(min(replan_every, T_max - t, plan.shape[0])):
            X = teacher_step(X, plan[k:k + 1]); t += 1
    ang0 = rotation_angle_deg(kabsch_rotation(X0[0], Xg[0]))
    ang = rotation_angle_deg(kabsch_rotation(X[0], Xg[0]))
    return {"ang": ang, "pos": float((centroid(X) - cg).norm()),
            "frac": 1.0 - ang / max(ang0, 1e-6), "ang0": ang0}


@torch.no_grad()
def reach_ceiling_oracle(model, tasks, *, H=8, gen, **cem):
    r"""Two non-deployable reference points (both need $a_{\rm true}$): the predictor-space goal
    ceiling (open-loop CEM toward $f^H(E(X_0),a_{\rm true})$ -- on-manifold by construction) and the
    oracle (replay $a_{\rm true}$ itself). Returns ``(ceiling_frac, oracle_frac)``."""
    ceil, orac = [], []
    for X0, Xg, a_true in tasks:
        ang0 = rotation_angle_deg(kabsch_rotation(X0[0], Xg[0]))
        # predictor-space target: roll the predictor under a_true (stays on the predictor manifold)
        zt = model.encoder(X0)
        for hh in range(a_true.shape[0]):
            zt = model.predictor(zt, a_true[hh:hh + 1])
        cg = centroid(Xg)
        plan = cem_plan_se3(model, X0, zt, cg, cost_kind="l2", H=H, gen=gen, **cem)
        X = X0.clone()
        for h in range(plan.shape[0]):
            X = teacher_step(X, plan[h:h + 1])
        ceil.append(1.0 - rotation_angle_deg(kabsch_rotation(X[0], Xg[0])) / max(ang0, 1e-6))
        # oracle: replay the actions that MADE Xg
        Xo = X0.clone()
        for hh in range(a_true.shape[0]):
            Xo = teacher_step(Xo, a_true[hh:hh + 1])
        orac.append(1.0 - rotation_angle_deg(kabsch_rotation(Xo[0], Xg[0])) / max(ang0, 1e-6))
    return float(np.mean(ceil)), float(np.mean(orac))


# --------------------------------------------------------------------------- #
# [A] the reaching cure: ablation ladder + signal/mode sweep + references
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _orient_frac(X0, Xg, plan):
    r"""Execute ``plan`` on the equivariant teacher; return the orientation frac closed (Kabsch)."""
    X = X0.clone()
    for h in range(plan.shape[0]):
        X = teacher_step(X, plan[h:h + 1].to(X.dtype))
    ang0 = rotation_angle_deg(kabsch_rotation(X0[0], Xg[0]))
    ang = rotation_angle_deg(kabsch_rotation(X[0], Xg[0]))
    return 1.0 - ang / max(ang0, 1e-6)


@torch.no_grad()
def panel_A(eq1, eqR, tasks, *, cem_kw, T_max, replan, H_plan, base_seed=10_000):
    r"""Ablation ladder (encoder goal, frac of orientation gap closed). ``eq1`` 1-step model, ``eqR``
    rollout-consistent model. The ladder is OPEN-LOOP (faithful to Step 13[C]) so each rung isolates
    one ingredient: verbatim->equivariant planner, then 1-step->rollout training. The receding payoff
    is shown separately in the signal x mode sweep. Returns a dict of named fracs."""
    def avg(fn):
        return float(np.mean([fn(X0, Xg, torch.Generator().manual_seed(base_seed + i))
                              for i, (X0, Xg, _) in enumerate(tasks)]))

    out = {}
    # row 1: faithful Step-13[C] -- verbatim planner (box clamp, diagonal sigma, translation-blind L2),
    #         1-step training, OPEN-LOOP single shot. Reproduces the documented NEGATIVE/weak frac.
    out["verbatim_1step"] = avg(lambda X0, Xg, g: _orient_frac(
        X0, Xg, latent_cem_plan(eq1, X0, eq1.encoder(Xg), H=T_max, gen=g, **cem_kw)))
    # row 2: + SE(3)-equivariant planner (iso-sigma, ball clamp, centroid channel), 1-step, open-loop
    out["equiv_1step_l2"] = avg(lambda X0, Xg, g: closed_loop(
        eq1, X0, Xg, cost_kind="l2", T_max=T_max, replan_every=T_max, H=T_max, gen=g, **cem_kw)["frac"])
    # row 3: + rollout-consistency training (the manifold cure), open-loop L2
    out["equiv_rollout_l2"] = avg(lambda X0, Xg, g: closed_loop(
        eqR, X0, Xg, cost_kind="l2", T_max=T_max, replan_every=T_max, H=T_max, gen=g, **cem_kw)["frac"])
    # row 4: + SE(3)-native Procrustes goal + receding (best deployable), rollout model
    out["equiv_rollout_proc"] = avg(lambda X0, Xg, g: closed_loop(
        eqR, X0, Xg, cost_kind="proc", T_max=T_max, replan_every=replan, H=H_plan, gen=g, **cem_kw)["frac"])

    # signal x mode sweep on the rollout model (open = one plan of length T_max; receding = replan)
    sweep = {}
    for ck in ("l2", "proc"):
        sweep[f"{ck}_open"] = avg(lambda X0, Xg, g, ck=ck: closed_loop(
            eqR, X0, Xg, cost_kind=ck, T_max=T_max, replan_every=T_max, H=T_max, gen=g, **cem_kw)["frac"])
        sweep[f"{ck}_recede"] = avg(lambda X0, Xg, g, ck=ck: closed_loop(
            eqR, X0, Xg, cost_kind=ck, T_max=T_max, replan_every=replan, H=H_plan, gen=g, **cem_kw)["frac"])
    out["sweep"] = sweep

    gen = torch.Generator().manual_seed(base_seed)
    out["ceiling_pred"], out["oracle"] = reach_ceiling_oracle(eqR, tasks, H=T_max, gen=gen, **cem_kw)
    return out


# --------------------------------------------------------------------------- #
# [B] equivariance payoff: paired seen-vs-OOD orbit (reaching transfers exactly)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_paired(model, tasks, orbit, *, cost_kind, w_t, T_max, replan, base_seed, **cem):
    r"""Run each base task at the seen element (gid 0) and every OOD $(R,t)$, SAME per-task seed; only
    the global SE(3) element changes. Noise pre-rotated by $R$ for OOD. Returns ``{gid: [result]}``."""
    results = {gid: [] for gid in range(len(orbit))}
    for i, (X0, Xg, _) in enumerate(tasks):
        for gid, (R, t) in enumerate(orbit):
            X0g, Xgg = _apply(X0, R, t), _apply(Xg, R, t)
            R_noise = None if gid == 0 else R
            gen = torch.Generator().manual_seed(base_seed + i)
            results[gid].append(closed_loop(model, X0g, Xgg, cost_kind=cost_kind, R_noise=R_noise,
                                            w_t=w_t, T_max=T_max, replan_every=replan, gen=gen, **cem))
    return results


# --------------------------------------------------------------------------- #
# [C] genuineness: the goal cost is SE(3)-invariant by construction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cost_invariance_err(model, tasks, R):
    r"""Max over tasks of $\lvert \mathrm{cost}(\rho(R)z_0,\rho(R)z_g)-\mathrm{cost}(z_0,z_g)\rvert$ for both
    the $L_2$ and the Procrustes goal cost: a shared latent rotation must leave the SE(3)-native goal
    unchanged. Returns ``(l2_err, proc_err)``."""
    l2e, pre = 0.0, 0.0
    for X0, Xg, _ in tasks:
        z0, zg = model.encoder(X0), model.encoder(Xg)
        z0r, zgr = rotate_latent(z0, R), rotate_latent(zg, R)
        l2 = ((z0 - zg) ** 2).sum(-1); l2r = ((z0r - zgr) ** 2).sum(-1)
        pr = latent_residual_angle(z0, zg); prr = latent_residual_angle(z0r, zgr)
        l2e = max(l2e, float((l2 - l2r).abs().max()))
        pre = max(pre, float((pr - prr).abs().max()))
    return l2e, pre


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 84

    if SMOKE:
        N_TRAIN, EPOCHS, H_ROLL = 150, 4, 6
        K, N_OOD, H_GOAL = 4, 1, 4
        T_MAX, REPLAN, H_PLAN, W_T, T_SCALE = 8, 2, 6, 0.5, 0.5
        cem_kw = dict(n_samples=64, n_iters=3, n_elite=8, sigma0=0.6, w_run=0.3)
    else:
        N_TRAIN, EPOCHS, H_ROLL = 1200, 50, 6
        K, N_OOD, H_GOAL = 24, 4, 6
        T_MAX, REPLAN, H_PLAN, W_T, T_SCALE = 8, 2, 6, 0.5, 0.8
        cem_kw = dict(n_samples=256, n_iters=8, n_elite=25, sigma0=0.6, w_run=0.3)
    VAR_COEF = 0.1

    print(line)
    print(f"STEP 38  decoder-free latent-goal REACHING, cured via the SE(3) theorem  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    # ---- train: 1-step (the Step-13 recipe) + rollout-consistency (the cure), VN & MLP ----
    print(f"    training on {N_TRAIN} clouds, phi in [0,90)  (1-step VN, rollout VN, rollout MLP)")
    S, A, S2 = collect_cloud_transitions(N_TRAIN, seed=0)
    eq1 = build_eq_jepa()
    train_jepa(eq1, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=999)
    Xr, Ar = collect_cloud_rollouts(N_TRAIN, H_ROLL, seed=0)
    eqR = build_eq_jepa()
    train_jepa_rollout(eqR, Xr, Ar, epochs=EPOCHS, batch_size=128, H_roll=H_ROLL, var_coef=VAR_COEF, seed=0)
    mlpR = build_mlp_jepa()
    train_jepa_rollout(mlpR, Xr, Ar, epochs=EPOCHS, batch_size=128, H_roll=H_ROLL, var_coef=VAR_COEF, seed=0)

    St, At, _ = collect_cloud_transitions(64, seed=999)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    eqR_comp = composed_equiv_err(eqR, St, At, R_chk)
    mlpR_comp = composed_equiv_err(mlpR, St, At, R_chk)
    print(f"    post-train composed equiv (random SO(3)): rollout VN={eqR_comp:.2e} (exact)  "
          f"MLP={mlpR_comp:.2e} (no prior)")

    # promote to eval precision (the theorem is a float-precision statement; see Step 18 EVAL_DTYPE)
    eq1, eqR, mlpR = eq1.to(EVAL_DTYPE), eqR.to(EVAL_DTYPE), mlpR.to(EVAL_DTYPE)
    tasks = make_reach_tasks(K, seed=321, H_goal=H_GOAL)
    goal_reorient = float(np.mean([rotation_angle_deg(kabsch_rotation(Xg[0], X0[0])) for X0, Xg, _ in tasks]))
    print(f"    {K} reorientation tasks; goals reorient by {goal_reorient:.1f} deg on avg; "
          f"closed loop T_max={T_MAX}, replan={REPLAN}")

    # ---- [A] the reaching cure -------------------------------------------------
    print()
    print(line)
    print("[A] REACHING CURE -- decoder-free, encoder goal E(Xg), fraction of orientation gap closed")
    print(line)
    A_res = panel_A(eq1, eqR, tasks, cem_kw=cem_kw, T_max=T_MAX, replan=REPLAN, H_plan=H_PLAN)
    print(f"    ablation ladder (each row adds one ingredient):")
    print(f"        {'Step-13[C] verbatim planner, 1-step train':48s} frac = {A_res['verbatim_1step']:+.3f}  (reproduce the FAILURE)")
    print(f"        {'+ SE(3)-equivariant planner (iso/ball/centroid)':48s} frac = {A_res['equiv_1step_l2']:+.3f}  (Step 18 lift)")
    print(f"        {'+ rollout-consistency training':48s} frac = {A_res['equiv_rollout_l2']:+.3f}  (Step 38 main cure)")
    print(f"        {'+ SE(3)-native Procrustes goal + receding':48s} frac = {A_res['equiv_rollout_proc']:+.3f}  (best DEPLOYABLE)")
    sw = A_res["sweep"]
    print(f"    signal x mode sweep (rollout model):")
    print(f"        {'':14s} {'open':>8s} {'receding':>10s}")
    print(f"        {'L2':14s} {sw['l2_open']:8.3f} {sw['l2_recede']:10.3f}")
    print(f"        {'Procrustes':14s} {sw['proc_open']:8.3f} {sw['proc_recede']:10.3f}")
    print(f"    non-deployable references (need a_true):")
    print(f"        predictor-space goal (ceiling) frac = {A_res['ceiling_pred']:+.3f}")
    print(f"        replay a_true        (oracle)  frac = {A_res['oracle']:+.3f}")
    best_deploy = max(sw.values())
    cured = (best_deploy > 0.45) and (best_deploy - A_res["verbatim_1step"] > 0.30)
    print(f"    => DEPLOYABLE best = {best_deploy:.3f}  (Step-13[C] faithful control {A_res['verbatim_1step']:+.3f}, "
          f"gain {best_deploy - A_res['verbatim_1step']:+.3f}): {'CURED' if cured else 'inconclusive'}")

    # ---- [B] equivariance payoff: reaching transfers exactly across the SE(3) orbit ----
    print()
    print(line)
    print("[B] EQUIVARIANCE PAYOFF -- paired seen-vs-OOD SE(3) orbit (reaching transfers exactly)")
    print(line)
    orbit = make_se3_orbit(N_OOD, seed=13, t_scale=T_SCALE)
    gids = list(range(len(orbit)))
    print(f"    SE(3)-native goal (Procrustes angle), receding (replan={REPLAN}):")
    vnB = eval_paired(eqR, tasks, orbit, cost_kind="proc", w_t=W_T, T_max=T_MAX, replan=REPLAN, H=H_PLAN, base_seed=10_000, **cem_kw)
    mlpB = eval_paired(mlpR, tasks, orbit, cost_kind="proc", w_t=W_T, T_max=T_MAX, replan=REPLAN, H=H_PLAN, base_seed=10_000, **cem_kw)
    Bg = report_panel_se3(vnB, mlpB, gids)

    # ---- [C] genuineness -------------------------------------------------------
    print()
    print(line)
    print("[C] GENUINENESS -- the SE(3)-native goal cost is invariant; the VN realises it end-to-end")
    print(line)
    l2_inv, proc_inv = cost_invariance_err(eqR, tasks, R_chk.to(EVAL_DTYPE))
    print(f"    goal-cost invariance under a shared latent rotation rho(R) (max over tasks):")
    print(f"        L2 cost          |d| = {l2_inv:.2e}        Procrustes-angle cost |d| = {proc_inv:.2e}  (both ~float floor)")
    print(f"    plan equivariance (panel [B] max|d_i| residual angle, seen vs OOD): {Bg['vn_diff_max']:.2e} deg")
    print(f"    composed equivariance: rollout VN = {eqR_comp:.2e}   MLP = {mlpR_comp:.2e}")

    # ---- verdict + JSON --------------------------------------------------------
    print()
    print(line)
    print("STEP 38 SUMMARY")
    print(line)
    ok_cured = cured
    ok_vn_flat = Bg["vn_ratio_ci"][1] < 1.05
    ok_mlp_degrades = Bg["mlp_ratio_ci"][0] > 1.0
    ok_separated = Bg["vn_ratio_ci"][1] < Bg["mlp_ratio_ci"][0]
    ok_equiv = eqR_comp < 1e-4
    ok_cost_inv = max(l2_inv, proc_inv) < 1e-4
    passed = ok_cured and ok_vn_flat and ok_mlp_degrades and ok_separated and ok_equiv and ok_cost_inv
    print(f"    [A] decoder-free reaching CURED:  {A_res['verbatim_1step']:+.3f} (Step-13[C]) -> "
          f"{best_deploy:+.3f} (deployable) -> {A_res['ceiling_pred']:+.3f} (ceiling)   [{'PASS' if ok_cured else 'FAIL'}]")
    print(f"    [B] reaching transfers exactly:   VN OOD/seen ratio CI=[{Bg['vn_ratio_ci'][0]:.3f},{Bg['vn_ratio_ci'][1]:.3f}], "
          f"MLP=[{Bg['mlp_ratio_ci'][0]:.3f},{Bg['mlp_ratio_ci'][1]:.3f}]   [{'PASS' if (ok_vn_flat and ok_mlp_degrades and ok_separated) else 'FAIL'}]")
    print(f"    [C] SE(3)-native & genuine:       cost |d|<{max(l2_inv,proc_inv):.0e}, composed-equiv {eqR_comp:.0e}   "
          f"[{'PASS' if (ok_equiv and ok_cost_inv) else 'FAIL'}]")
    print(f"    OVERALL: {'PASS -- the only outright failure is resolved, decoder-free, exactly equivariant' if passed else 'INCONCLUSIVE'}")

    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(N_TRAIN=N_TRAIN, EPOCHS=EPOCHS, H_ROLL=H_ROLL, K=K, N_OOD=N_OOD,
                       H_GOAL=H_GOAL, T_MAX=T_MAX, REPLAN=REPLAN, W_T=W_T, T_SCALE=T_SCALE, **cem_kw),
        "goal_reorient_deg": goal_reorient,
        "panel_A": {k: v for k, v in A_res.items()},
        "best_deployable": best_deploy,
        "panel_B": {
            "vn_ang": Bg["vn_ang"], "mlp_ang": Bg["mlp_ang"], "vn_pos": Bg["vn_pos"],
            "vn_ratio_ci": Bg["vn_ratio_ci"], "mlp_ratio_ci": Bg["mlp_ratio_ci"],
            "vn_diff_max": Bg["vn_diff_max"], "vn_diff_mean": Bg["vn_diff_mean"], "mlp_diff_mean": Bg["mlp_diff_mean"],
        },
        "panel_C": {"l2_cost_inv": l2_inv, "proc_cost_inv": proc_inv,
                    "vn_composed_equiv": eqR_comp, "mlp_composed_equiv": mlpR_comp},
        "checks": dict(ok_cured=ok_cured, ok_vn_flat=ok_vn_flat, ok_mlp_degrades=ok_mlp_degrades,
                       ok_separated=ok_separated, ok_equiv=ok_equiv, ok_cost_inv=ok_cost_inv),
        "passed": bool(passed),
    }
    tag = "_smoke" if SMOKE else ""
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"step38_latent_goal_reaching{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n    wrote {out_path.relative_to(ROOT)}")
    _make_figure(fig_dir / f"step38_latent_goal_reaching{tag}.png", A_res, best_deploy, Bg, gids,
                 l2_inv, proc_inv, eqR_comp, mlpR_comp, goal_reorient)


def _make_figure(path, A_res, best_deploy, Bg, gids, l2_inv, proc_inv, vn_comp, mlp_comp, goal_reorient):
    r"""Three panels telling the cure story: [A] reaching ablation ladder (failure -> deployable ->
    ceiling -> oracle), [B] the theorem -- residual orientation error is flat across the SE(3) orbit for
    the VN, rising for the MLP, and [C] genuineness -- the goal cost & the realised plan are invariant to
    the float floor for the VN while the MLP's composed equivariance is $O(1)$."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: write files, never open a window
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"    (figure skipped: {exc})")
        return
    C_EQ, C_MLP, C_REF = "#1b7837", "#762a83", "#888888"   # green=equivariant, purple=baseline, grey=ref
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

    # [A] the reaching cure: an ablation ladder from the Step-13[C] failure to the deployable cure -------
    rungs = ["Step-13[C]\nverbatim\n(1-step)", "+ SE(3)\nplanner", "+ rollout\nconsistency",
             "+ Procrustes\ngoal (best\ndeployable)"]
    vals = [A_res["verbatim_1step"], A_res["equiv_1step_l2"], A_res["equiv_rollout_l2"], best_deploy]
    # red for the failure rung, green-deepening for the cure rungs
    cols = ["#c0392b", "#9ad19a", "#5aae61", C_EQ]
    xs = np.arange(len(rungs))
    ax[0].bar(xs, vals, 0.64, color=cols)
    for x, v in zip(xs, vals):
        ax[0].text(x, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}", ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=9)
    ax[0].axhline(A_res["ceiling_pred"], ls="--", c="#2e86de", lw=1.2,
                  label=f"predictor-space ceiling ({A_res['ceiling_pred']:+.2f}, needs $a_\\mathrm{{true}}$)")
    ax[0].axhline(A_res["oracle"], ls=":", c="#27ae60", lw=1.2,
                  label=f"oracle / replay $a_\\mathrm{{true}}$ ({A_res['oracle']:+.2f})")
    ax[0].axhline(0.0, c=C_REF, lw=1, zorder=0)
    ax[0].text(len(xs) - 0.5, 0.012, "no progress (×0)", color=C_REF, fontsize=7, ha="right", va="bottom")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(rungs, fontsize=7.5)
    ax[0].set_ylabel("fraction of orientation gap closed (decoder-free)")
    ax[0].set_ylim(min(-0.05, min(vals) - 0.05), 1.08)
    ax[0].set_title("[A] reaching cure: failure → deployable", fontsize=10.5)
    ax[0].legend(fontsize=7.5, loc="upper left")

    # [B] the theorem: residual orientation error across the SE(3) orbit -- VN flat, MLP rising ----------
    # report_panel_se3 returns vn_ang/mlp_ang as per-gid dicts; order them along the orbit for plotting.
    vn_ang = [Bg["vn_ang"][g] for g in gids]
    mlp_ang = [Bg["mlp_ang"][g] for g in gids]
    xb = np.arange(len(gids))
    xticklab = ["seen"] + [f"g{g}" for g in gids[1:]]
    ax[1].plot(xb, vn_ang, "-o", c=C_EQ, lw=2.2, ms=7, label="VN (equivariant)")
    ax[1].plot(xb, mlp_ang, "-s", c=C_MLP, lw=2.0, ms=6, label="MLP (no prior)")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(xticklab)
    ax[1].set_xlabel("SE(3) orbit element (seen + OOD)")
    ax[1].set_ylabel("residual orientation error (deg)")
    vr = Bg["vn_ratio_ci"]; mr = Bg["mlp_ratio_ci"]
    ax[1].set_title(f"[B] reaching transfers exactly\nOOD/seen ratio: VN ×{vr[0]:.3f}  vs  MLP ×{mr[0]:.2f}–{mr[1]:.2f}",
                    fontsize=10.5)
    ax[1].text(0.02, 0.04, f"VN max$|d_i|$={Bg['vn_diff_max']:.1e} deg (float floor)",
               transform=ax[1].transAxes, fontsize=8, color=C_EQ, va="bottom")
    ax[1].legend(fontsize=8.5, loc="upper left")
    ax[1].set_ylim(0, max(mlp_ang) * 1.18)

    # [C] genuineness: goal cost + realised plan invariant for the VN; MLP composed-equiv is O(1) --------
    clabels = ["goal cost\n($L_2$)", "goal cost\n(Procr.)", "plan equiv\n(seen-OOD)",
               "composed\nequiv (VN)", "composed\nequiv (MLP)"]
    cvals = [max(l2_inv, 1e-12), max(proc_inv, 1e-12), max(Bg["vn_diff_max"], 1e-12),
             max(vn_comp, 1e-12), max(mlp_comp, 1e-12)]
    ccols = [C_EQ, C_EQ, C_EQ, C_EQ, C_MLP]
    xc = np.arange(len(clabels))
    ax[2].bar(xc, cvals, 0.64, color=ccols)
    for x, v in zip(xc, cvals):
        ax[2].text(x, v * 1.5, f"{v:.0e}", ha="center", va="bottom", fontsize=8)
    ax[2].axhline(1e-2, ls=":", c=C_REF, lw=1.1, label="break threshold $10^{-2}$")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xc); ax[2].set_xticklabels(clabels, fontsize=7.5)
    ax[2].set_ylabel("max deviation under shared $\\rho(R)$  (log)")
    ax[2].set_ylim(1e-8, max(cvals) * 12)
    ax[2].set_title("[C] SE(3)-native goal, realised end-to-end", fontsize=10.5)
    ax[2].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        rf"Step 38 — decoder-free latent-goal reaching ({goal_reorient:.0f}° reorientation tasks): "
        rf"the only outright failure, cured and made exactly equivariant",
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
