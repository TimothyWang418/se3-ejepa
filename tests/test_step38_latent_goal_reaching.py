r"""Step 38 -- decoder-free latent-goal REACHING (``experiments/step38_latent_goal_reaching.py``).

Step 38 re-attacks Step 13's panel [C] -- the project's only outright failure, where a purely-latent
CEM-MPC planner pushed the cloud *away* from the goal (NEGATIVE fraction-of-gap-closed). The cure has
three ingredients: rollout-consistency training (puts $f^H(E(x_0),a)$ back on the encoder manifold), an
$\mathrm{SE}(3)$-equivariant CEM planner (Step 18), and an $\mathrm{SE}(3)$-native goal *signal* -- the
geodesic angle of the residual rotation $R^\star$ that Kabsch-aligns the 16 type-1 vectors of $z_0,z_g$.

The reaching *cure* itself (NEGATIVE $\to$ strongly positive frac) needs full training and is certified by
the experiment's panel [A]. This test certifies the **structural / architectural invariants** the cure
stands on -- all fast, the model-based ones run at *random init* because equivariance is architectural,
not learned:

1. **The Procrustes goal signal is a true geodesic residual angle** -- ``latent_residual_angle`` is $0$
   when the two latents coincide, recovers exactly $\lvert R\rvert$ when $z_H=\rho(R)z_g$, lies in
   $[0,\pi]$, and is symmetric in its arguments ($\lvert R\rvert=\lvert R^{-1}\rvert$). A pure function.
2. **The Procrustes goal cost is $\mathrm{SE}(3)$-invariant by construction** -- a *shared* latent rotation
   $\rho(R)$ conjugates the Kabsch fit and leaves the angle (a function of the trace) unchanged.
3. **The $L_2$ goal cost is $\mathrm{SE}(3)$-invariant** -- $\rho(R)$ acts block-diagonally as $R$ on each
   of the 16 type-1 vectors, hence orthogonally on $\mathbb R^{48}$, so it preserves $\lVert z_H-z_g\rVert$.
   (Invariance is *necessary but not sufficient*: the $L_2$ signal still under-reaches because the encoder
   goal sits off the predictor manifold -- the experiment's diagnosis, cured by rollout-consistency.)
4. **The VN encoder is exactly $\mathrm{SE}(3)$-equivariant at init** (composed-equivariance $\sim$ float
   floor); the MLP control has no such prior. (The MLP's *break* is the experiment's post-train panel [C].)
5. **The goal cost is invariant through the actual VN encoder, end-to-end** -- rotating the *cloud* by a
   random $(R,t)$, encoding, and recomputing both costs reproduces them to the float floor (equivariant
   encoder $\circ$ invariant cost). This is the architectural foundation of the experiment's panel [C].
6. **Reaching transfers exactly across the $\mathrm{SE}(3)$ orbit -- at init** -- the equivariant CEM loop
   (isotropic $\sigma$, ball clamp, centroid channel, noise pre-rotated by $R$) makes the OOD plan the exact
   $R$-image of the seen plan, so the executed residual orientation error is identical seen-vs-OOD to the
   float floor. This is the headline theorem of panel [B]; being architectural it already holds untrained.

Run:
    .venv/bin/python tests/test_step38_latent_goal_reaching.py
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    LATENT_DIM, build_eq_jepa, build_mlp_jepa, collect_cloud_transitions,
    composed_equiv_err, rand_so3, rotate_latent,
)
from step18_se3_closed_loop import (  # noqa: E402
    EVAL_DTYPE, _apply, make_se3_orbit, rotation_angle_deg,
)
from step38_latent_goal_reaching import (  # noqa: E402
    closed_loop, cost_invariance_err, latent_residual_angle, make_reach_tasks,
)

torch.set_default_dtype(torch.float32)
NV = LATENT_DIM // 3        # 16 type-1 latent vectors
DEG = 180.0 / math.pi


def _rand_latents(B, *, gen, dtype=EVAL_DTYPE):
    r"""``B`` random latents ``(B, 48)`` viewed as $B\times16$ type-1 vectors."""
    return torch.randn(B, LATENT_DIM, generator=gen, dtype=dtype)


def _so3_64(gen):
    r"""A proper $\mathrm{SO}(3)$ rotation accurate to *float64*. ``rand_so3`` emits float32-precision
    orthogonality ($R^\top R - I\sim10^{-7}$); we re-orthogonalise via SVD so the float-floor invariance
    claims test a genuinely orthogonal $\rho(R)$, not the cast-from-float32 residual."""
    R = rand_so3(gen).to(EVAL_DTYPE)
    U, _, Vh = torch.linalg.svd(R)
    R = U @ Vh
    if float(torch.det(R)) < 0:               # guarantee det +1 (proper rotation, no reflection)
        U = U.clone(); U[:, -1] *= -1
        R = U @ Vh
    return R


def main() -> None:
    torch.manual_seed(0)
    print("Step 38 -- decoder-free latent-goal reaching: structural / architectural invariants\n")
    gen = torch.Generator().manual_seed(0)

    # ---- 1. latent_residual_angle is a true geodesic residual angle -----------------------------
    z = _rand_latents(64, gen=gen)
    self_ang = latent_residual_angle(z, z)
    assert float(self_ang.abs().max()) < 1e-6, f"residual angle of a latent with itself must be 0, got {float(self_ang.max()):.2e}"
    worst_recover, worst_range, worst_sym = 0.0, 0.0, 0.0
    for _ in range(20):
        R = _so3_64(gen)
        zg = _rand_latents(64, gen=gen)
        zH = rotate_latent(zg, R)                                  # z_H = rho(R) z_g, exactly
        ang = latent_residual_angle(zH, zg) * DEG                  # should equal |R|
        true_ang = rotation_angle_deg(R)
        worst_recover = max(worst_recover, float((ang - true_ang).abs().max()))
        worst_range = max(worst_range, float(ang.max()) - 180.0, -float(ang.min()))
        sym = latent_residual_angle(zg, zH) * DEG                  # |R| == |R^{-1}|
        worst_sym = max(worst_sym, float((ang - sym).abs().max()))
    assert worst_recover < 1e-3, f"residual angle must recover |R| when z_H=rho(R)z_g, worst {worst_recover:.2e} deg"
    assert worst_range < 1e-6, f"residual angle must lie in [0,180] deg, worst overshoot {worst_range:.2e}"
    assert worst_sym < 1e-3, f"residual angle must be symmetric (|R|=|R^-1|), worst {worst_sym:.2e} deg"
    print(f"  [1] Procrustes signal: angle(z,z)=0, recovers |R| (worst {worst_recover:.1e} deg), "
          f"in [0,180], symmetric ({worst_sym:.1e} deg)  OK")

    # ---- 2. Procrustes goal cost is SE(3)-invariant under a SHARED latent rotation --------------
    worst_proc = 0.0
    for _ in range(30):
        R = _so3_64(gen)
        zH, zg = _rand_latents(64, gen=gen), _rand_latents(64, gen=gen)
        base = latent_residual_angle(zH, zg)
        rot = latent_residual_angle(rotate_latent(zH, R), rotate_latent(zg, R))
        worst_proc = max(worst_proc, float((base - rot).abs().max()))
    assert worst_proc < 1e-5, f"Procrustes cost not SE(3)-invariant under shared rho(R), worst {worst_proc:.2e}"
    print(f"  [2] Procrustes goal cost SE(3)-invariant (shared rho(R)): max|d| = {worst_proc:.1e}  OK")

    # ---- 3. L2 goal cost is SE(3)-invariant (rho(R) is block-orthogonal, hence norm-preserving) -
    worst_l2 = 0.0
    for _ in range(30):
        R = _so3_64(gen)
        zH, zg = _rand_latents(64, gen=gen), _rand_latents(64, gen=gen)
        base = ((zH - zg) ** 2).sum(-1)
        rot = ((rotate_latent(zH, R) - rotate_latent(zg, R)) ** 2).sum(-1)
        worst_l2 = max(worst_l2, float((base - rot).abs().max()))
    assert worst_l2 < 1e-9, f"L2 cost not SE(3)-invariant (rho(R) should be orthogonal), worst {worst_l2:.2e}"
    print(f"  [3] L2 goal cost SE(3)-invariant (rho(R) block-orthogonal): max|d| = {worst_l2:.1e}  OK  "
          f"(invariant != reachable -- L2 still under-reaches off-manifold; the experiment's cure)")

    # ---- 4. VN encoder is exactly SE(3)-equivariant at init; the MLP control has no such prior --
    S, A, _ = collect_cloud_transitions(16, seed=7)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    vn = build_eq_jepa()
    mlp = build_mlp_jepa()
    vn_comp = composed_equiv_err(vn, S, A, R_chk)
    mlp_comp = composed_equiv_err(mlp, S, A, R_chk)
    assert vn_comp < 1e-4, f"VN encoder not SE(3)-equivariant at init: composed-equiv {vn_comp:.2e}"
    assert mlp_comp > 100 * vn_comp, f"MLP should have no equivariance prior, but {mlp_comp:.2e} !>> {vn_comp:.2e}"
    print(f"  [4] composed equivariance at init: VN = {vn_comp:.1e} (architectural), MLP = {mlp_comp:.1e} "
          f"(no prior; post-train break is the experiment's panel [C])  OK")

    # ---- 5. goal cost invariant through the ACTUAL VN encoder, end-to-end (rotate the CLOUD) ----
    # cost_invariance_err rotates the encoded latent (matches the experiment's panel [C]); here we also
    # rotate the raw cloud and re-encode, so the check exercises encoder-equivariance o invariant-cost.
    vn64 = vn.to(EVAL_DTYPE)
    tasks = make_reach_tasks(4, seed=321, H_goal=4)
    l2_lat, proc_lat = cost_invariance_err(vn64, tasks, R_chk.to(EVAL_DTYPE))   # latent-rotation form
    worst_l2_e2e = worst_proc_e2e = 0.0
    for _ in range(6):
        R = _so3_64(gen)
        t = ((torch.rand(3, generator=gen) * 2 - 1) * 0.8).to(EVAL_DTYPE)
        for X0, Xg, _ in tasks:
            with torch.no_grad():
                z0, zg = vn64.encoder(X0), vn64.encoder(Xg)
                z0r, zgr = vn64.encoder(_apply(X0, R, t)), vn64.encoder(_apply(Xg, R, t))
            worst_l2_e2e = max(worst_l2_e2e, abs(float(((z0 - zg) ** 2).sum(-1) - ((z0r - zgr) ** 2).sum(-1))))
            worst_proc_e2e = max(worst_proc_e2e, abs(float(latent_residual_angle(z0, zg) - latent_residual_angle(z0r, zgr))))
    assert max(l2_lat, proc_lat) < 1e-4, f"goal cost not invariant to a latent rotation: {max(l2_lat, proc_lat):.2e}"
    assert worst_l2_e2e < 1e-4 and worst_proc_e2e < 1e-4, (
        f"goal cost not invariant end-to-end through the VN encoder: L2 {worst_l2_e2e:.2e}, proc {worst_proc_e2e:.2e}")
    print(f"  [5] goal cost invariant through the VN encoder: latent-rot L2={l2_lat:.1e}/proc={proc_lat:.1e}, "
          f"cloud-rot end-to-end L2={worst_l2_e2e:.1e}/proc={worst_proc_e2e:.1e}  OK")

    # ---- 6. reaching transfers EXACTLY across the SE(3) orbit -- at random init (architectural) --
    # The equivariant CEM loop makes the OOD plan the exact R-image of the seen plan, so the executed
    # residual orientation error is identical seen-vs-OOD to the float floor. Untrained: it reaches
    # poorly, but EQUALLY -- equivariance is a property of the architecture, not of the weights.
    orbit = make_se3_orbit(2, seed=13, t_scale=0.8)
    cem_kw = dict(n_samples=48, n_iters=2, n_elite=8, sigma0=0.6, w_run=0.3)
    worst_orbit = 0.0
    for i, (X0, Xg, _) in enumerate(tasks[:2]):
        seen = None
        for gid, (R, t) in enumerate(orbit):
            X0g, Xgg = _apply(X0, R, t), _apply(Xg, R, t)
            g = torch.Generator().manual_seed(2024 + i)
            res = closed_loop(vn64, X0g, Xgg, cost_kind="proc", R_noise=(None if gid == 0 else R),
                              w_t=0.5, T_max=4, replan_every=2, H=4, gen=g, **cem_kw)
            if gid == 0:
                seen = res["ang"]
            else:
                worst_orbit = max(worst_orbit, abs(res["ang"] - seen))
    assert worst_orbit < 1e-3, f"reaching not orbit-invariant at init: worst seen-vs-OOD |d| = {worst_orbit:.2e} deg"
    print(f"  [6] reaching transfers exactly across the SE(3) orbit at init (equivariant CEM): "
          f"worst seen-vs-OOD |d| = {worst_orbit:.1e} deg  OK  (MLP degrades -- experiment's panel [B])")

    print("\nPASS: the Procrustes goal signal is a genuine geodesic residual angle, both goal costs are")
    print("SE(3)-invariant by construction, the VN encoder realises that invariance end-to-end while the")
    print("MLP has no such prior, and decoder-free reaching transfers exactly across the SE(3) orbit even")
    print("at random init -- the architectural scaffolding on which the trained cure (panel [A]) stands.")


def test_step38_latent_goal_reaching() -> None:
    """pytest: Procrustes-angle signal, goal-cost SE(3)-invariance (L2 + Procrustes, latent & end-to-end), VN init equivariance, exact orbit-transfer of reaching at init."""
    main()


if __name__ == "__main__":
    main()
