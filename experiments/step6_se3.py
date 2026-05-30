r"""Step 6: the geometric world model closes the loop — in 2D *and* 3D.

Steps 3–5 made the *encoder* equivariant. Step 6 makes the **whole latent world
model** equivariant by giving the *predictor* the same structure, and lifts the
construction from SO(2) to **SE(3)**.

  [A] 2D, on real PushT state.  Encoder = :class:`StructuredStateEncoder`,
      predictor = :class:`VNPredictor` (``dim=2``). We roll the latent dynamics
      forward $H$ steps and check the *rollout* is exactly equivariant:
      $$ z_H(R_\alpha\!\cdot\! X,\; R_\alpha\!\cdot\! a_{0:H}) \;=\; \rho(\alpha)\, z_H(X, a_{0:H}) $$
      for arbitrary planar angles $\alpha$ — so the **planning cost is invariant
      to a joint rotation of (state, goal, action sequence)**, not just the
      encoder output.

  [B] 3D, on a synthetic point cloud.  Encoder = :class:`SE3PointEncoder`,
      predictor = :class:`VNPredictor` (``dim=3``, the *same* Vector-Neuron
      dynamics). The whole model is exactly **SE(3)**-equivariant: invariant to
      translation, equivariant to arbitrary $R\in\mathrm{SO}(3)$ — no grid, no
      interpolation floor, for *every* rotation.

This is the thesis payoff: a geometric inductive bias that holds *exactly* and
*continuously*, end-to-end (perception + latent dynamics), where the pixel path
could only manage $90^\circ$.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step6_se3.py
"""

import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

import stable_worldmodel as swm  # noqa: E402

from src.geometry.so2 import rotate_vectors  # noqa: E402
from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.models.structured import (  # noqa: E402
    StructuredStateEncoder,
    VNPredictor,
    extract_pusht_vectors,
)


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def rollout(enc, pred, x, actions):
    r"""Roll the latent dynamics $H$ steps. ``actions: (H, B, action_dim)``.

    $z_0 = E(x)$, $z_{t+1} = f_\phi(z_t, a_t)$; returns $z_H$ of shape ``(B, D)``.
    """
    z = enc(x)
    for a in actions:
        z = pred(z, a)
    return z


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)

    # =====================================================================
    banner("[A] 2D — whole world model equivariant on REAL PushT (rollout)")
    # =====================================================================
    world = swm.World("swm/PushT-v1", num_envs=8, image_shape=(65, 65))
    world.reset(seed=0)
    X = extract_pusht_vectors(world.infos)  # (B, 6, 2) type-1 vectors
    world.close()

    enc2d = StructuredStateEncoder(n_vec=6, latent_dim=128, hidden=64).eval()
    pred2d = VNPredictor(latent_dim=128, action_dim=2, hidden=64, dim=2).eval()
    H = 4
    actions2d = torch.randn(H, X.shape[0], 2)  # type-1 action vectors per step

    z_base = rollout(enc2d, pred2d, X, actions2d)  # (B, 128)
    n_lat = z_base.shape[1] // 2

    print(f"    state X {tuple(X.shape)}  |  rollout H={H}  |  latent {tuple(z_base.shape)}")
    print("    angle    | max|z_H(R.x,R.a) - R.z_H(x,a)| | cost drift")
    print("    ---------+-------------------------------+-----------")
    worst2d = 0.0
    for deg in [37.0, 45.0, 90.0, 123.4, 211.7]:
        a = math.radians(deg)
        Xr = rotate_vectors(X, a)
        ar = rotate_vectors(actions2d, a)  # rotate every action in the sequence
        z_rot = rollout(enc2d, pred2d, Xr, ar)  # z_H(R.x, R.a)
        ref = rotate_vectors(z_base.reshape(-1, n_lat, 2), a).reshape(z_base.shape)
        eq = (z_rot - ref).abs().max().item()
        # planning cost between two halves of the batch, base vs jointly rotated
        base_cost = (z_base[0:4] - z_base[4:8]).norm().item()
        rot_cost = (z_rot[0:4] - z_rot[4:8]).norm().item()
        drift = abs(rot_cost - base_cost)
        worst2d = max(worst2d, eq, drift)
        print(f"    {deg:6.1f}d  |          {eq:.2e}           |  {drift:.2e}")
    print(f"    worst (2D whole-model, all angles) = {worst2d:.2e}")

    # =====================================================================
    banner("[B] 3D — whole world model exactly SE(3)-equivariant (point cloud)")
    # =====================================================================
    enc3d = SE3PointEncoder(n_out_vec=8, lmax=2, mul=8).eval()
    latent_dim = enc3d.latent_dim  # 24
    pred3d = VNPredictor(latent_dim=latent_dim, action_dim=3, hidden=32, dim=3).eval()

    B, N = 8, 24
    pos = torch.randn(B, N, 3)
    actions3d = torch.randn(H, B, 3)  # type-1 3D action vectors per step
    z3_base = rollout(enc3d, pred3d, pos, actions3d)  # (B, 24)
    n_lat3 = z3_base.shape[1] // 3

    print(f"    cloud {tuple(pos.shape)}  |  rollout H={H}  |  latent {tuple(z3_base.shape)}")

    # translation invariance of the whole rollout
    t = torch.randn(1, 1, 3)
    z3_shift = rollout(enc3d, pred3d, pos + t, actions3d)
    trans_err = (z3_shift - z3_base).abs().max().item()
    print(f"    translation invariance  E(x+t) rollout : {trans_err:.2e}")

    print("    rotation # | max|z_H(R.x,R.a) - R.z_H(x,a)| | cost drift")
    print("    -----------+-------------------------------+-----------")
    worst3d = trans_err
    for i in range(5):
        R = o3.rand_matrix()
        pos_r = pos @ R.transpose(-1, -2)
        act_r = actions3d @ R.transpose(-1, -2)
        z_rot = rollout(enc3d, pred3d, pos_r, act_r)
        ref = (z3_base.reshape(B, n_lat3, 3) @ R.transpose(-1, -2)).reshape(z3_base.shape)
        eq = (z_rot - ref).abs().max().item()
        base_cost = (z3_base[0:4] - z3_base[4:8]).norm().item()
        rot_cost = (z_rot[0:4] - z_rot[4:8]).norm().item()
        drift = abs(rot_cost - base_cost)
        worst3d = max(worst3d, eq, drift)
        print(f"    R_{i}        |          {eq:.2e}           |  {drift:.2e}")
    print(f"    worst (3D whole-model, rand SO(3)) = {worst3d:.2e}")

    # =====================================================================
    banner("STEP 6 SUMMARY")
    # =====================================================================
    print(f"    2D (real PushT) : whole WM equivariant, all angles  worst {worst2d:.1e}")
    print(f"    3D (point cloud): whole WM SE(3)-equivariant        worst {worst3d:.1e}")
    print("    => encoder AND predictor share one representation, so the latent")
    print("       dynamics + planning cost are exactly symmetric — end to end,")
    print("       continuously, in 2D and 3D (no 90-degree pixel-grid limit).")
    worst = max(worst2d, worst3d)
    assert worst < 1e-4, f"whole-model equivariance broke: {worst:.2e}"
    print(f"    PASS: full geometric world model is exact to {worst:.1e}.")


if __name__ == "__main__":
    main()
