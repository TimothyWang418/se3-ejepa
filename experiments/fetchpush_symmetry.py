r"""SO(2) symmetry of the FetchPush state (Step 72, Stage 1).

Gymnasium-Robotics ``FetchPush-v4`` exposes a 25-D state observation. About the vertical (z) axis, rotating the scene
by $\theta$ acts on the *planar* $(x,y)$ components of every positional/velocity 3-vector and shifts the object yaw by
$\theta$; the $z$-components, gripper state, roll/pitch, and angular-velocity $z$ are invariant. The fixed arm base
makes this an **approximate** dynamical symmetry (Theorem B regime, exactly like PushT) — but the *representation*
$\rho(\theta)$ on the state is exact and orthogonal, which is what the equivariant world model needs.

Documented 25-D layout (Fetch ``_get_obs``):
  grip_pos(3) object_pos(3) object_rel_pos(3) gripper_state(2) object_rot(3) object_velp(3) object_velr(3)
  grip_velp(3) gripper_vel(2)  ->  3+3+3+2+3+3+3+3+2 = 25.

We expose two maps used downstream:
  * ``rotate_obs(obs, theta)`` — the group action $\rho(\theta)$ on the raw 25-D obs (rotate the 6 planar 2-vectors,
    add $\theta$ to the object yaw). A valid SO(2) action: composing rotations adds angles.
  * ``obs_to_vn(obs)`` — split into (vectors $(\cdot,n_{\rm vec},2)$, scalars $(\cdot,n_{\rm sca})$) for the VN encoder,
    encoding yaw as a $(\cos,\sin)$ 2-vector so that under ``rotate_obs`` the VN vectors transform by the exact
    orthogonal $R(\theta)$ and the scalars are invariant (verified in tests/test_fetchpush_symmetry.py).
"""

import numpy as np

# (name, start, length) for the 9 obs blocks
LAYOUT = [
    ("grip_pos", 0, 3), ("object_pos", 3, 3), ("object_rel_pos", 6, 3), ("gripper_state", 9, 2),
    ("object_rot", 11, 3), ("object_velp", 14, 3), ("object_velr", 17, 3), ("grip_velp", 20, 3),
    ("gripper_vel", 23, 2),
]
OBS_DIM = 25
# (x,y) index pairs of the planar position/velocity 3-vectors that rotate under SO(2)-z
PLANAR_XY = [(0, 1), (3, 4), (6, 7), (14, 15), (17, 18), (20, 21)]   # grip_pos, object_pos, object_rel_pos, object_velp, object_velr, grip_velp
YAW_IDX = 13                                                          # object_rot = euler[roll(11), pitch(12), yaw(13)]
N_VEC = len(PLANAR_XY) + 1                                            # 6 planar 2-vecs + 1 yaw (cos,sin) 2-vec = 7
# scalar indices = everything that is not an (x,y) of a planar vec and not the yaw
_planar_flat = {i for xy in PLANAR_XY for i in xy}
SCALAR_IDX = [i for i in range(OBS_DIM) if i not in _planar_flat and i != YAW_IDX]
N_SCA = len(SCALAR_IDX)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotate_obs(obs: np.ndarray, theta: float) -> np.ndarray:
    r"""Apply the SO(2)-z group action $\rho(\theta)$ to a raw 25-D obs (or a batch ``(...,25)``)."""
    obs = np.array(obs, dtype=np.float64, copy=True)
    R = _rot(theta)
    for ix, iy in PLANAR_XY:
        xy = np.stack([obs[..., ix], obs[..., iy]], axis=-1)        # (...,2)
        rot = xy @ R.T
        obs[..., ix], obs[..., iy] = rot[..., 0], rot[..., 1]
    obs[..., YAW_IDX] = obs[..., YAW_IDX] + theta                   # yaw shifts additively
    return obs


def obs_to_vn(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Split a raw obs ``(...,25)`` into (vectors ``(...,N_VEC,2)``, scalars ``(...,N_SCA)``). Yaw -> (cos,sin) 2-vec,
    so under ``rotate_obs`` the vectors transform by the exact orthogonal $R(\theta)$ and scalars are invariant."""
    obs = np.asarray(obs, dtype=np.float64)
    vecs = [np.stack([obs[..., ix], obs[..., iy]], axis=-1) for ix, iy in PLANAR_XY]
    yaw = obs[..., YAW_IDX]
    vecs.append(np.stack([np.cos(yaw), np.sin(yaw)], axis=-1))      # yaw as a unit 2-vec (rotates cleanly)
    vectors = np.stack(vecs, axis=-2)                              # (...,N_VEC,2)
    scalars = obs[..., SCALAR_IDX]                                  # (...,N_SCA)
    return vectors, scalars


def rotate_goal(goal_xyz: np.ndarray, theta: float) -> np.ndarray:
    r"""SO(2)-z on a 3-D goal/achieved-goal position (rotate the planar part)."""
    g = np.array(goal_xyz, dtype=np.float64, copy=True)
    xy = np.stack([g[..., 0], g[..., 1]], axis=-1) @ _rot(theta).T
    g[..., 0], g[..., 1] = xy[..., 0], xy[..., 1]
    return g
