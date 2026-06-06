r"""Guard for Step 72 / Stage 1 — the SO(2)-z symmetry of the FetchPush state (experiments/fetchpush_symmetry.py).

Project rule: every symmetry the model relies on gets an equivariance unit test BEFORE any result trusts it. Here we
verify (i) the layout covers all 25 dims; (ii) ``rotate_obs`` is a genuine SO(2) *representation* (composition adds
angles; identity at 0); (iii) the load-bearing consistency ``obs_to_vn(rotate_obs(obs, theta))`` rotates each VN vector
by the exact orthogonal $R(\theta)$ and leaves the scalars invariant — so a VN encoder on these vectors is
SO(2)-equivariant by construction. An optional real-env sanity (skipped if gymnasium-robotics is absent) checks the
split matches a true FetchPush obs.

Run:  .venv/bin/python tests/test_fetchpush_symmetry.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import fetchpush_symmetry as sym  # noqa: E402


def test_layout_covers_all_dims() -> None:
    assert sym.N_VEC == 7 and sym.N_SCA == 12, (sym.N_VEC, sym.N_SCA)
    covered = {i for xy in sym.PLANAR_XY for i in xy} | {sym.YAW_IDX} | set(sym.SCALAR_IDX)
    assert covered == set(range(sym.OBS_DIM)), "layout does not partition the 25 dims exactly"
    print(f"PASS: layout partitions all {sym.OBS_DIM} dims ({sym.N_VEC} 2-vecs + {sym.N_SCA} scalars).")


def test_rotate_obs_is_representation() -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((8, sym.OBS_DIM))
    a, b = 0.7, -1.1
    # rho(0) = identity
    assert np.allclose(sym.rotate_obs(obs, 0.0), obs, atol=1e-12), "rho(0) != identity"
    # rho(a) o rho(b) = rho(a+b)
    lhs = sym.rotate_obs(sym.rotate_obs(obs, b), a)
    rhs = sym.rotate_obs(obs, a + b)
    assert np.allclose(lhs, rhs, atol=1e-10), "rho is not a representation (composition != angle sum)"
    print("PASS: rotate_obs is an SO(2) representation (rho(0)=I, rho(a)rho(b)=rho(a+b)).")


def test_obs_to_vn_equivariance() -> None:
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((16, sym.OBS_DIM))
    theta = 0.9
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v0, s0 = sym.obs_to_vn(obs)
    v1, s1 = sym.obs_to_vn(sym.rotate_obs(obs, theta))
    v0_rot = v0 @ R.T                                              # rotate every 2-vec by R(theta)
    assert np.allclose(v1, v0_rot, atol=1e-10), "VN vectors are not SO(2)-equivariant under rotate_obs"
    assert np.allclose(s1, s0, atol=1e-10), "VN scalars are not invariant under rotate_obs"
    # orthogonality of the per-vector action (JEPA cost stays rotation-invariant)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
    print("PASS: obs_to_vn equivariance — vectors transform by the exact orthogonal R(theta); scalars invariant.")


def test_real_env_obs_split_optional() -> None:
    try:
        import gymnasium as gym
        import gymnasium_robotics
    except Exception:
        print("SKIP: gymnasium-robotics not installed (core symmetry tests already passed).")
        return
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPush-v4")
    obs, _ = env.reset(seed=0)
    env.close()
    o = np.asarray(obs["observation"])
    assert o.shape == (sym.OBS_DIM,), f"real FetchPush obs is {o.shape}, expected ({sym.OBS_DIM},)"
    v, s = sym.obs_to_vn(o)
    assert v.shape == (sym.N_VEC, 2) and s.shape == (sym.N_SCA,)
    print(f"PASS: real FetchPush obs splits into {v.shape} vectors + {s.shape} scalars.")


if __name__ == "__main__":
    test_layout_covers_all_dims()
    test_rotate_obs_is_representation()
    test_obs_to_vn_equivariance()
    test_real_env_obs_split_optional()
