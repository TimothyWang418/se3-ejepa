r"""P4-step1 gates: predictor equivariance (exact), full-stack sign calibration, env smoke (G0a).

Three layers of test, in dependency order:

1. **CNRegularPredictor fiber equivariance (exact).** With $\rho(g) = \mathrm{roll}(\cdot, g)$ on
   the group axis and $R(\theta_g)$ the CCW rotation by $2\pi g/N$:
   $f(\rho(g)z, R(\theta_g)a) = \rho(g) f(z, a)$ to float precision — no pixels involved, so no
   interpolation floor. This is the new math of P4-step1; it must be exact.
2. **Full-stack sign calibration.** e2cnn's fiber permutation convention (the encoder's
   ``GeometricTensor.transform``) is determined *empirically*, and the matching action-rotation
   sign $s \in \{+1,-1\}$ for the full stack $f(E(g\cdot x), R(s\,\theta_g)a) \approx \rho_{\rm
   e2cnn}(g) f(E(x), a)$ is found and printed. Exactly one sign must pass (within the encoder's
   grid-angle floor). The experiment pipeline must use the recorded sign for augmentation.
3. **G0a env smoke.** ``gym.make("swm/PushT-v1", resolution=96, damping=0.95)`` — kwargs
   pass-through asserted (the κ lane's load-bearing assumption) + obs contract (pixels + state).

Run: .venv/bin/python tests/test_p4_step1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.models.cn_regular import CNRegularPredictor  # noqa: E402

N_ROT = 16
LATENT = 128
TOL_EXACT = 1e-5


def rot(theta: float) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    return torch.tensor([[c, -s], [s, c]])


def test_predictor_fiber_equivariance() -> float:
    r"""max over g of |f(rho(g)z, R(theta_g)a) - rho(g)f(z,a)| — must be float-exact."""
    torch.manual_seed(0)
    pred = CNRegularPredictor(latent_dim=LATENT, n_rot=N_ROT, hidden_fields=32)
    pred.eval()
    B = 4
    z = torch.randn(B, LATENT)
    # relative-mode displacement actions in [-1,1]^2 (rotate about the origin)
    a = pred.a_center + torch.empty(B, 2).uniform_(-1, 1)

    worst = 0.0
    with torch.no_grad():
        base = pred(z, a)  # (B, F*N)
        for g in range(N_ROT):
            theta = 2.0 * torch.pi * g / N_ROT
            zg = torch.roll(z.view(B, -1, N_ROT), shifts=g, dims=-1).reshape(B, LATENT)
            ag = pred.a_center + ((a - pred.a_center) @ rot(theta).T)
            left = pred(zg, ag)
            right = torch.roll(base.view(B, -1, N_ROT), shifts=g, dims=-1).reshape(B, LATENT)
            worst = max(worst, (left - right).abs().max().item())
    assert worst < TOL_EXACT, f"predictor fiber equivariance broken: max err {worst:.3e}"
    return worst


def test_fullstack_sign_calibration() -> tuple[int, float]:
    r"""Determine the action-rotation sign matching e2cnn's fiber convention; one sign must pass.

    Uses C_4 (90-degree grid) where the encoder's pixel rotation is interpolation-free, so the
    only tolerance needed is numerical. Returns (sign, max_err at that sign).
    """
    from e2cnn import nn as enn  # noqa: E402

    from src.models.eqjepa import SteerableEncoder  # noqa: E402

    torch.manual_seed(0)
    n_rot = 4  # exact pixel rotations only — isolates the SIGN question from the interp floor
    enc = SteerableEncoder(in_channels=3, latent_dim=LATENT, n_rot=n_rot)
    enc.eval()
    pred = CNRegularPredictor(latent_dim=LATENT, n_rot=n_rot, hidden_fields=16)
    pred.eval()

    x = torch.randn(2, 3, 65, 65)
    a = pred.a_center + torch.empty(2, 2).uniform_(-1, 1)
    gx = enn.GeometricTensor(x, enc.in_type)

    results = {}
    with torch.no_grad():
        z = enc.encode_geometric(x)  # GeometricTensor
        f_base = pred(z.tensor.flatten(1), a)
        for g in enc.r2.testing_elements:
            gi = int(g)
            if gi == 0:
                continue
            theta = 2.0 * torch.pi * gi / n_rot
            z_gx = enc.encode_geometric(gx.transform(g).tensor).tensor.flatten(1)  # E(g.x)
            rho_f = (
                enn.GeometricTensor(f_base.view(2, LATENT, 1, 1), enc.out_type).transform(g).tensor.flatten(1)
            )  # rho_e2cnn(g) f(z, a)
            for s in (+1, -1):
                ag = pred.a_center + ((a - pred.a_center) @ rot(s * theta).T)
                err = (pred(z_gx, ag) - rho_f).abs().max().item()
                results[(gi, s)] = err

    per_sign = {s: max(results[(gi, s)] for gi in range(1, n_rot)) for s in (+1, -1)}
    good = [s for s, e in per_sign.items() if e < 1e-3]
    assert len(good) == 1, f"sign calibration ambiguous/failed: per-sign max errs {per_sign}"
    return good[0], per_sign[good[0]]


def test_g0a_env_smoke() -> dict:
    r"""G0a: kwargs pass-through (resolution, damping) + obs contract."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/* envs)

    out = {}
    # default: quasi-static anchor
    env0 = gym.make("swm/PushT-v1", resolution=96)
    obs, _ = env0.reset(seed=0)
    out["default_damping"] = float(env0.unwrapped.space.damping)
    assert out["default_damping"] == 0.0, f"default damping changed: {out['default_damping']}"
    assert "state" in obs, f"obs keys: {list(obs.keys())}"
    # pixels come from render() (render_mode='rgb_array' is the env default)
    px = env0.render()
    assert px.shape[:2] == (96, 96), f"resolution kwarg ignored: {px.shape}"
    # state layout: [agent_x, agent_y, block_x, block_y, block_theta, agent_vx, agent_vy]
    out["state_dim"] = int(len(obs["state"]))
    assert out["state_dim"] == 7, f"state layout changed: dim {out['state_dim']}"
    out["relative_actions"] = bool(env0.unwrapped.relative)
    assert out["relative_actions"], "expected relative=True default (displacement actions)"
    env0.close()

    # kappa knob pass-through
    env1 = gym.make("swm/PushT-v1", resolution=96, damping=0.95)
    env1.reset(seed=0)
    out["damping_passthrough"] = float(env1.unwrapped.space.damping)
    assert abs(out["damping_passthrough"] - 0.95) < 1e-9, (
        f"damping kwarg NOT passed through: {out['damping_passthrough']}"
    )
    env1.close()
    return out


if __name__ == "__main__":
    e1 = test_predictor_fiber_equivariance()
    print(f"[1] predictor fiber equivariance (C_{N_ROT}, exact): max err {e1:.3e}  PASS")
    s, e2 = test_fullstack_sign_calibration()
    print(f"[2] full-stack sign calibration (C_4 grid): sign={s:+d}, max err {e2:.3e}  PASS")
    g0a = test_g0a_env_smoke()
    print(f"[3] G0a env smoke: {g0a}  PASS")
    print("ALL P4-step1 gate tests PASS")
