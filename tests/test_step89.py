r"""Guard for Step 89 — certifying a PUBLIC PRETRAINED world model (TD-MPC2).

Checkpoint-free unit tests for the faithful slice rebuild in ``experiments/step89_pretrained_wm_audit.py``: the layer
math must match tdmpc2/common/layers.py EXACTLY (SimNorm = per-group softmax over groups of ``simnorm_dim``;
NormedLinear = Mish(LayerNorm(Linear)); mlp = NormedLinear hidden + NormedLinear(act)/Linear tail), the key-subset
loader must accept a synthetic state_dict in that layout, and the autonomous policy-prior loop $g(z)=d(z,\tanh(\mu_\pi
(z)))$ must be differentiable with a square Jacobian. Fidelity to the REAL checkpoint is the experiment's G0 (smoke).
"""
import torch

from experiments import step89_pretrained_wm_audit as s89

DT = torch.float64


def test_simnorm_is_per_group_softmax():
    r"""SimNorm reshapes (..., L) into groups of ``dim`` and softmaxes each: outputs positive, each group sums to 1,
    and matches a hand-computed softmax on the first group."""
    sn = s89.SimNorm(dim=8)
    x = torch.randn(3, 32, dtype=DT)
    y = sn(x)
    assert y.shape == x.shape and (y > 0).all()
    sums = y.view(3, 4, 8).sum(-1)
    assert torch.allclose(sums, torch.ones(3, 4, dtype=DT), atol=1e-12)
    assert torch.allclose(y[0, :8], torch.softmax(x[0, :8], dim=-1), atol=1e-12)


def test_normed_linear_is_mish_layernorm_linear():
    r"""NormedLinear forward = act(LayerNorm(Linear(x))) with Mish default — verified against the explicit composition."""
    torch.manual_seed(0)
    nl = s89.NormedLinear(5, 7).double()
    x = torch.randn(4, 5, dtype=DT)
    expect = torch.nn.functional.mish(nl.ln(torch.nn.functional.linear(x, nl.weight, nl.bias)))
    assert torch.allclose(nl(x), expect, atol=1e-12)


def test_mlp_layout_matches_tdmpc2():
    r"""s89.mlp([in]+2*[h]+[out], act=SimNorm) = two NormedLinear hidden + a NormedLinear tail carrying SimNorm —
    state_dict keys {i}.weight/{i}.bias/{i}.ln.weight/{i}.ln.bias for i=0,1,2; with act=None the tail is a plain
    Linear (keys 2.weight/2.bias only)."""
    m = s89.mlp(6, [8, 8], 16, act=s89.SimNorm(8))
    keys = set(m.state_dict().keys())
    assert {"0.weight", "0.ln.weight", "1.ln.bias", "2.weight", "2.ln.weight"} <= keys
    m2 = s89.mlp(6, [8, 8], 4, act=None)
    k2 = set(m2.state_dict().keys())
    assert "2.weight" in k2 and "2.ln.weight" not in k2


def test_loader_accepts_synthetic_state_dict_and_g_is_differentiable():
    r"""Build a synthetic checkpoint in the TD-MPC2 key layout (_encoder.state / _dynamics / _pi + extra Q-keys that
    must be IGNORED), load it through the slice loader, and check the autonomous loop $g$ runs, is differentiable, and
    yields a square (L, L) Jacobian."""
    L, A, obs, h, enc_h, sim = 16, 2, 5, 12, 10, 8
    enc = s89.mlp(obs, [enc_h], L, act=s89.SimNorm(sim))
    dyn = s89.mlp(L + A, [h, h], L, act=s89.SimNorm(sim))
    pi = s89.mlp(L, [h, h], 2 * A, act=None)
    sd = {f"_encoder.state.{k}": v for k, v in enc.state_dict().items()}
    sd |= {f"_dynamics.{k}": v for k, v in dyn.state_dict().items()}
    sd |= {f"_pi.{k}": v for k, v in pi.state_dict().items()}
    sd |= {"_Qs.params.0": torch.zeros(3), "_target_Qs.params.0": torch.zeros(3)}   # must be ignored
    slices = s89.load_tdmpc2_slices({"model": sd}, latent_dim=L, action_dim=A, obs_dim=obs,
                                    enc_dim=enc_h, mlp_dim=h, simnorm_dim=sim)
    z = torch.randn(L, dtype=DT)
    g = s89.make_autonomous(slices)
    out = g(z)
    assert out.shape == (L,)
    J = torch.autograd.functional.jacobian(g, z)
    assert J.shape == (L, L) and torch.isfinite(J).all()
    # encode round-trip: obs -> z lands on the SimNorm simplex product (groups sum to 1)
    z0 = slices.encode(torch.randn(obs, dtype=DT))
    assert torch.allclose(z0.view(-1, sim).sum(-1), torch.ones(L // sim, dtype=DT), atol=1e-10)
