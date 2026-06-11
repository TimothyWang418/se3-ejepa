r"""P4 3D lane G0 gates — VN equivariance (V-I..V) + VN pairing equality (V-VI).

Registered before any 3D training/audit (spec 2026-06-12-p4-3d-lane-bootstrap.md):

- V-I   VNLinear exact equivariance (f64, 1e-12) — associativity, no learning involved.
- V-II  VNLeakyReLU equivariance (f64, 1e-10) — invariant gate + co-rotating direction.
- V-III encoder end-to-end: $E(X R^\top) = \rho(R)\, E(X)$ (f64, 1e-8; knn regraph included).
- V-IV  translation invariance: $E(X + t) = E(X)$ (centering quotient; f64, 1e-9).
- V-V   predictor equivariance: $f(\rho(R) z, R \cdot a) = \rho(R) f(z, a)$ (f64, 1e-10).
- V-VI  pairing equality (incident #9's lesson transplanted): after an EMA-style update,
        in-process target == save/load-reloaded target **bit-exactly** on fresh inputs
        (plain PyTorch modules — no e2cnn caches — but the gate is installed BEFORE any
        audit, not after the first inconsistency).

Run: .venv/bin/python tests/test_p4_3d_g0.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vn_jepa import (  # noqa: E402
    VNDGCNNEncoder, VNLeakyReLU, VNLinear, VNPredictor, random_rotation, rotate_action,
    rotate_latent,
)

torch.manual_seed(0)
G = torch.Generator().manual_seed(123)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def t1_vnlinear() -> None:
    lin = VNLinear(8, 16).double()
    v = torch.randn(4, 8, 3, dtype=torch.float64, generator=G)
    r = random_rotation(generator=G)
    d = _maxdiff(lin(v @ r.T), lin(v) @ r.T)
    assert d < 1e-12, f"V-I FAIL: VNLinear equivariance off by {d:.2e}"
    print(f"V-I   PASS  VNLinear        maxdiff {d:.2e}")


def t2_vnrelu() -> None:
    act = VNLeakyReLU(8).double()
    v = torch.randn(4, 8, 3, dtype=torch.float64, generator=G)
    r = random_rotation(generator=G)
    d = _maxdiff(act(v @ r.T), act(v) @ r.T)
    assert d < 1e-10, f"V-II FAIL: VNLeakyReLU equivariance off by {d:.2e}"
    print(f"V-II  PASS  VNLeakyReLU     maxdiff {d:.2e}")


def t3_encoder() -> None:
    enc = VNDGCNNEncoder(c_vec=8, k=8, width=16).double()
    x = torch.randn(2, 64, 3, dtype=torch.float64, generator=G)   # generic cloud: no knn ties
    r = random_rotation(generator=G)
    d = _maxdiff(enc(x @ r.T), rotate_latent(enc(x), r, enc.c_vec))
    assert d < 1e-8, f"V-III FAIL: encoder equivariance off by {d:.2e}"
    print(f"V-III PASS  encoder SO(3)   maxdiff {d:.2e}")


def t4_translation() -> None:
    enc = VNDGCNNEncoder(c_vec=8, k=8, width=16).double()
    x = torch.randn(2, 64, 3, dtype=torch.float64, generator=G)
    t = torch.randn(1, 1, 3, dtype=torch.float64, generator=G) * 5.0
    d = _maxdiff(enc(x + t), enc(x))
    assert d < 1e-9, f"V-IV FAIL: translation invariance off by {d:.2e}"
    print(f"V-IV  PASS  translation     maxdiff {d:.2e}")


def t5_predictor() -> None:
    c_vec, n_inv = 8, 24
    pred = VNPredictor(c_vec, n_inv).double()
    z = torch.randn(4, 3 * c_vec + n_inv, dtype=torch.float64, generator=G)
    a = torch.randn(4, 7, dtype=torch.float64, generator=G)       # Δp(3)+ω(3)+gripper(1)
    r = random_rotation(generator=G)
    d = _maxdiff(pred(rotate_latent(z, r, c_vec), rotate_action(a, r)),
                 rotate_latent(pred(z, a), r, c_vec))
    assert d < 1e-10, f"V-V FAIL: predictor equivariance off by {d:.2e}"
    print(f"V-V   PASS  predictor       maxdiff {d:.2e}")


def t6_pairing_equality() -> None:
    r"""EMA pair save/load == in-process, bit-exact (the deployable pair is (target, pred))."""
    enc = VNDGCNNEncoder(c_vec=8, k=8, width=16)
    pred = VNPredictor(8, 24)
    target = copy.deepcopy(enc)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    x = torch.randn(8, 64, 3, generator=G)
    a = torch.randn(8, 7, generator=G)
    for _ in range(5):                                            # 5 train+EMA steps
        z = enc(x)
        with torch.no_grad():
            zt = target(x)
        loss = (pred(z, a) - zt).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                     # EMA update on eval-mode target
            for p_t, p_o in zip(target.parameters(), enc.parameters()):
                p_t.mul_(0.99).add_(p_o, alpha=0.01)
    fresh = torch.randn(4, 64, 3, generator=G)
    fa = torch.randn(4, 7, generator=G)
    with tempfile.TemporaryDirectory() as td:
        pt = Path(td) / "pair.pt"
        torch.save({"target": target.state_dict(), "pred": pred.state_dict()}, pt)
        ck = torch.load(pt, weights_only=True)
        target2 = VNDGCNNEncoder(c_vec=8, k=8, width=16)
        pred2 = VNPredictor(8, 24)
        target2.load_state_dict(ck["target"]); pred2.load_state_dict(ck["pred"])
    with torch.no_grad():
        d_t = _maxdiff(target(fresh), target2(fresh))
        d_p = _maxdiff(pred(target(fresh), fa), pred2(target2(fresh), fa))
    assert d_t == 0.0 and d_p == 0.0, f"V-VI FAIL: pairing not bit-exact ({d_t:.2e}/{d_p:.2e})"
    print(f"V-VI  PASS  pairing equality  target {d_t} | pred∘target {d_p} (bit-exact)")


if __name__ == "__main__":
    for t in (t1_vnlinear, t2_vnrelu, t3_encoder, t4_translation, t5_predictor,
              t6_pairing_equality):
        t()
    print("ALL G0 MODEL GATES PASS")
