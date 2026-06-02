r"""Exact SE(3) + permutation equivariance of the *attention* pooling (Step 46).

Steps 42-44 localized the residual interaction cap to a single component of the
:class:`~src.models.se3.SE3PointEncoder`: the permutation-invariant **sum-pool**
$h=\sum_k \mathrm{msg}_k$. The design-rule cure ("enrich the aggregator, keep the
prior") replaces that sum with a multi-head **equivariant attention pool**
(``pool="attn"``): per-point scores read off the *invariant* $\ell=0$ channels,
$\mathrm{softmax}$ over the $N$ points, $K$ distinct weighted sums of the equivariant
per-point features, then an ``o3.Linear`` recombination.

A richer aggregator is only admissible if it does **not** spend the prior it was
meant to keep. So before trusting any downstream number we verify, to the float
floor (~$10^{-5}$), that the attention pool is still:

1. **Translation invariant**: $E(x+t)=E(x)$ (the cloud is centred upstream).
2. **SO(3) equivariant**: $E(R\,x)=\rho(R)\,E(x)$ for arbitrary $R\in\mathrm{SO}(3)$
   — the attention weights are invariant scalars, so the weighted sum of the
   equivariant ``msg`` stays equivariant, and ``o3.Linear`` preserves the type.
3. **Permutation invariant**: $E(\pi\cdot x)=E(x)$ — $\mathrm{softmax}$+sum over the
   $N$ points is symmetric, so re-indexing the cloud leaves the latent unchanged.

If any of these breaks, the cure is a confound (it bought capacity by leaking the
symmetry), and its in-distribution gains would be uninterpretable. We also re-check
the original ``pool="sum"`` encoder so the additive ``pool`` option is proven not to
regress the published default.

Run:
    .venv/bin/python tests/test_step46_attn_pool_equivariance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from src.models.se3 import SE3PointEncoder  # noqa: E402


def rotate_points(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Apply $v \mapsto R v$ to every vector in ``x: (..., 3)`` (right-multiply)."""
    return x @ R.transpose(-1, -2)


@torch.no_grad()
def translation_error(enc: SE3PointEncoder, pos: torch.Tensor) -> float:
    r"""$\max\lvert E(x+t) - E(x)\rvert$ for a random global shift $t$."""
    t = torch.randn(1, 1, 3)
    return (enc(pos + t) - enc(pos)).abs().max().item()


@torch.no_grad()
def equivariance_error(enc: SE3PointEncoder, pos: torch.Tensor, R: torch.Tensor) -> float:
    r"""$\max\lvert E(R x) - R\,E(x)\rvert$ on the $\ell=1$ vector latent."""
    z = enc.encode_vectors(pos)  # (B, n_vec, 3)
    left = enc.encode_vectors(rotate_points(pos, R))  # E(R.x)
    right = rotate_points(z, R)  # R.E(x)
    return (left - right).abs().max().item()


@torch.no_grad()
def permutation_error(enc: SE3PointEncoder, pos: torch.Tensor) -> float:
    r"""$\max\lvert E(\pi\cdot x) - E(x)\rvert$ for a random point permutation $\pi$."""
    perm = torch.randperm(pos.shape[1])
    return (enc(pos[:, perm, :]) - enc(pos)).abs().max().item()


@torch.no_grad()
def _audit(pool: str, n_heads: int = 4) -> float:
    torch.manual_seed(0)
    enc = SE3PointEncoder(n_out_vec=16, lmax=2, mul=8, pool=pool, n_heads=n_heads).eval()
    B, N = 6, 24
    pos = torch.randn(B, N, 3)

    e_trans = translation_error(enc, pos)
    e_perm = permutation_error(enc, pos)
    e_eq = max(equivariance_error(enc, pos, o3.rand_matrix()) for _ in range(8))
    worst = max(e_trans, e_perm, e_eq)

    tag = f'pool="{pool}"' + (f" (K={n_heads})" if pool == "attn" else "")
    print(f"\n    {tag}: latent = {enc.n_out_vec} type-1 vectors")
    print("    test                              | error")
    print("    ----------------------------------+----------")
    print(f"    translation invariance E(x+t)=E(x)|  {e_trans:.2e}")
    print(f"    SO(3) equivariance E(Rx)=R.E(x)   |  {e_eq:.2e}")
    print(f"    permutation invariance E(pi.x)    |  {e_perm:.2e}")
    print(f"    worst                             |  {worst:.2e}")
    return worst


def main() -> None:
    print("Step 46 — attention-pool encoder: exact SE(3) + permutation symmetry\n")
    worst_sum = _audit("sum")
    worst_attn = _audit("attn", n_heads=4)
    worst = max(worst_sum, worst_attn)
    print(f"\n    worst over both pools = {worst:.2e}")
    assert worst_sum < 1e-4, f'pool="sum" regressed: worst={worst_sum:.2e}'
    assert worst_attn < 1e-4, f'pool="attn" not exactly equivariant: worst={worst_attn:.2e}'
    print('PASS: attention pooling is exactly SE(3)-equivariant + permutation-invariant')
    print("=> the design-rule cure enriches the aggregator WITHOUT spending the prior.")


def test_attn_pool_se3_and_permutation_equivariance() -> None:
    """pytest entry point: the ``pool="attn"`` cure keeps every symmetry of ``pool="sum"``."""
    main()


if __name__ == "__main__":
    main()
