r"""Step 32 — the **tensor-product degree ladder** predictor (:class:`VNTPLadderPredictor`).

Step 27 showed a *single* tensor-product stack recovers a substantial part of the degree-1
Vector-Neuron capacity cap diagnosed in Step 24. Step 32 turns that one architectural point into a
**ladder**: a predictor whose maximum representable polynomial degree $d_{\max}=2^{L}$ is set by the
number $L$ of front-loaded tensor-product blocks. The empirical *recovery curve vs degree* lives in
``experiments/step32_tp_degree_ladder.py``; this test certifies the two structural invariants the
experiment relies on, at **every** rung $L\in\{0,1,2,3\}$:

1. **SO(3) equivariance** (to the float floor) of the full predictor at every rung,
   $f(\rho(R)z, Ra)=\rho(R)\,f(z,a)$ — the "exact-and-flat **for free at every degree**" claim;
2. a **degree-separation witness** on the residual update $\delta=f(z,a)-z$: rung $L=0$ is exactly
   degree-1 homogeneous ($\delta(\lambda x)=\lambda\,\delta(x)$), while every rung $L\ge1$ is **not**
   (it carries genuine higher-degree, $\ge$ degree-2, content) — so the ladder really does climb in
   representable degree, not just in parameters.

Run:
    .venv/bin/python tests/test_step32_degree_ladder.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.models.structured import VNTPLadderPredictor  # noqa: E402

torch.set_default_dtype(torch.float32)

LADDER = (0, 1, 2, 3)  # the rungs under test: L = number of tensor-product blocks (d_max = 2^L)


def rand_so3(gen: torch.Generator) -> torch.Tensor:
    r"""A Haar-random **proper** rotation $R\in\mathrm{SO}(3)$ via QR. ``-> (3, 3)``."""
    a = torch.randn(3, 3, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def rotate_channels(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""Apply $v\mapsto Rv$ to every type-1 vector. ``(..., 3) -> (..., 3)``."""
    return x @ R.transpose(-1, -2)


@torch.no_grad()
def predictor_equivariance_error(
    pred: VNTPLadderPredictor, z: torch.Tensor, a: torch.Tensor, R: torch.Tensor
) -> float:
    r"""$\max\lVert f(\rho(R)z, Ra)-\rho(R)f(z,a)\rVert_\infty$ over the batch."""
    b = z.shape[0]
    z_rot = rotate_channels(z.reshape(b, -1, 3), R).reshape(b, -1)  # rho(R) z
    a_rot = rotate_channels(a.reshape(b, -1, 3), R).reshape(b, -1)  # R a
    left = pred(z_rot, a_rot)  # f(rho(R)z, Ra)
    right = rotate_channels(pred(z, a).reshape(b, -1, 3), R).reshape(b, -1)  # rho(R) f(z, a)
    return (left - right).abs().max().item()


@torch.no_grad()
def degree1_violation(pred: VNTPLadderPredictor, z: torch.Tensor, a: torch.Tensor, lam: float) -> float:
    r"""Relative failure of degree-1 homogeneity of the residual $\delta=f(z,a)-z$.

    $\big\lVert\delta(\lambda z,\lambda a)-\lambda\,\delta(z,a)\big\rVert\big/\big\lVert\lambda\,\delta(z,a)\big\rVert$.
    Zero (float floor) iff $\delta$ is purely degree-1; strictly positive once $\delta$ carries any
    higher-degree (tensor-product) content.
    """
    b = z.shape[0]
    delta = pred(z, a) - z
    delta_scaled = pred(lam * z, lam * a) - lam * z
    num = (delta_scaled - lam * delta).norm().item()
    den = (lam * delta).norm().item()
    return num / max(den, 1e-12)


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    print("Step 32 — tensor-product degree ladder: SO(3) equivariance + degree separation\n")
    # latent 48 = 16 type-1 vectors, action 6 = 2 type-1 vectors (the Step-24/27 per-object shape).
    z = torch.randn(8, 48, generator=gen)
    a = torch.randn(8, 6, generator=gen)

    print(f"  {'rung L':>7s} | {'d_max=2^L':>9s} | {'max SO(3) resid':>15s} | {'deg-1 violation':>15s}")
    print("  " + "-" * 56)
    for L in LADDER:
        pred = VNTPLadderPredictor(latent_dim=48, action_dim=6, hidden=64, dim=3,
                                   n_tp_blocks=L, n_blocks_total=3).eval()

        # ---- 1. SO(3) equivariance at this rung ----
        worst = max(predictor_equivariance_error(pred, z, a, rand_so3(gen)) for _ in range(20))

        # ---- 2. degree separation: L=0 is degree-1, L>=1 is not ----
        viol = degree1_violation(pred, z, a, lam=2.0)

        dmax = 2 ** L
        print(f"  {L:>7d} | {dmax:>9d} | {worst:15.2e} | {viol:15.2e}")
        assert worst < 1e-4, f"rung L={L} broke SO(3) equivariance: {worst:.2e}"
        if L == 0:
            assert viol < 1e-4, f"rung L=0 should be degree-1 homogeneous, got violation {viol:.2e}"
        else:
            assert viol > 1e-2, (f"rung L={L} should carry higher-degree content "
                                 f"(deg-1 violation > 1e-2), got {viol:.2e}")

    print("\nPASS: every ladder rung is exactly SO(3)-equivariant (< 1e-4); L=0 is degree-1 and")
    print("every L>=1 carries genuine higher-degree tensor-product content.")
    print("=> the predictor can climb the representable-degree ladder WITHOUT spending equivariance.")


def test_vntp_ladder_equivariance_and_degree() -> None:
    """pytest: VNTPLadderPredictor SO(3)-equivariant at every rung; L=0 degree-1, L>=1 higher-degree."""
    main()


if __name__ == "__main__":
    main()
