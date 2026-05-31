r"""Step 33 — learnable Lie-algebra **symmetry discovery** (``experiments/step33_symmetry_discovery.py``).

The experiment lets gradient descent rediscover the generators of the world's symmetry from a blank
slate of $3\times3$ matrices. This test certifies the structural invariants that experiment relies on:

1. **$\mathfrak{so}(3)$ reference basis** is orthonormal, antisymmetric, and closes under the Lie
   bracket with the unit-frame structure constant $[\hat L_x,\hat L_y]=\tfrac{1}{\sqrt2}\hat L_z$.
2. **TRUE teacher is exactly $\mathrm{SO}(3)$-equivariant** (the symmetry we try to recover).
3. **BROKEN teacher has exactly the intended residual symmetry**: it is equivariant under rotations
   about $z$ (the surviving $\mathrm{SO}(2)$) but **not** under a generic rotation — so the
   falsifiability control really does break $\mathrm{SO}(3)\to\mathrm{SO}(2)_z$ and nothing else.
4. **The orthonormal frame** carried during discovery is genuinely orthonormal (QR).
5. **The discovery pipeline recovers the symmetry** end-to-end (fast settings): on TRUE with $K{=}3$ it
   lands in $\mathfrak{so}(3)$ (antisymmetric, bracket-closing); on BROKEN with $K{=}1$ it lands on $L_z$.

Run:
    .venv/bin/python tests/test_step33_symmetry_discovery.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step13_se3_latent_jepa import rand_so3, rot_z  # noqa: E402
from step24_object_interaction import N_OBJ, make_interacting_transitions  # noqa: E402
from step33_symmetry_discovery import (  # noqa: E402
    B_SO3,
    LZ_HAT,
    antisym_frac,
    apply_g,
    axis_align,
    broken_teacher,
    closure_resid,
    discover,
    frac_in_so3,
    orthonormalize,
    so3_basis,
    true_teacher,
)

torch.set_default_dtype(torch.float32)


@torch.no_grad()
def teacher_equiv_error(teacher, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> float:
    r"""$\max\lVert f(R\!\cdot\!S, R\!\cdot\!A) - R\!\cdot\! f(S,A)\rVert_\infty$ for a single $R$."""
    lhs = teacher(s @ r.transpose(-1, -2), a @ r.transpose(-1, -2))
    rhs = teacher(s, a) @ r.transpose(-1, -2)
    return (lhs - rhs).abs().max().item()


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    print("Step 33 — symmetry discovery: structural invariants + end-to-end recovery\n")

    # ---- 1. so(3) reference basis: orthonormal, antisymmetric, closes under the bracket ----
    b = so3_basis()                                            # (3, 9) rows
    gram = b @ b.transpose(-1, -2)
    assert torch.allclose(gram, torch.eye(3), atol=1e-6), "so(3) basis not orthonormal"
    mats = b.reshape(3, 3, 3)
    asym = max((m + m.transpose(-1, -2)).abs().max().item() for m in mats)
    assert asym < 1e-6, f"so(3) basis not antisymmetric: {asym:.2e}"
    # [Lx_hat, Ly_hat] = (1/sqrt2) Lz_hat  (unit-frame structure constant)
    lx, ly, lz = mats
    bracket = lx @ ly - ly @ lx
    close = (bracket - (1.0 / math.sqrt(2.0)) * lz).abs().max().item()
    assert close < 1e-6, f"so(3) bracket [Lx,Ly] != (1/sqrt2) Lz: {close:.2e}"
    print(f"  [1] so(3) basis: orthonormal (gram-I {((gram - torch.eye(3)).abs().max()):.1e}), "
          f"antisym ({asym:.1e}), bracket-closes ({close:.1e})  OK")

    # ---- data shared by the teacher checks ----
    s, a_flat, _ = make_interacting_transitions(48, seed=0)
    a = a_flat.reshape(48, N_OBJ, 3)

    # ---- 2. TRUE teacher is exactly SO(3)-equivariant ----
    true_worst = max(teacher_equiv_error(true_teacher, s, a, rand_so3(gen)) for _ in range(10))
    assert true_worst < 1e-5, f"TRUE teacher not SO(3)-equivariant: {true_worst:.2e}"
    print(f"  [2] TRUE teacher SO(3)-equivariant: worst residual over 10 random R = {true_worst:.2e}  OK")

    # ---- 3. BROKEN teacher: equivariant under R_z, NOT under a generic rotation ----
    rz_err = max(teacher_equiv_error(broken_teacher, s, a, rot_z(deg))
                 for deg in (17.0, 53.0, 121.0, 210.0))
    generic_err = min(teacher_equiv_error(broken_teacher, s, a, rand_so3(gen)) for _ in range(10))
    assert rz_err < 1e-5, f"BROKEN teacher should stay equivariant under R_z, got {rz_err:.2e}"
    assert generic_err > 1e-2, (f"BROKEN teacher should break under a generic rotation, "
                                f"got only {generic_err:.2e}")
    print(f"  [3] BROKEN teacher: R_z-equivariant ({rz_err:.2e}) but breaks under generic R "
          f"(min {generic_err:.2e})  OK  -> exactly SO(3)->SO(2)_z")

    # ---- 4. orthonormalize carries a genuine orthonormal frame ----
    v = torch.randn(3, 9, generator=gen)
    q = orthonormalize(v)
    qgram = q @ q.transpose(-1, -2)
    assert torch.allclose(qgram, torch.eye(3), atol=1e-5), "orthonormalize did not produce an orthonormal frame"
    print(f"  [4] orthonormalize: gram-I = {((qgram - torch.eye(3)).abs().max()):.1e}  OK")

    # ---- 5. end-to-end recovery (fast settings) ----
    gh_true, r_true = discover(true_teacher, 3, s, a, n_steps=400, seed=0, n_restarts=2)
    f_so3, anti, clo = frac_in_so3(gh_true), antisym_frac(gh_true), closure_resid(gh_true)
    assert f_so3 > 0.9, f"TRUE@K=3 should span so(3), got frac_in_so3={f_so3:.3f}"
    assert anti < 0.15, f"TRUE@K=3 generators should be antisymmetric, got sym-frac={anti:.3f}"
    assert clo < 0.15, f"TRUE@K=3 should close under the bracket, got closure={clo:.3f}"
    gh_brk, r_brk = discover(broken_teacher, 1, s, a, n_steps=400, seed=0, n_restarts=2)
    align = axis_align(gh_brk)
    assert align > 0.85, f"BROKEN@K=1 should recover L_z, got |<G,L_z>|={align:.3f}"
    print(f"  [5] recovery: TRUE@K=3 frac_in_so3={f_so3:.3f} antisym={anti:.3f} closure={clo:.3f} "
          f"(resid {r_true:.1e}); BROKEN@K=1 |<G,L_z>|={align:.3f} (resid {r_brk:.1e})  OK")

    print("\nPASS: so(3) basis + teachers + frame are structurally sound, and the discovery pipeline")
    print("recovers so(3) on the symmetric world and L_z on the rotation-broken world.")


def test_step33_symmetry_discovery() -> None:
    """pytest: so(3) basis, teacher (broken-)symmetry, orthonormal frame, and end-to-end recovery."""
    main()


if __name__ == "__main__":
    main()
