r"""Unit guards for the Step-22 symmetry-break $\times$ data phase diagram.

Step 22 sweeps a $g\times N$ grid (symmetry-break strength $g$, data size $N$) and reads, at each
cell, the equivariant VN's vs the non-equivariant MLP's latent 1-step relMSE both **in-wedge**
(``seen``) and **across the whole group** (``ood``). The experiment's narrative rests on three
*model-free, exact* properties of the misspecified teacher and one pure interpolation helper -- this
file pins exactly those, so the phase diagram's axes and labels are trustworthy before any training:

  * **The knob is honest.** $\mathrm{Dyn}_g$ reduces to the exactly-SO(3) Step-13 teacher at $g=0$
    ($\mathrm{noneq}=0$), the added lab-$z$ term is **centering-invariant** (a *real* prediction
    target, not a translation the VN encoder washes out) yet lies in the **complement of the
    equivariant maps** (it genuinely breaks SO(3)), and the break is **monotone** in $g$ -- so $g$ /
    the non-equivariance fraction is a faithful x-axis.

  * **Why OOD must be re-sampled, not rotated** (the Step-16 methodological crux). At $g=0$ the
    teacher is equivariant, so a *rotated* held-out transition $(Rx,Ra,Rx')$ is a genuine label and
    Step 21's "rotate the test set" is valid. At $g>0$ that identity **fails**, so a rotated target
    is a *fake* label -- Steps 16/22 must therefore sample fresh full-SO(3) clouds and push them
    through the true $\mathrm{Dyn}_g$. This test shows the equivariance residual of the *data* jumps
    from the float floor to $O(1)$ the instant $g>0$, which is precisely what makes re-sampling
    mandatory.

  * **The phase-extraction is correct.** ``first_overtake_N`` (the in-distribution crossover
    $N^\star(g)$) interpolates / walls / orders as specified.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python tests/test_symmetry_data_phase.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                  # for `src.*`
sys.path.insert(0, str(ROOT / "experiments"))  # for the Step 13/16/22 helpers

import torch  # noqa: E402

from step13_se3_latent_jepa import rand_so3, rotate_points, teacher_step  # noqa: E402
from step16_misspecification import collect_transitions_g, noneq_fraction, teacher_step_g  # noqa: E402
from step22_symmetry_data_phase import first_overtake_N  # noqa: E402

# Tolerances. The SO(3) identity at g=0 sits at the e3nn/synthetic-teacher float floor; the break at
# g>0 is an O(1) effect, so the two are separated by orders of magnitude.
_EQ_FLOOR = 1e-5    # relative residual of the equivariance identity at g=0
_BREAK = 1e-2       # minimum genuine symmetry violation we require at g>0


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative RMS residual ``||a-b|| / ||b||`` (scale-free)."""
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12))


def test_teacher_g0_recovers_equivariant_teacher() -> None:
    r"""$\mathrm{Dyn}_0 = $ the exactly-SO(3) Step-13 teacher, and $\mathrm{noneq}(0)\approx 0$."""
    S, A, _ = collect_transitions_g(24, seed=5, g=0.0, full_so3=False)
    same = teacher_step_g(S, A, 0.0)
    base = teacher_step(S, A)
    assert torch.allclose(same, base, atol=1e-6), "Dyn_g at g=0 must equal the Step-13 equivariant teacher"
    nf0 = noneq_fraction(0.0, S, A)
    assert nf0 < 1e-6, f"g=0 must be SO(3)-equivariant (noneq={nf0:.2e}); the x-axis origin is exact symmetry"


def test_break_is_a_real_target_yet_breaks_so3() -> None:
    r"""The added lab-$z$ term is centering-invariant (real target), $z$-only, nonzero, and monotone-breaking."""
    S, A, _ = collect_transitions_g(24, seed=6, g=0.0, full_so3=False)
    g = 0.3
    grav = teacher_step_g(S, A, g) - teacher_step(S, A)             # the pure symmetry-breaking term
    # centering-invariant: it contributes nothing to the centroid (so the VN encoder cannot wash it out)
    assert grav.mean(dim=1).abs().max() < 1e-5, "break term must be centering-invariant (a real target, not a translation)"
    # it is a genuine, visible target and lies along the fixed lab axis e_z only
    assert grav.abs().max() > 1e-3, "break term must be nonzero (a real prediction target)"
    assert grav[..., :2].abs().max() < 1e-6, "the fixed-axis field must point along lab-z only"
    # and it genuinely breaks SO(3), monotonically in g
    seq = [noneq_fraction(gv, S, A) for gv in (0.0, 0.1, 0.4)]
    assert seq[1] > _BREAK, f"g=0.1 must already break SO(3) (noneq={seq[1]:.2e})"
    assert seq[0] <= seq[1] <= seq[2], f"the symmetry-break knob must be monotone in g (got {seq})"


def test_g0_label_genuine_under_rotation_but_gpos_is_fake() -> None:
    r"""Why Steps 16/22 re-sample OOD: the rotated label is exact at $g=0$ but a *fake* label once $g>0$."""
    S, A, _ = collect_transitions_g(16, seed=7, g=0.0, full_so3=False)
    R = rand_so3(torch.Generator().manual_seed(0))

    # g=0: Dyn_0(Rx,Ra) == R Dyn_0(x,a) -> rotating a held-out transition gives a GENUINE OOD label.
    lhs0 = teacher_step_g(rotate_points(S, R), rotate_points(A, R), 0.0)
    rhs0 = rotate_points(teacher_step_g(S, A, 0.0), R)
    assert _rel(lhs0, rhs0) < _EQ_FLOOR, "at g=0 the rotated target must be a genuine label (teacher is equivariant)"

    # g>0: the identity fails by O(1) -> a rotated target would be FAKE, so OOD must be re-sampled.
    g = 0.4
    lhsg = teacher_step_g(rotate_points(S, R), rotate_points(A, R), g)
    rhsg = rotate_points(teacher_step_g(S, A, g), R)
    assert _rel(lhsg, rhsg) > _BREAK, (
        "at g>0 the rotated target must NOT be a valid label (teacher not equivariant); "
        "this is why Steps 16/22 sample genuine full-SO(3) transitions instead of rotating the test set"
    )


def test_first_overtake_N_pure() -> None:
    r"""``first_overtake_N``: grid-point crossover, first-point win, the wall (None), and tie handling."""
    N_grid = [32, 64, 128, 256, 512]

    # MLP (challenger) overtakes the VN (base) between N=128 and N=256 -> returns 256 (grid resolution)
    base = [0.90, 0.80, 0.70, 0.60, 0.50]        # VN seen (descending)
    chall = [0.95, 0.85, 0.72, 0.55, 0.40]       # MLP seen: only below base from N=256 on
    assert first_overtake_N(N_grid, base, chall) == 256

    # challenger already wins at the first grid point
    assert first_overtake_N(N_grid, [1.0] * 5, [0.5] * 5) == 32

    # challenger never wins on the grid -> a wall (None): the prior holds in-distribution at all N
    assert first_overtake_N(N_grid, [0.3] * 5, [0.9] * 5) is None

    # exact ties do NOT count as overtaking (strict inequality)
    assert first_overtake_N([10, 20], [0.5, 0.5], [0.5, 0.4]) == 20


def main() -> None:
    """Print the teacher-knob probe (noneq + the g=0-vs-g>0 label residual) and run all assertions."""
    S, A, _ = collect_transitions_g(16, seed=7, g=0.0, full_so3=False)
    R = rand_so3(torch.Generator().manual_seed(0))
    print("Step 22 phase-diagram unit guards: the misspecified teacher's knob + the phase extractor\n")
    print(f"    {'g':>5} | {'noneq(g)':>10} | {'rotated-label rel-residual':>26}")
    print("    " + "-" * 48)
    for g in (0.0, 0.1, 0.2, 0.4, 0.8):
        lhs = teacher_step_g(rotate_points(S, R), rotate_points(A, R), g)
        rhs = rotate_points(teacher_step_g(S, A, g), R)
        print(f"    {g:>5.2f} | {noneq_fraction(g, S, A):>10.3e} | {_rel(lhs, rhs):>26.3e}")

    test_teacher_g0_recovers_equivariant_teacher()
    test_break_is_a_real_target_yet_breaks_so3()
    test_g0_label_genuine_under_rotation_but_gpos_is_fake()
    test_first_overtake_N_pure()
    print("\nPASS: Dyn_g reduces to the exact SO(3) teacher at g=0 and breaks it monotonically for g>0;")
    print("      the break is a real (centering-invariant, lab-z) target; the rotated OOD label is exact")
    print("      at g=0 but fake at g>0 (=> re-sample, not rotate); and the crossover extractor is correct.")


if __name__ == "__main__":
    main()
