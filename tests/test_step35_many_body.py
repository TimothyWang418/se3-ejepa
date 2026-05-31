r"""Step 35 — few-body $\to$ many-body **combinatorial** 举一反三: structural invariants.

This certifies the *architectural* and *teacher* properties Step 35 leans on, all at
**random initialisation** (no training): they are exact-by-construction facts, so a unit
test is the right place to pin them. The trained-model generalisation numbers live in
``experiments/step35_many_body.py``; here we prove the scaffolding that makes those
numbers meaningful holds at counts the model never trains on.

Six properties (see the module docstring of ``step35_many_body`` for the derivations):

1. **count-stable mean message** $\bar r_i=\frac1{O-1}\sum_{j\ne i}\hat r_{ij}$ lies in the
   unit ball $\lVert\bar r_i\rVert\le 1$ at *every* count -- the whole reason a single-count
   predictor transfers (a *sum* would grow with $O$). At $O=2$ it is a single unit vector
   ($\lVert\bar r_i\rVert=1$); at $O\ge 3$ it strictly contracts.
2. **translation invariance** of the message: $\bar r_i(S+t)=\bar r_i(S)$ (built from
   centroid differences), so the interaction never leaks absolute position.
3. **teacher global $\mathrm{SE}(3)$ equivariance** at counts $\{2,3,4,5\}$ (incl. unseen):
   $\mathrm{teacher}(RS+t,\,RA)=R\,\mathrm{teacher}(S,A)+t$, exact to the float floor.
4. **teacher permutation $S_O$ equivariance** at counts $\{3,4,5\}$: relabelling the objects
   permutes the output, because $\bar r_i$ is a symmetric mean over the *other* objects.
5. **$O=2$ recovers Step 24 verbatim** ($\bar r_i=\hat r_{ij}$) and **$O=1$ reduces to pure
   Step-13 self-dynamics** (message $\equiv 0$, no torque) -- the two boundary counts.
6. **VN-MP whole-pipeline $\mathrm{SE}(3)\rtimes S_O$ equivariance at an UNSEEN count** $O=5$
   (the model is built for $O=3$): the slot encoder/predictor are count-agnostic by
   construction, so the residual sits at the float floor at init, before any optimisation.

Run:
    .venv/bin/python tests/test_step35_many_body.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    rand_so3,
    rotate_points,
    teacher_step,
)
from step24_object_interaction import scene_teacher_interact  # noqa: E402
from step35_many_body import (  # noqa: E402
    A_OBJ,
    _build,
    build_msg_action_many,
    make_many_body_transitions,
    mean_rel_dirs,
    scene_teacher_many,
    vnmp_perm_err,
    vnmp_se3_err,
)

torch.set_default_dtype(torch.float32)


def _scene(o: int, n: int = 12, seed: int = 0):
    r"""A held-out batch of ``o``-object transitions. ``S:(n,o,P,3)``, ``A:(n,o,3)``, ``A_flat:(n,o*A_OBJ)``."""
    S, A_flat, _ = make_many_body_transitions(n, o, seed=seed)
    A = A_flat.reshape(n, o, A_OBJ)
    return S, A, A_flat


# --------------------------------------------------------------------------- #
# 1+2. mean message: count-stable (unit ball) + translation-invariant
# --------------------------------------------------------------------------- #
def _check_message() -> None:
    print("  [1] count-stable mean message  |rbar_i| <= 1 at every count:")
    for o in range(1, 7):
        S, _, _ = _scene(o, seed=10 + o)
        norms = mean_rel_dirs(S).norm(dim=-1)            # (n,o)
        mx = norms.max().item()
        tag = ("== 0 (no neighbours)" if o == 1
               else "~ 1 (single unit vector)" if o == 2 else "< 1 (contracts)")
        print(f"      O={o}: max|rbar| = {mx:.4f}  {tag}")
        assert mx <= 1.0 + 1e-5, f"O={o}: mean message left the unit ball ({mx:.4f}) -- not count-stable"
        if o == 1:
            assert mx < 1e-8, f"O=1 message must be identically 0, got {mx:.2e}"
        if o == 2:
            assert abs(mx - 1.0) < 1e-4, f"O=2 message must be a unit vector, got {mx:.4f}"

    # translation invariance: rbar(S + t) == rbar(S)
    S, _, _ = _scene(4, seed=77)
    t = torch.randn(1, 1, 1, 3) * 5.0
    drift = (mean_rel_dirs(S + t) - mean_rel_dirs(S)).abs().max().item()
    print(f"  [2] translation invariance  max|rbar(S+t)-rbar(S)| = {drift:.2e}")
    assert drift < 1e-5, f"mean message is not translation-invariant: {drift:.2e}"


# --------------------------------------------------------------------------- #
# 3. teacher global SE(3) equivariance at several counts (incl. unseen)
# --------------------------------------------------------------------------- #
def _check_teacher_se3() -> None:
    gen = torch.Generator().manual_seed(0)
    print("  [3] teacher SE(3)  max|teacher(RS+t,RA)-(R teacher(S,A)+t)| over 8 (R,t):")
    worst_all = 0.0
    for o in (2, 3, 4, 5):
        S, A, _ = _scene(o, seed=100 + o)
        worst = 0.0
        for _ in range(8):
            R = rand_so3(gen)
            t = torch.randn(3, generator=gen)
            lhs = scene_teacher_many(rotate_points(S, R) + t.reshape(1, 1, 1, 3),
                                     rotate_points(A, R))
            rhs = rotate_points(scene_teacher_many(S, A), R) + t.reshape(1, 1, 1, 3)
            worst = max(worst, (lhs - rhs).abs().max().item())
        print(f"      O={o}: {worst:.2e}")
        assert worst < 1e-4, f"teacher broke SE(3) equivariance at O={o}: {worst:.2e}"
        worst_all = max(worst_all, worst)
    print(f"      => worst over all counts {worst_all:.2e} (float floor)")


# --------------------------------------------------------------------------- #
# 4. teacher permutation S_O equivariance
# --------------------------------------------------------------------------- #
def _check_teacher_perm() -> None:
    gen = torch.Generator().manual_seed(1)
    print("  [4] teacher S_O  max|teacher(sigma S, sigma A) - sigma teacher(S,A)| over 6 perms:")
    for o in (3, 4, 5):
        S, A, _ = _scene(o, seed=200 + o)
        base = scene_teacher_many(S, A)
        worst = 0.0
        for _ in range(6):
            perm = torch.randperm(o, generator=gen)
            lhs = scene_teacher_many(S[:, perm], A[:, perm])
            rhs = base[:, perm]
            worst = max(worst, (lhs - rhs).abs().max().item())
        print(f"      O={o}: {worst:.2e}")
        assert worst < 1e-5, f"teacher broke permutation equivariance at O={o}: {worst:.2e}"


# --------------------------------------------------------------------------- #
# 5. boundary counts: O=2 recovers Step 24, O=1 is pure self-dynamics
# --------------------------------------------------------------------------- #
def _check_boundaries() -> None:
    # O=2: scene_teacher_many must equal Step 24's scene_teacher_interact bit-for-bit
    S2, A2, _ = _scene(2, seed=321)
    diff24 = (scene_teacher_many(S2, A2) - scene_teacher_interact(S2, A2)).abs().max().item()
    print(f"  [5a] O=2 recovers Step 24  max|many - interact| = {diff24:.2e}")
    assert diff24 < 1e-5, f"O=2 does not recover the Step 24 two-body teacher: {diff24:.2e}"

    # O=1: message identically zero, output is pure per-object Step-13 self-dynamics
    S1, A1, _ = _scene(1, seed=654)
    msg1 = mean_rel_dirs(S1).abs().max().item()
    self_only = teacher_step(S1[:, 0], A1[:, 0])[:, None]    # (n,1,P,3)
    diff1 = (scene_teacher_many(S1, A1) - self_only).abs().max().item()
    print(f"  [5b] O=1 no-interaction limit  |msg|={msg1:.2e}  max|many - self_step|={diff1:.2e}")
    assert msg1 < 1e-8, f"O=1 message must vanish: {msg1:.2e}"
    assert diff1 < 1e-6, f"O=1 must reduce to pure self-dynamics: {diff1:.2e}"


# --------------------------------------------------------------------------- #
# 6. VN-MP whole-pipeline equivariance at an UNSEEN count (random init)
# --------------------------------------------------------------------------- #
def _check_pipeline_unseen_count() -> None:
    gen = torch.Generator().manual_seed(2)
    torch.manual_seed(0)
    model = _build("VN-MP").eval()      # built for O=3
    o = 5                               # a count it never sees in the experiment
    S, _, A_flat = _scene(o, n=8, seed=909)

    se3 = max(vnmp_se3_err(model, S, A_flat, rand_so3(gen), torch.randn(3, generator=gen), o)
              for _ in range(8))
    perm = max(vnmp_perm_err(model, S, A_flat, torch.randperm(o, generator=gen), o)
               for _ in range(6))
    print(f"  [6] VN-MP @ UNSEEN O={o} (random init):  SE(3) {se3:.2e}   perm {perm:.2e}")
    assert se3 < 1e-4, f"VN-MP pipeline broke SE(3) at the unseen count: {se3:.2e}"
    assert perm < 1e-5, f"VN-MP pipeline broke permutation at the unseen count: {perm:.2e}"


def main() -> None:
    print("Step 35 — combinatorial 举一反三: structural invariants (random init, no training)\n")
    _check_message()
    _check_teacher_se3()
    _check_teacher_perm()
    _check_boundaries()
    _check_pipeline_unseen_count()
    print("\nPASS: the count-stable mean message lives in the unit ball at every count, the teacher")
    print("is exactly SE(3) x S_O-equivariant (incl. unseen counts), recovers Step 24 at O=2 and pure")
    print("self-dynamics at O=1, and the VN-MP pipeline is equivariant at a count it never trains on.")


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #
def test_mean_message_count_stable_and_translation_invariant() -> None:
    """[1+2] mean message stays in the unit ball at every count and ignores translation."""
    _check_message()


def test_teacher_se3_equivariance_all_counts() -> None:
    """[3] teacher(RS+t, RA) = R teacher(S,A) + t at counts {2,3,4,5}."""
    _check_teacher_se3()


def test_teacher_permutation_equivariance() -> None:
    """[4] teacher commutes with object relabelling at counts {3,4,5}."""
    _check_teacher_perm()


def test_boundary_counts_recover_step24_and_self_dynamics() -> None:
    """[5] O=2 == Step 24 two-body teacher; O=1 == pure Step-13 self-dynamics (message 0)."""
    _check_boundaries()


def test_vnmp_pipeline_equivariant_at_unseen_count() -> None:
    """[6] VN-MP whole-pipeline SE(3) x S_O residual at the float floor at an unseen count, at init."""
    _check_pipeline_unseen_count()


if __name__ == "__main__":
    main()
