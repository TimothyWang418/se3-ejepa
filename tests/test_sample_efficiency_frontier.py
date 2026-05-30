r"""$\mathrm{SO}(3)$-exactness of the sample-efficiency **frontier** (Step 21).

Step 21 sweeps the training-set size $N$ and draws two learning curves per model: the in-wedge
held-out error and the whole-group error (the same held-out transition rotated by random
$\mathrm{SO}(3)$). The claim this test guards is the **geometric theorem** that makes the
equivariant curve special:

  For the VN latent JEPA, $E(Rx)=\rho(R)E(x)$ and $f(\rho(R)z,Ra)=\rho(R)f(z,a)$ with $\rho(R)$
  *orthogonal*, so the pooled latent relMSE carries $\rho(R)$ in **both** numerator and
  denominator and cancels:
  $$\mathrm{relMSE}(Rx,Ra,Rx')=\frac{\lVert\rho(R)\,(f(E(x),a)-E(x'))\rVert^2}{\lVert\rho(R)\,(E(x')-E(x))\rVert^2}=\mathrm{relMSE}(x,a,x')\quad\forall R\in\mathrm{SO}(3),\ \forall\text{ weights.}$$
  Hence the VN's **whole-group learning curve coincides with its in-wedge one at every $N$ and
  even at init** -- training on the thin wedge $\phi\in[0,90°)$ *determines* the dynamics on all
  of $\mathrm{SO}(3)$ (举一反三). The non-equivariant MLP has no such cancellation: trained on the
  wedge it never sees off-wedge orientations, so its whole-group error is a **wall**
  (``ood / seen`` $\gg 1$), structurally even at init and as a learned pathology after training.

Two things this test pins:

  * **The theorem (VN)** -- ``ood / seen`` $= 1$ to the e3nn float floor at init AND post-train.
    This is the unit-level mirror of the experiment's ``g/s = 1.000`` column.
  * **Non-vacuity (MLP)** -- the broken baseline misses it by a wide margin, so the VN assertion
    bites; and ``seen`` is a genuine non-trivial error (the field is not degenerately zero).

Plus a pure-function check of ``frontier_n_at_target`` (the log-$N$ interpolation that turns a
learning curve into a "smallest $N$ to reach target error", and reports a *wall* as ``None``),
since the VN-vs-MLP sample multiplier is read straight off it.

**Honest scope.** This test guards the *across-group* exactness (the thesis) and the wall, NOT an
in-distribution sample-efficiency win -- the experiment's own numbers show the higher-capacity MLP
is competitive in-wedge, so there is deliberately no "VN fits with fewer samples" assertion here.

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python tests/test_sample_efficiency_frontier.py
"""

import math
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                  # for `src.*`
sys.path.insert(0, str(ROOT / "experiments"))  # for the Step 13 backbone + Step 21 helpers

import torch  # noqa: E402

from src.training.jepa import train_jepa  # noqa: E402

from step13_se3_latent_jepa import (  # noqa: E402
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    rand_so3,
)
from step21_sample_efficiency_frontier import eval_seen_ood, frontier_n_at_target  # noqa: E402

# Tolerances. The whole-group invariance is exact-by-construction (orthogonal $\rho(R)$ cancels in
# the relMSE), so the VN ``ood/seen`` deviation sits at the e3nn spherical-harmonic float floor.
_FLOOR = 1e-4    # |ood/seen - 1| for the VN (whole-group curve == in-wedge curve)
_BROKEN = 0.30   # the non-equivariant control must MISS it by at least this (=> assertion not vacuous)


@lru_cache(maxsize=1)
def _probe() -> tuple[dict, dict]:
    r"""Build VN + MLP, probe ``(seen, ood)`` at init, train each on the wedge, probe post-train.

    Returns ``(init, post)``, each ``{model_name: (seen, ood)}``. Cached so the two short training
    runs happen once across all pytest entry points. A real EMA-target + VICReg + Muon/AdamW run
    (the same ``train_jepa`` the experiment uses) => a genuine "symmetry survives optimisation"
    guard, not an init-only check.
    """
    St, At, S2t = collect_cloud_transitions(64, seed=999)            # held-out wedge probe
    gen = torch.Generator().manual_seed(7)
    ood_Rs = [rand_so3(gen) for _ in range(6)]                       # the across-group sweep
    S, A, S2 = collect_cloud_transitions(128, seed=1000)            # wedge training data

    builders = {"VN": build_eq_jepa, "MLP": build_mlp_jepa}
    init: dict[str, tuple[float, float]] = {}
    post: dict[str, tuple[float, float]] = {}
    for name, build in builders.items():
        torch.manual_seed(0)
        model = build()
        init[name] = eval_seen_ood(model, St, At, S2t, ood_Rs)
        train_jepa(model, S, A, S2, epochs=12, batch_size=128, var_coef=0.1, seed=0,
                   log_every=10**9, verbose=False)
        post[name] = eval_seen_ood(model, St, At, S2t, ood_Rs)
    return init, post


def test_vn_whole_group_curve_equals_in_wedge() -> None:
    r"""VN: whole-group relMSE $=$ in-wedge relMSE to the float floor -- init AND post-train (举一反三)."""
    init, post = _probe()
    for phase, tbl in (("init", init), ("post-train", post)):
        seen, ood = tbl["VN"]
        dev = abs(ood / max(seen, 1e-12) - 1.0)
        assert dev < _FLOOR, (
            f"VN whole-group curve != in-wedge curve @{phase} (ood/seen-1 = {dev:.2e}); "
            "the relMSE must be exactly SO(3)-invariant"
        )
        # non-vacuity: the in-wedge error is a genuine non-trivial field, not degenerately zero
        assert seen > 1e-6, f"VN in-wedge relMSE ~0 @{phase} ({seen:.2e}); invariance would be vacuous"


def test_mlp_breaks_whole_group_invariance() -> None:
    r"""Non-equivariant control: the whole-group curve breaks away from the in-wedge one -- so the VN exactness assertion is not vacuous.

    Two faces, mirroring the structural-vs-learned split in ``test_efe_invariance``:

      * **init** -- the MLP is *not* whole-group-invariant by construction (a non-equivariant encoder
        maps $Rx$ to an unrelated latent), so ``ood/seen`` is off the VN's exact floor. But at random
        init the *sign* of the deviation is unconstrained (the denominator moves under rotation too),
        so we assert magnitude only.
      * **post-train** -- the genuine **wall**: trained on the wedge, the MLP fits it (``seen`` drops)
        while its across-group error stays high, so ``ood/seen`` $\gg 1$ -- exactly the experiment's
        ``g/s`` column (a few-fold to $>10\times$).
    """
    init, post = _probe()
    seen_i, ood_i = init["MLP"]
    dev_i = abs(ood_i / max(seen_i, 1e-12) - 1.0)
    assert dev_i > 1e-2, (
        f"MLP whole-group ratio sits at the invariance floor @init ({dev_i:.2e}); "
        "it should not be SO(3)-invariant by construction (unlike the VN's exact 1.0)"
    )
    seen_p, ood_p = post["MLP"]
    dev_p = ood_p / max(seen_p, 1e-12) - 1.0
    assert dev_p > _BROKEN, (
        f"trained MLP unexpectedly whole-group-invariant @post-train (ood/seen-1 = {dev_p:.2e}); "
        "the VN exactness assertion would be vacuous"
    )


def test_frontier_n_at_target_pure() -> None:
    r"""``frontier_n_at_target``: log-$N$ interpolation, grid-point hits, the *wall*, and monotonicity."""
    N_grid = [10, 100, 1000]
    curve = [1.0, 0.5, 0.1]  # a descending learning curve

    # crosses between N=100 (0.5) and N=1000 (0.1): frac=(0.5-0.3)/(0.5-0.1)=0.5 => 100*10^0.5
    n = frontier_n_at_target(N_grid, curve, 0.30)
    assert n is not None and math.isclose(n, 100.0 * math.sqrt(10.0), rel_tol=1e-9), n

    # target reached already at the first grid point => returns N_grid[0]
    assert frontier_n_at_target(N_grid, curve, 1.0) == 10.0

    # target hit exactly at an interior grid point => that grid point's N
    assert math.isclose(frontier_n_at_target(N_grid, curve, 0.5), 100.0, rel_tol=1e-9)

    # never reached on the grid => a wall (None)
    assert frontier_n_at_target(N_grid, curve, 0.05) is None

    # monotonicity: a harder (smaller) target needs at least as many samples
    n_easy = frontier_n_at_target(N_grid, curve, 0.40)
    n_hard = frontier_n_at_target(N_grid, curve, 0.30)
    assert n_easy is not None and n_hard is not None and n_hard > n_easy, (n_easy, n_hard)


def main() -> None:
    """Print the frontier probe (init + post-train seen/ood) and run all assertions."""
    init, post = _probe()
    print("Step 21 sample-efficiency frontier: whole-group vs in-wedge relMSE\n")
    hdr = f"    {'model':>6} | {'phase':>10} | {'seen':>10} | {'ood (group)':>11} | {'ood/seen':>9}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name in ("VN", "MLP"):
        for phase, tbl in (("init", init), ("post-train", post)):
            seen, ood = tbl[name]
            print(f"    {name:>6} | {phase:>10} | {seen:>10.3e} | {ood:>11.3e} | {ood / max(seen, 1e-12):>9.4f}")

    test_vn_whole_group_curve_equals_in_wedge()
    test_mlp_breaks_whole_group_invariance()
    test_frontier_n_at_target_pure()
    print("\nPASS: VN whole-group learning curve == its in-wedge curve to the float floor (init + post);")
    print("      the non-equivariant control breaks it by a wide margin (=> assertion bites); and the")
    print("      frontier statistic interpolates / walls / orders correctly. 举一反三 across SO(3).")


if __name__ == "__main__":
    main()
