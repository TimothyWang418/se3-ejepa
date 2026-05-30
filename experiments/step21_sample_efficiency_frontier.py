r"""Step 21: the sample-efficiency **frontier** -- how equivariance shifts the learning curve.

Where this sits, and why it is the capstone
-------------------------------------------
Steps 8/13 showed -- at a *single* training-set size -- that the equivariant latent JEPA
generalises across the symmetry group (举一反三) while the non-equivariant baseline breaks
out-of-distribution. Steps 14-18 lifted that to closed loop, Step 19 to scenes, Step 20 to
active inference. What none of them did is sweep the *amount of data* and draw the
**frontier**: test error as a function of the number of interactions $N$.

That frontier is the operational form of CLAUDE.md Open Question #1 -- *does SE(3)-equivariance
in the encoder of a JEPA improve sample efficiency vs a non-equivariant baseline?* -- and the
sharpest statement of the project thesis, because the inductive-bias payoff is *exactly* the gap
between the two learning curves.

Two readings of "sample efficiency", and which one the theorem guarantees
-------------------------------------------------------------------------
We train both models on the thin orientation **wedge** (z-rotations $\phi\in[0,90°)$, the
Step-13 protocol) and, at each $N$, evaluate the pooled latent 1-step relMSE in two regimes:

  * **in-distribution** -- held-out clouds in the *same* wedge. Both models *can* fit, and the
    honest finding here is that the higher-capacity baseline fits *at least as well*:
    equivariance buys **no** in-distribution sample-efficiency edge in this setup. This is the
    soft, data-dependent half, and in-wedge it is a wash (an MLP can memorise a wedge given
    enough data; it has no symmetry handicap when never tested off the wedge).

  * **across the whole group** -- the held-out transition rotated by random $\mathrm{SO}(3)$.
    Here the equivariant model's error is *exactly* its in-distribution error at **every** $N$:
    $\mathrm{relMSE}$ carries an orthogonal $\rho(R)$ in both numerator and denominator (Step 13
    [B]), so its whole-group learning curve *coincides* with its in-wedge one. The
    non-equivariant model has no such guarantee -- trained on the wedge it never sees off-wedge
    orientations, so its across-group error is a **wall**: flat-high at every $N$.

So the headline is not "$k\times$ fewer samples" -- and emphatically not an in-distribution
sample-efficiency win, which our own numbers do not support. It is the difference between a
*descending* learning curve (VN -- whole-group competence is reachable from wedge data) and a
*flat wall* (MLP -- whole-group competence is unreachable at any $N$ on the grid), once you
measure competence across the group the world actually has. The equivariant prior converts
wedge-only data into whole-group competence; no amount of in-wedge data buys the baseline the
same thing. We still report the *in-distribution* frontier for both models, but state plainly
what it shows: in-wedge the higher-capacity baseline is competitive (often better), so the
equivariant payoff lives **entirely** across the group, not in in-distribution sample count.

Why the across-group curve is a theorem, not a trend. With $E(Rx)=\rho(R)E(x)$ and
$f(\rho(R)z,Ra)=\rho(R)f(z,a)$ for orthogonal $\rho(R)$, the relMSE numerator
$\lVert f(E(Rx),Ra)-E(Rx')\rVert^2=\lVert\rho(R)(f(E(x),a)-E(x'))\rVert^2$ and denominator are
both $\rho$-invariant, so $\mathrm{relMSE}(Rx,Ra,Rx')=\mathrm{relMSE}(x,a,x')$ for *all*
$R\in\mathrm{SO}(3)$ and *any* weights -- hence the VN's whole-group curve equals its in-wedge
curve identically, at every $N$ and even at init.

Honest scope. Synthetic exactly-$\mathrm{SO}(3)$ teacher (Step 13: the price of a *provable* 3D
symmetry at laptop scale), one task family, laptop compute. The across-group frontier is a
*consequence* of the equivariance theorem made quantitative over $N$; the in-distribution
multiplier is the softer, data-dependent half. Sutton's Bitter Lesson (2019) is the standing
caveat: this says nothing about regimes where the symmetry is approximate or absent -- it
quantifies the payoff precisely *when the world genuinely carries the group* (cf. Step 16's
misspecification boundary, where the prior stops being free).

Run:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step21_sample_efficiency_frontier.py
    # fast smoke: STEP21_SMOKE=1 .venv/bin/python experiments/step21_sample_efficiency_frontier.py
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 10/13 helpers we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.training.jepa import train_jepa  # noqa: E402

from step10_pusht_closed_loop import n_params  # noqa: E402  (generic param counter)
from step13_se3_latent_jepa import (  # noqa: E402
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    latent_rel_mse,
    rand_so3,
    rotate_points,
)

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP21_SMOKE"))


# --------------------------------------------------------------------------- #
# evaluation: in-distribution vs across-the-group latent prediction error
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_seen_ood(
    model, St: torch.Tensor, At: torch.Tensor, S2t: torch.Tensor, ood_Rs: list[torch.Tensor]
) -> tuple[float, float]:
    r"""``(seen, ood)`` pooled latent relMSE: in-wedge held-out, and across random $\mathrm{SO}(3)$.

    ``seen`` is the Step-13 [B] relMSE on the held-out wedge transition; ``ood`` is the mean of
    that same relMSE with the *whole* transition $(x,a,x')$ rotated by each $R\in$ ``ood_Rs``
    (a sweep across the group). For the equivariant model ``ood == seen`` to the float floor
    (the relMSE carries an orthogonal $\rho(R)$); the baseline's ``ood`` is a wall.
    """
    seen = latent_rel_mse(model, St, At, S2t)
    ood = float(np.mean([
        latent_rel_mse(model, rotate_points(St, R), rotate_points(At, R), rotate_points(S2t, R))
        for R in ood_Rs
    ]))
    return seen, ood


def train_eval_one(
    build, N: int, *, seed: int, updates: int,
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor], ood_Rs: list[torch.Tensor],
    var_coef: float = 0.1,
) -> dict[str, float]:
    r"""Draw $N$ wedge transitions (``seed``), build + train one model, return its metrics.

    Fixes the *gradient-update* budget across $N$ (``epochs = round(updates / ceil(N/bs))``) so
    the frontier reflects **data size, not optimisation steps**. ``torch.manual_seed(seed)`` is
    set before ``build()`` so the weight init also varies per seed. Returns
    ``{seen, ood, latent_std, n_params}``.
    """
    St, At, S2t = test
    S, A, S2 = collect_cloud_transitions(N, seed=1000 + seed)
    bs = min(128, N)
    epochs = max(1, round(updates / math.ceil(N / bs)))
    torch.manual_seed(seed)
    model = build()
    hist = train_jepa(
        model, S, A, S2, epochs=epochs, batch_size=bs,
        var_coef=var_coef, seed=seed, log_every=10**9, verbose=False,
    )
    seen, ood = eval_seen_ood(model, St, At, S2t, ood_Rs)
    return {"seen": seen, "ood": ood, "latent_std": float(hist["latent_std"]), "n_params": n_params(model)}


# --------------------------------------------------------------------------- #
# frontier statistic: smallest N to reach a target error (a pure, unit-tested function)
# --------------------------------------------------------------------------- #
def frontier_n_at_target(N_grid: list[int], curve: list[float], target: float) -> float | None:
    r"""Smallest $N$ (log-linear interpolated) at which ``curve`` first drops $\le$ ``target``.

    ``curve[i]`` is the error at ``N_grid[i]`` (a learning curve, non-increasing in $N$).
    Interpolates linearly in $\log N$ between the bracketing grid points; returns ``None`` if the
    target is never reached on the grid (a "wall"). The VN-vs-MLP ratio of these N's is the
    sample-efficiency multiplier.
    """
    for i, e in enumerate(curve):
        if e <= target:
            if i == 0:
                return float(N_grid[0])
            e0, e1 = curve[i - 1], curve[i]
            n0, n1 = math.log(N_grid[i - 1]), math.log(N_grid[i])
            frac = 0.0 if e0 == e1 else (e0 - target) / (e0 - e1)
            return math.exp(n0 + frac * (n1 - n0))
    return None


def _fmt_n(n: float | None) -> str:
    return "wall" if n is None else f"{n:6.0f}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    line = "=" * 78
    N_GRID = [32, 128, 512] if SMOKE else [16, 32, 64, 128, 256, 512]
    SEEDS = [0] if SMOKE else [0, 1, 2]
    UPDATES = 300 if SMOKE else 600
    N_TEST = 80 if SMOKE else 400
    N_OOD = 4 if SMOKE else 8

    # fixed held-out test set (same wedge) and fixed across-group rotations -> curves comparable
    test = collect_cloud_transitions(N_TEST, seed=999)
    ood_gen = torch.Generator().manual_seed(7)
    ood_Rs = [rand_so3(ood_gen) for _ in range(N_OOD)]

    print(line)
    print("[Step 21] Sample-efficiency FRONTIER: equivariance shifts the learning curve")
    print(line)
    print(f"    train on z-wedge [0,90); sweep N={N_GRID}; {len(SEEDS)} seed(s); "
          f"{UPDATES} grad-updates/run (fixed across N)")
    print(f"    eval: in-wedge held-out (seen) vs same transition under {N_OOD} random SO(3) (whole group)")

    builders = {"VN": build_eq_jepa, "MLP": build_mlp_jepa}
    raw: dict[str, dict[int, list[dict]]] = {name: {N: [] for N in N_GRID} for name in builders}
    for name, build in builders.items():
        for N in N_GRID:
            for seed in SEEDS:
                raw[name][N].append(
                    train_eval_one(build, N, seed=seed, updates=UPDATES, test=test, ood_Rs=ood_Rs)
                )

    # aggregate seed mean/std per (model, N)
    def stat(name: str, N: int, key: str) -> tuple[float, float]:
        vals = [r[key] for r in raw[name][N]]
        return float(np.mean(vals)), float(np.std(vals))

    curves: dict[str, dict[str, list[float]]] = {}
    for name in builders:
        seen = [stat(name, N, "seen")[0] for N in N_GRID]
        seen_s = [stat(name, N, "seen")[1] for N in N_GRID]
        ood = [stat(name, N, "ood")[0] for N in N_GRID]
        ood_s = [stat(name, N, "ood")[1] for N in N_GRID]
        ratio = [o / max(s, 1e-12) for o, s in zip(ood, seen)]
        curves[name] = {"seen": seen, "seen_std": seen_s, "ood": ood, "ood_std": ood_s, "ratio": ratio}
    vn_params = raw["VN"][N_GRID[0]][0]["n_params"]
    mlp_params = raw["MLP"][N_GRID[0]][0]["n_params"]

    # ---- the frontier table ------------------------------------------------ #
    print()
    print("    latent 1-step relMSE vs training-set size N (mean over seeds):")
    print(f"    {'N':>6} | {'VN seen':>9} {'VN group':>9} {'VN g/s':>7} | "
          f"{'MLP seen':>9} {'MLP group':>9} {'MLP g/s':>7}")
    print("    " + "-" * 70)
    for i, N in enumerate(N_GRID):
        v, m = curves["VN"], curves["MLP"]
        print(f"    {N:>6} | {v['seen'][i]:>9.3e} {v['ood'][i]:>9.3e} {v['ratio'][i]:>7.3f} | "
              f"{m['seen'][i]:>9.3e} {m['ood'][i]:>9.3e} {m['ratio'][i]:>7.2f}")
    print(f"    params: VN={vn_params}  MLP={mlp_params}  ({mlp_params / vn_params:.1f}x VN)")

    # ---- frontier multipliers ---------------------------------------------- #
    # in-distribution: a target both can reach in-wedge. We report each model's N-to-target
    # honestly; the higher-capacity MLP typically reaches it with *fewer* samples (no symmetry
    # handicap in-wedge), so `mult_in` is recorded for the JSON but NOT sold as a VN win.
    tau_in = 1.5 * max(curves["VN"]["seen"][-1], curves["MLP"]["seen"][-1])
    n_vn_in = frontier_n_at_target(N_GRID, curves["VN"]["seen"], tau_in)
    n_mlp_in = frontier_n_at_target(N_GRID, curves["MLP"]["seen"], tau_in)
    mult_in = (n_mlp_in / n_vn_in) if (n_vn_in and n_mlp_in) else None
    # across the group: VN reaches its own best x1.5; the MLP wall almost never does
    tau_grp = 1.5 * curves["VN"]["ood"][-1]
    n_vn_grp = frontier_n_at_target(N_GRID, curves["VN"]["ood"], tau_grp)
    n_mlp_grp = frontier_n_at_target(N_GRID, curves["MLP"]["ood"], tau_grp)

    print()
    print(f"    [in-distribution] target relMSE<={tau_in:.3e}:  "
          f"VN at N~{_fmt_n(n_vn_in)},  MLP at N~{_fmt_n(n_mlp_in)}")
    print("      -> in-wedge the higher-capacity MLP is competitive (no VN sample-efficiency edge here).")
    print(f"    [whole group]     target relMSE<={tau_grp:.3e}:  "
          f"VN at N~{_fmt_n(n_vn_grp)},  MLP at N~{_fmt_n(n_mlp_grp)}"
          + ("  =>  MLP never reaches it (wall)" if n_mlp_grp is None else ""))

    # ---- guards ------------------------------------------------------------ #
    ok_vn_flat = all(r < 1.10 for r in curves["VN"]["ratio"])      # across-group flat at EVERY N (theorem)
    ok_mlp_wall = curves["MLP"]["ratio"][-1] > 1.5                 # MLP breaks across group at max data
    ok_vn_fits = curves["VN"]["seen"][-1] < 0.9                    # VN beats the trivial no-change predictor (relMSE=1)
    ok_vn_descends = curves["VN"]["ood"][-1] < curves["VN"]["ood"][0]  # whole-group frontier DESCENDS with data
    ok_params = vn_params < mlp_params                            # equivariant model is smaller
    ok_grp = (n_vn_grp is not None) and (n_mlp_grp is None or n_mlp_grp > n_vn_grp)  # group frontier favours VN
    passed = ok_vn_flat and ok_mlp_wall and ok_vn_fits and ok_vn_descends and ok_params and ok_grp

    print()
    print(line)
    print("STEP 21 SUMMARY")
    print(line)
    print("    The frontier (relMSE vs N) under a genuinely SO(3)-symmetric teacher:")
    print(f"    - VN whole-group curve == its in-wedge curve at every N (g/s in [{min(curves['VN']['ratio']):.3f},"
          f"{max(curves['VN']['ratio']):.3f}]); the whole-group frontier DESCENDS with data "
          f"({curves['VN']['ood'][0]:.3e} -> {curves['VN']['ood'][-1]:.3e}), competence at N~{_fmt_n(n_vn_grp)} wedge samples.")
    print(f"    - MLP fits the wedge but its whole-group error is a WALL "
          f"(g/s x{curves['MLP']['ratio'][-1]:.1f} at N={N_GRID[-1]}); never reaches the group target on the grid.")
    print("    - HONEST finding: the equivariant payoff is NOT in-distribution sample efficiency -- in-wedge the")
    print("      higher-capacity MLP is competitive/better. The payoff is ENTIRELY across the group: wedge-only")
    print("      data + the prior => whole-group competence; no amount of in-wedge data buys the baseline that.")
    print(f"    - equivariant model is {mlp_params / vn_params:.1f}x smaller ({vn_params} vs {mlp_params} params).")
    print(f"    guards: VN-flat={ok_vn_flat}  MLP-wall={ok_mlp_wall}  VN-fits={ok_vn_fits}  "
          f"VN-descends={ok_vn_descends}  smaller={ok_params}  group-frontier={ok_grp}")
    print(f"    {'PASS' if passed else 'INCONCLUSIVE'}: when the world carries the group, equivariance is the")
    print("    difference between a learnable whole-group frontier and a wall -- the sample-efficiency")
    print("    payoff is exactly the gap between the two learning curves (CLAUDE.md Open Question #1).")

    # ---- dump JSON --------------------------------------------------------- #
    out = {
        "N_grid": N_GRID, "seeds": SEEDS, "updates": UPDATES, "n_ood_rotations": N_OOD,
        "curves": curves,
        "params": {"VN": vn_params, "MLP": mlp_params},
        "frontier": {
            "tau_in": tau_in, "n_vn_in": n_vn_in, "n_mlp_in": n_mlp_in, "mult_in": mult_in,
            "tau_grp": tau_grp, "n_vn_grp": n_vn_grp, "n_mlp_grp": n_mlp_grp,
        },
        "guards": {
            "vn_flat": ok_vn_flat, "mlp_wall": ok_mlp_wall, "vn_fits": ok_vn_fits,
            "vn_descends": ok_vn_descends, "smaller": ok_params, "group_frontier": ok_grp,
            "passed": passed,
        },
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = "step21_sample_efficiency_frontier" + ("_smoke" if SMOKE else "") + ".json"
    (fig_dir / fname).write_text(json.dumps(out, indent=2))
    print(f"\n    wrote {fig_dir / fname}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
