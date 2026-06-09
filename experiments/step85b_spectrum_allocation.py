r"""Step 85b PRECHECK — does the non-equivariant model's garbage spectrum MISALLOCATE a fixed observation budget
across chaoticity regimes? (design: docs/specs/2026-06-08-step85b-spectrum-allocation-seed.md)

The C story (full-spectrum allocation) is alive ONLY if the MLP's $\lambda_1(F)$ is wrong in an **F-dependent** way.
If the inflation is roughly **uniform** across the forcing sweep $F$, it CANCELS under normalization — the per-regime
allocation weights $w(F)\propto\lambda_1(F)$ (more chaos → shorter horizon → more observations) are then the same as
the truth's, so an agent allocating by the MLP's spectrum allocates *correctly* and C adds nothing over A. This precheck
trains conv + MLP at a handful of $F$ (N=40) and compares the allocation weight vectors. CHEAP and decisive — run before
committing the 3080-scale full C.

Reuses step74 wholesale: its ``rk4`` / ``l96_rhs`` already take ``F`` as an argument (only the module defaults are
hardcoded to 8.0), and ``train_model`` / ``lyapunov_spectrum`` are F-agnostic. We only add an F-parameterized attractor.
CPU + float64 (Benettin-QR needs float64).

Run:    .venv/bin/python experiments/step85b_spectrum_allocation.py
Writes: papers/figures/step85b_allocation_precheck.json ; exit 0 if C is ALIVE.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as step74  # noqa: E402

DT = torch.float64


def true_map_F(x: torch.Tensor, F: float) -> torch.Tensor:
    r"""step74's $\Delta t$-map at an arbitrary forcing ``F`` (``step74.rk4`` takes ``F`` as a kwarg; default is 8.0)."""
    for _ in range(step74.SUB):
        x = step74.rk4(x, F=F)
    return x


def attractor_traj_F(N: int, n_steps: int, seed: int, F: float, burn: int = 2000) -> torch.Tensor:
    r"""Burn in to the F-forced Lorenz-96 attractor, then return ``n_steps+1`` states. Mirrors
    :func:`step74.attractor_traj` with the forcing exposed."""
    g = torch.Generator().manual_seed(seed)
    x = F + 0.01 * torch.randn(N, generator=g, dtype=DT)
    for _ in range(burn):
        x = true_map_F(x, F)
    out = [x]
    for _ in range(n_steps):
        x = true_map_F(x, F)
        out.append(x)
    return torch.stack(out, 0)


def _lambda1(step_fn, x0, ly_steps, ly_warm) -> float:
    return float(step74.lyapunov_spectrum(step_fn, x0, ly_steps, ly_warm)[0])


def run() -> int:
    N = int(os.environ.get("STEP85B_N", "40"))
    seed = int(os.environ.get("STEP85B_SEED", "0"))
    F_list = [float(x) for x in os.environ.get("STEP85B_F", "6,8,10,13").split(",")]
    n_train = int(os.environ.get("STEP85B_NTRAIN", "12000"))
    ly_steps = int(os.environ.get("STEP85B_LYSTEPS", "1800"))
    ly_warm = 300
    os.environ["STEP74_SMOKE"] = "0"
    os.environ["STEP74_N"] = str(N)
    print(f"[85b] precheck: N={N}, seed={seed}, F={F_list}, n_train={n_train}, ly_steps={ly_steps}", file=sys.stderr)

    rows = []
    for F in F_list:
        traj = attractor_traj_F(N, n_train, seed, F)
        mu, sd = traj.mean(0), traj.std(0) + 1e-8
        x0 = (traj[len(traj) // 2] - mu) / sd
        tl1 = _lambda1(lambda xn: (true_map_F(xn * sd + mu, F) - mu) / sd, x0, ly_steps, ly_warm)

        os.environ["STEP74_MODEL"] = "conv"
        cmodel, cmu, csd, cre = step74.train_model(traj, N, "cpu", seed)
        cl1 = _lambda1(lambda xn: cmodel(xn), (traj[len(traj) // 2] - cmu) / csd, ly_steps, ly_warm)

        os.environ["STEP74_MODEL"] = "mlp"
        mmodel, mmu, msd, mre = step74.train_model(traj, N, "cpu", seed)
        ml1 = _lambda1(lambda xn: mmodel(xn), (traj[len(traj) // 2] - mmu) / msd, ly_steps, ly_warm)

        rows.append({"F": F, "true_l1": tl1, "conv_l1": cl1, "mlp_l1": ml1,
                     "conv_relmse": cre, "mlp_relmse": mre})
        print(f"[85b] F={F:>4}: true l1={tl1:.3f} | conv l1={cl1:.3f} | mlp l1={ml1:.3f} "
              f"(infl {ml1 / cl1:.2f}x, conv relMSE {cre:.4f})", file=sys.stderr)

    # Allocation weights w(F) ∝ lambda1(F). A UNIFORM inflation cancels here (same weights as truth) => C dead.
    def weights(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return v / v.sum()

    wt, wc, wm = weights("true_l1"), weights("conv_l1"), weights("mlp_l1")
    d_conv = float(np.abs(wc - wt).sum())          # L1 allocation error: conv vs truth (should be small)
    d_mlp = float(np.abs(wm - wt).sum())           # L1 allocation error: MLP vs truth (large iff misallocation)

    def _rank(key):
        v = np.array([r[key] for r in rows]); return np.argsort(np.argsort(v))
    rc_conv = float(np.corrcoef(_rank("conv_l1"), _rank("true_l1"))[0, 1])
    rc_mlp = float(np.corrcoef(_rank("mlp_l1"), _rank("true_l1"))[0, 1])

    # C ALIVE iff the MLP's allocation is clearly worse than conv's AND the distortion is materially non-zero.
    c_alive = bool(d_mlp > 2.0 * d_conv and d_mlp > 0.10)
    verdict = {"C_alive": c_alive, "alloc_L1_conv_vs_true": d_conv, "alloc_L1_mlp_vs_true": d_mlp,
               "rankcorr_conv": rc_conv, "rankcorr_mlp": rc_mlp,
               "weights_true": wt.tolist(), "weights_conv": wc.tolist(), "weights_mlp": wm.tolist(),
               "rows": rows, "N": N, "seed": seed, "F_list": F_list}
    print(f"[85b] {'C ALIVE' if c_alive else 'C DEAD/WEAK'}: alloc-L1 mlp-vs-true={d_mlp:.3f} vs conv-vs-true="
          f"{d_conv:.3f}; rankcorr mlp={rc_mlp:.2f} conv={rc_conv:.2f}. "
          f"{'MLP misallocates (F-dependent inflation) -> full C worth a 3080 run.' if c_alive else 'MLP inflation ~uniform -> allocation ~correct -> C adds nothing over A; downgrade C honestly.'}",
          file=sys.stderr)
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "step85b_allocation_precheck.json").write_text(json.dumps(verdict, indent=2))
    return 0 if c_alive else 1


if __name__ == "__main__":
    raise SystemExit(run())
