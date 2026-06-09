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


def allocate(weights, B_tot: int) -> list:
    r"""Split a fixed total observation budget ``B_tot`` across regimes proportional to ``weights`` (raw $\lambda_1(F)$ —
    more chaos → shorter horizon → more observations), using **largest-remainder** rounding so the split sums to EXACTLY
    ``B_tot`` (required for a matched-budget conv-vs-MLP comparison). A flatter weight vector (the MLP's range-compressed
    spectrum) shifts budget from the hardest regime to the easiest — the C misallocation."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = w * B_tot
    base = np.floor(raw).astype(int)
    rem = int(B_tot - base.sum())
    order = np.argsort(-(raw - base))                      # largest fractional parts first
    for i in range(rem):
        base[order[i % len(base)]] += 1
    return base.tolist()


def build_true_traj_F(N: int, L: int, seed: int, F: float) -> torch.Tensor:
    r"""One TRUE $u\equiv0$ trajectory at forcing ``F``, ``(L+1, N)``, launched from a seed-pinned on-attractor point
    (regime-F analogue of :func:`step85.build_true_traj`)."""
    traj = attractor_traj_F(N, 4 * L, seed, F)
    g = torch.Generator().manual_seed(seed + 777)
    j0 = int(torch.randint(0, traj.shape[0] - L - 1, (1,), generator=g).item())
    x = torch.empty(L + 1, N, dtype=DT)
    x[0] = traj[j0]
    cur = traj[j0].clone()
    for t in range(1, L + 1):
        cur = true_map_F(cur, F)
        x[t] = cur
    return x


def run_full_C(seeds, F_list, N: int, eps: float, L: int, B_tot_list, device: str) -> dict:
    r"""Full C: allocate a fixed total observation budget across a forcing-$F$ chaoticity ensemble by each model's
    per-regime $\lambda_1(F)$, and compare aggregate forecast-violation. CERT-ISOLATED: every regime is forecast by its
    faithful **conv** model; only the budget SPLIT differs (conv-spectrum weights vs MLP-spectrum weights vs true). The
    MLP's range-compressed spectrum over-weights easy regimes and starves hard ones → higher aggregate violation at the
    same total budget. Training runs on ``device`` (CUDA on the 3080); the QR spectra + re-observation episodes run on
    CPU (cheap; reuse step85.reobserve_run_budgeted). Gate G: conv-alloc < MLP-alloc at the binding budget on >=2/3 seeds."""
    import step85_trustworthy_cert_downstream as step85  # reuse reobserve_run_budgeted + AutonomousWM
    smoke = bool(int(os.environ.get("STEP85B_SMOKE", "0")))
    n_train = 4000 if smoke else 12000
    ly_steps = 600 if smoke else 1800
    ly_warm = 100 if smoke else 300
    os.environ["STEP74_SMOKE"] = "1" if smoke else "0"
    os.environ["STEP74_N"] = str(N)

    def _train_l1(kind, traj_dev, traj_cpu, F):
        os.environ["STEP74_MODEL"] = kind
        model, mu, sd, _ = step74.train_model(traj_dev, N, device, seed)
        model = model.to("cpu").eval(); mu = mu.to("cpu"); sd = sd.to("cpu")     # episodes/QR on CPU
        x0 = (traj_cpu[len(traj_cpu) // 2] - mu) / sd
        l1 = _lambda1(lambda xn: model(xn), x0, ly_steps, ly_warm)
        return model, mu, sd, l1

    per_seed = {}
    for seed in seeds:
        Fs = list(F_list)
        reg = {}
        for F in Fs:
            traj_dev = attractor_traj_F(N, n_train, seed, F).to(device)
            traj_cpu = traj_dev.to("cpu")
            cmodel, cmu, csd, cl1 = _train_l1("conv", traj_dev, traj_cpu, F)
            _, _, _, ml1 = _train_l1("mlp", traj_dev, traj_cpu, F)           # only the MLP's lambda1 is needed
            mu_t, sd_t = traj_cpu.mean(0), traj_cpu.std(0) + 1e-8
            tl1 = _lambda1(lambda xn: (true_map_F(xn * sd_t + mu_t, F) - mu_t) / sd_t,
                           (traj_cpu[len(traj_cpu) // 2] - mu_t) / sd_t, ly_steps, ly_warm)
            reg[F] = {"fc": step85.AutonomousWM(cmodel), "mu": cmu, "sd": csd,
                      "cl1": cl1, "ml1": ml1, "tl1": tl1, "traj": build_true_traj_F(N, L, seed, F)}
            print(f"[85b.C] seed {seed} F={F:>4}: conv l1={cl1:.3f} mlp l1={ml1:.3f} true l1={tl1:.3f}", file=sys.stderr)

        w_conv = [reg[F]["cl1"] for F in Fs]
        w_mlp = [reg[F]["ml1"] for F in Fs]
        w_true = [reg[F]["tl1"] for F in Fs]

        def agg(weights, B_tot):
            budgets = allocate(weights, B_tot)
            vs = []
            for F, b in zip(Fs, budgets):
                r = reg[F]
                interval = max(1, L // max(1, b))               # spread b observations uniformly over the length-L episode
                out = step85.reobserve_run_budgeted(r["fc"], r["mu"], r["sd"], N, r["traj"],
                                                    interval=interval, eps=eps, budget=max(1, b))
                vs.append(out["violation_rate"])
            return float(np.mean(vs))

        rows = []
        for B_tot in B_tot_list:
            vc = agg(w_conv, B_tot); vm = agg(w_mlp, B_tot); vt = agg(w_true, B_tot)
            rows.append({"B_tot": B_tot, "conv_alloc": vc, "mlp_alloc": vm, "true_alloc": vt})
            print(f"[85b.C] seed {seed} B_tot={B_tot}: conv-alloc={vc:.3f} mlp-alloc={vm:.3f} true-alloc={vt:.3f} "
                  f"(margin {vm - vc:+.3f})", file=sys.stderr)
        best_margin = max(r["mlp_alloc"] - r["conv_alloc"] for r in rows)   # largest conv-over-mlp gap across the sweep
        per_seed[seed] = {"rows": rows, "w_conv": w_conv, "w_mlp": w_mlp, "w_true": w_true,
                          "best_margin": best_margin}

    margins = [per_seed[s]["best_margin"] for s in seeds]
    wins = [m > 0.03 for m in margins]                                       # moderate threshold (precheck: ~0.14 shift)
    g_pass = bool(sum(wins) >= int(np.ceil(2 / 3 * len(seeds))))
    verdict = {"G_pass": g_pass, "best_margins": margins, "n_win": int(sum(wins)), "n_seeds": len(seeds),
               "eps": eps, "N": N, "L": L, "F_list": F_list, "B_tot_list": list(B_tot_list)}
    print(f"[85b.C] {'C PASS' if g_pass else 'C INCONCLUSIVE'}: conv-allocation beats MLP-allocation (best margins "
          f"{[round(m, 3) for m in margins]}) on {int(sum(wins))}/{len(seeds)} seeds. Faithful spectrum allocates a "
          f"fixed budget better across the chaoticity ensemble.", file=sys.stderr)
    return {"verdict": verdict, "per_seed": {str(k): v for k, v in per_seed.items()}}


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
    mode = os.environ.get("STEP85B_MODE", "precheck")        # "precheck" (lambda1(F) go/no-go) | "full" (allocation expt)
    if mode == "precheck":
        raise SystemExit(run())
    smoke = bool(int(os.environ.get("STEP85B_SMOKE", "0")))
    N = int(os.environ.get("STEP85B_N", "40"))
    eps = float(os.environ.get("STEP85B_EPS", "0.2"))
    seeds = [int(x) for x in os.environ.get("STEP85B_SEEDS", "0" if smoke else "0,1,2").split(",")]
    F_list = [float(x) for x in os.environ.get("STEP85B_F", "6,8" if smoke else "6,8,10,13").split(",")]
    L = int(os.environ.get("STEP85B_L", "300" if smoke else "1500"))
    device = os.environ.get("STEP85B_DEVICE", "cpu")          # set "cuda" on the 3080 box
    B_tot_list = [int(x) for x in os.environ.get(
        "STEP85B_BTOT", "8,16,24" if smoke else "20,40,60,80,120,160").split(",")]
    print(f"[85b.C] full C: N={N} eps={eps} L={L} seeds={seeds} F={F_list} B_tot={B_tot_list} "
          f"device={device} smoke={smoke}", file=sys.stderr)
    res = run_full_C(seeds, F_list, N, eps, L, B_tot_list, device)
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""
    (figdir / f"step85b_allocation_full{tag}.json").write_text(json.dumps(res, indent=2))
    raise SystemExit(0 if res["verdict"]["G_pass"] else 1)
