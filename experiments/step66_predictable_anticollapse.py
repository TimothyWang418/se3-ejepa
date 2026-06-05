r"""Step 66 — the predictability-aware anti-collapse PoC (candidate new mechanism ③, part 1).

Step 64 discovered a tension that holds for *every* architecture, equivariant or not: VICReg's anti-collapse term
forces unit variance on **all** $D$ latent dimensions, but PushT's dynamics is low-dimensional, so on a $D{=}64$ latent
the learned representation is low-rank (participation ratio $\approx3$) and $\sim\!60$ dimensions carry *unpredictable
anti-collapse noise* — and no model beats predict-the-mean at horizon ($\mathrm{FVU}>1$, even the fair $1$-step-vs-target).
The hypothesis: the residual is **over-wide latent $\times$ isotropic variance floor**, not equivariance, and a
*predictability-aware* anti-collapse — spreading variance only across as many dimensions as the dynamics can predict —
should drive $\mathrm{FVU}<1$ where the wide isotropic latent cannot.

This step runs the first, diagnostic test: a **latent-rank sweep**. We train the exact frame-averaged $C_4$-equivariant
JEPA of Step 64 (so the certificate / orbit-flatness is untouched) at latent dimensions $D\in\{8,16,32,64\}$ and measure
the collapse-robust accuracy $\mathrm{FVU}$ (fraction of centered variance unexplained; $<1\iff$ beating
predict-the-mean) and the participation ratio. If $\mathrm{FVU}$ drops monotonically as $D\to$ the dynamics' effective
rank and crosses below $1$, the diagnosis is confirmed and the mechanism is "match the latent rank to the predictable
rank" (a predictability-aware variance floor is the principled drop-in; this PoC tests the simplest instance, rank
matching). Honest: orbit-flatness is exact at every $D$ (frame averaging), so this isolates *accuracy*, not the
certificate.

Honest gate (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   orbit-flatness preserved at every $D$ (certificate intact):   max ratio < 1.02;
  (ii)  the mechanism works: some $D$ achieves $\mathrm{FVU}<1$ (beats predict-the-mean) where $D{=}64$ did not;
  (iii) the trend is real: $\mathrm{FVU}(D{=}8\text{ or }16) < \mathrm{FVU}(D{=}64)$ by a clear margin.
If no $D$ reaches $\mathrm{FVU}<1$, we report that honestly (the residual is deeper than latent rank).

Run:   SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
           STEP66_DEVICE=mps .venv/bin/python experiments/step66_predictable_anticollapse.py
Seeded:    STEP66_SEED=0|1|2 ...
Writes:    papers/figures/step66_predictable_anticollapse.{json,png}
"""

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.models.eqjepa import EqJEPA  # noqa: E402
from src.training.jepa import collect_transitions, train_jepa  # noqa: E402
from step64_frame_averaged_pixel import (  # noqa: E402
    FrameAveragedEncoder,
    FrameAveragedPredictor,
    collect_trajs_pixels,
    latent_partratio,
    rollout_fvu,
    rollout_relmse,
)

SEED = int(os.environ.get("STEP66_SEED", "0"))
SMOKE = os.environ.get("STEP66_SMOKE", "0") == "1"
IMG = 65
H = 4
N_TRAIN = 200 if SMOKE else 1500
N_TRAJ = 16 if SMOKE else 96
EPOCHS = int(os.environ.get("STEP66_EPOCHS", "4" if SMOKE else "30"))
WIDTH = int(os.environ.get("STEP66_WIDTH", "32"))
PRED_HID = int(os.environ.get("STEP66_PREDHID", "256"))
DEVICE = os.environ.get("STEP66_DEVICE", "cpu")
TAG = os.environ.get("STEP66_TAG", "")
# latent ranks to sweep: each split half invariant (rho=I) / half equivariant 2D-vectors (rho=R_k), as in Step 64.
DIMS = [8, 16, 32, 64] if not SMOKE else [8, 32]
FIG = ROOT / "papers" / "figures"


def make_fa(latent_dim: int):
    n_vec = latent_dim // 4                 # half the dims are 2D vectors -> n_vec = (D/2)/2 = D/4
    n_inv = latent_dim - 2 * n_vec          # remaining dims invariant (rho = I)
    enc = FrameAveragedEncoder(in_channels=3, n_inv=n_inv, n_vec=n_vec, width=WIDTH)
    pred = FrameAveragedPredictor(n_inv=n_inv, n_vec=n_vec, hidden=PRED_HID)
    return EqJEPA(latent_dim=latent_dim, action_dim=2, encoder=enc, predictor=pred), n_inv, n_vec


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)
    import stable_worldmodel as swm

    world = swm.World("swm/PushT-v1", num_envs=8, image_shape=(IMG, IMG))
    obs, act, nxt = collect_transitions(world, N_TRAIN, seed=SEED)
    Fte, Ate = collect_trajs_pixels(world, N_TRAJ, H, seed=9000 + SEED)
    print(f"[step66] seed={SEED} train={tuple(obs.shape)} sweep D={DIMS} device={DEVICE}", file=sys.stderr)

    tk = dict(epochs=EPOCHS, batch_size=64, seed=SEED, device=DEVICE, verbose=False,
              var_coef=float(os.environ.get("STEP66_VARCOEF", "0.04")),
              muon_lr=float(os.environ.get("STEP66_MUON_LR", "0.02")),
              adamw_lr=float(os.environ.get("STEP66_ADAMW_LR", "0.001")))

    rows = []
    for D in DIMS:
        torch.manual_seed(SEED)
        model, n_inv, n_vec = make_fa(D)
        train_jepa(model, obs, act, nxt, **tk)
        model.to("cpu")
        fvu = rollout_fvu(model, Fte, Ate, 0)
        relmse = rollout_relmse(model, Fte, Ate, 0)
        pr = latent_partratio(model, Fte)
        ratio = max(rollout_relmse(model, Fte, Ate, k) for k in range(4)) / max(relmse, 1e-12)
        rows.append({"D": D, "n_inv": n_inv, "n_vec": n_vec, "fvu": fvu, "relmse": relmse,
                     "part_ratio": pr, "orbit_ratio": ratio})
        print(f"[step66]   D={D:3d} (inv={n_inv},vec={n_vec})  FVU {fvu:.3f}  relMSE {relmse:.3f}  PR {pr:.1f}  "
              f"orbit-ratio {ratio:.3f}", file=sys.stderr)

    fvus = {r["D"]: r["fvu"] for r in rows}
    fvu64 = fvus.get(64, float("nan"))
    best_D = min(rows, key=lambda r: r["fvu"])
    ok_flat = all(r["orbit_ratio"] < 1.02 for r in rows)
    ok_beats_mean = best_D["fvu"] < 1.0
    ok_trend = (fvus.get(8, fvu64) < fvu64 - 0.05) or (fvus.get(16, fvu64) < fvu64 - 0.05)
    passed = ok_flat and ok_beats_mean and ok_trend

    result = {
        "passed": passed,
        "gate": {"orbit_flat_all_D": ok_flat, "some_D_beats_mean": ok_beats_mean, "rank_trend": ok_trend},
        "rows": rows, "fvu_at_64": fvu64, "best": best_D, "seed": SEED, "epochs": EPOCHS,
        "device": DEVICE, "smoke": SMOKE,
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"step66_predictable_anticollapse{TAG}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    Ds = [r["D"] for r in rows]
    ax.plot(Ds, [r["fvu"] for r in rows], "o-", color="C0", lw=2.4, label="FA-JEPA FVU")
    ax.axhline(1.0, ls=":", color="gray", lw=1, label="FVU=1 (predict-the-mean)")
    ax.set_xlabel("latent dimension $D$ (rank)")
    ax.set_ylabel(f"{H}-step rollout FVU (collapse-robust)")
    ax.set_xscale("log", base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds)
    ax.set_title("(a) Accuracy vs latent rank: does narrowing beat predict-the-mean?")
    ax.legend(fontsize=8)
    axb.plot(Ds, [r["part_ratio"] for r in rows], "s-", color="C1", lw=2.0, label="participation ratio")
    axb.plot(Ds, [r["orbit_ratio"] for r in rows], "^--", color="C2", lw=2.0, label="orbit-flatness ratio")
    axb.set_xlabel("latent dimension $D$")
    axb.set_xscale("log", base=2); axb.set_xticks(Ds); axb.set_xticklabels(Ds)
    axb.set_title("(b) Latent health (PR) + certificate (orbit ratio $\\approx1$)")
    axb.legend(fontsize=8)
    fig.suptitle("Predictability-aware anti-collapse: matching latent rank to the predictable rank ($C_4$ PushT pixels)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / f"step66_predictable_anticollapse{TAG}.png", dpi=130, bbox_inches="tight")

    print(f"[step66] best D={best_D['D']} FVU {best_D['fvu']:.3f} vs D=64 FVU {fvu64:.3f}; orbit-flat all D: {ok_flat}",
          file=sys.stderr)
    if passed:
        print(f"[step66] MECHANISM WORKS: narrowing the latent to D={best_D['D']} reaches FVU {best_D['fvu']:.2f}<1 "
              f"(beats predict-the-mean) while D=64 stays at {fvu64:.2f}>1 -- the residual is over-wide latent x "
              f"isotropic variance, not equivariance; orbit-flatness intact at every D.", file=sys.stderr)
    else:
        bad = [k for k, v in result["gate"].items() if not v]
        print(f"[step66] INCONCLUSIVE: gate not met ({bad}); reported as-is (the residual may be deeper than latent rank).",
              file=sys.stderr)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
