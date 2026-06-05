r"""Step 68 — predictability-gated anti-collapse (candidate mechanism ③-1, the real version).

Step 64 found, and Step 66 confirmed, that the FVU>1 pixel residual is NOT latent width: at every dimension the
isotropic VICReg variance floor fills the latent with per-frame-*unpredictable* variance, so the rollout cannot beat
predict-the-mean. The principled fix (not the rank knob) is to change the *objective*: a **predictability-gated**
variance floor that protects variance only in dimensions the predictor can actually predict, letting the unpredictable
"anti-collapse noise" dimensions collapse. `train_jepa(..., predictability_gated_var=True)` implements it: per-dim
weight $w_d=\exp(-e_d/\bar e)$ from the per-dim one-step error $e_d$ (predictable $\to$ protected, noise $\to$ free),
normalized to mean $1$.

This step is a clean A/B on the exact Step-64 frame-averaged $C_4$ pixel JEPA (so the certificate / orbit-flatness is
untouched): isotropic variance vs gated variance, both at $D{=}64$, measured by the collapse-robust FVU. The mechanism
WORKS iff the gated model reaches $\mathrm{FVU}<1$ (beats predict-the-mean) where the isotropic baseline stays $>1$.

Honest gate (prints INCONCLUSIVE rather than loosen a threshold):
  (i)   both models orbit-flat (certificate intact):           ratio < 1.02;
  (ii)  the mechanism works: gated FVU < 1.0 AND gated FVU < baseline FVU by a clear margin (> 0.15).
If gated does not reach FVU<1, we report it honestly (the residual survives a predictability-gated objective too).

Run:   SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
           STEP68_DEVICE=mps .venv/bin/python experiments/step68_gated_variance.py
Seeded:    STEP68_SEED=0|1|2 ...
Writes:    papers/figures/step68_gated_variance.{json,png}
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

from src.training.jepa import collect_transitions, train_jepa  # noqa: E402
from step64_frame_averaged_pixel import (  # noqa: E402
    collect_trajs_pixels,
    latent_partratio,
    make_fa_jepa,
    rollout_fvu,
    rollout_relmse,
)

SEED = int(os.environ.get("STEP68_SEED", "0"))
SMOKE = os.environ.get("STEP68_SMOKE", "0") == "1"
IMG = 65
H = 4
N_TRAIN = 200 if SMOKE else 1500
N_TRAJ = 16 if SMOKE else 96
EPOCHS = int(os.environ.get("STEP68_EPOCHS", "4" if SMOKE else "30"))
DEVICE = os.environ.get("STEP68_DEVICE", "cpu")
TAG = os.environ.get("STEP68_TAG", "")
FIG = ROOT / "papers" / "figures"


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)
    import stable_worldmodel as swm

    world = swm.World("swm/PushT-v1", num_envs=8, image_shape=(IMG, IMG))
    obs, act, nxt = collect_transitions(world, N_TRAIN, seed=SEED)
    Fte, Ate = collect_trajs_pixels(world, N_TRAJ, H, seed=9000 + SEED)
    print(f"[step68] seed={SEED} train={tuple(obs.shape)} device={DEVICE}", file=sys.stderr)

    tk = dict(epochs=EPOCHS, batch_size=64, seed=SEED, device=DEVICE, verbose=False,
              var_coef=float(os.environ.get("STEP68_VARCOEF", "0.04")),
              muon_lr=float(os.environ.get("STEP68_MUON_LR", "0.02")),
              adamw_lr=float(os.environ.get("STEP68_ADAMW_LR", "0.001")))

    rows = {}
    for name, gated in [("isotropic", False), ("gated", True)]:
        torch.manual_seed(SEED)
        model = make_fa_jepa()
        train_jepa(model, obs, act, nxt, predictability_gated_var=gated, **tk)
        model.to("cpu")
        fvu = rollout_fvu(model, Fte, Ate, 0)
        relmse = rollout_relmse(model, Fte, Ate, 0)
        pr = latent_partratio(model, Fte)
        ratio = max(rollout_relmse(model, Fte, Ate, k) for k in range(4)) / max(relmse, 1e-12)
        rows[name] = {"fvu": fvu, "relmse": relmse, "part_ratio": pr, "orbit_ratio": ratio}
        print(f"[step68]   {name:10s}  FVU {fvu:.3f}  relMSE {relmse:.3f}  PR {pr:.1f}  orbit-ratio {ratio:.3f}",
              file=sys.stderr)

    iso, gat = rows["isotropic"], rows["gated"]
    ok_flat = iso["orbit_ratio"] < 1.02 and gat["orbit_ratio"] < 1.02
    ok_works = gat["fvu"] < 1.0 and (iso["fvu"] - gat["fvu"]) > 0.15
    passed = ok_flat and ok_works

    result = {
        "passed": passed,
        "gate": {"orbit_flat_both": ok_flat, "gated_beats_mean_and_isotropic": ok_works},
        "isotropic": iso, "gated": gat,
        "fvu_improvement": iso["fvu"] - gat["fvu"], "seed": SEED, "epochs": EPOCHS, "device": DEVICE, "smoke": SMOKE,
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"step68_gated_variance{TAG}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    labels = ["isotropic\nvariance\n(Step 64)", "predictability-\ngated variance\n(this step)"]
    fvus = [iso["fvu"], gat["fvu"]]
    bars = ax.bar(labels, fvus, color=["C3", "C0"])
    ax.axhline(1.0, ls=":", color="gray", lw=1, label="FVU=1 (predict-the-mean)")
    ax.set_ylabel(f"{H}-step rollout FVU (collapse-robust)")
    ax.set_title("Predictability-gated anti-collapse ($C_4$ PushT pixels)")
    for b, v in zip(bars, fvus):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"step68_gated_variance{TAG}.png", dpi=130, bbox_inches="tight")

    print(f"[step68] gated FVU {gat['fvu']:.3f} vs isotropic {iso['fvu']:.3f} (Δ {iso['fvu']-gat['fvu']:+.3f}); "
          f"orbit-flat both: {ok_flat}", file=sys.stderr)
    if passed:
        print(f"[step68] MECHANISM WORKS: predictability-gated variance reaches FVU {gat['fvu']:.2f}<1 (beats "
              f"predict-the-mean) where the isotropic floor stays {iso['fvu']:.2f}>1 -- resolving the "
              f"variance<->predictability tension, certificate intact.", file=sys.stderr)
    else:
        bad = [k for k, v in result["gate"].items() if not v]
        print(f"[step68] INCONCLUSIVE: gate not met ({bad}); reported as-is (the residual survives gating too).",
              file=sys.stderr)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
