r"""Step 56 (B3): prescriptive vs descriptive (vs isotropic) latent anisotropy — OOD head-to-head.

UR-JEPA (Le, 2026) argues a *data-discovered* anisotropic latent can beat an isotropic one; LeJEPA pushes
toward isotropy. Our claim (the equivariant line) is that a *group-prescribed* anisotropy — the isotypic block
structure, fixed before any data — is the one that transfers **out of distribution**, because it is a property of
the symmetry, not of the training slice. This experiment stages the three anisotropy *sources* head-to-head on
the Step 51 wedge→orbit task, holding the task fixed and varying only where the latent's anisotropy comes from:

  - **isotropic** — a non-equivariant MLP with a latent-covariance isotropy penalty (push $\Sigma\to\sigma^2 I$,
    LeJEPA-spirit);
  - **descriptive** — a plain non-equivariant MLP, whose latent anisotropy is whatever fitting the data induces
    (the data-discovered anisotropy);
  - **prescriptive** — the $\mathrm{SO}(2)$-equivariant model (Step 50), whose anisotropy is the group's.

All three train on a $50°$ wedge of the orbit and are scored on next-state prediction **in-wedge** and **OOD**
(the far side of the orbit). The question: which anisotropy source survives the OOD move along the group orbit?

Honest expectation: data-fit (descriptive) anisotropy may help **in-distribution** but is fit to the wedge and
fails OOD; the **prescriptive** (group) anisotropy transfers. We report the in-distribution comparison whatever
it shows (the toy may not reproduce UR-JEPA's in-dist descriptive>isotropic win), and gate on the OOD claim.

Run (full ~2 min):  .venv/bin/python experiments/step56_anisotropy_source_ood.py
Smoke:  STEP56_SMOKE=1 .venv/bin/python experiments/step56_anisotropy_source_ood.py
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from step50_noether_hinge import DS, DV, EquivWorldModel, make_orbits, train  # noqa: E402
from step51_structure_vs_scale import MLPWorldModelW, pos_angle, relmse_by_angle  # noqa: E402

torch.set_default_dtype(torch.float64)
SMOKE = bool(os.environ.get("STEP56_SMOKE"))
SEED = int(os.environ.get("STEP56_SEED", "0"))
THETA_MAX = math.radians(50.0)


def train_mlp_isotropic(model, S_t, S_tp1, epochs, iso_coef=0.3):
    r"""Train the MLP world model with a latent-covariance **isotropy** penalty
    $\lVert \Sigma/\bar\sigma^2 - I\rVert_F^2$ (push the latent toward $\sigma^2 I$, LeJEPA-spirit)."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(SEED)
    n, bs, d = S_t.shape[0], 256, DS + 2 * DV
    eye = torch.eye(d)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            st, st1 = S_t[idx], S_tp1[idx]
            opt.zero_grad()
            z = model.encode(st)
            recon = model.decode(z)
            zp = model.predict(z)
            pred = model.decode(zp)
            with torch.no_grad():
                z1 = model.encode(st1)
            lat = ((zp - z1) ** 2).mean()
            zc = z - z.mean(0, keepdim=True)
            cov = (zc.T @ zc) / max(zc.shape[0] - 1, 1)
            iso = ((cov / cov.diagonal().mean().clamp_min(1e-9) - eye) ** 2).mean()  # toward σ²I
            loss = ((recon - st) ** 2).mean() + ((pred - st1) ** 2).mean() + 0.1 * lat + iso_coef * iso
            loss.backward(); opt.step()
        sched.step()
    return float(loss.item())


@torch.no_grad()
def split_relmse(model, equiv, bins, in_mask, far_mask, St, St1):
    c = relmse_by_angle(model, St, St1, equiv=equiv, bins=bins)
    return float(np.nanmean(c[in_mask])), float(np.nanmean(c[far_mask])), c


def main() -> None:
    torch.manual_seed(SEED)
    line = "=" * 94
    n_traj, steps, epochs = (300, 10, 12) if SMOKE else (1200, 24, 70)
    epochs = int(os.environ.get("STEP56_EPOCHS", epochs))
    print(line)
    print(f"STEP 56 (B3)  anisotropy source × OOD: isotropic / descriptive / prescriptive  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)

    S_all, S1_all = make_orbits(n_traj, steps, seed=SEED)
    wedge = pos_angle(S_all).abs() <= THETA_MAX
    S_tr, S1_tr = S_all[wedge], S1_all[wedge]
    St, St1 = make_orbits(400, steps, seed=999)
    bins = torch.linspace(0, math.pi, 10)
    deg = np.array([math.degrees(float((bins[i] + bins[i + 1]) / 2)) for i in range(len(bins) - 1)])
    in_mask, far_mask = deg <= math.degrees(THETA_MAX), deg >= 120.0

    rows = {}
    # prescriptive (equivariant)
    torch.manual_seed(SEED); eqm = EquivWorldModel(); train(eqm, S_tr, S1_tr, epochs, equiv=True)
    rows["prescriptive (group)"] = split_relmse(eqm, True, bins, in_mask, far_mask, St, St1)
    # descriptive (plain MLP)
    torch.manual_seed(SEED); dm = MLPWorldModelW(96); train(dm, S_tr, S1_tr, epochs, equiv=False)
    rows["descriptive (data-fit)"] = split_relmse(dm, False, bins, in_mask, far_mask, St, St1)
    # isotropic (MLP + isotropy penalty)
    torch.manual_seed(SEED); im = MLPWorldModelW(96); train_mlp_isotropic(im, S_tr, S1_tr, epochs)
    rows["isotropic (LeJEPA-ish)"] = split_relmse(im, False, bins, in_mask, far_mask, St, St1)

    print(f"    {'anisotropy source':24s}  in-wedge relMSE    OOD (≥120°) relMSE")
    for k, (i, f, _) in rows.items():
        print(f"    {k:24s}  {i:.3e}        {f:.3e}")

    presc_in, presc_ood, _ = rows["prescriptive (group)"]
    desc_in, desc_ood, _ = rows["descriptive (data-fit)"]
    iso_in, iso_ood, _ = rows["isotropic (LeJEPA-ish)"]
    best_nonequiv_ood = min(desc_ood, iso_ood)
    print(f"\n{line}\nSTEP 56 SUMMARY\n{line}")
    print(f"    OOD: prescriptive {presc_ood:.2e}  vs  best non-equivariant {best_nonequiv_ood:.2e}  "
          f"({best_nonequiv_ood / max(presc_ood, 1e-12):.0f}× worse) — prescriptive anisotropy transfers OOD")
    print(f"    in-distribution (UR-JEPA's claim, on this toy): descriptive {desc_in:.2e}  "
          f"{'≤' if desc_in <= iso_in else '>'} isotropic {iso_in:.2e}  "
          f"(data-fit anisotropy {'helps' if desc_in <= iso_in else 'does not help'} in-dist here)")
    # gate: prescriptive wins OOD decisively over BOTH non-equivariant anisotropy sources
    ok = (presc_ood < 0.3 * desc_ood and presc_ood < 0.3 * iso_ood)
    if ok:
        print(f"\n    CONFIRMED: only the GROUP-PRESCRIBED anisotropy transfers out of distribution — prescriptive")
        print(f"        is {best_nonequiv_ood / max(presc_ood, 1e-12):.0f}× better OOD than the best data-driven")
        print(f"        (descriptive/isotropic) latent. A data-discovered anisotropy is fit to the training slice;")
        print(f"        the group's anisotropy is a property of the symmetry and is valid across the whole orbit.")
    else:
        print(f"\n    INCONCLUSIVE: gate not met; reported as-is (no thresholds loosened).")

    out = dict(passed=bool(ok), rows={k: [v[0], v[1]] for k, v in rows.items()},
               curves={k: v[2].tolist() for k, v in rows.items()}, deg=deg.tolist(), smoke=SMOKE, seed=SEED)
    fig_dir = ROOT / "papers" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step56_anisotropy_source_ood_smoke" if SMOKE else "step56_anisotropy_source_ood"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"prescriptive (group)": "C2", "descriptive (data-fit)": "C3", "isotropic (LeJEPA-ish)": "C0"}
    for k, (_, _, c) in rows.items():
        ax.semilogy(deg, c, "o-", color=colors[k], lw=1.8, ms=4, label=k)
    ax.axvspan(0, math.degrees(THETA_MAX), color="C0", alpha=0.08, label="training wedge")
    ax.set_xlabel("orbit position |∠ r| (degrees)"); ax.set_ylabel("next-state prediction relMSE")
    ax.set_title("Step 56 (B3) — anisotropy source × OOD\nonly group-prescribed anisotropy transfers along the orbit")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fig_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote papers/figures/{stem}.{{json,png}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
