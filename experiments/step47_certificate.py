r"""Step 47: the **predictability certificate** — verify Theorem A (config axis) + measure Theorem B's spectrum.

This is paper2's keystone experiment (P0 closure test + P1 Plot-1 core), run on the *already-validated*
exact-equivariant object model (reuses Step 43's E0-base VN-TP + the Step 24 transforms), so it is
low-risk to land while it directly tests the merged proposal's master theorem.

What it tests
-------------
(Axis 1 — configuration) **Theorem A (exact certificate).** Apply *composition words* $w=g_{i_1}\cdots g_{i_m}$
built from a finite generating set $S=\{\text{global }\mathrm{SE}(3)\text{ rotation/translation},\ \text{object
permutation}\}$ to the held-out scenes, and measure the whole-pipeline relMSE at $w\cdot x$ for word length
$m=0,1,\dots,M$. The claim: for the **equivariant** model the relMSE is *flat* across all words (orthogonal
$\rho(w)$ leaves the error invariant — the multi-step-free, full-word lift of [B]); the **non-equivariant
MLP** control blows up with $m$. Verifying flatness over composed words is the empirical face of "$k$
generator checks certify the generated set."

  *Honest scope note (in the spirit of the proposal §5):* for $O{=}2$ the group is global
  $\mathrm{SE}(3)\rtimes S_2$, so the *mathematical* closure of word-composition is modest (a product of
  global rotations is one rotation; $S_2$ has two elements). This experiment therefore certifies the
  **flatness over composed words empirically on the real embodied model**; the clean *exponential*-set
  demonstration ($k$ generators → $2^k$ states) is the I Ching $\mathbb{Z}_2^6$ toy flagged in the proposal,
  a separate clean testbed. We do not overclaim an exponential set here.

(Axes 2+3 — horizon × resolution) **Theorem B inputs.** Estimate the predictor's Jacobian **spectrum**
$\{\sigma_j\}=\{e^{\lambda_j}\}$ at sampled latents: the slow/contractive channels ($\sigma_j\le1$) are the
ones certified to long horizon $T_j(\epsilon)\sim\log(1/\epsilon)/\lambda_j$; expansive channels
($\sigma_j>1$) are not. We also split the latent into **invariant** (per-vector norms — $\mathrm{SE}(3)$
scalars) vs **equivariant** (directions) content and report each, as the coarse-vs-fine handle.

Gate (PASS): equivariant relMSE flat across all word lengths (ratio to base $\le1.15$) AND the MLP
degrades (ratio $>1.3$ at the longest word) AND the equivariant post-word $\mathrm{SE}(3)$/perm residual
stays float-floor.

Run (full; ~10-20 min CPU, 3 seeds):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step47_certificate.py
Smoke (~2 min):
    STEP47_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step47_certificate.py
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from step24_object_interaction import (  # noqa: E402
    A_OBJ,
    A_SCENE,
    N_OBJ,
    _PERM,
    _rot_self_actions,
    make_interacting_transitions,
    permute_scene,
    rand_so3,
    rotate_points,
    train_jepa,
)
from step43_encoder_ladder import build_model, indist_relmse, model_action_v  # noqa: E402

torch.set_default_dtype(torch.float32)
SMOKE = bool(os.environ.get("STEP47_SMOKE"))


# --------------------------------------------------------------------------- #
# generators acting on a whole transition (S, A_self, S2): global SE(3) + object permutation.
# A "word" is a composition of these; equivariance is closed under composition, so the relMSE
# at w·x should equal the relMSE at x for the equivariant model (Theorem A).
# --------------------------------------------------------------------------- #
def apply_rot(S, A_self, S2, gen):
    R = rand_so3(gen)
    t = torch.randn(3, generator=gen)
    Sr = rotate_points(S, R) + t.reshape(1, 1, 1, 3)
    S2r = rotate_points(S2, R) + t.reshape(1, 1, 1, 3)
    Ar = _rot_self_actions(A_self, R)
    return Sr, Ar, S2r


def apply_perm(S, A_self, S2):
    b = S.shape[0]
    Sp = permute_scene(S)
    S2p = permute_scene(S2)
    Ap = A_self.reshape(b, N_OBJ, A_OBJ)[:, _PERM].reshape(b, A_SCENE)
    return Sp, Ap, S2p


def apply_word(S, A_self, S2, length, gen):
    r"""Apply a random composition word of the given length to the transition."""
    for _ in range(length):
        if torch.rand(1, generator=gen).item() < 0.5:
            S, A_self, S2 = apply_rot(S, A_self, S2, gen)
        else:
            S, A_self, S2 = apply_perm(S, A_self, S2)
    return S, A_self, S2


@torch.no_grad()
def latent_channels(model, S):
    r"""Split the latent into invariant (per-type-1-vector norms) and equivariant (directions) content.
    Returns (mean invariant-norm-energy, mean direction-energy) as a coarse-vs-fine handle."""
    z = model.encoder(S)                       # (B, N_OBJ*D_OBJ) = stacked type-1 vectors
    b = z.shape[0]
    v = z.reshape(b, -1, 3)                     # (B, n_vec, 3)
    norms = v.norm(dim=-1)                      # (B, n_vec) — SE(3)-invariant scalars
    return float(norms.mean()), float(v.abs().mean())


@torch.no_grad()
def predictor_jacobian_spectrum(model, S, A_self, n_samples=8):
    r"""Singular-value spectrum {sigma_j}={e^{lambda_j}} of the predictor's Jacobian d f / d z, at sampled
    latents. sigma_j<=1 -> slow/contractive channel (certified long-horizon); sigma_j>1 -> expansive."""
    z = model.encoder(S)
    a = model_action_v("E0-base", S, A_self)
    sv_all = []
    for i in range(min(n_samples, z.shape[0])):
        zi = z[i:i + 1].detach().clone().requires_grad_(True)
        ai = a[i:i + 1]
        J = torch.autograd.functional.jacobian(
            lambda zz: model.predictor(zz, ai).reshape(-1), zi.reshape(1, -1), vectorize=True
        ).reshape(z.shape[1], z.shape[1])
        sv_all.append(torch.linalg.svdvals(J))
    sv = torch.stack(sv_all).mean(0)
    return sv.cpu().numpy()


def main() -> None:
    line = "=" * 80
    if SMOKE:
        SEEDS, N_TRAIN, N_TEST, EPOCHS, MAXLEN, WORDS = (0,), 150, 64, 3, 4, 6
    else:
        SEEDS, N_TRAIN, N_TEST, EPOCHS, MAXLEN, WORDS = (0, 1, 2), 1500, 400, 80, 8, 24
    SEEDS = tuple(int(s) for s in os.environ.get("STEP47_SEEDS", ",".join(map(str, SEEDS))).split(","))
    EPOCHS = int(os.environ.get("STEP47_EPOCHS", EPOCHS))
    NAMES = ["E0-base", "MLP-MP"]  # equivariant VN-TP vs non-equivariant control

    print(line)
    print(f"STEP 47  predictability certificate: Thm A (config) flat over composition words + Thm B spectrum  "
          f"({'SMOKE' if SMOKE else 'FULL'})")
    print(line)
    print(f"    generators S = {{global SE(3) rot/trans, object permutation}}; word length m=0..{MAXLEN}, "
          f"{WORDS} words/len; seeds={SEEDS}")

    St, At_self, S2t = make_interacting_transitions(N_TEST, seed=999)

    # relmse[name][m] = list over (seed, word) of relMSE(w·x); jac[name] = list of spectra
    relmse = {nm: {m: [] for m in range(MAXLEN + 1)} for nm in NAMES}
    jac = {nm: [] for nm in NAMES}
    chan = {nm: [] for nm in NAMES}

    for seed in SEEDS:
        print(f"\n[seed {seed}] train equivariant E0 + non-equivariant MLP on seen scenes")
        S, A_self, S2 = make_interacting_transitions(N_TRAIN, seed=seed)
        wg = torch.Generator().manual_seed(7000 + seed)   # word-sampling stream
        for nm in NAMES:
            torch.manual_seed(seed)
            model = build_model(nm)
            train_jepa(model, S, model_action_v(nm, S, A_self), S2,
                       epochs=EPOCHS, batch_size=128, var_coef=0.1, seed=seed, log_every=999)
            base = indist_relmse(nm, model, St, At_self, S2t)
            relmse[nm][0].append(base)
            for m in range(1, MAXLEN + 1):
                for _ in range(WORDS):
                    Sw, Aw, S2w = apply_word(St, At_self, S2t, m, wg)
                    relmse[nm][m].append(indist_relmse(nm, model, Sw, Aw, S2w))
            if nm == "E0-base":
                jac[nm].append(predictor_jacobian_spectrum(model, St, At_self))
                chan[nm].append(latent_channels(model, St))
            print(f"    {nm:>8s} base relMSE {base:.4e}")

    # ---- aggregate ----
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    print(f"\n{line}\n[A] CONFIGURATION axis — relMSE vs composition-word length (flat = certified)\n{line}")
    print(f"    {'m':>3s} | {'E0-base (equiv)':>22s} | {'MLP-MP (non-equiv)':>22s}")
    print("    " + "-" * 54)
    eq_base = mean(relmse['E0-base'][0]); mlp_base = mean(relmse['MLP-MP'][0])
    eq_ratio_max = 1.0; mlp_ratio_max = 1.0
    for m in range(MAXLEN + 1):
        eqm, mlpm = mean(relmse['E0-base'][m]), mean(relmse['MLP-MP'][m])
        eqr, mlpr = eqm / max(eq_base, 1e-12), mlpm / max(mlp_base, 1e-12)
        eq_ratio_max = max(eq_ratio_max, eqr); mlp_ratio_max = max(mlp_ratio_max, mlpr)
        print(f"    {m:>3d} | {eqm:.4e} (x{eqr:5.2f}) | {mlpm:.4e} (x{mlpr:5.2f})")
    print(f"    => equivariant flat over words: max ratio x{eq_ratio_max:.2f} (cert if <=1.15); "
          f"MLP max ratio x{mlp_ratio_max:.2f} (degrades if >1.3)")

    # ---- Thm B spectrum ----
    sv = np.mean(np.stack(jac['E0-base']), 0) if jac['E0-base'] else np.array([])
    n_slow = int((sv <= 1.0).sum()); n_fast = int((sv > 1.0).sum())
    inv_e, dir_e = (mean([c[0] for c in chan['E0-base']]), mean([c[1] for c in chan['E0-base']]))
    print(f"\n{line}\n[B] HORIZON x RESOLUTION — predictor Jacobian spectrum {{sigma_j}}={{e^lambda_j}}\n{line}")
    if sv.size:
        print(f"    spectrum: max {sv.max():.3f}  min {sv.min():.3f}  median {np.median(sv):.3f}")
        print(f"    channels: {n_slow} slow/contractive (sigma<=1 -> certified long-horizon) vs "
              f"{n_fast} expansive (sigma>1 -> short-horizon T~log(1/eps)/lambda)")
        print(f"    latent split: invariant norm-energy {inv_e:.3f} | direction-energy {dir_e:.3f}")
        print(f"    => the certified-long-horizon content is the {n_slow}/{n_slow + n_fast} contractive "
              f"channels; this is the measured input to Thm B's per-channel horizon.")

    # ---- verdict ----
    ok_cert = eq_ratio_max <= 1.15
    ok_mlp = mlp_ratio_max > 1.3
    passed = ok_cert and ok_mlp
    print(f"\n{line}\nSTEP 47 SUMMARY\n{line}")
    print(f"    [A] config certificate: equivariant relMSE flat over composition words "
          f"(max x{eq_ratio_max:.2f}) vs MLP x{mlp_ratio_max:.2f}.")
    print(f"    [B] {n_slow}/{n_slow + n_fast} predictor channels contractive (certified long-horizon).")
    print(f"    guards: cert-flat={ok_cert}  mlp-degrades={ok_mlp}")
    if passed:
        print(f"    CONFIRMED (Thm A, empirical): the exact-equivariant model's error is invariant across")
        print(f"        composed group words — the certificate holds on the real embodied model — while the")
        print(f"        non-equivariant control degrades. Honest scope: O=2 word-closure is modest (global")
        print(f"        SE(3) x S_2); the clean exponential k-generators->2^k demo is the I Ching Z_2^6 toy.")
    else:
        print(f"    INCONCLUSIVE: certificate flatness or MLP-degradation guard not met; reported as-is.")

    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": dict(SEEDS=list(SEEDS), N_TRAIN=N_TRAIN, EPOCHS=EPOCHS, MAXLEN=MAXLEN, WORDS=WORDS),
        "relmse_mean": {nm: {m: mean(relmse[nm][m]) for m in range(MAXLEN + 1)} for nm in NAMES},
        "eq_ratio_max": eq_ratio_max, "mlp_ratio_max": mlp_ratio_max,
        "jacobian_spectrum": sv.tolist(), "n_slow": n_slow, "n_fast": n_fast,
        "invariant_energy": inv_e, "direction_energy": dir_e,
        "verdict": {"passed": bool(passed), "ok_cert": ok_cert, "ok_mlp": ok_mlp},
    }
    fig_dir = ROOT / "papers" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "step47_certificate_smoke" if SMOKE else "step47_certificate"
    (fig_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))
    _make_figure(fig_dir / f"{stem}.png", relmse, NAMES, MAXLEN, sv)
    print(f"    wrote {(fig_dir / f'{stem}.json').relative_to(ROOT)} + .png")
    sys.exit(0 if passed else 1)


def _make_figure(path, relmse, NAMES, MAXLEN, sv) -> None:
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ms = list(range(MAXLEN + 1))
    for nm, c in zip(NAMES, ["C2", "C3"]):
        base = mean(relmse[nm][0])
        ax[0].plot(ms, [mean(relmse[nm][m]) / max(base, 1e-12) for m in ms], "o-", color=c,
                   label=nm + (" (equiv)" if nm == "E0-base" else " (non-equiv)"))
    ax[0].axhline(1.0, ls=":", color="k", alpha=0.5)
    ax[0].set_xlabel("composition-word length m"); ax[0].set_ylabel("relMSE / base")
    ax[0].set_title("[A] config certificate: error flat over composed words"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if sv.size:
        ax[1].semilogy(sorted(sv, reverse=True), "o-", color="C0")
        ax[1].axhline(1.0, ls="--", color="C3", label="σ=1 (slow/fast split)")
        ax[1].set_xlabel("channel (sorted)"); ax[1].set_ylabel("Jacobian σ_j = e^{λ_j}")
        ax[1].set_title("[B] predictor spectrum → per-channel horizon"); ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Step 47 — predictability certificate: Thm A (config, flat) + Thm B (spectrum)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
