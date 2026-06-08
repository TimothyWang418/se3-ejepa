r"""Step 77 — the RECURRENT (GRU-BPTT) baseline for the high-$N$ spectrum-recovery claim (defends Step 74 / Exp 18).

The cleanest possible answer to the reviewer attack "an unstructured *recurrent* model (RNN/RC) recovers chaotic
Lyapunov spectra without symmetry (Vlachas et al. 2020 with GRU+BPTT; Pathak/Kobayashi with reservoirs), so your
*feedforward* dense-MLP baseline is the wrong one." We add a GRU trained with the **identical** recipe as Step 74's
$\mathbb{Z}_N$-conv and dense MLP — same data, same residual parametrization, same multi-step rollout loss, same Adam
— so the *only* difference from the equivariant model is the architecture's inductive bias (recurrent-but-dense vs
$\mathbb{Z}_N$-banded). This isolates "structure" from "training regime" and "timescale" (it stays at the paper's
native $\Delta t{=}0.01$, leaving Step 74's canonical conv/MLP numbers untouched).

**Mechanism prediction.** The GRU's autonomous map lives on the joint state $(x,h)\in\mathbb R^{N+H}$ and its Jacobian
is *dense* (no banded-circulant structure). Step 74's thesis — unstructured dense Jacobian mass wrecks the high-$N$
spectrum — predicts the GRU should recover at low $N$ (positive control) yet fail at $N{=}40$, just like the MLP,
*despite* being recurrent. That is the falsifiable thing we test.

**Positive control (the linchpin).** Vlachas et al. recover the low-D Lorenz spectrum with exactly this GRU+BPTT tool,
so our GRU MUST recover at low $N$; otherwise the implementation/tuning is inadequate and the $N{=}40$ number cannot
be trusted. We run $N{=}N_{\mathrm{ctrl}}$ (default 12) first.

**GRU.** $h_{t+1}=\mathrm{GRUCell}(x_t,h_t)$, residual readout $x_{t+1}=x_t+W_o h_{t+1}$ (the $\Delta t$-map is
identity + increment, matching the conv/MLP residual). Trained with a teacher-forced warmup (build $h$) then a
$K$-step closed-loop rollout loss — the regime that already makes the conv/MLP stable. **Lyapunov spectrum:** the
autonomous joint map $M(x,h)=(x+W_o\,\mathrm{GRUCell}(x,h),\ \mathrm{GRUCell}(x,h))$ is fed to Step 74's *already-unit-
tested* ``lyapunov_spectrum`` (Benettin QR, autograd Jacobian) over $\mathbb R^{N+H}$; the leading $N$ exponents
approximate the true spectrum (the extra $H$ are the GRU's spurious contracting modes, like a reservoir's).

Honest gate (prints which branch; never loosens a threshold):
  (PC)  positive control: GRU at N_ctrl recovers (R^2 > 0.80). If it FAILS -> ABORT ("GRU impl/tuning not trustworthy").
  Then the headline is whichever is TRUE at N=40:
   (A) STRUCTURE-STILL-WINS : GRU R^2 < 0.50 AND (conv R^2 - GRU R^2) > 0.40  -> claim holds, hardened.
   (B) SOFTEN               : GRU R^2 > 0.80                                  -> a same-trained recurrent model also
                                                                                recovers; narrow the claim.
   (else) INCONCLUSIVE.

Run:    .venv/bin/python experiments/step77_lorenz96_gru_baseline.py
smoke:  STEP77_SMOKE=1 .venv/bin/python experiments/step77_lorenz96_gru_baseline.py
seeded: STEP77_SEED=0|1|2     (loads papers/figures/step74_lorenz96_spectrum{_seedS}.json for conv/MLP)
Writes: papers/figures/step77_gru_baseline{_seedS}.{json,png}     device: cpu, float64 (matches Step 74)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step74_lorenz96_spectrum as s74  # noqa: E402  (shares dynamics, true spectrum, unit-tested QR, kaplan_yorke)

SMOKE = bool(int(os.environ.get("STEP77_SMOKE", "0")))
SEED = int(os.environ.get("STEP77_SEED", "0"))
DEVICE = "cpu"
DTYPE = torch.float64
NCTRL = int(os.environ.get("STEP77_NCTRL", "12"))
HIDDEN = int(os.environ.get("STEP77_HIDDEN", "32" if "1" == os.environ.get("STEP77_SMOKE", "0") else "96"))
# Hidden-state noise injection (Jaeger/Pathak regularizer): forces the DRIVEN hidden dynamics to contract so the H
# spurious conditional-Lyapunov exponents fall well below the true minimum (Hart 2024) — without it the GRU's near-
# neutral memory modes pollute the top-N spectrum. Validated: hnoise=0.05 recovers the N=12 control to R^2=0.95.
HNOISE = float(os.environ.get("STEP77_HNOISE", "0.05"))


class L96GRU(nn.Module):
    r"""Residual GRU one-step model of the $\Delta t$-map: $h'=\mathrm{GRUCell}(x,h)$, $x'=x+W_o h'$."""

    def __init__(self, N: int, hidden: int):
        super().__init__()
        self.N, self.hidden = N, hidden
        self.cell = nn.GRUCell(N, hidden)
        self.readout = nn.Linear(hidden, N)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h2 = self.cell(x, h)
        return x + self.readout(h2), h2


def train_gru(traj: torch.Tensor, N: int, hidden: int, seed: int):
    r"""Identical recipe to Step 74's ``train_model``: normalized coords, residual map, **multi-step rollout loss**;
    plus a teacher-forced warmup of length ``W_tf`` so the hidden state is "developed" (matching the synced state used
    at spectrum time, and giving a stable autonomous rollout)."""
    torch.manual_seed(seed)
    mu = traj.mean(0); sd = traj.std(0) + 1e-8
    xn = (traj - mu) / sd
    K = 2 if SMOKE else 5
    W_tf = 3 if SMOKE else 10
    epochs = 8 if SMOKE else 80
    L = W_tf + K
    model = L96GRU(N, hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xn.shape[0] - L - 1
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            h = torch.zeros(idx.shape[0], hidden, dtype=DTYPE)
            for j in range(W_tf):                                  # teacher-forced warmup: feed TRUE x, build h
                _, h = model(xn[idx + j], h)
                if HNOISE > 0:                                     # contraction regularizer (see HNOISE note)
                    h = h + HNOISE * torch.randn(h.shape, generator=g, dtype=DTYPE)
            z = xn[idx + W_tf]
            loss = 0.0
            for j in range(K):                                     # closed-loop rollout: feed own prediction
                z, h = model(z, h)
                if HNOISE > 0:
                    h = h + HNOISE * torch.randn(h.shape, generator=g, dtype=DTYPE)
                loss = loss + ((z - xn[idx + W_tf + 1 + j]) ** 2).mean()
            loss = loss / K
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():                                          # one-step (teacher-forced) relMSE, h warmed from 0
        h = torch.zeros(xn.shape[0] - 1, hidden, dtype=DTYPE)
        p, _ = model(xn[:-1], h)
        relmse = (((p - xn[1:]) ** 2).sum(-1) / (xn[1:] ** 2).sum(-1).clamp_min(1e-12)).mean().item()
    return model, mu, sd, relmse


@torch.no_grad()
def sync_state(model: L96GRU, xn_window: torch.Tensor) -> torch.Tensor:
    r"""Teacher-force through ``xn_window`` (W, N) to synchronize the hidden state; return joint state ``[x, h]``."""
    h = torch.zeros(model.hidden, dtype=DTYPE)
    x = xn_window[0]
    for t in range(xn_window.shape[0]):
        x = xn_window[t]
        _, h = model(x.unsqueeze(0), h.unsqueeze(0))
        h = h.squeeze(0)
    return torch.cat([x, h], 0)


def autonomous_map(model: L96GRU):
    r"""Return the joint autonomous map $M:[x,h]\mapsto[x+W_o\,\mathrm{GRUCell}(x,h),\ \mathrm{GRUCell}(x,h)]$ on
    $\mathbb R^{N+H}$, whose Lyapunov spectrum Step 74's QR estimates (leading $N$ exponents approximate the truth)."""
    N = model.N

    def M(state: torch.Tensor) -> torch.Tensor:
        x, h = state[:N], state[N:]
        x2, h2 = model(x.unsqueeze(0), h.unsqueeze(0))
        return torch.cat([x2.squeeze(0), h2.squeeze(0)], 0)

    return M


def eval_gru(N: int, seed: int, lam_true: torch.Tensor, hidden: int, tag: str) -> dict:
    n_train = 3000 if SMOKE else 20000
    ly_steps = 500 if SMOKE else 2500
    ly_warm = 100 if SMOKE else 400
    washout = 60 if SMOKE else 200

    traj = s74.attractor_traj(N, n_train, seed, DEVICE).to(DTYPE)
    model, mu, sd, relmse = train_gru(traj, N, hidden, seed)
    xn = (traj - mu) / sd
    state0 = sync_state(model, xn[len(xn) // 2 - washout: len(xn) // 2])     # synced joint state on the attractor
    lam_full = s74.lyapunov_spectrum(autonomous_map(model), state0, ly_steps, ly_warm)   # N+H exponents
    lamN = lam_full[:N]                                                      # leading N approximate the true spectrum
    lt = lam_true.cpu().numpy(); ll = lamN.cpu().numpy()
    l1_t = float(lam_true[0])
    r2 = 1.0 - float(((ll - lt) ** 2).sum()) / max(float(((lt - lt.mean()) ** 2).sum()), 1e-12)
    out = {"tag": tag, "N": N, "hidden": hidden, "one_step_relmse": relmse,
           "lambda1": float(lamN[0]), "lambda1_relerr": abs(float(lamN[0]) - l1_t) / abs(l1_t),
           "sum_topN": float(lamN.sum()), "n_positive": int((lamN > 0).sum()),
           "ky": s74.kaplan_yorke(lamN), "spectrum_r2": r2,
           "spurious_lyap_max": float(lam_full[N:].max()) if lam_full.shape[0] > N else float("nan"),
           "lambda_learned": ll.tolist()}
    print(f"[step77]   {tag} N={N} H={hidden}: relMSE {relmse:.1e}  spectrum R^2 {r2:.3f}  "
          f"lambda1 {float(lamN[0]):.3f} (err {out['lambda1_relerr']:.0%})  #pos {out['n_positive']}  "
          f"KY {out['ky']:.2f}", file=sys.stderr)
    return out


def _load_step74(seed: int) -> dict:
    tag = f"_seed{seed}" if seed else ""
    p = Path(__file__).resolve().parent.parent / "papers" / "figures" / f"step74_lorenz96_spectrum{tag}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run(seed: int) -> int:
    N = int(os.environ.get("STEP77_N", "10" if SMOKE else "40"))
    print(f"[step77] GRU-BPTT baseline (seed {seed}, {DEVICE}, float64, H={HIDDEN}); "
          f"positive control N={NCTRL}, test N={N}", file=sys.stderr)

    # ---- positive control: the GRU must recover at low N (else implementation/tuning is untrustworthy) ----
    traj_c = s74.attractor_traj(NCTRL, 1500 if not SMOKE else 600, seed, DEVICE).to(DTYPE)
    mu_c, sd_c = traj_c.mean(0), traj_c.std(0) + 1e-8
    x0_c = (traj_c[len(traj_c) // 2] - mu_c) / sd_c
    lam_true_c = s74.lyapunov_spectrum(lambda xn: (s74.true_map(xn * sd_c + mu_c) - mu_c) / sd_c, x0_c,
                                       600 if not SMOKE else 200, 100 if not SMOKE else 50)
    ctrl = eval_gru(NCTRL, seed, lam_true_c, HIDDEN, tag="ctrl")
    pc_pass = bool(ctrl["spectrum_r2"] > 0.80)

    # ---- the test at N (same data/Delta t as Step 74); compare to Step 74's conv/MLP ----
    s74j = _load_step74(seed)
    if s74j and s74j.get("N") == N:
        lam_true = torch.tensor(s74j["lambda_true"], dtype=DTYPE)
        conv_r2, mlp_r2 = s74j["equivariant"]["spectrum_r2"], s74j["mlp"]["spectrum_r2"]
    else:
        traj = s74.attractor_traj(N, 20000 if not SMOKE else 3000, seed, DEVICE).to(DTYPE)
        mu, sd = traj.mean(0), traj.std(0) + 1e-8
        x0 = (traj[len(traj) // 2] - mu) / sd
        lam_true = s74.lyapunov_spectrum(lambda xn: (s74.true_map(xn * sd + mu) - mu) / sd, x0,
                                         2500 if not SMOKE else 600, 400 if not SMOKE else 100)
        conv_r2, mlp_r2 = float("nan"), float("nan")
    gru = eval_gru(N, seed, lam_true, HIDDEN, tag="test")

    gru_r2 = gru["spectrum_r2"]
    have_ref = bool(s74j and s74j.get("N") == N)
    structure_wins = bool(pc_pass and gru_r2 < 0.50 and have_ref and (conv_r2 - gru_r2) > 0.40)
    soften = bool(pc_pass and gru_r2 > 0.80)
    branch = ("ABORT_PC_FAIL" if not pc_pass else
              "STRUCTURE_STILL_WINS" if structure_wins else
              "SOFTEN" if soften else "INCONCLUSIVE")

    res = {"seed": seed, "smoke": SMOKE, "N": N, "N_ctrl": NCTRL, "hidden": HIDDEN, "hidden_noise": HNOISE, "branch": branch,
           "positive_control_pass": pc_pass, "positive_control": ctrl, "gru_test": gru,
           "conv_r2": conv_r2, "mlp_r2": mlp_r2, "gru_r2": gru_r2,
           "conv_minus_gru": (conv_r2 - gru_r2) if have_ref else None,
           "lambda_true": lam_true.cpu().numpy().tolist()}
    _save(res)

    print(f"\n[step77] === RESULT (seed {seed}) ===", file=sys.stderr)
    print(f"[step77] positive control N={NCTRL}: GRU R^2 {ctrl['spectrum_r2']:.3f} -> "
          f"{'PASS' if pc_pass else 'FAIL (impl/tuning untrustworthy)'}", file=sys.stderr)
    print(f"[step77] test N={N}: Z_N-conv R^2 {conv_r2:.3f}  dense-MLP R^2 {mlp_r2:.3f}  GRU R^2 {gru_r2:.3f}",
          file=sys.stderr)
    if branch == "STRUCTURE_STILL_WINS":
        print(f"[step77] >>> STRUCTURE STILL WINS: a same-trained *recurrent* GRU (Vlachas's spectrum-recoverer, valid "
              f"at N={NCTRL}) fails at N={N} (R^2 {gru_r2:.3f}) just like the dense MLP; only the Z_N prior recovers "
              f"the 40-D spectrum. Structure — not recurrence, training, or scale — is what closes it.", file=sys.stderr)
    elif branch == "SOFTEN":
        print(f"[step77] >>> SOFTEN: a same-trained recurrent GRU ALSO recovers the N={N} spectrum (R^2 {gru_r2:.3f}). "
              f"Narrow the claim — recurrence, not the Z_N prior, may be sufficient.", file=sys.stderr)
    elif branch == "INCONCLUSIVE":
        print(f"[step77] >>> INCONCLUSIVE: GRU R^2 {gru_r2:.3f} between the gates (partial recovery).", file=sys.stderr)
    else:
        print(f"[step77] >>> ABORT: positive control failed — fix/tune the GRU before trusting the N={N} number.",
              file=sys.stderr)
    return 0 if branch in ("STRUCTURE_STILL_WINS", "SOFTEN") else 1


def _save(res: dict) -> None:
    figdir = Path(__file__).resolve().parent.parent / "papers" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"_seed{res['seed']}" if res["seed"] else ""
    (figdir / f"step77_gru_baseline{tag}.json").write_text(json.dumps(res, indent=2))
    try:
        lt = np.array(res["lambda_true"]); lg = np.array(res["gru_test"]["lambda_learned"])
        s74j = _load_step74(res["seed"])
        fig, ax = plt.subplots(1, 1, figsize=(5.6, 5.0))
        lo, hi = float(min(lt.min(), lg.min())), float(max(lt.max(), lg.max()))
        if s74j and s74j.get("N") == res["N"]:
            lc = np.array(s74j["equivariant"]["lambda_learned"]); lm = np.array(s74j["mlp"]["lambda_learned"])
            lo = float(min(lo, lc.min(), lm.min())); hi = float(max(hi, lc.max(), lm.max()))
            ax.scatter(lt, lc, s=22, color="#1f77b4", zorder=3, label=f"$\\mathbb{{Z}}_N$-conv ($R^2={res['conv_r2']:.2f}$)")
            ax.scatter(lt, lm, s=26, color="#d62728", marker="x", zorder=2, label=f"dense MLP ($R^2={res['mlp_r2']:.2f}$)")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
        ax.scatter(lt, lg, s=30, facecolors="none", edgecolors="#2ca02c", zorder=4,
                   label=f"GRU-BPTT ($R^2={res['gru_r2']:.2f}$)")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("true Lyapunov exponent $\\lambda_j$")
        ax.set_ylabel("learned-model exponent $\\hat\\lambda_j$")
        ax.set_title(f"{res['N']}-D Lorenz-96: recurrent (GRU) baseline vs structure\n"
                     f"(positive control N={res['N_ctrl']}: GRU $R^2={res['positive_control']['spectrum_r2']:.2f}$)")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(figdir / f"step77_gru_baseline{tag}.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[step77]   (figure skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run(SEED))
