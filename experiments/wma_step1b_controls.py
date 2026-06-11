r"""wma-step1b — POST-HOC diagnostic controls for the step1 G2 FAIL (NOT pre-registered gates).

step1 found the AITS-P-like signature on LeWM/PushT under WeakPolicy data:
$\tilde D_{T,T} = 0.936$ (actions barely decodable from REAL encoded transitions) vs
$\tilde D_{P,P} = 0.462$ (decodable from PREDICTED transitions), transfer collapse
$\tilde D_{P,T} = 3.65$, $L_{\mathrm{sym}} = 6.9$ — stable across probe widths {64, 256, 1024}.

Hypothesis H-imprint: the action-conditioned predictor $\hat z_{t+1} = \mathrm{Pred}(z_{\mathrm{win}},
\mathrm{emb}(a))$ *imprints* the fed action into its output; the action code in domain $P$
originates from the conditioning channel, not from environment-grounded dynamics (cf. the
action-conditioning externalization paradox in arXiv:2606.07687, Table 6).

Two controls:

A) **Alignment A/B** (pipeline-confound check): median one-step rel-err under
   (i) the registered alignment (last action row = chunk driving $F_t \to F_{t+1}$,
   source-verified in swm buffer._gather_clip), (ii) shifted-by-one rows, (iii) zero
   actions — paired, same transitions. If (i) is not minimal, step1's interpretation HALTS.

B) **Action-imprint permutation**: feed a permuted last-row chunk $a'$ (roll by 1 over the
   dataset), recompute $\hat z'_{t+1}$, then run the ATM audit against TARGET $a'$:
   expect $\tilde D_{T,T}(a') \approx 1$ (real transitions independent of $a'$ — sanity) and
   $\tilde D_{P,P}(a') \ll 1$ (predicted transitions encode whatever was fed — imprint).

Run: .venv/bin/python experiments/wma_step1b_controls.py   (WMA1_SMOKE=1 for smoke)
Writes: papers/figures/wma_step1b_controls{_smoke}.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from step91_lewm_audit import DTYPE, HS, ZD, encode_seq, load_lewm  # noqa: E402
from wma_step1_atm_probe import EP_SEED0, SPLIT_SEED, TEST_FRAC_EP, collect_episode  # noqa: E402

from src.audit.atm import atm_audit  # noqa: E402

SMOKE = bool(int(os.environ.get("WMA1_SMOKE", "0")))


def predict_batch(model, Zwin: torch.Tensor, Awin: torch.Tensor) -> torch.Tensor:
    """zhat_{t+1} = Pred(window, emb(action rows))[:, -1], batched."""
    N = Zwin.shape[0]
    out = torch.empty(N, ZD, dtype=DTYPE)
    with torch.no_grad():
        for lo in range(0, N, 256):
            ab = model.action_encoder(Awin[lo:lo + 256].float()).double()
            out[lo:lo + 256] = model.predict(Zwin[lo:lo + 256], ab)[:, -1]
    return out


def main() -> int:
    t0 = time.time()
    n_ep, n_step = (4, 12) if SMOKE else (132, 40)
    audit_kw = (dict(widths=(256,), main_width=256, probe_seeds=(0,), max_steps=400)
                if SMOKE else dict(widths=(256,), main_width=256, probe_seeds=(0, 1, 2)))

    print("[wma1b] loading LeWM + recollecting (seeded => identical to step1) ...", file=sys.stderr)
    model = load_lewm()
    scale01 = True                                    # step1/G3-verified choice

    # collect with one extra lookback so the shifted alignment has c_{t-3}
    Zt, Znext, Zwin, Awin, Aprev, Atgt, ep_ids = [], [], [], [], [], [], []
    for ep in range(n_ep):
        frames, chunks, _ = collect_episode(EP_SEED0 + ep, n_step)
        n = chunks.shape[0]
        if n < HS + 1:
            continue
        with torch.no_grad():
            z = encode_seq(model, frames, scale01)
        for t in range(HS, n):                        # t >= HS so rows t-3..t-1 exist
            Zt.append(z[t])
            Znext.append(z[t + 1])
            Zwin.append(z[t - HS + 1:t + 1])
            Awin.append(chunks[t - HS + 1:t + 1])     # registered: c_{t-2}, c_{t-1}, c_t
            Aprev.append(chunks[t - HS:t])            # shifted:    c_{t-3}, c_{t-2}, c_{t-1}
            Atgt.append(chunks[t])
            ep_ids.append(ep)
    Zt, Znext, Zwin = torch.stack(Zt), torch.stack(Znext), torch.stack(Zwin)
    Awin = torch.tensor(np.stack(Awin))
    Aprev = torch.tensor(np.stack(Aprev))
    Atgt = torch.tensor(np.stack(Atgt), dtype=torch.float32)
    ep_ids = np.asarray(ep_ids)
    N = Zt.shape[0]
    print(f"[wma1b] {N} transitions ({time.time() - t0:.0f}s)", file=sys.stderr)

    def med_rel(zhat):
        return float(((zhat - Znext).norm(dim=-1) / Znext.norm(dim=-1).clamp_min(1e-12)).median())

    # ---- A) alignment A/B (paired) ----
    err_reg = med_rel(predict_batch(model, Zwin, Awin))
    err_shift = med_rel(predict_batch(model, Zwin, Aprev))
    err_zero = med_rel(predict_batch(model, Zwin, torch.zeros_like(Awin)))
    spread = (max(err_reg, err_shift, err_zero) - min(err_reg, err_shift, err_zero)) / err_reg
    if spread < 0.01:
        # predictor one-step error numerically insensitive to fed actions (norm-level);
        # an imprint can still live in a low-norm decodable subspace — see control B.
        align_verdict = "INSENSITIVE (spread < 1%; source-level verification stands)"
    elif err_reg <= min(err_shift, err_zero):
        align_verdict = "REGISTERED-MINIMAL (empirically confirmed)"
    else:
        align_verdict = "MISALIGNMENT SUSPECTED (halt step1 interpretation)"
    print(f"[wma1b] A) one-step rel-err: registered={err_reg:.4f} shifted={err_shift:.4f} "
          f"zero={err_zero:.4f} spread={spread:.4f} -> {align_verdict}", file=sys.stderr)

    # ---- B) action-imprint permutation control ----
    perm = torch.roll(torch.arange(N), shifts=1)      # a'_i = a_{i-1}: env-independent fed action
    Aperm = Atgt[perm]
    Awin_perm = Awin.clone()
    Awin_perm[:, -1] = Aperm                          # replace ONLY the decoded (last) row
    Zpred_perm = predict_batch(model, Zwin, Awin_perm)

    eps_unique = np.unique(ep_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(eps_unique)
    n_test_ep = max(1, int(round(TEST_FRAC_EP * eps_unique.shape[0])))
    test_mask = np.isin(ep_ids, eps_unique[:n_test_ep].tolist())
    train_idx = torch.tensor(np.where(~test_mask)[0])
    test_idx = torch.tensor(np.where(test_mask)[0])

    print("[wma1b] B) ATM audit vs permuted fed action a' ...", file=sys.stderr)
    res = atm_audit(Zt, Znext, Zpred_perm, Aperm,
                    train_idx=train_idx, test_idx=test_idx, **audit_kw)
    dn = res.D_norm
    imprint_confirmed = bool(dn[("T", "T")] > 0.9 and dn[("P", "P")] < 0.7)
    print(f"[wma1b] B) Dnorm_TT(a')={dn[('T', 'T')]:.3f} (sanity ~1)  "
          f"Dnorm_PP(a')={dn[('P', 'P')]:.3f} (imprint if <<1)  "
          f"-> imprint_confirmed={imprint_confirmed}", file=sys.stderr)

    out = {
        "kind": "POST-HOC diagnostic controls (not pre-registered gates)",
        "alignment_ab": {"registered": err_reg, "shifted_minus_one": err_shift,
                         "zero_actions": err_zero, "spread_rel": spread,
                         "verdict": align_verdict},
        "imprint_control": {"atm_vs_fed_action": res.to_dict(),
                            "Dnorm_TT_sanity": dn[("T", "T")],
                            "Dnorm_PP_imprint": dn[("P", "P")],
                            "imprint_confirmed": imprint_confirmed,
                            "construction": "last action row replaced by roll(1)-permuted "
                                            "chunk; probe target = the fed (permuted) chunk"},
        "n_transitions": int(N),
        "wall_seconds": round(time.time() - t0, 1),
        "smoke": SMOKE,
    }
    tag = "_smoke" if SMOKE else ""
    (ROOT / "papers/figures" / f"wma_step1b_controls{tag}.json").write_text(
        json.dumps(out, indent=2))
    print(f"[wma1b] wrote wma_step1b_controls{tag}.json  wall={out['wall_seconds']}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
