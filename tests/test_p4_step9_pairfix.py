r"""P4 #9 — the in-process == reloaded equality test for the EMA-target pairing fix.

The incident (paper3_record.md, 2026-06-11): e2cnn R2Conv caches its expanded filter at .eval()
and never refreshes it on parameter mutation (and train(False) re-expands only `if self.training`)
⇒ under the old train_jepa, the EMA target's FUNCTION was frozen at its init filters while its
state_dict carried EMA weights — in-process and reloaded targets were different functions, and all
pixel-eq audit numbers were quarantined.

Three gates:

- **E-I (fix closes the loop):** with ``refresh_target_cache=True``, the reloaded target equals
  the in-process target as a FUNCTION (relative output gap < 1e-3) and the target-pair δ̂ agrees
  in-process vs reloaded (rel < 1e-2).
- **E-II (the test has teeth):** with ``refresh_target_cache=False`` (paper2-era behaviour), the
  same round trip REPRODUCES the bug (relative output gap > 0.05) — if this ever starts passing,
  e2cnn changed and the comment in train_jepa needs revisiting.
- **E-III (plain control):** the plain base (no e2cnn) round-trips exactly under either flag.

Run: .venv/bin/python tests/test_p4_step9_pairfix.py   (~3 min, MPS)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from experiments.p4_step1_pipeline import (  # noqa: E402
    CHUNK, RES, build_eq, build_plain, circ_mask, collect_weakpolicy, to_transitions,
)
from src.audit.gap_mode import one_step_bias  # noqa: E402
from src.training.jepa import train_jepa  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Pair(torch.nn.Module):
    def __init__(self, enc, pred):
        super().__init__()
        self.encoder = enc
        self.pred = pred

    def encode(self, px):
        return self.encoder(px)

    @property
    def predictor(self):
        return self.pred


def heldout():
    ho = collect_weakpolicy(8, seed=1)
    f = torch.from_numpy(ho["frames"]).float().div_(255.0).permute(0, 1, 4, 2, 3)
    f = circ_mask(f.reshape(-1, 3, RES, RES)).reshape(f.shape)
    fb = f[:, ::CHUNK].double()
    a = torch.from_numpy(ho["actions"])
    n_ch = a.shape[1] // CHUNK
    ach = a[:, : n_ch * CHUNK].reshape(a.shape[0], n_ch, CHUNK * 2).double()
    return fb, ach


def round_trip(builder, refresh: bool, fb, ach, obs, act, nxt):
    torch.manual_seed(0)
    m = builder()
    hist, tgt = train_jepa(m, obs, act, nxt, epochs=8, batch_size=64, device=DEVICE, seed=0,
                           verbose=False, return_target_encoder=True,
                           refresh_target_cache=refresh)
    m.eval().cpu()
    tgt.eval().cpu()  # no-op re-expansion guard path (already eval)

    d_in = float(one_step_bias(Pair(tgt, m.predictor), fb, ach).mean())

    blob = {"model": m.state_dict(), "target_encoder": tgt.state_dict()}
    m2 = builder()
    m2.load_state_dict(blob["model"])
    t2 = builder()
    t2.encoder.load_state_dict(blob["target_encoder"])
    m2.eval()
    t2.eval()  # fresh module was train-mode -> this re-expands from the loaded weights

    d_re = float(one_step_bias(Pair(t2.encoder, m2.predictor), fb, ach).mean())
    with torch.no_grad():
        x = fb.reshape(-1, 3, RES, RES)[:48].float()
        z1, z2 = tgt(x), t2.encoder(x)
        rel_gap = float((z1 - z2).norm() / z1.norm().clamp_min(1e-9))
    return {"delta_inproc": d_in, "delta_reload": d_re, "fn_rel_gap": rel_gap,
            "pred_loss": hist["pred_loss"][-1]}


if __name__ == "__main__":
    tr = collect_weakpolicy(20, seed=0)
    obs, act, nxt = to_transitions(tr, 20)
    fb, ach = heldout()

    fixed = round_trip(build_eq, True, fb, ach, obs, act, nxt)
    print(f"[E-I] eq + refresh=True : {fixed}")
    assert fixed["fn_rel_gap"] < 1e-3, f"E-I FAIL: function gap {fixed['fn_rel_gap']:.2e}"
    assert abs(fixed["delta_reload"] - fixed["delta_inproc"]) / max(fixed["delta_inproc"], 1e-9) < 1e-2, \
        f"E-I FAIL: delta mismatch {fixed}"
    print("[E-I] PASS — in-process == reloaded under the fix")

    legacy = round_trip(build_eq, False, fb, ach, obs, act, nxt)
    print(f"[E-II] eq + refresh=False: {legacy}")
    assert legacy["fn_rel_gap"] > 0.05, \
        f"E-II UNEXPECTED PASS: legacy path no longer reproduces the bug ({legacy['fn_rel_gap']:.2e}) — e2cnn changed?"
    print("[E-II] PASS — legacy path reproduces the bug (test has teeth)")

    plain = round_trip(lambda: build_plain(16), True, fb, ach, obs, act, nxt)
    print(f"[E-III] plain control   : {plain}")
    assert plain["fn_rel_gap"] < 1e-5, f"E-III FAIL: plain round trip gap {plain['fn_rel_gap']:.2e}"
    print("[E-III] PASS — plain base round-trips exactly")
    print("ALL P4 #9 equality gates PASS")
