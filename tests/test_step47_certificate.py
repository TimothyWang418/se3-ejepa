r"""Closure / certificate unit test (Step 47, paper2 P0).

Theorem A (exact certificate) says: for an exact-equivariant world model with orthogonal $\rho$, the
whole-pipeline prediction error is **invariant** under every composition word $w=g_{i_1}\cdots g_{i_m}$
drawn from the generating set $S$ — *architecturally*, independent of training. We verify this at **init**
(no optimisation needed, since equivariance is a property of the architecture, not the weights):

1. **Composition closure ⇒ certificate.** Build the equivariant model, draw composition words of length
   $m=1,\dots,4$ from $S=\{\text{global }\mathrm{SE}(3)\text{ rot/trans},\ \text{object permutation}\}$, and
   assert the relMSE at $w\cdot x$ equals the relMSE at $x$ to the float floor. This is the empirical face
   of "verify on the generators ⇒ certified on the generated set" (the multi-element lift of [B]).
2. **Residual stays float-floor under words.** The post-word $\mathrm{SE}(3)$+permutation residual of the
   whole pipeline is $\le10^{-4}$ for every word length.
3. **Discrimination.** The non-equivariant MLP control's relMSE is **not** word-invariant — so the test
   actually distinguishes the property it claims.

Run:
    .venv/bin/python tests/test_step47_certificate.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step24_object_interaction import make_interacting_transitions, rand_so3  # noqa: E402
from step43_encoder_ladder import build_model, indist_relmse, se3_resid  # noqa: E402
from step47_certificate import apply_word  # noqa: E402


@torch.no_grad()
def _certificate_spread(name, n_words=12, max_len=4, seed=0):
    r"""Return (max relMSE deviation from base over words, max SE(3) residual over words) for a model
    built at init (equivariance is architectural — no training required)."""
    torch.manual_seed(seed)
    model = build_model(name).eval()
    S, A, S2 = make_interacting_transitions(48, seed=123)
    base = indist_relmse(name, model, S, A, S2)
    wg = torch.Generator().manual_seed(900 + seed)
    R_chk = rand_so3(torch.Generator().manual_seed(3))
    t_chk = torch.randn(3, generator=torch.Generator().manual_seed(4))
    dev, res = 0.0, 0.0
    for _ in range(n_words):
        m = int(torch.randint(1, max_len + 1, (1,), generator=wg).item())
        Sw, Aw, S2w = apply_word(S, A, S2, m, wg)
        dev = max(dev, abs(indist_relmse(name, model, Sw, Aw, S2w) - base) / max(base, 1e-12))
        res = max(res, se3_resid(name, model, Sw, Aw, R_chk, t_chk))
    return base, dev, res


def test_certificate_is_word_invariant_for_equivariant_model() -> None:
    base, dev, res = _certificate_spread("E0-base")
    print(f"E0-base (equivariant): base relMSE {base:.4e} | max word-deviation {dev:.2e} | "
          f"max SE(3) residual {res:.2e}")
    assert dev < 1e-3, f"certificate broken: relMSE varies by {dev:.2e} across composition words"
    assert res < 1e-4, f"post-word equivariance residual too large: {res:.2e}"


def test_non_equivariant_control_is_not_word_invariant() -> None:
    base, dev, res = _certificate_spread("MLP-MP")
    print(f"MLP-MP (non-equivariant): base relMSE {base:.4e} | max word-deviation {dev:.2e}")
    assert dev > 1e-2, (
        f"the non-equivariant control should NOT be word-invariant, but deviation was only {dev:.2e} "
        "— the test would not discriminate the certificate property"
    )


def main() -> None:
    print("Step 47 — composition-word certificate (Theorem A), verified at init\n")
    test_certificate_is_word_invariant_for_equivariant_model()
    test_non_equivariant_control_is_not_word_invariant()
    print("\nPASS: the equivariant model's error is invariant across composition words to the float floor;")
    print("      the non-equivariant control is not — Theorem A's certificate holds architecturally.")


if __name__ == "__main__":
    main()
