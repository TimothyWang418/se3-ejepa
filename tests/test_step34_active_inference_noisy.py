r"""Step 34 -- active inference under a **noisy** cue (``experiments/step34_active_inference_noisy.py``).

Step 34 de-constructs Step 25's noiseless one-bit reveal into a *noisy* binary channel whose epistemic
value is the **exact mutual information** of the sensor. This test certifies the structural invariants
the experiment relies on -- all fast, no training:

1. **Noisy channel** $\epsilon(d)=\tfrac12-(\tfrac12-\epsilon_0)e^{-d^2/2\delta^2}$ has the right limits
   ($\epsilon(0)=\epsilon_0$, $\epsilon(\infty)=\tfrac12$) and is monotone increasing in $d$.
2. **Exact mutual information** $\mathrm{IG}(p;\epsilon)$ has the right limits: $\epsilon{=}0\Rightarrow
   \mathrm{IG}=\mathcal H(p)$ (noiseless collapse, *recovers Step 25*); $\epsilon{=}\tfrac12\Rightarrow0$
   (useless channel); $p\in\{0,1\}\Rightarrow0$ (nothing to learn); $\mathrm{IG}\ge0$, monotone $\downarrow$
   in $\epsilon$.
3. **MI = expected belief-entropy reduction under soft Bayes** -- the planner's drive
   $\mathrm{IG}(p;\epsilon)$ equals $\mathcal H(p)-\mathbb E_o[\mathcal H(p')]$ computed by the loop's
   actual ``bayes_update``. This ties the epistemic *objective* to the belief *mechanism* exactly.
4. **Soft Bayes accumulates and never collapses** for $\epsilon>0$; two consistent bits beat one.
5. **The mutual-information field is exactly $\mathrm{SE}(3)$-invariant at init** for the equivariant VN
   encoder (it is a function of the invariant latent distance only).

Run:
    .venv/bin/python tests/test_step34_active_inference_noisy.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from step13_se3_latent_jepa import build_eq_jepa, rand_so3  # noqa: E402
from step18_se3_closed_loop import EVAL_DTYPE, _apply  # noqa: E402
from step25_active_inference_task import belief_entropy, make_cue_tasks  # noqa: E402
from step34_active_inference_noisy import (  # noqa: E402
    bayes_update,
    crossover,
    info_gain,
    info_value_invariance,
)

torch.set_default_dtype(torch.float32)


def main() -> None:
    torch.manual_seed(0)
    print("Step 34 -- active inference under a noisy cue: structural invariants\n")

    # ---- 1. noisy channel epsilon(d): limits + monotonicity --------------------------------------
    eps0, delta = 0.15, 1.0
    e_at0 = float(crossover(torch.tensor(0.0), eps0=eps0, delta=delta))
    e_far = float(crossover(torch.tensor(50.0), eps0=eps0, delta=delta))
    ds = torch.linspace(0.0, 8.0, 64)
    es = crossover(ds, eps0=eps0, delta=delta)
    assert abs(e_at0 - eps0) < 1e-6, f"crossover(0) should be eps0={eps0}, got {e_at0:.4f}"
    assert abs(e_far - 0.5) < 1e-3, f"crossover(inf) should be 1/2, got {e_far:.4f}"
    assert torch.all(es[1:] - es[:-1] >= -1e-7), "crossover must be monotone increasing in d"
    assert float(es.min()) >= eps0 - 1e-6 and float(es.max()) <= 0.5 + 1e-6, "crossover out of [eps0,1/2]"
    print(f"  [1] channel eps(d): eps(0)={e_at0:.3f}=eps0, eps(inf)={e_far:.4f}~1/2, "
          f"monotone-increasing  OK")

    # ---- 2. exact mutual information IG(p;eps): limits + sign + monotonicity ----------------------
    for p in (0.5, 0.3, 0.7):
        Hp = belief_entropy(p)
        ig_noiseless = float(info_gain(p, torch.tensor(0.0)))
        ig_useless = float(info_gain(p, torch.tensor(0.5)))
        assert abs(ig_noiseless - Hp) < 1e-5, (
            f"IG(p={p}, eps=0) should be H(p)={Hp:.4f} (Step 25 limit), got {ig_noiseless:.4f}")
        assert ig_useless < 1e-6, f"IG(p={p}, eps=1/2) should be 0 (useless), got {ig_useless:.2e}"
    for p_deg in (0.0, 1.0):
        assert float(info_gain(p_deg, torch.tensor(0.2))) < 1e-6, "IG at p in {0,1} should be 0"
    eg = torch.linspace(0.0, 0.5, 64)
    igs = info_gain(0.5, eg)
    assert torch.all(igs >= -1e-7), "IG must be non-negative"
    assert torch.all(igs[1:] - igs[:-1] <= 1e-7), "IG must be monotone decreasing in eps"
    print(f"  [2] IG(p;eps): IG(eps=0)=H(p) (Step 25 limit), IG(eps=1/2)=0, IG(p in {{0,1}})=0, "
          f">=0 & decreasing  OK")

    # ---- 3. IG == expected belief-entropy reduction under the loop's actual soft Bayes -----------
    worst = 0.0
    for p in (0.5, 0.35, 0.62, 0.8):
        for eps in (0.05, 0.15, 0.3, 0.45):
            q = p * (1.0 - eps) + (1.0 - p) * eps             # P(o=+)
            p_plus = bayes_update(p, +1, eps)
            p_minus = bayes_update(p, -1, eps)
            exp_red = belief_entropy(p) - (q * belief_entropy(p_plus) + (1.0 - q) * belief_entropy(p_minus))
            ig = float(info_gain(p, torch.tensor(eps)))
            worst = max(worst, abs(ig - exp_red))
    assert worst < 1e-5, f"IG must equal E_o[H(p)-H(p')] under soft Bayes, worst mismatch {worst:.2e}"
    print(f"  [3] IG == H(p)-E_o[H(p')] via the actual bayes_update: worst |mismatch| = {worst:.2e}  OK")

    # ---- 4. soft Bayes accumulates and never collapses (eps>0); two consistent bits beat one -----
    p0, eps = 0.5, 0.2
    p1 = bayes_update(p0, +1, eps)
    p2 = bayes_update(p1, +1, eps)
    assert 0.5 < p1 < 1.0, f"one '+' bit should raise belief into (0.5,1), got {p1:.4f}"
    assert p1 < p2 < 1.0, f"two consistent bits should beat one and still not collapse, got {p2:.4f}"
    assert abs(bayes_update(p0, +1, 0.5) - p0) < 1e-6, "eps=1/2 (useless) must leave belief unchanged"
    assert bayes_update(p0, +1, 0.0) > 1.0 - 1e-6, "eps=0 (noiseless) collapses belief to certainty"
    print(f"  [4] soft Bayes: 1 bit 0.5->{p1:.3f}, 2 bits ->{p2:.3f} (no collapse); "
          f"eps=1/2 inert; eps=0 collapses  OK")

    # ---- 5. mutual-information field is exactly SE(3)-invariant at init (equivariant VN encoder) --
    vn = build_eq_jepa().to(EVAL_DTYPE)
    task = make_cue_tasks(1, seed=0)[0]
    X0, Xc = task[0], task[1]
    gen = torch.Generator().manual_seed(1)
    worst_inv = 0.0
    for _ in range(5):
        R = rand_so3(gen).to(EVAL_DTYPE)
        t = ((torch.rand(3, generator=gen) * 2 - 1) * 0.8).to(EVAL_DTYPE)
        g = torch.Generator().manual_seed(7)
        worst_inv = max(worst_inv, info_value_invariance(
            vn, X0, Xc, R, t, eps0=0.15, p=0.5, H=8, n_samples=48, gen=g))
    assert worst_inv < 1e-4, f"VN mutual-information field not SE(3)-invariant at init: {worst_inv:.2e}"
    print(f"  [5] VN IG-field SE(3)-invariance at init: max|IG(x)-IG(Rx+t)| = {worst_inv:.2e}  OK")

    print("\nPASS: the noisy channel, the exact mutual-information drive, its agreement with the soft-Bayes")
    print("belief update, and the SE(3)-invariance of the information field are all structurally sound.")


def test_step34_active_inference_noisy() -> None:
    """pytest: noisy channel, exact-MI limits, MI=soft-Bayes-entropy-drop, accumulation, SE(3)-invariance."""
    main()


if __name__ == "__main__":
    main()
