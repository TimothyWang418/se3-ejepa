r"""Step 37 -- active inference on a generic $K$-target search (``experiments/step37_active_inference_search.py``).

Step 37 de-constructs Steps 25/34's *mirror-pair* POMDP into a **generic $K\!\ge\!3$ identification search**
with a single **off-path** cue, and lifts the belief from one bit to a **categorical** $p\in\Delta^{K-1}$
driven by the exact $K$-ary channel mutual information. This test certifies the structural invariants the
experiment relies on -- all fast, only the last builds an (untrained) model:

1. **$K$-ary channel** $\epsilon(d)=\epsilon_\star-(\epsilon_\star-\epsilon_0)e^{-d^2/2\delta^2}$ has the
   right limits ($\epsilon(0)=\epsilon_0$, $\epsilon(\infty)=\epsilon_\star=(K-1)/K$), is monotone increasing
   in $d$, and stays in $[\epsilon_0,\epsilon_\star]$.
2. **Exact categorical MI** $\mathrm{IG}(p;\epsilon)$ has the right limits: $\epsilon{=}0\Rightarrow
   \mathrm{IG}=\mathcal H(p)$ (noiseless collapse); $\epsilon{=}(K{-}1)/K\Rightarrow0$ (useless channel,
   identical rows); $p$ one-hot $\Rightarrow0$; $\mathrm{IG}\ge0$, monotone $\downarrow$ in $\epsilon$.
3. **Step 34's binary cue is exactly the $K=2$ case** -- ``cat_info_gain`` at $K{=}2$ equals the binary
   ``info_gain``, ``crossover_cat`` at $K{=}2$ equals the binary ``crossover``, and $\epsilon_\star(2)=\tfrac12$.
4. **MI = expected belief-entropy reduction under the loop's actual categorical soft Bayes** -- the drive
   $\mathrm{IG}(p;\epsilon)$ equals $\mathcal H(p)-\mathbb E_o[\mathcal H(p')]$ computed by ``cat_bayes_categorical``.
5. **Categorical soft Bayes accumulates and never collapses** for $\epsilon>0$; two consistent reads beat one;
   the useless channel leaves belief unchanged; $\epsilon{=}0$ collapses to a vertex.
6. **The affordance-collapse control is the right reduction** -- a proximity test of candidate $k$ is a binary
   channel on $y_k=\mathbb 1[b{=}k]$; since $o_k\perp b\mid y_k$ ($b\to y_k\to o_k$), its information about the
   full $b$ is *exactly* the binary $\mathrm{IG}(p_k,\epsilon)$ on the marginal -- verified via ``cat_bayes_binary``.
7. **The generic constellation is de-constructed** -- ``_inplane_axes`` yields $K$ coplanar unit axes that are
   well-separated (no duplicate), **non-antipodal (no mirror)**, and balanced (small centroid).
8. **The categorical-MI field is exactly $\mathrm{SE}(3)$-invariant at init** for the equivariant VN encoder
   (it is a function of the invariant latent cue-distance only) -- an architectural guarantee. The MLP
   control breaks it once *trained* (the experiment's panel [C] verifies that empirically).

Run:
    .venv/bin/python tests/test_step37_active_inference_search.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from step13_se3_latent_jepa import build_eq_jepa, rand_so3  # noqa: E402
from step18_se3_closed_loop import EVAL_DTYPE  # noqa: E402
from step34_active_inference_noisy import crossover, info_gain  # noqa: E402  (binary K=2 reference)
from step37_active_inference_search import (  # noqa: E402
    _inplane_axes,
    cat_bayes_binary,
    cat_bayes_categorical,
    cat_entropy,
    cat_info_gain,
    crossover_cat,
    eps_useless,
    info_value_invariance_cat,
    make_search_tasks,
)

torch.set_default_dtype(torch.float32)


def main() -> None:
    torch.manual_seed(0)
    print("Step 37 -- active inference on a generic K-target search: structural invariants\n")

    # ---- 1. K-ary channel eps(d): limits + monotonicity + range ----------------------------------
    eps0, delta = 0.15, 1.0
    for K in (3, 4, 5):
        es_star = eps_useless(K)
        assert abs(es_star - (K - 1.0) / K) < 1e-12, f"eps_useless({K}) should be {(K-1)/K}"
        e_at0 = crossover_cat(0.0, eps0, delta, K)
        e_far = crossover_cat(50.0, eps0, delta, K)
        ds = torch.linspace(0.0, 8.0, 64)
        es = crossover_cat(ds, eps0, delta, K)                       # tensor path
        assert abs(e_at0 - eps0) < 1e-6, f"K={K}: crossover_cat(0) should be eps0, got {e_at0:.4f}"
        assert abs(e_far - es_star) < 1e-3, f"K={K}: crossover_cat(inf) should be {es_star:.3f}, got {e_far:.4f}"
        assert torch.all(es[1:] - es[:-1] >= -1e-7), f"K={K}: crossover_cat must be monotone increasing in d"
        assert float(es.min()) >= eps0 - 1e-6 and float(es.max()) <= es_star + 1e-6, "out of [eps0, eps_star]"
    print("  [1] K-ary channel eps(d): eps(0)=eps0, eps(inf)=(K-1)/K, monotone-increasing, in-range "
          "(K=3,4,5)  OK")

    # ---- 2. exact categorical MI IG(p;eps): limits + sign + monotonicity -------------------------
    for K in (3, 4, 5):
        es_star = eps_useless(K)
        p_unif = torch.full((K,), 1.0 / K)
        Hunif = cat_entropy(np.full(K, 1.0 / K))
        ig_noiseless = float(cat_info_gain(p_unif, torch.tensor([0.0]), K))
        ig_useless = float(cat_info_gain(p_unif, torch.tensor([es_star]), K))
        assert abs(ig_noiseless - Hunif) < 1e-5, (
            f"K={K}: IG(eps=0) should be H(p)={Hunif:.4f} (noiseless collapse), got {ig_noiseless:.4f}")
        assert ig_useless < 1e-6, f"K={K}: IG(eps=(K-1)/K) should be 0 (useless), got {ig_useless:.2e}"
        # a generic (non-uniform) prior + a one-hot (degenerate) prior
        rng = np.random.default_rng(K)
        p_gen = rng.dirichlet(np.ones(K))
        p_gen_t = torch.as_tensor(p_gen)
        Hgen = cat_entropy(p_gen)
        assert abs(float(cat_info_gain(p_gen_t, torch.tensor([0.0]), K)) - Hgen) < 1e-5, "IG(eps=0)=H(p) generic"
        p_hot = torch.zeros(K); p_hot[0] = 1.0
        assert float(cat_info_gain(p_hot, torch.tensor([0.2]), K)) < 1e-6, f"K={K}: IG at one-hot p should be 0"
        eg = torch.linspace(0.0, es_star, 64)
        igs = cat_info_gain(p_unif, eg, K)
        assert torch.all(igs >= -1e-7), f"K={K}: IG must be non-negative"
        assert torch.all(igs[1:] - igs[:-1] <= 1e-6), f"K={K}: IG must be monotone decreasing in eps"
    print("  [2] IG(p;eps): IG(eps=0)=H(p), IG(eps=(K-1)/K)=0, IG(one-hot)=0, >=0 & decreasing "
          "(K=3,4,5)  OK")

    # ---- 3. Step 34's binary cue is EXACTLY the K=2 case -----------------------------------------
    assert abs(eps_useless(2) - 0.5) < 1e-12, "eps_useless(2) should be 1/2 (the binary useless floor)"
    worst_cx = worst_ig = 0.0
    eg = torch.linspace(0.0, 0.5, 48)
    ds = torch.linspace(0.0, 6.0, 48)
    for p in (0.5, 0.3, 0.72):
        ig_bin = info_gain(p, eg)                                    # Step 34 binary MI
        ig_cat = cat_info_gain(torch.tensor([p, 1.0 - p]), eg, 2)    # K=2 categorical MI
        worst_ig = max(worst_ig, float((ig_bin - ig_cat).abs().max()))
    worst_cx = float((crossover(ds, eps0=eps0, delta=delta) - crossover_cat(ds, eps0, delta, 2)).abs().max())
    assert worst_ig < 1e-5, f"cat_info_gain(K=2) must equal binary info_gain, worst {worst_ig:.2e}"
    assert worst_cx < 1e-6, f"crossover_cat(K=2) must equal binary crossover, worst {worst_cx:.2e}"
    print(f"  [3] K=2 specialisation: cat_info_gain==binary info_gain ({worst_ig:.1e}), "
          f"crossover_cat==crossover ({worst_cx:.1e}), eps*(2)=1/2  OK")

    # ---- 4. IG == expected belief-entropy reduction under the loop's actual categorical soft Bayes -
    worst = 0.0
    for K in (3, 4, 5):
        rng = np.random.default_rng(100 + K)
        for _ in range(6):
            p = rng.dirichlet(np.ones(K))
            for eps in (0.05, 0.15, 0.3, eps_useless(K) - 1e-3):
                # marginal P(o=j) = sum_i p_i P(o=j|b=i); posterior via the loop's cat_bayes_categorical
                exp_H = 0.0
                for o in range(K):
                    like = np.full(K, eps / (K - 1)); like[o] = 1.0 - eps
                    Po = float((p * like).sum())
                    exp_H += Po * cat_entropy(cat_bayes_categorical(p, o, eps, K))
                exp_red = cat_entropy(p) - exp_H
                ig = float(cat_info_gain(torch.as_tensor(p), torch.tensor([eps]), K))
                worst = max(worst, abs(ig - exp_red))
    assert worst < 1e-5, f"IG must equal E_o[H(p)-H(p')] under cat_bayes_categorical, worst {worst:.2e}"
    print(f"  [4] IG == H(p)-E_o[H(p')] via the actual cat_bayes_categorical: worst |mismatch| = "
          f"{worst:.2e} (K=3,4,5)  OK")

    # ---- 5. categorical soft Bayes accumulates and never collapses (eps>0) ------------------------
    K, eps = 4, 0.2
    p0 = np.full(K, 1.0 / K)
    p1 = cat_bayes_categorical(p0, 2, eps, K)
    p2 = cat_bayes_categorical(p1, 2, eps, K)
    assert p1[2] > 1.0 / K and p1[2] < 1.0, f"one read of '2' should raise p[2] into (1/K,1), got {p1[2]:.4f}"
    assert p1[2] < p2[2] < 1.0, f"two consistent reads should beat one and not collapse, got {p2[2]:.4f}"
    p_inert = cat_bayes_categorical(p0, 1, eps_useless(K), K)
    assert np.max(np.abs(p_inert - p0)) < 1e-9, "useless channel eps=(K-1)/K must leave belief unchanged"
    p_collapse = cat_bayes_categorical(p0, 3, 0.0, K)
    assert p_collapse[3] > 1.0 - 1e-9, "eps=0 (noiseless) must collapse belief to the read vertex"
    print(f"  [5] categorical soft Bayes: 1 read 1/K->{p1[2]:.3f}, 2 reads ->{p2[2]:.3f} (no collapse); "
          f"useless inert; eps=0 collapses  OK")

    # ---- 6. affordance-collapse control == the right binary reduction (b -> y_k -> o_k) ----------
    # A proximity test of candidate k carries info about the FULL b equal to the binary IG on the marginal p_k,
    # because o_k _|_ b | y_k. Verify via the loop's actual cat_bayes_binary update.
    worst_red = 0.0
    for K in (3, 4, 5):
        rng = np.random.default_rng(200 + K)
        for _ in range(6):
            p = rng.dirichlet(np.ones(K))
            for eps in (0.05, 0.15, 0.3):
                for k in range(K):
                    pk = float(p[k])
                    Pplus = pk * (1.0 - eps) + (1.0 - pk) * eps           # P(o=+) for the k-vs-rest test
                    exp_H = (Pplus * cat_entropy(cat_bayes_binary(p, k, 1, eps))
                             + (1.0 - Pplus) * cat_entropy(cat_bayes_binary(p, k, 0, eps)))
                    exp_red = cat_entropy(p) - exp_H                      # I(b; o_k)
                    ig_marg = float(info_gain(pk, torch.tensor([eps])))   # binary IG on the marginal p_k
                    worst_red = max(worst_red, abs(exp_red - ig_marg))
    assert worst_red < 1e-5, f"proximity-test info must reduce to binary IG(p_k,eps), worst {worst_red:.2e}"
    print(f"  [6] affordance-collapse reduction: H(p)-E_o[H(p')] via cat_bayes_binary == binary IG(p_k,eps), "
          f"worst {worst_red:.2e} (K=3,4,5)  OK")

    # ---- 7. the generic constellation is de-constructed: no mirror, separated, balanced, coplanar -
    cos_anti = math.cos(math.radians(30.0))    # anti_margin_deg default
    cos_sep = math.cos(math.radians(38.0))     # min_sep_deg default
    for K in (3, 4, 5):
        rng = np.random.default_rng(300 + K)
        for _ in range(40):
            Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            e1, e2, nrm = Q[:, 0], Q[:, 1], Q[:, 2]
            axes = np.stack(_inplane_axes(rng, K, e1, e2), axis=0)       # (K,3) unit axes
            norms = np.linalg.norm(axes, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6), f"K={K}: axes must be unit vectors"
            assert np.max(np.abs(axes @ nrm)) < 1e-6, f"K={K}: axes must be coplanar (orthogonal to normal)"
            G = axes @ axes.T
            iu = np.triu_indices(K, k=1)
            cos_pairs = G[iu]
            assert np.all(cos_pairs > -cos_anti - 1e-6), f"K={K}: a near-antipodal (MIRROR) pair slipped in"
            assert np.all(cos_pairs < cos_sep + 1e-6), f"K={K}: a near-duplicate (collinear) pair slipped in"
            assert float(np.linalg.norm(axes.mean(axis=0))) < 0.25 + 1e-6, f"K={K}: constellation not balanced"
    print("  [7] generic constellation: coplanar, no mirror (|angle-180|>30deg), separated (>38deg), "
          "balanced (K=3,4,5)  OK")

    # ---- 8. categorical-MI field is exactly SE(3)-invariant at init for the equivariant VN encoder ---
    # The equivariant encoder makes the IG field a function of the SE(3)-INVARIANT latent cue-distance, so
    # it is exactly invariant at init AND post-train -- an architectural guarantee, not a learned one. (The
    # MLP control *breaks* this once trained -- an empirical fact the experiment verifies in panel [C], e.g.
    # IG-inv ~ 3e-1 post-train; at random init it is not yet broken, so we do not assert it here.)
    K = 3
    vn = build_eq_jepa().to(EVAL_DTYPE)
    task = make_search_tasks(1, K, seed=0)[0]
    X0, Xc = task[0], task[2]
    gen = torch.Generator().manual_seed(1)
    worst_vn = 0.0
    for _ in range(5):
        R = rand_so3(gen).to(EVAL_DTYPE)
        t = ((torch.rand(3, generator=gen) * 2 - 1) * 0.8).to(EVAL_DTYPE)
        g = torch.Generator().manual_seed(7)
        worst_vn = max(worst_vn, info_value_invariance_cat(vn, X0, Xc, R, t, eps0=0.15, K=K, H=8, n_samples=48, gen=g))
    assert worst_vn < 1e-4, f"VN categorical-MI field not SE(3)-invariant at init: {worst_vn:.2e}"
    print(f"  [8] categorical-MI field SE(3)-invariance at init (equivariant VN): "
          f"max|IG(x)-IG(Rx+t)| = {worst_vn:.2e}  OK (MLP-break verified post-train in the experiment)")

    print("\nPASS: the K-ary noisy channel, the exact categorical mutual-information drive (with Step 34's")
    print("binary cue recovered as its K=2 case), its agreement with the categorical soft-Bayes update, the")
    print("affordance-collapse control's reduction to binary IG, the de-constructed (no-mirror, balanced)")
    print("constellation, and the SE(3)-invariance of the information field are all structurally sound.")


def test_step37_active_inference_search() -> None:
    """pytest: K-ary channel, categorical-MI limits, K=2->binary, MI=soft-Bayes-drop, affordance reduction, no-mirror geometry, SE(3)-invariance."""
    main()


if __name__ == "__main__":
    main()
