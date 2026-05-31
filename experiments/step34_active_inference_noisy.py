r"""Step 34: active inference **de-constructed** -- invariant curiosity under a *noisy* cue.

Where this sits, and the precise gap it closes
-----------------------------------------------
Step 25 proved that an Expected-Free-Energy planner in the exactly-$\mathrm{SE}(3)$-equivariant latent
converts an $\mathrm{SE}(3)$-invariant epistemic drive into a real **task win** a reward-only planner
*provably* cannot match (the ambiguous-goal cue-foraging POMDP). But Step 25 named its own ceiling
verbatim (its "Honest scope"): *"This is a constructed POMDP ... and the cue reveal is **noiseless** ...
The sophistication of the belief update (here a one-bit Bayesian collapse) is deliberately minimal so
the geometry is the only moving part."* A skeptic's fair complaint: of course information-seeking pays
when one visit to the cue hands you the answer **for free and with certainty**. Step 34 removes exactly
that crutch -- it de-constructs the noiseless reveal -- while keeping the geometry as the controlled
variable, and asks whether the invariant curiosity survives.

The single surgical change: a noisy sensor, and a *principled* epistemic value
-------------------------------------------------------------------------------
Everything (the exactly-equivariant teacher and latent, the CEM scaffolding, the belief-weighted
pragmatic cost, the reward-only / EFE / oracle agents, the invariance machinery) is inherited verbatim
from Step 25. Two things change, both at the **cue sensor**:

1. **The cue is a noisy binary channel that never gives certainty.** Sensing at executed latent
   $\hat z$ (distance $d=\lVert\hat z-z_c\rVert$ to the cue latent) returns a bit $o\in\{+,-\}$ about the
   hidden goal index $b$ with a distance-dependent crossover (flip) probability
   $$\epsilon(d)=\tfrac12-\big(\tfrac12-\epsilon_0\big)\exp\!\Big(\!-\tfrac{d^2}{2\delta^2}\Big),\qquad
     \epsilon(\infty)=\tfrac12\ (\text{pure noise}),\quad \epsilon(0)=\epsilon_0\in(0,\tfrac12).$$
   The floor $\epsilon_0>0$ is the de-construction: there is **no noiseless reveal anywhere**. The belief
   updates by a *soft* Bayes rule and never collapses in one shot, so the agent must **accumulate**
   noisy evidence over repeated senses:
   $$p'=\frac{p\,L_+(o)}{p\,L_+(o)+(1-p)\,L_-(o)},\quad L_+(o{=}{+})=1-\epsilon,\ L_+(o{=}{-})=\epsilon\ (\text{$L_-$ swaps}).$$

2. **The epistemic value is the exact mutual information of that channel** -- not a hand-set Gaussian
   proxy. For belief $p$ and crossover $\epsilon$, with predicted-observation marginal
   $q=P(o{=}{+})=p(1-\epsilon)+(1-p)\epsilon$, the information gain is the expected belief-entropy drop
   $$\mathrm{IG}(d;p)=\mathcal H(p)-\big[q\,\mathcal H(p'_+)+(1-q)\,\mathcal H(p'_-)\big]
     = I(b;o\mid d),\qquad \mathcal H(x)=-x\ln x-(1-x)\ln(1-x).$$
   $\mathrm{IG}$ is exact, analytic, monotone decreasing in $d$, and *self-extinguishing on two fronts at
   once*: $\mathrm{IG}\!\to\!0$ as $p\!\to\!\{0,1\}$ ($\mathcal H\!\to\!0$, nothing left to learn) **and**
   as $\epsilon\!\to\!\tfrac12$ ($d\!\to\!\infty$, the channel is information-free). It strictly upgrades
   Step 25's hand-built $\eta\,\mathcal H(p)$: at $\epsilon_0\!=\!0$ and $d\!=\!0$ one has
   $\mathrm{IG}=\mathcal H(p)$ -- so **Step 25 is exactly the $\epsilon_0\!\to\!0$ corner of Step 34.**
   The planner's per-trajectory epistemic surrogate is the best (closest) approach,
   $\mathrm{IG}_{\rm traj}=\max_h \mathrm{IG}(d_h;p)=\mathrm{IG}(\min_h d_h;p)$ (monotone in $d$); the
   *closed loop* then does the real multi-sense accumulation step by step.

The geometric theorem still holds (and that is the point)
---------------------------------------------------------
$\mathrm{IG}$ depends on $\hat z$ **only** through the latent distance $d=\lVert\hat z-z_c\rVert$, and on
the goal index through the label-probability $p$ -- both $\mathrm{SE}(3)$-invariant. Under a global
$x\mapsto Rx+t$ the equivariant encoder sends every latent by the same orthogonal $\rho(R)$, so $d$ is
unchanged and therefore the entire mutual-information field, the EFE plan, and the resulting task
outcome are **exactly $\mathrm{SE}(3)$-invariant / -equivariant** to the float floor -- now for a
genuinely noisy, information-theoretically-optimal epistemic drive, not a constructed proxy. The MLP
control breaks every line of it.

Four panels (most to least decisive)
-------------------------------------
  [A]  The task win **under noise**. Mean true-goal error over many POMDPs for reward-only ($\beta{=}0$),
       EFE ($\beta{>}0$), and an oracle, in the *same* equivariant latent, at a clearly-noisy headline
       $\epsilon_0$. EFE $\ll$ reward-only (pinned at the hedge floor) but now sits an honest *noise gap*
       above the oracle -- and EFE senses the cue **several** times (graded accumulation), where Step 25
       sensed once. Paired bootstrap CIs.
  [B]  The de-construction made quantitative -- a **noise sweep** $\epsilon_0:0\!\to\!\tfrac12$. At
       $\epsilon_0{=}0$ EFE error $\to$ oracle (we recover Step 25); as $\epsilon_0$ grows the error rises
       gracefully toward the hedge floor and the agent senses *more* (works harder for weaker bits);
       at $\epsilon_0\!\to\!\tfrac12$ the cue is information-free and EFE $\to$ reward-only -- the win
       **correctly vanishes**. A built-in falsifiable negative control: no free lunch from a useless cue.
  [C]  The theorem realised, **for the mutual-information drive**. Rotate the whole episode by a random
       global $(R,t)$: the IG field, the EFE plan, and the true-goal outcome are invariant/equivariant to
       the float floor (init AND post-train) for the VN agent and break for the MLP. Plus the standard
       encoder/predictor equivariance unit tests and ``tests/test_step34_*.py``.

Honest scope
------------
Still a constructed POMDP over the synthetic equivariant teacher; what changes -- and what Step 34 adds
over Step 25 -- is that the reveal is **noisy and never certain**, the belief is a *graded* accumulation,
and the epistemic drive is the **exact mutual information** of the sensor rather than a hand-tuned
salience. The claim is therefore stronger and more honest: the $\mathrm{SE}(3)$-invariant curiosity
earns its task win even when information is *expensive and unreliable*, it degrades gracefully to "no
win" exactly when the cue stops carrying information, and the whole noisy loop stays exactly
$\mathrm{SE}(3)$-equivariant. Active inference as geometric structure, carried into a noisy decision
problem -- not a benchmark-beating claim in the wild.

Run (full ~30-45 min on a laptop CPU):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step34_active_inference_noisy.py
Smoke (~2-3 min):
    STEP34_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step34_active_inference_noisy.py
"""

import json
import math
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
sys.path.insert(0, str(ROOT))   # for `src.*`
sys.path.insert(0, str(HERE))   # for the Step 13/18/25 backbone we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Reuse the *validated* Step 13/18 machinery verbatim, and the Step 25 POMDP construction + helpers.
from step13_se3_latent_jepa import (  # noqa: E402
    C_T,
    build_eq_jepa,
    build_mlp_jepa,
    collect_cloud_transitions,
    encoder_equiv_err,
    predictor_equiv_err,
    rand_so3,
    teacher_step,
)
from step18_se3_closed_loop import (  # noqa: E402
    EVAL_DTYPE,
    _apply,
    _ball_clamp,
    boot_mean_ci,
    boot_ratio_ci,
    centroid,
    kabsch_rotation,
    rotation_angle_deg,
)
from step10_pusht_closed_loop import n_params  # noqa: E402
from step25_active_inference_task import (  # noqa: E402
    _agent_arrays,
    _fmt,
    _zscore,
    belief_entropy,
    make_cue_tasks,
)
from src.training.jepa import train_jepa  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP34_SMOKE"))


# --------------------------------------------------------------------------- #
# the noisy cue channel: crossover epsilon(d), soft Bayes, and exact mutual information
# --------------------------------------------------------------------------- #
def crossover(d: torch.Tensor | float, *, eps0: float, delta: float) -> torch.Tensor | float:
    r"""Distance-dependent flip probability $\epsilon(d)=\tfrac12-(\tfrac12-\epsilon_0)e^{-d^2/2\delta^2}$.

    $\epsilon(\infty)=\tfrac12$ (far from the cue the bit is pure noise), $\epsilon(0)=\epsilon_0\in(0,
    \tfrac12)$ (at the cue it is reliable but, for $\epsilon_0>0$, **never perfect**). Works on a Python
    float or a tensor of latent distances. SE(3)-invariant because it is a function of $d$ alone.
    """
    g = torch.exp(-(d * d) / (2.0 * delta * delta)) if torch.is_tensor(d) else math.exp(
        -(d * d) / (2.0 * delta * delta))
    return 0.5 - (0.5 - eps0) * g


def _h_nats(x: torch.Tensor) -> torch.Tensor:
    r"""Binary entropy $\mathcal H(x)=-x\ln x-(1-x)\ln(1-x)$ (nats), *exact* at the $\{0,1\}$ ends.

    A naive ``x.log()`` is $-\infty$ at $x=0$ and the clamp trick $x\in[\delta,1-\delta]$ is dtype-fragile:
    in float32 (machine $\epsilon\approx 1.2\times10^{-7}$) a margin $\delta=10^{-12}$ rounds $1-\delta$ back
    to *exactly* $1.0$, so $(1-x)=0$ and $0\cdot\ln 0=\mathrm{NaN}$; a float32-safe $\delta=10^{-6}$ removes
    the NaN but injects $\sim1.5\times10^{-5}$ nats of error -- enough to break the exact $\mathrm{IG}(\epsilon{=}0)=\mathcal H(p)$
    limit (tolerance $10^{-5}$). Instead use $\operatorname{xlogy}(a,b)=a\ln b$ with the convention
    $\operatorname{xlogy}(0,0)=0$, which makes $\mathcal H(0)=\mathcal H(1)=0$ *exactly* with no clamp bias.
    The single clamp to $[0,1]$ only guards against tiny floating-point excursions of Bayes posteriors.
    """
    xc = x.clamp(0.0, 1.0)
    return -(torch.special.xlogy(xc, xc) + torch.special.xlogy(1.0 - xc, 1.0 - xc))


def info_gain(p: float, eps: torch.Tensor) -> torch.Tensor:
    r"""Exact mutual information $\mathrm{IG}=\mathcal H(p)-\mathbb E_o[\mathcal H(p')]=I(b;o)$ (nats).

    For belief $p=P(b{=}{+})$ and crossover tensor ``eps``, with $q=P(o{=}{+})=p(1-\epsilon)+(1-p)\epsilon$
    and soft-Bayes posteriors $p'_+=p(1-\epsilon)/q$, $p'_-=p\epsilon/(1-q)$:
    $$\mathrm{IG}=\mathcal H(p)-\big[q\,\mathcal H(p'_+)+(1-q)\,\mathcal H(p'_-)\big]\ge 0.$$
    Limits (both exact): $\epsilon\!=\!0\Rightarrow\mathrm{IG}=\mathcal H(p)$ (noiseless collapse, = Step 25);
    $\epsilon\!=\!\tfrac12\Rightarrow\mathrm{IG}=0$ (useless channel); $p\!\in\!\{0,1\}\Rightarrow0$
    (nothing left to learn). ``eps`` of any shape; returns the same shape.
    """
    Hp = belief_entropy(p)
    q = (p * (1.0 - eps) + (1.0 - p) * eps).clamp(1e-12, 1.0 - 1e-12)
    p_plus = (p * (1.0 - eps)) / q
    p_minus = (p * eps) / (1.0 - q)
    exp_post = q * _h_nats(p_plus) + (1.0 - q) * _h_nats(p_minus)
    return (Hp - exp_post).clamp_min(0.0)


def bayes_update(p: float, o: int, eps: float) -> float:
    r"""Soft Bayes posterior after observing bit ``o``$\in\{+1,-1\}$ through crossover ``eps``.

    $L_+(o{=}{+})=1-\epsilon,\ L_+(o{=}{-})=\epsilon$ (and $L_-$ swaps). Never collapses to $\{0,1\}$ for
    $\epsilon>0$, so evidence accumulates across repeated noisy senses.
    """
    if o == 1:
        lp, lm = (1.0 - eps), eps
    else:
        lp, lm = eps, (1.0 - eps)
    num = p * lp
    return num / (num + (1.0 - p) * lm + 1e-12)


# --------------------------------------------------------------------------- #
# the EFE planner: belief-weighted pragmatic - beta * (mutual-information salience)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def efe_pomdp_plan_noisy(
    model, X0: torch.Tensor, *, p: float, zgp: torch.Tensor, zgm: torch.Tensor,
    cgp: torch.Tensor, cgm: torch.Tensor, zc: torch.Tensor, delta: float, beta: float,
    eps0: float, R_noise: torch.Tensor | None = None, w_t: float = 0.5, H: int = 12,
    n_samples: int = 256, n_iters: int = 5, n_elite: int = 25, sigma0: float = 0.6,
    w_run: float = 0.3, gen: torch.Generator | None = None,
) -> torch.Tensor:
    r"""SE(3)-equivariant CEM minimising $G=\text{(belief-weighted pragmatic)}-\beta\,g_{\rm epi}(p)\,\mathrm{IG}_{\rm traj}$.

    Identical equivariant scaffolding to Step 25's :func:`efe_pomdp_plan` (iso-$\sigma$ CEM, ball clamp,
    $R$-rotated exploration noise, exact closed-form centroid channel, belief-weighted pragmatic), with
    **one** change: the epistemic channel is the exact mutual information of the noisy sensor rather than
    a hand-built Gaussian salience. For each rolled candidate we form per-step crossovers
    $\epsilon_h=\epsilon(\lVert\hat z_h-z_c\rVert)$ and per-step information $\mathrm{IG}(p;\epsilon_h)$,
    and take the trajectory value $\mathrm{IG}_{\rm traj}=\max_h\mathrm{IG}_h$ (the closest, most
    informative approach; the closed loop accumulates the rest). Each channel is z-scored separately
    across the (jointly rotated) candidate population -- and every per-candidate value is a function of
    SE(3)-invariant latent distances only, so the whole EFE stays exactly SE(3)-invariant.

    Shapes: ``X0:(1,P,3); zg*,zc:(1,D); cg*:(3,)|(1,3)``. Returns the plan ``(H,3)`` in the unit ball.
    """
    dtype = model.encoder(X0).dtype
    z0 = model.encoder(X0)                                   # (1,D)
    K = n_samples
    c0 = centroid(X0).reshape(1, 3)
    cgp, cgm = cgp.reshape(1, 3), cgm.reshape(1, 3)
    zgp_e, zgm_e, zc_e = zgp.expand(K, -1), zgm.expand(K, -1), zc.expand(K, -1)
    Rn = None if R_noise is None else R_noise.to(dtype)
    # Self-extinguishing envelope, restored. Step 25's salience was $\eta\,\mathcal H(p)$, so a *noiseless*
    # collapse to $p\in\{0,1\}$ made $\mathcal H(p)=0$ *exactly*, the salience channel constant-zero, and
    # its z-score vanish -- the cue drive switched off and the agent committed. Soft Bayes never reaches
    # $\{0,1\}$ exactly, so $\mathrm{IG}$ stays small-but-nonzero and, crucially, still *varies* across
    # candidates; z-scoring would renormalise that vanishing signal back to unit std and keep pulling the
    # agent to the cue forever (it never commits). We therefore gate the epistemic channel by the
    # **normalised belief entropy** $g_{\rm epi}=\mathcal H(p)/\ln 2\in[0,1]$ -- the mutual information's
    # own ceiling ($\mathrm{IG}\le\mathcal H(p)$): the agent values information exactly to the extent there
    # is information left to gain, and $g_{\rm epi}\to0$ as the belief firms, handing control to the
    # pragmatic term so it commits. $g_{\rm epi}$ is a belief scalar => SE(3)-invariance is untouched, and
    # it leaves the $\beta{=}0$ reward-only baseline (pure hedge) identical.
    g_epi = min(max(belief_entropy(p) / math.log(2.0), 0.0), 1.0)
    mean = torch.zeros(H, 3, dtype=dtype)
    sigma = torch.full((H, 3), sigma0, dtype=dtype)

    def rollout(cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Belief-weighted latent cost, exact-centroid cost, and best-approach IG. ``cand:(ns,H,3)``."""
        ns = cand.shape[0]
        z = z0.expand(ns, -1).contiguous()
        prag_lat = torch.zeros(ns, dtype=dtype)
        ig_best = torch.zeros(ns, dtype=dtype)               # running max_h IG(p; eps(d_h))
        for h in range(H):
            z = model.predictor(z, cand[:, h])               # (ns,D)
            d2 = p * ((z - zgp_e[:ns]) ** 2).sum(-1) + (1.0 - p) * ((z - zgm_e[:ns]) ** 2).sum(-1)
            prag_lat = prag_lat + (w_run * d2 if h < H - 1 else d2)
            d_cue = ((z - zc_e[:ns]) ** 2).sum(-1).clamp_min(0.0).sqrt()   # (ns,)
            eps_h = crossover(d_cue, eps0=eps0, delta=delta)              # (ns,)
            ig_best = torch.maximum(ig_best, info_gain(p, eps_h))         # exact MI, self-extinguishing
        pred_centroid = c0 + C_T * cand.sum(dim=1)           # (ns,3) exact translation channel
        cen = p * ((pred_centroid - cgp) ** 2).sum(-1) + (1.0 - p) * ((pred_centroid - cgm) ** 2).sum(-1)
        return prag_lat, cen, ig_best

    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 3, generator=gen, dtype=dtype)
        if Rn is not None:
            eps = torch.einsum("ij,nhj->nhi", Rn, eps)       # rotate exploration noise (equivariance)
        cand = _ball_clamp(mean.unsqueeze(0) + sigma.unsqueeze(0) * eps)
        prag_lat, cen, ig = rollout(cand)
        # z-score EACH channel separately (see Step 25): the latent term sums over D=48 dims and
        # accumulates over H, so in raw units it dwarfs the 3-D centroid and the scalar IG; per-channel
        # standardisation makes w_t, beta clean dimensionless trade-offs. Each channel is SE(3)-invariant
        # per candidate, so its population mean/std are invariant scalars => the EFE stays SE(3)-invariant.
        # The epistemic channel carries the belief-entropy gate g_epi (see above): it fades to zero as the
        # belief firms, so the agent stops chasing the (now-uninformative) cue and commits to the goal.
        cost = _zscore(prag_lat) + w_t * _zscore(cen) - beta * g_epi * _zscore(ig)
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        var_iso = ((elite - mean.unsqueeze(0)) ** 2).mean(dim=(0, 2))
        sigma = var_iso.sqrt().clamp_min(1e-3).unsqueeze(-1).expand(H, 3)
    return mean


# --------------------------------------------------------------------------- #
# the closed loop: plan -> execute -> *noisy* sense in the cue band -> soft Bayes accumulate -> commit
# --------------------------------------------------------------------------- #
@torch.no_grad()
def closed_loop_pomdp_noisy(
    model, task: tuple, *, beta: float, eps0: float, R_noise: torch.Tensor | None = None,
    R: torch.Tensor | None = None, t: torch.Tensor | None = None, oracle: bool = False,
    delta_frac: float = 0.45, w_t: float = 0.5, T_max: int = 18, replan_every: int = 6,
    gen: torch.Generator | None = None, np_rng: np.random.Generator | None = None, **cem,
) -> dict:
    r"""One noisy-cue POMDP episode under the EFE planner; optionally on a transformed copy $(R,t)$.

    Unlike Step 25's noiseless one-bit collapse, sensing here is graded: whenever the executed latent is
    within the cue band $d<\delta=\texttt{delta\_frac}\cdot\lVert z_c-z_0\rVert$, the agent draws a noisy
    bit (flip probability $\epsilon(d)$, more reliable the closer it is) and applies a **soft** Bayes
    update -- the belief *accumulates* and never collapses. The mutual-information drive
    self-extinguishes as $\mathcal H(p)\!\to\!0$, so the planner peels off to the (now-believed) goal once
    the evidence is strong enough that the pragmatic term dominates. ``oracle=True`` starts with the
    belief already correct (the achievable floor). Returns the **true-goal** error (Kabsch angle +
    centroid distance), the number of *informative* senses (band entries), and the min latent distance.
    """
    X0, Xc, Xgp, Xgm, b = task
    if R is not None:
        X0, Xc, Xgp, Xgm = (_apply(Z, R, t) for Z in (X0, Xc, Xgp, Xgm))
    Xtrue = Xgp if b == 1 else Xgm
    zgp, zgm, zc = model.encoder(Xgp), model.encoder(Xgm), model.encoder(Xc)
    cgp, cgm = centroid(Xgp), centroid(Xgm)
    z0 = model.encoder(X0)
    delta = delta_frac * float((zc - z0).norm()) + 1e-9
    if np_rng is None:
        np_rng = np.random.default_rng(0)
    p = (1.0 if b == 1 else 0.0) if oracle else 0.5
    X = X0.clone()
    step = 0
    n_sense = 0
    min_cue_dist = float("inf")
    while step < T_max:
        plan = efe_pomdp_plan_noisy(
            model, X, p=p, zgp=zgp, zgm=zgm, cgp=cgp, cgm=cgm, zc=zc, delta=delta,
            beta=(0.0 if oracle else beta), eps0=eps0, R_noise=R_noise, w_t=w_t, gen=gen, **cem,
        )
        for kk in range(min(replan_every, T_max - step)):
            X = teacher_step(X, plan[kk : kk + 1])
            step += 1
            d_cue = float((model.encoder(X) - zc).norm())
            min_cue_dist = min(min_cue_dist, d_cue)
            if (not oracle) and d_cue < delta:               # inside the band: draw ONE noisy bit
                eps_d = float(crossover(d_cue, eps0=eps0, delta=delta))
                o = b if (np_rng.random() > eps_d) else -b   # correct w.p. 1-eps, else flipped
                p = bayes_update(p, o, eps_d)                # soft accumulation, no collapse
                n_sense += 1
    R_resid = kabsch_rotation(X[0], Xtrue[0])
    return {
        "ang": rotation_angle_deg(R_resid),
        "pos": float((centroid(X) - centroid(Xtrue)).norm()),
        "n_sense": n_sense,
        "sensed": bool(n_sense > 0) if not oracle else False,
        "p_final": p,
        "min_cue_dist": min_cue_dist,
        "delta": delta,
        "d": float((centroid(Xgp) - centroid(Xgm)).norm()) / 2.0,
    }


# --------------------------------------------------------------------------- #
# [C] SE(3) invariance of the mutual-information field (the theorem, at the planner level)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _ig_field(
    model, X0: torch.Tensor, Xc: torch.Tensor, cand: torch.Tensor, delta: float, *,
    p: float, eps0: float,
) -> torch.Tensor:
    r"""Best-approach mutual information $\max_h\mathrm{IG}(p;\epsilon(d_h))$ for a fixed candidate batch.

    Input-level (re-encodes the clouds), so the invariance test assumes nothing about how the latent
    transforms -- it rotates clouds+actions and checks the IG field is unchanged. ``cand:(ns,H,3)->(ns,)``.
    """
    ns = cand.shape[0]
    zc = model.encoder(Xc)
    z = model.encoder(X0).expand(ns, -1).contiguous()
    ig_best = torch.zeros(ns, dtype=z.dtype)
    for h in range(cand.shape[1]):
        z = model.predictor(z, cand[:, h])
        d_cue = ((z - zc) ** 2).sum(-1).clamp_min(0.0).sqrt()
        ig_best = torch.maximum(ig_best, info_gain(p, crossover(d_cue, eps0=eps0, delta=delta)))
    return ig_best


@torch.no_grad()
def info_value_invariance(
    model, X0: torch.Tensor, Xc: torch.Tensor, R: torch.Tensor, t: torch.Tensor, *,
    eps0: float, p: float = 0.5, H: int = 12, n_samples: int = 64,
    gen: torch.Generator | None = None,
) -> float:
    r"""$\max_n|\mathrm{IG}_n(X_0,X_c)-\mathrm{IG}_n(RX_0{+}t,\,RX_c{+}t,\,R\,\text{cand})|$ -- $0$ for VN.

    A global $(R,t)$ rotates clouds and candidate actions; the IG field depends only on the latent
    distance to the cue, so for the equivariant encoder it is unchanged to the float floor. The MLP
    conflates pose with cue-proximity and breaks it.
    """
    z0, zc = model.encoder(X0), model.encoder(Xc)
    delta = 0.45 * float((zc - z0).norm()) + 1e-9
    cand = _ball_clamp(0.6 * torch.randn(n_samples, H, 3, generator=gen, dtype=X0.dtype))
    ig = _ig_field(model, X0, Xc, cand, delta, p=p, eps0=eps0)
    cand_r = torch.einsum("ij,nhj->nhi", R.to(X0.dtype), cand)
    ig_r = _ig_field(model, _apply(X0, R, t), _apply(Xc, R, t), cand_r, delta, p=p, eps0=eps0)
    return (ig - ig_r).abs().max().item()


# --------------------------------------------------------------------------- #
# evaluation helper (paired CEM seeds + paired noise streams across agents)
# --------------------------------------------------------------------------- #
def _eval_agent(
    model, tasks: list[tuple], *, beta: float, eps0: float, oracle: bool, base_seed: int, **kw
) -> list[dict]:
    r"""Run every POMDP once, pairing BOTH the CEM seed and the observation-noise stream across agents."""
    out = []
    for i, task in enumerate(tasks):
        gen = torch.Generator().manual_seed(base_seed + i)
        np_rng = np.random.default_rng(base_seed + i)        # paired noisy-bit stream
        out.append(closed_loop_pomdp_noisy(
            model, task, beta=beta, eps0=eps0, oracle=oracle, gen=gen, np_rng=np_rng, **kw))
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    line = "=" * 76

    if SMOKE:
        N_TRAIN, EPOCHS = 200, 3
        K_TASKS, K_SWEEP, N_OOD = 6, 4, 1
        BETA = 12.0
        EPS0_HEAD = 0.15
        EPS0_LADDER = (0.0, 0.25, 0.45)
        # Noisy soft evidence is *expensive*: unlike Step 25's one-touch noiseless collapse, the agent
        # must dwell near the cue, spiral inward for cleaner bits, and accumulate over many senses before
        # it is confident enough to peel off and commit -- so it needs finer replanning (react to the
        # growing belief) and a longer horizon (accumulate, then travel to the goal). 6 replan windows.
        T_MAX, REPLAN, W_T = 18, 3, 2.0
        cem_kw = dict(H=8, n_samples=64, n_iters=3, n_elite=8, sigma0=0.6, w_run=0.3)
    else:
        N_TRAIN, EPOCHS = 1500, 60
        K_TASKS, K_SWEEP, N_OOD = 24, 16, 3
        BETA = 12.0
        EPS0_HEAD = 0.15
        EPS0_LADDER = (0.0, 0.05, 0.15, 0.25, 0.35, 0.45)
        # 6 replan windows (vs Step 25's 3): noisy soft evidence must be *accumulated*, so the agent
        # needs to react to its growing belief (finer replanning) and have horizon to both spiral into
        # the cue for clean bits and then travel to the now-believed goal (T_max 18->24).
        T_MAX, REPLAN, W_T = 24, 4, 2.0
        cem_kw = dict(H=12, n_samples=256, n_iters=5, n_elite=25, sigma0=0.6, w_run=0.3)
    VAR_COEF = 0.1
    DELTA_FRAC = 0.45
    loop_kw = dict(w_t=W_T, T_max=T_MAX, replan_every=REPLAN, delta_frac=DELTA_FRAC, **cem_kw)

    print(line)
    print("STEP 34: active inference de-constructed -- invariant curiosity under a NOISY cue")
    print(f"    mode={'SMOKE' if SMOKE else 'FULL'}  N_train={N_TRAIN}  epochs={EPOCHS}  "
          f"K_tasks={K_TASKS}  beta={BETA}  eps0_head={EPS0_HEAD}")
    print(line)

    # ---- train the two world models (Step 13/18/25 recipe, z-wedge [0,90)) ---------------------
    print(f"\n    training VN/MLP latent JEPA on {N_TRAIN} cloud transitions, phi in [0,90)")
    S, A, S2 = collect_cloud_transitions(N_TRAIN, seed=0)
    vn, mlp = build_eq_jepa(), build_mlp_jepa()
    train_jepa(vn, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=999)
    train_jepa(mlp, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=VAR_COEF, seed=0, log_every=999)
    params = {"VN": n_params(vn), "MLP": n_params(mlp)}
    print(f"    params: VN={params['VN']}  MLP={params['MLP']}")

    Sc, Ac, _ = collect_cloud_transitions(64, seed=999)
    Rchk = rand_so3(torch.Generator().manual_seed(3))
    equiv = {
        "VN": {"enc": encoder_equiv_err(vn.encoder, Sc, Rchk),
               "pred": predictor_equiv_err(vn.predictor, vn.encoder(Sc), Ac, Rchk)},
        "MLP": {"enc": encoder_equiv_err(mlp.encoder, Sc, Rchk),
                "pred": predictor_equiv_err(mlp.predictor, mlp.encoder(Sc), Ac, Rchk)},
    }
    print("    post-train equivariance |rho.E(x)-E(Rx)| / |rho.f-f(rho.,R.)|:")
    print(f"        VN  : enc {_fmt(equiv['VN']['enc'])}  pred {_fmt(equiv['VN']['pred'])}  (exact)")
    print(f"        MLP : enc {_fmt(equiv['MLP']['enc'])}  pred {_fmt(equiv['MLP']['pred'])}  (no prior)")
    vn, mlp = vn.to(EVAL_DTYPE), mlp.to(EVAL_DTYPE)

    # ---- tasks ---------------------------------------------------------------------------------
    tasks = make_cue_tasks(K_TASKS, seed=321)
    hedge_floor = float(np.mean([
        float((centroid(Xgp) - centroid(Xgm)).norm()) / 2.0 for _, _, Xgp, Xgm, _ in tasks
    ]))
    print(f"\n    {K_TASKS} ambiguous-goal POMDPs; hedge-floor d (origin->goal centroid)={hedge_floor:.3f}")
    print(f"    noisy cue: eps(d)=1/2-(1/2-eps0)exp(-d^2/2delta^2), delta={DELTA_FRAC}*|z_c-z_0|; "
          f"soft-Bayes accumulation (no collapse)")
    print(f"    closed loop: T_max={T_MAX}, replan_every={REPLAN}, beta={BETA}, w_t={W_T}")

    # ---- [A] the task win UNDER NOISE: reward-only vs EFE vs oracle, same equivariant latent -----
    print("\n" + line)
    print(f"[A] TASK WIN UNDER NOISE (eps0={EPS0_HEAD}): reward-only (b=0) vs EFE (b>0) vs oracle")
    print(line)
    res_reward = _eval_agent(vn, tasks, beta=0.0, eps0=EPS0_HEAD, oracle=False, base_seed=10_000, **loop_kw)
    res_efe = _eval_agent(vn, tasks, beta=BETA, eps0=EPS0_HEAD, oracle=False, base_seed=10_000, **loop_kw)
    res_oracle = _eval_agent(vn, tasks, beta=0.0, eps0=EPS0_HEAD, oracle=True, base_seed=10_000, **loop_kw)
    agents = {"reward-only": res_reward, "EFE": res_efe, "oracle": res_oracle}
    print(f"    {'agent':>12} | {'pos err':>22} | {'ang(deg)':>10} | {'#senses':>8}")
    print("    " + "-" * 64)
    stats = {}
    for name, res in agents.items():
        pos, ang = _agent_arrays(res, "pos"), _agent_arrays(res, "ang")
        nse = float(np.mean([r["n_sense"] for r in res])) if name != "oracle" else float("nan")
        pm, plo, phi = boot_mean_ci(pos, seed=1)
        am = float(ang.mean())
        stats[name] = {"pos": pm, "pos_ci": (plo, phi), "ang": am, "n_sense": nse}
        nstr = "   n/a" if name == "oracle" else f"{nse:>6.2f}"
        print(f"    {name:>12} | {pm:7.4f} CI[{plo:6.4f},{phi:6.4f}] | {am:10.3f} | {nstr:>8}")
    rp, ep, op = (_agent_arrays(res_reward, "pos"), _agent_arrays(res_efe, "pos"),
                  _agent_arrays(res_oracle, "pos"))
    drop_m, drop_lo, drop_hi = boot_mean_ci(rp - ep, seed=2)
    ratio_m, ratio_lo, ratio_hi = boot_ratio_ci(ep, rp, seed=3)
    gap_m, gap_lo, gap_hi = boot_mean_ci(ep - op, seed=4)
    sense_efe = stats["EFE"]["n_sense"]
    print(f"    paired EFE-vs-reward pos-error drop: mean={drop_m:+.4f} CI[{drop_lo:+.4f},{drop_hi:+.4f}] "
          f"(>0 => EFE wins)")
    print(f"    EFE/reward-only pos-error ratio: {ratio_m:.3f} CI[{ratio_lo:.3f},{ratio_hi:.3f}] (<1 => win)")
    print(f"    EFE-minus-oracle pos gap: {gap_m:+.4f} CI[{gap_lo:+.4f},{gap_hi:+.4f}] (the honest NOISE gap)")
    print(f"    => EFE senses the noisy cue x{sense_efe:.2f} times (graded accumulation; Step 25 sensed ~1) "
          f"and collapses the hedge to ~oracle+noise.")

    # ---- [B] the de-construction made quantitative: a noise sweep eps0: 0 -> 1/2 -----------------
    print("\n" + line)
    print("[B] NOISE SWEEP eps0: 0 (Step 25 limit) -> 1/2 (useless cue) -- the de-construction, quantified")
    print(line)
    sweep_tasks = tasks[:K_SWEEP]
    sweep = []
    print(f"    {'eps0':>6} | {'EFE pos':>16} | {'reward pos':>11} | {'oracle pos':>11} | "
          f"{'#senses':>8} | {'win?':>5}")
    print("    " + "-" * 74)
    for e0 in EPS0_LADDER:
        r_efe = _eval_agent(vn, sweep_tasks, beta=BETA, eps0=e0, oracle=False, base_seed=20_000, **loop_kw)
        r_rew = _eval_agent(vn, sweep_tasks, beta=0.0, eps0=e0, oracle=False, base_seed=20_000, **loop_kw)
        r_orc = _eval_agent(vn, sweep_tasks, beta=0.0, eps0=e0, oracle=True, base_seed=20_000, **loop_kw)
        pe, pr, po = (_agent_arrays(r_efe, "pos"), _agent_arrays(r_rew, "pos"), _agent_arrays(r_orc, "pos"))
        pem, pelo, pehi = boot_mean_ci(pe, seed=11)
        nse = float(np.mean([r["n_sense"] for r in r_efe]))
        # "win" at this eps0: EFE beats reward-only by a clear margin (paired)
        rat_m, _, rat_hi = boot_ratio_ci(pe, pr, seed=13)
        won = bool(rat_hi < 0.85)
        sweep.append({"eps0": e0, "efe_pos": pem, "efe_pos_ci": (pelo, pehi),
                      "reward_pos": float(pr.mean()), "oracle_pos": float(po.mean()),
                      "n_sense": nse, "ratio": rat_m, "ratio_hi": rat_hi, "won": won})
        print(f"    {e0:>6.2f} | {pem:6.3f} CI[{pelo:5.3f},{pehi:5.3f}] | {float(pr.mean()):11.3f} | "
              f"{float(po.mean()):11.3f} | {nse:8.2f} | {'YES' if won else ' no':>5}")
    s0 = sweep[0]                                            # eps0 = 0   (noiseless limit -> recover Step 25)
    slast = sweep[-1]                                        # eps0 ~ 1/2 (useless cue -> win must vanish)
    near_oracle_at_0 = (s0["efe_pos"] - s0["oracle_pos"]) < 0.4 * max(s0["reward_pos"] - s0["oracle_pos"], 1e-9)
    no_win_when_useless = (not slast["won"])
    monotone_senses = sweep[0]["n_sense"] <= sweep[-1]["n_sense"] + 1e-6   # noisier => work harder (loose)
    print(f"    eps0=0: EFE {s0['efe_pos']:.3f} ~ oracle {s0['oracle_pos']:.3f} (recovers Step 25); "
          f"eps0={slast['eps0']:.2f}: EFE {slast['efe_pos']:.3f} ~ reward {slast['reward_pos']:.3f} "
          f"(win {'gone' if no_win_when_useless else 'PRESENT?!'})")

    # ---- [C] the theorem realised FOR THE MI DRIVE: rotate the whole episode by a global (R,t) ---
    print("\n" + line)
    print("[C] SE(3) THEOREM for the mutual-information drive: rotate the POMDP by a global (R,t)")
    print(line)
    gen_inv = torch.Generator().manual_seed(7)
    X0c, Xcc = tasks[0][0], tasks[0][1]
    ig_inv = {}
    for name, m in (("VN", vn), ("MLP", mlp)):
        worst = 0.0
        for _ in range(N_OOD):
            R = rand_so3(gen_inv).to(EVAL_DTYPE)
            tt = ((torch.rand(3, generator=gen_inv) * 2 - 1) * 0.8).to(EVAL_DTYPE)
            g = torch.Generator().manual_seed(99)
            worst = max(worst, info_value_invariance(
                m, X0c, Xcc, R, tt, eps0=EPS0_HEAD, H=cem_kw["H"], gen=g))
        ig_inv[name] = worst
    print(f"    IG-field invariance  max|IG(x)-IG(Rx+t)|:  VN {_fmt(ig_inv['VN'])}   MLP {_fmt(ig_inv['MLP'])}")

    orbit = [(torch.eye(3, dtype=EVAL_DTYPE), torch.zeros(3, dtype=EVAL_DTYPE))]
    for _ in range(N_OOD):
        R = rand_so3(gen_inv).to(EVAL_DTYPE)
        tt = ((torch.rand(3, generator=gen_inv) * 2 - 1) * 0.8).to(EVAL_DTYPE)
        orbit.append((R, tt))
    task0 = tasks[0]
    out_inv = {}
    for name, m in (("VN", vn), ("MLP", mlp)):
        errs = []
        for gid, (R, tt) in enumerate(orbit):
            g = torch.Generator().manual_seed(55)
            np_rng = np.random.default_rng(55)               # SAME noise stream across the orbit
            Rn = None if gid == 0 else R
            r = closed_loop_pomdp_noisy(m, task0, beta=BETA, eps0=EPS0_HEAD, R_noise=Rn, R=R, t=tt,
                                        gen=g, np_rng=np_rng, **loop_kw)
            errs.append((r["pos"], r["ang"]))
        seen_pos, seen_ang = errs[0]
        worst_pos = max(abs(p - seen_pos) for p, _ in errs[1:]) if len(errs) > 1 else 0.0
        worst_ang = max(abs(a - seen_ang) for _, a in errs[1:]) if len(errs) > 1 else 0.0
        out_inv[name] = {"pos": worst_pos, "ang": worst_ang}
    print("    true-goal-outcome invariance under (R,t)  max|err(seen)-err(R,t)|:")
    print(f"        VN  : pos {_fmt(out_inv['VN']['pos'])}  ang {_fmt(out_inv['VN']['ang'])}  (invariant)")
    print(f"        MLP : pos {_fmt(out_inv['MLP']['pos'])}  ang {_fmt(out_inv['MLP']['ang'])}  (breaks)")

    # plan equivariance of the Phase-1 EFE plan: plan(Rx) =? R.plan(x)
    R = rand_so3(gen_inv).to(EVAL_DTYPE)
    X0, Xc, Xgp, Xgm, _b = task0
    zero3 = torch.zeros(3, dtype=EVAL_DTYPE)

    def _phase1_plan(m, X0_, Xc_, Xgp_, Xgm_, Rn, gen):
        z0_ = m.encoder(X0_)
        delta_ = DELTA_FRAC * float((m.encoder(Xc_) - z0_).norm()) + 1e-9
        return efe_pomdp_plan_noisy(
            m, X0_, p=0.5, zgp=m.encoder(Xgp_), zgm=m.encoder(Xgm_), cgp=centroid(Xgp_),
            cgm=centroid(Xgm_), zc=m.encoder(Xc_), delta=delta_, beta=BETA, eps0=EPS0_HEAD,
            R_noise=Rn, w_t=W_T, gen=gen, **cem_kw,
        )

    base_plan = _phase1_plan(vn, X0, Xc, Xgp, Xgm, None, torch.Generator().manual_seed(321))
    rot_plan = _phase1_plan(
        vn, _apply(X0, R, zero3), _apply(Xc, R, zero3), _apply(Xgp, R, zero3), _apply(Xgm, R, zero3),
        R, torch.Generator().manual_seed(321))
    plan_equiv = (rot_plan - torch.einsum("ij,hj->hi", R, base_plan)).abs().max().item()
    print(f"    EFE plan equivariance  ||plan(Rx) - R.plan(x)||_inf = {_fmt(plan_equiv)}  (VN, beta={BETA})")

    # ---- verdict --------------------------------------------------------------------------------
    print("\n" + line)
    print("STEP 34 SUMMARY")
    print(line)
    ok_task_win = (ratio_hi < 0.75) and (drop_lo > 0.0)              # noise raises the floor: looser than .6
    ok_accumulate = sense_efe > 1.5                                  # multi-sense: graded, unlike Step 25
    ok_recover_step25 = near_oracle_at_0                             # eps0=0 -> near oracle
    ok_no_free_lunch = no_win_when_useless                          # eps0~1/2 -> win vanishes (falsifiable)
    ok_vn_inv = (ig_inv["VN"] < 1e-4 and out_inv["VN"]["pos"] < 1e-2
                 and out_inv["VN"]["ang"] < 1e-2 and plan_equiv < 1e-2)
    ok_mlp_breaks = ig_inv["MLP"] > 1e-2 or out_inv["MLP"]["pos"] > 1e-2
    ok_equiv = equiv["VN"]["enc"] < 1e-4 and equiv["VN"]["pred"] < 1e-4
    passed = (ok_task_win and ok_accumulate and ok_recover_step25 and ok_no_free_lunch
              and ok_vn_inv and ok_mlp_breaks and ok_equiv)
    print(f"    [A] EFE/reward ratio {ratio_m:.3f} (CI_hi<0.75 => win); noise gap to oracle {gap_m:+.4f}; "
          f"#senses {sense_efe:.2f}")
    print(f"    [B] eps0=0 EFE {s0['efe_pos']:.3f}~oracle {s0['oracle_pos']:.3f} (Step 25 limit); "
          f"eps0={slast['eps0']:.2f} win {'vanishes' if no_win_when_useless else 'PERSISTS?!'}")
    print(f"    [C] VN IG-inv {_fmt(ig_inv['VN'])}, outcome-inv pos {_fmt(out_inv['VN']['pos'])}, "
          f"plan-equiv {_fmt(plan_equiv)}; MLP IG-inv {_fmt(ig_inv['MLP'])} (breaks)")
    print(f"    guards: task-win={ok_task_win}  accumulate={ok_accumulate}  recover-step25={ok_recover_step25} "
          f" no-free-lunch={ok_no_free_lunch}  vn-inv={ok_vn_inv}  mlp-breaks={ok_mlp_breaks}  "
          f"equiv={ok_equiv}")
    print("    headline: with a NOISY cue that never gives certainty, the SE(3)-invariant epistemic drive")
    print("        -- now the EXACT mutual information of the sensor -- still earns the task win by")
    print("        accumulating graded evidence, degrades gracefully to 'no win' exactly when the cue stops")
    print("        carrying information (a built-in falsifiable negative), recovers Step 25 as eps0->0, and")
    print("        keeps the entire noisy loop exactly SE(3)-equivariant (VN) while the MLP breaks it.")
    print("    " + ("PASS" if passed else "FAIL"))

    tag = "_smoke" if SMOKE else ""
    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": {"N_TRAIN": N_TRAIN, "EPOCHS": EPOCHS, "K_TASKS": K_TASKS, "K_SWEEP": K_SWEEP,
                   "N_OOD": N_OOD, "BETA": BETA, "EPS0_HEAD": EPS0_HEAD, "EPS0_LADDER": list(EPS0_LADDER),
                   "T_MAX": T_MAX, "REPLAN": REPLAN, "W_T": W_T, "DELTA_FRAC": DELTA_FRAC, "cem": cem_kw},
        "params": params,
        "equivariance": equiv,
        "hedge_floor": hedge_floor,
        "task_win": {
            "eps0": EPS0_HEAD,
            "reward_pos": stats["reward-only"]["pos"], "reward_pos_ci": stats["reward-only"]["pos_ci"],
            "efe_pos": stats["EFE"]["pos"], "efe_pos_ci": stats["EFE"]["pos_ci"],
            "oracle_pos": stats["oracle"]["pos"], "oracle_pos_ci": stats["oracle"]["pos_ci"],
            "reward_ang": stats["reward-only"]["ang"], "efe_ang": stats["EFE"]["ang"],
            "oracle_ang": stats["oracle"]["ang"],
            "efe_n_sense": sense_efe,
            "drop_mean": drop_m, "drop_ci": (drop_lo, drop_hi),
            "ratio_mean": ratio_m, "ratio_ci": (ratio_lo, ratio_hi),
            "oracle_gap_mean": gap_m, "oracle_gap_ci": (gap_lo, gap_hi),
        },
        "noise_sweep": sweep,
        "invariance": {"ig_field": ig_inv, "outcome": out_inv, "plan_equiv": plan_equiv},
        "verdict": {
            "passed": bool(passed), "ok_task_win": bool(ok_task_win), "ok_accumulate": bool(ok_accumulate),
            "ok_recover_step25": bool(ok_recover_step25), "ok_no_free_lunch": bool(ok_no_free_lunch),
            "ok_vn_inv": bool(ok_vn_inv), "ok_mlp_breaks": bool(ok_mlp_breaks), "ok_equiv": bool(ok_equiv),
        },
    }
    out_path = ROOT / "papers" / "figures" / f"step34_active_inference_noisy{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")

    _make_figure(ROOT / "papers" / "figures" / f"step34_active_inference_noisy{tag}.png",
                 stats, sweep, hedge_floor, EPS0_HEAD)
    sys.exit(0 if passed else 1)


def _make_figure(path: Path, stats: dict, sweep: list[dict], hedge_floor: float, eps0_head: float) -> None:
    r"""Three panels: [A] task win under noise, [B] noise sweep eps0:0->1/2, [C] #senses vs eps0."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"    (figure skipped: {exc})")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # [A] task win under noise
    names = ["reward-only", "EFE", "oracle"]
    cols = ["#c0392b", "#2e86de", "#27ae60"]
    xs = np.arange(3)
    pos = [stats[n]["pos"] for n in names]
    los = [stats[n]["pos"] - stats[n]["pos_ci"][0] for n in names]
    his = [stats[n]["pos_ci"][1] - stats[n]["pos"] for n in names]
    ax[0].bar(xs, pos, yerr=[los, his], color=cols, capsize=5, width=0.6, alpha=0.9)
    ax[0].axhline(hedge_floor, ls="--", c="#7f8c8d", lw=1.3, label=f"hedge floor $d$={hedge_floor:.2f}")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names, rotation=12)
    ax[0].set_ylabel("true-goal position error")
    ax[0].set_title(f"[A] task win under noise ($\\epsilon_0$={eps0_head})")
    ax[0].legend(fontsize=8, loc="upper right")
    for x, n in zip(xs, names):
        if not math.isnan(stats[n]["n_sense"]):
            ax[0].text(x, pos[x] + (his[x] if his[x] > 0 else 0) + 0.02,
                       f"sns {stats[n]['n_sense']:.1f}", ha="center", fontsize=7, color="#34495e")

    # [B] noise sweep: EFE / reward / oracle position error vs eps0
    e = [s["eps0"] for s in sweep]
    efe = [s["efe_pos"] for s in sweep]
    elo = [s["efe_pos"] - s["efe_pos_ci"][0] for s in sweep]
    ehi = [s["efe_pos_ci"][1] - s["efe_pos"] for s in sweep]
    rew = [s["reward_pos"] for s in sweep]
    orc = [s["oracle_pos"] for s in sweep]
    ax[1].errorbar(e, efe, yerr=[elo, ehi], marker="o", c="#2e86de", capsize=3, label="EFE", lw=2)
    ax[1].plot(e, rew, marker="s", c="#c0392b", ls="--", label="reward-only (hedge)")
    ax[1].plot(e, orc, marker="^", c="#27ae60", ls=":", label="oracle")
    ax[1].set_xlabel("cue noise floor $\\epsilon_0$")
    ax[1].set_ylabel("true-goal position error")
    ax[1].set_title("[B] noise sweep: Step 25 limit $\\to$ useless cue")
    ax[1].annotate("recover\nStep 25", xy=(e[0], efe[0]), xytext=(e[0] + 0.05, efe[0] + 0.4),
                   fontsize=7, color="#2e86de",
                   arrowprops=dict(arrowstyle="->", color="#2e86de", lw=1))
    ax[1].annotate("win vanishes\n(useless cue)", xy=(e[-1], efe[-1]),
                   xytext=(e[-1] - 0.22, efe[-1] + 0.4), fontsize=7, color="#7f8c8d",
                   arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1))
    ax[1].legend(fontsize=8, loc="lower right")

    # [C] #senses vs eps0: the agent works harder for noisier bits
    nse = [s["n_sense"] for s in sweep]
    ax[2].plot(e, nse, marker="o", c="#8e44ad", lw=2)
    ax[2].set_xlabel("cue noise floor $\\epsilon_0$")
    ax[2].set_ylabel("# informative senses (EFE)")
    ax[2].set_title("[C] graded accumulation: noisier $\\Rightarrow$ sense more")
    ax[2].axhline(1.0, ls=":", c="#7f8c8d", lw=1, label="Step 25 $\\approx$ 1 sense")
    ax[2].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
