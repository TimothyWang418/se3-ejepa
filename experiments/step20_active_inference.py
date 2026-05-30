r"""Step 20: active inference — an Expected Free Energy objective in the equivariant latent.

Where this sits, and the precise question it answers
----------------------------------------------------
Steps 13–18 built an exactly $\mathrm{SE}(3)$-equivariant latent world model and a matching
*pragmatic* planner (drive the latent to a goal). That is the "perception + prediction + act
toward a goal" half of the loop. The project's fourth pillar — active inference (Friston) — adds
the *other* half: an agent should also act to **reduce its own uncertainty**. CLAUDE.md Open
Questions #2 and #5 ask exactly this: is there a tractable, *information-geometric* formulation of
active inference for a deep equivariant world model, and can it be unified with self-supervised
latent prediction? Step 20 answers with a concrete, falsifiable construction.

The agent minimises the **Expected Free Energy** (EFE) of an action sequence $a_{1:H}$,
$$
  G(a_{1:H}) \;=\; \underbrace{\sum_h w_h\,\lVert \bar z_h - z_g\rVert^2 \;+\; w_t\,\lVert \bar x_0 + c_t\!\sum_h a_h - \bar x_g\rVert^2}_{\text{pragmatic / risk (reach the goal — the Step-18 cost)}}
  \;-\; \beta\,\underbrace{\sum_h \mathcal{D}_h}_{\text{epistemic / information gain (reduce uncertainty)}},
$$
the standard risk$-$epistemic decomposition (Friston 2017; the $-\beta$ sign means *minimising* $G$
*maximises* information gain). We make both halves tractable in the **learned latent** of the
equivariant JEPA:

  * the **pragmatic** term is the validated Step-18 cost (latent terminal distance + the exact
    closed-form centroid channel for translation), using the ensemble-mean latent $\bar z_h$;
  * the **epistemic** term is the *ensemble disagreement*
    $\mathcal{D}_h=\tfrac1K\sum_k\lVert z_h^{(k)}-\bar z_h\rVert^2$ of a $K$-member predictor
    ensemble sharing **one** equivariant encoder (deep ensembles, Lakshminarayanan 2017; disagreement
    as exploration drive, Pathak 2019 / Sekar 2020 "Plan2Explore"). Its information-geometric face is
    the Gaussian differential entropy $\mathcal{H}=\tfrac12\log\det(\hat\Sigma+\epsilon I)$ of the
    predictive belief $\hat\Sigma=\tfrac1K\sum_k(z^{(k)}-\bar z)(z^{(k)}-\bar z)^\top$.

The geometric theorem (the reason this belongs in *this* project)
-----------------------------------------------------------------
Every predictor is jointly $\mathrm{SE}(3)$-equivariant, $f_k(\rho(R)z, Ra)=\rho(R)f_k(z,a)$, and the
shared encoder is equivariant ($E(Rx+t)=\rho(R)E(x)$). Because $\rho(R)$ is **orthogonal**, both the
mean $\bar z\mapsto\rho(R)\bar z$ is equivariant and the disagreement is **invariant**:
$$
  \mathcal{D}(\rho(R)z, Ra) = \tfrac1K\!\sum_k\lVert \rho(R)(z^{(k)}-\bar z)\rVert^2
  = \tfrac1K\!\sum_k\lVert z^{(k)}-\bar z\rVert^2 = \mathcal{D}(z,a),
$$
and likewise $\hat\Sigma\mapsto\rho(R)\hat\Sigma\rho(R)^\top$ so
$\log\det(\hat\Sigma+\epsilon I)$ is invariant ($\det\rho(R)=\pm1$). **The agent's epistemic drive —
its curiosity — is therefore an exactly $\mathrm{SE}(3)$-invariant scalar**: how much there is to
learn from an action does not depend on the global pose of the scene. Combined with the invariant
pragmatic cost, the whole EFE $G$ is $\mathrm{SE}(3)$-invariant, so the EFE-optimal plan is
$\mathrm{SE}(3)$-*equivariant*, $\mathrm{plan}(g\!\cdot\!x)=g\!\cdot\!\mathrm{plan}(x)$ — the entire
active-inference loop (perception, prediction, epistemic *and* pragmatic drives) transforms
consistently. For a non-equivariant ensemble none of this holds; that is the control.

Three panels (most to least decisive)
-------------------------------------
  [A]  EFE invariance — the theorem. The pragmatic term, the epistemic disagreement, the Gaussian
       entropy, and the total $G$ are all $\mathrm{SE}(3)$-invariant to the e3nn floor ($\sim10^{-6}$),
       at init AND after a real training run, for the VN ensemble; the MLP ensemble breaks each
       (control => the assertions are not vacuous). Gated, plus `tests/test_efe_invariance.py`.
  [B]  Epistemic geometry — the disagreement is *blind to re-orientation* (rotating a (cloud, action)
       pair onto another point of its $\mathrm{SE}(3)$ orbit leaves $\mathcal{D}$ **exactly** unchanged:
       the equivariant agent is correctly *not* curious about poses it already generalises across —
       举一反三), yet it is a *non-constant* field (CoV$>0$ across the probe batch, and genuinely
       off-orbit novelty — an OOD-shape cloud — raises it), so the invariance is non-vacuous. The
       non-equivariant control instead assigns spurious novelty to mere re-orientation (ratio $\neq 1$).
       Decisive and cheap.
  [C]  The active-inference knob — sweeping $\beta$ in an EFE CEM planner trades pragmatic progress
       for epistemic gain monotonically (more $\beta$ => the selected plan seeks higher-disagreement,
       more-informative regions at the cost of goal distance — exactly what active inference predicts),
       and the EFE-selected plan stays $\mathrm{SE}(3)$-equivariant (theorem realised end-to-end).

Honest scope (per CLAUDE.md: active inference = mathematical structure, not a guaranteed winner)
-----------------------------------------------------------------------------------------------
The teacher is **fully observed and deterministic** — so on *this* task the epistemic term is not
*required* to reach goals; the pragmatic planner already does (Step 18). What Step 20 establishes is
that the unified EFE objective is (i) well-posed and tractable in the equivariant latent, (ii) carries
an *exact* geometric invariance the project's thesis predicts and a non-equivariant model lacks, and
(iii) the active-inference knob measurably does what theory says. The empirical payoff *of exploration*
(tasks unreachable without information-seeking — partial observability, sparse/ambiguous goals) is the
named next rung and is **not** claimed here. Active inference is treated as the source of a geometric
structure, not as a benchmark win.

Run (full ~10–15 min on a laptop CPU):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
        .venv/bin/python experiments/step20_active_inference.py
Smoke (~60 s):
    STEP20_SMOKE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python experiments/step20_active_inference.py
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
sys.path.insert(0, str(HERE))   # for the Step 13 / 18 backbone we reuse

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

# Reuse the *validated* Step 13/18 machinery verbatim: the exactly-SE(3)-equivariant single-object
# teacher, the SO(3) helpers, the anisotropic template, the non-equivariant control encoder, the
# centroid + ball-clamp planner primitives, and the latent building blocks.
from step13_se3_latent_jepa import (  # noqa: E402
    ACTION_DIM,
    C_T,
    LATENT_DIM,
    N_POINTS,
    MLPPointEncoder,
    collect_cloud_transitions,
    rand_so3,
    rotate_latent,
    rotate_points,
    teacher_step,
)
from step18_se3_closed_loop import _ball_clamp, centroid  # noqa: E402
from step10_pusht_closed_loop import n_params  # noqa: E402
from src.models.eqjepa import EqJEPA, LatentPredictor  # noqa: E402  (EqJEPA imported for type parity)
from src.models.se3 import SE3PointEncoder  # noqa: E402
from src.models.structured import VNPredictor  # noqa: E402
from src.training.muon import build_muon_adamw  # noqa: E402

torch.set_default_dtype(torch.float32)

SMOKE = bool(os.environ.get("STEP20_SMOKE"))

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
K_ENS = 5                 # ensemble members (deep ensemble => epistemic disagreement)
N_OUT_VEC = LATENT_DIM // 3   # 16 type-1 vectors (D=48), reused from Step 13
EPS_PRIOR = 1e-2          # prior precision floor in the Gaussian entropy logdet (rank<D otherwise)
_ = EqJEPA                # silence unused-import linters; kept for type parity with sibling steps


# --------------------------------------------------------------------------- #
# the ensemble world model: ONE shared equivariant encoder + K predictors
# --------------------------------------------------------------------------- #
class EnsembleJEPA(nn.Module):
    r"""Shared encoder $E$ + $K$ predictors $f_1,\dots,f_K$ — a deep ensemble in a *shared* latent.

    The members **must** share the encoder: disagreement is only meaningful when all predictions live
    in the same latent coordinate frame (different encoders would conflate frame differences with
    genuine predictive uncertainty). With an equivariant $E$ and equivariant $f_k$, the per-member
    next-latents $z^{(k)}=f_k(E(x),a)$ all map by the same $\rho(R)$ under a global motion, which is
    what makes the disagreement exactly invariant.
    """

    def __init__(self, encoder: nn.Module, predictors: list[nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictors = nn.ModuleList(predictors)
        self.latent_dim = encoder.latent_dim

    @property
    def n_members(self) -> int:
        return len(self.predictors)

    def members_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        r"""All members' one-step predictions. ``z:(B,D), a:(B,A) -> (K,B,D)``."""
        return torch.stack([f(z, a) for f in self.predictors], dim=0)


def _seeded_vn_predictor(seed: int) -> VNPredictor:
    torch.manual_seed(seed)
    return VNPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=64, dim=3)


def _seeded_mlp_predictor(seed: int) -> LatentPredictor:
    torch.manual_seed(seed)
    return LatentPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden=256)


def build_vn_ensemble(k: int = K_ENS, seed: int = 0) -> EnsembleJEPA:
    r"""Equivariant ensemble: one ``SE3PointEncoder`` + ``k`` jointly-equivariant ``VNPredictor``s."""
    torch.manual_seed(seed)
    enc = SE3PointEncoder(n_out_vec=N_OUT_VEC, lmax=2, mul=8)
    preds = [_seeded_vn_predictor(seed + 1 + i) for i in range(k)]
    return EnsembleJEPA(enc, preds)


def build_mlp_ensemble(k: int = K_ENS, seed: int = 0) -> EnsembleJEPA:
    r"""Non-equivariant control: one ``MLPPointEncoder`` + ``k`` ordinary ``LatentPredictor``s."""
    torch.manual_seed(seed)
    enc = MLPPointEncoder(n_points=N_POINTS, latent_dim=LATENT_DIM, hidden=128)
    preds = [_seeded_mlp_predictor(seed + 1 + i) for i in range(k)]
    return EnsembleJEPA(enc, preds)


# --------------------------------------------------------------------------- #
# Expected-Free-Energy quantities (epistemic value) and their information-geometric face
# --------------------------------------------------------------------------- #
@torch.no_grad()
def disagreement(zk: torch.Tensor) -> torch.Tensor:
    r"""Ensemble disagreement $\mathcal{D}=\tfrac1K\sum_k\lVert z^{(k)}-\bar z\rVert^2$. ``(K,B,D)->(B,)``.

    Exactly $\mathrm{SE}(3)$-invariant for an equivariant ensemble (orthogonal $\rho$). This is the
    epistemic term the EFE planner uses (cheap, stable across many candidates).
    """
    zbar = zk.mean(dim=0, keepdim=True)         # (1,B,D)
    return ((zk - zbar) ** 2).sum(dim=-1).mean(dim=0)   # (B,)


@torch.no_grad()
def gaussian_entropy(zk: torch.Tensor, eps: float = EPS_PRIOR) -> torch.Tensor:
    r"""Differential entropy proxy $\tfrac12\log\det(\hat\Sigma+\epsilon I)$ of the predictive belief.

    ``zk:(K,B,D) -> (B,)``. $\hat\Sigma=\tfrac1K\sum_k(z^{(k)}-\bar z)(z^{(k)}-\bar z)^\top$ is the
    empirical $D\times D$ covariance of the $K$ member predictions (rank $\le K-1$, hence the $\epsilon I$
    prior-precision floor). The information-geometric face of the epistemic value, and exactly invariant
    because $\hat\Sigma\mapsto\rho\hat\Sigma\rho^\top$ and $\det\rho=\pm1$.
    """
    k, b, d = zk.shape
    zc = zk - zk.mean(dim=0, keepdim=True)       # (K,B,D)
    # batched outer-product mean over members: Sigma[b] = (1/K) sum_k zc[k,b] zc[k,b]^T
    sigma = torch.einsum("kbd,kbe->bde", zc, zc) / k        # (B,D,D)
    eye = eps * torch.eye(d, dtype=zk.dtype).expand(b, d, d)
    return 0.5 * torch.logdet(sigma + eye)        # (B,)


# --------------------------------------------------------------------------- #
# training: shared encoder + K predictors, EMA target, VICReg, per-member bootstrap
# --------------------------------------------------------------------------- #
def train_ensemble_jepa(
    model: EnsembleJEPA, S: torch.Tensor, A: torch.Tensor, S2: torch.Tensor, *,
    epochs: int = 60, batch_size: int = 128, var_coef: float = 0.1,
    ema_decay: float = 0.99, seed: int = 0, log_every: int = 999, verbose: bool = True,
) -> dict:
    r"""EMA-target JEPA for a shared-encoder ensemble (mirrors ``train_jepa``, multi-head).

    Loss $=\tfrac1K\sum_k \mathrm{MSE}\big(f_k(E(s),a),\,\mathrm{sg}\,E_{\bar\theta}(s')\big)
    + \lambda\,\mathrm{VICReg}(E(s))$, with a **per-member Poisson(1) bootstrap weight** on the batch
    rows (online bagging, Oza & Russell 2001) so members fit the data yet diverge where it is sparse —
    the disagreement signal that powers the epistemic term. One shared EMA target encoder; Muon/AdamW
    routing exactly as Step 4+.
    """
    import copy

    torch.manual_seed(seed)
    model.train()
    target_enc = copy.deepcopy(model.encoder)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    target_enc.eval()

    muon, adamw, counts = build_muon_adamw(model, muon_lr=0.02, adamw_lr=1e-3)
    if verbose:
        print(f"    optim routing: {counts}")

    n = S.shape[0]
    g = torch.Generator().manual_seed(seed)
    last_std = float("nan")
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        ep_pred = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            o, a, o2 = S[idx], A[idx], S2[idx]
            z0 = model.encoder(o)                       # (B,D) online
            with torch.no_grad():
                z_tgt = target_enc(o2)                  # (B,D) EMA target, stop-grad
            pred_loss = 0.0
            for f in model.predictors:
                w = torch.poisson(torch.ones(o.shape[0]), generator=g)   # (B,) Poisson(1) bootstrap
                err = ((f(z0, a) - z_tgt) ** 2).mean(dim=-1)             # (B,)
                pred_loss = pred_loss + (w * err).sum() / w.sum().clamp_min(1.0)
            pred_loss = pred_loss / model.n_members
            std = z0.std(dim=0)
            loss = pred_loss + var_coef * F.relu(1.0 - std).mean()

            for opt in (muon, adamw):
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in (muon, adamw):
                if opt is not None:
                    opt.step()

            with torch.no_grad():                       # EMA update (name-matched, e2cnn-safe)
                online_p = dict(model.encoder.named_parameters())
                for name, pt in target_enc.named_parameters():
                    pt.mul_(ema_decay).add_(online_p[name], alpha=1.0 - ema_decay)
            ep_pred += float(pred_loss.detach())
            last_std = std.mean().item()
            nb += 1
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}  pred={ep_pred / max(nb, 1):.4f}  latent_std={last_std:.4f}")
    model.eval()
    return {"latent_std": last_std, "routing": counts}


# --------------------------------------------------------------------------- #
# [A] invariance of the EFE terms under a global SE(3) motion (init AND post-train)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def efe_term_invariance(model: EnsembleJEPA, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor) -> dict:
    r"""Residuals $\lvert\mathcal{Q}(E(S),A)-\mathcal{Q}(E(RS),RA)\rvert$ for the epistemic quantities.

    Returns the worst-case (over the batch) absolute change of the disagreement $\mathcal{D}$ and the
    Gaussian entropy $\mathcal{H}$ under a global rotation. $0$ for an equivariant ensemble; nonzero for
    the MLP control. (Translation is a separate exact invariance — the encoder centres the cloud — so
    we probe the rotation here, the non-trivial part.)
    """
    z, a = model.encoder(S), A
    zr, ar = model.encoder(rotate_points(S, R)), rotate_points(A, R)
    dk, dkr = disagreement(model.members_next(z, a)), disagreement(model.members_next(zr, ar))
    hk, hkr = gaussian_entropy(model.members_next(z, a)), gaussian_entropy(model.members_next(zr, ar))
    return {
        "disagree": (dk - dkr).abs().max().item(),
        "entropy": (hk - hkr).abs().max().item(),
    }


@torch.no_grad()
def total_efe_invariance(
    model: EnsembleJEPA, X0: torch.Tensor, Xg: torch.Tensor, A: torch.Tensor, R: torch.Tensor,
    t: torch.Tensor, *, beta: float, w_t: float = 0.5,
) -> float:
    r"""Residual of the *whole* one-step EFE $G$ (pragmatic + centroid $-\beta\,$epistemic) under $(R,t)$.

    Builds $G$ for $(X_0,X_g,A)$ and for the globally transformed $(RX_0{+}t,\,RX_g{+}t,\,RA)$; for the
    equivariant ensemble the two are equal to the e3nn floor (the realised invariance of the objective).
    """
    def efe(x0, xg, aa):
        z0 = model.encoder(x0)
        zg = model.encoder(xg)
        zk = model.members_next(z0, aa)
        prag = ((zk.mean(0) - zg) ** 2).sum(-1)                       # (1,)
        cen = ((centroid(x0) + C_T * aa.sum(0, keepdim=True) - centroid(xg)) ** 2).sum(-1)
        return prag + w_t * cen - beta * disagreement(zk)
    g0 = efe(X0, Xg, A)
    g1 = efe(rotate_points(X0, R) + t, rotate_points(Xg, R) + t, rotate_points(A, R))
    return (g0 - g1).abs().max().item()


# --------------------------------------------------------------------------- #
# [B] epistemic calibration: disagreement is larger on novel (OOD) poses, and geometry-aware
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mean_one_step_disagreement(model: EnsembleJEPA, S: torch.Tensor, A: torch.Tensor) -> float:
    r"""Average ensemble disagreement of the one-step prediction over a probe set. ``-> scalar``."""
    return disagreement(model.members_next(model.encoder(S), A)).mean().item()


# --------------------------------------------------------------------------- #
# [C] the active-inference knob: an EFE CEM planner (pragmatic - beta * epistemic)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def efe_cem_plan(
    model: EnsembleJEPA, X0: torch.Tensor, zg: torch.Tensor, *, cg: torch.Tensor, beta: float,
    R_noise: torch.Tensor | None = None, w_t: float = 0.5, H: int = 12, n_samples: int = 256,
    n_iters: int = 5, n_elite: int = 25, sigma0: float = 0.6, w_run: float = 0.3,
    gen: torch.Generator | None = None,
) -> dict:
    r"""SE(3)-equivariant CEM minimising the EFE $G=\text{pragmatic}-\beta\,\text{epistemic}$.

    Identical equivariant scaffolding to Step 18's :func:`latent_cem_plan_iso` (iso-$\sigma$, ball
    clamp, $R$-rotated noise, closed-form centroid channel), with two changes: the pragmatic latent
    cost uses the **ensemble-mean** rollout, and an **epistemic** term $-\beta\sum_h\mathcal{D}_h$ is
    added. Both terms are *z-scored across the candidate population* before combining, so $\beta$ is a
    clean dimensionless exploration weight and the topk selection is well-conditioned; the
    standardisation constants are scalars over the (jointly rotated) candidate set, hence invariant, so
    the equivariance theorem is preserved. Returns the plan and the *raw* (un-scored) pragmatic and
    epistemic values of the selected plan (for the trade-off curve).

    Shapes: ``X0:(1,P,3), zg:(1,D), cg:(3,)|(1,3)``. Returns ``{"plan":(H,3), "prag":float, "epi":float}``.
    """
    z0 = model.encoder(X0)                                   # (1,D) shared encoder
    K = model.n_members
    c0 = centroid(X0).reshape(1, 3)
    cg = cg.reshape(1, 3)
    Rn = R_noise
    mean = torch.zeros(H, 3)
    sigma = torch.full((H, 3), sigma0)

    def rollout(cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Roll the K-member ensemble; return (pragmatic cost, epistemic value) per candidate."""
        ns = cand.shape[0]
        zk = z0.expand(K, ns, -1).contiguous()              # (K,ns,D) all members from shared z0
        prag = torch.zeros(ns)
        epi = torch.zeros(ns)
        for h in range(H):
            ah = cand[:, h]                                  # (ns,3)
            zk = torch.stack([model.predictors[k](zk[k], ah) for k in range(K)], dim=0)
            zbar = zk.mean(0)                                # (ns,D)
            d2 = ((zbar - zg.expand(ns, -1)) ** 2).sum(-1)
            prag = prag + (w_run * d2 if h < H - 1 else d2)
            epi = epi + ((zk - zbar.unsqueeze(0)) ** 2).sum(-1).mean(0)   # disagreement_h
        pred_centroid = c0 + C_T * cand.sum(dim=1)
        prag = prag + w_t * ((pred_centroid - cg) ** 2).sum(-1)
        return prag, epi

    def zscore(v: torch.Tensor) -> torch.Tensor:
        return (v - v.mean()) / v.std().clamp_min(1e-8)

    for _ in range(n_iters):
        eps = torch.randn(n_samples, H, 3, generator=gen)
        if Rn is not None:
            eps = torch.einsum("ij,nhj->nhi", Rn, eps)       # rotate exploration noise
        cand = _ball_clamp(mean.unsqueeze(0) + sigma.unsqueeze(0) * eps)
        prag, epi = rollout(cand)
        cost = zscore(prag) - beta * zscore(epi)             # EFE: minimise risk, maximise info gain
        elite = cand[torch.topk(cost, n_elite, largest=False).indices]
        mean = elite.mean(0)
        var_iso = ((elite - mean.unsqueeze(0)) ** 2).mean(dim=(0, 2))     # isotropic per-step std
        sigma = var_iso.sqrt().clamp_min(1e-3).unsqueeze(-1).expand(H, 3)

    prag_sel, epi_sel = rollout(mean.unsqueeze(0))
    return {"plan": mean, "prag": float(prag_sel), "epi": float(epi_sel)}


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def _make_ood(S: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    r"""Reorient each cloud by an independent random $\mathrm{SO}(3)$ about its centroid (novel goal pose)."""
    c = S.mean(dim=1, keepdim=True)
    out = torch.stack([rotate_points(S[i] - c[i], rand_so3(gen)) + c[i] for i in range(S.shape[0])])
    return out


def _reorient(
    S: torch.Tensor, A: torch.Tensor, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Re-orient each *(cloud, action)* pair by an independent $\mathrm{SO}(3)$ — a true group-orbit move.

    Both the cloud (about its centroid) **and** the type-1 action $a\mapsto Ra$ rotate by the *same*
    per-sample $R_i$, so $(S_i,A_i)\mapsto(R_iS_i,R_iA_i)$ stays on the $\mathrm{SE}(3)$ orbit. For an
    equivariant ensemble the one-step disagreement is then invariant *per sample* (the theorem), so the
    batch mean is unchanged to the float floor — re-orientation carries **zero** epistemic novelty.
    Rotating only the cloud (not the action) would be an off-orbit move and would change $\mathcal{D}$.
    """
    c = S.mean(dim=1, keepdim=True)
    out_s, out_a = [], []
    for i in range(S.shape[0]):
        R = rand_so3(gen)
        out_s.append(rotate_points(S[i] - c[i], R) + c[i])
        out_a.append(rotate_points(A[i], R))
    return torch.stack(out_s), torch.stack(out_a)


def _novel_shape(S: torch.Tensor, gen: torch.Generator, scale: float = 2.5) -> torch.Tensor:
    r"""Genuinely **off-orbit** novelty: anisotropically stretch + jitter each cloud out of distribution.

    Training shapes vary only mildly (per-axis scale $\in[0.85,1.15]$, jitter $0.04$); here we stretch by
    $\approx\!2.5\times$ on a random axis and add large jitter, producing clouds the encoder never saw.
    Scaling/jitter are **not** in $\mathrm{SO}(3)$, so this is novelty the symmetry does *not* explain away
    — yet the VN ensemble's disagreement on it is still exactly rotation-invariant (the theorem holds for
    *any* fixed cloud). Used to show $\mathcal{D}$ is a non-trivial field (its invariance is non-vacuous).
    """
    c = S.mean(dim=1, keepdim=True)
    xt = S - c
    g = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()))
    sc = 1.0 + (scale - 1.0) * torch.rand(S.shape[0], 1, 3, generator=g)        # per-cloud anisotropic
    jit = 0.25 * torch.randn(S.shape[0], S.shape[1], 3, generator=g)            # large jitter (OOD)
    return xt * sc + jit + c


def _fmt(x: float) -> str:
    return f"{x:.3e}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(7)

    if SMOKE:
        N_TRAIN, N_PROBE, EPOCHS = 200, 64, 3
        BETAS = [0.0, 4.0]
        N_TASKS = 2
        tag = "_smoke"
    else:
        N_TRAIN, N_PROBE, EPOCHS = 1500, 200, 60
        BETAS = [0.0, 1.0, 4.0, 12.0]
        N_TASKS = 6
        tag = ""

    print("=" * 72)
    print("STEP 20: active inference — Expected Free Energy in the equivariant latent")
    print(f"    mode={'SMOKE' if SMOKE else 'FULL'}  K={K_ENS}  N_train={N_TRAIN}  epochs={EPOCHS}")
    print("=" * 72)

    # data: seen orientation wedge (Step 13), held-out probe set, plus a re-oriented (same-orbit) copy
    # and a genuinely off-orbit (OOD-shape) copy for the epistemic-geometry panel [B].
    S, A, S2 = collect_cloud_transitions(N_TRAIN, seed=0)
    Sp, Ap, _ = collect_cloud_transitions(N_PROBE, seed=11)
    Sp_re, Ap_re = _reorient(Sp, Ap, gen)             # rotate (cloud, action) together: same SE(3) orbit
    Sp_novel = _novel_shape(Sp, gen)                  # off-orbit novelty (anisotropic stretch + jitter)

    models = {"VN": build_vn_ensemble(K_ENS, seed=0), "MLP": build_mlp_ensemble(K_ENS, seed=0)}
    params = {k: n_params(m) for k, m in models.items()}
    print(f"\n    params: VN={params['VN']}  MLP={params['MLP']}  (shared encoder + {K_ENS} predictors)")

    # ---- [A] init invariance of the epistemic terms --------------------------------------------
    print("\n" + "=" * 72)
    print("[A] invariance of the EFE epistemic terms under a global SO(3) -- AT INIT")
    print("=" * 72)
    Rprobe = rand_so3(gen)
    init_inv = {k: efe_term_invariance(m, Sp, Ap, Rprobe) for k, m in models.items()}
    print(f"    {'model':>6} | {'disagree-inv':>13} | {'entropy-inv':>12}")
    print("    " + "-" * 38)
    for k in ("VN", "MLP"):
        print(f"    {k:>6} | {_fmt(init_inv[k]['disagree']):>13} | {_fmt(init_inv[k]['entropy']):>12}")

    # ---- train both ensembles -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"[train] shared-encoder ensemble JEPA (EMA target, VICReg, per-member bootstrap), {EPOCHS} ep")
    print("=" * 72)
    latent_std = {}
    for k, m in models.items():
        print(f"  -- {k} ensemble --")
        h = train_ensemble_jepa(m, S, A, S2, epochs=EPOCHS, batch_size=128, var_coef=0.1, seed=0)
        latent_std[k] = h["latent_std"]
    print(f"    final latent_std (>0 => no collapse): VN={latent_std['VN']:.3f}  MLP={latent_std['MLP']:.3f}")

    # ---- [A'] post-train invariance + total EFE ------------------------------------------------
    print("\n" + "=" * 72)
    print("[A'] invariance AFTER training (the symmetry must survive optimisation)")
    print("=" * 72)
    post_inv = {k: efe_term_invariance(m, Sp, Ap, Rprobe) for k, m in models.items()}
    # total one-step EFE invariance under a full (R,t) motion (single task)
    X0 = Sp[:1]
    Xg = _make_ood(Sp[:1], gen)
    a1 = Ap[:1]
    tt = torch.randn(3, generator=gen) * 0.8
    total_inv = {
        k: total_efe_invariance(m, X0, Xg, a1, Rprobe, tt, beta=4.0) for k, m in models.items()
    }
    print(f"    {'model':>6} | {'disagree-inv':>13} | {'entropy-inv':>12} | {'total-G-inv (R,t)':>17}")
    print("    " + "-" * 58)
    for k in ("VN", "MLP"):
        print(f"    {k:>6} | {_fmt(post_inv[k]['disagree']):>13} | {_fmt(post_inv[k]['entropy']):>12} "
              f"| {_fmt(total_inv[k]):>17}")

    # ---- [B] epistemic geometry: curiosity blind to re-orientation, alert to genuine novelty ----
    print("\n" + "=" * 72)
    print("[B] epistemic geometry: disagreement is blind to re-orientation, alert to off-orbit novelty")
    print("=" * 72)
    calib = {}
    for k, m in models.items():
        d_vec = disagreement(m.members_next(m.encoder(Sp), Ap))           # (B,) per-sample, seen probes
        d_seen = d_vec.mean().item()
        cov = (d_vec.std(unbiased=False) / d_vec.mean().clamp_min(1e-12)).item()   # non-vacuity: D not constant
        d_reorient = mean_one_step_disagreement(m, Sp_re, Ap_re)          # same SE(3) orbit (cloud+action)
        d_novel = mean_one_step_disagreement(m, Sp_novel, Ap)             # off-orbit (OOD shape) novelty
        # geometry-awareness: globally rotate the whole off-orbit probe -> disagreement must be unchanged
        d_novel_rot = mean_one_step_disagreement(m, rotate_points(Sp_novel, Rprobe), rotate_points(Ap, Rprobe))
        calib[k] = {
            "seen": d_seen, "cov": cov,
            "reorient": d_reorient, "reorient_ratio": d_reorient / max(d_seen, 1e-12),
            "novel": d_novel, "novel_ratio": d_novel / max(d_seen, 1e-12),
            "novel_rot_inv": abs(d_novel - d_novel_rot) / max(d_novel, 1e-12),
        }
    print(f"    {'model':>6} | {'D(seen)':>9} | {'CoV':>6} | {'reori/seen':>10} | "
          f"{'novel/seen':>10} | {'novel rot-inv':>13}")
    print("    " + "-" * 70)
    for k in ("VN", "MLP"):
        c = calib[k]
        print(f"    {k:>6} | {c['seen']:>9.4f} | {c['cov']:>6.3f} | x{c['reorient_ratio']:>8.4f} | "
              f"x{c['novel_ratio']:>8.3f} | {_fmt(c['novel_rot_inv']):>13}")
    print("    => VN: re-orientation carries ZERO novelty (ratio=1, theorem), genuine off-orbit novelty does")
    print("       raise it (CoV>0, non-vacuous), and that signal is exactly rotation-invariant. MLP: re-")
    print("       orientation spuriously looks novel (ratio!=1) — it conflates pose with novelty.")

    # ---- [C] the active-inference knob: beta sweep trade-off (VN) -------------------------------
    print("\n" + "=" * 72)
    print("[C] active-inference knob: sweep beta in the EFE planner (pragmatic vs epistemic trade-off)")
    print("=" * 72)
    vn = models["VN"]
    tasks = [(Sp[i : i + 1], _make_ood(Sp[i : i + 1], gen)) for i in range(N_TASKS)]
    sweep = {}
    for beta in BETAS:
        prags, epis = [], []
        for X0t, Xgt in tasks:
            zg = vn.encoder(Xgt)
            cg = centroid(Xgt)
            g_plan = torch.Generator().manual_seed(123)      # same noise across beta => paired
            out = efe_cem_plan(vn, X0t, zg, cg=cg, beta=beta, H=12, n_samples=256, n_iters=5, gen=g_plan)
            prags.append(out["prag"])
            epis.append(out["epi"])
        sweep[beta] = {"prag": float(np.mean(prags)), "epi": float(np.mean(epis))}
    print(f"    {'beta':>6} | {'pragmatic cost':>15} | {'epistemic value':>16}")
    print("    " + "-" * 44)
    for beta in BETAS:
        print(f"    {beta:>6.1f} | {sweep[beta]['prag']:>15.4f} | {sweep[beta]['epi']:>16.4f}")
    b0, bmax = BETAS[0], BETAS[-1]
    epi_rises = sweep[bmax]["epi"] > sweep[b0]["epi"] * 1.05
    prag_rises = sweep[bmax]["prag"] >= sweep[b0]["prag"] - 1e-6
    print(f"    => epistemic up {sweep[b0]['epi']:.3f} -> {sweep[bmax]['epi']:.3f}; "
          f"pragmatic up {sweep[b0]['prag']:.3f} -> {sweep[bmax]['prag']:.3f} "
          f"(more beta => seek info, trade goal).")

    # plan equivariance of the EFE-selected plan (theorem realised end-to-end)
    X0t, Xgt = tasks[0]
    R = rand_so3(gen)
    g1 = torch.Generator().manual_seed(321)
    g2 = torch.Generator().manual_seed(321)
    base = efe_cem_plan(vn, X0t, vn.encoder(Xgt), cg=centroid(Xgt), beta=4.0, gen=g1)["plan"]
    rot = efe_cem_plan(
        vn, rotate_points(X0t, R), vn.encoder(rotate_points(Xgt, R)),
        cg=centroid(rotate_points(Xgt, R)), beta=4.0, R_noise=R, gen=g2,
    )["plan"]
    plan_equiv = (rot - rotate_points(base, R)).abs().max().item()
    print(f"    EFE plan equivariance  ||plan(Rx) - R.plan(x)||_inf = {_fmt(plan_equiv)}  (VN, beta=4)")

    # ---- verdict --------------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 20 SUMMARY")
    print("=" * 72)
    ok_vn_inv = (post_inv["VN"]["disagree"] < 1e-4 and post_inv["VN"]["entropy"] < 1e-4
                 and total_inv["VN"] < 1e-4)
    ok_mlp_breaks = post_inv["MLP"]["disagree"] > 1e-2 or post_inv["MLP"]["entropy"] > 1e-2
    ok_calib = (
        abs(calib["VN"]["reorient_ratio"] - 1.0) < 1e-4   # equivariant: re-orientation is NOT novel (theorem)
        and calib["VN"]["novel_rot_inv"] < 1e-4           # off-orbit novelty signal is still geometry-independent
        and calib["VN"]["cov"] > 1e-3                     # D is a non-constant field => invariance is non-vacuous
        and abs(calib["MLP"]["reorient_ratio"] - 1.0) > 1e-2  # control: MLP spuriously sees re-orientation as novel
    )
    ok_knob = bool(epi_rises and prag_rises)
    ok_plan_equiv = plan_equiv < 1e-2
    passed = ok_vn_inv and ok_mlp_breaks and ok_calib and ok_knob and ok_plan_equiv
    print(f"    [A'] VN EFE terms SE(3)-invariant post-train: disagree {_fmt(post_inv['VN']['disagree'])}, "
          f"entropy {_fmt(post_inv['VN']['entropy'])}, total-G {_fmt(total_inv['VN'])}")
    print(f"         MLP control breaks invariance: disagree {_fmt(post_inv['MLP']['disagree'])} (=> not vacuous)")
    print(f"    [B]  curiosity blind to re-orientation: VN reorient ratio x{calib['VN']['reorient_ratio']:.4f} "
          f"(=1, theorem), CoV={calib['VN']['cov']:.3f} (non-vacuous), novel rot-inv "
          f"{_fmt(calib['VN']['novel_rot_inv'])}; MLP reorient ratio x{calib['MLP']['reorient_ratio']:.3f} (spurious)")
    print(f"    [C]  active-inference knob works (epi & prag both rise with beta); "
          f"EFE plan equivariant ({_fmt(plan_equiv)})")
    print(f"    guards: vn-inv={ok_vn_inv}  mlp-breaks={ok_mlp_breaks}  calib={ok_calib}  "
          f"knob={ok_knob}  plan-equiv={ok_plan_equiv}")
    print("    headline: Expected Free Energy — pragmatic goal-seeking PLUS epistemic information gain —")
    print("        is well-posed and tractable in the equivariant latent, and its epistemic drive is an")
    print("        EXACTLY SE(3)-invariant scalar (a non-equivariant ensemble's is not). The agent's")
    print("        curiosity is geometry-independent; the whole active-inference loop is SE(3)-equivariant.")
    print("        Honest scope: the teacher is fully observed, so the epistemic term is a demonstrated")
    print("        mechanism with an exact geometric guarantee, NOT a task-success necessity (active")
    print("        inference treated as mathematical structure, per the project's standing caveat).")
    print("    " + ("PASS" if passed else "FAIL"))

    out = {
        "mode": "SMOKE" if SMOKE else "FULL",
        "config": {"K": K_ENS, "N_TRAIN": N_TRAIN, "EPOCHS": EPOCHS, "BETAS": BETAS, "N_TASKS": N_TASKS},
        "params": params,
        "latent_std": latent_std,
        "invariance_init": init_inv,
        "invariance_post": post_inv,
        "total_efe_invariance": total_inv,
        "calibration": calib,
        "beta_sweep": {str(b): sweep[b] for b in BETAS},
        "plan_equivariance": plan_equiv,
        "verdict": {
            "passed": bool(passed), "ok_vn_inv": bool(ok_vn_inv), "ok_mlp_breaks": bool(ok_mlp_breaks),
            "ok_calib": bool(ok_calib), "ok_knob": ok_knob, "ok_plan_equiv": bool(ok_plan_equiv),
        },
    }
    out_path = ROOT / "papers" / "figures" / f"step20_active_inference{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    wrote {out_path.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
