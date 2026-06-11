r"""wma-step2 — instrument upgrades: echo excess, imprint ratio, derangement (pre-registered).

Spec: docs/specs/2026-06-11-wma-step2-tdmpc2-imprint-seed.md (v1.2 readout taxonomy).

The discriminator test encodes the v1.1/v1.2 amendments as executable math:

- **Perfect counterfactual model** (action-revealing world, $F$ = true map):
  $\eta \approx 0$ AND $\rho_{\mathrm{imp}} \approx 1$ AND $L_{\mathrm{sym}}$ small —
  decoding the fed action from a correct counterfactual is COMPETENCE, not pathology.
  ($\rho$ alone is non-discriminative — the reason the v1.1 amendment exists.)
- **Echo model in an action-HIDING world** ($B = 0$: actions never touch true dynamics;
  predictor injects $2a$ verbatim): $\eta$ large — manufactured surplus (the LeWM regime).
- **Shortcut code in an action-revealing world** (G1c, tests/test_wma_step1.py):
  $\eta$ vacuous-by-ceiling but $L_{\mathrm{sym}}$ explodes — the third regime; covered there.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit.atm import atm_audit, derangement, echo_excess, imprint_ratio  # noqa: E402
from tests.test_wma_step1 import FAST, _linear_world  # noqa: E402


# ---------------------------------------------------------------------------
# derangement
# ---------------------------------------------------------------------------

def test_derangement_no_fixed_points_and_deterministic():
    perm1, _ = derangement(500, seed=7)
    perm2, _ = derangement(500, seed=7)
    perm3, _ = derangement(500, seed=8)
    assert torch.equal(perm1, perm2)
    assert not torch.equal(perm1, perm3)
    assert (perm1 != torch.arange(500)).all(), "derangement must have no fixed points"
    assert sorted(perm1.tolist()) == list(range(500)), "must be a permutation"


def test_derangement_cross_episode_fraction():
    ep_ids = np.repeat(np.arange(20), 25)                      # 20 episodes x 25 transitions
    perm, cross = derangement(500, seed=2026, ep_ids=ep_ids)
    assert cross is not None and cross > 0.9
    assert cross == pytest.approx(float((ep_ids[perm.numpy()] != ep_ids).mean()))


# ---------------------------------------------------------------------------
# closed-form readouts
# ---------------------------------------------------------------------------

def test_imprint_ratio_closed_form():
    # LeWM step1 reference: (1-0.445)/(1-0.462)
    assert imprint_ratio(0.462, 0.445) == pytest.approx((1 - 0.445) / (1 - 0.462))
    assert imprint_ratio(0.5, 1.0) == pytest.approx(0.0)       # grounded: perm not decodable
    assert imprint_ratio(0.95, 0.5) is None                    # NO-SIGNAL above threshold


def test_echo_excess_closed_form():
    d_norm = {("T", "T"): 0.936, ("T", "P"): 1.578, ("P", "T"): 3.654, ("P", "P"): 0.462}
    assert echo_excess(d_norm) == pytest.approx(0.474)         # LeWM step1 reference


# ---------------------------------------------------------------------------
# three-arm discriminator (the v1.1/v1.2 amendments as executable math)
# ---------------------------------------------------------------------------

def _audit_pair(z, z_next, z_pred_fn, a, ep_ids):
    """Run the real-action audit and the derangement-control audit; return both."""
    res_real = atm_audit(z, z_next, z_pred_fn(a), a, **FAST)
    perm, _ = derangement(z.shape[0], seed=2026, ep_ids=ep_ids)
    a_perm = a[perm]
    res_perm = atm_audit(z, z_next, z_pred_fn(a_perm), a_perm, **FAST)
    return res_real, res_perm


def test_perfect_model_is_competent_not_pathological():
    z, a, z_next, A, B = _linear_world()
    ep_ids = np.repeat(np.arange(40), 100)
    res_real, res_perm = _audit_pair(z, z_next, lambda act: z @ A.T + act @ B.T, a, ep_ids)
    eta = echo_excess(res_real.D_norm)
    rho = imprint_ratio(res_real.D_norm[("P", "P")], res_perm.D_norm[("P", "P")])
    assert abs(eta) < 0.15, f"perfect model must have ~zero echo excess, eta={eta}"
    assert rho is not None and rho > 0.7, f"counterfactual competence: rho={rho}"
    assert res_real.readouts["L_sym"] < 0.5


def test_echo_model_in_action_hiding_world_has_large_excess():
    # B = 0: actions never touch the true dynamics (the LeWM/PushT regime, distilled);
    # the predictor manufactures action info by injecting 2a into the last coordinates.
    g = torch.Generator().manual_seed(3)
    dz, da, n = 16, 4, 4000
    A = 0.9 * torch.linalg.qr(torch.randn(dz, dz, generator=g, dtype=torch.float64))[0].float()
    z = torch.randn(n, dz, generator=g)
    a = torch.rand(n, da, generator=g) * 2.0 - 1.0
    z_next = z @ A.T + 0.05 * torch.randn(n, dz, generator=g)      # actions hidden
    ep_ids = np.repeat(np.arange(40), 100)

    def echo_pred(act):
        return torch.cat([(z @ A.T)[:, :-da], 2.0 * act], dim=1)

    res_real, res_perm = _audit_pair(z, z_next, echo_pred, a, ep_ids)
    eta = echo_excess(res_real.D_norm)
    rho = imprint_ratio(res_real.D_norm[("P", "P")], res_perm.D_norm[("P", "P")])
    assert res_real.D_norm[("T", "T")] > 0.9, "world must hide actions from T"
    assert eta > 0.5, f"manufactured surplus expected, eta={eta}"
    assert rho is not None and rho > 0.7, f"echo tracks any fed action, rho={rho}"
