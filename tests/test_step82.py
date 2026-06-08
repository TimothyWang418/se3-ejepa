r"""Guard for Step 82 — certified predictability horizon FROM the (learned) model.

Fast, training-free unit tests for the cone / adapted-metric certificate (Theorem B', see
``experiments/step82_certified_horizon_from_model.py`` and docs/specs/2026-06-08-certified-horizon-from-model-design.md).
Phase A covers the rigorous certificate on the TRUE Henon map (exact Jacobian-Lipschitz constant $L_J=2.8$):
the Henon Jacobian (A1), the Theorem-B' closed form ``t_guar`` (A2), the constant adapted-metric solve (A3), the
Lipschitz sample->continuum bridge (A4), and the make-or-break ``run_true_henon`` wiring + Gate G1 (A5).
"""
import math

import numpy as np
from experiments import step82_certified_horizon_from_model as s82


def test_henon_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.uniform(-1.0, 1.0, size=2)
        J = s82.henon_jac(s)                       # analytic 2x2
        fd = np.zeros((2, 2))
        eps = 1e-6
        for j in range(2):
            sp = s.copy(); sp[j] += eps
            sm = s.copy(); sm[j] -= eps
            fd[:, j] = (s82.henon_map(sp) - s82.henon_map(sm)) / (2 * eps)
        assert np.max(np.abs(J - fd)) < 1e-7


def test_henon_jacobian_lipschitz_is_exact():
    # D phi depends only on x via the (0,0) entry = -2a x ; Lip = 2a = 2.8
    assert abs(s82.HENON_JAC_LIP - 2.8) < 1e-12


def test_t_guar_matches_closed_form_and_is_monotone():
    Lam, kappa, eps_res = 1.5, 4.0, 1.0
    expected = math.floor((math.log(eps_res / 0.01) - 0.5 * math.log(kappa)) / math.log(Lam))
    assert s82.t_guar(Lam, kappa, 0.01, eps_res) == expected
    # smaller eps (sharper resolution demand at fixed eps_res) => longer horizon
    assert s82.t_guar(Lam, kappa, 1e-4, eps_res) >= s82.t_guar(Lam, kappa, 1e-2, eps_res)
    # Lambda = 1 (no expansion) => unbounded horizon sentinel
    assert s82.t_guar(1.0, 1.0, 0.01, 1.0) == s82.HORIZON_INF


def test_adapted_metric_on_symmetric_matrix_recovers_spectral_norm():
    # For a single symmetric matrix, the optimal constant metric gives Lambda = sigma_max = |max eig|.
    A = np.array([[1.3, 0.0], [0.0, 0.7]], dtype=np.float64)
    jacs = np.stack([A, A, A])
    P, Lam, _ = s82.adapted_metric(jacs)
    assert abs(Lam - 1.3) < 1e-2
    # P is SPD
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_adapted_metric_beats_euclidean_on_rotated_expansion():
    # A non-normal matrix: Euclidean op-norm overestimates; the adapted metric tightens toward rho(A)=1.2.
    A = np.array([[1.2, 5.0], [0.0, 1.2]], dtype=np.float64)
    jacs = np.stack([A, A])
    P, Lam, _ = s82.adapted_metric(jacs)
    assert Lam < np.linalg.norm(A, 2) - 1e-3        # strictly better than Euclidean
    assert Lam >= 1.2 - 1e-2                          # cannot beat the spectral radius


def test_lipschitz_bridge_inflates_lambda_and_is_sound():
    # Continuum-certified Lambda = Lambda_samples + sqrt(kappa) * L_J * h
    out = s82.lipschitz_bridge(lambda_samples=1.30, kappa=4.0, L_J=2.8, h=0.01)
    assert abs(out["lambda_cert"] - (1.30 + 2.0 * 2.8 * 0.01)) < 1e-12   # sqrt(4)=2
    assert out["lambda_cert"] > 1.30                                      # bridge only inflates (sound)


def test_lipschitz_bridge_vacuous_when_slack_dominates():
    # Huge grid spacing => Lambda_cert explodes => horizon vacuous => certified=False
    out = s82.lipschitz_bridge(lambda_samples=1.30, kappa=4.0, L_J=2.8, h=10.0,
                               eps=0.01, eps_res=1.0)
    assert out["certified"] is False


def test_run_true_henon_smoke_is_sound_and_non_vacuous():
    # n_samples=2000: the make-or-break Gate G1 (non-vacuous AND beats-Euclidean) passes here. G1 depends on the
    # covering radius h via the bridge slack sqrt(kappa)*L_J*h: an under-sampled cloud (n<=1500) has h~0.06-0.09 so the
    # inflated Lambda_cert loses to the Euclidean bound (beats_euclidean=False) -- a real covering-radius effect, NOT a
    # loosening. By n=2000 (h~0.04) the adapted metric beats Euclidean and the certificate is non-vacuous; the full
    # run() default n=4000 is denser still. This is a fast (~3s) wiring smoke of that PASS, not a re-tuned threshold.
    out = s82.run_true_henon(n_samples=2000, seed=0, eps=0.01, eps_res=1.0)
    # certificate is sound by construction; on the TRUE map at adequate resolution it must be non-vacuous (G1)
    assert out["certified"] is True
    assert out["t_guar"] >= 1
    # certified exponent brackets the textbook Henon exponent from ABOVE (sound: log Lambda_cert >= lambda_1)
    assert math.log(out["lambda_cert"]) >= 0.419 - 0.05


# =============================================================================================================== #
# Phase A' — the TIGHT certificate on the uniformly-hyperbolic cat map. Tests stay fast and training-free.
# =============================================================================================================== #

# --- A'-1: linear cat map -------------------------------------------------------------------------------------- #
def test_cat_jacobian_is_constant_A():
    # The cat-map Jacobian is the constant matrix A = [[2,1],[1,1]] at every torus point (the mod-1 wrap is a
    # translation and contributes nothing to Dphi).
    A = np.array([[2.0, 1.0], [1.0, 1.0]])
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = rng.uniform(0.0, 1.0, size=2)
        assert np.max(np.abs(s82.cat_jac(s) - A)) < 1e-14


def test_cat_jacobian_matches_finite_difference_interior():
    # Finite-diff at interior points (away from the mod-1 seam) recovers A.
    rng = np.random.default_rng(1)
    for _ in range(10):
        s = rng.uniform(0.2, 0.8, size=2)          # interior: no wrap within +-1e-6
        fd = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            sp = s.copy(); sp[j] += h
            sm = s.copy(); sm[j] -= h
            fd[:, j] = (s82.cat_map(sp) - s82.cat_map(sm)) / (2 * h)
        assert np.max(np.abs(s82.cat_jac(s) - fd)) < 1e-7


def test_cat_lambda1_is_analytic_log_golden():
    # lambda_1 = log((3+sqrt5)/2) exactly.
    assert abs(s82.CAT_LAMBDA1 - math.log((3.0 + math.sqrt(5.0)) / 2.0)) < 1e-15


def test_cat_lambda1_matches_benettin_estimate():
    # A hand-rolled Benettin power-iteration estimate of lambda_1 matches the analytic value to ~1e-6
    # (constant Jacobian => the running log-stretch converges to log rho(A) with no fluctuation).
    est = s82.benettin_lambda1(lambda z: s82.cat_jac(z), s82.cat_map, s0=np.array([0.3, 0.7]),
                               n_steps=3000, warmup=100, seed=0)
    assert abs(est - s82.CAT_LAMBDA1) < 1e-5


# --- A'-2: perturbed cat map + cone margin --------------------------------------------------------------------- #
def test_perturbed_cat_jacobian_matches_finite_difference():
    delta = 0.1
    rng = np.random.default_rng(2)
    for _ in range(15):
        s = rng.uniform(0.15, 0.85, size=2)        # interior to dodge the seam
        J = s82.perturbed_cat_jac(s, delta)
        fd = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            sp = s.copy(); sp[j] += h
            sm = s.copy(); sm[j] -= h
            fd[:, j] = (s82.perturbed_cat_map(sp, delta) - s82.perturbed_cat_map(sm, delta)) / (2 * h)
        assert np.max(np.abs(J - fd)) < 1e-7


def test_perturbed_cat_is_area_preserving():
    # det Dphi == 1 for every x and any delta (the perturbation only shears).
    rng = np.random.default_rng(3)
    for delta in (0.05, 0.1, 0.3):
        for _ in range(8):
            s = rng.uniform(0.0, 1.0, size=2)
            assert abs(np.linalg.det(s82.perturbed_cat_jac(s, delta)) - 1.0) < 1e-12


def test_perturbed_cat_jac_lipschitz_is_analytic_and_dominates_local_slopes():
    delta = 0.1
    L = s82.perturbed_cat_jac_lipschitz(delta)
    assert abs(L - 2.0 * math.sqrt(2.0) * math.pi * delta) < 1e-12     # analytic value
    # The bound must dominate every empirical finite-difference slope ||Dphi(z)-Dphi(z')|| / ||z-z'|| on a grid.
    xs = np.linspace(0.0, 1.0, 200, endpoint=False)
    worst = 0.0
    for i in range(len(xs) - 1):
        za = np.array([xs[i], 0.3]); zb = np.array([xs[i + 1], 0.3])
        slope = np.linalg.norm(s82.perturbed_cat_jac(za, delta) - s82.perturbed_cat_jac(zb, delta), 2) \
            / np.linalg.norm(za - zb)
        worst = max(worst, slope)
    assert worst <= L + 1e-9                       # sound: the analytic L_J upper-bounds all sampled slopes


def test_cone_margin_positive_on_catmap_negative_on_nonhyperbolic():
    # Cat map: a single golden-direction cone is forward-invariant and uniformly expanding => margin > 0.
    rng = np.random.default_rng(4)
    cat_jacs = np.stack([s82.cat_jac(rng.uniform(0, 1, size=2)) for _ in range(50)])
    assert s82.cone_margin(cat_jacs) > 0.0
    # Perturbed cat (delta=0.1): still Anosov => margin > 0.
    pert_jacs = np.stack([s82.perturbed_cat_jac(rng.uniform(0, 1, size=2), 0.1) for _ in range(50)])
    assert s82.cone_margin(pert_jacs) > 0.0
    # A rotation by 90 degrees has NO invariant cone (it rotates every direction) and is not expanding => margin <= 0.
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert s82.cone_margin(np.stack([R, R, R])) <= 0.0
    # An identity set is non-expanding (margin <= 0): no uniform-hyperbolic expansion.
    assert s82.cone_margin(np.stack([np.eye(2)] * 3)) <= 0.0


# --- A'-3: torus horizon + the tight runs ---------------------------------------------------------------------- #
def test_true_horizon_torus_uses_mod1_distance():
    # Two points 0.02 apart on the circle but 0.98 apart in raw coords must register as CLOSE (toroidal), so a
    # stationary identity map never "crosses" eps_res=0.4.
    out = s82.true_horizon_torus(lambda z: z % 1.0, eps=0.01, eps_res=0.4, n_starts=50, seed=0, max_t=20)
    assert out == 20.0                              # identity: never separates => pinned at the budget


def test_run_true_catmap_is_TIGHT_and_sound():
    # THE tight anchor. Linear cat map: P=I, kappa=1, L_J=0 => certified exponent == lambda_1 to machine precision.
    out = s82.run_true_catmap(n_samples=1500, seed=0, eps=0.01, eps_res=0.4)
    # Tightness gate (the point of A'): ratio <= 1.02 (analytic, essentially exact).
    assert out["tightness_ratio"] <= 1.02
    assert abs(out["log_lambda_cert"] - s82.CAT_LAMBDA1) < 1e-6      # exact to machine precision
    # Symmetric A => optimal P = I => kappa = 1. The Nelder-Mead solve converges to P=I within xatol=1e-8, so kappa is
    # 1 to ~5 decimals (a tiny optimizer residual, NOT a real ill-conditioning); the certified exponent above is exact
    # regardless because it is the honest op-norm of the returned P. A 1e-4 tolerance asserts "P = I".
    assert abs(out["kappa"] - 1.0) < 1e-4
    # Soundness: certified exponent is an UPPER bound on lambda_1; T_guar <= T_true on the torus.
    assert out["sound_exponent"] is True
    assert out["t_guar"] <= out["t_true"]
    # Uniform hyperbolicity verified from the geometry.
    assert out["anosov"] is True


def test_run_true_perturbed_catmap_is_anosov_and_near_tight():
    # The nonlinear upgrade: still Anosov, tightness ratio <= 1.3 (small analytic L_J slack).
    out = s82.run_true_perturbed_catmap(delta=0.1, n_samples=1500, seed=0, eps=0.01, eps_res=0.4)
    assert out["anosov"] is True
    assert out["cone_margin"] > 0.0
    assert out["tightness_ratio"] <= 1.3
    assert out["sound_exponent"] is True            # log Lambda_cert >= its own lambda_1
    assert out["t_guar"] <= out["t_true"]


def test_tightness_comparison_cat_tight_henon_loose():
    cmp = s82.tightness_comparison(n_samples=2000, seed=0, eps=0.01)
    cat = cmp["CatMap(linear)"]
    hen = cmp["Henon(true)"]
    assert cat["tightness_ratio"] <= 1.02           # cat map is tight
    assert hen["tightness_ratio"] >= 2.0            # Henon is the loose (sound-but-conservative) companion
    assert cat["anosov"] is True and hen["anosov"] is False


# =============================================================================================================== #
# Phase B — the contribution: the SAME certificate read off a LEARNED model's Jacobian field. Tests stay fast and
# training-free (the autograd-Jacobian, the sound net-Lipschitz bound, the bootstrap fallback, and the routing); the
# heavy training-and-certify runs live in run_learned_henon / run_learned_catmap, smoke-tested with tiny nets.
# =============================================================================================================== #
import torch  # noqa: E402


# --- B1: autograd Jacobian of a learned map -------------------------------------------------------------------- #
def test_learned_jacobian_matches_finite_difference():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).double()
    z = torch.tensor([0.1, -0.2], dtype=torch.float64)
    J = s82.learned_jacobian(net, z)                        # (2,2) numpy
    fd = np.zeros((2, 2)); e = 1e-6
    base = net(z).detach().numpy()
    for j in range(2):
        zp = z.clone(); zp[j] += e
        fd[:, j] = (net(zp).detach().numpy() - base) / e
    assert np.max(np.abs(J - fd)) < 1e-4


# --- B2: a SOUND net Jacobian-Lipschitz bound from layer spectral norms ---------------------------------------- #
def test_net_jacobian_lipschitz_upper_bounds_empirical_on_samples():
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).double()
    Lhat = s82.net_jacobian_lipschitz(net)
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        a = torch.tensor(rng.uniform(-1, 1, 2), dtype=torch.float64)
        b = a + torch.tensor(rng.uniform(-1e-3, 1e-3, 2), dtype=torch.float64)
        Ja, Jb = s82.learned_jacobian(net, a), s82.learned_jacobian(net, b)
        worst = max(worst, np.linalg.norm(Ja - Jb, 2) / max(np.linalg.norm((a - b).numpy()), 1e-12))
    assert Lhat >= worst - 1e-9     # sound: the analytic bound dominates every sampled local slope


# --- B3: the step78 block-bootstrap hybrid fallback ------------------------------------------------------------ #
def test_bootstrap_fallback_returns_a_horizon_interval():
    # synthetic chaotic logR (one positive channel) -> a finite horizon interval
    rng = np.random.default_rng(0)
    logR = rng.normal(0.4, 0.1, size=(2000, 2)) * np.array([1.0, -1.0])  # ch0 expand, ch1 contract
    out = s82.bootstrap_fallback(logR, dt_map=1.0, eps=0.01)
    assert out["t_lo"] >= 1 and out["t_hi"] >= out["t_lo"]
    assert out["lambda1_hi"] >= out["lambda1"]    # upper CB used for the conservative horizon


# --- B4: run_learned_henon (train step71 MLP -> certificate or hybrid fallback) -------------------------------- #
def test_run_learned_henon_smoke_routes_and_is_sound():
    out = s82.run_learned_henon(n_samples=300, seed=0, eps=0.01, eps_res=1.0, smoke=True)
    assert out["route"] in ("cone", "bootstrap")          # certifies or falls back, never crashes
    assert out["t_guar"] >= 1


def test_true_horizon_and_soundness_checker():
    th = s82.true_horizon(s82.henon_map, eps=0.01, eps_res=1.0, n_starts=20, seed=0)
    assert th >= 1
    # soundness checker: a deliberately too-long certified horizon is flagged as a violation
    assert s82.is_sound(t_guar=th + 5, t_true=th) is False
    assert s82.is_sound(t_guar=max(1, th - 1), t_true=th) is True


# --- The tight-from-the-learned-model showpiece: certificate on a LEARNED cat map ------------------------------ #
def test_run_learned_catmap_is_near_tight_and_sound():
    # Trains a small MLP on the lifted linear cat map (~3 s). The certificate read off the LEARNED Jacobian field must
    # stay SOUND (cert is an upper bound; T_guar <= T_true) and NEAR-TIGHT -- the net learns ~A, so the bridge only
    # mildly inflates the exponent. We assert ratio <= 1.5 (far below the learned-Henon ~8-9x), the honest expectation.
    out = s82.run_learned_catmap(n_samples=1200, seed=0, eps=0.01, eps_res=0.4)
    assert out["route"] in ("cone", "bootstrap")            # certifies or falls back; report which
    assert out["one_step_relmse"] < 1e-2                    # the net actually learned A s
    assert out["sound_exponent"] is True                   # log Lambda_cert >= lambda_1 (sound upper bound)
    assert out["t_guar"] <= out["t_true"]                  # T_guar <= T_true on the torus (sound horizon)
    assert out["tightness_ratio"] <= 1.5                   # near-tight: survives the learned-model transfer
    # the learned-model tightness is far better than the learned Henon's (sanity on the headline claim)
    assert out["tightness_ratio"] < 2.0


def test_full_step_jacobian_matches_finite_difference():
    # The chain-rule Jacobian of the un-normalized learned map phi(s)=denorm(model(norm(s))) must match a finite
    # difference of that same composed map (validates _full_step_jacobian's diag(sd)*Dmodel*diag(1/sd) factoring).
    net, mu, sd = s82._train_henon_mlp(seed=0, smoke=True)
    rng = np.random.default_rng(0)
    for _ in range(8):
        s = rng.uniform(-0.3, 0.3, size=2)
        J = s82._full_step_jacobian(net, mu, sd, s)
        fd = np.zeros((2, 2)); e = 1e-6

        def phi(z):  # un-normalized one-step map: denorm(model(norm(z)))
            zt = torch.tensor((z - mu) / sd, dtype=torch.float64)
            return net(zt).detach().numpy() * sd + mu

        base = phi(s)
        for j in range(2):
            sp = s.copy(); sp[j] += e
            fd[:, j] = (phi(sp) - base) / e
        assert np.max(np.abs(J - fd)) < 1e-4
