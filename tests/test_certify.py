r"""Unit tests for the runnable certificate procedure (Algorithm 1, ``src/certify.py``)."""

import math

import pytest

from src.certify import certify


def test_contractive_channels_are_certified_to_all_horizons():
    cert = certify({"g": 1e-7}, [1.0, 0.5, 0.75], eps=0.01)
    assert cert.config_certified                       # residual below tolerance
    assert cert.n_unbounded == 3                       # all sigma<=1 => lambda<=0 => infinite horizon
    assert all(math.isinf(h) for h in cert.horizons)
    assert cert.min_finite_horizon is None


def test_expansive_channel_obeys_theorem_B_log_law():
    sigma = 2.0                                        # chaotic channel, lambda = ln 2
    cert = certify({"g": 1e-7}, [1.0, sigma], eps=0.01)
    expected = math.log(1.0 / 0.01) / math.log(sigma)  # T_j(eps) = log(1/eps)/lambda_j
    assert cert.n_unbounded == 1                       # the sigma=1 channel
    assert cert.min_finite_horizon == pytest.approx(expected, rel=1e-9)


def test_horizon_grows_logarithmically_as_eps_coarsens():
    sv = [1.0, 2.0]
    fine = certify({"g": 1e-7}, sv, eps=0.001).min_finite_horizon
    coarse = certify({"g": 1e-7}, sv, eps=0.1).min_finite_horizon
    assert coarse < fine                               # coarser tolerance => longer certified horizon
    # the difference is exactly the log-law slope 1/lambda
    assert (fine - coarse) == pytest.approx(math.log(0.1 / 0.001) / math.log(2.0), rel=1e-9)


def test_configuration_axis_gate_on_residual():
    assert certify({"g1": 5e-4, "g2": 8e-4}, [0.5], eps=0.01).config_certified        # both below 1e-3
    assert not certify({"g1": 5e-4, "g2": 2e-3}, [0.5], eps=0.01).config_certified    # one above 1e-3
    assert certify({"g": 1e-7}, [0.5], eps=0.01).n_generators == 1


def test_eps_out_of_range_raises():
    with pytest.raises(ValueError):
        certify({"g": 1e-7}, [0.5], eps=0.0)
    with pytest.raises(ValueError):
        certify({"g": 1e-7}, [0.5], eps=1.0)
