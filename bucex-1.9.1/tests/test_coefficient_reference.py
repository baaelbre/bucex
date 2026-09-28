"""Numerical targets and long-record regression for the GEV coefficient fix."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, gumbel_r
import bucex as bx
from bucex.inference.fit.coefficient_reference import coefficient_reference
from bucex.inference.fit.conditional_margin import ConditionalMargin
from bucex.inference.fit.continuous import gaussian_reference, reference_slice
from bucex.inference.fit import fs_utils as fs


def make_reference(X, y, margin, obs, pm, L, controls=None):
    controls = controls or bx.Laplace()
    pseudo, variance = fs._ncp_laplace_pseudo_data(
        y, y, SimpleNamespace(obs=margin), obs,
        curvature_floor=controls.curvature_floor,
        maximum_variance=controls.maximum_variance, shift_limit=10*np.max(obs['sigma']))
    old = gaussian_reference(X, pseudo, variance, pm, L)
    new, metrics = coefficient_reference(X, y, margin, obs, pm, L, old, controls)
    return old, new, metrics


@pytest.mark.parametrize('xi', [-.45, 0., .45])
@pytest.mark.parametrize('copula', [False, True])
def test_exact_slice_matches_gev_quadrature_including_copula(xi, copula):
    y = np.array([-.4, .1, .7, 1.2, .3])
    X, pm, L = np.ones((len(y), 1)), np.zeros(1), np.eye(1)
    obs = dict(sigma=1., xi=xi)
    margin = ConditionalMargin(bx.GEV(), np.linspace(-.2, .6, len(y)) if copula else 0.,
                               .6 if copula else 1.)
    # Even an unfinished optimization must retain the exact posterior target.
    _, (mean, root, precision), _ = make_reference(
        X, y, margin, obs, pm, L, bx.Laplace(max_iterations=1))
    def logtarget(w):
        return np.sum(margin.logpdf(y, np.full_like(y, w[0]), obs)) - .5*(w@w)
    def correction(w):
        d = w-mean
        return logtarget(w) + .5*(d@precision@d)
    def density(a):
        return float(np.exp(logtarget(np.array([a]))))
    mass = quad(density, -7, 7, epsabs=1e-10)[0]
    truth = quad(lambda a: a*density(a), -7, 7, epsabs=1e-10)[0]/mass
    variance = quad(lambda a: (a-truth)**2*density(a), -7, 7, epsabs=1e-10)[0]/mass
    rng, current = np.random.default_rng(167), np.zeros(1)
    values = []
    for i in range(2400):
        current, _ = reference_slice(current, mean, root, correction, rng)
        if i >= 400:
            values.append(current[0])
    assert np.mean(values) == pytest.approx(truth, abs=.045)
    assert np.var(values) == pytest.approx(variance, rel=.15)


def test_long_record_reference_is_near_conditional_mode_and_moves():
    rng = np.random.default_rng(417)
    X = np.column_stack([np.ones(1200), rng.normal(size=(1200, 6))])
    y = X @ np.array([2., .2, -.1, 0., .3, 0., -.2]) + gumbel_r.rvs(
        size=len(X), random_state=rng)
    pm, L = np.zeros(7), np.diag([20., 1., 1., 1., 1., 1., 1.])
    margin, obs = ConditionalMargin(bx.GEV(), 0., 1.), dict(sigma=1., xi=0.)
    old, new, metrics = make_reference(X, y, margin, obs, pm, L)
    mean, root, precision = new
    assert metrics['coefficient_reference_fallback'] == 0
    assert metrics['coefficient_reference_converged'] == 1
    assert np.linalg.norm(np.linalg.solve(root, old[0]-mean)) > 5
    np.testing.assert_allclose(root @ root.T @ precision, np.eye(7), atol=1e-10)
    def correction(w):
        delta = w-mean
        return np.sum(margin.logpdf(y, X@(pm+L@w), obs)) - .5*(w@w) + .5*(delta@precision@delta)
    current, values = mean.copy(), []
    for _ in range(400):
        current, _ = reference_slice(current, mean, root, correction, rng)
        values.append(current[0])
    assert np.corrcoef(values[:-1], values[1:])[0, 1] < .4


@pytest.mark.parametrize('xi,wrong_center', [(-.4, -20.), (.4, 20.)])
def test_deterministic_support_repair_does_not_need_current_coefficients(xi, wrong_center):
    X, y = np.ones((5, 1)), np.array([-.5, 0., .4, .8, 1.])
    margin, obs = ConditionalMargin(bx.GEV(), .2, .8), dict(sigma=1., xi=xi)
    initial = (np.array([wrong_center]), np.eye(1), np.eye(1))
    args = (X, y, margin, obs, np.zeros(1), np.eye(1), initial, bx.Laplace())
    first, metrics = coefficient_reference(*args)
    second, _ = coefficient_reference(*args)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert metrics['coefficient_reference_support_repaired'] == 1
    assert metrics['coefficient_reference_fallback'] == 0
    assert np.all(np.isfinite(margin.logpdf(y, X@first[0], obs)))
