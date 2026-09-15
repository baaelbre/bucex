"""Full FS/copula posterior against direct Gaussian integration + quadrature.

The reference integrates regression coefficients analytically, enumerates the
four zero/fixed slope combinations, then integrates both observation scales.
It uses neither the FS design builder nor any sampler's likelihood helpers.
"""
from itertools import product

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import bucex as bx


def reference(y, R, nodes=48):
    T = len(y)
    grid, weights = np.polynomial.legendre.leggauss(nodes)
    grid = -.1 + 2.3*grid  # log sigma in [-2.4, 2.2]
    weights = weights*2.3
    records = []
    for a, b in product((0, 1), repeat=2):
        X = np.zeros((2*T, 2+a+b))
        X[0::2,0], X[1::2,1] = 1, 1
        variance = [1.5**2, 1.5**2]
        if a:
            X[0::2,2] = np.arange(1,T+1); variance.append(.2**2)
        if b:
            X[1::2,2+a] = np.arange(1,T+1); variance.append(.2**2)
        V = np.diag(variance)
        latent = X @ V @ X.T
        for i, l1 in enumerate(grid):
            for j, l2 in enumerate(grid):
                sd = np.exp([l1,l2])
                covariance = np.kron(np.eye(T), R*sd[:,None]*sd[None,:]) + latent
                logweight = multivariate_normal.logpdf(y.ravel(), cov=covariance)
                logweight += np.log((.3,.7)[a])+np.log((.3,.7)[b])
                logweight += -4*(l1+l2) - np.exp(-2*l1) - np.exp(-2*l2)
                logweight += np.log(weights[i]*weights[j])
                mean = V @ X.T @ np.linalg.solve(covariance, y.ravel())
                records.append((logweight, mean[0], mean[1], a, b))
    values = np.asarray(records)
    normalized = np.exp(values[:,0]-logsumexp(values[:,0]))
    return normalized @ values[:,1:]


def test_full_gaussian_copula_ssvs_matches_integrated_reference():
    y = np.array([[-.8,-.3], [.2,.5], [-.3,.6], [.5,.3], [.7,1.1], [.5,.9]])
    R = np.array([[1.,.65],[.65,1.]])
    prior = bx.ssvs_gaussian_priors(1, alpha_sd=1.5, beta_sd=.2,
        level_dynamic_probability=0., trend_probabilities=(.3,.7,0.),
        season_probabilities=(1.,0.,0.), sigma2_prior=bx.InverseGammaPrior(2,1))
    model = bx.MultiSeriesModel([bx.Channel(name, bx.Gaussian(), [bx.LocalLinearTrend()]) for name in ('a','b')],
                                 copula=bx.GaussianCopula(correlation=R))
    fit = bx.fit(y, model, priors=bx.MarginalPriors({'a':prior,'b':prior}), parameterization='fs',
                 mcmc=bx.MCMC(chains=1,warmup=500,draws=3000,seed=7921))
    observed = [fit.parameter(f'initial.channel.{name}.level').mean() for name in ('a','b')]
    observed += [np.mean(fit.parameter(f'state.{name}.trend') == 1) for name in ('a','b')]
    expected = reference(y,R)
    np.testing.assert_allclose(expected, reference(y,R,nodes=64), atol=3e-4, rtol=0)
    # Monte Carlo tolerance is much larger than quadrature error; the fixed
    # seed makes this an auditable regression check, not a coverage study.
    np.testing.assert_allclose(observed, expected, atol=.06, rtol=0)
