"""GEV SSVS against direct integration in a nearly fixed-nuisance submodel."""
from dataclasses import replace

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gumbel_r, norm

import bucex as bx


def test_gumbel_slope_selection_matches_direct_quadrature():
    y = np.array([-.4,.3,.2,1.3,.9,1.6])
    time = np.arange(1, len(y)+1)
    grid, weight = np.polynomial.legendre.leggauss(160)
    alpha, wa = 5*grid, 5*weight
    beta, wb = 1.5*grid, 1.5*weight
    absent = gumbel_r.logpdf(y[:,None], loc=alpha[None,:]).sum(axis=0) + norm.logpdf(alpha,scale=1.5)+np.log(wa)+np.log(.3)
    present = gumbel_r.logpdf(y[:,None,None], loc=alpha[None,:,None]+time[:,None,None]*beta[None,None,:]).sum(axis=0)
    present += norm.logpdf(alpha[:,None],scale=1.5)+norm.logpdf(beta[None,:],scale=.2)
    present += np.log(wa[:,None])+np.log(wb[None,:])+np.log(.7)
    normalizer = logsumexp([logsumexp(absent),logsumexp(present)])
    probability = np.exp(logsumexp(present)-normalizer)
    alpha_mean = np.exp(absent-normalizer) @ alpha + np.sum(np.exp(present-normalizer)*alpha[:,None])
    prior = bx.ssvs_gev_priors(1, alpha_sd=1.5,beta_sd=.2,
        level_dynamic_probability=0.,trend_probabilities=(.3,.7,0.),season_probabilities=(1.,0.,0.))
    prior = replace(prior, sigma2=None, log_sigma=bx.NormalPrior(0.,1e-4), xi=bx.UniformPrior(-1e-6,1e-6))
    model = bx.MultiSeriesModel([bx.Channel('a',bx.GEV(),[bx.LocalLinearTrend()])])
    fit = bx.fit(y[:,None],model,priors=bx.MarginalPriors({'a':prior}),
        mcmc=bx.MCMC(chains=1,warmup=300,draws=2000,seed=20102))
    np.testing.assert_allclose([fit.parameter('initial.channel.a.level').mean(),np.mean(fit.parameter('state.a.trend')==1)],
                               [alpha_mean,probability],atol=.065,rtol=0)
