"""A univariate likelihood conditional on the other Gaussian-copula scores.

This adapter reuses the exact FS Laplace--MH kernel. The Gaussian proposal
includes copula curvature; the MH correction uses the same exact conditional
likelihood. Other channels are held fixed throughout a channel update.
"""
import numpy as np

from ...dependence.gaussian import _gev_scores


class ConditionalMargin:
    def __init__(self, observation, mean, variance):
        self.observation = observation
        self.mean = np.asarray(mean, float)  # INTERNAL score orientation
        self.variance = np.asarray(variance, float)
        self.name = observation.name

    def scores(self, y, eta, params):
        sigma = np.asarray(params["sigma"], float)
        if self.name == "gaussian":
            return (np.asarray(y) - eta) / sigma
        return _gev_scores(np.asarray(y), np.asarray(eta), sigma, float(params["xi"]))

    def logpdf(self, y, eta, params):
        marginal = np.asarray(self.observation.logpdf(y, eta, params))
        if not np.all(np.isfinite(marginal)):
            return np.full(np.shape(marginal), -np.inf)
        try:
            z = self.scores(y, eta, params)
            correction = (-.5 * np.log(self.variance)
                          -.5 * (z-self.mean)**2 / self.variance + .5*z*z)
            result = marginal + correction
            return result if np.all(np.isfinite(result)) else np.full_like(result, -np.inf)
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(np.shape(marginal), -np.inf)

    def derivatives(self, y, eta, params):
        z = self.scores(y, eta, params)
        gradient = self.observation.grad_eta(y, eta, params)
        hessian = self.observation.hess_eta(y, eta, params)
        if self.name == "gaussian":
            first = -1. / np.asarray(params["sigma"])
            second = np.zeros_like(z)
        else:
            density = np.exp(self.observation.logpdf(y, eta, params)
                             + .5*z*z + .5*np.log(2*np.pi))
            first = -density
            second = -density * gradient + z * density**2
        correction_gradient = z - (z-self.mean) / self.variance
        return (gradient + first*correction_gradient,
                hessian + (1-1/self.variance)*first**2 + second*correction_gradient)

    def grad_eta(self, y, eta, params):
        return self.derivatives(y, eta, params)[0]

    def hess_eta(self, y, eta, params):
        return self.derivatives(y, eta, params)[1]

    def support_ok(self, y, eta, params):
        return self.observation.support_ok(y, eta, params)


def conditional_score_parameters(scores, correlation, index, sign=1.):
    """Internal z_j | z_-j parameters, using original-orientation R."""
    correlation = np.asarray(correlation)
    if correlation.ndim == 3:
        # Work per distinct phase rather than inverting T identical matrices.
        matrices, labels = np.unique(correlation, axis=0, return_inverse=True)
        mean, variance = np.empty(len(scores)), np.empty(len(scores))
        for group, matrix in enumerate(matrices):
            selected = labels == group
            mean[selected], variance[selected] = conditional_score_parameters(scores[selected], matrix, index, sign)
        return mean, variance
    k = correlation.shape[0]
    others = np.arange(k) != index
    if not np.any(others):
        return np.zeros(scores.shape[0]), 1.
    weights = np.linalg.solve(correlation[np.ix_(others, others)], correlation[others, index])
    variance = float(1 - correlation[index, others] @ weights)
    if not np.isfinite(variance) or variance <= 0:
        raise FloatingPointError("Copula conditional variance is not positive.")
    return float(sign) * (scores[:, others] @ weights), variance
