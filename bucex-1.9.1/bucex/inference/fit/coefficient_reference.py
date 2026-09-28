"""Deterministic Gaussian preconditioning for non-Gaussian FS coefficients.

This module constructs a proposal, not a posterior approximation to report.
The caller divides the exact conditional target by this Gaussian inside an
elliptical slice update. The reference depends only on quantities held fixed
during that update: data, paths, nuisance parameters and conditional priors.
In particular, neither optimization nor fallback uses the current coefficient
vector. Finite optimization tolerances therefore affect efficiency, not the
invariant posterior distribution.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def _support_safe_center(design, y, params, prior_root, mean, predictor):
    """Deterministically shift an intercept into the GEV location support."""
    xi = float(params['xi'])
    if xi == 0:
        return mean, False
    sigma = np.broadcast_to(np.asarray(params['sigma'], float), np.shape(y))
    boundary = np.asarray(y) + sigma / xi
    # A strictly interior location, not a likelihood/support modification.
    shift = (max(0., float(np.max(boundary + .05*sigma - predictor))) if xi < 0
             else min(0., float(np.min(boundary - .05*sigma - predictor))))
    if shift == 0:
        return mean, False
    intercepts = np.flatnonzero(np.all(design == 1., axis=0))
    if not len(intercepts):
        return mean, False
    delta = np.zeros_like(mean)
    delta[intercepts[0]] = shift
    return mean + np.linalg.solve(prior_root, delta), True


def coefficient_reference(design, y, conditional, params, prior_mean, prior_root,
                          initial_reference, controls):
    """Refine a fixed Gaussian reference at the exact conditional mode.

    Optimization is in the initial reference's standardized coordinates, which
    handles very different level/slope/innovation scales. The final curvature
    includes the Gaussian prior and, where present, the copula likelihood.
    A failed optimization retains its best finite point; a failed construction
    falls back to the original reference. All decisions are deterministic and
    recorded. Curvature regularization changes only the proposal covariance.
    """
    mean, root, precision = initial_reference
    A = np.asarray(design) @ prior_root
    offset = np.asarray(design) @ prior_mean
    center, repaired = _support_safe_center(
        np.asarray(design), y, params, prior_root, mean, offset + A @ mean)
    metric = dict(coefficient_reference_converged=0.,
                  coefficient_reference_iterations=0.,
                  coefficient_reference_fallback=0.,
                  coefficient_reference_support_repaired=float(repaired),
                  coefficient_reference_regularized=0.)
    B = A @ root
    best_value, best_point = np.inf, np.zeros_like(mean)

    def objective(v):
        nonlocal best_value, best_point
        w = center + root @ v
        eta = offset + A @ w
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ll = float(np.sum(conditional.logpdf(y, eta, params)))
            if not np.isfinite(ll):
                return np.inf, np.zeros_like(v)
            gradient = root.T @ w - B.T @ conditional.grad_eta(y, eta, params)
        value = .5 * (w @ w) - ll
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return np.inf, np.zeros_like(v)
        if value < best_value:
            best_value, best_point = value, np.array(v, copy=True)
        return value, gradient

    try:
        value, _ = objective(np.zeros_like(mean))
        if not np.isfinite(value):
            raise FloatingPointError('No finite deterministic coefficient reference.')
        result = minimize(objective, np.zeros_like(mean), jac=True, method='BFGS',
                          options={'gtol': controls.tolerance,
                                   'maxiter': controls.max_iterations})
        metric['coefficient_reference_iterations'] = float(result.nit)
        # Use our best evaluated finite point, also on line-search failure.
        mode = center + root @ best_point
        _, gradient = objective(best_point)
        metric['coefficient_reference_converged'] = float(
            np.max(np.abs(gradient)) <= controls.tolerance)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            curvature = conditional.hess_eta(y, offset + A @ mode, params)
            H = root.T @ root - B.T @ (curvature[:, None] * B)
        H = (H + H.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        floor = controls.curvature_floor * max(1., float(np.max(np.abs(eigenvalues))))
        if np.min(eigenvalues) < floor:
            H = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
            metric['coefficient_reference_regularized'] = 1.
        factor = np.linalg.cholesky(H)
        refined_root = root @ solve_triangular(factor.T, np.eye(len(mean)), lower=False)
        inverse_root = np.linalg.solve(refined_root, np.eye(len(mean)))
        refined_precision = inverse_root.T @ inverse_root
        if not all(np.all(np.isfinite(v)) for v in (mode, refined_root, refined_precision)):
            raise FloatingPointError('Nonfinite coefficient reference.')
        return (mode, refined_root, refined_precision), metric
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        metric['coefficient_reference_fallback'] = 1.
        metric['coefficient_reference_converged'] = 0.
        return (mean, root, precision), metric
