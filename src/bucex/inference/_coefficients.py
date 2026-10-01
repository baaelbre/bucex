from __future__ import annotations
import numpy as np
from scipy.linalg import cho_solve, solve_triangular

def gaussian_reference(design, response, variance, prior_mean, prior_root):
    """Gaussian posterior in prior-whitened coordinates; no covariance floors."""
    X = np.asarray(design, float)
    variance = np.broadcast_to(np.asarray(variance,float), (len(X),))
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ValueError('Gaussian reference variances must be positive and finite.')
    A = (X @ prior_root) / np.sqrt(variance[:,None])
    b = (np.asarray(response)-X @ prior_mean) / np.sqrt(variance)
    precision = np.eye(A.shape[1]) + A.T @ A
    factor = np.linalg.cholesky(precision)
    mean = cho_solve((factor,True), A.T @ b)
    # root @ root.T = precision^{-1}; the root need not be triangular.
    root = solve_triangular(factor.T, np.eye(len(mean)), lower=False)
    return mean, root, precision

def reference_slice(current, mean, root, log_correction, rng, *, max_steps=10000):
    """Elliptical slice for N(mean, root root.T) times exp(log_correction)."""
    log_current = float(log_correction(current))
    if not np.isfinite(log_current):
        raise FloatingPointError('Current coefficient vector has an invalid exact target.')
    threshold = log_current + np.log(rng.random())
    direction = root @ rng.normal(size=len(current))
    centered = current-mean
    angle = rng.uniform(0.,2*np.pi); lo,hi = angle-2*np.pi,angle
    for n in range(1,max_steps+1):
        candidate = mean + centered*np.cos(angle) + direction*np.sin(angle)
        if log_correction(candidate) >= threshold:
            return candidate,n
        if angle < 0: lo=angle
        else: hi=angle
        angle = rng.uniform(lo,hi)
    raise FloatingPointError('Coefficient elliptical slice exhausted its bracket budget.')
