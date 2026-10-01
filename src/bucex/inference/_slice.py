"""Stepping-out slice sampling on the real line."""
from __future__ import annotations
import numpy as np

def _slice_sample_real(
    current: float,
    log_density,
    rng: np.random.Generator,
    *,
    width: float = 1.0,
    max_steps_out: int = 50,
    max_shrink: int = 500,
) -> tuple[float, int]:
    """Univariate stepping-out slice sampler on the real line.

    The routine updates log observation scales and unconstrained GEV shapes.
    Support restrictions are enforced by the exact log density.
    It leaves the exact conditional invariant; the returned count is the
    number of target evaluations, useful for future performance diagnostics.
    """

    current = float(current)
    width = max(float(width), 1e-6)
    current_density = float(log_density(current))
    if not np.isfinite(current_density):
        raise FloatingPointError("Slice sampler started outside finite support.")
    log_height = current_density + np.log(max(float(rng.random()), 1e-300))
    left = current - width * float(rng.random())
    right = left + width
    evaluations = 1
    left_steps = int(rng.integers(max_steps_out + 1))
    right_steps = max_steps_out - left_steps
    while left_steps > 0:
        value = float(log_density(left))
        evaluations += 1
        if not np.isfinite(value) or value <= log_height:
            break
        left -= width
        left_steps -= 1
    while right_steps > 0:
        value = float(log_density(right))
        evaluations += 1
        if not np.isfinite(value) or value <= log_height:
            break
        right += width
        right_steps -= 1
    for _ in range(max_shrink):
        proposal = float(rng.uniform(left, right))
        value = float(log_density(proposal))
        evaluations += 1
        if np.isfinite(value) and value >= log_height:
            return proposal, evaluations
        if proposal < current:
            left = proposal
        else:
            right = proposal
    raise RuntimeError("Slice sampler failed to find a point on the slice.")

