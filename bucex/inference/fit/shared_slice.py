"""Exact Gaussian-prior elliptical slice refreshes for shared state models.

Each complete private channel and each complete shared/departure group is a
Gaussian prior block. Grouping all departure contrasts makes these updates
independent of the chosen orthonormal contrast basis. This is a scheduled
additional kernel, never a fallback conditional on rejecting another proposal.

Algorithm: Murray, Adams and MacKay (2010), AISTATS 9:541--548.
https://proceedings.mlr.press/v9/murray10a.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.numerics import sample_mvn


@dataclass(frozen=True)
class GaussianPriorBlock:
    name: str
    indices: np.ndarray
    channels: np.ndarray
    mean_path: np.ndarray
    transition: np.ndarray
    initial_cov: np.ndarray
    loading: np.ndarray
    innovation_indices: np.ndarray
    design: np.ndarray


@dataclass(frozen=True)
class SliceRefresh:
    path: np.ndarray
    evaluations: int
    angle: float
    moved: bool
    stochastic: bool


def gaussian_prior_blocks(compiled: Any) -> tuple[GaussianPriorBlock, ...]:
    """Partition the compiler's independent Gaussian prior into update blocks."""
    definitions = [(f"channel.{block.name}", np.arange(block.state_slice.start, block.state_slice.stop))
                   for block in compiled.blocks]
    for name, slices in compiled.group_slices.items():
        indices = np.concatenate([np.arange(item.start, item.stop) for item in slices])
        definitions.append((f"{compiled.group_kinds[name]}.{name}", indices))
    all_indices = np.concatenate([indices for _, indices in definitions])
    if not np.array_equal(np.sort(all_indices), np.arange(compiled.state_dim)):
        raise ValueError("Elliptical-slice blocks must partition every state exactly once.")
    full_design = compiled.design(compiled.n_time)
    result = []
    for name, indices in definitions:
        if indices.size == 0:
            continue
        outside = np.setdiff1d(np.arange(compiled.state_dim), indices)
        if (np.any(compiled.transition[np.ix_(indices, outside)] != 0.0)
                or np.any(compiled.transition[np.ix_(outside, indices)] != 0.0)
                or np.any(compiled.initial_cov[np.ix_(indices, outside)] != 0.0)):
            raise ValueError("Elliptical-slice blocks require independent Gaussian prior transitions and initial states.")
        innovations = np.flatnonzero(np.any(compiled.loading[indices] != 0.0, axis=0))
        if np.any(compiled.loading[np.ix_(outside, innovations)] != 0.0):
            raise ValueError("An innovation cannot cross elliptical-slice prior blocks.")
        transition = compiled.transition[np.ix_(indices, indices)]
        mean = np.empty((compiled.n_time + 1, len(indices)))
        mean[0] = compiled.initial_mean[indices]
        for t in range(1, len(mean)):
            mean[t] = transition @ mean[t - 1]
        local_design = full_design[:, :, indices]
        channels = (np.arange(len(compiled.channel_names)) if getattr(compiled, "has_copula", False)
                    else np.flatnonzero(np.any(local_design != 0.0, axis=(0, 2))))
        result.append(GaussianPriorBlock(
            name=name, indices=indices, channels=channels, mean_path=mean,
            transition=transition, initial_cov=compiled.initial_cov[np.ix_(indices, indices)],
            loading=compiled.loading[np.ix_(indices, innovations)],
            innovation_indices=innovations, design=local_design[:, channels],
        ))
    return tuple(result)


def _zero_mean_prior_draw(block: GaussianPriorBlock, process_sd, rng) -> tuple[np.ndarray, bool]:
    sd = np.asarray(process_sd)[block.innovation_indices]
    stochastic = bool(np.any(block.initial_cov != 0.0) or np.any(sd != 0.0))
    auxiliary = np.zeros_like(block.mean_path)
    if not stochastic:
        return auxiliary, False
    auxiliary[0] = sample_mvn(np.zeros(block.indices.size), block.initial_cov, rng)
    innovations = rng.normal(size=(len(auxiliary) - 1, len(sd))) * sd[None, :]
    increments = innovations @ block.loading.T
    for t in range(1, len(auxiliary)):
        auxiliary[t] = block.transition @ auxiliary[t - 1] + increments[t - 1]
    return auxiliary, True


def _affected_log_likelihood(y, eta, compiled, params, channels) -> float:
    if getattr(compiled, "has_copula", False):
        return compiled.observation_log_likelihood(y, eta, params)
    total = 0.0
    for local, index in enumerate(channels):
        channel = compiled.model.channels[index]
        mask = np.isfinite(y[:, index])
        values = channel.observation.logpdf(
            y[mask, index], eta[mask, local], sigma=params[f"sigma.{channel.name}"],
            xi=params.get(f"xi.{channel.name}"),
        )
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise FloatingPointError("Invalid observation density during an elliptical-slice refresh.")
        if np.any(np.isneginf(values)):
            return -np.inf
        total += float(np.sum(values))
    return total


def elliptical_slice_block(y, path, compiled, params, block, rng, *, maximum_evaluations=200) -> SliceRefresh:
    """Refresh one block conditional on all other blocks and static parameters.

    Exact zeros in the Gaussian prior remain deterministic. A finite GEV
    endpoint violation simply shrinks the angle bracket. Exhausting the
    numerical evaluation budget raises, so it cannot create a biased atom at
    the incumbent trajectory.
    """
    if int(maximum_evaluations) < 1:
        raise ValueError("maximum_evaluations must be positive.")
    auxiliary, stochastic = _zero_mean_prior_draw(block, compiled.process_vector(params), rng)
    if not stochastic:
        return SliceRefresh(path, 0, 0.0, False, False)
    incumbent = path[:, block.indices]
    eta = compiled.eta(path, params=params)[:, block.channels]
    current = _affected_log_likelihood(y, eta, compiled, params, block.channels)
    if not np.isfinite(current):
        raise FloatingPointError("Elliptical-slice incumbent is outside observation support.")
    uniform = float(rng.random())
    threshold = current + (np.log(uniform) if uniform > 0.0 else -np.inf)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    lower, upper = angle - 2.0 * np.pi, angle
    centered = incumbent - block.mean_path
    for evaluation in range(1, int(maximum_evaluations) + 1):
        candidate = block.mean_path + centered * np.cos(angle) + auxiliary * np.sin(angle)
        eta_candidate = eta + np.einsum('tki,ti->tk', block.design, candidate[1:] - incumbent[1:])
        likelihood = _affected_log_likelihood(y, eta_candidate, compiled, params, block.channels)
        if np.isfinite(likelihood) and likelihood >= threshold:
            result = path.copy()
            result[:, block.indices] = candidate
            moved = bool(np.any(candidate != incumbent))
            principal_angle = float((angle + np.pi) % (2.0 * np.pi) - np.pi)
            return SliceRefresh(result, evaluation, principal_angle, moved, True)
        if angle < 0.0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise FloatingPointError(
        f"Elliptical-slice block {block.name!r} exhausted {maximum_evaluations} evaluations; no posterior result is returned."
    )


__all__ = ["GaussianPriorBlock", "SliceRefresh", "gaussian_prior_blocks", "elliptical_slice_block"]
