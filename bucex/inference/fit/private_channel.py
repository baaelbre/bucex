"""Initialization of a private FS channel for the marginal sampler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ...models.structural import Model
from ...priors.structural import FSGaussianPriors, FSGEVPriors
from ..config import MCMC
from .fs_gev import FSGEVKernel, _gev_support_ok
from .fs_utils import (canonicalize_ncp_params, infer_ncp_layout,
                       map_centered_to_ncp, mu_from_ncp,
                       seasonal_state_from_phase_effects, static_seasonal_design)
from .model_space import StructuralModelState, initial_structural_state

Array = np.ndarray

@dataclass
class _ChannelState:
    name: str
    family: str
    y: Array
    model: Model
    compiled: Any
    layout: Any
    base_prior: FSGaussianPriors | FSGEVPriors
    params_state: dict[str, Any]
    params_obs: dict[str, float]
    z_path: Array
    model_state: StructuralModelState
    model_index: int
    gev_kernel: FSGEVKernel | None = None


def _seasonal_initial(y: Array, period: int | None) -> Array:
    if period is None:
        return np.zeros(0, dtype=float)
    phase = np.arange(y.size) % int(period)
    overall = float(np.mean(y))
    full = np.asarray(
        [float(np.mean(y[phase == index])) - overall if np.any(phase == index)
         else 0.0 for index in range(int(period))]
    )
    full -= float(np.mean(full))
    return seasonal_state_from_phase_effects(full)


def _linear_initial(y: Array, gamma: Array, period: int | None) -> tuple[float, float]:
    """Data-informed chain start for the level and slope; both remain sampled."""

    values = np.asarray(y, dtype=float)
    if period is None:
        adjusted = values
    else:
        adjusted = values - static_seasonal_design(values.size, int(period)-1) @ gamma
    time = np.arange(values.size, dtype=float)
    centered = time - float(np.mean(time))
    denominator = float(centered @ centered)
    slope = 0.0 if denominator == 0.0 else float(centered @ adjusted) / denominator
    level = float(np.mean(adjusted) - slope * np.mean(time))
    return level, slope


def _initial_channel_state(
    name: str,
    family: str,
    y: Array,
    compiled: Any,
    prior: FSGaussianPriors | FSGEVPriors,
    rng: np.random.Generator,
    initial: Mapping[str, Any] | None,
) -> _ChannelState:
    model = compiled.model
    layout = infer_ncp_layout(model)
    gamma = _seasonal_initial(y, model.period)
    intercept, beta0 = _linear_initial(y, gamma, model.period)
    # baseline_mu_path evaluates the first observation at t=1.
    alpha0 = intercept - beta0
    fitted = alpha0 + beta0 * np.arange(1, y.size+1, dtype=float)
    if model.period is not None:
        fitted = fitted + static_seasonal_design(y.size, int(model.period)-1) @ gamma
    residual = y - fitted
    scale = max(float(np.std(residual, ddof=1)), 0.05)
    state = {
        "alpha0": alpha0,
        "beta0": beta0,
        "gamma0_season": gamma,
        "s_level": 0.01,
        "s_trend": 0.0001,
        "s_season": 0.01,
        "q_level": 0.01**2,
        "q_trend": 0.0001**2,
        "q_season": 0.01**2,
    }
    observation = {"sigma": scale, "sigma2": scale**2}
    if family == "gev":
        # Gumbel initialization has no finite support boundary.
        observation["xi"] = 0.0
    supplied = {} if initial is None else dict(initial)
    prefixes = (f"channel.{name}.", f"{name}.")
    local: dict[str, Any] = {}
    for key, value in supplied.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                local[key.removeprefix(prefix)] = value
                break
    centered_path = local.pop("__centered_path", None)
    aliases = {
        "intercept": "alpha0",
        "initial.level": "alpha0",
        "initial.slope": "beta0",
        "initial.seasonal": "gamma0_season",
        "initial_slope": "beta0",
        "sd.level": "s_level",
        "sd.slope": "s_trend",
        "sd.seasonal": "s_season",
    }
    for key, value in local.items():
        target = aliases.get(key, key)
        if target in state:
            state[target] = value
        elif target in observation:
            observation[target] = float(value)
        else:
            raise ValueError(f"Unknown initial parameter '{key}' for channel '{name}'.")
    state = canonicalize_ncp_params(state, layout)
    observation["sigma2"] = float(observation["sigma"]) ** 2
    model_state = initial_structural_state(state, layout)
    if centered_path is None:
        z_path = np.zeros((y.size + 1, layout.ncp_state_dim), dtype=float)
    else:
        centered = np.asarray(centered_path, dtype=float)
        expected = (y.size + 1, layout.centered_state_dim)
        if centered.shape != expected or not np.all(np.isfinite(centered)):
            raise ValueError(
                f"Warm-start path for channel '{name}' must have shape {expected} "
                "and contain only finite values."
            )
        z_path = map_centered_to_ncp(centered, state, layout)
    mu = mu_from_ncp(z_path, state, layout)
    if family == "gev" and not _gev_support_ok(y, mu, model, observation):
        # A supplied nonzero shape can introduce a finite endpoint even when
        # the data-informed location path is sensible.  Enlarge only the
        # starting scale enough to place every observation strictly inside
        # support; sigma remains sampled immediately afterwards.  This is an
        # initialization repair, not a likelihood or prior modification.
        xi = float(observation["xi"])
        required = float(np.max(-xi * (np.asarray(y, dtype=float) - mu)))
        if xi != 0.0 and np.isfinite(required) and required > 0.0:
            scale = max(float(observation["sigma"]), required / .8)
            observation.update(sigma=scale, sigma2=scale * scale)
    if family == "gev" and not _gev_support_ok(y, mu, model, observation):
        raise ValueError(
            f"Initial GEV values violate support for channel '{name}'. "
            "Use channel-prefixed init values closer to the data."
        )
    kernel = None
    if family == "gev":
        kernel = FSGEVKernel(model, prior, config=MCMC(draws=1, chains=1))
        kernel.rng = rng
    return _ChannelState(
        name=name,
        family=family,
        y=np.asarray(y, dtype=float),
        model=model,
        compiled=compiled,
        layout=layout,
        base_prior=prior,
        params_state=state,
        params_obs=observation,
        z_path=z_path,
        model_state=model_state,
        model_index=-1,
        gev_kernel=kernel,
    )
