"""Simulation from a compiled structural model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.compiler import compile_model
from ..models.multiseries import MultiSeriesModel
from ..models.structural import Model


@dataclass
class Simulation:
    y: np.ndarray
    eta: np.ndarray
    states: np.ndarray
    params: dict[str, Any]
    model: Model | MultiSeriesModel
    exog: object = None

    @property
    def channel_names(self) -> tuple[str, ...]:
        return (
            self.model.channel_names
            if isinstance(self.model, MultiSeriesModel)
            else ()
        )

    @property
    def sigma(self) -> np.ndarray:
        """Observation-scale path (stationary scales are expanded to length T)."""

        if isinstance(self.model, MultiSeriesModel):
            raise ValueError("Choose a channel-specific scale for multiseries simulations.")
        values = np.asarray(self.params.get("sigma_path", self.params["sigma"]), dtype=float)
        return np.full(self.y.shape[0], float(values)) if values.ndim == 0 else values

    @property
    def phi(self) -> np.ndarray:
        """The simulated log-scale path."""

        return np.log(self.sigma)


def simulate(
    model: Model | MultiSeriesModel,
    n_time: int,
    params: dict[str, Any],
    *,
    exog=None,
    dates=None,
    initial_state=None,
    seed: int | None = None,
) -> Simulation:
    n_time = int(n_time)
    if n_time < 1:
        raise ValueError("n_time must be positive.")
    rng = np.random.default_rng(seed)
    placeholder = (
        np.zeros((n_time, len(model.channel_names)))
        if isinstance(model, MultiSeriesModel)
        else np.zeros(n_time)
    )
    compiled = compile_model(model, placeholder, exog=exog)
    missing = [f"sd.{name}" for name in compiled.noise_names if f"sd.{name}" not in params]
    if isinstance(model, MultiSeriesModel):
        required = list(compiled.observation_parameter_names)
    else:
        required = ["sigma"] + (["xi"] if model.family == "gev" else [])
    missing.extend(name for name in required if name not in params)
    if missing:
        raise ValueError(f"Missing simulation parameters: {missing}")
    scale_paths = {}
    channels = model.channels if isinstance(model, MultiSeriesModel) else (None,)
    for channel in channels:
        observation = channel.observation if channel else model.observation
        if observation.scale is None:
            continue
        suffix = f".{channel.name}" if channel else ""
        key = "scale.seasonal"+suffix
        if key not in params and observation.scale.period > 1:
            raise ValueError(f"Supply {key}: zero-sum seasonal log-scale effects.")
        effects = np.asarray(params.get(key, np.zeros(1)), float)
        if effects.shape != (observation.scale.period,) or not np.all(np.isfinite(effects)) or not np.isclose(effects.sum(), 0., atol=1e-10):
            raise ValueError(f"{key} must contain period finite effects summing to zero.")
        phase = observation.scale.phases(n_time, dates)
        offset = np.zeros(n_time)
        mode = getattr(observation.scale,"mode","constant")
        if mode == "structural":
            from ..inference.fit.evolution import simulate_evolution
            offset = simulate_evolution(observation.scale, n_time, params, rng, suffix)
        elif mode == "linear":
            if "scale_slope"+suffix not in params:
                raise ValueError("Linear scale simulation requires scale_slope"+suffix)
            offset = float(params["scale_slope"+suffix])*np.arange(1,n_time+1)/observation.scale.time_unit
        elif mode == "rw":
            if "scale_signed_sd"+suffix not in params:
                raise ValueError("RW scale simulation requires scale_signed_sd"+suffix)
            z = (np.asarray(params["scale_z"+suffix]) if "scale_z"+suffix in params else np.cumsum(rng.normal(size=n_time)))
            if z.shape != (n_time,):
                raise ValueError("scale_z must have length n_time.")
            offset = float(params["scale_signed_sd"+suffix])*z
        scale_paths["sigma"+suffix] = np.asarray(params["sigma"+suffix]) * np.exp(effects[phase]+offset)
        if np.any(~np.isfinite(scale_paths["sigma"+suffix])) or np.any(scale_paths["sigma"+suffix] <= 0):
            raise ValueError("Simulated scale path is not positive and finite.")
    sigma_path = None
    if not isinstance(model, MultiSeriesModel):
        sigma_values = np.asarray(scale_paths.get("sigma", params["sigma"]), dtype=float)
        if sigma_values.ndim == 0:
            sigma_path = np.full(n_time, float(sigma_values), dtype=float)
        elif sigma_values.shape == (n_time,):
            sigma_path = sigma_values
        else:
            raise ValueError("Simulation sigma must be scalar or have length n_time.")
        if np.any(~np.isfinite(sigma_path)) or np.any(sigma_path <= 0.0):
            raise ValueError("Simulation sigma values must be finite and positive.")
    states = np.zeros((n_time + 1, compiled.state_dim))
    if initial_state is None:
        states[0] = np.zeros(compiled.state_dim)
    else:
        states[0] = np.asarray(initial_state, dtype=float).reshape(compiled.state_dim)
    process_sd = compiled.process_vector(params)
    copula = getattr(model, "copula", None)
    copula_phases = copula.phases(n_time, dates) if copula is not None and copula.seasonal else None
    design = compiled.design(params=params)
    if isinstance(model, MultiSeriesModel):
        eta = np.zeros((n_time, len(model.channel_names)))
        y = np.zeros_like(eta)
    else:
        eta = np.zeros(n_time)
        y = np.zeros(n_time)
    for t in range(1, n_time + 1):
        states[t] = (
            compiled.transition @ states[t - 1]
            + compiled.loading @ (process_sd * rng.normal(size=compiled.noise_dim))
        )
        if isinstance(model, MultiSeriesModel):
            eta[t - 1] = design[t - 1] @ states[t]
            time_params = {**params, **{name: float(path[t-1]) for name,path in scale_paths.items()}}
            if copula_phases is not None:
                time_params["__copula_phase"] = int(copula_phases[t-1])
            y[t - 1] = compiled.sample_observation(eta[t - 1], time_params, rng)
        else:
            eta[t - 1] = float(design[t - 1] @ states[t])
            y[t - 1] = float(
                model.observation.sample(
                    eta=eta[t - 1],
                    sigma=float(sigma_path[t - 1]),
                    xi=params.get("xi"),
                    rng=rng,
                )
            )
    if isinstance(model, MultiSeriesModel):
        signs = model.transform_signs[None, :]
        y = signs * y
        eta = signs * eta
    output_params = dict(params)
    if scale_paths and sigma_path is not None:
        output_params["sigma_path"] = sigma_path
    elif scale_paths:
        output_params.update({key.replace("sigma.","sigma_path."):v for key,v in scale_paths.items()})
    return Simulation(y=y, eta=eta, states=states, params=output_params, model=model, exog=compiled.exog)
