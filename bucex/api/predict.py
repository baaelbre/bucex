"""Posterior predictive simulation and forecast summaries."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..diagnostics.calibration import pit_diagnostics
from ..diagnostics.scores import evaluate_ensemble, log_predictive_score
from ..inference.fit.phi import phi_future_basis
from ..core.shared_results import channel_design, group_design, univariate_design
from ..core.calendar import seasonal_phases


Array = np.ndarray


def _seasonal_scale_paths(fit, indices, n_time, dates, *, start_index=0, rng=None):
    result = {}
    copula = getattr(fit.model, "copula", None)
    if copula is not None and copula.seasonal:
        result["copula_phase"] = np.broadcast_to(copula.phases(n_time, dates, start_index=start_index), (len(indices), n_time))
    channels = fit.model.channels if fit.is_multiseries_model else (None,)
    for channel in channels:
        observation = channel.observation if channel else fit.model.observation
        if observation.scale is None:
            continue
        suffix = f".{channel.name}" if channel else ""
        if start_index == 0 and "sigma_path"+suffix in fit.parameter_draws:
            result["sigma_path"+suffix] = fit.parameter("sigma_path"+suffix)[indices]
            continue
        phase = observation.scale.phases(n_time, dates, start_index=start_index)
        effects = fit.parameter("scale.seasonal"+suffix)[indices]
        baseline = fit.parameter("sigma"+suffix)[indices]
        offset = np.zeros((len(indices), n_time))
        mode = getattr(observation.scale, "mode", "constant")
        if mode == "structural":
            if rng is None:
                raise ValueError("Structural scale forecasts require a random generator.")
            from ..inference.fit.evolution import forecast_evolution
            offset = forecast_evolution(observation.scale, fit, indices, n_time, rng, suffix)
        elif mode == "linear":
            slope = fit.parameter("scale_slope"+suffix)[indices]
            offset = slope[:,None]*np.arange(start_index+1,start_index+n_time+1)/observation.scale.time_unit
        elif mode == "rw":
            if rng is None:
                raise ValueError("Random-walk scale forecasting requires an explicit random generator.")
            signed_sd = fit.parameter("scale_signed_sd"+suffix)[indices]
            last = fit.parameter("scale_z"+suffix)[indices,-1]
            offset = signed_sd[:,None]*(last[:,None]+np.cumsum(rng.normal(size=(len(indices),n_time)),axis=1))
        result["sigma_path"+suffix] = baseline[:, None] * np.exp(effects[:, phase] + offset)
    return result


def _time_parameters(parameters, values, draw, time):
    result = dict(parameters)
    if "copula_phase" in values:
        result["__copula_phase"] = int(values["copula_phase"][draw, time])
    for name, array in values.items():
        if name.startswith("sigma_path."):
            result["sigma."+name.removeprefix("sigma_path.")] = float(array[draw, time])
    return result


def _component_designs(fit, n_time: int) -> dict[str, Array]:
    """Retain scientific decompositions in joint forecast objects."""
    if not fit.is_multiseries_model:
        return {f"univariate.{component}": univariate_design(fit.compiled, component, fit.transform_sign)
                for component in ("level", "slope", "seasonal")}
    designs = {
        f"channel.{channel}.{component}": channel_design(fit.compiled, channel, component, n_time)
        for channel in fit.channel_names
        for component in ("level", "slope", "seasonal")
    }
    if getattr(fit.compiled, "is_shared", False):
        for name, kind in fit.compiled.group_kinds.items():
            for component in ("level", "slope", "seasonal"):
                try:
                    if kind == "shared":
                        designs[f"shared.{name}.{component}"] = group_design(fit.compiled, name, component)
                    else:
                        for channel in fit.channel_names:
                            designs[f"departure.{name}.{channel}.{component}"] = group_design(fit.compiled, name, component, channel=channel)
                except ValueError:
                    # A local-level process, for example, has no slope.
                    continue
    return designs


def _future_dates(dates: Array | None, horizon: int) -> Array:
    if dates is None:
        return np.arange(1, horizon + 1)
    values = np.asarray(dates)
    try:
        import pandas as pd

        parsed = pd.to_datetime(values)
        frequency = pd.infer_freq(parsed)
        if frequency is not None:
            return pd.date_range(parsed[-1], periods=horizon + 1, freq=frequency)[1:].to_numpy()
        if parsed.size >= 2:
            delta = parsed[-1] - parsed[-2]
            return np.asarray([parsed[-1] + (j + 1) * delta for j in range(horizon)])
    except Exception:
        pass
    if np.issubdtype(values.dtype, np.number) and values.size >= 2:
        step = values[-1] - values[-2]
        return values[-1] + step * np.arange(1, horizon + 1)
    return np.arange(1, horizon + 1)


@dataclass
class Forecast:
    observations: Array
    eta: Array
    states: Array
    parameters: dict[str, Array]
    dates: Array
    family: str
    tail: Any
    observation_model: Any
    transform_sign: Any = 1.0
    channel_names: tuple[str, ...] = ()
    state_names: tuple[str, ...] = ()
    period: int | None = None
    phases: Array | None = None
    component_designs: dict[str, Array] = field(default_factory=dict)
    copula: Any = None
    channels: tuple[Any, ...] = ()

    @property
    def is_multiseries_forecast(self) -> bool:
        return bool(self.channel_names)

    def _channel_index(self, channel: str | None) -> int:
        if not self.is_multiseries_forecast:
            if channel is not None:
                raise ValueError("channel= is only valid for a multiseries forecast.")
            return 0
        if channel is None:
            raise ValueError(f"Choose channel from {self.channel_names}.")
        if channel not in self.channel_names:
            raise KeyError(f"Unknown channel '{channel}'. Available: {self.channel_names}")
        return self.channel_names.index(channel)

    @property
    def n_draws(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[1])

    def _phase_indices(self, phase: int | None) -> Array:
        if phase is None:
            return np.arange(self.horizon)
        if self.period is None or int(self.period) < 2:
            raise ValueError("phase= requires a forecast with a seasonal period.")
        selected_phase = int(phase)
        if selected_phase != phase or not 1 <= selected_phase <= int(self.period):
            raise ValueError(f"phase must be an integer from 1 to {self.period}.")
        phases = (
            np.asarray(self.phases, dtype=int).reshape(-1)
            if self.phases is not None
            else seasonal_phases(self.period, self.horizon, self.dates)
        )
        if phases.size != self.horizon:
            raise ValueError("Forecast phases must have length equal to horizon.")
        indices = np.flatnonzero(phases == selected_phase)
        if indices.size == 0:
            raise ValueError(
                f"phase {selected_phase} is not present in this forecast horizon."
            )
        return indices

    def component_draws(self, name: str, *, channel: str | None = None) -> Array:
        """Return one latent state component on the response orientation.

        In particular, ``component_draws("level", channel=...)`` is the seasonally adjusted
        structural trajectory: it excludes both the seasonal state and future
        observation noise.
        """

        if self.is_multiseries_forecast:
            self._channel_index(channel)
            key = f"channel.{channel}.{'seasonal' if name == 'season' else name}"
            return self._project_component(key)
        if channel is not None:
            raise ValueError("channel= is only valid for a multiseries forecast.")
        key = f"univariate.{'seasonal' if name == 'season' else name}"
        if key in self.component_designs:
            return self._project_component(key)
        if name in {"season", "seasonal"}:
            name = "seasonal[1]"
        if name not in self.state_names:
            raise KeyError(
                f"Unknown forecast state '{name}'. Available: {self.state_names}"
            )
        values = np.asarray(self.states[..., self.state_names.index(name)], dtype=float)
        return float(self.transform_sign) * values

    def _project_component(self, key: str) -> Array:
        if key not in self.component_designs:
            raise KeyError(f"Unknown forecast component {key!r}; choose from {tuple(self.component_designs)}.")
        design = np.asarray(self.component_designs[key])
        if design.ndim == 1:
            return np.einsum("dtm,m->dt", self.states, design)
        return np.einsum("dtm,tm->dt", self.states, design)

    def shared_draws(self, name: str = "warming", *, component: str = "level") -> Array:
        """Future shared trajectory before its fixed channel loadings."""
        return self._project_component(f"shared.{name}.{component}")

    def departure_draws(self, channel: str, *, name: str = "departure", component: str = "level") -> Array:
        """Future constrained departure on the original response scale."""
        self._channel_index(channel)
        return self._project_component(f"departure.{name}.{channel}.{component}")

    def _target_draws(self, target: str, *, channel: str | None) -> Array:
        key = str(target).lower().replace("-", "_")
        if key in {"observation", "observations", "predictive"}:
            index = self._channel_index(channel)
            return (
                self.observations[:, :, index]
                if self.is_multiseries_forecast
                else self.observations
            )
        if key in {"eta", "predictor", "latent_predictor"}:
            index = self._channel_index(channel)
            return self.eta[:, :, index] if self.is_multiseries_forecast else self.eta
        if key in {"level", "latent_level", "seasonally_adjusted"}:
            return self.component_draws("level", channel=channel)
        raise ValueError(
            "target must be 'observations', 'predictor', or 'level'."
        )

    def summary(
        self,
        level: float = 0.90,
        *,
        channel: str | None = None,
        target: str = "observations",
        phase: int | None = None,
    ):
        if not 0.0 < float(level) < 1.0:
            raise ValueError("level must lie in (0, 1).")
        alpha = 1.0 - float(level)
        target_key = str(target).lower().replace("-", "_")
        selected_horizons = self._phase_indices(phase)
        rows = []
        channels = (
            self.channel_names
            if self.is_multiseries_forecast and channel is None
            else (channel,) if self.is_multiseries_forecast else (None,)
        )
        for selected in channels:
            index = self._channel_index(selected) if self.is_multiseries_forecast else 0
            target_draws = self._target_draws(target_key, channel=selected)
            for h in selected_horizons:
                values = target_draws[:, h]
                eta = self.eta[:, h, index] if self.is_multiseries_forecast else self.eta[:, h]
                row = {
                    "time": self.dates[h],
                    "mean": float(np.mean(values)),
                    "lower": float(np.quantile(values, alpha / 2.0)),
                    "median": float(np.median(values)),
                    "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                }
                if target_key in {"observation", "observations", "predictive"}:
                    row.update(
                        {
                            "eta_lower": float(np.quantile(eta, alpha / 2.0)),
                            "eta_median": float(np.median(eta)),
                            "eta_upper": float(np.quantile(eta, 1.0 - alpha / 2.0)),
                        }
                    )
                if phase is not None:
                    row["phase"] = int(phase)
                if selected is not None:
                    row["channel"] = selected
                rows.append(row)
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def tail_probability(self, threshold: float, *, channel: str | None = None) -> Array:
        index = self._channel_index(channel)
        observations = (
            self.observations[:, :, index]
            if self.is_multiseries_forecast
            else self.observations
        )
        tail = self.tail[index] if self.is_multiseries_forecast else self.tail
        if tail == "upper":
            return np.mean(observations > float(threshold), axis=0)
        return np.mean(observations < float(threshold), axis=0)

    def parameter_path(self, parameter: str, *, channel=None, link_scale=False) -> Array:
        """Future mu, sigma or xi paths, using the same names as fitted paths."""
        index = self._channel_index(channel)
        if parameter == "mu":
            return self.eta[...,index] if self.is_multiseries_forecast else self.eta
        if parameter == "sigma":
            values = self.sigma_draws(channel=channel)
            return np.log(values) if link_scale else values
        key = "xi"+("."+channel if self.is_multiseries_forecast else "")
        if parameter == "xi" and key in self.parameters:
            return np.broadcast_to(self.parameters[key][:,None],(self.n_draws,self.horizon))
        raise ValueError("Supported parameters are mu, sigma and (GEV only) xi.")

    def sigma_draws(self, *, channel: str | None = None) -> Array:
        """Monthly observation SD (Gaussian) or scale (GEV), for every path."""
        self._channel_index(channel)
        suffix = f".{channel}" if self.is_multiseries_forecast else ""
        values = self.parameters.get("sigma_path" + suffix)
        if values is None:
            values = np.asarray(self.parameters["sigma" + suffix])[:, None]
        return np.broadcast_to(values, (self.n_draws, self.horizon))

    def probability_draws(self, threshold, *, channel=None, direction=None):
        """Monthly risk integrating observation noise within each parameter/state draw.

        Bands across these draws include future-state uncertainty. Use their
        mean for the posterior predictive probability, not a median band line.
        """
        index = self._channel_index(channel)
        tail = self.tail[index] if self.is_multiseries_forecast else self.tail
        direction = direction or ("<" if tail == "lower" else ">")
        if direction not in {"<", ">"} or not np.isfinite(threshold):
            raise ValueError("Use a finite threshold and direction '<' or '>'.")
        cdf = self.conditional_cdf(np.full(self.horizon, threshold), channel=channel)
        return cdf if direction == "<" else 1-cdf

    def risk_summary(self, threshold, *, channel=None, direction=None, phase=None, level=.95):
        """Monthly conditional-risk bands and integrated predictive probabilities."""
        import pandas as pd
        from .aggregate import _band
        values = self.probability_draws(threshold, channel=channel, direction=direction)
        indices = self._phase_indices(phase)
        index = self._channel_index(channel)
        tail = self.tail[index] if self.is_multiseries_forecast else self.tail
        direction = direction or ("<" if tail == "lower" else ">")
        return pd.DataFrame(dict(time=np.asarray(self.dates)[indices], threshold=float(threshold), direction=direction,
                                 **_band(values[:, indices], level)))

    def aggregate(self, *, frequency="year", reduction=None, channel=None,
                  months=None, weighting="days", include_partial=False):
        """Calendar averages or block extremes; see :func:`aggregate_forecast`.

        Partial years/seasons are omitted by default, never labelled complete.
        ``frequency='season'`` uses DJF/MAM/JJA/SON; DJF belongs to its ending year.
        """
        from .aggregate import aggregate_forecast
        return aggregate_forecast(self, frequency=frequency, reduction=reduction,
                                  channel=channel, months=months, weighting=weighting,
                                  include_partial=include_partial)

    def return_level(self, return_period: float, *, channel: str | None = None) -> Array:
        index = self._channel_index(channel)
        if self.is_multiseries_forecast:
            assert channel is not None
            observation_model = self.observation_model[channel]
            if getattr(observation_model, "name", None) != "gev":
                raise ValueError("Return levels require a GEV channel.")
            sign = float(np.asarray(self.transform_sign)[index])
            eta = self.eta[:, :, index]
            sigma = self.parameters.get(f"sigma_path.{channel}", self.parameters[f"sigma.{channel}"][:, None])
            xi = self.parameters[f"xi.{channel}"][:, None]
        elif self.family != "gev":
            raise ValueError("Return levels require a GEV forecast.")
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta = self.eta
            sigma = self.parameters.get("sigma_path")
            if sigma is None:
                sigma = self.parameters["sigma"][:, None]
            xi = self.parameters["xi"][:, None]
        if float(return_period) <= 1.0:
            raise ValueError("return_period must exceed 1.")
        probability = 1.0 - 1.0 / float(return_period)
        eta_model = sign * eta
        level = observation_model.ppf(probability, eta_model, sigma=sigma, xi=xi)
        return sign * level

    def conditional_log_density(
        self,
        observed: Array,
        *,
        channel: str | None = None,
    ) -> Array:
        """Per-draw log densities for held-out observations.

        The result has shape ``(draws, horizon)`` and uses the analytic
        Gaussian or GEV density, including every posterior observation-
        parameter draw.  It is therefore preferable to a kernel-density score
        computed from the simulated ensemble.
        """

        index = self._channel_index(channel)
        observed_array = np.asarray(observed, dtype=float).reshape(-1)
        if observed_array.size != self.horizon:
            raise ValueError("observed must have length equal to the forecast horizon.")
        if self.is_multiseries_forecast:
            assert channel is not None
            observation_model = self.observation_model[channel]
            sign = float(np.asarray(self.transform_sign)[index])
            eta_model = sign * self.eta[:, :, index]
            sigma = self.parameters.get(f"sigma_path.{channel}", self.parameters[f"sigma.{channel}"][:, None])
            xi = (
                self.parameters[f"xi.{channel}"][:, None]
                if getattr(observation_model, "name", None) == "gev"
                else None
            )
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta_model = sign * self.eta
            sigma = self.parameters.get("sigma_path")
            if sigma is None:
                sigma = self.parameters["sigma"][:, None]
            xi = self.parameters.get("xi")
            if xi is not None:
                xi = np.asarray(xi, dtype=float)[:, None]
        values = observation_model.logpdf(
            sign * observed_array[None, :],
            eta_model,
            sigma=sigma,
            xi=xi,
        )
        return np.asarray(values, dtype=float)

    def joint_conditional_log_density(self, observed: Array) -> Array:
        """Joint density per posterior draw and time, before mixture averaging.

        Marginal densities and the residual copula are evaluated at the same
        state/parameter draw. Even conditional independence generally does not
        justify multiplying posterior-averaged marginal predictive densities.
        Missing margins (NaN) are marginalized using the corresponding copula
        submatrix; a completely unobserved time block cannot be scored.
        """
        if not self.is_multiseries_forecast:
            return self.conditional_log_density(observed)
        values = np.asarray(observed, dtype=float)
        if values.shape != (self.horizon, len(self.channel_names)):
            raise ValueError("observed must have shape (horizon, channels).")
        if np.any(np.isinf(values)) or np.any(np.all(np.isnan(values), axis=1)):
            raise ValueError("Each time must have at least one observed finite channel; NaN marks missing margins.")
        result = sum(np.where(np.isnan(values[:, index])[None, :], 0.0,
                              self.conditional_log_density(values[:, index], channel=name))
                     for index, name in enumerate(self.channel_names))
        if self.copula is not None:
            from ..dependence.gaussian import normal_scores, gaussian_copula_logpdf

            if not self.channels:
                raise ValueError("Copula density evaluation requires the original channel specifications.")
            signs = np.asarray(self.transform_sign, dtype=float)
            for draw in range(self.n_draws):
                params = {name: float(array[draw]) for name, array in self.parameters.items()
                          if np.asarray(array[draw]).ndim == 0}
                # Density is zero if any marginal lies outside its support.
                finite = np.isfinite(result[draw])
                if not np.any(finite):
                    continue
                for channel in self.channel_names:
                    key = f"sigma_path.{channel}"
                    if key in self.parameters:
                        params[f"sigma.{channel}"] = self.parameters[key][draw, finite]
                z = normal_scores(values[finite] * signs[None, :],
                                  self.eta[draw, finite] * signs[None, :],
                                  self.channels, params)
                if self.copula.seasonal:
                    phase = self.parameters["copula_phase"][draw, finite].astype(int)
                    correlation = self.copula.correlation_matrix(params, self.channel_names)[phase-1]
                else:
                    correlation = self.copula.correlation_matrix(params, self.channel_names)
                result[draw, finite] += gaussian_copula_logpdf(z, correlation)
        return np.asarray(result)

    def joint_log_score(self, observed: Array) -> Array:
        """Negative log joint posterior-predictive density at each time block."""
        return log_predictive_score(self.joint_conditional_log_density(observed))

    def ordering_diagnostics(self, constraints, *, observed=None, tolerance=0.0):
        """Quantify incompatible replicated outcomes without sorting or rejection."""
        from ..diagnostics.ordering import ordering_diagnostics

        if not self.is_multiseries_forecast:
            raise ValueError("Ordering diagnostics require a multiseries forecast.")
        return ordering_diagnostics(self.observations, channel_names=self.channel_names,
                                    constraints=constraints, observed=observed,
                                    dates=self.dates, tolerance=tolerance)

    def compound_probability(self, events, *, operation="all") -> Array:
        """Probability of simultaneous channel threshold events at each time.

        Uses unchanged joint predictive draws, preserving copula, shared-state
        and parameter dependence. This Monte Carlo average is not a credible
        interval for a conditional probability.
        """
        from ..diagnostics.ordering import compound_event_probability

        if not self.is_multiseries_forecast:
            raise ValueError("Compound probabilities require a multiseries forecast.")
        return compound_event_probability(self.observations, channel_names=self.channel_names,
                                          events=events, operation=operation)

    def compound_probability_draws(self, events, *, operation="all", rtol=1e-8) -> Array:
        """Bivariate event probabilities per draw/time, integrating residual noise.

        Returns (draws, time), including parameter and state-path uncertainty.
        Forecast draws include sampled future states; intervals therefore
        describe conditional future-path risks. Their average estimates the
        posterior predictive event probability. To isolate epistemic risk
        uncertainty, additionally integrate many future paths per parameter
        draw. This method does not impose physical ordering constraints.
        """
        from ..dependence.probability import pair_probability_draws
        return pair_probability_draws(self,events,operation=operation,rtol=rtol)

    def log_score(
        self,
        observed: Array,
        *,
        channel: str | None = None,
    ) -> Array:
        """Negative log posterior-predictive density at each horizon."""

        return log_predictive_score(
            self.conditional_log_density(observed, channel=channel)
        )

    def conditional_cdf(self, observed: Array, *, channel: str | None = None) -> Array:
        """Original-scale marginal CDF for each posterior draw and time block."""

        index = self._channel_index(channel)
        observed_array = np.asarray(observed, dtype=float).reshape(-1)
        if observed_array.size != self.horizon:
            raise ValueError("observed must have length equal to the forecast horizon.")
        if self.is_multiseries_forecast:
            assert channel is not None
            observation_model = self.observation_model[channel]
            sign = float(np.asarray(self.transform_sign)[index])
            eta_model = sign * self.eta[:, :, index]
            sigma = self.parameters.get(f"sigma_path.{channel}", self.parameters[f"sigma.{channel}"][:, None])
            xi = (
                self.parameters[f"xi.{channel}"][:, None]
                if getattr(observation_model, "name", None) == "gev"
                else None
            )
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta_model = sign * self.eta
            sigma = self.parameters.get("sigma_path")
            if sigma is None:
                sigma = self.parameters["sigma"][:, None]
            xi = self.parameters.get("xi")
            if xi is not None:
                xi = np.asarray(xi, dtype=float)[:, None]
        cdf = observation_model.cdf(
            sign * observed_array[None, :],
            eta_model,
            sigma=sigma,
            xi=xi,
        )
        cdf = np.asarray(cdf, dtype=float)
        # If W=-Y is modelled for a lower extreme, F_Y(y)=1-F_W(-y) for the
        # continuous Gaussian/GEV families used here.
        if sign < 0.0:
            cdf = 1.0 - cdf
        return cdf

    def pit(self, observed: Array, *, channel: str | None = None) -> Array:
        """Posterior-predictive marginal PIT for held-out observations."""
        return np.mean(self.conditional_cdf(observed, channel=channel), axis=0)

    def pit_diagnostics(
        self,
        observed: Array,
        *,
        channel: str | None = None,
        bins: int = 10,
    ):
        """Return histogram, uniformity, and lag-one PIT diagnostics."""

        return pit_diagnostics(self.pit(observed, channel=channel), bins=bins)

    def score(
        self,
        observed: Array,
        *,
        thresholds: list[float] | tuple[float, ...] = (),
        quantiles: list[float] | tuple[float, ...] = (0.9, 0.95, 0.99),
        channel: str | None = None,
        aggregate: bool = True,
    ):
        if self.is_multiseries_forecast and channel is None:
            observed_array = np.asarray(observed, dtype=float)
            if observed_array.shape != (self.horizon, len(self.channel_names)):
                raise ValueError(
                    "Multiseries observed values must have shape (horizon, n_channels), "
                    "or choose channel=."
                )
            tables = []
            for index, name in enumerate(self.channel_names):
                table = evaluate_ensemble(
                    self.observations[:, :, index],
                    observed_array[:, index],
                    thresholds=thresholds,
                    quantiles=quantiles,
                    tail=self.tail[index],
                    log_density_draws=self.conditional_log_density(
                        observed_array[:, index], channel=name
                    ),
                    aggregate=aggregate,
                )
                if hasattr(table, "assign"):
                    table = table.assign(channel=name)
                else:
                    for row in table:
                        row["channel"] = name
                tables.append(table)
            try:
                import pandas as pd

                return pd.concat(tables, ignore_index=True)
            except ImportError:
                return [row for table in tables for row in table]
        index = self._channel_index(channel)
        observed_array = np.asarray(observed, dtype=float).reshape(-1)
        if observed_array.size != self.horizon:
            raise ValueError("observed must have length equal to the forecast horizon.")
        ensemble = self.observations[:, :, index] if self.is_multiseries_forecast else self.observations
        tail = self.tail[index] if self.is_multiseries_forecast else self.tail
        return evaluate_ensemble(
            ensemble,
            observed_array,
            thresholds=thresholds,
            quantiles=quantiles,
            tail=tail,
            log_density_draws=self.conditional_log_density(
                observed_array, channel=channel
            ),
            aggregate=aggregate,
        )

    def plot(
        self,
        *,
        level: float = 0.90,
        channel: str | None = None,
        target: str = "observations",
        phase: int | None = None,
        phase_label: str | None = None,
        ax=None,
        color: str = "C0",
        observed=None,
        history=None,
        history_dates=None,
        history_points: int | None = None,
        show_eta: bool = False,
        title: str | None = None,
        ylabel: str | None = None,
        save=None,
    ):
        """Plot a posterior predictive sample or an out-of-sample forecast.

        Supply ``observed`` for an in-sample posterior predictive check.
        Supply ``history`` (and, for dated fits, ``history_dates``) to place
        recent observations immediately before a forecast.  The optional
        ``history_points`` argument keeps only the most recent values.

        ``target="level"`` plots the seasonally adjusted latent level without
        observation noise. ``phase=`` retains one one-based seasonal phase,
        for example ``phase=7`` for July in a dated monthly annual cycle.
        """

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        if self.is_multiseries_forecast and channel is None:
            channel = self.channel_names[0]
        target_key = str(target).lower().replace("-", "_")
        selected_horizons = self._phase_indices(phase)
        summary = self.summary(
            level=level,
            channel=channel,
            target=target_key,
            phase=phase,
        )
        if hasattr(summary, "columns"):
            x = summary["time"]
            lower = summary["lower"].to_numpy()
            median = summary["median"].to_numpy()
            upper = summary["upper"].to_numpy()
        else:
            x = np.asarray([row["time"] for row in summary])
            lower = np.asarray([row["lower"] for row in summary])
            median = np.asarray([row["median"] for row in summary])
            upper = np.asarray([row["upper"] for row in summary])

        def selected_values(values, *, label: str):
            array = np.asarray(values, dtype=float)
            if self.is_multiseries_forecast:
                index = self._channel_index(channel)
                if array.ndim == 2:
                    if array.shape[1] != len(self.channel_names):
                        raise ValueError(
                            f"{label} must have one column per forecast channel."
                        )
                    array = array[:, index]
                elif array.ndim != 1:
                    raise ValueError(f"{label} must be one- or two-dimensional.")
            return array.reshape(-1)

        if history is not None:
            history_values = selected_values(history, label="history")
            if history_dates is None:
                if np.issubdtype(np.asarray(x).dtype, np.number):
                    history_x = np.arange(-history_values.size, 0) + float(
                        np.asarray(x)[0]
                    )
                else:
                    raise ValueError(
                        "history_dates are required when forecast dates are not numeric."
                    )
            else:
                history_x = np.asarray(history_dates).reshape(-1)
                if history_x.size != history_values.size:
                    raise ValueError("history_dates and history must have the same length.")
            if history_points is not None:
                history_points = int(history_points)
                if history_points < 1:
                    raise ValueError("history_points must be positive.")
                history_values = history_values[-history_points:]
                history_x = history_x[-history_points:]
            if phase is not None:
                if self.period is None:
                    raise ValueError("phase= requires a seasonal forecast.")
                first_future_phase = (
                    int(np.asarray(self.phases, dtype=int).reshape(-1)[0])
                    if self.phases is not None
                    else 1
                )
                history_phases = (
                    first_future_phase
                    - history_values.size
                    - 1
                    + np.arange(history_values.size)
                ) % int(self.period) + 1
                selected_history = history_phases == int(phase)
                history_values = history_values[selected_history]
                history_x = history_x[selected_history]
            if target_key in {"level", "latent_level", "seasonally_adjusted"}:
                ax.plot(
                    history_x,
                    history_values,
                    color="0.45",
                    linewidth=1.0,
                    label=r"$\hat{\alpha}_t$ (history)",
                )
            else:
                ax.scatter(
                    history_x,
                    history_values,
                    s=10,
                    color="0.55",
                    alpha=0.55,
                    label=r"$y_t$ (history)",
                )
            ax.axvline(np.asarray(x)[0], color="0.45", linestyle=":", linewidth=1.0)

        latent_target = target_key not in {"observation", "observations", "predictive"}
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.2,
            label=f"{level:.0%} {'credible' if latent_target else 'predictive'} interval",
        )
        line_label = {
            "eta": r"$\hat{\mu}_t$",
            "predictor": r"$\hat{\mu}_t$",
            "latent_predictor": r"$\hat{\mu}_t$",
            "level": r"$\hat{\alpha}_t$",
            "latent_level": r"$\hat{\alpha}_t$",
            "seasonally_adjusted": r"$\hat{\alpha}_t$",
        }.get(target_key, "posterior predictive median")
        ax.plot(x, median, color=color, label=line_label)
        if show_eta:
            if latent_target:
                raise ValueError("show_eta is only valid for target='observations'.")
            eta_values = (
                self.eta[:, :, self._channel_index(channel)]
                if self.is_multiseries_forecast
                else self.eta
            )
            ax.plot(
                x,
                np.median(eta_values[:, selected_horizons], axis=0),
                color=color,
                linestyle="--",
                linewidth=1.0,
                label=r"$\hat{\mu}_t$",
            )
        if observed is not None:
            observed_values = selected_values(observed, label="observed")
            if observed_values.size == self.horizon:
                observed_values = observed_values[selected_horizons]
            elif observed_values.size != selected_horizons.size:
                raise ValueError(
                    "observed must have length equal to the predictive horizon "
                    "or the selected phase."
                )
            ax.scatter(
                x,
                observed_values,
                s=10,
                color="0.35",
                alpha=0.6,
                label=r"$y_t$",
            )
        if target_key in {"level", "latent_level", "seasonally_adjusted"}:
            default_title = "Latent level forecast"
        elif target_key in {"eta", "predictor", "latent_predictor"}:
            default_title = "Latent predictor forecast"
        else:
            default_title = (
                "Posterior predictive check"
                if observed is not None
                else "Posterior predictive forecast"
            )
        if channel is not None:
            default_title = f"{default_title}: {channel}"
        if phase is not None:
            default_title = f"{default_title}: {phase_label or f'phase {phase}'}"
        if title is not None:
            ax.set_title(str(title))
        if ylabel is not None:
            ax.set_ylabel(str(ylabel))
        ax.legend()
        if save is not None:
            options = {}
            if isinstance(save, dict):
                options = dict(save)
                try:
                    path = options.pop("path")
                except KeyError as exc:
                    raise ValueError(
                        "A save mapping requires a 'path' entry."
                    ) from exc
            else:
                path = save
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            options.setdefault("bbox_inches", "tight")
            ax.figure.savefig(path, **options)
        return ax


def posterior_predictive(
    fit,
    *,
    draws: int | None = None,
    seed: int | None = None,
) -> Forecast:
    """Simulate replicated observations at the fitted time points.

    This is an in-sample posterior predictive check: latent states and static
    observation parameters are taken jointly from retained posterior draws,
    and a fresh observation is generated from each draw.  It is distinct from
    :func:`posterior_predict`, which propagates the state equation forward.
    """

    rng = np.random.default_rng(seed)
    total = fit.n_draws
    n_draws = total if draws is None else int(draws)
    if n_draws < 1:
        raise ValueError("draws must be positive.")
    indices = (
        np.arange(total)
        if n_draws == total
        else rng.choice(total, size=n_draws, replace=n_draws > total)
    )
    flat_states = fit.state_draws.reshape((total,) + fit.state_draws.shape[2:])
    state_paths = flat_states[indices, 1:].copy()
    eta_model = np.asarray(
        fit.eta_draws(combine_chains=True, original_scale=False), dtype=float
    )[indices]
    resolved_dates = (
        np.arange(fit.n_time)
        if fit.dates is None
        else np.asarray(fit.dates).reshape(-1)
    )

    if getattr(fit, "is_multiseries_model", False):
        required_parameters = tuple(fit.compiled.observation_parameter_names)
        missing = [
            name for name in required_parameters if name not in fit.parameter_draws
        ]
        if missing:
            raise ValueError(f"Fit is missing predictive parameters: {missing}")
        parameter_values = {
            name: fit.parameter(name)[indices] for name in required_parameters
        }
        parameter_values.update(_seasonal_scale_paths(fit, indices, fit.n_time, resolved_dates))
        observations_model = np.zeros_like(eta_model)
        for draw in range(n_draws):
            params = {
                name: float(values[draw]) for name, values in parameter_values.items() if values.ndim == 1
            }
            for time in range(fit.n_time):
                observations_model[draw, time] = fit.compiled.sample_observation(
                    eta_model[draw, time], _time_parameters(params, parameter_values, draw, time), rng
                )
        signs = np.asarray(fit.transform_sign, dtype=float)
        return Forecast(
            observations=observations_model * signs[None, None, :],
            eta=eta_model * signs[None, None, :],
            states=state_paths,
            parameters=parameter_values,
            dates=resolved_dates,
            family=fit.family,
            tail=tuple("lower" if sign < 0.0 else "upper" for sign in signs),
            observation_model=fit.model.observations,
            transform_sign=signs,
            channel_names=fit.channel_names,
            copula=getattr(fit.model, "copula", None),
            channels=tuple(fit.model.channels),
            state_names=fit.state_names,
            component_designs=_component_designs(fit, fit.n_time),
            period=fit.model.period,
            phases=seasonal_phases(fit.model.period, fit.n_time, resolved_dates),
        )

    required_parameters = ("sigma",) + (("xi",) if fit.family == "gev" else ())
    missing = [name for name in required_parameters if name not in fit.parameter_draws]
    if missing:
        raise ValueError(f"Fit is missing predictive parameters: {missing}")
    parameter_values = {
        name: fit.parameter(name)[indices] for name in required_parameters
    }
    sigma_path = fit.sigma_draws()[indices]
    parameter_values["sigma_path"] = sigma_path
    if fit.family == "gev":
        parameter_values["phi"] = fit.phi_draws()[indices]
        parameter_values["sigma_path"] = sigma_path
        for name in (
            "phi_intercept",
            "phi_slope",
            "phi_rw_sd",
            "phi_rw_variance",
            "phi_last",
            "phi_model",
        ):
            if name in fit.parameter_draws:
                parameter_values[name] = fit.parameter(name)[indices]
    observations_model = np.zeros_like(eta_model)
    for draw in range(n_draws):
        observations_model[draw] = fit.model.observation.sample(
            eta=eta_model[draw],
            sigma=sigma_path[draw],
            xi=(
                float(parameter_values["xi"][draw])
                if "xi" in parameter_values
                else None
            ),
            rng=rng,
        )
    sign = float(fit.transform_sign)
    return Forecast(
        observations=sign * observations_model,
        eta=sign * eta_model,
        states=state_paths,
        parameters=parameter_values,
        dates=resolved_dates,
        family=fit.family,
        tail="lower" if sign < 0.0 else "upper",
        observation_model=fit.model.observation,
        transform_sign=sign,
        state_names=fit.state_names,
        component_designs=_component_designs(fit, fit.n_time),
        period=fit.model.period,
        phases=seasonal_phases(fit.model.period, fit.n_time, resolved_dates),
    )


def posterior_predict(
    fit,
    horizon: int,
    *,
    exog_future=None,
    draws: int | None = None,
    seed: int | None = None,
    dates: Array | None = None,
) -> Forecast:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    requires_exog = (
        any(channel.n_exog for channel in fit.model.channels)
        if fit.is_multiseries_model else bool(fit.model.n_exog)
    )
    if requires_exog and exog_future is None:
        raise ValueError(
            "exog_future is required when forecasting a regression model; "
            "supply future covariates explicitly, even when the forecast "
            "horizon equals the training length."
        )
    rng = np.random.default_rng(seed)
    total = fit.n_draws
    n_draws = total if draws is None else int(draws)
    if n_draws < 1:
        raise ValueError("draws must be positive.")
    indices = np.arange(total) if n_draws == total else rng.choice(total, size=n_draws, replace=n_draws > total)
    flat_states = fit.state_draws.reshape((total,) + fit.state_draws.shape[2:])
    resolved_dates = (np.arange(fit.n_time, fit.n_time+horizon) if dates is None and fit.dates is None
                      else _future_dates(fit.dates, horizon) if dates is None else np.asarray(dates))
    if np.asarray(resolved_dates).reshape(-1).size != horizon:
        raise ValueError("dates must have length equal to horizon.")
    if getattr(fit, "is_multiseries_model", False):
        required_parameters = [
            *(f"sd.{name}" for name in fit.compiled.noise_names),
            *fit.compiled.observation_parameter_names,
        ]
        missing = [name for name in required_parameters if name not in fit.parameter_draws]
        if missing:
            raise ValueError(f"Fit is missing forecast parameters: {missing}")
        parameter_values = {
            name: fit.parameter(name)[indices] for name in required_parameters
        }
        parameter_values.update(_seasonal_scale_paths(fit, indices, horizon, resolved_dates, start_index=fit.n_time, rng=rng))
        n_channels = len(fit.channel_names)
        state_paths = np.zeros((n_draws, horizon, fit.compiled.state_dim))
        eta_model = np.zeros((n_draws, horizon, n_channels))
        observations_model = np.zeros_like(eta_model)
        for draw, posterior_index in enumerate(indices):
            state = flat_states[posterior_index, -1].copy()
            params = {name: float(values[draw]) for name, values in parameter_values.items() if values.ndim == 1}
            process_sd = fit.compiled.process_vector(params)
            design = fit.compiled.design(horizon, exog=exog_future, params=params)
            for h in range(horizon):
                innovation = fit.compiled.loading @ (
                    process_sd * rng.normal(size=fit.compiled.noise_dim)
                )
                state = fit.compiled.transition @ state + innovation
                state_paths[draw, h] = state
                eta_model[draw, h] = design[h] @ state
                observations_model[draw, h] = fit.compiled.sample_observation(
                    eta_model[draw, h], _time_parameters(params, parameter_values, draw, h), rng
                )
        signs = np.asarray(fit.transform_sign, dtype=float)
        resolved_dates = (
            np.arange(fit.n_time, fit.n_time + horizon)
            if dates is None and fit.dates is None
            else _future_dates(fit.dates, horizon)
            if dates is None
            else np.asarray(dates)
        )
        if np.asarray(resolved_dates).reshape(-1).size != horizon:
            raise ValueError("dates must have length equal to horizon.")
        return Forecast(
            observations=observations_model * signs[None, None, :],
            eta=eta_model * signs[None, None, :],
            states=state_paths,
            parameters=parameter_values,
            dates=resolved_dates,
            family=fit.family,
            tail=tuple(
                "lower" if sign < 0.0 else "upper" for sign in signs
            ),
            observation_model=fit.model.observations,
            transform_sign=signs,
            channel_names=fit.channel_names,
            copula=getattr(fit.model, "copula", None),
            channels=tuple(fit.model.channels),
            state_names=fit.state_names,
            component_designs=_component_designs(fit, horizon),
            period=fit.model.period,
            phases=seasonal_phases(fit.model.period, horizon, resolved_dates, start_index=fit.n_time),
        )
    required_parameters = [
        *(f"sd.{name}" for name in fit.compiled.noise_names),
        "sigma",
        *(["xi"] if fit.family == "gev" else []),
    ]
    missing = [name for name in required_parameters if name not in fit.parameter_draws]
    if missing:
        raise ValueError(f"Fit is missing forecast parameters: {missing}")
    parameter_values = {
        name: fit.parameter(name)[indices]
        for name in required_parameters
    }
    phi_mode = str(getattr(fit.model.observation, "phi", "stationary"))
    if fit.family == "gev" and phi_mode == "stationary":
        future_phi = np.repeat(
            np.log(np.asarray(parameter_values["sigma"], dtype=float))[:, None],
            horizon,
            axis=1,
        )
        parameter_values["phi"] = future_phi
        parameter_values["sigma_path"] = np.exp(future_phi)
    elif fit.family == "gev":
        needed = {
            "linear": ("phi_intercept", "phi_slope"),
            "rw": ("phi_last", "phi_rw_sd"),
            "ssvs": (
                "phi_model",
                "phi_intercept",
                "phi_slope",
                "phi_last",
                "phi_rw_sd",
            ),
        }[phi_mode]
        missing_phi = [name for name in needed if name not in fit.parameter_draws]
        if missing_phi:
            raise ValueError(f"Fit is missing phi forecast parameters: {missing_phi}")
        for name in needed:
            parameter_values[name] = fit.parameter(name)[indices]

        future_phi = np.zeros((n_draws, horizon), dtype=float)
        basis = phi_future_basis(fit.n_time, horizon)
        model_codes = (
            np.asarray(parameter_values["phi_model"], dtype=int)
            if phi_mode == "ssvs"
            else np.full(
                n_draws,
                {"linear": 1, "rw": 2}[phi_mode],
                dtype=int,
            )
        )
        for draw in range(n_draws):
            code = int(model_codes[draw])
            if code == 0:
                future_phi[draw] = float(parameter_values["phi_intercept"][draw])
            elif code == 1:
                future_phi[draw] = (
                    float(parameter_values["phi_intercept"][draw])
                    + float(parameter_values["phi_slope"][draw]) * basis
                )
            elif code == 2:
                innovations = rng.normal(
                    scale=float(parameter_values["phi_rw_sd"][draw]),
                    size=horizon,
                )
                future_phi[draw] = float(parameter_values["phi_last"][draw]) + np.cumsum(
                    innovations
                )
            else:
                raise ValueError("phi_model draws must use codes 0, 1, or 2.")
        parameter_values["phi"] = future_phi
        parameter_values["sigma_path"] = np.exp(future_phi)
    parameter_values.update(_seasonal_scale_paths(fit, indices, horizon, resolved_dates, start_index=fit.n_time, rng=rng))
    if fit.model.observation.scale is not None and fit.family == "gev":
        parameter_values["phi"] = np.log(parameter_values["sigma_path"])
    design = fit.compiled.design(horizon, exog=exog_future)
    state_paths = np.zeros((n_draws, horizon, fit.compiled.state_dim))
    eta_model = np.zeros((n_draws, horizon))
    observations_model = np.zeros((n_draws, horizon))

    for draw, posterior_index in enumerate(indices):
        state = flat_states[posterior_index, -1].copy()
        params = {
            name: float(values[draw])
            for name, values in parameter_values.items()
            if np.asarray(values).ndim == 1
        }
        process_sd = fit.compiled.process_vector(params)
        for h in range(horizon):
            innovation = fit.compiled.loading @ (process_sd * rng.normal(size=fit.compiled.noise_dim))
            state = fit.compiled.transition @ state + innovation
            state_paths[draw, h] = state
            eta = float(design[h] @ state)
            eta_model[draw, h] = eta
            observations_model[draw, h] = float(
                fit.model.observation.sample(
                    eta=eta,
                    sigma=(
                        float(parameter_values["sigma_path"][draw, h])
                        if "sigma_path" in parameter_values
                        else params["sigma"]
                    ),
                    xi=params.get("xi"),
                    rng=rng,
                )
            )

    sign = float(fit.transform_sign)
    resolved_dates = (
        np.arange(fit.n_time, fit.n_time + horizon)
        if dates is None and fit.dates is None
        else _future_dates(fit.dates, horizon)
        if dates is None
        else np.asarray(dates)
    )
    if np.asarray(resolved_dates).reshape(-1).size != horizon:
        raise ValueError("dates must have length equal to horizon.")
    return Forecast(
        observations=sign * observations_model,
        eta=sign * eta_model,
        states=state_paths,
        parameters=parameter_values,
        dates=resolved_dates,
        family=fit.family,
        tail="lower" if sign < 0.0 else "upper",
        observation_model=fit.model.observation,
        transform_sign=sign,
        state_names=fit.state_names,
        component_designs=_component_designs(fit, horizon),
        period=fit.model.period,
        phases=seasonal_phases(fit.model.period, horizon, resolved_dates, start_index=fit.n_time),
    )
