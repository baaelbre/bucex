"""Posterior predictive simulation and forecast summaries."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..diagnostics.calibration import pit_diagnostics
from ..diagnostics.scores import evaluate_ensemble, log_predictive_score


Array = np.ndarray


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
            else np.arange(self.horizon, dtype=int) % int(self.period) + 1
        )
        if phases.size != self.horizon:
            raise ValueError("Forecast phases must have length equal to horizon.")
        indices = np.flatnonzero(phases == selected_phase)
        if indices.size == 0:
            raise ValueError(
                f"phase {selected_phase} is not present in this forecast horizon."
            )
        return indices

    def component_draws(self, name: str) -> Array:
        """Return one latent state component on the response orientation.

        Component forecasts are currently defined for univariate models.  In
        particular, ``component_draws("level")`` is the seasonally adjusted
        structural trajectory: it excludes both the seasonal state and future
        observation noise.
        """

        if self.is_multiseries_forecast:
            raise ValueError(
                "Latent component forecasts are currently available only for "
                "univariate models."
            )
        if name not in self.state_names:
            raise KeyError(
                f"Unknown forecast state '{name}'. Available: {self.state_names}"
            )
        values = np.asarray(self.states[..., self.state_names.index(name)], dtype=float)
        return float(self.transform_sign) * values

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
            return self.component_draws("level")
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

    def return_level(self, return_period: float, *, channel: str | None = None) -> Array:
        index = self._channel_index(channel)
        if self.is_multiseries_forecast:
            assert channel is not None
            observation_model = self.observation_model[channel]
            if getattr(observation_model, "name", None) != "gev":
                raise ValueError("Return levels require a GEV channel.")
            sign = float(np.asarray(self.transform_sign)[index])
            eta = self.eta[:, :, index]
            sigma = self.parameters[f"sigma.{channel}"][:, None]
            xi = self.parameters[f"xi.{channel}"][:, None]
        elif self.family != "gev":
            raise ValueError("Return levels require a GEV forecast.")
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta = self.eta
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
            sigma = self.parameters[f"sigma.{channel}"][:, None]
            xi = (
                self.parameters[f"xi.{channel}"][:, None]
                if getattr(observation_model, "name", None) == "gev"
                else None
            )
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta_model = sign * self.eta
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

    def pit(self, observed: Array, *, channel: str | None = None) -> Array:
        """Posterior-predictive PIT for held-out observations."""

        index = self._channel_index(channel)
        observed_array = np.asarray(observed, dtype=float).reshape(-1)
        if observed_array.size != self.horizon:
            raise ValueError("observed must have length equal to the forecast horizon.")
        if self.is_multiseries_forecast:
            assert channel is not None
            observation_model = self.observation_model[channel]
            sign = float(np.asarray(self.transform_sign)[index])
            eta_model = sign * self.eta[:, :, index]
            sigma = self.parameters[f"sigma.{channel}"][:, None]
            xi = (
                self.parameters[f"xi.{channel}"][:, None]
                if getattr(observation_model, "name", None) == "gev"
                else None
            )
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta_model = sign * self.eta
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
        return np.mean(cdf, axis=0)

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
        for example ``phase=7`` for July when a monthly fit starts in January.
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
                    label="latent level history",
                )
            else:
                ax.scatter(
                    history_x,
                    history_values,
                    s=10,
                    color="0.55",
                    alpha=0.55,
                    label="observed history",
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
            "eta": "predictor median",
            "predictor": "predictor median",
            "latent_predictor": "predictor median",
            "level": "latent level median",
            "latent_level": "latent level median",
            "seasonally_adjusted": "latent level median",
        }.get(target_key, "predictive median")
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
                label="latent predictor median",
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
                label="observed",
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
        ax.set_title(default_title if title is None else str(title))
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
        observations_model = np.zeros_like(eta_model)
        for draw in range(n_draws):
            params = {
                name: float(values[draw]) for name, values in parameter_values.items()
            }
            for time in range(fit.n_time):
                observations_model[draw, time] = fit.compiled.sample_observation(
                    eta_model[draw, time], params, rng
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
            state_names=fit.state_names,
            period=fit.model.period,
            phases=(
                np.arange(fit.n_time, dtype=int) % int(fit.model.period) + 1
                if fit.model.period is not None
                else None
            ),
        )

    required_parameters = ("sigma",) + (("xi",) if fit.family == "gev" else ())
    missing = [name for name in required_parameters if name not in fit.parameter_draws]
    if missing:
        raise ValueError(f"Fit is missing predictive parameters: {missing}")
    parameter_values = {
        name: fit.parameter(name)[indices] for name in required_parameters
    }
    observations_model = np.zeros_like(eta_model)
    for draw in range(n_draws):
        observations_model[draw] = fit.model.observation.sample(
            eta=eta_model[draw],
            sigma=float(parameter_values["sigma"][draw]),
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
        period=fit.model.period,
        phases=(
            np.arange(fit.n_time, dtype=int) % int(fit.model.period) + 1
            if fit.model.period is not None
            else None
        ),
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
    rng = np.random.default_rng(seed)
    total = fit.n_draws
    n_draws = total if draws is None else int(draws)
    if n_draws < 1:
        raise ValueError("draws must be positive.")
    indices = np.arange(total) if n_draws == total else rng.choice(total, size=n_draws, replace=n_draws > total)
    flat_states = fit.state_draws.reshape((total,) + fit.state_draws.shape[2:])
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
        n_channels = len(fit.channel_names)
        state_paths = np.zeros((n_draws, horizon, fit.compiled.state_dim))
        eta_model = np.zeros((n_draws, horizon, n_channels))
        observations_model = np.zeros_like(eta_model)
        for draw, posterior_index in enumerate(indices):
            state = flat_states[posterior_index, -1].copy()
            params = {name: float(values[draw]) for name, values in parameter_values.items()}
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
                    eta_model[draw, h], params, rng
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
            state_names=fit.state_names,
            period=fit.model.period,
            phases=(
                np.arange(fit.n_time, fit.n_time + horizon, dtype=int)
                % int(fit.model.period)
                + 1
                if fit.model.period is not None
                else None
            ),
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
    design = fit.compiled.design(horizon, exog=exog_future)
    state_paths = np.zeros((n_draws, horizon, fit.compiled.state_dim))
    eta_model = np.zeros((n_draws, horizon))
    observations_model = np.zeros((n_draws, horizon))

    for draw, posterior_index in enumerate(indices):
        state = flat_states[posterior_index, -1].copy()
        params = {name: float(values[draw]) for name, values in parameter_values.items()}
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
                    sigma=params["sigma"],
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
        period=fit.model.period,
        phases=(
            np.arange(fit.n_time, fit.n_time + horizon, dtype=int)
            % int(fit.model.period)
            + 1
            if fit.model.period is not None
            else None
        ),
    )
