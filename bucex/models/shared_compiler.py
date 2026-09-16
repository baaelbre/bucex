"""Compile private and shared components to the same linear Gaussian system."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from scipy.linalg import block_diag

from ..components import DummySeasonal, LocalLevel, LocalLinearTrend, Regression
from .compiler import _rect_block_diag
from .multiseries_compiler import CompiledMultiSeriesModel, as_multiseries_array, compile_multiseries_model
from .shared import Departures


def _oriented(component, sign):
    """Convert declared original-response initial means to internal orientation."""
    if sign == 1:
        return component
    if isinstance(component, LocalLevel):
        return replace(component, initial_mean=None if component.initial_mean is None else sign * component.initial_mean)
    if isinstance(component, LocalLinearTrend):
        return replace(component, initial_level=None if component.initial_level is None else sign * component.initial_level,
                       initial_slope=sign * component.initial_slope)
    if isinstance(component, (DummySeasonal, Regression)):
        return replace(component, initial_mean=None if component.initial_mean is None else tuple(sign * np.asarray(component.initial_mean)))
    raise TypeError(type(component).__name__)


@dataclass
class CompiledSharedModel(CompiledMultiSeriesModel):
    """Joint private/shared states with independent or copula-coupled errors.

    ``noise_names`` names unique SD parameters; ``innovation_names`` names the
    parameter associated with every physical noise column. A departure group
    repeats each parameter across its exchangeable orthonormal contrasts.
    """

    innovation_names: tuple[str, ...] = ()
    group_slices: Mapping[str, tuple[slice, ...]] = None
    group_loadings: Mapping[str, np.ndarray] = None
    group_components: Mapping[str, Any] = None
    group_kinds: Mapping[str, str] = None
    is_shared: bool = True

    @property
    def has_copula(self):
        return self.model.copula is not None

    @property
    def observation_parameter_names(self):
        names = super().observation_parameter_names
        if self.has_copula:
            names += self.model.copula.parameter_names(self.channel_names)
        return names

    def marginal_observation_log_likelihood(self, y, eta, params):
        """Sum of marginal log densities, before the optional copula correction."""
        return super().observation_log_likelihood(y, eta, params)

    def observation_variance(self, params, n_time=None):
        if not (self.has_copula and self.all_gaussian):
            return super().observation_variance(params, n_time)
        n = self.n_time if n_time is None else int(n_time)
        scales = np.asarray([float(params[f"sigma.{name}"]) for name in self.channel_names])
        correlation = self.model.copula.correlation_matrix(params, self.channel_names)
        covariance = correlation * scales[:, None] * scales[None, :]
        return np.broadcast_to(covariance, (n, *covariance.shape)).copy()

    def observation_derivatives(self, y, eta, params):
        if not self.has_copula:
            return super().observation_derivatives(y, eta, params)
        from ..dependence import copula_observation_derivatives
        correlation = self.model.copula.correlation_matrix(params, self.channel_names)
        return copula_observation_derivatives(y, eta, self.model.channels, params, correlation)

    def observation_log_likelihood(self, y, eta, params):
        marginal = super().observation_log_likelihood(y, eta, params)
        if not np.isfinite(marginal) or not self.has_copula:
            return marginal
        from ..dependence import copula_log_likelihood
        correlation = self.model.copula.correlation_matrix(params, self.channel_names)
        return marginal + copula_log_likelihood(y, eta, self.model.channels, params, correlation)

    def sample_observation(self, eta_t, params, rng):
        if not self.has_copula:
            return super().sample_observation(eta_t, params, rng)
        from ..dependence import sample_normal_scores, quantiles_from_normal_scores
        phase = params.get("__copula_phase")
        if self.model.copula.seasonal and phase is None:
            raise ValueError("Seasonal copula simulation requires an explicit calendar phase.")
        correlation = self.model.copula.correlation_matrix(params, self.channel_names, phase=phase)
        scores = np.asarray(sample_normal_scores(correlation, 1, rng)).reshape(len(self.channel_names))
        return quantiles_from_normal_scores(scores, np.asarray(eta_t), self.model.channels, params)

    def process_vector(self, params):
        values = np.asarray([float(params[f"sd.{name}"]) for name in self.innovation_names])
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Process SDs must be finite and nonnegative.")
        return values

    def design(self, n_time=None, exog=None, *, params=None):
        design = super().design(n_time, exog, params=params)
        signs = self.model.transform_signs
        for name in self.group_slices:
            component = self.group_components[name]
            if component.state_dim == 0:
                continue
            h, _ = component.design_matrices(1, {})
            for j, sl in enumerate(self.group_slices[name]):
                design[:, :, sl] = (signs * self.group_loadings[name][:, j])[None, :, None] * h[None, :, :]
        return design

    def contribution_design(self, name, n_time=None, *, original_scale=True):
        if name not in self.group_slices:
            raise KeyError(f"Unknown shared group {name!r}; choose {tuple(self.group_slices)}.")
        n = self.n_time if n_time is None else int(n_time)
        out = np.zeros((n, len(self.channel_names), self.state_dim))
        component = self.group_components[name]
        if component.state_dim:
            h, _ = component.design_matrices(1, {})
            signs = np.ones(len(self.channel_names)) if original_scale else self.model.transform_signs
            for j, sl in enumerate(self.group_slices[name]):
                out[:, :, sl] = (signs * self.group_loadings[name][:, j])[None, :, None] * h[None, :, :]
        return out

    def channel_component_design(self, channel, component="level", n_time=None, *, original_scale=True):
        if channel not in self.channel_names:
            raise KeyError(f"Unknown channel {channel!r}.")
        if component not in {"level", "slope", "seasonal", "season", "regression", "predictor", "eta"}:
            raise ValueError("component must be level, slope, seasonal, regression, or predictor.")
        n = self.n_time if n_time is None else int(n_time)
        index = self.channel_names.index(channel)
        sign = self.model.transform_signs[index]
        if component in {"predictor", "eta"}:
            out = self.design(n)[:, index]
            return out * sign if original_scale else out
        out = np.zeros((n, self.state_dim))
        block = next(block for block in self.blocks if block.name == channel)
        target = "seasonal" if component == "season" else component
        for local in block.compiled.model.components:
            key = block.compiled._component_key(local)
            sl = block.compiled.component_slices[key]
            if sl.start == sl.stop:
                continue
            position = block.state_slice.start + sl.start
            if target == "level" and isinstance(local, (LocalLevel, LocalLinearTrend)):
                out[:, position] = sign
            elif target == "slope" and isinstance(local, LocalLinearTrend) and local.state_dim == 2:
                out[:, position + 1] = sign
            elif target == "seasonal" and isinstance(local, DummySeasonal):
                out[:, position] = sign
            elif target == "regression" and isinstance(local, Regression):
                local_design = block.compiled.design(n, exog=None if self.exog is None else self.exog.get(channel))
                out[:, position:position + local.state_dim] = sign * local_design[:, sl]
        for name, group in self.group_components.items():
            offset = None
            if target == "level" and isinstance(group, (LocalLevel, LocalLinearTrend)):
                offset = 0
            elif target == "slope" and isinstance(group, LocalLinearTrend) and group.state_dim == 2:
                offset = 1
            elif target == "seasonal" and isinstance(group, DummySeasonal) and group.state_dim:
                offset = 0
            if offset is not None:
                for j, sl in enumerate(self.group_slices[name]):
                    out[:, sl.start + offset] = self.group_loadings[name][index, j]
        return out if original_scale else sign * out

    def project_path(self, path, params):
        path = np.asarray(path, dtype=float).copy()
        if path.ndim != 2 or path.shape[1] != self.state_dim:
            raise ValueError("path must have shape (T+1, state_dim).")
        # This compiler constructs diagonal initial priors. Small positive
        # variances are uncertain states, never deterministic constraints.
        fixed = np.diag(self.initial_cov) == 0.0
        path[0, fixed] = self.initial_mean[fixed]
        return super().project_path(path, params)

    def support_feasible_path(self, y, path, params):
        """Deterministic intercept shift into all GEV endpoint constraints.

        Used only to initialize a Laplace proposal from data and parameters.
        An intercept is movable only when its initial prior permits it. Every
        shift preserves transition support; no posterior draw is repaired.
        """
        out = np.asarray(path, dtype=float).copy()
        eta = self.eta(out, params=params)
        for j, channel in enumerate(self.model.channels):
            if channel.family != "gev":
                continue
            xi = float(params[f"xi.{channel.name}"])
            if abs(xi) < 1e-10:
                continue
            sigma = float(params[f"sigma.{channel.name}"])
            mask = np.isfinite(y[:, j])
            bounds = np.asarray(y)[mask, j] + sigma / xi - eta[mask, j]
            margin = max(1e-7, 0.01 * sigma)
            delta = max(0.0, float(np.max(bounds)) + margin) if xi < 0 else min(0.0, float(np.min(bounds)) - margin)
            if delta == 0.0:
                continue
            block = next(b for b in self.blocks if b.name == channel.name)
            sl = block.compiled.component_slices.get("trend")
            if sl is None or sl.start == sl.stop:
                raise FloatingPointError(f"No private intercept can initialize GEV support for {channel.name}.")
            position = block.state_slice.start + sl.start
            if self.initial_cov[position, position] <= 0:
                raise FloatingPointError(f"Fixed intercept prevents deterministic GEV support initialization for {channel.name}.")
            out[:, position] += delta
        return out

    def with_data(self, y, exog=None):
        return compile_shared_model(self.model, y, exog=exog)


def compile_shared_model(model, y, exog=None):
    values = as_multiseries_array(y, model)
    # Preserve the established internal orientation of private channel states.
    # Shared states and contrast coordinates use original response units.
    channels = tuple(replace(channel, components=tuple(_oriented(c, channel.transform_sign) for c in channel.components))
                     for channel in model.channels)
    base = compile_multiseries_model(replace(model, channels=channels, shared=(), copula=None), values, exog=exog)
    transitions, loadings, covariances = [base.transition], [base.loading], [base.initial_cov]
    means = [base.initial_mean]
    state_names, innovation_names = list(base.state_names), list(base.noise_names)
    group_slices, group_loadings, group_components, group_kinds = {}, {}, {}, {}
    process_reference_scale = dict(base.process_reference_scale)
    position = base.state_dim
    for item in model.shared:
        component = item.component
        kind = "departure" if isinstance(item, Departures) else "shared"
        mixing = item.loading_matrix(model.channel_names)
        unit = {f"sd.{name}": 1.0 for name in component.spec.noise_names}
        transition, loading, covariance, offset = component.system_matrices(1, unit)
        if np.any(offset) or (covariance.size and not np.allclose(covariance, np.eye(component.noise_dim))):
            raise ValueError("Shared components require named independent unit innovations and zero offsets.")
        mean, variance = component.initial_mean_var({})
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(variance)) or np.any(variance < 0):
            raise ValueError("Shared initial priors must have finite means and nonnegative variances.")
        slices = []
        parameter_names = [f"{kind}.{item.name}.{name}" for name in component.spec.noise_names]
        for j in range(mixing.shape[1]):
            sl = slice(position, position + component.state_dim)
            position = sl.stop
            slices.append(sl)
            transitions.append(transition)
            loadings.append(loading)
            covariances.append(np.diag(variance))
            means.append(mean)
            prefix = f"{kind}.{item.name}" + (f".contrast{j+1}" if kind == "departure" else "")
            state_names.extend(f"{prefix}.{name}" for name in component.spec.state_names)
            innovation_names.extend(parameter_names)
        group_slices[item.name] = tuple(slices)
        group_loadings[item.name] = mixing
        group_components[item.name] = component
        group_kinds[item.name] = kind
        process_reference_scale.update({name: base.difference_scale for name in parameter_names})
    payload = dict(base.__dict__)
    payload.update(model=model, transition=block_diag(*transitions), loading=_rect_block_diag(loadings),
                   initial_mean=np.concatenate(means), initial_cov=block_diag(*covariances),
                   state_names=tuple(state_names), noise_names=tuple(dict.fromkeys(innovation_names)),
                   innovation_names=tuple(innovation_names), group_slices=group_slices,
                   group_loadings=group_loadings, group_components=group_components, group_kinds=group_kinds,
                   process_reference_scale=process_reference_scale)
    return CompiledSharedModel(**payload)


__all__ = ["CompiledSharedModel", "compile_shared_model"]
