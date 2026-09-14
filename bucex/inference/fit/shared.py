"""Joint posterior sampling for Gaussian/GEV models with shared latent states.

The state step is Gaussian FFBS or exact-invariant Laplace independence MH.
Mixed models additionally receive scheduled Gaussian-prior elliptical-slice
block refreshes. Scale updates alternate centered and disturbance representations. Optional Gaussian-copula dependence is included in every likelihood-bearing
update. No selection indicator or uncorrected non-Gaussian path draw is hidden
in this backend. Numerical errors abort; an ordinary MH rejection preserves the
previous valid state.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ...core.fit import FitResult
from ...core.numerics import singular_normal_logpdf
from ...priors.joint import JointPriors, resolve_joint_priors
from ...priors.process import FixedSD, InverseGammaVariance
from ...__about__ import __version__
from ..config import Laplace, MCMC, SharedSampler
from .shared_slice import gaussian_prior_blocks, elliptical_slice_block
from ..state.kalman import ffbs, kalman_filter, kalman_smoother
from ..state.laplace import (
    build_laplace_approximation, laplace_mh, observation_log_likelihood,
)


Array = np.ndarray


def _innovation_geometry(compiled: Any) -> tuple[Array, dict[str, Array]]:
    """Invert the fixed loading matrix, retaining repeated scale parameters."""
    loading = np.asarray(compiled.loading, dtype=float)
    names = tuple(getattr(compiled, "innovation_names", compiled.noise_names))
    if len(names) != loading.shape[1]:
        raise ValueError("innovation_names must label every physical innovation column.")
    if set(names) != set(compiled.noise_names):
        raise ValueError("noise_names must be the unique names in innovation_names.")
    if loading.shape[1] == 0:
        return np.empty((0, loading.shape[0])), {}
    if np.linalg.matrix_rank(loading) != loading.shape[1]:
        raise ValueError("Shared inference requires independent innovation columns; use an independent contrast basis.")
    inverse = np.linalg.pinv(loading)
    return inverse, {
        name: np.flatnonzero(np.asarray(names) == name) for name in compiled.noise_names
    }


def _innovations(path: Array, compiled: Any, inverse: Array) -> Array:
    residuals = path[1:] - path[:-1] @ compiled.transition.T
    epsilon = residuals @ inverse.T
    reconstruction = epsilon @ compiled.loading.T
    error = np.linalg.norm(residuals - reconstruction, axis=1)
    if np.any(error > 1e-7 * (1.0 + np.linalg.norm(residuals, axis=1))):
        raise FloatingPointError("State path is outside the declared transition support.")
    return epsilon


def _centered_sd_target(log_sd: float, prior: Any, count: int, sum_squares: float) -> float:
    """Log density w.r.t. log(SD), including its Jacobian."""
    if not np.isfinite(log_sd) or abs(log_sd) > 350.0:
        return -np.inf
    sd = float(np.exp(log_sd))
    prior_value = float(prior.logpdf(sd))
    if not np.isfinite(prior_value):
        return -np.inf
    return float(prior_value + log_sd - count * log_sd - 0.5 * sum_squares / (sd * sd))


def _draw_variance(prior: InverseGammaVariance, count: int, sum_squares: float, rng) -> float:
    """Exact IG variance conditional; repeated contrast innovations all count."""
    precision = rng.gamma(prior.shape + 0.5 * count, 1.0 / (prior.scale + 0.5 * sum_squares))
    if not np.isfinite(precision) or precision <= 0.0:
        raise FloatingPointError("Invalid conjugate variance draw.")
    return float(np.sqrt(1.0 / precision))


def _accept(log_new: float, log_old: float, rng) -> bool:
    if not np.isfinite(log_old):
        raise FloatingPointError("The current Markov state has zero or nonfinite density.")
    if np.isnan(log_new) or np.isposinf(log_new):
        raise FloatingPointError("A proposed log density is numerically invalid.")
    return bool(np.isfinite(log_new) and np.log(rng.random()) < min(0.0, log_new - log_old))


def _log_parameter_prior(params: Mapping[str, float], priors: JointPriors, compiled=None) -> float:
    result = sum(prior.logpdf(params[f"sd.{key}"]) for key, prior in priors.process.items())
    result += sum(prior.logpdf(params[f"sigma.{key}"]) for key, prior in priors.observation_sd.items())
    result += sum(prior.logpdf(params[f"xi.{key}"]) for key, prior in priors.shape.items())
    if compiled is not None and getattr(compiled, "has_copula", False):
        result += compiled.model.copula.log_prior(params, compiled.channel_names)
    return float(result)


def _log_state_density(path: Array, params: Mapping[str, float], compiled: Any, inverse: Array) -> float:
    """Transition density on its affine support without truncating small SDs.

    A fixed loading matrix maps independent innovations into states. Its active
    Gram determinant supplies the Hausdorff-volume term for singular states.
    """
    epsilon = _innovations(path, compiled, inverse)
    sd = compiled.process_vector(params)
    active = sd > 0.0
    if np.any(np.abs(epsilon[:, ~active]) > 1e-7):
        return -np.inf
    value = singular_normal_logpdf(path[0], compiled.initial_mean, compiled.initial_cov)
    if not np.any(active):
        return float(value)
    loading = compiled.loading[:, active]
    sign, logdet = np.linalg.slogdet(loading.T @ loading)
    if sign <= 0.0:
        raise FloatingPointError("Active innovation loading has singular Gram matrix.")
    scaled = epsilon[:, active] / sd[active][None, :]
    value -= 0.5 * (
        epsilon.shape[0] * (int(active.sum()) * np.log(2.0 * np.pi) + 2.0 * np.log(sd[active]).sum() + logdet)
        + float(np.square(scaled).sum())
    )
    return float(value)


def _initial_parameters(priors: JointPriors, supplied: Mapping[str, Any] | None, compiled=None) -> dict[str, float]:
    params = {f"sd.{key}": float(prior.initial()) for key, prior in priors.process.items()}
    params.update({f"sigma.{key}": float(prior.initial()) for key, prior in priors.observation_sd.items()})
    params.update({f"xi.{key}": float(prior.initial()) for key, prior in priors.shape.items()})
    if compiled is not None and getattr(compiled, "has_copula", False):
        params.update(compiled.model.copula.initial_parameters(compiled.channel_names))
    if supplied:
        unknown = set(supplied) - {"parameters", "state_path", "__warm_start__", "chains"}
        if unknown:
            raise ValueError(f"Unknown shared initialization fields: {sorted(unknown)}; put scalar values in parameters.")
        actual = supplied.get("parameters", {})
        if set(actual) - set(params):
            raise ValueError(f"Unknown shared initial parameters: {sorted(set(actual) - set(params))}.")
        params.update({key: float(value) for key, value in actual.items()})
    for prefix, collection in (("sd", priors.process), ("sigma", priors.observation_sd)):
        for name, prior in collection.items():
            if isinstance(prior, FixedSD) and params[f"{prefix}.{name}"] != prior.value:
                raise ValueError(f"Initial {prefix}.{name} must equal its FixedSD value exactly.")
    if not np.isfinite(_log_parameter_prior(params, priors, compiled)):
        raise ValueError("Initial parameters are outside their prior support.")
    if any(value <= 0.0 for key, value in params.items() if key.startswith("sigma.")):
        raise ValueError("Initial observation scales must be positive.")
    if any(value <= 0.0 and not isinstance(priors.process[key[3:]], FixedSD)
           for key, value in params.items() if key.startswith("sd.")):
        raise ValueError("Nonfixed process scales must start strictly above zero.")
    return params


def _initial_path(y, compiled, params, plan, laplace, supplied, inverse):
    if supplied is not None and "state_path" in supplied:
        path = np.asarray(supplied["state_path"], dtype=float).copy()
        if path.shape != (compiled.n_time + 1, compiled.state_dim) or not np.all(np.isfinite(path)):
            raise ValueError("Initial state_path must be finite with shape (T+1, state_dim).")
    elif plan.engine == "ffbs":
        filtered = kalman_filter(y, compiled, params)
        path = compiled.project_path(kalman_smoother(filtered, compiled).mean, params)
    else:
        approximation = build_laplace_approximation(
            y, compiled, params, max_iterations=laplace.max_iterations,
            tolerance=laplace.tolerance, curvature_floor=laplace.curvature_floor,
            maximum_variance=laplace.maximum_variance,
        )
        path = approximation.mode_path.copy()
    density = _log_state_density(path, params, compiled, inverse)
    likelihood = observation_log_likelihood(y, compiled.eta(path, params=params), compiled, params)
    if not np.isfinite(density + likelihood):
        raise ValueError("Initial path lies outside the initial, transition, or observation support.")
    return path


def _channel_log_likelihood(y: Array, eta: Array, compiled: Any, params, index: int) -> float:
    if getattr(compiled, "has_copula", False):
        return observation_log_likelihood(y, eta, compiled, params)
    channel = compiled.model.channels[index]
    mask = np.isfinite(y[:, index])
    values = channel.observation.logpdf(
        y[mask, index], eta[mask, index], sigma=params[f"sigma.{channel.name}"],
        xi=params.get(f"xi.{channel.name}"),
    )
    if not np.all(np.isfinite(values)):
        return -np.inf
    return float(np.sum(values))


def _noncentered_scale_step(y, path, params, name, prior, width, compiled, inverse, groups, rng):
    """MH for SD conditional on x0 and standardized innovation trajectories.

    At fixed standardized disturbances their N(0,1) density cancels. The target
    is observation likelihood times the SD prior and the log-SD Jacobian.
    Repeated physical columns are rescaled together by their common SD.
    """
    old_sd = params[f"sd.{name}"]
    old_log = float(np.log(old_sd))
    new_log = old_log + rng.normal(scale=width)
    if abs(new_log) > 350.0:
        return path, False
    new_sd = float(np.exp(new_log))
    log_prior_new = float(prior.logpdf(new_sd))
    if not np.isfinite(log_prior_new):
        return path, False
    eps = _innovations(path, compiled, inverse)
    eps[:, groups[name]] *= new_sd / old_sd
    proposal = np.empty_like(path)
    proposal[0] = path[0]
    increments = eps @ compiled.loading.T
    for t in range(1, len(path)):
        proposal[t] = compiled.transition @ proposal[t - 1] + increments[t - 1]
    candidate_params = dict(params, **{f"sd.{name}": new_sd})
    old_target = observation_log_likelihood(y, compiled.eta(path, params=params), compiled, params)
    old_target += prior.logpdf(old_sd) + old_log
    new_target = observation_log_likelihood(y, compiled.eta(proposal, params=candidate_params), compiled, candidate_params)
    new_target += log_prior_new + new_log
    accepted = _accept(new_target, old_target, rng)
    if accepted:
        params[f"sd.{name}"] = new_sd
        return proposal, True
    return path, False


def sample_shared_posterior(
    y: Array, compiled: Any, priors: JointPriors | None, plan: Any, *,
    mcmc: MCMC, laplace: Laplace, dates=None, initial_parameters=None,
    sampler: SharedSampler | None = None,
) -> FitResult:
    """Sample a shared-state model using exact conditional transition kernels.

    ``initial_parameters`` accepts ``{'parameters': {...}, 'state_path': ...}``
    or ``{'chains': [one such mapping per chain]}``. Retained paths are always
    in scientific centered coordinates, irrespective of interweaving.
    """
    values = np.asarray(y, dtype=float)
    if values.shape != (compiled.n_time, len(compiled.channel_names)):
        raise ValueError("Shared observations must have shape (T, n_channels).")
    if np.any(np.isinf(values)) or np.any(np.all(np.isnan(values), axis=0)):
        raise ValueError("Each shared channel needs finite observations; use NaN for missing values.")
    if plan.engine not in {"ffbs", "laplace_mh"}:
        raise ValueError("Shared-state models support ffbs (Gaussian) or laplace_mh (Gaussian/GEV).")
    has_copula = bool(getattr(compiled, "has_copula", False))
    if (plan.engine == "ffbs") != (bool(compiled.all_gaussian) and not has_copula):
        raise ValueError("Use ffbs for all-Gaussian models without a copula and laplace_mh otherwise.")
    priors = resolve_joint_priors(compiled, priors)
    sampler = SharedSampler() if sampler is None else sampler
    if not isinstance(sampler, SharedSampler):
        raise TypeError("sampler must be SharedSampler(...).")
    slice_blocks = gaussian_prior_blocks(compiled) if plan.engine == "laplace_mh" and sampler.elliptical_slice_steps else ()
    if slice_blocks:
        plan = replace(plan, state_update=plan.state_update + " + scheduled Gaussian-prior block elliptical slice")
    inverse, groups = _innovation_geometry(compiled)
    chains = None if initial_parameters is None else initial_parameters.get("chains")
    if chains is not None and len(chains) != mcmc.chains:
        raise ValueError("Initial chains must contain one entry per requested chain.")
    seeds = np.random.SeedSequence(mcmc.seed).spawn(mcmc.chains)
    shape = (mcmc.chains, mcmc.draws)
    state_draws = np.empty((*shape, compiled.n_time + 1, compiled.state_dim))
    log_posterior, log_likelihood = np.empty(shape), np.empty(shape)
    parameter_draws = {name: np.empty(shape) for name in _initial_parameters(priors, None, compiled)}
    correlation_draws = np.empty((*shape, len(compiled.channel_names), len(compiled.channel_names))) if has_copula else None
    metrics = {key: np.full(shape, np.nan) for key in (
        "laplace_converged", "laplace_iterations", "laplace_relative_change",
        "laplace_support_rejections", "laplace_mh_acceptance",
        "laplace_mh_mean_log_acceptance_ratio", "laplace_mh_support_rejections",
        "state_ess_evaluations", "state_ess_blocks_moved", "state_ess_stochastic_blocks",
        "state_ess_mean_absolute_angle",
    )}
    acceptance: dict[str, Array] = {}
    initial_values = []
    final_widths = []
    started = perf_counter()
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        supplied = chains[chain] if chains is not None else initial_parameters
        params = _initial_parameters(priors, supplied, compiled)
        path = _initial_path(values, compiled, params, plan, laplace, supplied, inverse)
        initial_values.append({"parameters": dict(params), "state_path": path.copy()})
        widths = {
            **{f"centered.{key}": 0.15 for key, p in priors.process.items() if not isinstance(p, FixedSD)},
            **{f"noncentered.{key}": 0.15 for key, p in priors.process.items() if not isinstance(p, FixedSD)},
            **{f"sigma.{key}": 0.10 for key, p in priors.observation_sd.items() if not isinstance(p, FixedSD)},
            **{f"xi.{key}": 0.03 for key in priors.shape},
            **{key: 0.12 for key in (compiled.model.copula.parameter_names(compiled.channel_names) if has_copula else ())},
        }
        counts = {key: [0, 0] for key in widths}
        counts["state"] = [0, 0]
        saved = 0
        for iteration in range(mcmc.iterations):
            step_accept = {}
            try:
                if plan.engine == "ffbs":
                    path, _ = ffbs(values, compiled, params, rng)
                    iteration_metrics = {key: np.nan for key in metrics}
                    counts["state"][0] += 1
                    counts["state"][1] += 1
                else:
                    state = laplace_mh(
                        values, compiled, params, path, rng,
                        mh_steps=laplace.mh_steps, max_iterations=laplace.max_iterations,
                        tolerance=laplace.tolerance, curvature_floor=laplace.curvature_floor,
                        maximum_variance=laplace.maximum_variance,
                    )
                    path = state.path
                    iteration_metrics = {
                        "laplace_converged": float(state.approximation.converged),
                        "laplace_iterations": state.approximation.iterations,
                        "laplace_relative_change": state.approximation.relative_change,
                        "laplace_support_rejections": state.proposal_support_failures,
                        "laplace_mh_acceptance": state.acceptance_rate,
                        "laplace_mh_mean_log_acceptance_ratio": float(np.mean(state.log_acceptance_ratio)),
                        "laplace_mh_support_rejections": state.proposal_support_failures,
                    }
                    counts["state"][0] += state.accepted_steps
                    counts["state"][1] += state.attempts
                slice_evaluations, slice_moved, slice_stochastic, slice_angles = 0, 0, 0, []
                for _ in range(sampler.elliptical_slice_steps if slice_blocks else 0):
                    for block in slice_blocks:
                        refresh = elliptical_slice_block(
                            values, path, compiled, params, block, rng,
                            maximum_evaluations=sampler.maximum_slice_evaluations,
                        )
                        path = refresh.path
                        slice_evaluations += refresh.evaluations
                        slice_moved += int(refresh.moved)
                        slice_stochastic += int(refresh.stochastic)
                        if refresh.stochastic:
                            slice_angles.append(abs(refresh.angle))
                iteration_metrics.update(
                    state_ess_evaluations=slice_evaluations,
                    state_ess_blocks_moved=slice_moved,
                    state_ess_stochastic_blocks=slice_stochastic,
                    state_ess_mean_absolute_angle=float(np.mean(slice_angles)) if slice_angles else np.nan,
                )
                epsilon = _innovations(path, compiled, inverse)
                # Centered innovation conditionals: each reused SD sees every
                # physical innovation attached to it, including K-1 contrasts.
                for name, prior in priors.process.items():
                    if isinstance(prior, FixedSD):
                        continue
                    key = f"centered.{name}"
                    subset = epsilon[:, groups[name]]
                    count, ss = subset.size, float(np.square(subset).sum())
                    if isinstance(prior, InverseGammaVariance):
                        params[f"sd.{name}"] = _draw_variance(prior, count, ss, rng)
                        accepted = True
                    else:
                        old = float(np.log(params[f"sd.{name}"]))
                        proposal = old + rng.normal(scale=widths[key])
                        accepted = _accept(_centered_sd_target(proposal, prior, count, ss),
                                           _centered_sd_target(old, prior, count, ss), rng)
                        if accepted:
                            params[f"sd.{name}"] = float(np.exp(proposal))
                        step_accept[key] = accepted
                    counts[key][0] += int(accepted)
                    counts[key][1] += 1
                # Ancillary augmentation: improve mixing when the process
                # variance is small, without changing the model or public API.
                for name, prior in priors.process.items():
                    if isinstance(prior, FixedSD):
                        continue
                    key = f"noncentered.{name}"
                    path, accepted = _noncentered_scale_step(
                        values, path, params, name, prior, widths[key], compiled, inverse, groups, rng,
                    )
                    step_accept[key] = accepted
                    counts[key][0] += int(accepted)
                    counts[key][1] += 1
                eta = compiled.eta(path, params=params)
                for index, channel in enumerate(compiled.model.channels):
                    name, prior = channel.name, priors.observation_sd[channel.name]
                    key = f"sigma.{name}"
                    if not isinstance(prior, FixedSD):
                        if channel.family == "gaussian" and isinstance(prior, InverseGammaVariance) and not has_copula:
                            mask = np.isfinite(values[:, index])
                            ss = float(np.square(values[mask, index] - eta[mask, index]).sum())
                            params[key] = _draw_variance(prior, int(mask.sum()), ss, rng)
                            accepted = True
                        else:
                            old_log = float(np.log(params[key]))
                            new_log = old_log + rng.normal(scale=widths[key])
                            candidate = dict(params)
                            candidate[key] = float(np.exp(new_log)) if abs(new_log) < 350.0 else np.inf
                            old_target = _channel_log_likelihood(values, eta, compiled, params, index) + prior.logpdf(params[key]) + old_log
                            new_target = -np.inf
                            if np.isfinite(candidate[key]):
                                new_target = _channel_log_likelihood(values, eta, compiled, candidate, index) + prior.logpdf(candidate[key]) + new_log
                            accepted = _accept(new_target, old_target, rng)
                            if accepted:
                                params[key] = candidate[key]
                            step_accept[key] = accepted
                        counts[key][0] += int(accepted)
                        counts[key][1] += 1
                    if channel.family == "gev":
                        key, shape_prior = f"xi.{name}", priors.shape[name]
                        candidate = dict(params)
                        candidate[key] += rng.normal(scale=widths[key])
                        old_target = _channel_log_likelihood(values, eta, compiled, params, index) + shape_prior.logpdf(params[key])
                        new_target = shape_prior.logpdf(candidate[key])
                        if np.isfinite(new_target):
                            new_target += _channel_log_likelihood(values, eta, compiled, candidate, index)
                        accepted = _accept(new_target, old_target, rng)
                        if accepted:
                            params[key] = candidate[key]
                        step_accept[key] = accepted
                        counts[key][0] += int(accepted)
                        counts[key][1] += 1
                if has_copula:
                    copula = compiled.model.copula
                    # Symmetric random walks in unconstrained partial coordinates.
                    # log_prior includes both LKJ density and its full Jacobian.
                    for key in copula.parameter_names(compiled.channel_names):
                        candidate = dict(params)
                        candidate[key] += rng.normal(scale=widths[key])
                        old_target = observation_log_likelihood(values, eta, compiled, params)
                        old_target += copula.log_prior(params, compiled.channel_names)
                        new_target = copula.log_prior(candidate, compiled.channel_names)
                        if np.isfinite(new_target):
                            new_target += observation_log_likelihood(values, eta, compiled, candidate)
                        accepted = _accept(new_target, old_target, rng)
                        if accepted:
                            params[key] = candidate[key]
                        step_accept[key] = accepted
                        counts[key][0] += int(accepted)
                        counts[key][1] += 1
                if iteration < mcmc.warmup and mcmc.adapt:
                    gain = (iteration + 10.0) ** -0.6
                    for key, accepted in step_accept.items():
                        widths[key] = float(np.clip(widths[key] * np.exp(gain * (float(accepted) - 0.44)), 1e-5, 2.0))
                retained = iteration >= mcmc.warmup and (iteration - mcmc.warmup) % mcmc.thin == 0
                if retained:
                    ll = observation_log_likelihood(values, compiled.eta(path, params=params), compiled, params)
                    lp = ll + _log_parameter_prior(params, priors, compiled) + _log_state_density(path, params, compiled, inverse)
                    if not np.isfinite(lp):
                        raise FloatingPointError("A retained shared posterior draw has nonfinite density.")
                    state_draws[chain, saved] = path
                    log_posterior[chain, saved], log_likelihood[chain, saved] = lp, ll
                    for key in parameter_draws:
                        parameter_draws[key][chain, saved] = params[key]
                    if correlation_draws is not None:
                        correlation_draws[chain, saved] = compiled.model.copula.correlation_matrix(params, compiled.channel_names)
                    for key, value in iteration_metrics.items():
                        metrics[key][chain, saved] = value
                    saved += 1
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, OverflowError) as error:
                raise RuntimeError(
                    f"Shared {plan.engine} sampler failed in chain {chain + 1}, iteration {iteration + 1}; no posterior result was returned. {error}"
                ) from error
            every = mcmc.progress_every or max(mcmc.iterations // 20, 1)
            if mcmc.progress and ((iteration + 1) % every == 0 or iteration + 1 == mcmc.iterations):
                print(f"shared {plan.engine}: chain {chain + 1}/{mcmc.chains}, iteration {iteration + 1}/{mcmc.iterations}, saved {saved}/{mcmc.draws}, elapsed {perf_counter() - started:.1f}s", flush=True)
        for key, (accepted, attempted) in counts.items():
            acceptance.setdefault(key, np.full(mcmc.chains, np.nan))[chain] = accepted / attempted if attempted else np.nan
        for name, prior in priors.process.items():
            if isinstance(prior, FixedSD):
                continue
            cent, ncp = counts[f"centered.{name}"], counts[f"noncentered.{name}"]
            acceptance.setdefault(f"sd.{name}", np.full(mcmc.chains, np.nan))[chain] = (cent[0] + ncp[0]) / (cent[1] + ncp[1])
        final_widths.append(widths)
    update_methods = {}
    for name, prior in priors.process.items():
        update_methods[f"sd.{name}"] = "fixed" if isinstance(prior, FixedSD) else (
            "centered IG + noncentered log-SD MH; pooled acceptance" if isinstance(prior, InverseGammaVariance)
            else "centered + noncentered log-SD MH; pooled acceptance")
    for channel in compiled.model.channels:
        prior = priors.observation_sd[channel.name]
        update_methods[f"sigma.{channel.name}"] = ("fixed" if isinstance(prior, FixedSD) else
            "conjugate inverse-gamma variance" if channel.family == "gaussian" and isinstance(prior, InverseGammaVariance) and not has_copula
            else "log-SD MH")
        if channel.family == "gev":
            update_methods[f"xi.{channel.name}"] = "prior-bounded random-walk MH"
    if has_copula:
        update_methods.update({key: "full-likelihood MH in unconstrained partial correlation; LKJ prior plus Jacobian"
            for key in compiled.model.copula.parameter_names(compiled.channel_names)})
    return FitResult(
        model=compiled.model, compiled=compiled, priors=priors, y=values,
        state_draws=state_draws, parameter_draws=parameter_draws,
        log_posterior=log_posterior, plan=plan,
        auxiliary_draws={"log_likelihood": log_likelihood, **({"copula_correlation": correlation_draws} if has_copula else {})},
        sampler_diagnostics={
            "acceptance": acceptance, "draw_metrics": metrics,
            "mcmc": asdict(mcmc), "laplace": asdict(laplace), "shared_sampler": asdict(sampler),
            "elliptical_slice_blocks": [block.name for block in slice_blocks],
            "chain_seeds": [int(seed.generate_state(1)[0]) for seed in seeds],
            "final_proposal_widths": final_widths,
            "update_methods": update_methods,
            "acceptance_includes_warmup": True,
        },
        exog=compiled.exog,
        dates=None if dates is None else np.asarray(dates),
        series_name=compiled.model.name, transform_sign=compiled.model.transform_signs,
        initial_values={"chains": initial_values},
        metadata={
            "bucex_version": __version__, "inference_contract": "joint_copula_state_model" if has_copula else "joint_shared_state_model",
            "joint_model": True, "joint_likelihood": True, "shared_temporal_state": compiled.model.has_shared_states,
            "conditional_channel_independence": not has_copula, "residual_copula": has_copula,
            "copula_family": "gaussian" if has_copula else None,
            "copula_orientation": "original response" if has_copula else None,
            "laplace_proposal": "copula_joint_laplace" if has_copula else "marginal_laplace" if plan.engine == "laplace_mh" else None,
            "observation_curvature": "full" if has_copula else "diagonal",
            "ordering_enforced": False,
            "fixed_factor_loadings": compiled.model.has_shared_states, "structural_ssvs": False,
            "noncentered_scale_interweaving": True,
            "state_elliptical_slice_refresh": bool(slice_blocks),
            "state_kernel": "joint Laplace-MH + Gaussian-prior block elliptical slice" if slice_blocks else plan.engine,
            "elliptical_slice_schedule": "fixed sweeps after every state update" if slice_blocks else None,
            "laplace_mh_exact_invariant": plan.engine == "laplace_mh",
            "log_posterior_available": True, "log_likelihood_available": True,
            "log_posterior_measure": "SD parameters, centered states on their affine Gaussian support, and unconstrained copula partial coordinates; includes normalizing constants and copula Jacobian",
            "numerical_failure_policy": "abort_without_returning_fit",
            "external_warm_start": initial_parameters is not None,
        },
    )


__all__ = ["sample_shared_posterior"]
