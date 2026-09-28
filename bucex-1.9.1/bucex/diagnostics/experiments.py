"""Reusable reviewer checks on priors, recovery and predictive uncertainty.

These helpers summarize draws; they do not run MCMC, certify convergence, or
silently condition endpoint summaries on a negative shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genextreme, truncnorm

from ..priors.structural import (
    BayesianLassoPrior, ComponentwiseBayesianLassoPrior, NormalPrior, UniformPrior,
)
from .contrasts import summarize_draws

_COMPONENTS = {"level": "level", "trend": "slope", "season": "seasonal"}


def draw_structural_prior(prior, size=2000, *, seed=None, xi_bounds=None):
    """Draw the *unconditional* FS prior, including exact SSVS atoms at zero.

    Lasso local scales and their common/componentwise Gamma hyperprior are
    integrated by simulation. In the original Gaussian hierarchy sigma^2 is
    also drawn, rather than substituted by a posterior estimate.
    """
    if int(size) != size or size < 1:
        raise ValueError("size must be a positive integer.")
    size = int(size)
    rng = np.random.default_rng(seed)
    if prior.sigma2 is not None:
        sigma2 = prior.sigma2.b / rng.gamma(prior.sigma2.a, size=size)
    elif getattr(prior, "log_sigma", None) is not None:
        sigma2 = np.exp(2 * rng.normal(prior.log_sigma.mean, prior.log_sigma.sd, size))
    else:
        raise ValueError("A declared sigma2 or log_sigma prior is required.")
    output = {"sigma": np.sqrt(sigma2),
              "initial.level": rng.normal(prior.alpha0.mean, prior.alpha0.sd, size),
              "initial.slope": rng.normal(prior.beta0.mean, prior.beta0.sd, size)}
    if prior.gamma0_season is not None:
        output["initial.seasonal"] = rng.normal(prior.gamma0_season.mean_array(),
            prior.gamma0_season.sd_array(), size=(size, len(prior.gamma0_season.mean)))
    if hasattr(prior, "xi"):
        lower,upper = -prior.xi_max_abs,prior.xi_max_abs
        if xi_bounds is not None:
            lower,upper = max(lower,xi_bounds[0]),min(upper,xi_bounds[1])
        if isinstance(prior.xi, UniformPrior):
            lower,upper = max(lower,prior.xi.lower),min(upper,prior.xi.upper)
            if lower >= upper:
                raise ValueError("Empty shape-prior support.")
            output["xi"] = rng.uniform(lower,upper,size)
        elif isinstance(prior.xi, NormalPrior):
            # Match FSGEV._xi_prior: Normal is truncated at +/-xi_max_abs.
            if lower >= upper:
                raise ValueError("Empty shape-prior support.")
            lo = (lower - prior.xi.mean) / prior.xi.sd
            hi = (upper - prior.xi.mean) / prior.xi.sd
            output["xi"] = truncnorm.rvs(lo, hi, loc=prior.xi.mean,
                scale=prior.xi.sd, size=size, random_state=rng)
        else:
            raise ValueError("Shape prior draws require UniformPrior or NormalPrior.")
    lasso = prior.lasso
    common_lambda = None
    if lasso is not None and not lasso.componentwise:
        common_lambda = (np.full(size,lasso.fixed_lambda2) if lasso.fixed_lambda2 is not None
                         else rng.gamma(lasso.a_lambda, 1.0 / lasso.b_lambda, size))
    tg = prior.triple_gamma
    if tg is not None:
        a = (.5*rng.beta(*tg.spike_shape_prior,size) if tg.learn_shapes else np.full(size,tg.spike_shape))
        c = (.5*rng.beta(*tg.tail_shape_prior,size) if tg.learn_shapes else np.full(size,tg.tail_shape))
        global_scale = (rng.gamma(c)/rng.gamma(a) if tg.learn_global else np.full(size,tg.global_scale))
        slab = tg.slab_df*tg.slab_scale**2/(2*rng.gamma(tg.slab_df/2,size=size)) if tg.regularized else None
    if prior.ssvs is not None and prior.ssvs.trend_model_probabilities is not None:
        raise ValueError("Joint trend-model prior draws are not implemented by this helper.")
    for key, canonical in _COMPONENTS.items():
        if prior.ssvs is not None:
            ssvs = prior.ssvs
            if key == "level":
                active = rng.binomial(1, ssvs.level_dynamic_probability, size).astype(bool)
            else:
                probs = getattr(ssvs, f"{key}_probabilities")
                state = rng.choice(3, p=probs, size=size)
                active = state == 2
                initial = "initial.slope" if key == "trend" else "initial.seasonal"
                if initial in output:
                    output[initial][state == 0] = 0.0
            signed = rng.normal(0, ssvs.innovation_slab_sd[key], size) * active
        elif lasso is not None:
            lam = (rng.gamma(lasso.a_lambda[key], 1.0 / lasso.b_lambda[key], size)
                   if lasso.componentwise else common_lambda)
            local = rng.exponential(2.0 / lam)
            variance = sigma2 if lasso.variance_mode == "observation" else lasso.fixed_variance
            signed = rng.normal(size=size) * np.sqrt(variance * local) * lasso.coefficient_scale_for(key)
        elif prior.pc is not None:
            rate = -np.log(prior.pc.alpha_for(key)) / prior.pc.upper[key]
            signed = rng.laplace(scale=1.0 / rate, size=size)
        elif tg is not None:
            variance = global_scale*rng.gamma(a)/rng.gamma(c)
            if tg.regularized:
                variance = 1/(1/slab+1/variance)
            signed = rng.normal(size=size)*np.sqrt(variance)*tg.coefficient_scale_for(key)
        elif getattr(prior, f"s_{key}") is not None:
            declared = getattr(prior, f"s_{key}")
            signed = rng.normal(declared.mean, declared.sd, size)
        else:
            raise ValueError("Prior draws support SSVS, lasso, PC, triple-gamma and normal profiles.")
        output[f"sd.{canonical}"] = np.abs(signed)
    return output


def compare_innovation_priors(fit, prior=None, *, channel=None, size=2000, seed=None, level=0.90):
    """SD/variance summaries against the unconditional declared prior.

    For joint fits select a channel. Shared hyperparameters are integrated in
    the prior draws, never fixed at their posterior estimates or anchors.
    """
    from ..priors import MarginalPriors
    prior = fit.priors if prior is None else prior
    if fit.is_multiseries_model:
        if not isinstance(prior, MarginalPriors) or channel not in fit.channel_names:
            raise ValueError("Select a channel and its complete MarginalPriors hierarchy.")
        from .shrinkage import draw_marginal_prior
        samples = draw_marginal_prior(prior, size, seed=seed)["channels"][channel]
        prefix = f"channel.{channel}."
        posterior_sd = {key[len(prefix):]: value for key, value in fit.process_sd_draws().items()
                        if key.startswith(prefix)}
    else:
        if channel is not None:
            raise ValueError("channel= requires a multiseries fit.")
        samples = draw_structural_prior(prior, size, seed=seed)
        posterior_sd = fit.process_sd_draws()
    rows = []
    for component, posterior in posterior_sd.items():
        key = f"sd.{component}"
        if key not in samples:
            raise ValueError(f"No matching prior draw for {key!r}; use a univariate FS fit.")
        for distribution, values in (("prior", samples[key]), ("posterior", posterior)):
            for scale, transformed in (("SD", values), ("variance", np.asarray(values)**2)):
                lower, median, upper = np.quantile(transformed, [(1-level)/2, .5, (1+level)/2])
                rows.append(dict(component=component, distribution=distribution, scale=scale,
                    lower=lower, median=median, upper=upper, probability_zero=np.mean(transformed == 0),
                    credible_interval=level, draws=np.size(transformed)))
    return pd.DataFrame(rows)


def scientific_summary(fit, threshold, *, event_index=-1, level=0.90):
    """Univariate warming/risk contrasts and honest finite-endpoint diagnostics."""
    if fit.is_multiseries_model:
        raise ValueError("Use channel-specific contrasts for a multiseries fit.")
    shape = (fit.n_chains, fit.draws_per_chain)
    eta = fit.eta_draws(original_scale=True)
    level_values = fit.component_draws("level")
    risk = fit.exceedance_probability_draws(threshold, return_labels=False)
    targets = {"level_change": level_values[:, -1]-level_values[:, 0],
               "risk_start": risk[:, 0], "risk_end": risk[:, -1],
               "risk_change": risk[:, -1]-risk[:, 0], "risk_event": risk[:, event_index],
               "location_event": eta[:, event_index]}
    if fit.family == "gev":
        xi = fit.parameter("xi")
        targets.update(shape=xi, finite_endpoint=(xi < 0).astype(float),
                       return_level_100_blocks=fit.return_level_draws(100)[:, event_index])
        # Infinite endpoints remain in the supplied quantity and are explicitly flagged.
        endpoint = fit.endpoint_draws()[:, event_index]
        targets["endpoint_event"] = endpoint
        targets["endpoint_gap_event"] = fit.transform_sign * (endpoint - threshold)
    return summarize_draws({key: np.asarray(value).reshape(shape) for key, value in targets.items()},
                           credible_interval=level)


def recovery_metrics(fit, simulation, threshold, *, levels=(0.90, 0.95, 0.99)):
    """Truth-based pointwise location/risk recovery for one simulation replicate.

    Replicate-level rows must be aggregated over independently generated data
    sets for a coverage study; time points are not independent replicates.
    """
    eta = fit.eta_draws(original_scale=True)
    truth = float(fit.transform_sign) * np.asarray(simulation.eta)
    risk = fit.exceedance_probability_draws(threshold, return_labels=False)
    true_risk = 1-simulation.model.observation.cdf(float(fit.transform_sign)*threshold, simulation.eta,
        sigma=simulation.sigma, xi=simulation.params.get("xi"))
    rows = []
    for target, draws, known in (("location", eta, truth), ("risk", risk, true_risk)):
        for level in levels:
            lower, upper = np.quantile(draws, [(1-level)/2, (1+level)/2], axis=0)
            rows.append(dict(target=target, nominal=level,
                rmse=float(np.sqrt(np.mean((np.mean(draws, axis=0)-known)**2))),
                bias=float(np.mean(np.mean(draws, axis=0)-known)),
                pointwise_coverage=float(np.mean((lower <= known) & (known <= upper))),
                mean_interval_width=float(np.mean(upper-lower))))
    return pd.DataFrame(rows)


def forecast_uncertainty(forecast, *, channel=None, levels=(0.90, 0.95, 0.99)):
    """Forecast quantiles and total-variance decomposition at every horizon.

    GEV location is not its conditional mean. The decomposition therefore uses
    Var(E[Y|state,parameters]) + E[Var(Y|state,parameters)], separately from
    Var(location). For any posterior mass at xi>=1/2 the GEV predictive second
    moment is infinite; finite Monte Carlo sample variance must not conceal it.
    Conversely, finite conditional moments for every retained draw do not
    establish integrability over the full posterior: even support approaching
    xi=1/2 from below may give a divergent posterior mean conditional variance.
    Reported finite decompositions are Monte Carlo estimates over retained paths.
    """
    joint = forecast.is_multiseries_forecast
    if joint:
        if channel not in forecast.channel_names:
            raise ValueError("Choose a forecast channel.")
        index = forecast.channel_names.index(channel)
        observations, eta = forecast.observations[..., index], forecast.eta[..., index]
        item = next(c for c in forecast.channels if c.name == channel)
        family, sign = item.family, item.transform_sign
        sigma = forecast.parameters.get(f"sigma_path.{channel}", forecast.parameters[f"sigma.{channel}"][:, None])
        xi = forecast.parameters.get(f"xi.{channel}")
    else:
        observations, eta = forecast.observations, forecast.eta
        family, sign = forecast.family, float(forecast.transform_sign)
        sigma = forecast.parameters.get("sigma_path", forecast.parameters["sigma"][:, None])
        xi = forecast.parameters.get("xi")
    latent_level = forecast.component_draws("level", channel=channel)
    if family == "gev":
        xi = np.asarray(xi)[:, None]
        standard_mean, standard_variance = genextreme.stats(-xi, moments="mv")
        conditional_mean = eta + sign * sigma * standard_mean
        conditional_variance = sigma**2 * standard_variance
        finite_second = bool(np.all(xi < .5))
    else:
        conditional_mean, conditional_variance = eta, sigma**2
        finite_second = True
    conditional_variance = np.broadcast_to(conditional_variance, eta.shape)
    between = np.var(conditional_mean, axis=0) if np.all(np.isfinite(conditional_mean)) else np.full(forecast.horizon, np.nan)
    within = np.mean(conditional_variance, axis=0) if finite_second else np.full(forecast.horizon, np.inf)
    rows = []
    for target, values in (("level", latent_level), ("location", eta), ("observation", observations)):
        for level in levels:
            low, median, high = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0)
            upper_quantile = np.quantile(values, level, axis=0)
            for h in range(forecast.horizon):
                rows.append(dict(channel=channel or "series", horizon=h+1, time=forecast.dates[h],
                    target=target, nominal=level, lower=low[h], median=median[h], upper=high[h],
                    interval_width=high[h]-low[h], upper_quantile=upper_quantile[h],
                    latent_location_variance=np.var(eta[:, h]),
                    estimated_conditional_mean_variance=between[h], estimated_expected_observation_variance=within[h],
                    estimated_total_observation_variance=between[h]+within[h],
                    all_retained_conditional_second_moments_finite=finite_second))
    return pd.DataFrame(rows)


__all__ = ["draw_structural_prior", "compare_innovation_priors",
           "scientific_summary", "recovery_metrics", "forecast_uncertainty"]


def prior_predictive_targets(model, prior, n_time, threshold, *, tail="upper",
                             size=200, seed=None, event_index=-1, level=.90):
    """Simulate scientific quantities from a declared univariate structural prior.

    This includes uncertainty in initial level/slope/season, innovation SDs,
    sigma and xi, and the declared GEV log-scale dynamics. It therefore
    complements an SD-only prior comparison. The model must expose a full
    dynamic local-linear trend and, if present, a dynamic dummy seasonal
    component: structural selection is controlled by the supplied FS prior.
    Infinite endpoints are retained; finite endpoint probability is separate.
    """
    from ..simulate import simulate
    from ..components import LocalLinearTrend, DummySeasonal
    from ..models.compiler import compile_model
    from ..models.structural import Model
    if not isinstance(model, Model) or any(not isinstance(c, (LocalLinearTrend, DummySeasonal)) for c in model.components):
        raise ValueError("Prior predictive targets require a univariate FS local-linear-trend model, optionally with dummy seasonality.")
    trends = [c for c in model.components if isinstance(c, LocalLinearTrend)]
    seasons = [c for c in model.components if isinstance(c, DummySeasonal)]
    if len(trends) != 1 or len(seasons) > 1:
        raise ValueError("Declare one trend and at most one dummy-seasonal component.")
    if prior.ssvs is not None and (trends[0].level_mode == "static" or trends[0].trend_mode == "static"
                                  or any(c.mode == "static" for c in seasons)):
        raise ValueError("Use full dynamic components with SSVS; explicit static reductions require continuous priors.")
    seasons = [c for c in seasons if c.mode != "off"]
    if int(n_time) != n_time or n_time < 1:
        raise ValueError("n_time must be a positive integer.")
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1).")
    if tail not in {"upper", "max", "maxima", "lower", "min", "minima"}:
        raise ValueError("tail must identify upper or lower extremes.")
    sign = -1.0 if tail in {"lower", "min", "minima"} else 1.0
    if model.family == "gaussian" and sign < 0:
        raise ValueError("Gaussian means do not use a lower-tail transformation.")
    n_time = int(n_time)
    prior_seed, simulation_seed = np.random.SeedSequence(seed).spawn(2)
    samples = draw_structural_prior(prior, size, seed=prior_seed,
        xi_bounds=model.observation.xi_bounds if model.family == "gev" else None)
    size = int(size)
    rng = np.random.default_rng(simulation_seed)
    compiled = compile_model(model, np.zeros(n_time))
    trend_slice = compiled.component_slices["trend"]
    season_slice = compiled.component_slices.get("seasonal")
    if seasons and (prior.gamma0_season is None or len(prior.gamma0_season.mean) != seasons[0].period - 1):
        raise ValueError("The FS seasonal prior must have period-1 initial-state coordinates.")
    scale_samples = None
    specification = model.observation.scale
    if getattr(specification, "mode", None) == "structural":
        scale_period = next((c.period for c in specification.components
                             if isinstance(c, DummySeasonal) and c.mode != "off"), None)
        scale_samples = draw_structural_prior(specification.priors.resolve(scale_period), size,
                                              seed=int(rng.integers(0, 2**32-1)))
    targets = []
    for i in range(size):
        params = {"sigma": samples["sigma"][i],
            f"sd.{trends[0].level_name}": samples["sd.level"][i],
            f"sd.{trends[0].slope_name}": samples["sd.slope"][i]}
        initial = np.zeros(compiled.state_dim)
        initial[trend_slice.start] = samples["initial.level"][i]
        if trends[0].trend_mode != "off":
            initial[trend_slice.start+1] = samples["initial.slope"][i]
        if seasons:
            params[f"sd.{seasons[0].name}"] = samples["sd.seasonal"][i]
            # FS gamma0 is the state at observation 1. Generic simulation
            # starts before observation 1, so invert one seasonal rotation.
            rotation = compiled.transition[season_slice, season_slice]
            initial[season_slice] = np.linalg.solve(rotation, samples["initial.seasonal"][i])
        if model.family == "gev":
            params["xi"] = samples["xi"][i]
            phi_mode = model.observation.phi
            if phi_mode == "ssvs":
                phi_mode = rng.choice(("stationary", "linear", "rw"), p=prior.phi.model_probabilities)
            log_sigma = np.log(params["sigma"])
            if phi_mode == "linear":
                basis = np.linspace(-.5, .5, n_time) if n_time > 1 else np.zeros(1)
                log_sigma = log_sigma + rng.normal(prior.phi.linear.mean, prior.phi.linear.sd) * basis
            elif phi_mode == "rw":
                variance = prior.phi.rw_variance.b / rng.gamma(prior.phi.rw_variance.a)
                log_sigma = log_sigma + np.cumsum(rng.normal(0, np.sqrt(variance), n_time))
            with np.errstate(over="ignore", under="ignore"):
                params["sigma"] = np.exp(log_sigma)
            if np.any(~np.isfinite(params["sigma"])) or np.any(params["sigma"] <= 0):
                raise FloatingPointError("Prior log-scale paths exceed numerical precision; reconsider the declared prior.")
        if model.observation.scale is not None:
            specification = model.observation.scale
            if scale_samples is not None:
                for component in ("level", "slope", "seasonal"):
                    params["scale.sd."+component] = scale_samples["sd."+component][i]
                params["scale.initial.slope"] = scale_samples["initial.slope"][i]
                params["scale.initial.seasonal"] = scale_samples["initial.seasonal"][i]
            else:
                params["scale.seasonal"] = specification.contrast() @ rng.normal(0, specification.prior_sd, specification.period-1)
            if getattr(specification,"mode","constant") == "linear":
                params["scale_slope"] = rng.normal(0,specification.slope_sd)
            elif getattr(specification,"mode","constant") == "rw":
                params["scale_signed_sd"] = rng.normal(0,specification.innovation_sd)
        simulation = simulate(model, n_time, params, initial_state=initial,
                              seed=int(rng.integers(0, 2**32-1)))
        eta = simulation.eta
        risk = 1-model.observation.cdf(sign*threshold, eta, sigma=simulation.sigma, xi=params.get("xi"))
        item = dict(level_change=sign*(simulation.states[-1, trend_slice.start]-simulation.states[1, trend_slice.start]),
            risk_start=risk[0], risk_end=risk[-1], risk_change=risk[-1]-risk[0],
            risk_event=risk[event_index], location_event=sign*eta[event_index])
        if model.family == "gev":
            xi = params["xi"]
            sigma_event = simulation.sigma[event_index]
            endpoint = sign*(eta[event_index]-sigma_event/xi) if xi < 0 else sign*np.inf
            item.update(shape=xi, finite_endpoint=float(xi < 0), endpoint_event=endpoint,
                endpoint_gap_event=sign*(endpoint-threshold),
                return_level_100_blocks=sign*model.observation.ppf(.99, eta[event_index], sigma=sigma_event, xi=xi))
        targets.append(item)
    table = pd.DataFrame(targets)
    rows = []
    for quantity in table:
        values = table[quantity].to_numpy()
        # Inverse-CDF empirical quantiles preserve mass at infinity without inf-inf interpolation.
        lower, median, upper = np.quantile(values, [(1-level)/2, .5, (1+level)/2], method="inverted_cdf")
        rows.append(dict(quantity=quantity, distribution="prior", lower=lower, median=median,
            upper=upper, finite_fraction=np.mean(np.isfinite(values)), draws=size))
    return pd.DataFrame(rows).set_index("quantity")


def annual_aggregation_check(forecast, threshold, *, channel=None):
    """Check draw-wise annual aggregation against conditional monthly CDFs.

    Uses complete calendar years only. The annual maximum is exactly the
    maximum of monthly maxima when the months partition that year. A product
    of CDFs additionally assumes residual independence across months conditional
    on states/parameters. The posterior average is taken *after* the product.
    MC standard errors condition on the retained posterior paths; they measure
    observation simulation error only, not MCMC uncertainty.
    """
    from ..core.calendar import annual_groups
    groups, labels = annual_groups(forecast.horizon, forecast.dates,
                                   period=forecast.period, include_partial=False)
    cdf = forecast.conditional_cdf(np.full(forecast.horizon, threshold), channel=channel)
    if forecast.is_multiseries_forecast:
        j = forecast.channel_names.index(channel)
        observations = forecast.observations[..., j]
        tail = forecast.tail[channel] if isinstance(forecast.tail, dict) else forecast.tail[j]
    else:
        observations, tail = forecast.observations, forecast.tail
    lower = tail == "lower"
    rows = []
    for group, label in zip(groups, labels):
        probability = (1-np.prod(1-cdf[:, group], axis=1) if lower else np.prod(cdf[:, group], axis=1))
        annual = np.min(observations[:, group], axis=1) if lower else np.max(observations[:, group], axis=1)
        expected, empirical = np.mean(probability), np.mean(annual <= threshold)
        se = np.sqrt(np.sum(probability*(1-probability)))/probability.size
        rows.append(dict(year=label, months=len(group), channel=channel or "series",
            aggregation="minimum" if lower else "maximum", threshold=threshold,
            analytic_predictive_cdf=expected, simulated_predictive_cdf=empirical,
            simulation_difference=empirical-expected, observation_mc_se=se,
            product_of_posterior_mean_cdfs=np.prod(np.mean(cdf[:, group], axis=0)) if not lower else np.nan,
            posterior_paths=probability.size))
    return pd.DataFrame(rows)


__all__ += ["prior_predictive_targets", "annual_aggregation_check"]
