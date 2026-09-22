"""Private FS trajectories with continuous shrinkage or exact SSVS with optional residual copula.

Gaussian channels use their exact copula-conditional Gaussian regression.
GEV paths use deterministic conditional Laplace independence proposals and
an exact MH correction. Structural moves, scale, shape and R updates all
target the joint likelihood. There is no two-stage/cut approximation.
"""
from dataclasses import asdict
from types import SimpleNamespace

from ..chains import independent_chains, chain_seeds, chain_position

import numpy as np

from ...__about__ import __version__
from ...core.fit import FitResult
from ...dependence.gaussian import gaussian_copula_logpdf
from ...priors import MarginalPriors, FSGaussianPriors, FSGEVPriors
from ..plan import InferencePlan
from .conditional_margin import ConditionalMargin, conditional_score_parameters
from .hierarchical import _initial_channel_state, _sign_move
from .fs_utils import (
    apply_theta_draw, baseline_mu_path, build_ncp_system, design_matrix_ncp,
    elliptical_slice_gaussian_prior, ffbs_gaussian_1d_tvR, map_ncp_to_centered,
    measurement_vector, mu_from_ncp, ncp_laplace_mh, theta_vector_from_params,
    _slice_sample_real, _ncp_laplace_pseudo_data,
)
from .model_space import sample_structural_regression, sample_structural_regression_exact
from .continuous import ShrinkageState, continuous_step
from .shrinkage import SharedShrinkageState


def marginal_plan(compiled, *, engine="auto", parameterization="auto", asis=False):
    from ..plan import normalize_parameterization, supports_fs
    resolved = normalize_parameterization(parameterization, fs_supported=supports_fs(compiled))
    if resolved != "fruehwirth_schnatter":
        raise ValueError("MarginalPriors and SeasonalScale require parameterization='fs'.")
    gaussian = bool(getattr(compiled, "all_gaussian", compiled.family == "gaussian"))
    actual = ("ffbs" if gaussian else "laplace_mh") if engine == "auto" else engine
    if actual != ("ffbs" if gaussian else "laplace_mh"):
        raise ValueError("Private FS inference requires ffbs for Gaussian models and laplace_mh for mixed/GEV models.")
    return InferencePlan(
        family=compiled.family, engine=actual, parameterization=resolved,
        asis=bool(asis), state_update="conditional Gaussian FFBS / exact Laplace-MH",
        targets_exact_posterior=True, approximation=None,
        proposal=None if gaussian else "copula_conditional_laplace",
        backend="marginal_fs", warnings=(),
    )


def _sigma(state, effects, phase):
    return float(state.params_obs["sigma"]) * np.exp(effects[phase] + state.params_obs.get("log_scale_offset", 0.))


def _observation_params(state, effects, phase):
    return {**state.params_obs, "sigma": _sigma(state, effects, phase)}


def _channel_parameters(state, effects, *, selection=True):
    name, p, s = state.name, state.params_state, state.model_state
    result = {
        f"initial.channel.{name}.level": float(p["alpha0"]),
        f"state.{name}.level": int(s.level), f"state.{name}.trend": int(s.trend),
        f"state.{name}.season": int(s.season), f"model_index.{name}": int(state.model_index),
        f"sigma.{name}": float(state.params_obs["sigma"]),
    }
    if not selection:
        result = {k:v for k,v in result.items() if not k.startswith(("state.", "model_index."))}
    if state.family == "gev":
        result[f"xi.{name}"] = float(state.params_obs["xi"])
    if state.layout.has_beta:
        result[f"initial.channel.{name}.slope"] = float(p["beta0"])
    if state.layout.season_dim:
        result[f"initial.channel.{name}.seasonal"] = np.asarray(p["gamma0_season"])
    for process, signed in (("level", "s_level"), ("slope", "s_trend"), ("seasonal", "s_season")):
        if process in state.compiled.noise_names:
            result[f"sd.channel.{name}.{process}"] = abs(float(p[signed]))
            result[f"signed_sd.channel.{name}.{process}"] = float(p[signed])
    if state.model.observation.scale is not None:
        result[f"scale.seasonal.{name}"] = effects.copy()
    for key in ("scale_slope", "scale_signed_sd", "scale_z", "log_scale_offset"):
        if key in state.params_obs:
            result[f"{key}.{name}"] = np.asarray(state.params_obs[key]).copy()
    if "scale_signed_sd" in state.params_obs:
        result[f"scale_rw_sd.{name}"] = abs(state.params_obs["scale_signed_sd"])
    result.update({f"{key}.{name}": np.asarray(value).copy()
                   for key,value in state.params_obs.items() if key.startswith("evolution.")})
    return result


def _structure_step(state, conditional, params, prior, laplace, rng):
    layout = state.layout
    model = SimpleNamespace(obs=conditional)
    metric = {}
    if state.family == "gaussian":
        G, Q = build_ncp_system(layout)
        adjusted = state.y - params["sigma"] * conditional.mean
        variance = params["sigma"]**2 * conditional.variance
        state.z_path = ffbs_gaussian_1d_tvR(
            adjusted - baseline_mu_path(state.y.size, state.params_state, layout),
            G, Q, measurement_vector(state.params_state, layout), variance,
            m0=np.zeros(layout.ncp_state_dim), C0=np.zeros((layout.ncp_state_dim,)*2), rng=rng,
        )
    else:
        result = ncp_laplace_mh(
            state.y, model, state.params_state, params, layout, state.z_path,
            rng=rng, mh_steps=laplace.mh_steps, max_iterations=laplace.max_iterations,
            tolerance=laplace.tolerance, curvature_floor=laplace.curvature_floor,
            maximum_variance=laplace.maximum_variance,
        )
        state.z_path = result.z_path
        metric = {"laplace_mh_acceptance": result.acceptance_rate,
                  "laplace_converged": float(result.approximation.converged),
                  "laplace_iterations": result.approximation.iterations,
                  "laplace_relative_change": result.approximation.relative_change,
                  "laplace_mh_mean_log_acceptance_ratio": float(np.mean(result.log_acceptance_ratio)),
                  "laplace_initial_support_repaired": float(result.approximation.initial_support_repaired),
                  "laplace_mh_support_rejections": result.proposal_support_failures,
                  "laplace_support_rejections": result.proposal_support_failures}
    X, names, tbar = design_matrix_ncp(state.z_path, layout, center_time=True)
    common = dict(X=X, theta_names=names, tbar=tbar, priors=prior,
                  layout=layout, rng=rng, apply_theta_draw=apply_theta_draw)
    if state.family == "gaussian":
        selection = sample_structural_regression(y=adjusted, noise_variance=variance, **common)
    else:
        # A fixed, support-safe observation anchor makes q independent of the
        # coefficients and model being updated. Re-expanding at their current
        # predictor would require a different reverse-proposal calculation.
        eta = state.y
        pseudo_y, pseudo_variance = _ncp_laplace_pseudo_data(
            state.y, eta, model, params, curvature_floor=laplace.curvature_floor,
            maximum_variance=laplace.maximum_variance, shift_limit=10*float(np.max(params["sigma"])),
        )
        selection = sample_structural_regression_exact(
            y=state.y, pseudo_y=pseudo_y, pseudo_variance=pseudo_variance,
            current_state=state.model_state, current_params_state=state.params_state,
            theta_vector_from_params=theta_vector_from_params,
            elliptical_slice=elliptical_slice_gaussian_prior,
            log_likelihood=lambda eta: float(np.sum(conditional.logpdf(state.y, eta, params))),
            **common,
        )
        metric["structural_move_acceptance"] = float(selection.move_accepted)
    state.params_state.update(selection.params_state)
    state.model_state, state.model_index = selection.state, int(selection.selected_index)
    _, error = _sign_move(state, rng)
    metric["sign_error"] = error
    return metric


def _observation_step(state, conditional, coefficients, scale, phase, prior, rng, mixing=None):
    basis = scale.contrast() if scale else np.zeros((1, 0))
    effects = basis @ coefficients
    eta = mu_from_ncp(state.z_path, state.params_state, state.layout)

    def likelihood(log_scale, effect=effects, xi=None, offset=None):
        if offset is None:
            offset = state.params_obs.get("log_scale_offset", 0.)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            sigma = np.exp(log_scale + effect[phase] + offset)
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            return -np.inf
        params = {**state.params_obs, "sigma": sigma}
        if xi is not None:
            params["xi"] = xi
        return float(np.sum(conditional.logpdf(state.y, eta, params)))

    def log_scale_target(value):
        # IG(a,b) on sigma^2, transformed to log(sigma): -2*a*l-b*exp(-2*l).
        with np.errstate(over="ignore", invalid="ignore"):
            if getattr(prior, "log_sigma", None) is not None:
                lp = -.5*((value-prior.log_sigma.mean)/prior.log_sigma.sd)**2
            else:
                lp = -2*prior.sigma2.a*value - prior.sigma2.b*np.exp(-2*value)
        if mixing is not None and prior.lasso is not None and prior.lasso.variance_mode == "observation":
            terms = [(state.params_state["s_"+k] / prior.lasso.coefficient_scale_for(k))**2 / v
                     for k,v in mixing.tau.items()]
            with np.errstate(over="ignore", invalid="ignore"):
                lp += -len(terms)*value - .5*sum(terms)*np.exp(-2*value)
        return float(lp + likelihood(value)) if np.isfinite(lp) else -np.inf

    log_scale, evaluations = _slice_sample_real(np.log(state.params_obs["sigma"]), log_scale_target, rng, width=.08)
    state.params_obs.update(sigma=float(np.exp(log_scale)), sigma2=float(np.exp(2*log_scale)))
    metric = {"scale_slice_evaluations": evaluations}
    if scale and scale.period > 1:
        coefficients, evaluations = elliptical_slice_gaussian_prior(
            coefficients, np.zeros(scale.period-1), scale.prior_sd**2*np.eye(scale.period-1),
            lambda b: likelihood(log_scale, basis @ b), rng,
        )
        effects = basis @ coefficients
        metric["seasonal_scale_slice_evaluations"] = evaluations
    from .scale_path import secular_scale_step
    metric.update(secular_scale_step(state, scale,
        lambda offset: likelihood(log_scale, effects, offset=offset), rng))
    if state.family == "gev":
        lower, upper = state.model.observation.xi_bounds
        lower, upper = max(lower, -prior.xi_max_abs), min(upper, prior.xi_max_abs)

        def xi_target(value):
            if not lower <= value <= upper:
                return -np.inf
            lp = (prior.xi.logpdf(value) if hasattr(prior.xi, "logpdf")
                  else -.5*((value-prior.xi.mean)/prior.xi.sd)**2)
            return lp + likelihood(log_scale, effects, value)

        xi, evaluations = _slice_sample_real(state.params_obs["xi"], xi_target, rng, width=.03)
        state.params_obs["xi"] = xi
        metric["xi_slice_evaluations"] = evaluations
    return coefficients, effects, metric


@independent_chains()
def sample_marginal_posterior(y, compiled, priors, plan, *, mcmc, laplace, dates=None, initial_parameters=None):
    if not isinstance(priors, MarginalPriors):
        raise TypeError("Use MarginalPriors(channels={name: fs_prior}).")
    model = compiled.model
    if plan.asis and any(p.ssvs is not None for p in priors.channels.values()):
        raise ValueError("ASIS requires continuous priors for every channel; exact SSVS has point-mass zero scales.")
    if model.shared or not model.supports_fs_parameterization:
        raise ValueError("Private FS/SSVS requires local-linear trends with optional dummy seasonality, without shared states or regressions.")
    if set(priors.channels) != set(compiled.channel_names):
        raise ValueError("MarginalPriors names must match model channels exactly.")
    values = np.asarray(y, float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Private FS/SSVS currently requires a complete aligned observation matrix; missing-data inference is not implemented in this backend.")
    for channel in model.channels:
        expected = FSGaussianPriors if channel.family == "gaussian" else FSGEVPriors
        if not isinstance(priors.channels[channel.name], expected):
            raise TypeError(f"Wrong structural prior family for {channel.name}.")
        if channel.family == "gev" and channel.observation.phi != "stationary":
            raise ValueError("Joint FS/SSVS supports stationary or seasonal observation scales, not a dynamic phi path.")
    C, D, T, K = mcmc.chains, mcmc.draws, len(values), len(model.channels)
    paths = np.empty((C, D, T+1, compiled.state_dim))
    loglik = np.empty((C, D))
    parameters, metrics = {}, {}
    scales = [channel.observation.scale for channel in model.channels]
    phases = [s.phases(T, dates) if s else np.zeros(T, int) for s in scales]
    initial = dict(initial_parameters or {})
    restart = initial.pop("__fs_marginal__", None)
    seeds = chain_seeds(mcmc.seed, C)
    def correlation_for(parameters):
        if model.copula is None:
            return np.eye(K)
        if model.copula.seasonal:
            return model.copula.correlation_path(parameters, compiled.channel_names, T, dates)
        return model.copula.correlation_matrix(parameters, compiled.channel_names)
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        # Each channel starts at a support-safe Gumbel fit. A restart restores
        # all nuisance parameters together before any transition is run.
        states = [_initial_channel_state(b.name, model.channel(b.name).family, values[:,j], b.compiled,
                  priors.channels[b.name], rng, initial) for j,b in enumerate(compiled.blocks)]
        for state in states:
            from .model_space import enumerate_structural_models, structural_model_log_prior, enforce_structural_state
            prior = priors.channels[state.name]
            from .fs_utils import _active_scale_names
            inactive = {"level", "trend", "season"} - set(_active_scale_names(state.layout))
            has_fixed = (any(getattr(c, "level_mode", None) == "static" or
                             getattr(c, "trend_mode", None) == "static" or
                             getattr(c, "mode", None) == "static" for c in state.model.components))
            if has_fixed and prior.ssvs is not None:
                raise ValueError("Explicit static components use continuous priors; for SSVS declare dynamic components and select them through SSVSPrior.")
            for component in inactive:
                state.params_state["s_"+component] = state.params_state["q_"+component] = 0.
            if prior.ssvs is not None and not np.isfinite(structural_model_log_prior(state.model_state, prior.ssvs)):
                candidates = enumerate_structural_models(state.layout, prior.ssvs)
                state.model_state = max(candidates, key=lambda m: structural_model_log_prior(m, prior.ssvs))
                state.params_state = enforce_structural_state(state.params_state, state.model_state, state.layout)
            if state.family != "gev":
                continue
            lower = max(state.model.observation.xi_bounds[0], -prior.xi_max_abs, getattr(prior.xi, "lower", -np.inf))
            upper = min(state.model.observation.xi_bounds[1], prior.xi_max_abs, getattr(prior.xi, "upper", np.inf))
            if not lower < upper:
                raise ValueError(f"Empty fitted shape support for {state.name}.")
            if not lower <= state.params_obs["xi"] <= upper:
                state.params_obs["xi"] = (lower+upper)/2
            residual = state.y-mu_from_ncp(state.z_path, state.params_state, state.layout)
            scale = max(state.params_obs["sigma"], float(np.max(-state.params_obs["xi"]*residual))/.8)
            state.params_obs.update(sigma=scale, sigma2=scale*scale)
        mixing = [ShrinkageState.initialize(priors.channels[s.name], s.layout) if priors.channels[s.name].ssvs is None else None for s in states]
        coefficients = [np.zeros(s.period-1) if s else np.zeros(0) for s in scales]
        effects = [s.contrast() @ b if s else np.zeros(1) for s,b in zip(scales, coefficients)]
        cp = model.copula.initial_parameters(compiled.channel_names) if model.copula else {}
        if restart:
            for j, state in enumerate(states):
                saved = restart["channels"][state.name]
                state.params_state = dict(saved["state"])
                state.params_obs = dict(saved["observation"])
                state.z_path = np.asarray(saved["z_path"]).copy()
                from .model_space import initial_structural_state
                state.model_state = initial_structural_state(state.params_state, state.layout)
                if scales[j]:
                    effects[j] = np.asarray(saved["effects"])
                    coefficients[j] = scales[j].contrast().T @ effects[j]
            cp.update(restart["copula"])
            for state, mixture in zip(states,mixing):
                if mixture is not None: mixture.restore(restart.get("mixing",{}),state.name)
        shared = None
        if priors.shrinkage is not None:
            start_scales = {**initial, **((restart or {}).get("mixing", {}))}
            shared = SharedShrinkageState.initialize(priors.shrinkage, states, rng, start_scales)
        saved_draw = 0
        for iteration in range(mcmc.iterations):
            correlation = correlation_for(cp)
            scores = np.column_stack([
                model.transform_signs[j] * ConditionalMargin(s.model.observation, 0, 1).scores(
                    s.y, mu_from_ncp(s.z_path, s.params_state, s.layout), _observation_params(s, effects[j], phases[j]))
                for j,s in enumerate(states)
            ])
            row_metrics = {}
            for j, state in enumerate(states):
                mean, variance = conditional_score_parameters(scores, correlation, j, model.transform_signs[j])
                conditional = ConditionalMargin(state.model.observation, mean, variance)
                prior = priors.channels[state.name]
                if shared is not None:
                    prior = priors.shrinkage.conditional_prior(prior, shared.medians)
                if prior.ssvs is not None:
                    metric = _structure_step(state, conditional, _observation_params(state, effects[j], phases[j]), prior, laplace, rng)
                else:
                    metric = continuous_step(state, conditional, _observation_params(state, effects[j], phases[j]),
                                             prior, mixing[j], laplace, rng, asis=plan.asis)
                coefficients[j], effects[j], obs_metric = _observation_step(
                    state, conditional, coefficients[j], scales[j], phases[j], prior, rng, mixing[j])
                row_metrics.update({f"{key}.{state.name}": value for key,value in {**metric, **obs_metric}.items()})
                scores[:,j] = model.transform_signs[j] * conditional.scores(
                    state.y, mu_from_ncp(state.z_path, state.params_state, state.layout), _observation_params(state, effects[j], phases[j]))
            if shared is not None:
                row_metrics.update(shared.update(rng))
            if model.copula and model.copula.estimated:
                for name in cp:
                    def target(value):
                        proposed = {**cp, name: value}
                        try:
                            return model.copula.log_prior(proposed, compiled.channel_names) + float(np.sum(
                                model.copula.logpdf(scores, proposed, compiled.channel_names, dates=dates)))
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                            return -np.inf
                    cp[name], evaluations = _slice_sample_real(cp[name], target, rng, width=.08)
                    row_metrics[f"slice_evaluations.{name}"] = evaluations
                correlation = correlation_for(cp)
            retain = iteration >= mcmc.warmup and (iteration-mcmc.warmup) % mcmc.thin == 0
            if retain:
                current = dict(cp)
                if shared is not None:
                    current.update(shared.parameter_values())
                total = float(np.sum(gaussian_copula_logpdf(scores, correlation)))
                for j, (state, block) in enumerate(zip(states, compiled.blocks)):
                    paths[chain,saved_draw,:,block.state_slice] = map_ncp_to_centered(state.z_path, state.params_state, state.layout)
                    current.update(_channel_parameters(state, effects[j], selection=priors.channels[state.name].ssvs is not None))
                    if scales[j] is not None:
                        current[f"sigma_path.{state.name}"] = _sigma(state, effects[j], phases[j])
                    if mixing[j] is not None: current.update(mixing[j].parameter_values(state.name))
                    total += float(np.sum(state.model.observation.logpdf(state.y,
                        mu_from_ncp(state.z_path, state.params_state, state.layout), _observation_params(state, effects[j], phases[j]))))
                if not np.isfinite(total):
                    raise FloatingPointError("Joint FS sampler left finite likelihood support.")
                loglik[chain,saved_draw] = total
                for key,value in current.items():
                    if key not in parameters:
                        parameters[key] = np.empty((C,D,*np.shape(value)))
                    parameters[key][chain,saved_draw] = value
                for key,value in row_metrics.items():
                    if key not in metrics:
                        metrics[key] = np.empty((C,D))
                    metrics[key][chain,saved_draw] = value
                saved_draw += 1
            if mcmc.progress and ((iteration+1) % max(1, mcmc.progress_every or 50) == 0 or iteration+1 == mcmc.iterations):
                display_chain, display_total = chain_position(chain+1, C)
                print(f"Private FS {plan.engine}: chain {display_chain}/{display_total}, iteration {iteration+1}/{mcmc.iterations}, retained {saved_draw}/{D}", flush=True)
    # Preserve per-channel diagnostics and expose pooled computational summaries.
    for key in {name.rsplit(".", 1)[0] for name in metrics
                if name.startswith(("laplace_", "coefficient_"))}:
        metrics[key] = np.mean([value for name,value in metrics.items() if name.startswith(key+".")], axis=0)
    return FitResult(
        model=model, compiled=compiled, priors=priors, y=values, state_draws=paths,
        parameter_draws=parameters, log_posterior=np.full((C,D), np.nan),
        auxiliary_draws={"log_likelihood": loglik}, plan=plan,
        sampler_diagnostics={"mcmc": asdict(mcmc), "laplace": asdict(laplace),
                             "draw_metrics": metrics,
                             "acceptance": {k: v.mean(axis=1) for k,v in metrics.items() if "acceptance" in k},
                             "chain_seeds": [int(s.generate_state(1)[0]) for s in seeds]},
        dates=None if dates is None else np.asarray(dates), series_name=model.name,
        transform_sign=model.transform_signs,
        metadata={"bucex_version": __version__, "inference_contract": "private_fs_joint_likelihood",
                  "marginal_fs": True, "structural_ssvs": any(p.ssvs is not None for p in priors.channels.values()),
                  "model_selection_exact": any(p.ssvs is not None for p in priors.channels.values()),
                  "innovation_prior_families": {k:p.profile for k,p in priors.channels.items()},
                  "parameter_updates": {c.name: {"mu": plan.state_update,
                      "sigma": "Gaussian evolution elliptical slice" if getattr(c.observation.scale,"mode",None)=="structural" else "exact likelihood scale updates",
                      **({"xi": "constant shape slice"} if c.family=="gev" else {})} for c in model.channels},
                  "asis": plan.asis,
                  "continuous_coefficient_update": "exact Gaussian draw / conditional-mode reference elliptical slice",
                  "coefficient_reference_initialization": "deterministic, independent of current coefficients",
                  "shared_temporal_state": False, "hierarchical_model_selection": False,
                  "hierarchical_innovations": priors.shrinkage is not None,
                  "shared_shrinkage": None if priors.shrinkage is None else asdict(priors.shrinkage),
                  "shared_shrinkage_members": {} if shared is None else {c: [s.name for s in members] for c, members in shared.members.items()},
                  "innovation_marginal_prior": "normal scale mixture" if priors.shrinkage else "declared channel priors",
                  "joint_model": True, "joint_likelihood": True,
                  "conditional_channel_independence": model.copula is None,
                  "copula_feedback": model.copula is not None,
                  "signed_innovation_scales": True, "sign_switching": True,
                  "log_posterior_available": False, "log_likelihood_available": True},
    )
