"""One exact-target MCMC chain; channels factorize given shared scales.

FFBS/Laplace-MH, Gaussian/elliptical coefficient updates, observation slice
updates and direct GIG shared-scale updates are kept in separate modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from ..distributions import GEV, contrasts
from ..priors import Normal, Fixed
from . import _state as fs
from ._model import layout_for
from ._laplace import ncp_laplace_mh, _ncp_laplace_pseudo_data, _deterministic_feasible_ncp_path, _project_ncp_path_to_support
from ._coefficients import gaussian_reference, reference_slice
from ._reference import coefficient_reference
from ._slice import _slice_sample_real
from ._shared import _half_normal_gig_log_multiplier

FIELDS = {'level': 's_level', 'slope': 's_trend', 'seasonal': 's_season'}


@dataclass
class State:
    name: str
    model: object
    y: np.ndarray
    layout: object
    obs: object
    sign: int
    params: dict
    sigma2: float
    xi: float
    scale_contrasts: np.ndarray
    z: np.ndarray

    def observation_params(self, *, sigma2=None, xi=None, u=None):
        variance = self.sigma2 if sigma2 is None else sigma2
        effects = self.scale_contrasts if u is None else u
        if effects.size:
            log_effect = contrasts(self.model.period) @ effects
            sigma = np.sqrt(variance) * np.exp(log_effect[np.arange(len(self.y)) % self.model.period])
        else:
            sigma = np.full(len(self.y), np.sqrt(variance))
        return {'sigma': sigma, 'xi': self.xi if xi is None else xi}


def _prior_value(p, sign=1):
    return sign * (p.value if isinstance(p, Fixed) else p.mean)


def initialize(channel, y, rng, shared):
    model, layout = channel.model, layout_for(channel.model)
    sign = model.observation.sign if isinstance(model.observation, GEV) else 1
    obs = GEV() if isinstance(model.observation, GEV) else model.observation
    y = np.asarray(y)*sign
    p = model.priors
    t = np.arange(1, len(y)+1)
    X = np.column_stack([np.ones(len(y))] + ([t] if layout.has_beta else []) +
                         ([fs.static_seasonal_design(len(y), layout.season_dim)] if layout.season_dim else []))
    start = np.linalg.lstsq(X, y, rcond=None)[0]
    params = dict(alpha0=float(start[0]), beta0=float(start[1]) if layout.has_beta else 0.,
                  gamma0_season=start[1+int(layout.has_beta):].copy(),
                  s_level=0., s_trend=0., s_season=0.)
    for field, prior in [('alpha0', p.initial_level), ('beta0', p.initial_slope)]:
        if isinstance(prior, Fixed):
            params[field] = sign*prior.value
    if not layout.has_beta:
        params['beta0'] = 0.
    if layout.season_dim and isinstance(p.initial_seasonal, Fixed):
        effects = contrasts(model.period) @ np.full(layout.season_dim, sign*p.initial_seasonal.value)
        params['gamma0_season'] = fs.seasonal_state_from_phase_effects(effects)
    for name in model.active:
        prior = getattr(p, name)
        params[FIELDS[name]] = prior.value if isinstance(prior, Fixed) else (
            shared.get(name, prior.sd) * rng.choice([-1, 1]) * rng.uniform(.2, .8))
    residual = y - fs.baseline_mu_path(len(y), params, layout)
    sigma2 = p.variance.value if isinstance(p.variance, Fixed) else max(np.std(residual), .05)**2
    xi = p.shape.value if isinstance(p.shape, Fixed) else 0.
    state = State(channel.name, model, y, layout, obs, sign, params, sigma2, xi,
        np.zeros(model.period-1 if model.observation.scale else 0),
        np.zeros((len(y)+1, layout.ncp_state_dim)))
    if isinstance(model.observation, GEV):
        # Only start values are adapted; the priors and likelihood stay fixed.
        for _ in range(60):
            op = state.observation_params()
            ll = obs.logpdf(y, fs.mu_from_ncp(state.z, params, layout), op)
            if np.all(np.isfinite(ll)):
                break
            if not isinstance(p.variance, Fixed):
                state.sigma2 *= 4
                continue
            # A known nonzero shape/scale can invalidate least-squares start
            # values even for an entirely static, otherwise feasible model.
            # Shift a free initial location (or rate) strictly into support.
            eta = fs.mu_from_ncp(state.z, params, layout)
            boundary = y + op['sigma']/xi if xi != 0 else y + 500*op['sigma']
            delta = boundary + (.05 if xi < 0 else -.05)*op['sigma'] - eta
            if not isinstance(p.initial_level, Fixed):
                params['alpha0'] += max(0., delta.max()) if xi < 0 else min(0., delta.min())
                continue
            if layout.has_beta and not isinstance(p.initial_slope, Fixed):
                delta /= np.arange(1, len(y)+1)
                params['beta0'] += max(0., delta.max()) if xi < 0 else min(0., delta.min())
                continue
            G, Q = fs.build_ncp_system(layout)
            state.z, _ = _deterministic_feasible_ncp_path(y, SimpleNamespace(obs=obs), params, op, layout, G, Q,
                fs.measurement_vector(params, layout), fs.baseline_mu_path(len(y), params, layout))
            break
        if not np.all(np.isfinite(obs.logpdf(y, fs.mu_from_ncp(state.z, params, layout), state.observation_params()))):
            raise ValueError(f'{channel.name}: cannot construct a finite initial GEV state; check fixed parameters.')
    return state


def coefficient_prior(state, names, shared):
    """Full exact prior moments; fixed coefficients have zero variance."""
    p, k = state.model.priors, state.layout.season_dim
    mean, cov = np.zeros(len(names)), np.zeros((len(names), len(names)))
    for i, name in enumerate(names):
        if name.startswith('gamma0_season_'):
            continue
        if name == 'alpha0':
            prior, sign = p.initial_level, state.sign
        elif name == 'beta0':
            prior, sign = p.initial_slope, state.sign
        else:
            component = next(c for c, field in FIELDS.items() if field == name)
            prior, sign = getattr(p, component), 1
            if component in shared and component in state.model.active:
                prior = Normal(0, shared[component])
            if component not in state.model.active:
                prior = Fixed(0)
        mean[i] = _prior_value(prior, sign)
        cov[i, i] = 0 if isinstance(prior, Fixed) else prior.sd**2
    ids = [i for i, name in enumerate(names) if name.startswith('gamma0_season_')]
    if ids:
        p0 = p.initial_seasonal
        effects = contrasts(k+1) @ np.full(k, _prior_value(p0, state.sign))
        mean[ids] = fs.seasonal_state_from_phase_effects(effects)
        if isinstance(p0, Normal):
            cov[np.ix_(ids, ids)] = p0.sd**2 * (np.eye(k) - np.ones((k, k))/(k+1))
    return mean, cov


def coefficients_step(state, shared, controls, rng):
    X, names, _ = fs.design_matrix_ncp(state.z, state.layout, center_time=False)
    mean, cov = coefficient_prior(state, names, shared)
    free = np.flatnonzero(np.diag(cov) > 0)
    fixed = np.flatnonzero(np.diag(cov) == 0)
    values = mean.copy()
    if free.size:
        offset = X[:, fixed] @ mean[fixed]
        y = state.y - offset
        design, pm = X[:, free], mean[free]
        L = np.linalg.cholesky(cov[np.ix_(free, free)])
        op = state.observation_params()
        if state.obs.name == 'gaussian':
            response, variance = y, op['sigma']**2
        else:
            response, variance = _ncp_laplace_pseudo_data(y, y, SimpleNamespace(obs=state.obs), op,
                curvature_floor=controls.curvature_floor, maximum_variance=controls.maximum_variance,
                shift_limit=10*float(np.max(op['sigma'])))
        reference = gaussian_reference(design, response, variance, pm, L)
        metric = {}
        if state.obs.name == 'gaussian':
            m, root, _ = reference
            w, evaluations = m + root @ rng.normal(size=len(m)), 1
        else:
            (m, root, precision), metric = coefficient_reference(design, y, state.obs, op,
                                                   pm, L, reference, controls)
            current = fs.theta_vector_from_params(state.params, names, state.layout, tbar=0)[free]
            current = np.linalg.solve(L, current - pm)
            def correction(w):
                with np.errstate(over='ignore', invalid='ignore'):
                    ll = float(np.sum(state.obs.logpdf(y, design @ (pm+L@w), op)))
                delta = w-m
                return ll - .5*(w@w) + .5*(delta@precision@delta) if np.isfinite(ll) else -np.inf
            w, evaluations = reference_slice(current, m, root, correction, rng)
        values[free] = pm + L@w
    else:
        metric, evaluations = {}, 0
    state.params = fs.apply_theta_draw(state.params, values, names, state.layout)
    return {**metric, 'coefficient_evaluations': evaluations}


def observation_step(state, rng):
    priors = state.model.priors
    eta = fs.mu_from_ncp(state.z, state.params, state.layout)
    def ll(**kwargs):
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            value = float(np.sum(state.obs.logpdf(state.y, eta, state.observation_params(**kwargs))))
        return value if np.isfinite(value) else -np.inf
    if not isinstance(priors.variance, Fixed):
        if state.obs.name == 'gaussian':
            factors = state.observation_params()['sigma']/np.sqrt(state.sigma2)
            b = priors.variance.scale + .5*np.sum(((state.y-eta)/factors)**2)
            state.sigma2 = b/rng.gamma(priors.variance.shape + len(eta)/2)
        else:
            def target(kappa):
                with np.errstate(over='ignore', under='ignore'):
                    v, inv = np.exp(2*kappa), np.exp(-2*kappa)
                if not np.isfinite(v) or v <= 0 or not np.isfinite(inv):
                    return -np.inf
                return ll(sigma2=v) - 2*priors.variance.shape*kappa - priors.variance.scale*inv
            kappa, _ = _slice_sample_real(.5*np.log(state.sigma2), target, rng, width=.1)
            state.sigma2 = np.exp(2*kappa)
    if state.scale_contrasts.size:
        size = state.scale_contrasts.size
        root = np.eye(size)*state.model.observation.scale.prior_sd
        state.scale_contrasts, _ = reference_slice(state.scale_contrasts, np.zeros(size), root,
                                                     lambda u: ll(u=u), rng)
    if state.obs.name == 'gev' and not isinstance(priors.shape, Fixed):
        def target(xi):
            return ll(xi=xi) - .5*((xi-priors.shape.mean)/priors.shape.sd)**2
        state.xi, _ = _slice_sample_real(state.xi, target, rng, width=.05)


def step(state, shared, controls, rng):
    G, Q = fs.build_ncp_system(state.layout)
    op = state.observation_params()
    if state.obs.name == 'gaussian':
        state.z = fs.ffbs_gaussian_1d_tvR(state.y-fs.baseline_mu_path(len(state.y), state.params, state.layout),
            G, Q, fs.measurement_vector(state.params, state.layout), op['sigma']**2, rng=rng)
        state.z = _project_ncp_path_to_support(state.z, G, Q)
        metric = {}
    else:
        proposal = ncp_laplace_mh(state.y, SimpleNamespace(obs=state.obs), state.params, op, state.layout,
            state.z, rng=rng, mh_steps=controls.mh_steps, max_iterations=controls.max_iterations,
            tolerance=controls.tolerance, curvature_floor=controls.curvature_floor,
            maximum_variance=controls.maximum_variance)
        state.z = proposal.z_path
        metric = dict(path_acceptance=proposal.acceptance_rate,
            path_support_rejections=proposal.proposal_support_failures,
            laplace_iterations=proposal.approximation.iterations,
            laplace_converged=float(proposal.approximation.converged))
    metric.update(coefficients_step(state, shared, controls, rng))
    # Exact sign symmetry for zero-centred normal amplitudes; fixed ones stay fixed.
    for c in state.model.active:
        if isinstance(getattr(state.model.priors, c), Fixed) or rng.random() >= .5:
            continue
        indices = {'level': [state.layout.idx_tilde_alpha],
            'slope': [state.layout.idx_tilde_beta, state.layout.idx_A],
            'seasonal': list(range(state.layout.season_ncp_slice.start, state.layout.season_ncp_slice.stop))}[c]
        state.params[FIELDS[c]] *= -1
        state.z[:, indices] *= -1
    observation_step(state, rng)
    metric['log_likelihood'] = float(np.sum(state.obs.logpdf(state.y,
        fs.mu_from_ncp(state.z, state.params, state.layout), state.observation_params())))
    if not np.isfinite(metric['log_likelihood']):
        raise FloatingPointError(f'{state.name}: sampler reached a nonfinite exact likelihood.')
    return metric


def parameter_values(state):
    p = state.params
    result = dict(initial_level=state.sign*p['alpha0'], initial_slope=state.sign*p['beta0'],
                  sigma=np.sqrt(state.sigma2))
    for c, field in FIELDS.items():
        result['variance.'+c] = p[field]**2
    if state.layout.season_dim:
        g = state.sign*p['gamma0_season']
        phase = fs.static_seasonal_design(state.model.period, state.layout.season_dim) @ g
        for i, value in enumerate(phase):
            result[f'initial_seasonal[{i}]'] = value
    if state.scale_contrasts.size:
        sigmas = np.sqrt(state.sigma2)*np.exp(contrasts(state.model.period)@state.scale_contrasts)
        for i, value in enumerate(sigmas):
            result[f'sigma[{i}]'] = value
    if state.obs.name == 'gev':
        result['xi'] = state.xi
    return result


def run_chain(payload):
    model, data, mcmc, controls, chain_id = payload
    start = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence(mcmc.seed, spawn_key=(chain_id,)))
    shared = {c: getattr(model.pooling, c).sd for c in FIELDS
              if model.pooling and getattr(model.pooling, c) is not None}
    with threadpool_limits(limits=1):
        states = [initialize(c, data[c.name], rng, shared) for c in model.channels]
        saved = {s.name: dict(states=np.empty((mcmc.draws, len(s.y)+1, s.layout.centered_state_dim)),
                    parameters={k: np.empty(mcmc.draws) for k in parameter_values(s)}, metrics={}) for s in states}
        scales = {c: np.empty(mcmc.draws) for c in shared}
        for iteration in range(mcmc.warmup+mcmc.draws):
            metrics = {s.name: step(s, shared, controls, rng) for s in states}
            for c in shared:
                coefficients = [s.params[FIELDS[c]] for s in states if c in s.model.active]
                anchor = getattr(model.pooling, c).sd
                shared[c] = anchor*np.exp(_half_normal_gig_log_multiplier(coefficients, anchor=anchor, rng=rng))
                if not np.isfinite(shared[c]) or shared[c] <= 0:
                    raise FloatingPointError('Shared scale is not representable; no artificial floor is applied.')
            index = iteration-mcmc.warmup
            if index >= 0:
                for s in states:
                    target = saved[s.name]
                    target['states'][index] = s.sign*fs.map_ncp_to_centered(s.z, s.params, s.layout)
                    for k, value in parameter_values(s).items():
                        target['parameters'][k][index] = value
                    for k, value in metrics[s.name].items():
                        target['metrics'].setdefault(k, np.empty(mcmc.draws))[index] = value
                for c, value in shared.items():
                    scales[c][index] = value
            if mcmc.progress and ((iteration+1) % mcmc.progress_every == 0 or iteration+1 == mcmc.draws+mcmc.warmup):
                print(f'chain {chain_id}: {iteration+1}/{mcmc.draws+mcmc.warmup} iterations, {perf_counter()-start:.1f}s', flush=True)
    return saved, scales, perf_counter()-start
