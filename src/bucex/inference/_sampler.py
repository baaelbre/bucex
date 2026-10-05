"""One MCMC chain, coordinated through compiled component and parameter blocks."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from ..models import compile_model
from ..distributions import contrasts
from ..priors import Fixed
from .smoothers import sample_gaussian
from .paths import finite_start, project_path, laplace_mh
from .coefficients import coefficients_step, coefficient_prior
from .observation import observation_step
from ._shared import _half_normal_gig_log_multiplier


@dataclass
class State:
    name: str
    model: object
    y: np.ndarray
    compiled: object
    obs: object
    theta: np.ndarray
    sigma2: float
    xi: float
    scale_contrasts: np.ndarray
    z: np.ndarray

    def location(self):
        return self.compiled.coefficient_design(self.z) @ self.theta

    def observation_params(self, *, sigma2=None, xi=None, u=None):
        variance = self.sigma2 if sigma2 is None else sigma2
        effects = self.scale_contrasts if u is None else u
        if effects.size:
            log_effect = contrasts(self.model.period) @ effects
            sigma = np.sqrt(variance)*np.exp(log_effect[np.arange(len(self.y))%self.model.period])
        else:
            sigma = np.full(len(self.y), np.sqrt(variance))
        return {'sigma': sigma, 'xi': self.xi if xi is None else xi}


def initialize(channel, y, exog, rng, shared):
    model = channel.model
    compiled = compile_model(model, len(y), exog)
    p = model.priors
    theta = np.array([prior.value if isinstance(prior, Fixed) else prior.mean for prior in compiled.priors], dtype=float)
    n0 = compiled.initial_count
    free = np.array([i for i in range(n0) if not isinstance(compiled.priors[i], Fixed)], dtype=int)
    fixed = np.array([i for i in range(n0) if isinstance(compiled.priors[i], Fixed)], dtype=int)
    if free.size:
        theta[free] = np.linalg.lstsq(compiled.initial_design[:, free],
                          y-compiled.initial_design[:, fixed]@theta[fixed], rcond=None)[0]
    for g in compiled.groups:
        theta[g.coefficient] = g.prior.value if isinstance(g.prior, Fixed) else shared.get(g.name, g.prior.sd)*rng.choice([-1., 1.])*rng.uniform(.2, .8)
    residual = y-compiled.offset(theta)
    variance = p.variance.value if isinstance(p.variance, Fixed) else max(np.std(residual), .05)**2
    xi = p.shape.value if isinstance(p.shape, Fixed) else p.shape.mean
    state = State(channel.name, model, np.asarray(y), compiled, model.observation, theta,
        variance, xi, np.zeros(model.period-1 if model.observation.scale else 0),
        np.zeros((len(y)+1, compiled.ncp_dim)))
    for _ in range(60):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ok = np.all(np.isfinite(state.obs.logpdf(y, state.location(), state.observation_params())))
        if ok:
            return state
        if not isinstance(p.variance, Fixed):
            state.sigma2 *= 4
            continue
        # Repair an estimated intercept for fixed finite GEV endpoints.
        intercept = [i for i in free if np.all(compiled.initial_design[:, i] == 1)]
        if state.obs.name == 'gev' and xi != 0 and intercept:
            sigma = state.observation_params()['sigma']
            boundary = y+state.obs.sign*sigma/xi
            delta = boundary-state.location()
            theta[intercept[0]] += max(0., np.max(delta+.05*sigma)) if xi*state.obs.sign < 0 else min(0., np.min(delta-.05*sigma))
            continue
        if compiled.ncp_dim:
            state.z = finite_start(y, state.obs, state.observation_params(), compiled.ncp_transition,
                compiled.ncp_covariance, compiled.path_design(theta), compiled.offset(theta))
            continue
        break
    raise ValueError(f'{channel.name}: cannot construct a finite initial state; check fixed parameters and support.')


def step(state, shared, controls, rng):
    b, op = state.compiled, state.observation_params()
    F, Q, H, offset = b.ncp_transition, b.ncp_covariance, b.path_design(state.theta), b.offset(state.theta)
    metric = {}
    if b.ncp_dim:
        if getattr(state.obs, 'gaussian_location', False):
            state.z = project_path(sample_gaussian(state.y-offset, F, Q, H, op['sigma']**2, rng=rng), F, Q)
        else:
            state.z, metric = laplace_mh(state.y, state.obs, op, F, Q, H, offset, state.z, controls, rng)
    metric.update(coefficients_step(state, shared, controls, rng))
    for g in b.groups:
        if not isinstance(g.prior, Fixed) and rng.random() < .5:
            state.theta[g.coefficient] *= -1
            state.z[:, g.state_slice] *= -1
    observation_step(state, rng)
    metric['log_likelihood'] = float(np.sum(state.obs.logpdf(state.y, state.location(), state.observation_params())))
    if not np.isfinite(metric['log_likelihood']):
        raise FloatingPointError(f'{state.name}: nonfinite exact likelihood.')
    return metric


def parameter_values(state):
    result = state.compiled.parameter_values(state.theta)
    result['sigma'] = np.sqrt(state.sigma2)
    # These convenience zeroes describe absent paper components; the compiler
    # itself is generic and only exposes innovations present in the model.
    for name in ('level', 'slope', 'seasonal'):
        result.setdefault('variance.'+name, 0.)
    result.setdefault('initial_slope', 0.)
    if state.scale_contrasts.size:
        sigmas = np.sqrt(state.sigma2)*np.exp(contrasts(state.model.period)@state.scale_contrasts)
        for i, value in enumerate(sigmas):
            result[f'sigma[{i}]'] = value
    if state.obs.name == 'gev':
        result['xi'] = state.xi
    return result


def run_chain(payload):
    model, data, exog, mcmc, controls, chain_id = payload
    start = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence(mcmc.seed, spawn_key=(chain_id,)))
    pooling = model.pooling.scales if model.pooling else {}
    shared = {name: prior.sd for name, prior in pooling.items()}
    with threadpool_limits(limits=1):
        states = [initialize(c, data[c.name], exog[c.name], rng, shared) for c in model.channels]
        saved = {s.name: dict(states=np.empty((mcmc.draws, len(s.y)+1, s.compiled.state_dim)),
                    parameters={k: np.empty(mcmc.draws) for k in parameter_values(s)}, metrics={}) for s in states}
        scales = {name: np.empty(mcmc.draws) for name in shared}
        for iteration in range(mcmc.warmup+mcmc.draws):
            metrics = {s.name: step(s, shared, controls, rng) for s in states}
            for name in shared:
                amplitudes = [s.theta[g.coefficient] for s in states for g in s.compiled.groups if g.name == name]
                anchor = pooling[name].sd
                shared[name] = anchor*np.exp(_half_normal_gig_log_multiplier(amplitudes, anchor=anchor, rng=rng))
                if not np.isfinite(shared[name]) or shared[name] <= 0:
                    raise FloatingPointError('Shared scale is not representable; no artificial floor is applied.')
            index = iteration-mcmc.warmup
            if index >= 0:
                for s in states:
                    target = saved[s.name]
                    target['states'][index] = s.compiled.physical_path(s.theta, s.z)
                    for k, value in parameter_values(s).items():
                        target['parameters'][k][index] = value
                    for k, value in metrics[s.name].items():
                        target['metrics'].setdefault(k, np.empty(mcmc.draws))[index] = value
                for name, value in shared.items():
                    scales[name][index] = value
            if mcmc.progress and ((iteration+1)%mcmc.progress_every == 0 or iteration+1 == mcmc.draws+mcmc.warmup):
                print(f'chain {chain_id}: {iteration+1}/{mcmc.draws+mcmc.warmup} iterations, {perf_counter()-start:.1f}s', flush=True)
    return saved, scales, perf_counter()-start
