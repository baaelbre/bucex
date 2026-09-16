"""Conditional continuous FS coefficients, shrinkage variables and interweaving.

GEV coefficients use an elliptical slice with a fixed Gaussian reference.
The slice likelihood is the exact target divided by that reference. Thus
preconditioning changes efficiency without replacing the GEV/copula target.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from . import fs_utils as fs


@dataclass
class ShrinkageState:
    tau: dict
    lambda2: object
    horseshoe: dict
    triple_gamma: dict

    @classmethod
    def initialize(cls, prior, layout):
        tau, lam = fs.initialise_lasso(prior, layout)
        return cls(tau, lam, fs.initialise_horseshoe(prior, layout),
                   fs.initialise_triple_gamma(prior, layout))

    def variance_scale(self, prior, sigma2):
        return prior.lasso.variance_scale(sigma2) if prior.lasso is not None else 1.

    def arguments(self, prior, sigma2):
        return dict(tau=self.tau, lasso_variance_scale=self.variance_scale(prior, sigma2),
                    horseshoe_state=self.horseshoe or None,
                    triple_gamma_state=self.triple_gamma or None)

    def update(self, state, prior, rng):
        metric = {}
        if prior.lasso is not None:
            self.tau, self.lambda2 = fs.update_lasso_scales(
                state.params_state, self.tau, self.lambda2, prior, state.layout,
                variance_scale=self.variance_scale(prior, state.params_obs['sigma2']), rng=rng)
        if prior.pc is not None:
            self.tau = fs.update_pc_scales(state.params_state, self.tau, prior, state.layout, rng=rng)
        if prior.horseshoe is not None:
            self.horseshoe, moved = fs.update_horseshoe_scales(
                state.params_state, self.horseshoe, prior, state.layout, rng=rng)
            metric.update({k: float(v) for k,v in moved.items()})
        if prior.triple_gamma is not None:
            self.triple_gamma, moved = fs.update_triple_gamma_scales(
                state.params_state, self.triple_gamma, prior, state.layout, rng=rng)
            metric.update({k: float(v) for k,v in moved.items()})
        return metric

    def parameter_values(self, name):
        result = {}
        for k,v in self.tau.items(): result[f'shrinkage.{name}.tau.{k}'] = float(v)
        if isinstance(self.lambda2, dict):
            for k,v in self.lambda2.items(): result[f'shrinkage.{name}.lambda2.{k}'] = float(v)
        elif np.isfinite(self.lambda2): result[f'shrinkage.{name}.lambda2'] = float(self.lambda2)
        for group,values in [('horseshoe',self.horseshoe),('triple_gamma',self.triple_gamma)]:
            for k,v in values.items():
                if isinstance(v,dict):
                    for component,x in v.items(): result[f'{group}.{name}.{k}.{component}'] = float(x)
                elif v is not None: result[f'{group}.{name}.{k}'] = float(v)
        return result

    def restore(self, values, name):
        for k in self.tau:
            self.tau[k]=float(values.get(f'shrinkage.{name}.tau.{k}',self.tau[k]))
        if isinstance(self.lambda2,dict):
            for k in self.lambda2:
                self.lambda2[k]=float(values.get(f'shrinkage.{name}.lambda2.{k}',self.lambda2[k]))
        else:
            self.lambda2=float(values.get(f'shrinkage.{name}.lambda2',self.lambda2))
        for group,items in [('horseshoe',self.horseshoe),('triple_gamma',self.triple_gamma)]:
            for k,v in items.items():
                if isinstance(v,dict):
                    for component in v:
                        v[component]=float(values.get(f'{group}.{name}.{k}.{component}',v[component]))
                elif v is not None:
                    items[k]=float(values.get(f'{group}.{name}.{k}',v))


def gaussian_reference(design, response, variance, prior_mean, prior_root):
    """Gaussian posterior in prior-whitened coordinates; no covariance floors."""
    X = np.asarray(design, float)
    variance = np.broadcast_to(np.asarray(variance,float), (len(X),))
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ValueError('Gaussian reference variances must be positive and finite.')
    A = (X @ prior_root) / np.sqrt(variance[:,None])
    b = (np.asarray(response)-X @ prior_mean) / np.sqrt(variance)
    precision = np.eye(A.shape[1]) + A.T @ A
    factor = np.linalg.cholesky(precision)
    mean = cho_solve((factor,True), A.T @ b)
    # root @ root.T = precision^{-1}; the root need not be triangular.
    root = solve_triangular(factor.T, np.eye(len(mean)), lower=False)
    return mean, root, precision


def reference_slice(current, mean, root, log_correction, rng, *, max_steps=10000):
    """Elliptical slice for N(mean, root root.T) times exp(log_correction)."""
    log_current = float(log_correction(current))
    if not np.isfinite(log_current):
        raise FloatingPointError('Current coefficient vector has an invalid exact target.')
    threshold = log_current + np.log(rng.random())
    direction = root @ rng.normal(size=len(current))
    centered = current-mean
    angle = rng.uniform(0.,2*np.pi); lo,hi = angle-2*np.pi,angle
    for n in range(1,max_steps+1):
        candidate = mean + centered*np.cos(angle) + direction*np.sin(angle)
        if log_correction(candidate) >= threshold:
            return candidate,n
        if angle < 0: lo=angle
        else: hi=angle
        angle = rng.uniform(lo,hi)
    raise FloatingPointError('Coefficient elliptical slice exhausted its bracket budget.')


def coefficient_step(state, conditional, obs_params, prior, mixing, laplace, rng):
    X,names,tbar = fs.design_matrix_ncp(state.z_path,state.layout,center_time=True)
    active = {'s_'+key for key in fs._active_scale_names(state.layout)}
    selected = [i for i,key in enumerate(names) if not key.startswith('s_') or key in active]
    X,names = X[:,selected],[names[i] for i in selected]
    pm,L = coefficient_prior(prior,names,tbar,mixing,state.params_obs['sigma2'])
    if state.family == 'gaussian':
        y = state.y-obs_params['sigma']*conditional.mean
        variance = obs_params['sigma']**2*conditional.variance
    else:
        from types import SimpleNamespace
        y,variance = fs._ncp_laplace_pseudo_data(
            state.y,state.y,SimpleNamespace(obs=conditional),obs_params,
            curvature_floor=laplace.curvature_floor, maximum_variance=laplace.maximum_variance,
            shift_limit=10*float(np.max(obs_params['sigma'])))
    mean,root,precision = gaussian_reference(X,y,variance,pm,L)
    if state.family == 'gaussian':
        whitened = mean+root @ rng.normal(size=len(mean)); evaluations=1
    else:
        current = fs.theta_vector_from_params(state.params_state,names,state.layout,tbar=tbar)
        current = np.linalg.solve(L,current-pm)
        def correction(w):
            ll = float(np.sum(conditional.logpdf(state.y,X @ (pm+L @ w),obs_params)))
            if not np.isfinite(ll): return -np.inf
            delta = w-mean
            return ll-.5*(w@w)+.5*(delta @ precision @ delta)
        whitened,evaluations = reference_slice(current,mean,root,correction,rng)
    state.params_state.update(fs.apply_theta_draw(state.params_state,pm+L @ whitened,names,state.layout,tbar=tbar))
    return {'coefficient_slice_evaluations': evaluations}


def coefficient_prior(prior, names, tbar, mixing, sigma2):
    """Exact prior root in centered-time coordinates, retaining arbitrarily small SDs."""
    mean=np.zeros(len(names)); root=np.zeros((len(names),len(names)))
    for i,name in enumerate(names):
        if name in {'alpha0','alpha_c'}:
            mean[i],root[i,i]=prior.alpha0.mean,prior.alpha0.sd
            if 'beta0' in names:
                j=names.index('beta0')
                mean[i]+=tbar*prior.beta0.mean;root[i,j]=tbar*prior.beta0.sd
        elif name=='beta0': mean[i],root[i,i]=prior.beta0.mean,prior.beta0.sd
        elif name.startswith('gamma0_season_'):
            j=int(name.rsplit('_',1)[1])-1
            mean[i],root[i,i]=prior.gamma0_season.mean_array()[j],prior.gamma0_season.sd_array()[j]
        else:
            component={'s_level':'level','s_trend':'trend','s_season':'season'}[name]
            if prior.lasso is not None:
                v=mixing.variance_scale(prior,sigma2)*prior.lasso.coefficient_scale_for(component)**2*mixing.tau[component]
            elif prior.pc is not None:
                v=prior.pc.coefficient_scale_for(component)**2*mixing.tau[component]
            elif prior.horseshoe is not None:
                v=fs.horseshoe_conditional_variance(prior,mixing.horseshoe,component)
            elif prior.triple_gamma is not None:
                v=fs.triple_gamma_conditional_variance(prior,mixing.triple_gamma,component)
            else:
                p=getattr(prior,name);mean[i]=p.mean;v=p.sd**2
            root[i,i]=np.sqrt(v)
    if np.any(np.diag(root)<=0) or not np.all(np.isfinite(root)):
        raise FloatingPointError('Nonpositive or nonfinite FS prior scale; no prior floor is applied.')
    return mean,root


def sign_step(state, prior, mixing, rng):
    before=fs.mu_from_ncp(state.z_path,state.params_state,state.layout)
    candidate,params = fs.random_sign_switches(state.z_path,state.params_state,state.layout,rng)
    # Normal priors with nonzero means do not have an unconditional sign symmetry.
    ratio=0.
    for process,key in [('level','s_level'),('trend','s_trend'),('season','s_season')]:
        p=getattr(prior,key,None)
        if p is not None and p.mean != 0:
            ratio += -.5*((params.get(key,0)-p.mean)/p.sd)**2 + .5*((state.params_state.get(key,0)-p.mean)/p.sd)**2
    if np.log(rng.random()) < ratio:
        state.z_path,state.params_state=candidate,params
    error=float(np.max(np.abs(before-fs.mu_from_ncp(state.z_path,state.params_state,state.layout))))
    return {'sign_error': error}


def interweave_scales(state, prior, mixing, rng, *, step=.2):
    """Centred-scale MH via exact NCP rescaling, without subtracting baselines.

    This avoids loss of tiny innovations when a large temperature baseline
    is subtracted. Only active continuous scales are interwoven; fixed-zero
    reductions never receive an artificial lower bound or a scale proposal.
    """
    z, layout = state.z_path, state.layout
    indices = {'level': [layout.idx_tilde_alpha],
               'trend': [layout.idx_tilde_beta, layout.idx_A],
               'season': list(range(layout.season_ncp_slice.start,layout.season_ncp_slice.stop))}
    result = {}
    for component in fs._active_scale_names(layout):
        key = 's_'+component
        old = float(state.params_state[key])
        if old == 0:
            raise FloatingPointError('A continuous scale was rounded to zero before ASIS.')
        if component == 'season':
            standardized = z[1:,layout.season_ncp_slice.start] + np.sum(z[:-1,layout.season_ncp_slice],axis=1)
        else:
            standardized = np.diff(z[:,indices[component][0]])
        square = float(standardized @ standardized)
        log_ratio = step*rng.normal()
        proposed = old*np.exp(log_ratio)
        arguments = mixing.arguments(prior,state.params_obs['sigma2'])
        lp_old = fs._signed_scale_prior_logpdf(old,component,prior,**arguments)
        lp_new = fs._signed_scale_prior_logpdf(proposed,component,prior,**arguments)
        ratio = (-(len(standardized)-1)*log_ratio
                 - .5*square*np.expm1(-2*log_ratio) + lp_new-lp_old)
        take = np.log(rng.random()) < ratio
        if take:
            state.params_state[key] = proposed
            state.params_state['q_'+component] = proposed**2
            z[:,indices[component]] *= np.exp(-log_ratio)
        result['asis_'+key] = float(take)
    return result


def continuous_step(state, conditional, obs_params, prior, mixing, laplace, rng, *, asis=False):
    G,Q = fs.build_ncp_system(state.layout)
    if state.family == 'gaussian':
        y=state.y-obs_params['sigma']*conditional.mean
        state.z_path=fs.ffbs_gaussian_1d_tvR(
            y-fs.baseline_mu_path(len(y),state.params_state,state.layout),G,Q,
            fs.measurement_vector(state.params_state,state.layout),
            obs_params['sigma']**2*conditional.variance,
            m0=np.zeros(state.layout.ncp_state_dim),C0=np.zeros((state.layout.ncp_state_dim,)*2),rng=rng)
        metric={}
    else:
        from types import SimpleNamespace
        result=fs.ncp_laplace_mh(
            state.y,SimpleNamespace(obs=conditional),state.params_state,obs_params,state.layout,state.z_path,
            rng=rng,mh_steps=laplace.mh_steps,max_iterations=laplace.max_iterations,
            tolerance=laplace.tolerance,curvature_floor=laplace.curvature_floor,
            maximum_variance=laplace.maximum_variance)
        state.z_path=result.z_path
        metric={'laplace_mh_acceptance':result.acceptance_rate,
                'laplace_converged':float(result.approximation.converged),
                'laplace_iterations':result.approximation.iterations,
                'laplace_initial_support_repaired':float(result.approximation.initial_support_repaired),
                'laplace_support_rejections':result.proposal_support_failures}
    metric.update(coefficient_step(state,conditional,obs_params,prior,mixing,laplace,rng))
    if asis:
        before=fs.mu_from_ncp(state.z_path,state.params_state,state.layout)
        moved=interweave_scales(state,prior,mixing,rng)
        metric.update({key:float(value) for key,value in moved.items()})
        metric['asis_predictor_error']=float(np.max(np.abs(before-fs.mu_from_ncp(state.z_path,state.params_state,state.layout))))
    metric.update(sign_step(state,prior,mixing,rng))
    metric.update(mixing.update(state,prior,rng))
    return metric
