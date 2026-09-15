"""Result and restart adapters for the shared univariate/multiseries FS kernel."""
from dataclasses import replace

import numpy as np

from ...models import Channel, MultiSeriesModel
from ...models.compiler import compile_model
from ...priors import MarginalPriors
from .fs_utils import infer_ncp_layout, map_centered_to_ncp
from .marginal import sample_marginal_posterior


def export_marginal_start(fit, chain, draw):
    channels = fit.model.channels if fit.is_multiseries_model else [
        Channel("response", fit.model.observation, fit.model.components,
                tail="lower" if fit.transform_sign < 0 else None)]
    def value(key, default=0.):
        return np.asarray(fit.parameter_draws[key])[chain,draw].copy() if key in fit.parameter_draws else default
    result = {"channels": {}, "copula": {key: float(value(key)) for key in fit.parameter_draws if key.startswith("copula.z.")}}
    for channel in channels:
        name = channel.name
        multi = fit.is_multiseries_model
        suffix = f".{name}" if multi else ""
        prefix = f"channel.{name}." if multi else ""
        local = next(b for b in fit.compiled.blocks if b.name == name).compiled if multi else fit.compiled
        layout = infer_ncp_layout(local.model)
        state = {"alpha0": float(value(f"initial.{prefix}level")),
                 "beta0": float(value(f"initial.{prefix}slope")),
                 "gamma0_season": np.asarray(value(f"initial.{prefix}seasonal", np.zeros(0)))}
        for process, legacy in (("level", "level"), ("slope", "trend"), ("seasonal", "season")):
            sd = float(value(f"signed_sd.{prefix}{process}", value(f"sd.{prefix}{process}")))
            state[f"s_{legacy}"], state[f"q_{legacy}"] = sd, sd*sd
        sigma = float(value("sigma"+suffix))
        obs = {"sigma": sigma, "sigma2": sigma*sigma}
        if channel.family == "gev":
            obs["xi"] = float(value("xi"+suffix))
        path = fit.state_draws[chain,draw]
        if multi:
            path = path[:,next(b.state_slice for b in fit.compiled.blocks if b.name == name)]
        effects = value("scale.seasonal"+suffix, np.zeros(1))
        result["channels"][name] = {"state": state, "observation": obs, "effects": effects,
                                     "z_path": map_centered_to_ncp(path, state, layout)}
    return {"__fs_marginal__": result}


def sample_seasonal_univariate(y, model, compiled, prior, plan, *, mcmc, laplace,
                               dates, sign, initial, params_state, params_obs, name=None):
    joint_model = MultiSeriesModel(channels=(Channel("response", model.observation, model.components,
                       tail="lower" if sign < 0 else None),), name=name or "response")
    joint_compiled = compile_model(joint_model, np.asarray(y)[:,None])
    if "__fs_marginal__" in initial:
        start = initial
    else:
        start = {f"channel.response.{k}": v for k,v in params_state.items()
                 if k in {"alpha0", "beta0", "gamma0_season", "s_level", "s_trend", "s_season"}}
        start.update({f"channel.response.{k}": v for k,v in params_obs.items() if k in {"sigma", "xi"}})
        if "__centered_path__" in initial:
            start["channel.response.__centered_path"] = initial["__centered_path__"]
    result = sample_marginal_posterior(np.asarray(y)[:,None], joint_compiled,
        MarginalPriors({"response": prior}), plan, mcmc=mcmc, laplace=laplace,
        dates=dates, initial_parameters=start)
    parameters = {k.replace("channel.response.", "").replace("state.response.", "state_").removesuffix(".response"): v
                  for k,v in result.parameter_draws.items()}
    # Maintain scalar FS aliases used by established scientific scripts.
    for source, alias in (("initial.level", "alpha0"), ("initial.slope", "beta0"), ("initial.seasonal", "gamma0_season"),
                          ("signed_sd.level", "s_level"), ("signed_sd.slope", "s_trend"), ("signed_sd.seasonal", "s_season")):
        if source in parameters:
            parameters[alias] = parameters[source]
    return replace(result, model=model, compiled=compiled, priors=prior, y=np.asarray(y),
                   parameter_draws=parameters, transform_sign=float(sign), series_name=name,
                   metadata={**result.metadata, "joint_model": False})
