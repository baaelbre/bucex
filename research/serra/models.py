"""Explicit scientific declarations; configuration and output live elsewhere."""
import numpy as np
import bucex as bx


JOINT_MODES = {"joint", "shared", "copula", "shared_copula"}


def channel(name, data, config, *, shared=False, continuous=False):
    """One response distribution and its own cycle, in original Celsius units."""
    info, settings, p = bx.UCCLE_INFO[name], config["model"], config["priors"]
    observation = (bx.Gaussian() if info["family"] == "gaussian" else
                   bx.GEV(phi=settings.get("phi", "stationary"),
                          xi_bounds=tuple(p["xi_bounds"])))
    if shared:
        trend = bx.LocalLevel(mode="static", initial_mean=float(np.median(data[name])),
                              initial_sd=p["baseline_sd"])
    else:
        initial = dict(initial_level=float(np.median(data[name])),
                       initial_level_sd=p["baseline_sd"],
                       initial_slope_sd=p["initial_slope_sd"]) if continuous else {}
        trend = bx.LocalLinearTrend(level_mode=settings.get("level", "dynamic"),
                                    trend_mode=settings.get("trend", "dynamic"), **initial)
    seasonal = bx.DummySeasonal(period=settings["period"],
        mode=settings.get("seasonal", "static" if shared else "dynamic"),
        initial_sd=p["seasonal_initial_sd"] if continuous or shared else None)
    return bx.Channel(name, observation, (trend, seasonal), tail=info["tail"])


def marginal_prior(channel, data, config):
    """Structural selection is independent of optional GEV log-scale selection."""
    p = config["priors"]
    builder = bx.ssvs_gaussian_priors if channel.family == "gaussian" else bx.ssvs_gev_priors
    arguments = dict(
        period=config["model"]["period"],
        alpha_mean=channel.transform_sign * float(np.median(data[channel.name])),
        alpha_sd=p["baseline_sd"], beta_sd=p["initial_slope_sd"],
        seasonal_initial_sd=p["seasonal_initial_sd"],
        innovation_slab_sd=p["innovation_slab_sd"],
        season_probabilities=tuple(p.get("season_probabilities", (0.0, 0.5, 0.5))),
        sigma2_prior=bx.InverseGammaPrior(*p["observation_variance"]),
    )
    if channel.family == "gev":
        arguments["xi_prior"] = bx.UniformPrior(*p["xi_bounds"])
        if "phi" in p:
            phi = p["phi"]
            arguments["phi_prior"] = bx.PhiPrior(
                linear=bx.NormalPrior(**phi["linear"]),
                rw_variance=bx.InverseGammaPrior(**phi["rw_variance"]),
                model_probabilities=phi["model_probabilities"])
    return builder(**arguments)


def joint_model(data, config):
    """Separate selection pooling, shared trajectories and residual dependence."""
    mode, p = config["analysis"], config["priors"]
    if config["model"].get("phi", "stationary") != "stationary":
        raise ValueError("Dynamic GEV log scale currently requires analysis='independent'.")
    if mode == "hierarchical":
        model = bx.MultiSeriesModel(tuple(channel(name, data, config) for name in data))
        prior = bx.HierarchicalPriors(
            hierarchy=bx.HierarchicalPrior(pool="selection",
                coefficient_scale=p["innovation_slab_sd"]),
            channels={item.name: marginal_prior(item, data, config) for item in model.channels})
        return model, prior
    if mode not in JOINT_MODES:
        raise ValueError(f"Unknown joint analysis {mode!r}.")
    shared = mode in {"shared", "shared_copula"}
    components = () if not shared else (
        bx.Shared("warming", bx.LocalLinearTrend(initial_level=0, initial_level_sd=0,
                   initial_slope_sd=p["initial_slope_sd"])),
        bx.Departures("departure", bx.LocalLinearTrend(initial_level=0, initial_level_sd=0,
                      initial_slope_sd=p["departure_initial_slope_sd"]),
                      weights=config["model"].get("departure_weights")),
    )
    kwargs = {}
    if mode == "joint":
        kwargs["copula"] = bx.GaussianCopula(correlation=np.eye(len(data.columns)))
    elif mode in {"copula", "shared_copula"}:
        kwargs["copula"] = bx.GaussianCopula(**config.get("copula", {"eta": 2.0}))
    model = bx.MultiSeriesModel(
        tuple(channel(name, data, config, shared=shared, continuous=True) for name in data),
        shared=components, **kwargs)
    process = {name: bx.HalfNormalSD(sd) for name, sd in p.get("shared_process_sd", {}).items()} if shared else {}
    for item in model.channels:
        for component in item.components:
            for name in component.spec.noise_names:
                key = {"slope": "trend", "seasonal": "season"}.get(name, name)
                process[f"channel.{item.name}.{name}"] = bx.HalfNormalSD(p["innovation_slab_sd"][key])
    prior = bx.JointPriors(process=process,
        observation_sd={name: bx.InverseGammaVariance(*p["observation_variance"]) for name in data},
        shape={item.name: bx.UniformPrior(*p["xi_bounds"])
               for item in model.channels if item.family == "gev"})
    return model, prior


def shared_model(data, config):
    """Backward-compatible research helper; same public construction as joint_model."""
    return joint_model(data, {**config, "analysis": "shared"})


def inference_options(config):
    settings = config.get("inference", {})
    options = {"laplace": bx.Laplace(**settings.get("laplace", {}))}
    if config["analysis"] in JOINT_MODES:
        options["shared_sampler"] = bx.SharedSampler(**settings.get("shared_sampler", {}))
    return options


def fit_options(config, *, family="mixed", tail=None):
    """Numerical controls are shared by fitting and each held-out refit."""
    joint = config["analysis"] in JOINT_MODES
    options = dict(engine="laplace_mh" if joint or family != "gaussian" else "ffbs",
                   parameterization="centered" if joint else "fs",
                   mcmc=bx.MCMC(**config["mcmc"]), **inference_options(config))
    if tail is not None:
        options["tail"] = tail
    return options
