"""Shared setup for small, explicitly configured SERRA experiments."""
from dataclasses import asdict, replace
import json
import numpy as np
import pandas as pd
import bucex as bx
from research.serra.models import channel, marginal_prior, inference_options
from research.serra.report import save_band


def prior_for(item, data, config, variant):
    prior = marginal_prior(item, data, config)
    profile = variant.get("profile", "ssvs")
    if profile == "manuscript_original":
        builder = bx.manuscript_gaussian_priors if item.family == "gaussian" else bx.manuscript_gev_priors
        prior = builder(period=config["model"]["period"],
                        alpha_mean=item.transform_sign * float(np.median(data[item.name])))
    else:
        prior = bx.innovation_prior_variant(prior, profile=profile,
            multipliers=variant.get("multipliers"), lasso_shape=variant.get("lasso_shape", 1),
            lasso_rate=variant.get("lasso_rate", 1))
    if "initial_slope_multiplier" in variant:
        factor = float(variant["initial_slope_multiplier"])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("initial_slope_multiplier must be finite and positive.")
        prior = replace(prior, beta0=bx.NormalPrior(prior.beta0.mean, prior.beta0.sd * factor))
    if item.family == "gev":
        bounds = variant.get("xi_bounds", config["priors"]["xi_bounds"])
        prior = replace(prior, xi=bx.UniformPrior(*bounds), xi_max_abs=max(abs(v) for v in bounds))
    return prior


def fit_case(data, name, config, variant, *, engine="laplace_mh"):
    item = channel(name, data, config)
    if item.family == "gev":
        bounds = tuple(variant.get("xi_bounds", config["priors"]["xi_bounds"]))
        item = replace(item, observation=bx.GEV(phi=config["model"].get("phi", "stationary"), xi_bounds=bounds))
    model, prior = bx.Model(item.observation, item.components), prior_for(item, data, config, variant)
    fit = bx.fit(data[name], model=model, tail=item.tail, priors=prior,
        engine="ffbs" if item.family == "gaussian" else engine,
        parameterization="fs", mcmc=bx.MCMC(**config["mcmc"]), **inference_options(config))
    return fit, prior


def save_case(fit, prior, directory, config, *, threshold, event_index=-1):
    directory.mkdir(parents=True, exist_ok=True)
    if config.get("save_fits", True):
        fit.save(directory / "fit.bucex")
    bx.save_config(config, directory / "config.json")
    (directory / "declared_priors.json").write_text(json.dumps(asdict(prior),
        default=lambda value: value.tolist(), indent=2) + "\n")
    diagnostic = fit.diagnostics()
    bx.save_config(dict(bucex_version=bx.__version__, model=fit.model.to_dict(),
        inference=fit.plan.to_dict(), warnings=diagnostic["warnings"]), directory / "run.json")
    level = config.get("credible_interval", .90)
    diagnostic["parameters"].to_csv(directory / "mcmc.csv")
    (directory / "engine.json").write_text(json.dumps(
        {key: float(value) if np.isfinite(value) else None for key, value in diagnostic["engine"].items()}, indent=2))
    comparison = bx.compare_innovation_priors(fit, prior,
        size=config.get("prior_draws", 2000), seed=config["seed"], level=level)
    comparison.to_csv(directory / "prior_posterior.csv", index=False)
    scientific = bx.scientific_summary(fit, threshold, event_index=event_index, level=level)
    scientific.to_csv(directory / "scientific_targets.csv")
    prior_targets = bx.prior_predictive_targets(fit.model, prior, fit.n_time, threshold,
        tail="lower" if fit.transform_sign < 0 else "upper", size=config.get("prior_predictive_draws", 100),
        seed=config["seed"], event_index=event_index, level=level)
    pd.concat([prior_targets, scientific.assign(distribution="posterior")]).to_csv(
        directory / "prior_posterior_targets.csv")
    if config.get("figures", True):
        save_band(fit, fit.component_draws("level"), directory / "level")
        save_band(fit, fit.exceedance_probability_draws(threshold, return_labels=False),
                  directory / "risk", ylabel=fit.event_label(threshold))
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(10, 3))
        for axis, component in zip(axes, ("level", "slope", "seasonal")):
            rows = comparison[(comparison.component == component) & (comparison.scale == "SD")]
            for j, (_, row) in enumerate(rows.iterrows()):
                axis.errorbar(row["median"], j, xerr=[[row["median"]-row["lower"]], [row["upper"]-row["median"]]], fmt="o")
            axis.tick_params(labelsize=11)
            axis.xaxis.label.set_size(12)
            axis.set(yticks=[0, 1], yticklabels=["prior", "posterior"], xlabel=f"{component} innovation SD")
        figure.tight_layout(); figure.savefig(directory / "prior_posterior.pdf"); plt.close(figure)
    return scientific
