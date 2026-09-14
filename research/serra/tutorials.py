"""Readable API demonstrations: python -m research.serra.tutorials --kind all.

The smoke config checks execution; full settings remain a starting MCMC budget.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import genextreme
import bucex as bx
from research.serra.report import new_run, save_band, write_report


def gaussian(config):
    model = bx.Model(bx.Gaussian(), (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ))
    simulation = bx.simulate(model, config["n_time"], {
        "sigma": 0.8, "sd.level": 0.02, "sd.slope": 0.00005,
        "sd.seasonal": 0.02,
    }, initial_state=np.r_[10, 0.005, 5 * np.sin(2 * np.pi * np.arange(11) / 12)],
       seed=config["seed"])
    priors = bx.ssvs_gaussian_priors(
        period=12, alpha_mean=float(np.median(simulation.y)),
        innovation_slab_sd={"level": 0.02, "trend": 0.00005, "season": 0.02},
    )
    fit = bx.fit(simulation.y, model=model, priors=priors, engine="ffbs",
                 parameterization="fs", mcmc=bx.MCMC(**config["mcmc"]))
    print(fit.component_probabilities())
    print(fit.diagnostics()["parameters"])
    print(write_report(fit, new_run(config["output"], "gaussian"), config=config))



def gev(config):
    model = bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ))
    simulation = bx.simulate(model, config["n_time"], {
        "sigma": 1.5, "xi": -0.2, "sd.level": 0.02,
        "sd.slope": 0.00005, "sd.seasonal": 0.02,
    }, initial_state=np.r_[25, 0.005, 3 * np.sin(2 * np.pi * np.arange(11) / 12)],
       seed=config["seed"])
    priors = bx.ssvs_gev_priors(
        period=12, alpha_mean=float(np.median(simulation.y)),
        innovation_slab_sd={"level": 0.02, "trend": 0.00005, "season": 0.02},
        sigma2_prior=bx.InverseGammaPrior(2, 2),
        xi_prior=bx.UniformPrior(-0.5, 0.5),
    )
    fit = bx.fit(simulation.y, model=model, priors=priors, engine="laplace_mh",
                 parameterization="fs", mcmc=bx.MCMC(**config["mcmc"]))
    print(fit.component_probabilities())
    print(fit.diagnostics()["engine"])
    # Threshold probabilities integrate posterior uncertainty in states and tails.
    print(write_report(fit, new_run(config["output"], "gev"), config=config,
                       risks={"series": 35.0}))



def hierarchical(config):
    rng = np.random.default_rng(config["seed"])
    time = np.arange(config["n_time"])
    location = 0.005 * time + np.sin(2 * np.pi * time / 12)
    data = pd.DataFrame({
        "mean": location + rng.normal(0, 0.8, time.size),
        "maximum": genextreme.rvs(c=0.2, loc=location + 3, scale=1,
                                  random_state=rng),
    })
    model = bx.MultiSeriesModel(tuple(
        bx.Channel(name, observation, (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=12, mode="dynamic"),
        )) for name, observation in (("mean", bx.Gaussian()), ("maximum", bx.GEV()))
    ))
    prior = bx.HierarchicalPrior(pool="selection", coefficient_scale={
        "level": 0.02, "trend": 0.00005, "season": 0.02,
    })
    fit = bx.fit(data, model=model, priors=prior, engine="laplace_mh",
                 parameterization="fs", mcmc=bx.MCMC(**config["mcmc"]))
    print(fit.hierarchical_probabilities())
    print(write_report(fit, new_run(config["output"], "hierarchical"),
                       config=config, risks={"maximum": 5.0}))



def shared(config):
    rng = np.random.default_rng(config["seed"])
    time = np.arange(config["n_time"])
    common = 0.005 * time
    departure = 0.002 * time
    season = np.sin(2 * np.pi * time / 12)
    data = pd.DataFrame({
        "mean": common + departure + season + rng.normal(0, 0.8, time.size),
        "maximum": genextreme.rvs(c=0.2, loc=3 + common - departure + 2 * season,
                                  scale=1, random_state=rng),
    })
    model = bx.MultiSeriesModel(
        channels=tuple(bx.Channel(name, observation, (
            bx.LocalLevel(mode="static", initial_mean=baseline, initial_sd=3),
            bx.DummySeasonal(period=12, mode="static", initial_sd=3),
        )) for name, observation, baseline in (
            ("mean", bx.Gaussian(), 0), ("maximum", bx.GEV(), 3),
        )),
        shared=(
            bx.Shared("warming", bx.LocalLinearTrend(
                initial_level=0, initial_level_sd=0, initial_slope_sd=0.005,
            )),
            bx.Departures("departure", bx.LocalLinearTrend(
                initial_level=0, initial_level_sd=0, initial_slope_sd=0.003,
            )),
        ),
    )
    priors = bx.JointPriors(
        process={
            "shared.warming.level": bx.HalfNormalSD(0.02),
            "shared.warming.slope": bx.HalfNormalSD(0.00005),
            "departure.departure.level": bx.HalfNormalSD(0.01),
            "departure.departure.slope": bx.HalfNormalSD(0.000025),
        },
        observation_sd={name: bx.InverseGammaVariance(2, 2) for name in data},
        shape={"maximum": bx.UniformPrior(-0.5, 0.5)},
    )
    fit = bx.fit(data, model=model, priors=priors, engine="laplace_mh",
                 parameterization="centered", mcmc=bx.MCMC(**config["mcmc"]),
                 shared_sampler=bx.SharedSampler(**config.get("shared_sampler", {})))
    directory = write_report(fit, new_run(config["output"], "shared"),
                             config=config, risks={"maximum": 5.0})
    print(fit.diagnostics()["parameters"])
    print(directory)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("gaussian", "gev", "hierarchical", "shared", "all"), default="all")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config") / "tutorial_smoke.json")
    args = parser.parse_args()
    config = bx.load_config(args.config)
    demonstrations = {"gaussian": gaussian, "gev": gev, "hierarchical": hierarchical, "shared": shared}
    for name in demonstrations if args.kind == "all" else (args.kind,):
        demonstrations[name](config)


if __name__ == "__main__":
    main()
