"""Worker count must not change posterior transitions, pairing or diagnostics."""
from dataclasses import replace
import os

import numpy as np
import pandas as pd
import pytest

import bucex as bx


def problem(kind):
    rng = np.random.default_rng(371)
    y = pd.Series(rng.normal(size=16), index=pd.date_range("2000-01-01", periods=16, freq="MS"))
    components = (bx.LocalLinearTrend(), bx.DummySeasonal(4))
    options = {}
    if kind in {"fs_gaussian", "fs_gev", "monthly", "disturbance"}:
        family = "gev" if kind == "fs_gev" else "gaussian"
        scale = bx.SeasonalScale(4) if kind == "monthly" else None
        observation = bx.GEV(scale=scale) if family == "gev" else bx.Gaussian(scale=scale)
        model = bx.Model(observation, components)
        options["priors"] = bx.fs_priors(family, period=4)
        if kind == "disturbance":
            options = {"priors": "normal", "parameterization": "noncentered"}
    elif kind in {"copula", "hierarchy"}:
        channels = (bx.Channel("mean", bx.Gaussian(scale=bx.SeasonalScale(4)) if kind == "copula" else bx.Gaussian(), components),
                    bx.Channel("minimum", bx.GEV(), components, tail="lower"))
        model = bx.MultiSeriesModel(channels, copula=bx.GaussianCopula() if kind == "copula" else None)
        y = pd.DataFrame({"mean": y, "minimum": y * .4 + rng.normal(size=len(y))})
        options["priors"] = (bx.MarginalPriors({c.name: bx.fs_priors(c.family, period=4) for c in channels})
                             if kind == "copula" else bx.HierarchicalPrior(pool="both"))
    else:
        from tests.test_shared_sampler import _mixed_shared_fixture
        model, y, priors = _mixed_shared_fixture()
        options["priors"] = priors
    return y, model, options


@pytest.mark.parametrize("kind", ["fs_gaussian", "fs_gev", "monthly", "copula", "disturbance", "hierarchy", "shared"])
def test_serial_parallel_identical_draws_and_saved_chain_axes(kind, tmp_path):
    y, model, options = problem(kind)
    settings = bx.MCMC(chains=2, draws=4, warmup=2, thin=2, seed=371)
    serial = bx.fit(y, model, mcmc=settings, **options)
    parallel = bx.fit(y, model, mcmc=replace(settings, chain_workers=2), **options)
    np.testing.assert_array_equal(serial.state_draws, parallel.state_draws)
    np.testing.assert_array_equal(serial.log_posterior, parallel.log_posterior)
    for mapping in ("parameter_draws", "auxiliary_draws"):
        assert getattr(serial, mapping).keys() == getattr(parallel, mapping).keys()
        for name, value in getattr(serial, mapping).items():
            np.testing.assert_array_equal(value, getattr(parallel, mapping)[name], err_msg=name)
    for group in ("acceptance", "draw_metrics"):
        for name, values in serial.sampler_diagnostics[group].items():
            np.testing.assert_array_equal(values, parallel.sampler_diagnostics[group][name])
    assert not np.array_equal(parallel.state_draws[0], parallel.state_draws[1])
    execution = parallel.sampler_diagnostics["execution"]
    assert execution["start_method"] == "spawn"
    assert all(pid != os.getpid() for pid in execution["worker_pids"])
    assert execution["seed_states"] == serial.sampler_diagnostics["execution"]["seed_states"]
    parallel.save(tmp_path / "parallel.bucex")
    loaded = bx.load_fit(tmp_path / "parallel.bucex")
    assert loaded.config["mcmc"]["chain_workers"] == 2
    np.testing.assert_array_equal(loaded.state_draws, serial.state_draws)
    # Downstream randomness sees identical chain ordering and posterior pairing.
    np.testing.assert_array_equal(serial.forecast(2, draws=5, seed=31).observations,
                                  parallel.forecast(2, draws=5, seed=31).observations)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
def test_invalid_worker_counts_are_rejected(value):
    with pytest.raises(ValueError, match="chain_workers"):
        bx.MCMC(chain_workers=value)


def test_one_chain_caps_workers_and_keeps_seed():
    y, model, options = problem("monthly")
    settings = bx.MCMC(chains=1, draws=2, warmup=1, seed=49)
    first = bx.fit(y, model, mcmc=settings, **options)
    other = bx.fit(y, model, mcmc=replace(settings, chain_workers=4), **options)
    np.testing.assert_array_equal(first.state_draws, other.state_draws)
    assert other.sampler_diagnostics["execution"]["chain_workers"] == 1


def test_four_chains_keep_the_same_order_with_two_or_four_workers():
    y, model, options = problem('monthly')
    settings = bx.MCMC(chains=4, draws=2, warmup=1, seed=60, chain_workers=2)
    first = bx.fit(y, model, mcmc=settings, **options)
    second = bx.fit(y, model, mcmc=replace(settings, chain_workers=4), **options)
    np.testing.assert_array_equal(first.state_draws, second.state_draws)
    assert second.sampler_diagnostics['execution']['chain_order'] == [0,1,2,3]


def test_parallel_errors_do_not_return_partial_posterior():
    y, model, options = problem("copula")
    y.iloc[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="no partial fit returned"):
        bx.fit(y, model, mcmc=bx.MCMC(chains=2, draws=1, warmup=0, chain_workers=2), **options)


def test_shared_warm_start_selects_the_matching_chain():
    y, model, options = problem("shared")
    initial = bx.fit(y, model, mcmc=bx.MCMC(chains=2, draws=2, warmup=1, seed=21), **options)
    settings = bx.MCMC(chains=2, draws=2, warmup=1, seed=22)
    serial = bx.fit(y, model, init=initial, mcmc=settings, **options)
    parallel = bx.fit(y, model, init=initial, mcmc=replace(settings, chain_workers=2), **options)
    np.testing.assert_array_equal(serial.state_draws, parallel.state_draws)
