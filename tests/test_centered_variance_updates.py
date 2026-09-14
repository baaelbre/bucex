from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit.disturbance import (
    _draw_centered_inverse_gamma_sd,
)


class _RecordedGammaDraw:
    def __init__(self, value: float):
        self.value = float(value)
        self.shape: float | None = None
        self.scale: float | None = None

    def gamma(self, *, shape: float, scale: float) -> float:
        self.shape = float(shape)
        self.scale = float(scale)
        return self.value


def _model_and_data():
    model = bx.Model(
        bx.Gaussian(),
        [bx.LocalLevel(mode="dynamic", initial_mean=0.0, initial_sd=1.0)],
    )
    simulation = bx.simulate(
        model,
        24,
        {"sd.level": 0.08, "sigma": 0.40},
        initial_state=[0.0],
        seed=111,
    )
    priors = bx.Priors(
        process={
            "level": bx.InverseGammaVariance(shape=2.0, scale=0.0064)
        },
        observation_sd=bx.FixedSD(0.40),
        profile="centered_ig_test",
    )
    return model, simulation.y, priors


def test_centered_inverse_gamma_full_conditional_uses_shape_scale_formula():
    prior = bx.InverseGammaVariance(shape=2.0, scale=0.5)
    innovations = np.asarray([1.0, -2.0, 3.0])
    rng = _RecordedGammaDraw(4.0)

    sd = _draw_centered_inverse_gamma_sd(prior, innovations, rng)

    assert rng.shape == pytest.approx(3.5)
    assert rng.scale == pytest.approx(1.0 / 7.5)
    assert sd == pytest.approx(0.5)


def test_centered_inverse_gamma_prior_dispatches_to_gibbs_without_acceptance():
    model, y, priors = _model_and_data()
    fit = bx.fit(
        y,
        model=model,
        priors=priors,
        engine="ffbs",
        parameterization="centered",
        asis=False,
        mcmc=bx.MCMC(
            draws=6, warmup=6, chains=1, seed=112, progress=False
        ),
    )

    methods = fit.sampler_diagnostics["update_methods"]
    assert methods["sd.level"] == "inverse_gamma_gibbs"
    assert methods["sigma"] == "fixed"
    assert fit.methods["parameter_updates"]["sd.level"] == (
        "inverse_gamma_gibbs"
    )
    assert "sd.level" not in fit.sampler_diagnostics["acceptance"]
    assert fit.meta["conjugate_process_variance_updates"] == ["sd.level"]
    assert fit.diagnostics()["parameters"].loc["sd.level", "update"] == (
        "inverse_gamma_gibbs"
    )
    assert np.all(np.isfinite(fit.parameter("sd.level")))
    assert np.all(fit.parameter("sd.level") > 0.0)


def test_inverse_gamma_process_prior_remains_mh_when_noncentered():
    model, y, priors = _model_and_data()
    fit = bx.fit(
        y,
        model=model,
        priors=priors,
        engine="ffbs",
        parameterization="disturbance",
        asis=False,
        mcmc=bx.MCMC(
            draws=4, warmup=4, chains=1, seed=113, progress=False
        ),
    )

    assert fit.sampler_diagnostics["update_methods"]["sd.level"] == (
        "log_sd_random_walk_mh"
    )
    assert "sd.level" in fit.sampler_diagnostics["acceptance"]
    assert fit.meta["conjugate_process_variance_updates"] == []


@pytest.mark.parametrize(
    "parameterization,primary_method,asis_method,acceptance_key",
    [
        (
            "centered",
            "inverse_gamma_gibbs",
            "log_sd_random_walk_mh",
            "asis.sd.level",
        ),
        (
            "disturbance",
            "log_sd_random_walk_mh",
            "inverse_gamma_gibbs",
            "sd.level",
        ),
    ],
)
def test_asis_labels_centered_gibbs_and_noncentered_mh_separately(
    parameterization, primary_method, asis_method, acceptance_key
):
    model, y, priors = _model_and_data()
    fit = bx.fit(
        y,
        model=model,
        priors=priors,
        engine="ffbs",
        parameterization=parameterization,
        asis=True,
        mcmc=bx.MCMC(
            draws=3, warmup=3, chains=1, seed=114, progress=False
        ),
    )

    methods = fit.methods["parameter_updates"]
    assert methods["sd.level"] == primary_method
    assert methods["asis.sd.level"] == asis_method
    assert set(fit.sampler_diagnostics["acceptance"]) == {acceptance_key}
