"""Independent checks of the FS prior used for scientific sensitivity plots."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import genextreme, truncnorm

import bucex as bx


def _fixed_structure_prior(*, seasonal=True):
    prior = bx.ssvs_gev_priors(period=4)
    return replace(prior,
        alpha0=bx.NormalPrior(10, 1e-9), beta0=bx.NormalPrior(0, 1e-9),
        gamma0_season=bx.DiagonalNormalPrior((2, 4, 8), (1e-9,)*3) if seasonal else None,
        sigma2=None, log_sigma=bx.NormalPrior(0, 1e-9), xi=bx.NormalPrior(-.2, 1e-9),
        ssvs=replace(prior.ssvs, level_dynamic_probability=0,
            trend_probabilities=(0, 1, 0), season_probabilities=(0, 1, 0)))


def test_structural_normal_shape_draws_match_truncated_prior_and_log_sigma():
    prior = replace(_fixed_structure_prior(), xi=bx.NormalPrior(.4, .2),
        xi_max_abs=.3, log_sigma=bx.NormalPrior(.2, .4))
    draws = bx.draw_structural_prior(prior, size=80000, seed=109)
    expected_mean, expected_variance = truncnorm.stats((-0.3-.4)/.2, (.3-.4)/.2,
        loc=.4, scale=.2, moments="mv")
    assert np.max(draws["xi"]) < .3
    assert np.min(draws["xi"]) > -.3
    assert np.mean(draws["xi"]) == pytest.approx(expected_mean, abs=.002)
    assert np.var(draws["xi"]) == pytest.approx(expected_variance, abs=.0004)
    assert np.mean(np.log(draws["sigma"])) == pytest.approx(.2, abs=.005)
    assert np.std(np.log(draws["sigma"])) == pytest.approx(.4, abs=.005)


def test_pc_scale_draws_satisfy_declared_tail_probabilities():
    prior = replace(_fixed_structure_prior(), ssvs=None,
        pc=bx.PCInnovationPrior(upper={"level": .3, "trend": .004, "season": .2},
            alpha={"level": .05, "trend": .1, "season": .2}))
    draws = bx.draw_structural_prior(prior, size=100000, seed=42)
    for component, name in (("level", "level"), ("trend", "slope"), ("season", "seasonal")):
        probability = np.mean(draws[f"sd.{name}"] > prior.pc.upper[component])
        assert probability == pytest.approx(prior.pc.alpha_for(component), abs=.004)


@pytest.mark.parametrize("tail, sign", [("upper", 1), ("lower", -1)])
def test_prior_predictive_initial_season_is_at_first_observation(tail, sign):
    # Deliberately put season before trend and use noncanonical component names.
    model = bx.Model(bx.GEV(), (
        bx.DummySeasonal(4, name="cycle"),
        bx.LocalLinearTrend(level_name="baseline", slope_name="velocity")))
    prior = _fixed_structure_prior()
    result = bx.prior_predictive_targets(model, prior, 4, sign*13,
        tail=tail, size=20, seed=90, event_index=0)
    # FS gamma0=(2,4,8) is at observation 1, so eta1=10+2.
    assert result.loc["location_event", "median"] == pytest.approx(sign*12, abs=1e-7)
    assert result.loc["level_change", "median"] == pytest.approx(0, abs=1e-7)
    expected_risk = genextreme.sf(13, c=.2, loc=12, scale=1)
    assert result.loc["risk_event", "median"] == pytest.approx(expected_risk, abs=1e-7)
    assert result.loc["endpoint_event", "median"] == pytest.approx(sign*17, abs=1e-7)


def test_prior_predictive_linear_scale_uses_time_specific_risks_and_endpoint():
    prior = replace(_fixed_structure_prior(seasonal=False),
        phi=bx.PhiPrior(linear=bx.NormalPrior(np.log(4), 1e-9)))
    model = bx.Model(bx.GEV(phi="linear"), (bx.LocalLinearTrend(),))
    result = bx.prior_predictive_targets(model, prior, 4, 11, size=20, seed=80)
    # Centered log-scale basis gives sigma_start=.5, sigma_end=2.
    assert result.loc["risk_start", "median"] == pytest.approx(genextreme.sf(11, .2, loc=10, scale=.5), abs=1e-7)
    assert result.loc["risk_end", "median"] == pytest.approx(genextreme.sf(11, .2, loc=10, scale=2), abs=1e-7)
    assert result.loc["endpoint_event", "median"] == pytest.approx(20, abs=1e-7)
    assert result.loc["return_level_100_blocks", "median"] == pytest.approx(
        genextreme.ppf(.99, .2, loc=10, scale=2), abs=1e-7)


def test_prior_predictive_rejects_incompatible_component_and_season_dimensions():
    prior = _fixed_structure_prior()
    with pytest.raises(ValueError, match="period-1"):
        bx.prior_predictive_targets(bx.Model(bx.GEV(), (bx.LocalLinearTrend(), bx.DummySeasonal(12))),
            prior, 12, 20, size=2)
    with pytest.raises(ValueError, match="full dynamic"):
        bx.prior_predictive_targets(bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static"),)),
            prior, 12, 20, size=2)


def test_prior_predictive_preserves_infinite_endpoint_mass():
    prior = replace(_fixed_structure_prior(seasonal=False), xi=bx.UniformPrior(.1, .4))
    model = bx.Model(bx.GEV(), (bx.LocalLinearTrend(),))
    result = bx.prior_predictive_targets(model, prior, 4, 11, size=20, seed=80)
    assert result.loc["finite_endpoint", "median"] == 0
    assert result.loc["endpoint_event", "finite_fraction"] == 0
    assert np.isposinf(result.loc["endpoint_event", ["lower", "median", "upper"]].to_numpy(dtype=float)).all()
