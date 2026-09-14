from __future__ import annotations


import pytest

import bucex as bx
from bucex.datasets.uccle import _normalize_series_names


@pytest.mark.parametrize("builder", [bx.ssvs_gaussian_priors, bx.ssvs_gev_priors])
def test_ssvs_prior_builders_accept_readable_component_probabilities(builder):
    priors = builder(
        period=4,
        level_dynamic_probability=0.6,
        trend_probabilities=(0.1, 0.4, 0.5),
        season_probabilities=(0.0, 0.25, 0.75),
    )
    assert priors.ssvs.level_dynamic_probability == pytest.approx(0.6)
    assert tuple(priors.ssvs.trend_probabilities) == (0.1, 0.4, 0.5)
    assert tuple(priors.ssvs.season_probabilities) == (0.0, 0.25, 0.75)


def test_ssvs_prior_builder_rejects_object_and_direct_settings_together():
    with pytest.raises(ValueError, match="either ssvs=SSVSPrior"):
        bx.ssvs_gaussian_priors(
            ssvs=bx.SSVSPrior(),
            season_probabilities=(0.0, 0.5, 0.5),
        )


def test_ssvs_gev_prior_builder_exposes_observation_and_initial_hyperparameters():
    priors = bx.ssvs_gev_priors(
        period=4,
        alpha_mean=25.0,
        alpha_sd=1.7,
        beta_sd=0.02,
        seasonal_initial_sd=0.8,
        sigma2_prior=bx.InverseGammaPrior(3.0, 4.0),
        xi_prior=bx.UniformPrior(-0.4, 0.2),
        xi_max_abs=0.45,
    )
    assert priors.alpha0.sd == 1.7
    assert priors.beta0.sd == 0.02
    assert priors.sigma2 == bx.InverseGammaPrior(3.0, 4.0)
    assert priors.xi == bx.UniformPrior(-0.4, 0.2)
    assert priors.xi_max_abs == 0.45
    assert priors.gamma0_season.sd_array().tolist() == [0.8, 0.8, 0.8]


def test_uccle_series_normalization_does_not_split_a_string():
    assert _normalize_series_names("TXm") == ("TXm",)
    with pytest.raises(ValueError, match="Unknown Uccle series"):
        _normalize_series_names("T")
