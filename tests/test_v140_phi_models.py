from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import bucex as bx


ROOT = Path(__file__).resolve().parents[1]


def _sample() -> np.ndarray:
    model = bx.Model(
        bx.GEV(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
    )
    return bx.simulate(
        model,
        24,
        {
            "sd.level": 0.02,
            "sd.slope": 0.001,
            "sd.seasonal": 0.02,
            "sigma": 0.8,
            "xi": -0.1,
        },
        initial_state=np.zeros(5),
        seed=1400,
    ).y


def test_v140_public_phi_api_defaults_aliases_and_roundtrips():
    assert bx.__version__ == "1.5.2"
    assert bx.GEV().phi == "stationary"
    assert bx.GEV(phi="linear").phi == "linear"
    assert bx.GEV(phi="random_walk").phi == "rw"
    assert bx.GEV(phi="selection").phi == "ssvs"
    with pytest.raises(ValueError, match="stationary.*linear.*rw.*ssvs"):
        bx.GEV(phi="quadratic")

    model = bx.Model(bx.GEV(phi="rw"), (bx.LocalLinearTrend(),))
    restored = bx.Model.from_dict(model.to_dict())
    assert restored.observation.phi == "rw"
    readable_prior = bx.PhiPrior(
        model_probabilities={"stationary": 0.6, "linear": 0.3, "rw": 0.1}
    )
    assert readable_prior.model_probabilities == (0.6, 0.3, 0.1)


@pytest.mark.parametrize(
    "builder",
    [
        bx.manuscript_gev_priors,
        bx.normal_gev_priors,
        bx.regularized_gev_priors,
        bx.regularized_horseshoe_gev_priors,
        bx.triple_gamma_gev_priors,
        bx.regularized_triple_gamma_gev_priors,
        bx.pc_gev_priors,
        bx.ssvs_gev_priors,
    ],
)
def test_v140_every_gev_prior_profile_accepts_one_phi_prior(builder):
    phi_prior = bx.PhiPrior(
        linear=bx.NormalPrior(0.1, 0.2),
        rw_variance=bx.InverseGammaPrior(3.0, 0.001),
        model_probabilities=(0.2, 0.3, 0.5),
    )
    assert builder(period=4, phi_prior=phi_prior).phi is phi_prior


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_v140_log_scale_derivatives_match_finite_differences(xi: float):
    observation = bx.GEV()
    y = np.asarray([0.1, 0.4, -0.2])
    eta = np.asarray([0.4, 0.2, -0.1])
    phi = np.log(np.asarray([1.3, 0.9, 1.1]))
    step = 1e-5

    def objective(value: np.ndarray) -> np.ndarray:
        return np.asarray(
            observation.logpdf(y, eta, sigma=np.exp(value), xi=xi),
            dtype=float,
        )

    gradient = (objective(phi + step) - objective(phi - step)) / (2.0 * step)
    hessian = (
        objective(phi + step) - 2.0 * objective(phi) + objective(phi - step)
    ) / step**2
    np.testing.assert_allclose(
        observation.grad_phi(y, eta, sigma=np.exp(phi), xi=xi),
        gradient,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        observation.hess_phi(y, eta, sigma=np.exp(phi), xi=xi),
        hessian,
        atol=3e-3,
    )


@pytest.mark.parametrize("mode", ["stationary", "linear", "rw", "ssvs"])
def test_v140_every_phi_model_fits_predicts_and_has_clean_outputs(
    mode: str,
    tmp_path: Path,
):
    model = bx.Model(
        bx.GEV(phi=mode),
        (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
    )
    fit = bx.fit(
        _sample(),
        model=model,
        priors="ssvs",
        engine="laplace_mh",
        parameterization="fs",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=1410),
        laplace=bx.Laplace(max_iterations=12),
    )

    assert fit.plan.targets_exact_posterior
    assert fit.meta["phi"] == mode
    assert fit.meta["phi_exact_invariant"]
    assert fit.phi_draws().shape == (2, 24)
    assert fit.sigma_draws().shape == (2, 24)
    assert np.all(np.isfinite(fit.phi_draws()))
    assert np.all(fit.sigma_draws() > 0.0)
    assert fit.return_level_draws(20).shape == (2, 24)
    exceedance, labels = fit.exceedance_probability_draws(1.5)
    assert exceedance.shape == (2, 24)
    assert labels.shape == (24,)

    if mode == "stationary":
        assert "phi" not in fit.parameter_draws
    elif mode == "linear":
        assert {"phi", "sigma_path", "phi_intercept", "phi_slope"} <= set(
            fit.parameter_draws
        )
        assert "phi_rw_sd" not in fit.parameter_draws
        assert "phi_model" not in fit.parameter_draws
    elif mode == "rw":
        assert {
            "phi",
            "sigma_path",
            "phi_intercept",
            "phi_last",
            "phi_rw_sd",
            "phi_rw_variance",
        } <= set(fit.parameter_draws)
        assert "phi_slope" not in fit.parameter_draws
        assert "phi_model" not in fit.parameter_draws
    else:
        probabilities = fit.phi_model_probabilities()
        assert set(probabilities) == {"stationary", "linear", "rw"}
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert {"phi_stationary", "phi_linear", "phi_rw", "phi_model"} <= set(
            fit.parameter_draws
        )

    predictive = fit.posterior_predictive(draws=2, seed=1411)
    forecast = fit.forecast(3, draws=2, seed=1412)
    for result, length in ((predictive, 24), (forecast, 3)):
        assert result.parameters["phi"].shape == (2, length)
        assert result.parameters["sigma_path"].shape == (2, length)
        assert np.all(result.parameters["sigma_path"] > 0.0)

    archive = tmp_path / f"{mode}.bucex"
    fit.save(archive)
    restored = bx.FitResult.load(archive)
    assert restored.model.observation.phi == mode
    np.testing.assert_allclose(restored.phi_draws(), fit.phi_draws())


def test_v140_json_examples_are_complete_and_hpc_parallelizable():
    simulation_dir = ROOT / "examples" / "config" / "phi"
    simulation_files = sorted(simulation_dir.glob("simulation_*.json"))
    uccle_files = sorted((simulation_dir / "uccle").glob("*.json"))
    assert [path.stem for path in simulation_files] == [
        "simulation_linear",
        "simulation_rw",
        "simulation_ssvs",
        "simulation_stationary",
    ]
    assert len(uccle_files) == 16

    for path in (*simulation_files, *uccle_files):
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["schema_version"] == 2
        assert config["_comment"]
        assert config["model"]["phi"] == path.stem.rsplit("_", 1)[-1]
        assert config["model"]["location"] == {
            "_comment": [
                "The fitted location uses structural SSVS.",
                "These dynamic components form the full candidate envelope; priors below select zero/fixed/dynamic level, trend, and season states.",
            ],
            "structure": "ssvs",
            "trend_component": "local_linear_trend",
            "level_mode": "dynamic",
            "trend_mode": "dynamic",
            "seasonal_component": "dummy",
            "seasonal_mode": "dynamic",
        }
        assert config["mcmc"]["draws"] == 1000
        assert config["mcmc"]["warmup"] == 1000
        assert config["mcmc"]["chains"] == 4
        assert set(config["priors"]["phi"]) - {"_comment"} == {
            "linear",
            "rw_variance",
            "model_probabilities",
        }
        assert set(config["inference"]["phi"]) - {"_comment"} == {
            "step_intercept",
            "step_linear",
            "laplace_max_iterations",
            "laplace_tolerance",
            "curvature_floor",
            "maximum_variance",
            "shift_limit",
            "mh_steps",
            "draw_attempts",
        }
        assert config["figures"]["formats"] == ["pdf", "png"]
        assert config["figures"]["interval_probability"] == pytest.approx(0.9)

    for path in simulation_files:
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["simulation"]["location"]["trend_component"] == (
            "local_linear_trend"
        )
        assert config["simulation"]["location"]["level_mode"] == "dynamic"
        assert config["simulation"]["location"]["trend_mode"] == "dynamic"
        assert config["simulation"]["location"]["seasonal_mode"] == "dynamic"

    runner = (ROOT / "job_scripts" / "run_parallel_chains.py").read_text(
        encoding="utf-8"
    )
    assert '"10"' in runner and '"11"' in runner
    assert (ROOT / "bash_scripts" / "run_10_simulation_phi.sh").is_file()
    assert (ROOT / "bash_scripts" / "run_11_uccle_phi.sh").is_file()
    assert (ROOT / "job_scripts" / "submit_10_simulation_phi.pbs").is_file()
    assert (ROOT / "job_scripts" / "submit_11_uccle_phi.pbs").is_file()
