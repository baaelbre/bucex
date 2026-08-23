#!/usr/bin/env python3
"""Fixed-seed numerical release validation for bucex 1.1.4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx
from bucex.inference.fit.fs_utils import (
    _ncp_exact_observation_loglik,
    build_ncp_laplace_approximation,
    build_ncp_system,
    infer_ncp_layout,
    ncp_laplace_mh,
)


def _dates(n_time: int) -> np.ndarray:
    return np.arange(
        np.datetime64("2000-01"),
        np.datetime64("2000-01") + np.timedelta64(n_time, "M"),
        dtype="datetime64[M]",
    )


def _model(*, mixed: bool = False) -> bx.MultiSeriesModel:
    return bx.MultiSeriesModel(
        channels=(
            bx.Channel(
                "mean",
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
            bx.Channel(
                "maximum",
                bx.GEV() if mixed else bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
        ),
        name="release validation",
    )


def _finite_fit(fit: bx.FitResult) -> dict[str, object]:
    return {
        "backend": fit.plan.backend,
        "engine": fit.plan.engine,
        "finite_states": bool(np.all(np.isfinite(fit.state_draws))),
        "finite_log_posterior": bool(np.all(np.isfinite(fit.log_posterior))),
        "restored_iterations": int(fit.meta.get("restored_iterations", 0)),
        "initial_parameters": sorted(
            name for name in fit.parameter_draws if name.startswith("initial.")
        ),
    }


def run() -> dict[str, object]:
    started = time.perf_counter()
    if bx.__version__ != "1.1.4":
        raise RuntimeError(f"Expected bucex 1.1.4, found {bx.__version__}.")
    default_hierarchy = bx.HierarchicalPrior()
    if default_hierarchy.model_space != "componentwise":
        raise RuntimeError("The hierarchy must default to componentwise SSVS.")
    rng = np.random.default_rng(2500)
    record: dict[str, object] = {
        "bucex_version": bx.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    mapping_command = r'''
source config/laplace_mh_simulation_array.sh
for task in $(seq 1 24); do
  laplace_mh_map_task "$task" 4 "$LMH_DEFAULT_SCENARIO_KEYS"
  printf '%s,%s\n' "$LMH_TASK_SCENARIO" "$LMH_TASK_CHAIN"
done
'''
    mapping_result = subprocess.run(
        ["bash", "-c", mapping_command],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    mapping_pairs = [
        tuple(line.split(","))
        for line in mapping_result.stdout.splitlines()
        if line
    ]
    fit_pbs = (
        SOURCE_ROOT
        / "job_scripts"
        / "submit_08_simulation_laplace_mh_array.pbs"
    ).read_text(encoding="utf-8")
    final_pbs = (
        SOURCE_ROOT
        / "job_scripts"
        / "submit_08_simulation_laplace_mh_finalize.pbs"
    ).read_text(encoding="utf-8")
    record["laplace_mh_hpc_fanout"] = {
        "task_count": len(mapping_pairs),
        "unique_task_count": len(set(mapping_pairs)),
        "first_task": list(mapping_pairs[0]),
        "last_task": list(mapping_pairs[-1]),
        "fit_walltime_six_hours": "#PBS -l walltime=06:00:00" in fit_pbs,
        "one_core_per_fit": "#PBS -l nodes=1:ppn=1" in fit_pbs,
        "one_hour_finalizer": "#PBS -l walltime=01:00:00" in final_pbs,
        "submission_helper_present": (
            SOURCE_ROOT
            / "bash_scripts"
            / "qsub_08_simulation_laplace_mh.sh"
        ).is_file(),
    }
    if len(mapping_pairs) != 24 or len(set(mapping_pairs)) != 24:
        raise RuntimeError("Example-08 PBS fan-out must contain 24 unique tasks.")
    if mapping_pairs[0] != ("stationary", "1") or mapping_pairs[-1] != (
        "llt_season",
        "4",
    ):
        raise RuntimeError("Example-08 PBS task ordering changed unexpectedly.")

    # Regression for the 1.1.0--1.1.2 failure: the zero FS trajectory crosses
    # a negative-shape endpoint, while a valid trajectory exists through the
    # one-step-delayed integrated slope. Proposal construction must repair the
    # initializer deterministically without using that current trajectory.
    support_model = bx.Model(bx.GEV(), [bx.LocalLinearTrend()])
    support_layout = infer_ncp_layout(support_model)
    support_y = np.asarray([0.0, 5.0, 5.0, 5.0, 5.0])
    support_state = {
        "alpha0": 0.0,
        "beta0": 0.0,
        "s_level": 0.0,
        "q_level": 0.0,
        "s_trend": 1.0,
        "q_trend": 1.0,
    }
    support_observation = {"sigma": 1.0, "xi": -0.25}
    support_transition, support_covariance = build_ncp_system(support_layout)
    support_current = np.zeros(
        (support_y.size + 1, support_layout.ncp_state_dim)
    )
    for state_time in range(1, support_y.size + 1):
        support_current[state_time] = (
            support_transition @ support_current[state_time - 1]
        )
        if state_time == 1:
            support_current[
                state_time, int(support_layout.idx_tilde_beta)
            ] += 5.0
        elif state_time == 2:
            support_current[
                state_time, int(support_layout.idx_tilde_beta)
            ] -= 5.0
    zero_support_loglik = _ncp_exact_observation_loglik(
        support_y,
        np.zeros_like(support_current),
        support_state,
        support_observation,
        support_model,
        support_layout,
    )
    current_support_loglik = _ncp_exact_observation_loglik(
        support_y,
        support_current,
        support_state,
        support_observation,
        support_model,
        support_layout,
    )
    support_approximation = build_ncp_laplace_approximation(
        support_y,
        support_model,
        support_state,
        support_observation,
        support_layout,
        max_iterations=15,
    )
    support_repeat = build_ncp_laplace_approximation(
        support_y,
        support_model,
        support_state,
        support_observation,
        support_layout,
        max_iterations=15,
    )
    support_result = ncp_laplace_mh(
        support_y,
        support_model,
        support_state,
        support_observation,
        support_layout,
        support_current,
        rng=np.random.default_rng(2499),
        mh_steps=2,
        max_iterations=15,
    )
    support_residual = (
        support_approximation.mode_path[1:]
        - support_approximation.mode_path[:-1] @ support_transition.T
    )
    deterministic = np.flatnonzero(np.diag(support_covariance) == 0.0)
    record["negative_xi_fs_support_repair"] = {
        "zero_path_invalid": bool(not np.isfinite(zero_support_loglik)),
        "current_path_valid": bool(np.isfinite(current_support_loglik)),
        "initializer_repaired": bool(
            support_approximation.initial_support_repaired
        ),
        "initializer_deterministic": bool(
            np.array_equal(
                support_approximation.mode_path,
                support_repeat.mode_path,
            )
        ),
        "mode_converged": bool(support_approximation.converged),
        "mode_objective_finite": bool(
            np.isfinite(support_approximation.objective)
        ),
        "singular_recursion_exact": bool(
            np.array_equal(
                support_residual[:, deterministic],
                np.zeros_like(support_residual[:, deterministic]),
            )
        ),
        "laplace_mh_exact_invariant": bool(support_result.exact_invariant),
        "finite_returned_path": bool(
            np.all(np.isfinite(support_result.z_path))
        ),
    }

    # Long, nearly deterministic Gaussian path: regression for stable Joseph
    # covariance handling and estimated initial level/slope.
    n_long = 1000
    time_index = np.arange(n_long)
    long_values = (
        0.002 * time_index
        + 0.3 * np.sin(2.0 * np.pi * time_index / 12.0)
        + rng.normal(scale=0.4, size=n_long)
    )
    long_fit = bx.fit(
        long_values,
        family="gaussian",
        period=12,
        priors="normal",
        parameterization="fs",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2401),
    )
    record["long_gaussian"] = {
        **_finite_fit(long_fit),
        "sign_invariance_max_error": float(
            np.nanmax(long_fit.draws_aux["sign_invariance_error"])
        ),
        "initial_level_finite": bool(
            np.all(np.isfinite(long_fit.parameter("alpha0")))
        ),
        "initial_slope_finite": bool(
            np.all(np.isfinite(long_fit.parameter("beta0")))
        ),
    }

    n_time = 24
    season = 0.15 * np.sin(2.0 * np.pi * np.arange(n_time) / 4.0)
    gaussian_values = np.column_stack(
        (
            0.02 * np.arange(n_time) + season + rng.normal(scale=0.15, size=n_time),
            0.01 * np.arange(n_time) + 0.8 * season + rng.normal(scale=0.2, size=n_time),
        )
    )
    pooling = {}
    for index, mode in enumerate(("selection", "slab", "both")):
        fit = bx.fit(
            gaussian_values,
            _model(),
            priors=bx.HierarchicalPrior(pool=mode),
            mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=2410 + index),
            dates=_dates(n_time),
        )
        probabilities = fit.component_probabilities()
        pooling[mode] = {
            **_finite_fit(fit),
            "sign_invariance_max_error": float(
                np.nanmax(fit.draws_aux["sign_invariance_error"])
            ),
            "season_zero_probability": probabilities.xs(
                "seasonal", level="process"
            )["zero"].tolist(),
            "hierarchy_probability_rows": int(
                fit.hierarchical_probabilities().shape[0]
            ),
            "hierarchy_slab_rows": int(fit.hierarchical_slab_summary().shape[0]),
            "model_space": fit.meta["trend_model_space"],
            "joint_structure_rows": int(fit.structural_model_probabilities().shape[0]),
        }
    record["hierarchical_gaussian"] = pooling

    joint_prior = bx.HierarchicalPrior(
        pool="selection",
        model_space="joint_trend",
        trend_states=("fixed", "dynamic"),
        trend_concentration=(1.0, 1.0),
    )
    joint_fit = bx.fit(
        gaussian_values,
        _model(),
        priors=joint_prior,
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=2519),
        dates=_dates(n_time),
    )
    trend_models = joint_fit.hierarchical_trend_model_probabilities()
    record["optional_joint_trend_space"] = {
        **_finite_fit(joint_fit),
        "rows": int(trend_models.shape[0]),
        "probability_sum": float(trend_models["mean"].sum()),
    }

    mixed_values = np.column_stack(
        (
            rng.normal(scale=0.2, size=16),
            rng.gumbel(scale=0.3, size=16),
        )
    )
    laplace_screen = bx.fit(
        mixed_values,
        _model(mixed=True),
        priors=bx.HierarchicalPrior(pool="selection"),
        engine="laplace",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2420),
        dates=_dates(16),
    )
    record["mixed_hierarchical_laplace"] = {
        **_finite_fit(laplace_screen),
        "targets_exact_posterior": bool(
            laplace_screen.plan.targets_exact_posterior
        ),
        "warm_start_keys": sorted(laplace_screen.warm_start()),
    }

    mixed_fit = bx.fit(
        mixed_values,
        _model(mixed=True),
        priors=bx.HierarchicalPrior(pool="selection"),
        engine="pgas",
        init=laplace_screen,
        particles=bx.Particles(n=20, proposal="guided"),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2421),
        dates=_dates(16),
    )
    record["mixed_guided_pgas"] = {
        **_finite_fit(mixed_fit),
        "targets_exact_posterior": bool(mixed_fit.plan.targets_exact_posterior),
        "external_warm_start": bool(mixed_fit.meta["external_warm_start"]),
        "path_update_fraction_finite": bool(
            np.all(
                np.isfinite(
                    mixed_fit.draws_aux["particle_path_update_fraction"]
                )
            )
        ),
    }

    disturbance_fit = bx.fit(
        rng.gumbel(size=18),
        family="gev",
        period=4,
        priors="normal",
        engine="pgas",
        parameterization="disturbance",
        asis=False,
        particles=bx.Particles(n=20, proposal="guided"),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2422),
    )
    record["disturbance_guided_pgas"] = _finite_fit(disturbance_fit)

    laplace_mh_fit = bx.fit(
        rng.gumbel(size=18),
        family="gev",
        period=4,
        priors="normal",
        engine="laplace_mh",
        parameterization="fs",
        laplace=bx.Laplace(max_iterations=12, mh_steps=2),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2423),
    )
    laplace_mh_diagnostics = laplace_mh_fit.diagnostics()["engine"]
    record["fs_laplace_mh"] = {
        **_finite_fit(laplace_mh_fit),
        "targets_exact_posterior": bool(
            laplace_mh_fit.plan.targets_exact_posterior
        ),
        "state_acceptance": float(
            laplace_mh_diagnostics["state_acceptance"]
        ),
        "support_rejections": float(
            laplace_mh_diagnostics["mean_proposal_support_rejections"]
        ),
    }

    centered_ig_model = bx.Model(
        bx.GEV(xi_bounds=(-0.5, 0.5)),
        [bx.LocalLevel(mode="dynamic", initial_mean=20.0, initial_sd=2.0)],
    )
    centered_ig_simulation = bx.simulate(
        centered_ig_model,
        24,
        {"sd.level": 0.05, "sigma": 1.0, "xi": -0.2},
        initial_state=[20.0],
        seed=2424,
    )
    centered_ig_fit = bx.fit(
        centered_ig_simulation.y,
        model=centered_ig_model,
        priors=bx.Priors(
            process={
                "level": bx.InverseGammaVariance(
                    shape=2.0, scale=0.0025
                )
            },
            observation_sd=bx.InverseGammaVariance(
                shape=2.0, scale=1.0
            ),
            shape=bx.TruncatedNormalPrior(
                mean=0.0, sd=0.2, lower=-0.5, upper=0.5
            ),
            profile="centered_ig_validation",
        ),
        engine="laplace",
        parameterization="centered",
        asis=False,
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=2425),
    )
    record["centered_inverse_gamma_gibbs"] = {
        **_finite_fit(centered_ig_fit),
        "update_method": centered_ig_fit.sampler_diagnostics[
            "update_methods"
        ]["sd.level"],
        "has_mh_acceptance": "sd.level"
        in centered_ig_fit.sampler_diagnostics["acceptance"],
        "conjugate_updates": centered_ig_fit.meta[
            "conjugate_process_variance_updates"
        ],
    }
    record["total_seconds"] = time.perf_counter() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/release_validation_1.1.4.json"),
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
