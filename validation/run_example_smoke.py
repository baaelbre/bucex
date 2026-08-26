#!/usr/bin/env python3
"""Direct-API smoke validation for the bucex 1.5.1 examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def run(work_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    scripts = sorted((SOURCE_ROOT / "examples").glob("[0-9][0-9]_*.py"))
    if len(scripts) != 13:
        raise RuntimeError(f"Expected 13 examples, found {len(scripts)}.")
    for script in scripts:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")

    model = bx.Model(
        bx.GEV(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
        name="smoke model",
    )
    simulation = bx.simulate(
        model,
        24,
        {"sigma": 1.0, "xi": -0.2, "sd.level": 0.04, "sd.slope": 0.001, "sd.seasonal": 0.03},
        initial_state=[20.0, 0.0, -0.2, 0.0, 0.2],
        seed=2620,
    )
    priors = bx.ssvs_gev_priors(period=4, alpha_mean=20.0)
    laplace = bx.fit(
        simulation.y,
        model=model,
        priors=priors,
        engine="laplace",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=2621),
    )
    pgas = bx.fit(
        simulation.y,
        model=model,
        priors=laplace.priors,
        engine="pgas",
        parameterization="fruehwirth_schnatter",
        init=laplace,
        particles=bx.Particles(n=16, proposal="guided"),
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=2622),
    )
    laplace_mh = bx.fit(
        simulation.y,
        model=model,
        priors=priors,
        engine="laplace_mh",
        parameterization="fruehwirth_schnatter",
        laplace=bx.Laplace(max_iterations=12, mh_steps=2),
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=2627),
    )
    if laplace.plan.targets_exact_posterior:
        raise RuntimeError("The Laplace fit was incorrectly marked exact.")
    if not pgas.plan.targets_exact_posterior:
        raise RuntimeError("The PGAS fit was not marked exact-invariant.")
    if not laplace_mh.plan.targets_exact_posterior:
        raise RuntimeError("The Laplace-MH fit was not marked exact-invariant.")
    if pgas.meta.get("warm_start_source_engine") != "laplace":
        raise RuntimeError("PGAS did not retain Laplace warm-start provenance.")

    centered_model = bx.Model(
        bx.GEV(),
        (bx.LocalLevel(mode="dynamic", initial_mean=20.0, initial_sd=3.0),),
        name="centered inverse-gamma smoke model",
    )
    centered_simulation = bx.simulate(
        centered_model,
        16,
        {"sigma": 1.0, "xi": -0.2, "sd.level": 0.04},
        initial_state=[20.0],
        seed=2625,
    )
    centered_priors = bx.Priors(
        process={"level": bx.InverseGammaVariance(shape=2.0, scale=0.0016)},
        observation_sd=bx.InverseGammaVariance(shape=2.0, scale=1.0),
        shape=bx.TruncatedNormalPrior(
            mean=0.0,
            sd=0.2,
            lower=-0.5,
            upper=0.5,
        ),
    )
    centered_ig = bx.fit(
        centered_simulation.y,
        model=centered_model,
        priors=centered_priors,
        engine="laplace",
        parameterization="centered",
        asis=False,
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=2626),
    )
    if centered_ig.plan.parameterization != "centered" or centered_ig.plan.asis:
        raise RuntimeError("The centered/inverse-gamma benchmark plan changed.")
    if centered_ig.plan.targets_exact_posterior:
        raise RuntimeError("The centered/inverse-gamma Laplace fit was marked exact.")
    if centered_ig.methods["parameter_updates"]["sd.level"] != (
        "inverse_gamma_gibbs"
    ):
        raise RuntimeError("The centered process variance did not use Gibbs.")

    work_dir.mkdir(parents=True, exist_ok=True)
    laplace.save(work_dir / "laplace.bucex")
    pgas.save(work_dir / "pgas.bucex")
    laplace_mh.save(work_dir / "laplace_mh.bucex")
    centered_ig.save(work_dir / "centered_ig.bucex")
    figure, axis = pgas.plot("season", labels=("1", "2", "3", "4"))
    figure.savefig(work_dir / "season.png", dpi=72)
    plt.close(figure)
    level_figure, _ = pgas.plot("level")
    level_figure.savefig(work_dir / "level.png", dpi=72)
    plt.close(level_figure)
    clean_level_figure, _ = pgas.plot("level", show_observed=False)
    clean_level_figure.savefig(work_dir / "level_no_observations.png", dpi=72)
    plt.close(clean_level_figure)
    slope_figure, slope_axis = pgas.plot("slope")
    slope_figure.savefig(work_dir / "slope.png", dpi=72)
    plt.close(slope_figure)
    predictive = pgas.posterior_predictive(draws=2, seed=2623)
    predictive_axis = predictive.plot(observed=pgas.observed)
    predictive_axis.figure.savefig(work_dir / "posterior_predictive.png", dpi=72)
    plt.close(predictive_axis.figure)
    forecast = pgas.forecast(8, draws=2, seed=2624)
    forecast_axis = forecast.plot(
        history=pgas.observed,
        history_dates=np.arange(pgas.n_time),
        history_points=12,
    )
    forecast_axis.figure.savefig(work_dir / "forecast.png", dpi=72)
    plt.close(forecast_axis.figure)

    x = np.arange(41, dtype=float)
    y = 2.0 + 0.2 * x
    y[20] += 10.0
    loess = bx.loess_smooth(x, y, fraction=0.3)
    if not np.all(np.isfinite(loess)):
        raise RuntimeError("LOESS returned non-finite values.")

    return {
        "bucex_version": bx.__version__,
        "work_dir": str(work_dir),
        "examples": [script.name for script in scripts],
        "warm_start_source": pgas.meta["warm_start_source_engine"],
        "pgas_engine_diagnostics": pgas.diagnostics()["engine"],
        "laplace_mh_engine_diagnostics": laplace_mh.diagnostics()["engine"],
        "centered_ig_plan": centered_ig.plan.to_dict(),
        "centered_ig_update_methods": centered_ig.methods[
            "parameter_updates"
        ],
        "centered_ig_engine_diagnostics": centered_ig.diagnostics()["engine"],
        "season_lines": len(axis.lines),
        "slope_ylabel": slope_axis.get_ylabel(),
        "loess_max_error_without_outlier": float(np.max(np.abs(np.delete(loess - (2.0 + 0.2 * x), 20)))),
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("validation/example_smoke_artifacts_1.5.1"))
    parser.add_argument("--output", type=Path, default=Path("validation/example_smoke_1.5.1.json"))
    args = parser.parse_args()
    result = run(args.work_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
