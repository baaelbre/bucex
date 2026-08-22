"""Exact Laplace-MH fits for representative dynamic-GEV simulations.

Run with ``python examples/08_simulation_laplace_mh.py``. Environment variables
prefixed by ``BUCEX_`` control the record length and MCMC effort; the defaults
are suitable for a substantive run, while ``BUCEX_DRAWS=20 BUCEX_WARMUP=20``
is convenient for a smoke test.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
N_TIME = int(os.environ.get("BUCEX_N_TIME", "240"))
PERIOD = 4
DRAWS = int(os.environ.get("BUCEX_DRAWS", "400"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "400"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "2"))
SEED = int(os.environ.get("BUCEX_SEED", "11001"))
MH_STEPS = int(os.environ.get("BUCEX_LAPLACE_MH_STEPS", "1"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {
    "0",
    "false",
    "no",
}
RUN_SIGNATURE = f"n{N_TIME}p{PERIOD}_d{DRAWS}w{WARMUP}c{CHAINS}m{MH_STEPS}"
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


FIT_MODEL = bx.Model(
    bx.GEV(xi_bounds=(-0.50, 0.20)),
    [bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)],
    name="candidate dynamic GEV",
)

season = np.asarray([-0.30, 0.10, 0.25, -0.05])
season -= season.mean()
SCENARIOS = (
    {
        "name": "random_walk",
        "model": bx.Model(
            bx.GEV(),
            [bx.LocalLinearTrend(level_mode="dynamic", trend_mode="off")],
        ),
        "parameters": {"sd.level": 0.035, "sigma": 1.0, "xi": -0.20},
        "initial_state": np.asarray([25.0]),
    },
    {
        "name": "local_linear_trend",
        "model": bx.Model(bx.GEV(), [bx.LocalLinearTrend()]),
        "parameters": {
            "sd.level": 0.015,
            "sd.slope": 0.0005,
            "sigma": 1.0,
            "xi": -0.20,
        },
        "initial_state": np.asarray([25.0, 0.006]),
    },
    {
        "name": "evolving_season",
        "model": bx.Model(
            bx.GEV(),
            [bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)],
        ),
        "parameters": {
            "sd.level": 0.015,
            "sd.slope": 0.0005,
            "sd.seasonal": 0.025,
            "sigma": 1.0,
            "xi": -0.20,
        },
        "initial_state": np.r_[25.0, 0.006, season[1:]],
    },
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    run_config = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "bucex_version": bx.__version__,
        "engine": "laplace_mh",
        "target": "exact posterior",
        "n_time": N_TIME,
        "period": PERIOD,
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "seed": SEED,
        },
        "laplace": {"mh_steps": MH_STEPS},
        "scenarios": [scenario["name"] for scenario in SCENARIOS],
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    summary_rows = []
    for index, scenario in enumerate(SCENARIOS):
        simulation = bx.simulate(
            scenario["model"],
            N_TIME,
            scenario["parameters"],
            initial_state=scenario["initial_state"],
            seed=SEED + index,
        )
        fit = bx.fit(
            simulation.y,
            model=FIT_MODEL,
            priors="ssvs",
            engine="laplace_mh",
            parameterization="fs",
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED + 100 + index,
                progress=PROGRESS,
            ),
            laplace=bx.Laplace(mh_steps=MH_STEPS),
            name=scenario["name"],
        )
        scenario_dir = OUTPUT_DIR / scenario["name"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        fit.save(scenario_dir / "fit.bucex")

        diagnostics = fit.diagnostics()
        diagnostics["parameters"].to_csv(scenario_dir / "diagnostics.csv")
        fit.component_probabilities().to_csv(
            scenario_dir / "component_probabilities.csv"
        )
        lower, median, upper = np.quantile(
            fit.eta_draws(original_scale=True), [0.05, 0.50, 0.95], axis=0
        )
        pd.DataFrame(
            {
                "time": np.arange(N_TIME),
                "observed": simulation.y,
                "truth": simulation.eta,
                "lower": lower,
                "median": median,
                "upper": upper,
            }
        ).to_csv(scenario_dir / "trajectory.csv", index=False)

        engine = diagnostics["engine"]
        summary_rows.append({"scenario": scenario["name"], **engine})
        (scenario_dir / "algorithm.json").write_text(
            json.dumps(engine, indent=2, default=_json_value), encoding="utf-8"
        )

        figure, axis = fit.plot("predictor", credible_interval=0.90)
        axis.plot(
            np.arange(N_TIME),
            simulation.eta,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="truth",
        )
        axis.legend()
        axis.set_title(f"{scenario['name']}: exact Laplace-MH predictor")
        _save_figure(figure, scenario_dir / "predictor.png")

        figure, axis = fit.plot("level", credible_interval=0.90)
        axis.set_title(f"{scenario['name']}: latent level")
        _save_figure(figure, scenario_dir / "level.png")

        figure, axis = fit.plot("slope", credible_interval=0.90, scale="interval")
        axis.set_title(f"{scenario['name']}: latent slope")
        _save_figure(figure, scenario_dir / "slope.png")

        figure, axis = fit.plot("season", show_interval=False)
        axis.set_title(f"{scenario['name']}: seasonal component")
        _save_figure(figure, scenario_dir / "season.png")

        figure, axis = fit.plot("component_probabilities")
        axis.set_title(f"{scenario['name']}: structural probabilities")
        _save_figure(figure, scenario_dir / "selection.png")

        print(
            scenario["name"],
            f"state acceptance={engine['state_acceptance']:.3f}",
            f"support rejections={engine['mean_proposal_support_rejections']:.3f}",
        )

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "laplace_mh_diagnostics.csv", index=False
    )
    print(f"Simulation Laplace-MH outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
