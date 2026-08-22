"""Exact Laplace-MH analysis of the four Uccle monthly temperature extremes.

The lower-tail series are transformed internally by :func:`fit_uccle_series`.
Use ``BUCEX_UCCLE_SERIES=TXx`` for a quick single-series run.
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
DATA_DIR = (
    Path(os.environ["BUCEX_DATA_DIR"])
    if "BUCEX_DATA_DIR" in os.environ
    else None
)
START = os.environ.get("BUCEX_START", "1892-01-01")
END = os.environ.get("BUCEX_END") or None
SERIES = tuple(
    value.strip()
    for value in os.environ.get("BUCEX_UCCLE_SERIES", "TXx,TXn,TNx,TNn").split(",")
    if value.strip()
)
DRAWS = int(os.environ.get("BUCEX_DRAWS", "400"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "400"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "2"))
SEED = int(os.environ.get("BUCEX_SEED", "11010"))
MH_STEPS = int(os.environ.get("BUCEX_LAPLACE_MH_STEPS", "1"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {
    "0",
    "false",
    "no",
}
RUN_SIGNATURE = (
    f"y{START[:4]}-{(END or 'latest')[:4]}"
    f"_d{DRAWS}w{WARMUP}c{CHAINS}m{MH_STEPS}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    unknown = sorted(set(SERIES) - {"TXx", "TXn", "TNx", "TNn"})
    if unknown:
        raise ValueError(f"Laplace-MH example expects GEV series; unknown: {unknown}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    run_config = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "bucex_version": bx.__version__,
        "engine": "laplace_mh",
        "target": "exact posterior",
        "data": {
            "data_dir": None if DATA_DIR is None else str(DATA_DIR),
            "start": START,
            "end": END,
            "series": list(SERIES),
        },
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "seed": SEED,
        },
        "laplace": {"mh_steps": MH_STEPS},
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    algorithm_rows = []
    selection_rows = []
    for index, name in enumerate(SERIES):
        values = bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        fit = bx.fit_uccle_series(
            name,
            DATA_DIR,
            priors="ssvs",
            engine="laplace_mh",
            parameterization="fs",
            asis=False,
            start=START,
            end=END,
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED + index,
                progress=PROGRESS,
            ),
            laplace=bx.Laplace(mh_steps=MH_STEPS),
        )
        series_dir = OUTPUT_DIR / name
        series_dir.mkdir(parents=True, exist_ok=True)
        fit.save(series_dir / "fit.bucex")

        diagnostics = fit.diagnostics()
        diagnostics["parameters"].to_csv(series_dir / "diagnostics.csv")
        pd.DataFrame.from_dict(fit.static_summary(), orient="index").to_csv(
            series_dir / "parameters.csv"
        )
        selection = fit.component_probabilities().reset_index()
        selection.insert(0, "series", name)
        selection_rows.append(selection)
        selection.to_csv(series_dir / "component_probabilities.csv", index=False)

        lower, median, upper = np.quantile(
            fit.eta_draws(original_scale=True), [0.05, 0.50, 0.95], axis=0
        )
        pd.DataFrame(
            {
                "date": values.index,
                "observed": values.to_numpy(),
                "lower": lower,
                "median": median,
                "upper": upper,
            }
        ).to_csv(series_dir / "trajectory.csv", index=False)

        engine = diagnostics["engine"]
        algorithm_rows.append({"series": name, **engine})
        (series_dir / "algorithm.json").write_text(
            json.dumps(engine, indent=2), encoding="utf-8"
        )

        figure, axis = fit.plot("predictor", credible_interval=0.90)
        axis.set_title(f"{name}: monthly GEV location (exact Laplace-MH)")
        axis.set_ylabel("temperature / °C")
        _save_figure(figure, series_dir / "predictor.png")

        figure, axis = fit.plot("level", credible_interval=0.90)
        axis.set_title(f"{name}: latent climate level")
        axis.set_ylabel("temperature / °C")
        _save_figure(figure, series_dir / "level.png")

        figure, axis = fit.plot(
            "slope", credible_interval=0.90, scale="decade", unit="°C / decade"
        )
        axis.set_title(f"{name}: latent warming rate")
        _save_figure(figure, series_dir / "slope.png")

        figure, axis = fit.plot("season", show_interval=False)
        axis.set_title(f"{name}: seasonal component")
        _save_figure(figure, series_dir / "season.png")

        figure, axis = fit.plot("component_probabilities")
        axis.set_title(f"{name}: structural probabilities")
        _save_figure(figure, series_dir / "selection.png")

        print(
            name,
            f"state acceptance={engine['state_acceptance']:.3f}",
            f"support rejections={engine['mean_proposal_support_rejections']:.3f}",
        )
        if engine["state_acceptance"] < 0.10:
            print(
                f"warning: {name} has low whole-trajectory acceptance; "
                "compare PGAS or shorten the block before substantive use."
            )

    pd.DataFrame(algorithm_rows).to_csv(
        OUTPUT_DIR / "laplace_mh_diagnostics.csv", index=False
    )
    pd.concat(selection_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "component_probabilities.csv", index=False
    )
    print(f"Uccle Laplace-MH outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
