"""Reduce the Laplace sensitivity grid to compact tables and figures.

The script reads only ``run_config.json`` and the CSV/JSON files below each
combined run. It never loads the large ``.bucex`` posterior files, so it is
safe to run on a login node after the combination jobs have finished.

Example
-------
python examples/10_laplace_sensitivity_summary.py \
    --results-root results/laplace_sensitivity \
    --output results/laplace_sensitivity/summary
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = (
    "stationary",
    "linear",
    "random_walk",
    "llt",
    "dynamic_season",
    "llt_season",
)
PROCESS_ORDER = ("level", "slope", "seasonal")
STATE_ORDER = ("zero", "fixed", "dynamic")
TRUTH_PARAMETERS = ("sigma", "xi", "sd.level", "sd.slope", "sd.seasonal")
RUN_PATTERN = re.compile(
    r"^s(?P<seed>[0-9]+)_(?P<profile>.+)_combined__(?P<signature>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/laplace_sensitivity"),
        help="Sensitivity root containing 03_simulation_laplace/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Compact output directory (default: RESULTS_ROOT/summary).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any combined run is incomplete.",
    )
    return parser.parse_args()


def profile_order(values: Iterable[str]) -> list[str]:
    unique = sorted(set(values))
    return sorted(unique, key=lambda value: (value != "base", value))


def discover_runs(fit_root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for run_dir in sorted(fit_root.glob("s*_combined__*")):
        if not run_dir.is_dir():
            continue
        match = RUN_PATTERN.match(run_dir.name)
        if match is None:
            continue
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            runs.append(
                {
                    "seed": int(match.group("seed")),
                    "profile": match.group("profile"),
                    "signature": match.group("signature"),
                    "run_dir": run_dir,
                    "config": None,
                    "config_error": "missing run_config.json",
                }
            )
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            error = ""
        except (OSError, json.JSONDecodeError) as exc:
            config = None
            error = f"{type(exc).__name__}: {exc}"
        runs.append(
            {
                "seed": int(match.group("seed")),
                "profile": match.group("profile"),
                "signature": match.group("signature"),
                "run_dir": run_dir,
                "config": config,
                "config_error": error,
            }
        )
    return runs


def scenario_keys(run: dict[str, object]) -> list[str]:
    config = run["config"]
    if isinstance(config, dict):
        keys = config.get("scenario_keys")
        if isinstance(keys, list) and keys:
            return [str(key) for key in keys]
        directories = config.get("scenario_directories")
        if isinstance(directories, dict) and directories:
            present = {str(value) for value in directories.values()}
            return [key for key in SCENARIO_ORDER if key in present]
    table_root = Path(run["run_dir"]) / "tables"
    present = {path.name for path in table_root.iterdir()} if table_root.is_dir() else set()
    return [key for key in SCENARIO_ORDER if key in present]


def prior_columns(config: dict[str, object] | None) -> dict[str, float]:
    if not isinstance(config, dict):
        return {
            "level_slab_sd": math.nan,
            "trend_slab_sd": math.nan,
            "season_slab_sd": math.nan,
            "initial_season_sd": math.nan,
        }
    priors = config.get("priors", {})
    if not isinstance(priors, dict):
        priors = {}
    slabs = priors.get("innovation_slab_sd", {})
    if not isinstance(slabs, dict):
        slabs = {}
    return {
        "level_slab_sd": float(slabs.get("level", math.nan)),
        "trend_slab_sd": float(slabs.get("trend", math.nan)),
        "season_slab_sd": float(slabs.get("season", math.nan)),
        "initial_season_sd": float(priors.get("seasonal_initial_sd", math.nan)),
    }


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return table


def run_metadata(run: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "seed": run["seed"],
        "profile": run["profile"],
        "signature": run["signature"],
        "run_directory": str(run["run_dir"]),
    }
    result.update(prior_columns(run["config"]))
    return result


def collect(runs: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    switching_frames: list[pd.DataFrame] = []
    trajectory_rows: list[dict[str, object]] = []
    parameter_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    algorithm_rows: list[dict[str, object]] = []

    for run in runs:
        metadata = run_metadata(run)
        run_dir = Path(run["run_dir"])
        config = run["config"]
        keys = scenario_keys(run)
        config_valid = isinstance(config, dict)

        for scenario in SCENARIO_ORDER:
            table_dir = run_dir / "tables" / scenario
            truth_path = run_dir / "simulations" / f"{scenario}.json"
            expected_files = {
                "selection": table_dir / "selection.csv",
                "switching": table_dir / "switching.csv",
                "trajectory": table_dir / "trajectory.csv",
                "parameters": table_dir / "parameters.csv",
                "diagnostics": table_dir / "diagnostics.csv",
                "algorithm": table_dir / "algorithm.csv",
                "truth": truth_path,
                "fit": run_dir / "fits" / scenario / "combined.bucex",
            }
            present = {name: path.is_file() for name, path in expected_files.items()}
            inventory_rows.append(
                {
                    **metadata,
                    "scenario": scenario,
                    "listed_in_config": scenario in keys,
                    "config_valid": config_valid,
                    "config_error": run["config_error"],
                    **{f"has_{name}": value for name, value in present.items()},
                    "complete": config_valid and all(present.values()),
                }
            )
            if not all(
                present[name]
                for name in (
                    "selection",
                    "switching",
                    "trajectory",
                    "parameters",
                    "diagnostics",
                    "algorithm",
                    "truth",
                )
            ):
                continue

            truth = json.loads(truth_path.read_text(encoding="utf-8"))

            selection = read_csv(
                expected_files["selection"],
                (*STATE_ORDER, "process", "truth_state", "probability_true_state"),
            )
            probabilities = selection.loc[:, STATE_ORDER].to_numpy(float)
            truth_index = selection["truth_state"].map(
                {state: index for index, state in enumerate(STATE_ORDER)}
            ).to_numpy(int)
            one_hot = np.eye(len(STATE_ORDER))[truth_index]
            safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
            selection = selection.assign(
                hard_state=np.asarray(STATE_ORDER)[np.argmax(probabilities, axis=1)],
                correct_hard=(np.argmax(probabilities, axis=1) == truth_index),
                brier=np.sum((probabilities - one_hot) ** 2, axis=1),
                log_loss=-np.log(
                    np.clip(selection["probability_true_state"].to_numpy(float), 1e-12, 1.0)
                ),
                entropy=-np.sum(
                    probabilities * np.log(safe_probabilities),
                    axis=1,
                ),
            )
            for name, value in {**metadata, "scenario": scenario}.items():
                selection[name] = value
            selection_frames.append(selection)

            switching = read_csv(
                expected_files["switching"],
                ("process", "n_draws", "n_switches", "switch_rate", "status"),
            )
            for name, value in {**metadata, "scenario": scenario}.items():
                switching[name] = value
            switching_frames.append(switching)

            trajectory = read_csv(
                expected_files["trajectory"],
                ("truth", "lower", "median", "upper"),
            )
            error = trajectory["median"].to_numpy(float) - trajectory["truth"].to_numpy(float)
            covered = (
                (trajectory["truth"] >= trajectory["lower"])
                & (trajectory["truth"] <= trajectory["upper"])
            )
            trajectory_rows.append(
                {
                    **metadata,
                    "scenario": scenario,
                    "n_time": len(trajectory),
                    "bias": float(np.mean(error)),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "coverage_90": float(np.mean(covered)),
                    "mean_width_90": float(
                        np.mean(trajectory["upper"] - trajectory["lower"])
                    ),
                }
            )

            parameter_truth = truth.get("parameter_truth", {})
            parameters = read_csv(
                expected_files["parameters"],
                ("parameter", "mean", "median", "lower", "upper"),
            )
            parameters = parameters[parameters["parameter"].isin(TRUTH_PARAMETERS)].copy()
            parameters["truth"] = parameters["parameter"].map(parameter_truth)
            parameters["mean_error"] = parameters["mean"] - parameters["truth"]
            parameters["median_error"] = parameters["median"] - parameters["truth"]
            parameters["covered_90"] = (
                (parameters["truth"] >= parameters["lower"])
                & (parameters["truth"] <= parameters["upper"])
            )
            for name, value in {**metadata, "scenario": scenario}.items():
                parameters[name] = value
            parameter_frames.append(parameters)

            diagnostics = read_csv(
                expected_files["diagnostics"],
                ("parameter", "rhat", "ess_bulk", "constant", "diagnostic"),
            )
            sampled = diagnostics[
                (~diagnostics["constant"].astype(bool))
                & diagnostics["rhat"].notna()
                & diagnostics["ess_bulk"].notna()
            ].copy()
            diagnostic_rows.append(
                {
                    **metadata,
                    "scenario": scenario,
                    "n_diagnostic_parameters": len(sampled),
                    "max_rhat": float(sampled["rhat"].max()) if len(sampled) else math.nan,
                    "median_rhat": float(sampled["rhat"].median()) if len(sampled) else math.nan,
                    "min_ess_bulk": float(sampled["ess_bulk"].min()) if len(sampled) else math.nan,
                    "median_ess_bulk": float(sampled["ess_bulk"].median()) if len(sampled) else math.nan,
                    "fraction_rhat_le_1_01": float(np.mean(sampled["rhat"] <= 1.01)) if len(sampled) else math.nan,
                    "fraction_rhat_le_1_05": float(np.mean(sampled["rhat"] <= 1.05)) if len(sampled) else math.nan,
                }
            )

            algorithm = read_csv(expected_files["algorithm"], ("metric", "value"))
            algorithm_wide = {
                str(row.metric): float(row.value)
                for row in algorithm.itertuples(index=False)
            }
            algorithm_rows.append({**metadata, "scenario": scenario, **algorithm_wide})

    def concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "inventory": pd.DataFrame(inventory_rows),
        "selection": concatenate(selection_frames),
        "switching": concatenate(switching_frames),
        "trajectory": pd.DataFrame(trajectory_rows),
        "parameters": concatenate(parameter_frames),
        "diagnostics": pd.DataFrame(diagnostic_rows),
        "algorithm": pd.DataFrame(algorithm_rows),
    }


def summarize(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    selection = tables["selection"]
    switching = tables["switching"]
    trajectory = tables["trajectory"]
    parameters = tables["parameters"]
    diagnostics = tables["diagnostics"]
    algorithm = tables["algorithm"]

    selection_with_switching = selection.merge(
        switching[
            ["seed", "profile", "scenario", "process", "switch_rate", "n_switches", "status"]
        ],
        on=["seed", "profile", "scenario", "process"],
        how="left",
    )
    selection_summary = (
        selection_with_switching.groupby(["profile", "scenario", "process"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_true_probability=("probability_true_state", "mean"),
            median_true_probability=("probability_true_state", "median"),
            min_true_probability=("probability_true_state", "min"),
            hard_recovery_rate=("correct_hard", "mean"),
            mean_brier=("brier", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            min_switches=("n_switches", "min"),
        )
        .reset_index()
    )

    trajectory_summary = (
        trajectory.groupby(["profile", "scenario"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_bias=("bias", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_coverage_90=("coverage_90", "mean"),
            mean_width_90=("mean_width_90", "mean"),
        )
        .reset_index()
    )

    parameter_summary = (
        parameters.groupby(["profile", "scenario", "parameter"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            truth=("truth", "first"),
            mean_posterior_mean=("mean", "mean"),
            mean_posterior_median=("median", "mean"),
            mean_error=("mean_error", "mean"),
            rmse_posterior_mean=("mean_error", lambda value: float(np.sqrt(np.mean(value**2)))),
            coverage_90=("covered_90", "mean"),
        )
        .reset_index()
    )

    diagnostic_summary = (
        diagnostics.groupby(["profile", "scenario"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            worst_rhat=("max_rhat", "max"),
            mean_max_rhat=("max_rhat", "mean"),
            worst_min_ess_bulk=("min_ess_bulk", "min"),
            mean_min_ess_bulk=("min_ess_bulk", "mean"),
            mean_fraction_rhat_le_1_01=("fraction_rhat_le_1_01", "mean"),
            mean_fraction_rhat_le_1_05=("fraction_rhat_le_1_05", "mean"),
        )
        .reset_index()
    )

    algorithm_aggregations: dict[str, tuple[str, str]] = {
        "n_seeds": ("seed", "nunique")
    }
    if "convergence_rate" in algorithm:
        algorithm_aggregations.update(
            mean_convergence_rate=("convergence_rate", "mean"),
            minimum_convergence_rate=("convergence_rate", "min"),
        )
    if "median_iterations" in algorithm:
        algorithm_aggregations["mean_median_iterations"] = (
            "median_iterations",
            "mean",
        )
    if "mean_support_rejections" in algorithm:
        algorithm_aggregations.update(
            mean_support_rejections=("mean_support_rejections", "mean"),
            maximum_support_rejections=("mean_support_rejections", "max"),
        )
    algorithm_summary = (
        algorithm.groupby(["profile", "scenario"], sort=False)
        .agg(**algorithm_aggregations)
        .reset_index()
    )

    profile_selection = (
        selection.groupby("profile", sort=False)
        .agg(
            n_seed_scenario_process=("probability_true_state", "size"),
            mean_true_probability=("probability_true_state", "mean"),
            hard_recovery_rate=("correct_hard", "mean"),
            mean_brier=("brier", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
        .reset_index()
    )
    profile_trajectory = (
        trajectory.groupby("profile", sort=False)
        .agg(
            mean_trajectory_rmse=("rmse", "mean"),
            mean_trajectory_coverage_90=("coverage_90", "mean"),
        )
        .reset_index()
    )
    profile_diagnostics = (
        diagnostics.groupby("profile", sort=False)
        .agg(
            worst_rhat=("max_rhat", "max"),
            worst_ess_bulk=("min_ess_bulk", "min"),
            mean_fraction_rhat_le_1_05=("fraction_rhat_le_1_05", "mean"),
        )
        .reset_index()
    )
    profile_summary = (
        profile_selection.merge(profile_trajectory, on="profile", how="outer")
        .merge(profile_diagnostics, on="profile", how="outer")
    )
    profile_summary["coverage_distance_from_0_90"] = np.abs(
        profile_summary["mean_trajectory_coverage_90"] - 0.90
    )

    profile_settings = (
        tables["inventory"]
        .loc[
            :,
            [
                "profile",
                "level_slab_sd",
                "trend_slab_sd",
                "season_slab_sd",
                "initial_season_sd",
            ],
        ]
        .drop_duplicates()
        .sort_values("profile")
        .reset_index(drop=True)
    )

    return {
        "selection_summary": selection_summary,
        "trajectory_summary": trajectory_summary,
        "parameter_summary": parameter_summary,
        "diagnostic_summary": diagnostic_summary,
        "algorithm_summary": algorithm_summary,
        "profile_summary": profile_summary,
        "profile_settings": profile_settings,
    }


def heatmap(
    table: pd.DataFrame,
    value: str,
    profiles: list[str],
    output: Path,
    title: str,
    colorbar_label: str,
) -> None:
    rows = [f"{scenario} / {process}" for scenario in SCENARIO_ORDER for process in PROCESS_ORDER]
    index = pd.MultiIndex.from_product(
        [SCENARIO_ORDER, PROCESS_ORDER], names=["scenario", "process"]
    )
    matrix = (
        table.pivot_table(index=["scenario", "process"], columns="profile", values=value)
        .reindex(index=index, columns=profiles)
        .to_numpy(float)
    )
    figure_height = max(6.0, 0.36 * len(rows))
    figure, axis = plt.subplots(figsize=(1.35 * len(profiles) + 3.5, figure_height))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(profiles)), profiles)
    axis.set_yticks(np.arange(len(rows)), rows)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value_at_cell = matrix[row, column]
            if np.isfinite(value_at_cell):
                color = "white" if value_at_cell < 0.55 else "black"
                axis.text(column, row, f"{value_at_cell:.2f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def trajectory_plot(table: pd.DataFrame, profiles: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(8.0, 1.2 * len(profiles)), 5.0))
    x = np.arange(len(profiles))
    for scenario in SCENARIO_ORDER:
        values = (
            table[table["scenario"] == scenario]
            .set_index("profile")
            .reindex(profiles)["mean_rmse"]
            .to_numpy(float)
        )
        axis.plot(x, values, marker="o", linewidth=1.5, label=scenario)
    axis.set_xticks(x, profiles)
    axis.set_ylabel("Mean latent-predictor RMSE")
    axis.set_title("Trajectory recovery across simulation seeds")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    raw: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, table in {**raw, **summaries}.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)

    profiles = profile_order(raw["inventory"]["profile"].astype(str))
    heatmap(
        summaries["selection_summary"],
        "mean_true_probability",
        profiles,
        figure_dir / "true_state_probability",
        "Posterior probability assigned to the true structure",
        "Mean probability",
    )
    heatmap(
        summaries["selection_summary"],
        "hard_recovery_rate",
        profiles,
        figure_dir / "hard_recovery_rate",
        "Hard structural recovery across simulation seeds",
        "Recovery rate",
    )
    trajectory_plot(
        summaries["trajectory_summary"],
        profiles,
        figure_dir / "trajectory_rmse",
    )


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output = (args.output or (results_root / "summary")).resolve()
    fit_root = results_root / "03_simulation_laplace"
    if not fit_root.is_dir():
        raise FileNotFoundError(f"Sensitivity fit directory not found: {fit_root}")

    runs = discover_runs(fit_root)
    if not runs:
        raise FileNotFoundError(f"No combined sensitivity runs found below {fit_root}")
    raw = collect(runs)
    incomplete = raw["inventory"][~raw["inventory"]["complete"]]
    if args.strict and len(incomplete):
        missing_path = output / "tables" / "inventory.csv"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        raw["inventory"].to_csv(missing_path, index=False)
        raise RuntimeError(
            f"Found {len(incomplete)} incomplete run-scenario combinations; "
            f"see {missing_path}."
        )
    if raw["selection"].empty:
        raise RuntimeError("No complete scenario tables were available to summarize.")

    summaries = summarize(raw)
    write_outputs(raw, summaries, output)
    print(f"Combined runs found: {len(runs)}")
    print(f"Complete run-scenario combinations: {int(raw['inventory']['complete'].sum())}")
    print(f"Incomplete run-scenario combinations: {len(incomplete)}")
    print(f"Compact sensitivity analysis: {output}")


if __name__ == "__main__":
    main()