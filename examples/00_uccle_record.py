"""Inspect the complete Uccle temperature record before fitting a model.

Edit ``examples/config/record.json`` and run
``python examples/00_uccle_record.py``. No scientific setting is hidden in an
environment variable.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import bucex as bx
from _example_config import load_example_config


CONFIG, SETTINGS_PATH = load_example_config("record.json")
DATA = CONFIG["data"]
LOESS = CONFIG["loess"]
FIGURES = CONFIG["figures"]
OUTPUT = CONFIG["output"]

SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = OUTPUT.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
START = DATA.get("start")
END = DATA.get("end")
START_TAG = (START or "first").removesuffix("-01-01")
END_TAG = (END or "latest").removesuffix("-12-31")
OUTPUT_DIR = (
    Path(OUTPUT["results_root"])
    / SCRIPT_NAME
    / f"{RUN_TIMESTAMP}__y{START_TAG}-{END_TAG}"
)

data_dir = None if DATA.get("data_dir") is None else Path(DATA["data_dir"])
series_names = tuple(DATA["series"])
uccle = bx.load_uccle_multiseries(
    data_dir,
    series=series_names,
    start=START,
    end=END,
)

config_path = OUTPUT_DIR / "run_config.json"
if config_path.exists() and not bool(OUTPUT["overwrite"]):
    raise FileExistsError(f"Refusing to overwrite {config_path}; change output.run_id or output.overwrite in the JSON.")
table_dir = OUTPUT_DIR / "tables"
figure_dir = OUTPUT_DIR / "figures"
table_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

run_config = deepcopy(CONFIG)
run_config.update(
    {
        "script": SCRIPT_NAME,
        "settings_file": str(SETTINGS_PATH),
        "created_at": datetime.now().astimezone().isoformat(),
        "bucex_version": bx.__version__,
        "output_directory": str(OUTPUT_DIR),
        "actual_start": str(uccle.index.min().date()),
        "actual_end": str(uccle.index.max().date()),
    }
)
bx.save_config(run_config, config_path)

uccle.rename_axis("date").reset_index().to_csv(
    table_dir / "uccle_monthly_extremes.csv", index=False
)
integrity = bx.validate_uccle_data(data_dir, check_daily=False)
integrity.to_csv(table_dir / "uccle_integrity.csv")

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)
colors = {"navy": "#123B4A", "teal": "#1D7F7A", "grey": "#7A8589"}

# Monthly records.
x_all = (uccle.index - uccle.index[0]).days.to_numpy(float)
figure, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
for axis, name in zip(axes.ravel(), series_names):
    values = uccle[name].to_numpy(float)
    finite = np.isfinite(values)
    smooth = bx.loess_smooth(
        x_all[finite],
        values[finite],
        fraction=float(LOESS["monthly_fraction"]),
        robust_iterations=int(LOESS["robust_iterations"]),
    )
    axis.scatter(uccle.index[finite], values[finite], s=4, alpha=0.25, color=colors["grey"], label=r"$y_t$")
    axis.plot(uccle.index[finite], smooth, color=colors["teal"], linewidth=1.8, label="LOESS")
    axis.set_ylabel(f"{name} / °C")
    axis.grid(axis="y", alpha=0.35)
axes[0, 0].legend(loc="upper left")
monthly_title = bx.config_title(CONFIG, "monthly_extremes")
if monthly_title is not None:
    figure.suptitle(monthly_title)
figure.tight_layout()
for extension in FIGURES["formats"]:
    figure.savefig(figure_dir / f"00_uccle_extremes.{extension}", dpi=int(FIGURES["dpi"]), bbox_inches="tight")
plt.close(figure)

# Evolution of annual warm daytime extremes.
annual_txx = uccle["TXx"].resample("YE").max().dropna()
x_annual = (annual_txx.index - annual_txx.index[0]).days.to_numpy(float)
annual_smooth = bx.loess_smooth(
    x_annual,
    annual_txx.to_numpy(float),
    fraction=float(LOESS["annual_fraction"]),
    robust_iterations=int(LOESS["robust_iterations"]),
)
figure, axis = plt.subplots(figsize=(11, 4.5))
axis.scatter(annual_txx.index, annual_txx, s=18, alpha=0.45, color=colors["grey"], label=r"annual $TXx$")
axis.plot(annual_txx.index, annual_smooth, color=colors["teal"], linewidth=2.4, label="LOESS")
axis.set_ylabel("TXx / °C")
axis.grid(axis="y", alpha=0.35)
axis.legend()
annual_title = bx.config_title(CONFIG, "annual_txx")
if annual_title is not None:
    axis.set_title(annual_title)
figure.tight_layout()
for extension in FIGURES["formats"]:
    figure.savefig(figure_dir / f"01_txx_evolution.{extension}", dpi=int(FIGURES["dpi"]), bbox_inches="tight")
plt.close(figure)

# Early and recent within-year patterns.
available_years = sorted(uccle["TXx"].dropna().index.year.unique())
early_years, late_years = available_years[:30], available_years[-30:]
early = uccle.loc[uccle.index.year.astype(int).isin(early_years), "TXx"].groupby(lambda date: date.month).mean()
late = uccle.loc[uccle.index.year.astype(int).isin(late_years), "TXx"].groupby(lambda date: date.month).mean()
figure, axis = plt.subplots(figsize=(8.5, 4.5))
axis.plot(range(1, 13), early, marker="o", color=colors["grey"], label=f"{early_years[0]}–{early_years[-1]}")
axis.plot(range(1, 13), late, marker="o", color=colors["teal"], label=f"{late_years[0]}–{late_years[-1]}")
axis.set_xticks(range(1, 13), ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
axis.set_ylabel("mean monthly TXx / °C")
axis.grid(axis="y", alpha=0.35)
axis.legend()
seasonal_title = bx.config_title(CONFIG, "seasonal_cycle")
if seasonal_title is not None:
    axis.set_title(seasonal_title)
figure.tight_layout()
for extension in FIGURES["formats"]:
    figure.savefig(figure_dir / f"02_txx_seasonal_cycle.{extension}", dpi=int(FIGURES["dpi"]), bbox_inches="tight")
plt.close(figure)

print(f"Uccle record outputs: {OUTPUT_DIR}")
