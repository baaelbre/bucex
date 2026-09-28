# Exploratory manuscript Figures 1 and 2

From the source-release directory, after `python -m pip install -e ".[plot]"`:

```bash
python -m research.monthly.explore
```

This generates both figures from all six observed monthly temperature summaries.
It does not use fitted models, MCMC draws, priors, or posterior residuals. The
command prints a new directory below `results/serra_exploration/`.

| Output | Meaning |
|---|---|
| `figures/exploratory_seasonal_cycles.png` and `.pdf` | Figure 1: calendar-month means and empirical interquartile bands in two declared 30-year periods |
| `figures/exploratory_monthly_spread.png` and `.pdf` | Figure 2: within-month residual IQR after separate linear detrending in three declared eras |
| `data/exploratory_seasonal_cycles.csv` | The means, quartiles and sample counts plotted in Figure 1 |
| `data/exploratory_monthly_spread.csv` | The IQRs, fitted annual trends and sample counts underlying Figure 2 |
| `data/monthly_observations.csv` | Source observations, on their original temperature scale |
| `exploration_metadata.json` | Actual date span, input hash, methods and figure settings |
| `config.json` | Resolved SERRA configuration |

The image filenames match the manuscript's `includegraphics` references. Copy
the two PNGs or PDFs into the manuscript's `figures/` directory when updating it.
The old standalone `rebuild_exploration.py` is no longer required.

## Scientific definitions

Figure 1 compares January 1892–December 1921 with September 1996–August 2026.
Each window supplies 30 observations per calendar month. The curves are
arithmetic means of the observed summaries. Shading spans their empirical
25th–75th percentiles: it describes observation variability and is not a
confidence band or a posterior interval. Extreme summaries retain their
original signs; there is no GEV tail reflection in descriptive exploration.

Figure 2 uses January 1892–December 1936, January 1937–December 1981, and
January 1982–August 2026. For each response, era and calendar month, an OLS
intercept and linear year trend are fitted to those observations, and the
interquartile range of the residuals is calculated. There are 44 or 45
observations per group. This adjustment reduces contamination by changing
location. It is used only for the descriptive spread figure; Bayesian fits
continue to receive the unmodified observations.

Neither figure tests whether seasonality or dispersion is constant through
history. They motivate a specification whose adequacy is assessed separately.

## Edit the configuration

All application choices are in `research/monthly/config/exploration.json`:

- `data`: source directory, response order and loaded date range. `end: null`
  loads the latest available month, currently August 2026 in the bundled data.
- `periods`: labelled inclusive month ranges for Figure 1.
- `eras`: labelled inclusive month ranges for Figure 2.
- `figures`: colours, linestyles, units, panel layout, formats, DPI and optional
  figure titles. The defaults preserve the manuscript appearance.
- `min_count`: minimum finite observations per calendar month and window.
- `output`: root for timestamped result directories.

The comparison windows are explicit scientific choices. Extending the data
does not silently change them. Edit `periods` and `eras` when you intend to
change the figure windows. No exact record length such as 1,616 is hard-coded
in the exploration API. A window outside the available data raises an error
instead of silently clipping it; sparse groups are also rejected. The general
API counts missing observations and missing months in `n_missing`.

Useful overrides:

```bash
python -m research.monthly.explore --series TXm TNm --formats png
python -m research.monthly.explore --data-dir data --output results/my_exploration
python -m research.monthly.explore --config research/monthly/config/exploration.json
```

## General package API

The research script only loads data, reads the study choices and exports a
report. The numerical work and plotting are ordinary BUCEX functions:

```python
import bucex as bx

config = bx.load_config("research/monthly/config/exploration.json")
data = bx.load_uccle_multiseries(**config["data"])
exploration = bx.explore_monthly(
    data,
    periods=config["periods"],
    eras=config["eras"],
)
exploration.save("results/my_exploration", **config["figures"])
```

`data` can instead be any numeric monthly pandas Series or DataFrame with a
DatetimeIndex or monthly PeriodIndex. The same API applies to other stations
and other observation types, and does not select a likelihood for them.

Inspect `exploration.seasonal_cycles`, `exploration.monthly_spread` and
`exploration.metadata`, or call `exploration.plot_cycles(units="°C")` and
`exploration.plot_spread(units="°C")` to obtain Matplotlib Figures for further
editing. The corresponding public plot functions are
`bx.plot_exploratory_cycles` and `bx.plot_exploratory_spread`.
`exploration.save(..., figures=False)` exports tables without Matplotlib.

## Where the remaining paper figures come from

`research.monthly.explore` generates the two descriptive data figures.
`research.monthly.figures` continues to reconstruct level, slope, risk,
dependence, scale and diagnostic figures from fitted-model report exports.
Changing innovation priors affects those fitted-model figures; it does not
change Figures 1 and 2.
