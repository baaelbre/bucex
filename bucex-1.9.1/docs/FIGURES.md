# Reproducible manuscript figures

All aesthetics and layouts come from BUCEX. The short study script chooses
series, quantities, months and event thresholds through
`research/monthly/config/figures.json`.

The reference is the rewritten SERRA manuscript: blue TX panels, red TN
panels, 2-by-3 component layouts, readable sans-serif labels, light horizontal
guides and translucent pointwise bands. A colour mapping belongs to the study
configuration; the package also works with other response names.

## Rebuild from reports

```text
python -m research.monthly.figures --reports PATH_TO_RUN
python -m research.monthly.figures --reports PATH_TO_TXm PATH_TO_TNm PATH_TO_TXx PATH_TO_TXn PATH_TO_TNx PATH_TO_TNn --formats png pdf
```

Paths may be a single joint report, one independent-run root, or separate
series directories. Do not pass a parent containing several models for each
series: duplicate matches are an error. Output goes into a new dated directory.
No fit archive is loaded and no MCMC is run.

To start with one response and a few panels:

```text
python -m research.monthly.figures --reports PATH_TO_TNm --series TNm --panels levels slopes monthly_residual_spread qq_scores monthly_scales
```

| Figure stem | Source | Interpretation |
|---|---|---|
| `levels` | `*_level.csv` | Six original-orientation latent levels |
| `slopes` | `*_slope_C_per_decade.csv` | Local warming slopes; no further smoothing |
| `monthly_residual_spread` | `*_smoothed_pit.csv` | Within-month normal-score SD |
| `residual_dependence` | aligned `*_smoothed_pit.csv` | Descriptive pooled/monthly score correlations |
| `monthly_risks` | `*_risk.csv` and saved thresholds | Selected January/July conditional probabilities |
| `qq_scores` | `*_smoothed_pit.csv` | In-sample score QQ plots |
| `forecast_uncertainty` | `*_forecast_monthly.csv` | July location/observation bands and widths |
| `source_TXn_parameter_traces` | `TXn_parameter_traces.csv.gz` | Chains and ACFs, including initial slopes |
| `source_TNm_pit_qq_residuals` | `TNm_smoothed_pit.csv` | Four pooled predictive-check panels |
| `monthly_scales` | `*_scale_by_month.csv` | Observation scale for each calendar month |
| `annual_forecasts` | `*_forecast_year.csv` | Complete-year means or maxima/minima |

The two `source_` names are retained to replace the corresponding manuscript
image paths. Their content is generated from numerical exports, not restyled
pixels. Put generated PNGs in the manuscript's `figures/` directory and update
the results and captions when the model changes.

## Trace data and older archives

Version 1.7.1 saved trace PNGs but did not export their underlying parameter
draws. Summary medians/intervals cannot recover those traces. Re-export an
existing `.bucex` fit to obtain the new compressed CSVs:

```text
python -m research.monthly.report --fit PATH_TO_CASE/fit.bucex --format png --level 0.95
```

The new report includes unthinned `(chain, draw)` parameter, target and
monthly-scale trace tables. They are much smaller than the full posterior
state archive. Retain them when sharing reports. An absent table generates a
marked placeholder and an entry in `figure_manifest.json`; `--strict` raises
instead. Invalid data always raise, even without `--strict`.

## What the manifest protects

- Declared interval levels must match across selected reports. The tool never
  labels 90% bounds as 95%; use `report --level` to recompute from draws.
- Risk thresholds in recipes must match saved report thresholds.
- Score correlations require exactly aligned dates. They are not estimated
  copula parameters and do not replace joint posterior inference.
- PIT endpoints are counted before clipping for normal scores. No IID
  uniformity p-value or calibration claim is added to smoothed PIT figures.
- Filenames, source paths/checksums, report configurations, figure recipes and
  missing inputs are recorded. A manifest does not certify convergence.

Read recipe captions as interpretation notes. The final manuscript must state
which model each figure represents and whether its bounds are posterior,
predictive or conditional-risk intervals. Constant-scale and monthly-scale
reports should be exported separately, rather than mixed within a panel.

## General plotting API

```python
import bucex as bx

with bx.publication_style():
    figure, axis = fit.plot("slope", scale="decade", credible_interval=.95)
    bx.save_figure(figure, "figures/slope", formats=("png", "pdf"))

reports = bx.ReportCollection({"station_A": "reports/A", "station_B": "reports/B"})
recipes = [{"name": "levels", "kind": "bands", "table": "level",
            "ylabel": "Latent level / °C", "columns": 2}]
bx.save_publication_figures(reports, "figures/comparison", recipes=recipes,
                            colors={"station_A": "#24658a", "station_B": "#a44839"})
```

Style contexts restore the caller's settings on exit. Report configurations
accept `figure_style: "manuscript"` or `"default"`, `figure_dpi` and
`figure_format`. PNG at 180 dpi is the research default; PDF and SVG are
available. Change recipe `series`, `month`, `events`, `columns`, `ylabel`,
`formats` and `colors` in JSON without editing package plotting code.
