# SERRA workflow with BUCEX 1.8.2

The main route is [START_HERE](../../START_HERE.md): exploration, physical prior
calibration, one smoke run, one preliminary fit, figures, two forecast origins.
The detailed sensitivity grid is [deferred to the supplement](../../docs/SUPPLEMENTARY_RUNS.md).

| Task | Module | Configuration |
|---|---|---|
| Data Figures 1–2 | `explore` | `revision/exploration.json` |
| Dates, memory, physical prior effects | `preflight` | `draft/main.json` |
| Four-process execution check | `copula` | `draft/smoke.json` |
| Main preliminary model | `copula` | `draft/main.json` |
| Historical predictive checks | `validate` | `draft/predictive.json` |
| Fitted-model panels from CSVs | `figures --reports PATH` | `revision/figures.json` |
| Regenerate reports from saved draws | `report --fit PATH/fit.bucex` | Saved config |
| Shared shrinkage with R=I | `run` | `draft/independence.json` |
| Six separate fixed-prior models | `univariate` | `draft/independent.json` |
| Later higher-budget fit | `copula` | `draft/final.json` |
| Targeted sensitivity | `prior_assessment` | `draft/sensitivity.json` and related files |

Use `python -m research.serra.MODULE --config research/serra/config/CONFIG`.
Scripts expose `--help`; START_HERE lists the complete commands.

`models.py` declares the scientific model via public BUCEX APIs. `run.py`
orchestrates fits; `validate.py` refits on training data; `report.py` collects
public diagnostics, figures, forecasts and risks. The package owns the priors,
shared-scale updates, state samplers, persistence and scientific calculations.
`SharedShrinkage.from_effects` accepts physical changes; `calibration()` reports
what existing anchors imply. See [the hierarchy API](../../docs/SHARED_SHRINKAGE.md).

The main fit saves its posterior before generating reports. Compact results
can be shared without the large archive, but keep that archive locally for
replotting. A disconnected foreground SSH session may lose an unfinished fit;
use tmux or the documented nohup command before starting.

All new main fits extend through August 2026. Old archived fits retain the
model they were sampled under. New inherited configuration defaults are not
an exact reproduction of earlier releases; use saved resolved declarations.
