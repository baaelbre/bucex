# Migration from the former 2.6.2 package to 1.3.1

The general `Model`, `MultiSeriesModel`, `fit`, `FitResult`, prediction,
diagnostic, plotting, Uccle-loader, and hierarchical APIs remain available.

## Presentation workflow removal

The clean 1.x repository preserves the public API decisions made in former
version 2.6.2, including removal of `bucex.workflows`, `PresentationConfig`,
`PresentationWorkflow`, `WorkflowPaths`, the scenario factories, result
workflow helpers, and the `bucex-presentation` console command. These objects
hid analysis choices that are more useful when visible.

Replace a workflow call with the corresponding numbered script, or copy its
explicit public-API declarations into an analysis:

```python
import bucex as bx

model = bx.Model(
    bx.GEV(),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ),
)
laplace = bx.fit(y, model=model, priors=priors, engine="laplace", ...)
pgas = bx.fit(y, model=model, priors=laplace.priors, engine="pgas", init=laplace, ...)
```

The seven complete analysis examples and the centered/inverse-gamma diagnostic
benchmark are in `examples/`; normal runners are in `bash_scripts/`, and
matching PBS jobs are in `job_scripts/`.

## Result paths

Results are always indexed by script, timestamp, and identifying settings:

```text
results/<script>/<timestamp>__<settings-signature>/
  run_config.json
  simulations/               # simulation scripts
  fits/<scenario-or-series>/
  tables/<scenario-or-series>/
  figures/<scenario-or-series>/
```

Set `output.run_id` in the selected JSON when several scripts should share a
human-chosen prefix. Leave it as JSON `null` for an automatic timestamp. The
settings signature is always present.

## New plots

`bx.loess_smooth()` replaces rolling-median smoothing in the Uccle descriptive
example. `fit.plot("level")` and `fit.plot("slope")` provide separate trend
component figures, while `fit.plot("season")` plots phase-specific posterior
seasonal effects without adding the level. Version 1.0.1 also adds an
observation-free level option, conditional fixed/dynamic slope overlays,
posterior predictive checks, and forecasts.

## Warm starts and archives

The full-path Laplace-to-PGAS warm-start contract is unchanged. Compatibility
checks cover family, period, transformed observations, and state dimensions.
The persistence schema remains 2.6.2; loading remains checksum-verified and
pickle-free, with readers for all versions previously supported by 2.6.2.
