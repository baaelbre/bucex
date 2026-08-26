# Migration to bucex 1.4.1

The general `Model`, `MultiSeriesModel`, `fit`, `FitResult`, prediction,
diagnostic, plotting, Uccle-loader, and hierarchical APIs remain available.

## From 1.4.0 to 1.4.1

No Python API change is required. The patch changes the focused phi example
configuration schema and their report output only.

Use the supplied schema-2 JSONs as the new templates. The fitted location
fields that were flat or implicit in 1.4.0 now live under `model.location`.
For simulations, the data-generating location settings now live under
`simulation.location`. Report controls formerly fixed by the short scripts are
explicit under `figures`.

The fit archive paths in examples 10 and 11 now follow the same established
layout as the other fitting examples:

```text
fits/<case-or-series>/combined.bucex
```

The generic parallel-chain runner uses these paths automatically. Existing
1.4.0 result archives remain readable, but do not mix old and new per-chain
result directories in one combine operation.

## From 1.3.2 to 1.4.0

No change is required for an existing stationary GEV analysis:

```python
bx.GEV()
```

is still stationary and is equivalent to `bx.GEV(phi="stationary")`. To add a
scale sensitivity fit, change only the observation declaration:

```python
bx.GEV(phi="linear")
bx.GEV(phi="rw")
bx.GEV(phi="ssvs")
```

Dynamic scale requires a univariate FS model. Existing multiseries fits remain
stationary; an attempted dynamic declaration now fails explicitly. Prior
profiles receive a default `PhiPrior`, so old prior construction remains
valid. To customize scale priors with structural SSVS, pass
`phi_prior=bx.PhiPrior(...)` to `ssvs_gev_priors`.

Use `fit.phi_draws()` and `fit.sigma_draws()` instead of assuming that
`fit.parameter("sigma")` is the full scale trajectory. The scalar `sigma`
remains a stationary or reference-scale compatibility value. Forecast and
risk APIs use the scale path automatically.

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

The complete analysis examples, centered/inverse-gamma benchmark, exact-engine
comparisons, and new scale-sensitivity scripts are in `examples/`; normal
runners are in `bash_scripts/`, and matching PBS jobs are in `job_scripts/`.

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

Set `output.run_id` only when a manually chosen identifier is useful and unique
to the submission. Leave it as JSON `null` for the collision-safe automatic
identifier. Bash/PBS runs include the configuration name and PBS job ID (or
local process ID); the settings signature is always present.

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
The persistence schema is 2.7.0 so dynamic log-scale paths and model indicators
round-trip. Loading remains checksum-verified and pickle-free, with readers for
all versions previously supported by 2.6.2.
