# bucex 1.4.1 release notes

Version 1.4.1 is a documentation and example-output maintenance release for
the time-varying GEV log-scale API introduced in 1.4.0. The modelling and fit
API remains unchanged:

```python
bx.GEV()                  # stationary phi=log(sigma), still the default
bx.GEV(phi="linear")
bx.GEV(phi="rw")
bx.GEV(phi="ssvs")
```

## Self-documenting JSONs

All four simulation and 16 Uccle phi configurations now contain valid
`_comment` fields. Ordinary `//` comments are not legal JSON; `_comment` keeps
the files readable by standard tools and is ignored by the examples.

The location model is no longer implicit in example 10. Simulation files
separate:

- `simulation.location`: the known data-generating location structure;
- `model.location`: the fitted local-linear-trend plus dummy-seasonal
  structural-SSVS model;
- `simulation.phi`: the data-generating log-scale process;
- `model.phi`: the stationary, linear, RW, or SSVS scale model being fitted.

This also clarifies that the supplied `simulation_linear.json` fits a linear
scale model to the same stationary-scale truth used by the other three files.
It is a sensitivity fit, not a declaration of linear simulation truth.

`examples/config/phi/README.md` gives the location equations, SSVS probability
ordering, every field's role, edit examples, and the complete output tree.

## Full report parity

Examples 10 and 11 again produce the established simulation and Uccle output
set instead of only a fit archive and phi plot. Their tables include:

- parameter summaries and MCMC diagnostics;
- location trajectories and structural selection/model switching;
- posterior predictive checks and full/focused/level forecasts;
- phi and sigma paths and scale-model probabilities;
- a reproducibility summary and the effective JSON.

Their figures include predictor trajectories, posterior prediction, forecasts,
level and slope, structural selection, process scales, GEV parameters,
seasonality, and optional MCMC diagnostics. Uccle also retains the endpoint
figure when finite. `phi.*` is an additional figure in each report.

The result paths match the established layout:

```text
fits/<case-or-series>/combined.bucex
tables/<case-or-series>/...
figures/<case-or-series>/...
```

All report controls—formats, interval probability, predictive draws, focus
phase/month, forecast horizon/history, and diagnostics—are explicit in JSON.

## Compatibility

- The public GEV, prior, fit, result, prediction, and archive APIs are
  unchanged from 1.4.0.
- `GEV()` remains stationary.
- Archive schema remains 2.7.0 and all previously supported archives remain
  readable.
- Existing 1.4.0 phi JSONs are superseded by the documented schema-2 example
  files; the scientific defaults are unchanged.
- PBS continues to allocate resources only. It does not overwrite JSON draws,
  warmup, priors, model settings, or report settings.
