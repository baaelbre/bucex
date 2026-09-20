# BUCEX 1.7.2 - publication workflow

This release makes the manuscript figure style reproducible through the public
package API and integrates the revised research configurations. It preserves
the 1.7.1 inference kernels, archive schema and normal innovation priors.

## New public APIs

- `publication_style()` is a scoped Matplotlib context, with legible fonts,
  blue/red/ochre colours, restrained bands and no import-time style changes.
- `save_figure()` saves PNG, PDF or SVG consistently.
- `ReportCollection` reads compact report exports and rejects ambiguous input.
- `save_publication_figures()` assembles configured manuscript panels, records
  input checksums and preserves the interval level actually exported.
- `pit_by_month()` and `coverage_by_month()` preserve counts and distinguish
  overlapping forecast cases from distinct dates.
- `parameter_trace_draws()`, `trace_frame()` and `traces_from_frame()` retain
  chain identity for portable trace/ACF figures.

## Reporting fixes

Initial slopes and levels now appear in parameter traces. Reports save small
compressed draw tables, including physical process SDs, so trace figures no
longer depend on access to the large latent-state archive. Monthly PIT plots
and normal-score dispersion tables accompany pooled diagnostics. Scale
parameters are included in convergence screening.

Endpoint and forecast-width exports now follow `credible_interval`, normally
95%. Old 1.7.1 endpoint tables and forecast-width figures remain at 90% and
must not be relabelled. `save_fits: false` is now honoured by the main reporter.
Forecast checks use the saved fit's series name, rather than blindly assigning
the first configured series. Copula fitting accepts a marginal `--scale`
override, and model comparisons accept `--series`.

## Research scope

The primary baseline keeps constant unknown scales and shapes. Repeating
monthly scales are a targeted adequacy extension. Normal FS shrinkage remains
the reference; six independent analyses and matched joint R=I/copula routes
remain available. No new stochastic volatility, factor model, selection
method or sampler was introduced for this patch release.

The nine figure recipes from the rewritten manuscript are included, plus
monthly scales and annual forecasts. Existing summary CSVs reproduce all
manuscript panels except a trace panel when draws were omitted from the old
ZIPs. Missing data are shown as placeholders; no traces are fabricated.

See `START_HERE.md`, `docs/FIGURES.md` and `docs/PUBLICATION_RUNS.md`.
Execution evidence is recorded in `validation/RELEASE_VALIDATION.md`. Scientific
acceptance still requires satisfactory full runs, sensitivity and validation;
this release does not claim those experiments have been completed.
