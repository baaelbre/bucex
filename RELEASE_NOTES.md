# bucex 1.4.0 release notes

Version 1.4.0 adds optional linear, random-walk, and SSVS evolution for the GEV
log scale \(\phi_t=\log(\sigma_t)\). Stationary scale remains the default and
existing `GEV()` calls retain their previous model.

## Clean model API

The complete public choice is:

```python
bx.GEV()
bx.GEV(phi="linear")
bx.GEV(phi="rw")
bx.GEV(phi="ssvs")
```

No additional fit function or scale-component class is required. Model JSON
serialization records `phi`, and invalid choices fail at construction.
Time-varying scale currently uses univariate FS GEV inference; unsupported
multiseries or parameterization combinations fail explicitly.

## Scale priors and SSVS

`PhiPrior` groups:

- a normal prior for the whole-record linear log-scale change;
- an inverse-gamma prior for RW innovation variance;
- stationary/linear/RW model probabilities.

`ssvs_gev_priors(phi_prior=...)` exposes these settings without changing the
existing structural SSVS controls. `phi="ssvs"` uses an exact product-space
model update in exact engines, so scale-model selection and location-structure
selection can coexist.

## Conditional inference

The new scale kernel is separate from the structural location kernel. It uses
analytic GEV derivatives with respect to log scale. Linear and stationary
coefficients use the exact likelihood. The RW path uses an iterated-Laplace
Kalman smoother; under Laplace-MH or PGAS it is an independence proposal with
an exact-density MH correction. Ordinary Laplace remains explicitly
approximate.

## Results and prediction

`FitResult.phi_draws()` and `sigma_draws()` provide uniform time paths for all
four models. Dynamic fits retain mode-specific scalar parameters and schema
2.7.0 archives. Posterior predictive checks use fitted scale paths. Forecasts
keep stationary scale fixed, continue the linear basis, propagate RW
innovations, or use each SSVS draw's selected model. Risk summaries and PIT
calculations use time-specific scale.

## JSON examples and PBS

Two standalone scripts were added:

- `examples/10_simulation_phi.py`;
- `examples/11_uccle_phi.py`.

Four simulation JSONs and 16 Uccle JSONs make stationary, linear, RW, and SSVS
fits directly editable. Every hyperparameter and computational control is in
JSON. The defaults are 1,000 retained draws, 1,000 warmup iterations, and four
chains.

Matching Bash and PBS files use the generic process-level chain runner. Each
series/model job and every chain can be scheduled independently, with
collision-safe run IDs containing the source config name and PBS job ID. The
PBS files allocate resources only; they do not override JSON settings.

## Compatibility

- `GEV()` remains stationary.
- Existing stationary GEV fits and prior profiles continue to work.
- Archive schema advances to 2.7.0; all previously supported schemas remain
  readable.
- Dynamic scale is intentionally not enabled for `MultiSeriesModel` in this
  release.
