# Working manuscript alignment

The working analysis compares six private Gaussian/GEV location models under continuous FS priors. Gaussian copula dependence and shared regularization of prior scales are estimated in the joint paper model. The fitted scale baseline and GEV shape are time-constant; periodic observation-scale effects are a candidate to assess rather than an established necessity.

| Question | Experiment |
|---|---|
| Monthly versus complete seasonal blocks | `research/seasonal/compare.py` on identical seasonal forecast targets |
| Need for periodic observation scales | `constant_dispersion` variant in monthly and seasonal `adequacy.json` |
| Location seasonality | `fixed_location_seasonality` variant in both `adequacy.json` files |
| Sensitivity of innovations, shape, and prior pooling | `sensitivity.json`, `anchors.json`, and independent fits |
| Dependence and predictive adequacy | Matched copula/identity fits, block forecasts, PIT and ordering checks |

No long-run posterior findings are asserted by this software release. Archived analyses retain the priors and data under which they were sampled. Final manuscript claims require the planned fits, convergence and out-of-sample checks.
