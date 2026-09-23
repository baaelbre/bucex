# BUCEX 1.8.5

The seasonal reference now uses level, slope, seasonal and initial-slope
shared-scale medians `(0.01, 0.0001, 0.01, 0.003)` per three-month update.
The last value corrects an overly narrow initial-rate prior: after integrating
over its lognormal shared scale it corresponds to about 0.194°C per decade.
The new `research.seasonal.grid` runner screens ten level/slope settings with
two chains each, supports `--jobs 10` for concurrent settings, and scores identical held-out seasonal forecasts. The monthly
reference configurations retain their prior values. The manuscript has not
been edited as part of this release.

The paper release concentrates on private Gaussian/GEV structural trajectories, continuous innovation priors, optional shared shrinkage of prior scales, and an optional Gaussian residual copula. Monthly and seasonal research now have their own directories and configurations. A constant dispersion comparison in each adequacy study keeps the need for periodic observation scales open while experiments are running.

The public API no longer exposes shared latent warming factors, departures, hierarchical SSVS, dynamic GEV phi or evolving observation scales. Different continuous priors remain. `MultiSeriesModel` fits require `MarginalPriors`; archives with this paper model remain loadable. The private compiler now retains copula parameters in forecast objects and simulates correlated observation errors, including calendar-specific correlations.

The 1.8.3 source archive and earlier fitted results should be retained separately. This release does not report new climate estimates or interpret short smoke runs as converged evidence. See [START_HERE](START_HERE.md), [1.8.5 checks](validation/RELEASE_VALIDATION_185.md) and [earlier release validation](validation/RELEASE_VALIDATION.md).
