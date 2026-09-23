# BUCEX 1.8.4

The paper release concentrates on private Gaussian/GEV structural trajectories, continuous innovation priors, optional shared shrinkage of prior scales, and an optional Gaussian residual copula. Monthly and seasonal research now have their own directories and configurations. A constant dispersion comparison in each adequacy study keeps the need for periodic observation scales open while experiments are running.

The public API no longer exposes shared latent warming factors, departures, hierarchical SSVS, dynamic GEV phi or evolving observation scales. Different continuous priors remain. `MultiSeriesModel` fits require `MarginalPriors`; archives with this paper model remain loadable. The private compiler now retains copula parameters in forecast objects and simulates correlated observation errors, including calendar-specific correlations.

The 1.8.3 source archive and earlier fitted results should be retained separately. This release does not report new climate estimates or interpret short smoke runs as converged evidence. See [START_HERE](START_HERE.md) and [release validation](validation/RELEASE_VALIDATION.md).
