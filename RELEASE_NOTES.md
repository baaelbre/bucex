# BUCEX 1.9.4

- Adds direct Normal-SD hierarchy constructors and uses the same convention in
  conditional updates, prior draws, serialization, calibration and figures.
- Provides matched independent/shared marginal priors and a fixed Normal comparator.
- Keeps all six initial-rate priors separate with reference SD 0.01 per seasonal update.
- Replaces the active screen with 49 experiments / 159 fits, including width,
  shape, seasonality and pooling-influence checks.
- Uses 2,000 warm-up + 5,000 retained draws per chain for screening, with two chains.
- Routes shared fits to two CPUs and bundles of six separate fits to twelve CPUs.
- Retains the Gallade compute-node probe and dependency gate; submission does
  not execute the compute Python on the login CPU.
- Adds a biobot launcher and a collector accepting shared and separate report layouts.
- Exports scientific-target traces. Declared constant observation-scale
  placeholders are excluded from MCMC gates; sampled constants remain flagged.
- Removes historical contrasts and record/endpoint experiments from the active
  screen. The default screen does not run validation.

No copula parameters or common temperature trajectory are fitted. The shared
hierarchy borrows information about regularization and does not model residual
cross-summary correlation. The independent/shared comparison has identical
one-response marginal priors. Fixed Normal and scale-mixture priors differ.

All results go to a fresh `results/serra_194` tree. Existing 1.9.3 fits are not
relabelled or reused as 1.9.4 results. See `RELEASE_VALIDATION.json` for software
checks and their scope.
