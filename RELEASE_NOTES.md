# 1.9.6

The half-normal reference now has slope scale 0.0002. A 3×4 level/slope grid includes
slope 0.001; broader seasonal checks reach 0.10 and include a fixed cycle. Observation
scale and shape checks remain. Heavy-tail hyperpriors and leave-one-out experiments
are absent from the active matrix. Six matched private HN fits remain for the pooling
comparison; broader private checks are deferred.

Initial levels have SD 10; initial seasonal contrast coordinates have SD 10. The FS
sampler retains lag coordinates but now supports their equivalent full covariance.
This change affects coefficient updates, prior draws and archive round trips. A bridge
fit preserves the old SD-20 independent lag initialization at the new slope setting.

All four central-level slope settings have 11 expanding-window origins, 1970–2020,
and up to 140 held-out seasons, truncated only at the end of available observations.
Reports retain horizon bands and unique verifying-date counts. No future observations
are used in training or learning the shared scales.

Forecast simulation for independent-residual multiseries models is vectorized in
batches. Predictive quantiles can invert the mixture CDF to integrate observation
noise, while process and posterior uncertainty are still simulated. Seasonal full-fit
forecasts use 50,000 paths over 120 steps. Annual return-level calculations use a
recorded subset of whole joint paths. The public empirical-quantile default remains
available; this study explicitly selects CDF inversion.

A BIOBOT screen queue defaults to 48 chain workers with memory-aware scheduling;
HPC uses native Slurm arrays and an automatic dependent report collection job.
100 fits map to 95 HPC array elements. Both launch paths include startup checks,
prior simulation, provenance checks and completion/diagnostic manifests.

This is a computational release, not a claim that the new scientific runs converge
or confirm prior robustness. The manuscript's results must be updated from those runs.
