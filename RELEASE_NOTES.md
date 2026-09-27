# BUCEX 1.8.9

The 1.8.9 seasonal manuscript reference has medians `(0.01, 0.0001,
0.01, 0.01)` per season and initial seasonal coefficient SD 20°C.
The initial-rate prior SD is about 0.96°C/decade after integrating the
hierarchy. It forecasts 120 seasons with 12,000 predictive draws and 95%
intervals. See [FINAL_RUN.md](FINAL_RUN.md) for the 106 independent PBS array
fits, five separate long fits, result collection and numerical gates.

## 1.8.8 archive

The seasonal manuscript prior now has four median-absolute shared-scale
hyperprior medians `(0.01, 0.0001, 0.01, 0.001)` per seasonal transition.
The initial-slope conditional normal SD is `m_beta0 / Phi^-1(.75)`, matching
the other three components and the manuscript. The old SD-anchored API remains
available, and the archived 2026-09-23 configuration retains its prior.
At 30 years the integrated contribution SDs are about 0.263, 0.181, 0.186
and 0.288°C; the initial-rate SD is 0.096°C per decade. Existing 1.8.7
posteriors cannot be treated as fits under these revised priors.

The complete paper queue in `RUN_PAPER_EXPERIMENTS.sh` covers prior effects,
full fit, physical-scale and width/shape sensitivity, structural and
dependence alternatives, pre-2019 prediction, expanding-window validation,
time-resolved additive-gap and rate-difference contrasts, and matched
monthly/seasonal scoring. The main figures follow the chosen
records/LOESS, all-six scale and PIT/QQ layouts. See `FINAL_RUN.md` for
separate commands. Gates require adequate numerical diagnostics.

## 1.8.7

This release freezes the seasonal manuscript specification to the successful
`uccle_copula_20260923T222238_406159Z.zip` settings while retaining the 1.8.6.1
dummy-seasonal initialization correction. The four shared-scale medians are
`(0.0144513322, 0.000108839649, 0.00834348054, 0.00463877353)` per seasonal
update for level, slope, seasonal and initial-slope terms. They imply marginal
prior SDs of approximately 0.38°C, 0.20°C and 0.15°C for their 30-year
contributions and 0.30°C/decade for the initial rate.

The final run increases only the MCMC budget to four chains with 3,000 warm-up
and 8,000 retained draws. A strict post-run gate verifies configuration, data,
required exports and numerical diagnostics. A stable-name figure builder now
generates the main seasonal manuscript panels and records source checksums.
Targeted configs cover prior/structural sensitivity, the prospective JJA-2019
record event, expanding-window tail validation and matched monthly-block
sensitivity. See [FINAL_RUN.md](FINAL_RUN.md).

Release checks and the locked source/configuration checksums are recorded in
[validation/RELEASE_VALIDATION_187.md](validation/RELEASE_VALIDATION_187.md).

## 1.8.6.1

This patch corrects the lag ordering of the dummy-seasonal starting state in
single-series and joint fits. The earlier 2015 TXn pilot had a badly drifting
chain; the same-seed short TXn-only comparison now produces a stable shape and
observation scale. This is a starting-value correction; the model and priors
are unchanged. The old ten-setting grid is not suitable for selecting a prior.
Run the new four-chain mixing checks for both grid origins before repeating
any sensitivity fit. The grid enforces this gate and writes to a fresh
directory. See [seasonal research](research/seasonal/README.md).

## Previous releases

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
