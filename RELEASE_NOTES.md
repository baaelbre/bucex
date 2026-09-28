# BUCEX 1.9.5

## Pooled half-normal reference

Adds half-normal, half-t and half-Cauchy hyperpriors for the normal SDs of signed
non-centred innovation coefficients. The reference pools three scales across
six distinct latent paths. Initial slopes retain their fixed normal prior SD
of 0.01 per seasonal transition. Level and seasonal-coordinate initial SDs are
20. The exact log-scale hyperconditional includes the Jacobian and all normal
coefficient normalization terms. Legacy lognormal constructors and archives
retain their meaning.

## Experiments and execution

The default suite contains 106 fits per tier, including 41 full-record seasonal
settings, six pre-2019 fits, 52 standard/split validation fits, six matched
monthly/seasonal validation fits and a full-record monthly supplement. All use
pooled shrinkage. Fixed/unpooled and private-hierarchy reference fits are kept
in an explicit `deferred` batch, excluded from the default.

A single BIOBOT queue runs paper and screen together with bounded CPU and
estimated memory reservations. Chains and independent fits run in parallel.
The queue prioritizes reference fits, records process failures, preserves failed
attempts and skips completed matching configurations. It never treats screen
draws as paper draws. A startup probe exercises all three hyperprior families
before production fits begin. Collection and compact exports run automatically.

## Interpretation and reporting

Calibration uses the half-normal's marginal RMS coefficient scale. Half-t4
matches second moments; half-Cauchy matches a shared-scale upper quantile and
is marked as having no finite second moment. Monthly innovations are translated
to match the seasonal model's 30-year prior effects.

Risk reporting adds exact pre-2019 threshold probabilities, all six observed
summer predictions, period-average risks, local return periods, multiple return
levels and conditional annual extrema. Annual aggregation uses complete
meteorological years and pairs all seasonal distributions within a draw.
Forecasts span 30 years with 95% intervals; predictive validation additionally
reports 90%, 95% and 99% coverage. Numerical flags remain visible throughout.

See `BUCEX-1.9.5-commands.md` for all budgets, calibration constants, optional
batches, monitoring and the runtime limits of a full overnight study. No full
scientific fit has been run or declared converged during release preparation.
