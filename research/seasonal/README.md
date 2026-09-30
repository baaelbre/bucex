# Seasonal workflow — 1.9.8.1

See [the command guide](../../BUCEX-1.9.8.1-commands.md).

`config/sweetspot.json` defines a 3×4 level–slope grid plus two central seasonal checks.
`sweetspot_plan.py` creates 14 pooled HN, 14 separate HN and 14 fixed-Normal variants.
Fixed-Normal variants use `scope="fixed"`, remove both hierarchy declarations, and
retain calibrated zero-centred Normal priors on the signed innovation coefficients.

`job_plan.py` maps the study to disjoint host batches: `sweetspot_hpc` has 390 fits
in 100 scheduler elements; `sweetspot_validation` has 312 individual BIOBOT fits.
The complete `sweetspot` batch has 702 fits. Pooled and fixed-prior validation covers
all settings at five matched origins; separate HN validation covers the reference.

`jobs.py --verify` compiles models and checks priors, folds and resources.
`probe.py` tests tiny Gaussian/GEV runs for all three constructions on the compute host.
`prior_simulations.py --suite sweetspot` simulates all 14 pooled HN and 14 fixed-Normal
settings; the separate HN one-response marginal prior matches the pooled HN prior.

`sweetspot_report.py` collects all six separate responses, keeps incomplete bundles
flagged, produces 3×4 grids for each construction, and compares identical calibrations.
`matched_prior_crps.csv` pairs separate and pooled predictions on the same response,
origin, date and horizon. Fixed-Normal local sensitivity uses the squared coefficient
score, not a nonexistent hyperparameter. The original general-purpose batches remain
available but do not substitute for the focused grid.
