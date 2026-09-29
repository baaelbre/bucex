# BUCEX 1.9.6.1

This patch adds a focused joint level/slope calibration study. It preserves the
1.9.6 half-normal hierarchy, MH correction, initial-state priors, observation model,
conditional residual independence and both dynamic trend components. It does not
claim to resolve statistical identification through a change of coordinates.

The grid crosses five level scales (0.005–0.10) with five slope scales
(0.0001–0.002), while the seasonal scale stays at 0.01. The current reference
(0.01, 0.0002) is included. All 25 cells receive full-record fits and matched
hindcasts at 1990, 2000, 2010, 2015 and 2020. The requested horizon is 30 years,
truncated when observations end. Full-record forecasts also cover 30 years.

The focused HPC launcher runs 75 fits; the BIOBOT launcher runs the other 75.
Each screen fit uses two parallel chains with 1,000 warmup and 1,000 retained
draws each. The earlier 100-fit `all` suite remains available separately.

New outputs retain paired parameter draws, pointwise 95% level/rate intervals,
posterior allocation of new level/slope forecast variance, current-state forecast
uncertainty including covariance, and local posterior-mean hyperprior sensitivity.
Numerical checks include historical level/rate checkpoints, every final-30-year
state, hyperprior scores and forecast allocation targets. A job that finishes can
still be numerically flagged.

The automatic self-contained HTML compares all pairs and checks each adjacent
2×2 rectangle, including diagonal pairs and all six responses. Missing quantities
or numerical flags prevent a pass. It distinguishes whole-history from recent
stability and shows shared-scale learning, paired CRPS, interval coverage/width
and directional misses. A passing region is a descriptive candidate, not a prior
optimum or proof of robustness. It also reads disjoint compact exports from both
hosts without requiring posterior state archives.

HPC setup resolves physical directory paths before submission, preserves the
venv interpreter symlink, rejects an unresolved `/user/data` alias, and unloads
a conflicting Python module before loading the configured compute environment.
The existing working Gallade environment can be reused. The longer paper reference
requests 48 GiB to accommodate its retained states and diagnostic copies. The focused startup
probe exercises both grid corners and the reference; 25-cell prior simulations
run before the model array. Login planning uses Python 3.9-compatible standard
library code; numerical fitting requires Python ≥3.10.

Verification is recorded in `RELEASE_VALIDATION.json`. Local smoke fits are
software checks only and are not included as scientific results. No VSC or BIOBOT
jobs were submitted while preparing this package.
