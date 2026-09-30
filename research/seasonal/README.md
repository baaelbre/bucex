# Seasonal workflow — 1.9.8

See [the command guide](../../BUCEX-1.9.8-commands.md).

`config/sweetspot.json` declares the 3×3 level–slope grid and the central seasonal checks. `sweetspot_plan.py` creates 11 shared calibrations and 11 matched private variants. `job_plan.py` maps these to disjoint host batches. `jobs.py --verify` checks models, folds and priors; `probe.py` runs short sampler/report checks on the compute host.

The focused `sweetspot` batch contains 162 fits. `sweetspot_hpc` contains 111, and `sweetspot_validation` contains 51. Private full-record fits cover the whole grid; private validation covers the reference at five origins.

Reports are built by `sweetspot_report.py`; partial or numerically flagged fits cannot pass a stability gate. Archive compatibility APIs and the older general-purpose batches are retained.
