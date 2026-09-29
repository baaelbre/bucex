# Seasonal workflow — 1.9.6.1

See [the launch guide](../../BUCEX-1.9.6.1-commands.md) for the focused joint
level/slope study. `config/main.json` preserves the 1.9.6 statistical reference;
`config/sweetspot.json` declares the new grid, origins and descriptive tolerances.

- `sweetspot_hpc`: 25 full-record fits and 50 earlier-origin hindcasts.
- `sweetspot_validation`: 75 recent-origin hindcasts for BIOBOT.
- `sweetspot`: all 150 fits, when running the entire study on one host.
- `sweetspot_posterior` and `sweetspot_long`: the two parts of the HPC batch.

Use `jobs --verify` for configurations/folds and `overnight --dry-run` for the
resource-bounded local queue. The focused launchers keep the legacy `all` suite
separate. Do not run overlapping queues on the same results root.

`finish` builds the HTML, figures and compact evidence ZIP. `sweetspot_report`
combines disjoint host exports using `--other-root`. Scientific settings, source
and data must match; output paths may differ. Completed fits retain numerical
flags. Screen and paper outputs use separate tier directories.
