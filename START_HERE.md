# Start here — BUCEX 1.9.5

Read [BUCEX-1.9.5-commands.md](BUCEX-1.9.5-commands.md).

For BIOBOT, activate your working Python environment, select a fresh results
root and launch `bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch all`. Use
`--dry-run` first. The launcher checks all configurations and exercises the
three hyperprior families before starting the resource-bounded queue.

The reference pools half-normal innovation-prior SDs across six summaries.
Initial slopes keep separate fixed normal priors. There are 106 active fits
per tier; optional unpooled comparisons are in `deferred`. See the guide for
all priors, budgets, monitoring, automatic collection, exports and resumption.
The full paper suite may take longer than one night.

For overnight screening: all 106 fits, two chains each, 1,000 warm-up + 1,000 retained per chain; no wall-time cutoff. Use a fresh results root after this configuration update. Paper fits are a separate later run.
