# Paper runs in 1.9.3

Follow [the paper commands](BUCEX-1.9.3-commands.md). Both screen and paper use two chains per response, six simultaneous separate fits and twelve cores per experiment job. The long paper reference uses 3,000 warm-up and 8,000 retained iterations per chain; other paper fits use 2,000/4,000.

The declared reference is fixed Normal shrinkage with Normal prior SDs `(0.01, 0.0001, 0.01, 0.01)` for level, slope, seasonal innovations and initial slope. No shrinkage scale is learned. All six analyses use the same prior settings.

Each full-record reference report has its own `final_check.json`. The collector preserves missing tasks, failed computations and numerical warnings. Inspect those reports before using `COLLECT_HPC_RESULTS.sh paper experiments --require-complete`. A numerical pass does not establish forecast adequacy or select the best prior. The full forecast horizon is 120 seasons, with 95% reported intervals and 90/95/99% held-out coverage checks.
