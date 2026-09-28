# BUCEX 1.9.3

Start with [BUCEX-1.9.3-commands.md](BUCEX-1.9.3-commands.md) for the exact fixed-prior grid, Gallade setup, screen/paper submission and result collection.

The seasonal analysis fits six responses separately with identical calibrated fixed Normal prior SDs. There are no shrinkage hyperpriors or copula parameters. Each HPC experiment runs six fits in parallel, with two chains per fit and twelve cores in both tiers. Paper runs use longer chains and stricter diagnostics.

`RUN_*_EXPERIMENTS.sh` submits 27 experiment jobs (23 full-record settings plus four prospective TXx checks). `RUN_*_ALL.sh` submits 88 jobs, adding five-year and original-design ten-year validation. Setup and a mandatory compute-node probe check the environment before production starts. Full-record forecasts cover 30 years.

The release supplies tested code and a starting prior specification. Production fits still need to be run and assessed.
