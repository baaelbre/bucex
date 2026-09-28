# Paper runs in 1.9.2

Use the paper section of [BUCEX-1.9.2-commands.md](BUCEX-1.9.2-commands.md).

`bash bash_scripts/run_local.sh paper reference` runs the six long reference fits, one response per task. `RUN_PAPER_EXPERIMENTS.sh` includes all posterior sensitivities and prospective 2019 TXx checks. `RUN_PAPER_ALL.sh` also includes five-year and original-design ten-year validation.

Each reference report has its own `final_check.json`. The collector preserves missing tasks, failed computations and numerical warnings. Run its `--require-complete` gate only after inspecting the reports. No joint/copula model or pooled hyperparameter is used by this job grid.
