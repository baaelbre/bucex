# BUCEX 1.9.2

Start with [BUCEX-1.9.2-commands.md](BUCEX-1.9.2-commands.md).

The seasonal paper now uses six separate analyses, with private innovation hyperparameters and initial slopes, no copula and no cross-response borrowing. All current screen/paper launchers run locally on biobot with bounded fit concurrency. `RUN_SCREEN_ALL.sh` and `RUN_PAPER_ALL.sh` include posterior sensitivity and both validation designs. `RUN_*_EXPERIMENTS.sh` runs sensitivities without the held-out validation batches.

Screen results assess sensitivity; paper budgets and convergence gates do not themselves establish adequate forecasting. The doubled-slope and stronger seasonal-shrinkage alternatives are comparisons, not a preselected winner.
