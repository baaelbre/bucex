# BUCEX 1.9.1

Read [BUCEX-1.9.1-commands.md](BUCEX-1.9.1-commands.md) for installation,
settings, screen/paper budgets, collection and retries.

From this release directory, after activating your working Python environment:

```bash
export BUCEX_PYTHON="$(command -v python3)"
"$BUCEX_PYTHON" -m pip install -e '.[plot]'
export BUCEX_RESULTS_ROOT="$PWD/results/serra_191_parallel"
```

On HPC, submit posterior sensitivities with `bash RUN_SCREEN_EXPERIMENTS.sh`
or, after screening, `bash RUN_PAPER_EXPERIMENTS.sh`. Native Slurm is preferred;
`BUCEX_SCHEDULER=slurm` / `pbs` selects the scheduler explicitly.

On biobot, use `BUCEX_MAX_JOBS=8 bash RUN_SCREEN_VALIDATION.sh`, or later
`BUCEX_MAX_JOBS=6 bash RUN_PAPER_VALIDATION.sh`. These are foreground local
runners. The optional `validation10` argument runs the original 60/80/90%
splits with ten-year forecasts in a separate batch.

There are 23 posterior fits, 52 five-year validation fits and three optional
ten-year checks. All use identity residual dependence. Three innovation-scale
hyperparameters remain shared; six initial rates have separate fixed priors.
Full-record forecasts span 30 years. Paper results require numerical and
scientific review after running.
