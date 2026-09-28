# Correct 1.9.3 to the manuscript prior settings

The first 1.9.3 release divided the requested numbers by Phi^-1(.75). The latest manuscript instead declares Normal prior SDs directly. This correction uses innovation SDs (0.01, 0.0001, 0.01) and initial-slope SD 0.01 (variance 0.0001). Initial-rate SD is now 0.40 degrees C/decade. There are still six separate fits, no shrinkage hyperpriors, two chains per response and twelve cores per experiment job.

The four widths are fixed hyperparameters, not MCMC initial values. The coefficients and process variances are estimated. Sensitivity changes the fixed SDs in separate refits; initial-rate SDs are 0.20, 0.40 and 0.80 degrees C/decade for the half/reference/double variants.

## Apply on the HPC

Stop the earlier submissions before replacing any files they may still read:

```bash
scancel --clusters=gallade 27605591 27605590
squeue --clusters=gallade -j 27605590,27605591
```

Wait until neither job is running (they may briefly be completing). Copy `bucex-1.9.3-paper-priors-patch.zip` into your existing bucex directory and extract it there:

```bash
unzip -o bucex-1.9.3-paper-priors-patch.zip
```

The patch contains relative paths, updates the scientific configurations, verification, documentation and launcher defaults, and preserves your installed virtual environment and earlier results. The full release archive has also been corrected. Do not apply both or rerun environment setup: the existing editable installation uses the updated source.

```bash
cd -P .
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
export BUCEX_SCHEDULER=slurm
export VSC_ARRAY_LIMIT=16
unset VSC_PARTITION
export BUCEX_VENV="$PWD/bucex_env_gallade_py311_193"
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_193_paperpriors"

bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh
```

Keep this results root for the later paper tier too. A new startup probe checks all configurations and the short fits on Gallade. The old results are not resumed or overwritten.

## Scope

The existing 23-setting sensitivity grid is retained and re-centred at the manuscript SDs. Its numerical reference and all six analyses now match the manuscript prior table. Two broader sensitivity statements in the manuscript still go beyond the existing grid: a combined level–slope grid and variation of the base observation-variance prior IG(2,2). The release already includes individual level/slope changes, global changes, stronger-seasonal controls, seasonal log-scale changes, constant dispersion and GEV shape alternatives. Add the two further checks or adjust those manuscript sentences before describing all checks as completed.
