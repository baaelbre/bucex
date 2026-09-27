# BUCEX 1.8.9 manuscript run sheet

All commands run from the unpacked release root. Install dependencies in the
BIOBOT Python environment once and specify an absolute interpreter if needed:

```bash
python3 -m pip install -e '.[plot,test]'
export BUCEX_PYTHON="$(command -v python3)"
python3 -m research.seasonal.jobs --list
```

## Submit on BIOBOT (PBS)

```bash
export BUCEX_MAX_JOBS=8  # increase to 10 only if 40 cores and ~200 GB are available
bash bash_scripts/submit_biobot.sh paper
qstat -u "$USER"
```

The array contains **106 independent one-fit tasks**. Its elements include
posterior sensitivity to hyperprior width, the four physical medians (including
the old initial-slope anchor), GEV shape and bounds, seasonal dispersion and
initial state, seasonal dependence, pre-2019 record risks, two held-out
origins for every prior/structural variant, seven reviewer-comment-5 origins,
and four matched monthly/seasonal forecast origins. The separate long jobs
fit the reference, pre-2019 record, constant copula, independence, and monthly
supplement. The PBS array cap is eight at once by default; every task has its
own result and convergence file. Change the cap according to available
physical cores and memory. PBS logs are under `job_scripts/logs/`.

For a quick numerical screen instead, run
`bash bash_scripts/submit_biobot.sh screen`. These are two-chain,
700-warmup/1,300-retained fits. They are **not** manuscript results. The
four-chain paper array retains each study's declared budget (typically
2,000 warmup and 4,000 retained draws per chain), and the primary reference
retains 3,000/8,000. The preceding reference took roughly ten hours; running
fits concurrently reduces calendar time but cannot make the final reference
finish in one to three hours. Array elements have 12-hour and primary jobs
24-hour walltime requests. Adjust PBS directives to your site limits.

For an individual task in a separate terminal (without PBS):

```bash
python3 -m research.seasonal.jobs --task posterior_physical_sensitivity_double_slope
python3 -m research.seasonal.jobs --group long --task final_reference
```

If PBS is unavailable and you have a reserved compute node, the local runner
uses `BUCEX_MAX_JOBS` simultaneous processes:

```bash
BUCEX_MAX_JOBS=4 bash bash_scripts/run_biobot_local.sh screen
```

## Audit, aggregate, and make figures

```bash
python3 -m research.seasonal.collect_jobs --tier paper --require-complete --figures
```

The collector writes `results/serra_189_parallel/paper/collected/status.json`
even if a task is missing or a convergence check is flagged. It merges
nonoverlapping forecast origins, reports paired forecast cases, aggregates
matched block comparisons, and **refuses a complete-paper gate** if any job
is missing, numerically flagged or a two-chain screen. A failed/interrupted
task leaves `task.json`; inspect its traceback, remove that manifest only,
and rerun the same task. Completed tasks with an unchanged config are skipped.

Retrieve the reference path from
`results/serra_189_parallel/paper/final_reference/task.json` and run:

```bash
RUN=$(python3 -c 'import json; print(json.load(open("results/serra_189_parallel/paper/final_reference/task.json"))["result"])')
python3 -m research.seasonal.check_final --run "$RUN"
python3 -m research.seasonal.dynamic_comparison --run "$RUN" \
  --output results/serra_189_dynamic_comparison
python3 -m research.seasonal.manuscript_figures --run "$RUN" \
  --output results/serra_189_manuscript_figures
```

The full fit forecasts 120 seasonal steps (30 years) from JJA 2026, with
12,000 predictive draws and **95%** posterior/prediction bands. The companion
monthly full run forecasts 360 months with 8,000 draws. The figure script
includes forecast interval-width figures, a six-response scale figure,
observed seasonal records and LOESS overview, level/slope panels,
and PIT/QQ summaries. Check its CLI for optional pre-2019 inputs.

Prior sensitivity is reported with numerical warnings visible. The wider
initial-slope prior implies approximately 0.96°C/decade marginal initial-rate
SD and 2.88°C over 30 years; the reference choice is a modelling decision,
not a result. No posterior, forecast, additive-gap or threshold claim should
be transferred from the earlier 1.8.8 fits to this release.
