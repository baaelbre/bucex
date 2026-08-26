# Bash, PBS, and parallel-chain contract

Each numbered example has one small Bash runner and one matching PBS file:

```text
bash bash_scripts/run_<example>.sh [CONFIG] [MAX_WORKERS]
qsub -v CONFIG=<path> job_scripts/submit_<example>.pbs
```

The Bash runner locates Python, sets single-threaded BLAS variables, and calls
`job_scripts/run_parallel_chains.py`. The PBS file requests walltime, cores,
and memory and redirects the main log. Neither layer defines scientific or
MCMC settings.

For examples 03–12, `mcmc.chains > 1` activates process-level chain
parallelism. The runner:

1. reads the selected source JSON;
2. assigns a collision-safe automatic run ID when `output.run_id` is null;
3. writes one temporary JSON copy per chain;
4. keeps draws and warmup unchanged, sets `chains` to 1, and offsets the seed;
5. starts at most `MAX_WORKERS` independent Python processes;
6. waits until every chain succeeds;
7. asks the same numbered example to combine the fits and create tables and
   figures.

Temporary JSONs contain operational `_runner` provenance. Each
`run_config.json` stores the exact effective settings, while fit metadata
records the original selected JSON instead of the temporary path. Temporary
copies are removed after the job. The source JSON is never modified.

When `output.run_id` is null, an HPC run ID has the form:

```text
YYYYMMDD_HHMMSS_<config-name>_<PBS-job-id>
```

Local Bash runs use the process ID instead. This prevents simultaneous series
or scenario jobs from colliding even when they begin in the same second.

The requested chain count may not exceed the worker count. For the supplied
production files, use four chains and `nodes=1:ppn=4`. Parallel-chain stdout is
written to:

```text
logs/<script>_<run-id>_chainXX.log
```

Six primary Uccle submissions can run concurrently. Two select `TXm` or `TNm`
for exact Gaussian FFBS; four select `TXx`, `TXn`, `TNx`, or `TNn` for exact
stationary-scale GEV Laplace-MH. Each internally runs four chains in parallel,
for up to 24 simultaneous one-core processes when PBS grants all six jobs.

Example 11 also separates the stationary, linear, RW, and SSVS scale models
into independent JSON/job pairs. All 16 series-by-scale jobs are valid in
parallel, for up to 64 one-core chain processes when scheduler policy permits.
