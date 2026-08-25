# Bash and PBS runner contract

Every numbered example has exactly one Bash runner and one PBS file. The Bash
contract is deliberately uniform:

```text
bash bash_scripts/run_<example>.sh [CONFIG] [MAX_WORKERS]
```

`CONFIG` defaults to the matching file under `examples/config/`.
`MAX_WORKERS` defaults to `PBS_NP` (or one outside PBS). The runner uses the
active virtual environment, `${HOME}/venvs/bucex_env`, `BUCEX_VENV_DIR`, or an
explicit `BUCEX_PYTHON`; these operational variables never change scientific
settings.

Examples 03–09 support saved one-chain fits. If `mcmc.chains` is larger than
one, `hpc/run_example.py` launches one Python process per chain, gives each a
deterministic seed offset, waits for all chains, and invokes the same example
again with `runtime.combine_runs` to create the final fit, tables, and figures.
The requested chain count cannot exceed the allocated workers.

Parallel-chain stdout is written to
`logs/<script>_<run-id>_chainXX.log`. The main PBS log records launch,
completion, and combine status.

PBS receives a custom configuration with:

```bash
qsub -v CONFIG=examples/config/my_run.json job_scripts/submit_05_uccle_laplace.pbs
```

Edit `#PBS -l walltime`, `nodes=1:ppn=...`, and `mem=...` in the submission
file for the cluster. Set `mcmc.chains` in JSON equal to or below `ppn`.
