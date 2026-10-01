# Independent MCMC chains

Parallelism is over **independent complete chains**. In a pooled model, each
chain updates every channel and its own shared scales. Splitting the channels
across unrelated jobs while keeping a shared-scale prior would target a
different model.

## One machine

```python
import bucex as bx

def main():
    # Define model and y here, or read them from files.
    fit = bx.fit(y, model=model,
                 mcmc=bx.MCMC(chains=4, workers=4, seed=1234))
    fit.save("results/fit.bucex")

if __name__ == "__main__":
    main()
```

The API uses the portable `spawn` process context. Keep worker-importable code
in a script/module, and protect the entry point. Run notebooks with workers=1
or put the fitting call in an importable script. See the
[Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).

Every worker limits numerical-library threads to one. You can also set these
before Python starts:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m research.main --profile paper --workers 4
```

Allocate four CPUs for four pooled chains, not six times four. This release
does not add a nested per-channel process pool. Memory grows with retained
paths; 4 chains × 4,000 draws × 538 blocks × six response paths is much larger
than a smoke run. Estimate memory/runtime on your actual machine before a
large batch; the refactor does not promise the old wall-clock time.

## Separate jobs, one chain per job

Use the **same configuration and master seed**, with distinct chain IDs.
SeedSequence derives the random stream from `(seed, chain_id)`, so the chains
are identical to those from one multi-process call with those IDs.

```bash
python -m research.main --profile paper --chain-id 0 --no-figures
python -m research.main --profile paper --chain-id 1 --no-figures
python -m research.main --profile paper --chain-id 2 --no-figures
python -m research.main --profile paper --chain-id 3 --no-figures

python -m research.combine \
  results/main/paper/chains/chain_0 \
  results/main/paper/chains/chain_1 \
  results/main/paper/chains/chain_2 \
  results/main/paper/chains/chain_3 \
  --output results/main/paper_combined
```

Submit the four commands independently or in a scheduler array. No module
loads or environment paths are embedded in the package. The optional
`local/chain.slurm` template expects an absolute `BUCEX_PYTHON` and an optional
`BUCEX_ENV_SETUP` script. `local/` and all job files are gitignored. The template
uses the site's configured scheduler defaults rather than guessing a partition.

Combination rejects different data, priors/calendars, draw counts, sampler
settings or duplicated random streams. Warmup draws are never concatenated.
This is independent-chain execution, **not checkpoint/resume** of an interrupted
chain; resume is deliberately outside 1.0.0's API.

## Sensitivity jobs

```bash
python -m research.sensitivity --list
python -m research.sensitivity --profile screen --variant 0 --workers 2
```

There are 25 variants indexed **0 through 24**. A scheduler array can supply
one index per task. Each screen task uses two CPUs for its two chains. After
completion:

```bash
python -m research.sensitivity --profile screen --compare
```

## Truly separate response fits

The private-prior workflow can also be split by response:

```bash
python -m research.private --profile screen --channel TXm --workers 2
python -m research.private --profile screen --channel TNm --workers 2
python -m research.private --profile screen --channel TXx --workers 2
python -m research.private --profile screen --channel TXn --workers 2
python -m research.private --profile screen --channel TNx --workers 2
python -m research.private --profile screen --channel TNn --workers 2
```

These six commands may run concurrently using twelve CPUs in total. The
`--channel` option is rejected for pooled configurations. A single call to
`research.private` instead keeps all private channel updates in each chain and
uses only the requested chain workers; it targets the same product posterior.
