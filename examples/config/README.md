# JSON settings for the examples

The selected JSON is the authoritative run specification. It contains data or
simulation choices, model structure, priors, MCMC controls, inference controls,
figures, runtime behavior, and output policy. The numbered Python examples read
these values directly with `bx.load_config`.

## Simulation configurations

`simulation.json` selects all six structural truths. The `simulations/`
directory contains one complete input per truth:

- `01_stationary.json`;
- `02_linear_trend.json`;
- `03_random_walk.json`;
- `04_local_linear_trend.json`;
- `05_changing_seasonality.json`;
- `06_llt_fixed_seasonality.json`.

All six production files use 1,000 warmup iterations, 1,000 retained draws,
and four independent chains. Their scenario-specific simulation and prior
values remain explicit in each file.

## Uccle configurations

`uccle.json` selects all four temperature-extreme series. For independent
series jobs, use:

- `uccle/01_txx.json`;
- `uccle/02_txn.json`;
- `uccle/03_tnx.json`;
- `uccle/04_tnn.json`.

The matching narrower-slab sensitivity files are:

- `uccle/01_txx_narrow.json`;
- `uccle/02_txn_narrow.json`;
- `uccle/03_tnx_narrow.json`;
- `uccle/04_tnn_narrow.json`.

Within each group, the four files differ only in `data.series`. A narrow file
differs from its primary counterpart only in the three values under
`priors.innovation_slab_sd`: level `0.01`, trend `0.000025`, and season
`0.01`. Every file records the complete model, priors, 1,000 warmup
iterations, 1,000 retained draws, and four chains. The same file is accepted
by examples 05, 06, and 09; the numbered example selects Laplace, PGAS, or
Laplace-MH.

## Log-scale configurations

`phi/` contains the 1.4.1 sensitivity inputs for
`phi_t = log(sigma_t)`:

- `phi/simulation_stationary.json`;
- `phi/simulation_linear.json`;
- `phi/simulation_rw.json`;
- `phi/simulation_ssvs.json`;
- `phi/uccle/*.json`: four scale models for each of TXx, TXn, TNx, and TNn.

The fitted scale choice is the single readable field `model.phi`. The fitted
location choice is now equally explicit under `model.location`: these examples
use structural SSVS over a local-linear trend and dummy seasonality. Simulation
files separately record the known data-generating location under
`simulation.location`, so truth and fitted model cannot be confused.

All phi files contain valid `_comment` fields and retain the complete
`priors.phi` block, so switching models never hides inactive hyperparameters.
Linear, RW-variance, and model-probability settings can be edited directly.
Examples 10 and 11 read every scientific, computational, reporting, runtime,
and output value from these JSONs. See [`phi/README.md`](phi/README.md) for a
field-by-field guide, the location equations, and the complete output tree.

## Selecting a JSON

Every example exposes `DEFAULT_CONFIG_FILE` near its top. Change that path for
an IDE run, or use `--config` without editing Python:

```bash
python examples/03_simulation_laplace.py \
  --config examples/config/simulations/01_stationary.json

python examples/05_uccle_laplace.py \
  --config examples/config/uccle/01_txx.json

python examples/11_uccle_phi.py \
  --config examples/config/phi/uccle/01_txx_ssvs.json
```

For parallel local chains, use the matching Bash runner:

```bash
bash bash_scripts/run_05_uccle_laplace.sh \
  examples/config/uccle/01_txx.json 4
```

For PBS, pass exactly the same file:

```bash
qsub -v CONFIG=examples/config/uccle/01_txx.json \
  job_scripts/submit_05_uccle_laplace.pbs

qsub -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json \
  job_scripts/submit_11_uccle_phi.pbs
```

There are no environment-variable overrides of scientific settings. The
operational runner creates temporary one-chain copies for parallel execution,
but never edits the source JSON and never changes draws or warmup. Leave
`output.run_id` as `null` for collision-safe automatic IDs; set a manual value
only when it is unique to that submission.
