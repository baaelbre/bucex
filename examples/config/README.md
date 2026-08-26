# JSON settings for the examples

The selected JSON is the authoritative run specification. It contains data or
simulation choices, model structure, priors, MCMC controls, inference controls,
figures, runtime behavior, and output policy. The numbered Python examples read
these values directly with `bx.load_config`.

## Tail comparison configuration

`tail.json` drives example 01. Its single `simulation.location` block is
stationary, and all six scenarios reuse that same constant predictor. The
shape group changes only `tail_xi_values`; the scale group changes only
`scale_sigma_values`. The paired seeds give matched probability draws within
each group. No process-innovation SD is present or needed.

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

The recommended primary analysis has six independent series jobs. Monthly
means use exact Gaussian FFBS:

- `uccle_gaussian/01_txm.json`;
- `uccle_gaussian/02_tnm.json`.

Monthly extremes use exact stationary-scale GEV Laplace-MH:

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

The Gaussian files contain the same primary structural slabs and SSVS
probabilities. They additionally contain a readable `sigma2` observation-
variance prior and `inference.engine: "ffbs"`; they do not contain particles
or Laplace proposal controls. Example 12 reads them.

## Log-scale configurations

`phi/` contains the maintained sensitivity inputs for
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
Their location innovation slabs equal the primary GEV values
`(0.02, 0.00005, 0.02)`, so a scale comparison does not also change the
location prior.
Linear, RW-variance, and model-probability settings can be edited directly.
Examples 10 and 11 read every scientific, computational, reporting, runtime,
and output value from these JSONs. See [`phi/README.md`](phi/README.md) for a
field-by-field guide, the location equations, and the complete output tree.

## Seasonal-pattern figures

Every fitted seasonal example reads this reporting block directly from JSON:

```json
"seasonal_patterns": {
  "years": [1892, 2022],
  "cycles": [],
  "show_interval": true
}
```

For Uccle, list any complete calendar years under `years`; the output compares
the full January--December seasonal effect in those years. Use three or more
years if desired, or `[]` to disable the figure. Simulations have no calendar
dates and instead use readable selectors such as
`"cycles": ["first", "middle", "last"]`. Do not fill both arrays. The
pointwise band uses `figures.interval_probability`, and an optional title can
be placed at `figures.titles.seasonal_patterns`.

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

python examples/12_uccle_gaussian.py \
  --config examples/config/uccle_gaussian/01_txm.json
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

qsub -v CONFIG=examples/config/uccle_gaussian/01_txm.json \
  job_scripts/submit_12_uccle_gaussian.pbs
```

There are no environment-variable overrides of scientific settings. The
operational runner creates temporary one-chain copies for parallel execution,
but never edits the source JSON and never changes draws or warmup. Leave
`output.run_id` as `null` for collision-safe automatic IDs; set a manual value
only when it is unique to that submission.
