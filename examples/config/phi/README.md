# Reading the phi JSON files

The files in this directory are complete run specifications for examples 10
and 11. You should be able to understand and change a run without searching
through the Python script.

JSON itself does not permit `//` or `#` comments. Each file therefore uses
ordinary `_comment` fields. They keep the file valid JSON, and bucex ignores
them. You may leave, edit, or remove those fields without changing a fit.

## The most important distinction

In `simulation_linear.json`, `linear` describes the **model being fitted**, not
necessarily the simulated truth:

```json
"simulation": {
  "phi": {"mode": "stationary", "reference_sigma": 1.5}
},
"model": {
  "phi": "linear"
}
```

Thus the supplied file simulates stationary scale and asks whether a fitted
linear log-scale model invents a change. All four supplied simulation files use
the same stationary truth and fit a different scale model. This makes them a
controlled sensitivity comparison.

To simulate and fit a genuinely linear scale, copy the file and change only:

```json
"simulation": {
  "phi": {
    "mode": "linear",
    "reference_sigma": 1.5,
    "linear_change": 0.35
  }
},
"model": {"phi": "linear"}
```

`linear_change = 0.35` means
`phi_last - phi_first = 0.35`, so the scale ratio is
`sigma_last / sigma_first = exp(0.35)`, approximately 1.42.

## Which structure does the location follow?

There are again two answers: the data-generating location truth and the fitted
location model.

### Simulated location truth

The complete truth is under `simulation.location`:

```json
"location": {
  "trend_component": "local_linear_trend",
  "level_mode": "dynamic",
  "trend_mode": "dynamic",
  "initial_level": 25.0,
  "initial_slope": 0.002,
  "level_innovation_sd": 0.02,
  "trend_innovation_sd": 0.0005,
  "seasonal_component": "dummy",
  "seasonal_mode": "dynamic",
  "seasonal_amplitude": 0.25,
  "seasonal_innovation_sd": 0.03
}
```

In words, the simulation uses

```text
mu_t       = level_t + seasonal_t
level_t    = level_(t-1) + slope_(t-1) + level innovation
slope_t    = slope_(t-1) + slope innovation
seasonal_t = sum-to-zero dummy seasonal recursion + seasonal innovation
```

With the supplied `dynamic` modes, all three innovations are active. Change a
mode to `static` to set that component's innovation variance to zero. Set
`trend_mode` or `seasonal_mode` to `off` to omit that state from the simulated
truth.

### Fitted location structure

The model being fitted is under `model.location`:

```json
"location": {
  "structure": "ssvs",
  "trend_component": "local_linear_trend",
  "level_mode": "dynamic",
  "trend_mode": "dynamic",
  "seasonal_component": "dummy",
  "seasonal_mode": "dynamic"
}
```

Here `dynamic` defines the **largest candidate model**, not the structure that
must be selected in every posterior draw. `structure: "ssvs"` means the sampler
selects among location structures:

| Location process | Candidate posterior states |
|---|---|
| level | fixed or dynamic |
| trend/slope | zero, fixed, or dynamic |
| seasonality | zero, fixed, or dynamic |

Their prior probabilities are the readable fields:

```json
"level_dynamic_probability": 0.5,
"trend_probabilities": [0.333333333333, 0.333333333333, 0.333333333334],
"season_probabilities": [0.0, 0.5, 0.5]
```

Both arrays are ordered `[zero, fixed, dynamic]`. In this example, seasonal
`zero` has probability 0, so the fitted season is either fixed or dynamic.
The posterior location choice is written to `tables/<case>/selection.csv`,
`models.csv`, and `switching.csv`.

Location SSVS and scale SSVS are independent. `model.location.structure`
controls location selection; `model.phi: "ssvs"` controls selection among
stationary, linear, and random-walk log scale.

## Field guide

### `simulation` (simulation files only)

| Field | Meaning |
|---|---|
| `n_time` | Number of simulated observations. |
| `period` | Seasonal period; 4 in the supplied quarterly-style experiment. |
| `xi` | True GEV shape. |
| `location` | Complete location truth described above. |
| `phi.mode` | True scale process: `stationary`, `linear`, or `rw`. |
| `phi.reference_sigma` | Reference scale; its logarithm is the centered/reference phi. |
| `phi.linear_change` | Complete first-to-last change in log scale. Active only for linear truth. |
| `phi.rw_sd` | Per-step random-walk innovation SD in log scale. Active only for RW truth. |
| `seed` | Data-simulation seed. |

Inactive phi values stay in the file on purpose, so changing the truth requires
editing only `phi.mode`.

### `data` (Uccle files only)

| Field | Meaning |
|---|---|
| `data_dir` | `null` uses the data shipped with bucex; otherwise use a directory path. |
| `start`, `end` | Inclusive date window; `null` end means the latest available observation. |
| `series` | One or more of `TXx`, `TXn`, `TNx`, and `TNn`. Supplied HPC files select one. |
| `period` | Monthly seasonal period, normally 12. |

### `model`

| Field | Meaning |
|---|---|
| `location.structure` | `ssvs` in these focused examples. |
| `location.trend_component` | `local_linear_trend`. |
| `location.level_mode`, `trend_mode` | Candidate-envelope modes passed to `bx.LocalLinearTrend`. |
| `location.seasonal_component` | `dummy`. |
| `location.seasonal_mode` | Candidate-envelope mode passed to `bx.DummySeasonal`. |
| `phi` | Fitted log-scale model: `stationary`, `linear`, `rw`, or `ssvs`. |

The period comes from `simulation.period` or `data.period`; it is not duplicated
inside `model.location`.

### `priors`

The initial location priors are `alpha_*`, `beta_*`, and
`seasonal_initial_sd`. In a simulation file,
`alpha_mean: "simulated_series_median"` computes the prior center from the
simulated response. In a Uccle file,
`alpha_mean: "transformed_series_median"` accounts for the lower-tail sign
transformation of minima.

`sigma2` is the inverse-gamma scale prior. `xi_bounds` supplies both the GEV
support bound and the uniform shape prior. `innovation_slab_sd` contains the
slab scales for the location innovations. The three structural probability
fields control location SSVS as described above.

The scale-process hyperparameters are grouped under `priors.phi`:

| Field | Meaning |
|---|---|
| `linear.mean`, `linear.sd` | Normal prior for the complete linear log-scale change. |
| `rw_variance.a`, `rw_variance.b` | Inverse-gamma prior for RW innovation variance. |
| `model_probabilities` | Prior probabilities for stationary/linear/RW, used only by phi SSVS. |

### `mcmc` and `inference`

`mcmc.draws` is the retained number **per chain**, after
`mcmc.warmup`. The supplied production setting is 1,000 retained draws, 1,000
warmup iterations, and four chains. `thin` retains every nth post-warmup draw.

`inference.engine`, `parameterization`, and `asis` choose the inference path.
`inference.laplace` tunes the location-state proposal.
`inference.phi` tunes only the conditional log-scale proposal or smoother.
These are computational settings; they do not redefine the scientific model.

### `figures`, `runtime`, and `output`

Tables are always produced. `figures.enabled` turns figure creation on or off;
`formats` can contain `png`, `pdf`, or both. The credible interval, number of
posterior-predictive draws, focus phase/month, forecast horizon, and displayed
history are all explicit in this block.

The parallel-chain runner temporarily changes only `runtime.chain_only`,
`runtime.combine_runs`, one-chain execution, chain seed, and the collision-safe
run ID. It does not change draws, warmup, priors, data, or scientific model
settings. Normal users should leave `runtime` unchanged.

Leave `output.run_id` as `null` for an automatic unique directory.
`overwrite: false` protects existing results.

## Complete outputs

Simulation results use this layout:

```text
results/10_simulation_phi/<run>/
  run_config.json
  simulations/phi_<model>.csv
  simulations/phi_<model>.json
  fits/phi_<model>/combined.bucex
  tables/phi_<model>/
    parameters.csv diagnostics.csv algorithm.csv
    trajectory.csv selection.csv models.csv switching.csv
    posterior_predictive.csv forecast.csv forecast_phase_01.csv
    forecast_level.csv phi.csv phi_models.csv summary.json
  figures/phi_<model>/
    trajectory.* trajectory_phase_01.* posterior_predictive.*
    forecast.* forecast_phase_01.* forecast_level.*
    level.* level_no_observations.* slope.* selection.*
    process_sd.* gev.* season.* phi.*
```

Uccle uses the same core table and figure names under
`tables/<series>/` and `figures/<series>/`, with the fit at
`fits/<series>/combined.bucex`. It additionally writes the aggregate tables
`selection_all.csv`, `laplace_mh_diagnostics.csv`, and
`phi_summary_all.csv`.

## Run a file

```bash
python examples/10_simulation_phi.py \
  --config examples/config/phi/simulation_linear.json

python examples/11_uccle_phi.py \
  --config examples/config/phi/uccle/01_txx_linear.json
```

For PBS and all batch commands, see `../../../docs/HPC.md` and
`../../../docs/HPC_ALL_COMMANDS.md`.
