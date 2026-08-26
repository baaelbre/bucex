# Migration to bucex 1.5.2

## From 1.5.1 to 1.5.2

No model, prior, sampler, archive, or existing plotting call changed. Version
1.5.2 adds one optional reporting block to the maintained seasonal JSONs:

```json
"seasonal_patterns": {
  "years": [1892, 2022],
  "cycles": [],
  "show_interval": true
}
```

Uccle fits use `years`; undated simulations use `cycles`, including the named
selectors `"first"`, `"middle"`, and `"last"`. Custom 1.5.1 JSONs without
this block remain valid and simply omit `seasonal_patterns.*`.

The public equivalent is:

```python
fit.plot("seasonal_patterns", years=[1892, 2022])
fit.plot("seasonal_patterns", cycles=["first", "last"])
```

The old `fit.plot("season")` trajectory is unchanged. Existing complete runs
can be replotted from their `combined.bucex` files with
`examples/replot_seasonal_patterns.py`; no refit or archive migration is
required.

## From 1.5.0 to 1.5.1

No public Python API or Uccle configuration changes. The only scientific input
change is `examples/config/tail.json`, because example 01 is now deliberately
stationary.

Replace this 1.5.0 block:

```json
"period": 4,
"initial_level": 25.0,
"level_process_sd": 0.08
```

with:

```json
"frequency": "QS",
"location": {
  "mode": "stationary",
  "value": 25.0
}
```

`period` never affected the tail model; it only described the quarterly date
axis. `level_process_sd` is intentionally gone because a stationary location
has no innovation. Old result files remain readable, but rerunning the 1.5.0
tail JSON with example 01 is rejected instead of silently restoring a random
walk.

Packaging now uses `pyproject.toml` alone. Use a current pip/setuptools as
already documented in the HPC setup.

## From 1.4.1 to 1.5.0

No existing Python call or JSON needs to change. Version 1.5.0 adds a focused
Gaussian Uccle layer for the two monthly-mean series:

```bash
python examples/12_uccle_gaussian.py \
  --config examples/config/uccle_gaussian/01_txm.json
```

Use `02_tnm.json` for TNm. These models use exact FFBS and have no Laplace or
particle settings. The new PBS wrapper is
`job_scripts/submit_12_uccle_gaussian.pbs`.

The 16 phi-sensitivity JSONs now use the primary location slabs
`(0.02, 0.00005, 0.02)`. This is an intentional scientific correction: scale
sensitivity should vary the scale model without simultaneously narrowing the
location prior. Older 1.4.1 files remain valid if reproducing an old fit is the
goal.

## Primary versus sensitivity analyses

The primary Uccle specification is:

- exact Gaussian FFBS for TXm and TNm;
- exact stationary-scale GEV Laplace-MH for TXx, TXn, TNx, and TNn.

Linear, random-walk, and SSVS log-scale GEV models remain optional sensitivity
fits. `bx.GEV()` is unchanged and equivalent to
`bx.GEV(phi="stationary")`.

## Results and parallel chains

Result paths retain the established structure:

```text
results/<script>/<run-id>__<settings-signature>/
  run_config.json
  fits/<series>/combined.bucex
  tables/<series>/
  figures/<series>/
```

Leave `output.run_id` as JSON `null` for a collision-safe automatic ID. The
runner temporarily splits a four-chain request into four one-chain processes,
then combines compatible archives. It never edits the selected source JSON or
changes its draws, warmup, priors, model, or reporting settings.

## Earlier 1.4.x scale API

The scale interface remains:

```python
bx.GEV()
bx.GEV(phi="linear")
bx.GEV(phi="rw")
bx.GEV(phi="ssvs")
```

Use `fit.phi_draws()` and `fit.sigma_draws()` for scale trajectories. Dynamic
scale remains a univariate Fruehwirth-Schnatter model; attempts to put it in a
`MultiSeriesModel` fail explicitly.

## Archive compatibility

Persistence schema remains 2.7.0. Loading is checksum-verified and pickle-free,
and all archive versions supported by 1.4.1 remain readable.
