# bucex 1.4.0 examples

The 12 numbered files are direct uses of the public `bucex` API. Examples 10
and 11 are the focused log-scale additions:

1. `00_uccle_record.py` — Uccle record and LOESS summaries.
2. `01_tail_simulations.py` — matched GEV shape/scale comparisons.
3. `02_structural_simulations.py` — six structural data-generating models.
4. `03_simulation_laplace.py` — structural recovery with Laplace.
5. `04_simulation_pgas.py` — structural recovery with PGAS.
6. `05_uccle_laplace.py` — Uccle Laplace fits.
7. `06_uccle_pgas.py` — Uccle PGAS fits.
8. `07_centered_ig.py` — centered inverse-gamma benchmark.
9. `08_simulation_laplace_mh.py` — exact Laplace-MH simulations.
10. `09_uccle_laplace_mh.py` — exact Laplace-MH Uccle fits.
11. `10_simulation_phi.py` — stationary/linear/RW/SSVS log-scale sensitivity.
12. `11_uccle_phi.py` — the four scale models for each Uccle series.

## Pick a JSON

Every script exposes `DEFAULT_CONFIG_FILE` near the top and accepts
`--config PATH`. Examples 10 and 11 have one `main()` and read the selected
JSON directly; there is no configuration wrapper and no hidden scientific
environment-variable layer.

For an IDE run, edit one line:

```python
DEFAULT_CONFIG_FILE = Path(__file__).parent / "config" / "phi" / "simulation_rw.json"
```

For a terminal run, leave the file untouched:

```bash
python examples/10_simulation_phi.py --config examples/config/phi/simulation_rw.json
```

## Simulation scale sensitivity

The supplied simulation files share a stationary data-generating truth and
fit four alternative scale models. This is a clean sensitivity comparison:

```bash
python examples/10_simulation_phi.py --config examples/config/phi/simulation_stationary.json
python examples/10_simulation_phi.py --config examples/config/phi/simulation_linear.json
python examples/10_simulation_phi.py --config examples/config/phi/simulation_rw.json
python examples/10_simulation_phi.py --config examples/config/phi/simulation_ssvs.json
```

To simulate a changing scale as well, edit `simulation.phi.mode` in a copy of
the JSON to `"linear"` or `"rw"`. The same object contains
`linear_change`, `rw_sd`, and `reference_sigma`; all three remain visible even
when inactive so switching experiments is easy.

## Uccle scale sensitivity

There are four readable JSONs per series under `config/phi/uccle/`:

| Series | Stationary | Linear | Random walk | SSVS |
|---|---|---|---|---|
| TXx | `01_txx_stationary.json` | `01_txx_linear.json` | `01_txx_rw.json` | `01_txx_ssvs.json` |
| TXn | `02_txn_stationary.json` | `02_txn_linear.json` | `02_txn_rw.json` | `02_txn_ssvs.json` |
| TNx | `03_tnx_stationary.json` | `03_tnx_linear.json` | `03_tnx_rw.json` | `03_tnx_ssvs.json` |
| TNn | `04_tnn_stationary.json` | `04_tnn_linear.json` | `04_tnn_rw.json` | `04_tnn_ssvs.json` |

Run one directly:

```bash
python examples/11_uccle_phi.py --config examples/config/phi/uccle/04_tnn_ssvs.json
```

Run all 16 sequentially in a local shell:

```bash
for config in examples/config/phi/uccle/*.json; do
  python examples/11_uccle_phi.py --config "$config"
done
```

For local four-chain process parallelism, use the Bash wrappers:

```bash
bash bash_scripts/run_10_simulation_phi.sh examples/config/phi/simulation_rw.json 4
bash bash_scripts/run_11_uccle_phi.sh examples/config/phi/uccle/01_txx_ssvs.json 4
```

## What to edit

All scale hyperparameters are grouped under `priors.phi`:

```json
"phi": {
  "linear": {"mean": 0.0, "sd": 0.35},
  "rw_variance": {"a": 2.5, "b": 0.0000375},
  "model_probabilities": {"stationary": 0.5, "linear": 0.25, "rw": 0.25}
}
```

- `linear` is the prior for the whole-record change in log scale.
- `rw_variance` is an inverse-gamma prior in the package's `IG(a,b)`
  convention.
- `model_probabilities` are used only by `model.phi: "ssvs"` and must sum to
  one.
- `priors.sigma2` controls the stationary scale or the initial/reference scale
  of a dynamic model.
- `inference.phi` contains proposal and Laplace-smoother tuning only.
- `mcmc` contains draws, warmup, thinning, chain count, and seed.

The production JSONs use 1,000 retained draws, 1,000 warmup iterations, and
four chains. The PBS scripts never replace these values. The chain runner
creates temporary one-chain copies only to execute the four requested chains
concurrently, then combines them.

## Outputs

Outputs are stored below:

```text
results/10_simulation_phi/<run-id>__<signature>/
results/11_uccle_phi/<run-id>__<signature>/
```

Each run saves the exact effective `run_config.json`, one or more `.bucex`
fits, log-scale summaries, and optional figures. Automatic Bash/PBS run IDs
contain the source configuration name plus process or PBS job ID, so parallel
series and model jobs do not collide. Leave `output.run_id` as `null` unless
you deliberately provide a unique name.

See `../docs/HPC.md` for PBS commands and `../docs/LOG_SCALE.md` for the model
and inference details.
