# BUCEX 1.8.7 — final seasonal manuscript run

Version 1.8.7 uses the model, thresholds, contrast periods, seeds and calibrated
hyperparameters of `uccle_copula_20260923T222238_406159Z.zip`. It retains the
1.8.6.1 dummy-seasonal initialization correction. The final configuration
changes only the sampling budget: four chains, 3,000 warm-up and 8,000 retained
iterations per chain.

From the unpacked package root:

```bash
python -m pip install -e ".[plot,test]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
python -m research.seasonal.prepare --config research/seasonal/config/final.json --output results/serra_187_seasonal_data
python -m research.seasonal.preflight --config research/seasonal/config/final.json --output results/serra_187_final_plan
python -u -m research.seasonal.fit --config research/seasonal/config/final.json
```

The fit command prints a timestamped `uccle_copula_...` directory. Do not copy
numbers into the manuscript until the strict gate passes:

```bash
python -m research.seasonal.check_final --run PATH_PRINTED_BY_FIT
python -m research.seasonal.manuscript_figures --run PATH_PRINTED_BY_FIT --output results/serra_187_manuscript_figures
```

The figure command writes stable manuscript stems in PNG and PDF plus a source
checksum manifest. It refuses a run whose `convergence.json` has not passed.
See [FINAL_RUN.md](FINAL_RUN.md) for the sensitivity, held-out validation,
monthly-block comparison and pre-2019 record-event commands.

The analysis contains 538 complete meteorological seasons from MAM 1892
through JJA 2026. January and February 1892 are excluded; JJA 2026 includes
daily observations through 31 August 2026. Seasonal extrema use one extreme
per block (`r=1`).
