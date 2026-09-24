# Seasonal Uccle analysis

This workflow constructs one mean or extreme per complete DJF/MAM/JJA/SON
block. Seasonal means use all daily observations and seasonal extrema use one
block maximum or minimum (`r=1`). The reference contains 538 seasons from MAM
1892 through JJA 2026.

BUCEX 1.8.7 reproduces the scientific settings of
`uccle_copula_20260923T222238_406159Z.zip` while retaining the corrected
dummy-seasonal initialization introduced in 1.8.6.1. The exact archived MCMC
budget is in `config/reference_20260923.json`; `config/final.json` changes only
the final sampling budget to 3,000 warm-up and 8,000 retained draws per chain.

The shortest complete path is:

```bash
python -m research.seasonal.prepare --config research/seasonal/config/final.json --output results/serra_187_seasonal_data
python -m research.seasonal.preflight --config research/seasonal/config/final.json --output results/serra_187_final_plan
python -u -m research.seasonal.fit --config research/seasonal/config/final.json
python -m research.seasonal.check_final --run PATH_PRINTED_BY_FIT
python -m research.seasonal.manuscript_figures --run PATH_PRINTED_BY_FIT --output results/serra_187_manuscript_figures
```

See [FINAL_RUN.md](../../FINAL_RUN.md) for the prior/adequacy studies,
pre-2019 record-event refit, seven-origin validation, matched monthly-block
sensitivity and the complete figure command. Short smoke or grid runs are
computational checks, not manuscript evidence.
