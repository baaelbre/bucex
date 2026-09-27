# Seasonal Uccle analysis

This workflow constructs one mean or extreme per complete DJF/MAM/JJA/SON
block. Seasonal means use all daily observations and seasonal extrema use one
block maximum or minimum (`r=1`). The reference contains 538 seasons from MAM
1892 through JJA 2026.

BUCEX 1.8.9 uses the manuscript medians `(0.01, 0.0001, 0.01, 0.01)`
for the four shared components, including a median-absolute initial slope.
The exact archived 2026-09-23 scientific settings remain in
`config/reference_20260923.json`. The current `config/final.json` has the
1.8.9 priors, seasonal initial coefficient SD 20°C, 3,000 warm-up and
8,000 retained draws per chain, then a 30-year forecast with 12,000 draws.

The shortest complete path is:

```bash
python -m research.seasonal.prepare --config research/seasonal/config/final.json --output results/serra_189_seasonal_data
python -m research.seasonal.preflight --config research/seasonal/config/final.json --output results/serra_189_final_plan
python -u -m research.seasonal.fit --config research/seasonal/config/final.json
python -m research.seasonal.check_final --run PATH_PRINTED_BY_FIT
python -m research.seasonal.manuscript_figures --run PATH_PRINTED_BY_FIT --output results/serra_189_manuscript_figures
```

See [FINAL_RUN.md](../../FINAL_RUN.md) for parallel PBS jobs, prior/adequacy studies,
pre-2019 record-event refit, seven-origin validation, matched monthly-block
sensitivity and the complete figure command. Short smoke or grid runs are
computational checks, not manuscript evidence.
