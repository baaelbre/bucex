# Seasonal SERRA experiments

Use the ordered commands in [START_HERE](../../START_HERE.md).

- `prepare.py`: complete daily-derived seasons and exploratory figures.
- `preflight.py`: physical prior and parallel fitting plan.
- `fit.py`: the same public BUCEX fitting/report API on quarterly observations.
- `compare.py`: identical historical cutoffs, forecasts scored on common seasonal targets.
- `clusters.py`: raw r-largest ranks and runs-cluster diagnostics; no r>1 model fit.

All scientific choices live in `config/`. `main.json` has period 4 and an
explicit meteorological scale calendar; the monthly workflow has period 12.
DJF is named by its ending year. Summer 2026 remains included.

Shared model declarations and report orchestration remain in `research/serra`;
statistical calculations, seasonal extraction, scoring and diagnostic reports
live in `bucex`. Inference has not been copied into research scripts.
