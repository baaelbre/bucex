# Seasonal research

The seasonal model fits one mean or extreme per complete DJF/MAM/JJA/SON block, using the same model builder and sampler as [monthly research](../monthly/README.md). Season means weight all complete daily observations. The series ends with JJA 2026.

1. `python -m research.seasonal.prepare` audits and derives seasonal blocks.
2. `python -m research.seasonal.preflight` checks units, priors and computation.
3. `python -m research.seasonal.fit --config research/seasonal/config/smoke.json` checks execution.
4. `python -m research.seasonal.fit --config research/seasonal/config/main.json` fits the reference.
5. `python -m research.seasonal.compare --config research/seasonal/config/compare.json` scores matched monthly and seasonal forecasts on the same seasonal outcomes.

[`config/adequacy.json`](config/adequacy.json) compares the reference seasonal observation scales with `constant_dispersion` and fixed location seasonality. Run `research.monthly.prior_assessment --config research/seasonal/config/adequacy.json --stage plan` to inspect those candidates. `clusters.py` diagnoses raw daily ranks and clustering; the fitted likelihood remains one extreme per season.
