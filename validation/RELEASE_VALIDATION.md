# BUCEX 1.8.4 validation — 2026-09-23

The release was checked from its source tree with the same Python environment used to build the distribution.

| Check | Result |
|---|---|
| Active software suite: `python -m pytest -q` | 272 passed; one data quality warning reports two daily TN > TX pairs already retained in the source record |
| Focused private marginal, seasonal, forecast and archive tests during cleanup | 54 passed; the full suite was rerun after removing the SSVS constructor option |
| All research JSON configurations | 16 monthly and 12 seasonal configurations resolved through inheritance |
| Matched seasonal comparison dry run | Two monthly and two seasonal training windows aligned to common seasonal outcomes |
| Monthly and seasonal preflight | 1,614 monthly observations and 538 complete seasonal blocks, ending with JJA 2026 |
| Package build | `bucex-1.8.4` wheel built with `pip wheel . --no-build-isolation --no-deps` |
| 1.8.3 paper-model archive | A small private two-channel Gaussian/copula fit was saved under 1.8.3, loaded under 1.8.4, and produced finite correlated forecasts and joint scores |
| Legacy archive migration | Continuous Gaussian and GEV fit archives with an unused `ssvs: null` field load and resave; the `ssvs=` prior constructor option is absent |

These checks cover software behavior, configurations and data window calculations. The long paper fits and their convergence, adequacy and predictive comparisons remain separate experiments. The removed latent factor, structural SSVS and evolving-scale routes are not part of this release's tests or public API.
