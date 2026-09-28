# Supplementary checks for 1.9.4

See `SENSITIVITY_GRID_194.csv` for the 49 experiment settings and the commands
file for submission. The structural grid is matched across fixed, independent
mixture and shared specifications. Hyperprior-width checks are matched between
the two mixtures; additional shape, observation-scale and seasonality checks
use the shared reference. Six influence fits omit one summary at a time.

Compare recent levels/rates, 95% intervals, forecasts, finite-threshold risks,
and physically expressed innovation magnitudes. Review sampler diagnostics
first. No model is automatically selected by an in-sample fit statistic or a
preferred terminal warming rate.
