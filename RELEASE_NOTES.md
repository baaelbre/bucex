# bucex 1.5.2 release notes

Version 1.5.2 is a reporting release. The scientific models, priors, MCMC
kernels, production settings, and archive schema from 1.5.1 are unchanged.

## Seasonal patterns at selected times

The established `season.*` figure answers: how does the seasonal effect for
each month evolve over the record? The new `seasonal_patterns.*` figure turns
that view around and answers: what does the complete seasonal pattern look
like at selected points in the record?

```python
fit.plot("seasonal_patterns", years=[1892, 2022])
fit.plot("seasonal_patterns", cycles=["first", "last"])
```

Calendar fits put months on the horizontal axis and draw one posterior-median
line per requested complete year. Undated simulations use one-based cycles or
the readable selectors `first`, `middle`, and `last`. Pointwise credible bands
are optional, and simulations may overlay the known seasonal truth. Lower-tail
fits are displayed in their original temperature orientation.

## JSON and example integration

Every maintained seasonal configuration now contains:

```json
"seasonal_patterns": {
  "years": [1892, 2022],
  "cycles": [],
  "show_interval": true
}
```

Uccle configurations default to 1892 versus 2022. Simulation configurations
use empty `years` and `cycles: ["first", "last"]`. An empty applicable list
suppresses the new figure. The band probability remains the existing
`figures.interval_probability` value.

Examples 03, 04, 05, 06, 08, 09, 10, 11, and 12 write the new figure while
retaining all previous tables and figures. This covers simulation and Uccle
fits under Laplace, PGAS, Laplace-MH, dynamic log-scale models, and exact
Gaussian FFBS.

## Replot completed fits

`examples/replot_seasonal_patterns.py` adds the figure to completed runs in a
few seconds without rerunning MCMC. It discovers
`fits/<series>/combined.bucex`, reconstructs the selected posterior seasonal
patterns, and writes the result beside the existing figures. The script
defaults to the requested TNx and TXx run names and to years 1892 and 2022.

A lightweight transfer containing only `figures.zip` is insufficient because
it does not contain posterior draws; execute the script against the complete
result directories on the HPC.

## Compatibility

- `fit.plot("season")` is unchanged.
- Custom 1.5.1 JSONs without `figures.seasonal_patterns` remain valid and
  simply omit the new output.
- Public fitting, prediction, persistence, risk, and log-scale APIs are
  unchanged.
- Safe result archives remain at schema 2.7.0 and earlier supported archives
  remain readable.
