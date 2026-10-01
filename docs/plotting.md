# Plotting and saved results

`bx.plot(object, type=..., ax=...)` and `object.plot(...)` return a Matplotlib
Axes. Matplotlib is an optional dependency, imported only when plotting.
Plots have no automatic titles; a caller can add panel annotations or captions.
Mean curves and pointwise central 95% intervals are the defaults. Override
`interval`, `color`, `label`, `ax`, `path` and `dpi` as needed.

| Type | Object | Required/important arguments |
|---|---|---|
| `level`, `slope`, `location`, `seasonal` | fit, channel, prediction | `channel` for a multiseries object; optional `phase` for fitted paths |
| `cycle` | fit or channel | initial and final latent seasonal cycles; optional `phase_labels` |
| `normal_qq`, `pit`, `acf` | fit or channel | conditional in-sample score diagnostics |
| `trace`, `posterior`, `prior_posterior` | fit or channel | `parameter="variance.level"`, `"initial_slope"`, `"sigma"`, `"xi"`, etc. |
| `prior_posterior` | full fit | `parameter="tau.level"` for a pooled scale |
| `risk` | fit, channel, prediction | `threshold`, `tail="upper"` or `"lower"` |
| `return_level` | fit, channel, prediction | `years=20`, `tail` |
| `forecast` | prediction | optional `history=fit`, `observed=heldout` |

For pooled innovation prior/posterior plots pass the **full fit**, so its shared
hyperprior is included. A detached channel only knows its private prior.
`prior_posterior` estimates the marginal prior density from reproducible prior
samples. Its density curve is a visualization, not an analytic prior formula.

`location` denotes the observation location parameter, not universally the
mean of a GEV. Forecast figures show a predictive interval, the simulated
predictive mean, and a dashed latent location mean. The GEV expectation is not
finite when xi >= 1; the API does not relabel its location as an expectation.

`phase=0` means the first observed phase. In the Uccle seasonal analysis,
phase order is **MAM, JJA, SON, DJF**, because the incomplete first winter is
omitted. `phase=1` therefore selects summer. This avoids a hidden DJF-first
rotation. Seasonal forecast figures select calendar months explicitly.

Return-level plots use a fixed season's one-opportunity-per-year interpretation:
upper p=1-1/r, lower p=1/r. They are not annual maxima over all four seasons.
For risks over a window, `bx.window_risk` takes products within each simulated
path, then the caller averages. `bx.block_extremes` aggregates predictive
maxima/minima over explicitly selected complete groups of blocks. It never
sorts or discards draws to repair ordering relationships among responses.

```python
import matplotlib.pyplot as plt
import bucex as bx

fit = bx.load("results/main/paper/fit.bucex")
fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
bx.plot(fit, type="level", channel="TXx", ax=axes[0])
bx.plot(fit, type="slope", channel="TXx", ax=axes[1])
fig.savefig("results/level_rate.pdf")
```

Fit archives contain versioned JSON, compressed numerical arrays, a checksum
and a fingerprint of the model/data/calendar. They do not contain pickle or
code. `.save()` writes atomically. Reloading does not refit. Forecasts are
reproducible from a saved fit when their horizon, draw count and seed agree.
