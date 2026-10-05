# Native plotting

`bx.plot(fit, ...)`, `fit.plot(...)`, `fit['TXx'].plot(...)` and
`forecast.plot(...)` use the same handlers in `bucex.plots`. They return a
Matplotlib Axes, accept an existing `ax`, use posterior means and pointwise
central intervals, and never set a title. Matplotlib is imported lazily.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
bx.plot(fit, channel='TXx', type='slope', ax=ax, interval=.95)
ax.set_ylabel('Rate (°C per decade)')
fig.savefig('rate.pdf', bbox_inches='tight')
```

| Type | Additional arguments / meaning |
|---|---|
| `level`, `location`, `seasonal` | Named posterior/predictive paths |
| `slope` | Per decade by default; override `rate_scale` explicitly |
| `component` | `component='regression.wind'`, `'cycle'` or any named output |
| `seasonal_cycle` | Initial/final conditional seasonal patterns; `phase_labels` |
| `normal_qq`, `pit`, `acf` | Conditional fitted-score diagnostics; `max_lag` for ACF |
| `posterior`, `trace`, `prior_posterior` | `parameter='xi'`, `'variance.level'`, `'tau.level'`, etc. |
| `observation_scale` | Phase-specific scale means and intervals |
| `risk` | `threshold=35`, `tail='upper'` / `'lower'` |
| `risk_curve` | `thresholds=grid`, `time=index`, `tail=...` |
| `return_level` | Draw-wise conditional quantiles; `years=20`, `tail=...` |
| `forecast` | Predictive mean/band and location; optional `history` and `observed` |

`phase` on fitted trajectory/risk/return-level plots selects a zero-based
phase relative to the first observation. Use calendar filtering in the
research layer for forecasts. The compatibility alias `type='cycle'` means
a seasonal-cycle comparison; a stochastic cycle is
`type='component', component='cycle'`.

Prior/posterior plots of shared scales require a full `FitResult`, so the
hierarchy is retained when prior draws are generated. `prior=` can supply
explicit prior samples. `path='figure.png'` saves the figure, with `dpi=180`
by default. Numeric arrays are also directly accessible for specialized
figures. Observation prediction intervals and credible intervals for a
conditional risk are different objects; the relevant handlers preserve that
distinction.

To add a new plot, use `bx.register_plot(name, handler)`. A handler receives
`(channel_or_prediction, ax, **context)`, including `fit`, `name`, `type`,
`color`, `interval`, `label` and user keywords. Duplicate registrations are
rejected unless `replace=True`. Styling or custom panel arrangements do not
need to alter the inference layer.
