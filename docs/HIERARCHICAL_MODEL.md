# Joint hierarchical analysis

A `MultiSeriesModel` contains named `Channel` objects, each with an observation
family and components. Its joint hierarchy shares probabilities or scale
hyperparameters while retaining separate state trajectories and seasonal cycles.

```python
model = bx.MultiSeriesModel((
    bx.Channel("mean", bx.Gaussian(), (
        bx.LocalLinearTrend(), bx.DummySeasonal(period=12),
    )),
    bx.Channel("maximum", bx.GEV(), (
        bx.LocalLinearTrend(), bx.DummySeasonal(period=12),
    )),
))
fit = bx.fit(data, model=model, priors=bx.HierarchicalPrior(pool="selection"),
             engine="laplace_mh", parameterization="fs", mcmc=bx.MCMC())
```

`pool="selection"` learns shared selection probabilities. `pool="slab"` forces
available innovations to be dynamic and learns their scale distribution.
`pool="both"` combines selection and scale pooling. The default hierarchy permits
fixed or dynamic seasonality; seasonal absence is excluded unless requested.

Use `HierarchicalPriors(hierarchy=..., channels=...)` to supply channel-specific
baseline and observation priors. Hierarchical selection/scales replace the
corresponding per-channel SSVS settings. Match those scales explicitly when
comparing independent fits to hierarchical fits.

The hierarchy does not estimate factor loadings or a common path. With six
channels there are only six allocations informing each shared selection
probability. Those probabilities describe this model, not a well-estimated
population frequency of climate mechanisms.

Inspect `fit.component_probabilities(channel=...)`,
`fit.hierarchical_probabilities()`, `fit.channel_eta_draws(...)`, and
`fit.diagnostics()`. `python -m research.serra.tutorials --kind hierarchical` is the runnable tutorial.
