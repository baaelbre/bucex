# BUCEX 1.9.5

Bayesian unobserved-component models for environmental averages and extremes.

The paper reference uses **pooled half-normal hierarchical shrinkage** for three
signed innovation coefficients. Each response retains its own latent trajectory.
Initial slopes have separately calibrated fixed normal priors. Observation
residuals are conditionally independent; no copula enters the default study.

Start with **[BUCEX-1.9.5-commands.md](BUCEX-1.9.5-commands.md)**. It gives the
BIOBOT overnight launch, monitoring, resumption, all budgets and the complete
experiment inventory. The combined queue contains 212 fits (106 per tier), with
parallel chains and a common CPU/memory budget. It prioritizes reference fits;
it does not promise that the entire paper grid will finish overnight.

```python
import bucex as bx
shared = bx.SharedShrinkage.half_normal(
    {"level": 0.01, "slope": 0.0001, "seasonal": 0.01}
)
```

For `s[c,j] | tau[c] ~ Normal(0,tau[c]^2)`, the hyperprior is
`tau[c] ~ HalfNormal(A[c])`. The three anchors above are half-normal scales,
not absolute-coefficient medians. The reference initial-rate SD is 0.01 per
seasonal update; initial level and seasonal-coordinate SDs are 20.

Half-t4 and half-Cauchy sensitivity families are supported. Unpooled fixed and
private one-response hierarchies are retained in the optional `deferred` batch.
Legacy lognormal constructors and saved archives preserve their meanings.

- `research/seasonal/config/main.json`: pooled reference.
- `research/seasonal/config/experiments.json`: sensitivity settings and budgets.
- `docs/SENSITIVITY_GRID_195.csv`: 41 seasonal full-record settings.
- `docs/EXPERIMENT_PLAN_195_*.csv`: complete plans for both tiers.
- `research/seasonal/overnight.py`: resumable local queue.
- `research/seasonal/jobs.py`: execution and provenance checks.
- `research/seasonal/finish.py`: collection, review figures and compact exports.
- `validation/RELEASE_VALIDATION_195.md`: implementation checks and their limits.

Python >= 3.10 is required. In an existing environment, the launchers put this
release first on `PYTHONPATH`. For a new environment, install with
`python -m pip install -e '.[plot]'`. Run `python -m pytest` for regression tests.

Scientific quantities are reported with 95% intervals. Forecasts span 30 years;
validation also includes 90%, 95% and 99% coverage. Annual extreme aggregation
uses complete meteorological years and integrates uncertainty within paired
posterior paths. All numerical and prior-sensitivity checks must be assessed
before using a fit for manuscript claims.
