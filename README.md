# BUCEX 1.9.4

Bayesian unobserved-component models for temperature means and extremes.

This release implements a matched comparison of fixed Normal shrinkage priors,
independent Normal–lognormal mixtures, and shared hierarchical shrinkage.
Initial rates remain separate. Observation residuals are conditionally independent.

Start with **[BUCEX-1.9.4-commands.md](BUCEX-1.9.4-commands.md)** for the screen
commands on Gallade and biobot. The default screen contains 49 experiments
(159 fits). Validation is a separate batch.

- `research/seasonal/config/main.json`: the shared reference.
- `research/seasonal/config/experiments.json`: the complete comparison grid and budgets.
- `docs/SENSITIVITY_GRID_194.csv`: all settings in a table.
- `research/seasonal/jobs.py`: fitting, configuration checks and provenance.
- `research/seasonal/job_plan.py`: scheduler planning using only the standard library.
- `bucex/`: the model, inference, diagnostics and reporting API.

```python
import numpy as np
import bucex as bx

shared = bx.SharedShrinkage.from_sd(
    {"level": 0.01, "slope": 0.0001, "seasonal": 0.01},
    log_sd=np.log(3),
)
private = bx.IndependentShrinkage.from_sd(
    {"level": 0.01, "slope": 0.0001, "seasonal": 0.01},
    log_sd=np.log(3),
)
```

Both use direct conditional Normal SDs. The private specification is for a
single-response model; the shared specification pools regularization across
response-specific innovation coefficients. `from_sd` leaves initial rates
under their separately declared priors. Legacy median-parameterized archives
remain readable.

The package requires Python >= 3.10. For a new workstation environment:

```bash
python -m pip install -e ".[plot,test]"
```

Production inference is not run as part of building a release. Screen results
must be assessed for convergence, prior sensitivity and predictive adequacy.
