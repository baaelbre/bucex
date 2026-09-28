# BUCEX 1.9.3

Bayesian unobserved-component models for Gaussian temperature summaries and GEV block extremes.

The active seasonal workflow fits **six separate analyses with fixed Normal shrinkage priors**. All six use the fixed Normal prior SDs `(0.01, 0.0001, 0.01, 0.01)` for level, slope and seasonal innovations and initial slope. There are no shrinkage hyperpriors or copula parameters in the current grid. Estimated innovation coefficients and state trajectories remain private to each response.

[Start here](START_HERE.md) · [HPC commands and sensitivity grid](BUCEX-1.9.3-commands.md) · [Verification](validation/RELEASE_VALIDATION_193.md)

One HPC experiment runs six responses in parallel, with two chains each and **12 cores in both screen and paper tiers**. Paper runs use longer chains and stricter diagnostics. The 23 full-record settings test fixed prior SDs directly, together with shape-prior and model alternatives. Root launchers now submit HPC jobs; the local/biobot runner remains available separately.

```bash
# Local installation; for Gallade use SETUP_HPC_ENV.sh as documented.
python -m pip install -e ".[plot,test]"
python -m pytest
```

The innovation prior SDs are tau_alpha=0.01, tau_beta=0.0001 and tau_gamma=0.01; initial-slope SD is sqrt(P_beta0)=0.01. A simple univariate API example is:

```python
import bucex as bx

prior = bx.fs_priors(
    "gaussian", period=4, innovation="normal",
    innovation_sd={"level": .01, "trend": .0001, "season": .01},
    initial_slope=bx.NormalPrior(0., .01),
)
```

These are prior SDs for the signed FS coefficients, not fixed process innovation SDs. Initial-state and observation priors are separate. The full research configuration, including meteorological-season scale effects and initial seasonal SD 20, is in `research/seasonal/config/main.json`.

Full-record fits cover 538 complete seasons and produce 30-year forecasts. Reports retain 95% intervals, 90/95/99% held-out coverage, marginal scores, prior/posterior comparisons and individual numerical diagnostics. Production fits must still be run and assessed.

Gaussian fits use conditional FFBS; GEV fits use Laplace proposals with an exact-likelihood Metropolis–Hastings correction. The library retains optional historical copula and hierarchical-prior APIs for reproducibility; they are not enabled by this release's seasonal experiment grid. See the [seasonal workflow](research/seasonal/README.md) and [prior calibration](docs/PRIOR_CALIBRATION.md).
