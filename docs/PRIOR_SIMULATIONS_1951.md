# Joint prior checks in BUCEX 1.9.5.1

This patch adds prior simulation without changing the 1.9.5 posterior model,
MCMC budgets, sensitivity grid or calibration constants. A prior replication
samples the full hierarchy, not a posterior-conditioned trajectory. The default
initialization remains SD 20; the checks expose its consequences rather than
silently replacing it.

## Run without refitting

From this release directory, in the working Python environment:

```bash
export BUCEX_PYTHON="$(command -v python3)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_1951"
bash RUN_PRIOR_SIMULATIONS.sh screen --dry-run
bash RUN_PRIOR_SIMULATIONS.sh screen
bash RUN_PRIOR_SIMULATIONS.sh paper
```

These commands perform **no MCMC** and read no observed temperatures. They use
2,000 and 10,000 joint replications respectively, processed in batches of 250.
All six channels share each sampled innovation hyperparameter; their initial
rates and observation parameters remain private. Only the calendar and prior
specification enter. The 538 historical positions and 120 further seasons are
simulated unconditionally, ending in JJA 2056.

The core suite has 15 settings: the reference; half/double all innovation
hyperprior scales; matched half-t4 and half-Cauchy families with half/base/double
scales; half/double initial-rate SD; quarter seasonal innovation scale; previous
initial-seasonal SD (2.25); and narrow/wide xi priors. These use the existing
experiment declarations. All remaining active non-influence settings can be
checked with `--suite all` (35 settings). A comma-separated list selects settings:

```bash
bash RUN_PRIOR_SIMULATIONS.sh screen --suite reference,previous_initial_season_sd
bash RUN_PRIOR_SIMULATIONS.sh screen --suite all
```

The output root includes a tier. Repeating a matching completed simulation skips
its draws and regenerates the requested comparison. Changing draw count, seed,
configuration or simulator source requires a fresh results root. For a custom
small check, use a separate `BUCEX_RESULTS_ROOT`; do not mix it with the default
screen/paper checks.

## Outputs

Each variant under `results/serra_1951/<tier>/prior_simulations/` contains:

- `resolved_config.json` and `complete.json`: actual settings, seed, software
  version, simulator hash and completion identity.
- `target_summary.csv`: empirical 95% and 99% prior intervals for 10-, 20- and
  30-year component effects, warming rates, future changes, first-cycle ranges,
  tail quantiles, path extrema and nonfinite values.
- `initial_cycle.csv`: first-cycle latent and replicated temperature intervals.
- `ordering_checks.csv`: violations of the summaries' within-block inequalities.
- `target_draws.csv.gz`: all per-replication target values, retained locally.
- `example_paths.csv.gz`: the first six complete replications in generated order.
- `analytic/`: independent analytic calibration and endpoint-simulation tables.

The parent directory contains `comparison.csv`, `manifest.json` and four figures
in PNG and PDF. Figures have no titles and use the manuscript palette. Main
intervals are 95%; the family comparison additionally shows 99% intervals.
Compact result exports include the checks and figures but omit raw
`target_draws.csv.gz`, which remains on the run host.

## Integration

The BIOBOT queue runs the core prior suite once per selected tier before
production fitting. Existing startup smoke tests still run first. Use
`--skip-priors` only when you intentionally want to run the checks separately.
The HPC probe likewise runs the tier's core suite before its dependent array
starts; `BUCEX_SKIP_PRIOR_SIMULATIONS=1` skips that step. The 20-minute probe
allocation is unchanged. Collection exports already-generated prior checks.
No new posterior fits are added: there are still 106 active fits per tier.

## Scientific interpretation

The reference innovation coefficient RMS scales remain 0.01, 0.0001 and 0.01;
the private initial-rate SD is 0.01 per season. Half-t4 is matched by second
moments, half-Cauchy by the 95th percentile of the shared scale. A half-Cauchy
has no finite variance; finite sample quantiles must not be reported as proof
of a finite population SD.

The full prior is not automatically physically plausible because its evolution
coefficients are calibrated. In 10,000 reference replications, TXm's first-year
latent range has median 52.10°C and central 95% interval [14.60, 120.53]°C.
With the existing SD-2.25 initialization sensitivity, these become 5.86°C and
[1.65, 13.56]°C. This diagnostic identifies the initial-cycle scale for further
scientific calibration; it does not select the smaller setting as optimal.

The SD-20 prior is on the three **lag coordinates at the first observation**.
Their chronological cycle is (g1, -g1-g2-g3, g3, g2). Since the record begins in
MAM, JJA has initial seasonal SD sqrt(3)*20, and other seasons SD 20. It is not
an exchangeable four-season prior or an orthonormal-contrast prior.

The audit range [-50, 60]°C is deliberately transparent and illustrative. It is
not a truncation, physical law, prior support restriction or adequacy pass/fail
criterion. The model does not enforce cross-summary ordering, and replicated
violations are retained and counted. Shared shrinkage does not fix this.

No observations, posterior hyperparameters, sorted draws or rejection of
unusual datasets are used to improve the appearance of the prior checks.
