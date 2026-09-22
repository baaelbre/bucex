# Prior assessment: current and earlier workflows

The current all-six-response hierarchy workflow is documented in
[START_HERE](../START_HERE.md) and [SHARED_SHRINKAGE.md](SHARED_SHRINKAGE.md).
The original single-response 1.7.x assessment below remains available with
`config/priors/pilot.json`; it is not the default recommendation for 1.8.0.

# Parallel chains and focused prior assessment

This workflow asks which of a few declared innovation priors gives useful
smoothness while retaining predictive accuracy and uncertainty. It runs no
simulation study and never automatically selects a prior.

## Parallel execution in the public API

```python
import bucex as bx

def main():
    y = bx.load_uccle_multiseries(series=["TNm"], end="2026-08-01")["TNm"]
    model = bx.Model(
        bx.Gaussian(scale=bx.SeasonalScale(12, .3)),
        parameters={"mu": bx.Latent([bx.LocalLinearTrend(), bx.DummySeasonal(12)])},
    )
    priors = bx.fs_priors("gaussian", period=12, innovation="normal",
        innovation_median={"level": .01, "trend": .00005, "season": .02})
    fit = bx.fit(y, model, priors=priors, parameterization="fs", asis=False,
        mcmc=bx.MCMC(chains=4, chain_workers=4, warmup=500, draws=500, seed=173))
    print(fit.sampler_diagnostics["execution"])
    print(fit.diagnostics()["parameters"])

if __name__ == "__main__":
    main()
```

The research JSON contains the full paper prior; this short example illustrates
the general API. `chain_workers` is a positive integer capped by chain count.
Its Python default is one. The research base JSON now requests four workers.
One process runs one whole chain at a time. A joint/copula chain still contains
all six series and their joint updates. Hierarchical `channel_workers` is a
separate within-chain setting; normally leave it at one when using parallel
chains.

The executor uses `spawn` on all platforms. Custom scripts need the main guard
shown above; provided drivers already have it. Use those from a terminal if an
interactive IDE's process launcher is problematic. `chain_workers=1` retains
the same API and posterior target. Worker errors raise without returning a
partial fit or silently switching to serial execution.

Seed streams and result order do not depend on worker count. This is tested
for scalar Gaussian/GEV FS, monthly scales, mixed copula, hierarchical,
shared-state and disturbance backends. Saved archives preserve chain identity
and execution metadata, including complete seed states. Reproducibility is
within a fixed numerical environment; bitwise equality across BLAS, NumPy or
platform versions is not promised.

Numerical thread pools are limited to one thread per chain. Parallel execution
reduces elapsed time, not the iterations needed for mixing. Workers and result
assembly require additional RAM, including transfer copies. Speedup depends
on workload and available CPU cores. Inspect `preflight` for state storage,
which is a lower bound on total memory.

## One candidate list, two assessments

`config/priors/pilot.json` inherits the monthly-scale revision model. It uses
TNm through August 2026 with the same normal FS priors and fixed repeating
monthly observation-scale profile. The profile is estimated; it does not
change across years. The three candidates have innovation SD prior medians:

| Candidate | Level | Slope | Seasonality |
|---|---:|---:|---:|
| `normal_reference` | .010 | .000050 | .020 |
| `level_half` | .005 | .000050 | .020 |
| `level_slope_half` | .005 | .000025 | .020 |

A signed-normal coefficient prior induces a half-normal physical SD prior.
For a declared median `m`, the signed-normal prior SD is `m / 0.67448975`.
Initial-slope SD .0025 per month, initial seasonality, observation-scale and
shape priors remain unchanged. GEV shape is constant in time with its declared
bounded normal prior. There is no SSVS or ASIS in this workflow.

The second candidate isolates level shrinkage. The third checks the allocation
of variation between level and slope by also shrinking slope innovations.
After this comparison, a seasonal-prior check can be added as one JSON
candidate. For example, add `"multipliers": {"level": 0.5, "trend": 0.5,
"season": 0.5}` and compare it with `level_slope_half`. Multipliers are applied
to the base medians once, not sequentially across candidates.

Each candidate is refitted to historical training prefixes ending in December
2000, 2010 and 2020. Each fit forecasts the next 60 months without updating on
held-out observations. Those observations enter only scoring and calibration.
The full-record fit is a separate sensitivity analysis; it is not reused to
make past forecasts. Declared priors stay fixed across forecast folds.

Four chains each use 500 warmup and 500 retained iterations for screening.
Convergence thresholds stay at R-hat 1.01, bulk/tail ESS 400 and four chains.
Short pilots may fail those thresholds. Inspect initial slope, innovation and
scientific-target diagnostics before relying on a comparison. No convergence
threshold is relaxed to obtain a positive assessment.

## Stages and files

Start with the commands in [START_HERE](../START_HERE.md). `--stage plan` prints
resolved dates and fit counts without MCMC. `--stage all` runs sensitivity and
prediction. Alternatively use `--stage sensitivity`, then `--run PATH --stage
predictive`. Both stages use the same saved candidate list. `--run PATH --stage
report` rebuilds CSVs and figures without inference.

An existing run keeps its saved settings and completed stages. Changed priors,
data or budgets require a new run. An interrupted stage retains partial files
and is not silently mixed with a rerun. One process pool is active at a time:
series, candidates and forecast folds run sequentially around parallel chains.

The pilot saves no large fit archives by default. It retains configurations,
declared priors, posterior intervals, unthinned parameter/target trace CSVs,
level/slope/risk paths, held-out forecast bands and observations, and numerical
flags. Set `save_fits: true` for full-record archives; set
`validation.save_fits: true` for historical archives. Small traces cannot
restart the sampler or reconstruct arbitrary latent path functionals.

`config/priors/confirm.json` is an optional six-origin design using
1,000+1,000 iterations. It runs only if explicitly selected. Narrow it to the
relevant candidates with `--variants`, retaining the declared baseline.

## Reusable assessment API

```python
comparison = bx.compare_innovation_priors(fit, priors, size=20000, level=.95, seed=173)
updating = bx.innovation_prior_diagnostics(comparison)

# Tidy case-level losses labelled by variant, channel, origin, time and horizon.
paired = bx.compare_predictive_scores(scores, baseline="normal_reference")

report = bx.SensitivityReport(
    posterior_runs={
        "normal_reference": {"TNm": "run/sensitivity/normal_reference/TNm"},
        "level_half": {"TNm": "run/sensitivity/level_half/TNm"},
    },
    predictive_runs={
        "normal_reference": {"TNm": "run/predictive/normal_reference/TNm"},
        "level_half": {"TNm": "run/predictive/level_half/TNm"},
    },
    baseline="normal_reference",
)
report.save("run/comparison")
```

Statistical summaries, comparison and plotting belong in BUCEX. The research
driver owns data selection, candidate declarations and stage sequence. PNGs
use the existing manuscript palette and typography. Baseline and candidate
forecast cases are matched explicitly, with missing/duplicate cases rejected.

## Interpretation

Start with `convergence.csv`, then assess `prior_updates.csv` together with
`scientific_targets.csv`. A posterior-to-prior interval-width ratio below one
describes contraction; the median shift describes displacement. Neither is an
objective to maximize. Learning can concentrate a posterior around the prior
centre, and a shifted posterior can still be strongly prior-sensitive.
Continuous SD intervals above zero are not inclusion probabilities or tests
for dynamic structure. Period slope differences describe changes in average
latent slope; they do not establish sustained acceleration by themselves.

All BUCEX predictive scores are losses: lower is better. Paired improvement is
reference minus candidate. Inspect CRPS, negative log predictive density,
monthly coverage and held-out PIT together, including horizons 1–12 versus
13 onward. Nonfinite predictive scores are flagged, never silently dropped.
Monthly coverage tables retain counts; few observations cannot precisely
assess the far tails.

Three origins receive descriptive comparisons without a bootstrap interval.
With at least six origins, the general comparison supplies whole-origin
bootstrap intervals assuming approximately independent forecast blocks. These
are exploratory sampling intervals, not posterior intervals or a correction
for tuning priors. Similar noisy-observation predictions can leave the latent
level–slope decomposition weakly identified.

Prefer stronger regularization when predictions and coverage remain
comparable. Retain the nearby-prior sensitivity of warming, slopes and risks.
Final credible intervals condition on the selected prior; choosing it with
these observations is data-informed calibration. The comparison reports do
not select a winner or label a model publication-ready.
