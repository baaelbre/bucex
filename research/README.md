# Reproducing the paper

Run from the source-release root after `python -m pip install -e '.[research]'`.
This folder uses the public BUCEX API for model construction, fitting, prior
sampling, diagnostics, forecasts, risks and native plotting. It contains no
sampler implementation and is not imported by the installed package.

## Start here

```bash
# Check execution first; these draws are deliberately too short for inference.
python -m research.main --profile smoke --workers 1

# Inspect the complete plan without launching it.
python -m research.paper --profile screen --dry-run

# Run experiments concurrently, with two independent chains per job.
python -m research.paper --profile screen --jobs 4 --workers 2
```

The last command uses up to eight chain workers, plus orchestration/plotting
processes. Increase concurrency only within your CPU and memory allocation.
For a publication run use `--profile paper --jobs 2 --workers 4` as an example
of the same eight-worker budget. The full plan contains five structural
analyses, 23 pooled sensitivities, nine private sensitivities, main/recent
validation, a pre-2019 refit and prior calibration. `--groups` selects any
subset: `analyses sensitivity private_sensitivity validation pre2019 priors`.

The orchestrator saves each command and output to
`results/paper_suite/logs/<profile>/<task>.log`, with a `status.json` recording
successes and failures. It propagates failures instead of reporting a partly
completed suite as successful. Reruns reuse compatible completed fits; an
interrupted chain is restarted, not resumed. Comparison plots are generated
explicitly after the selected fits finish (commands below).

## Five required analyses

| JSON configuration | Blocks | Observation dispersion | Innovation priors |
|---|---|---|---|
| `main.json` | seasons | repeats by season | shared half-normal scales |
| `monthly.json` | months | repeats by month | shared half-normal scales |
| `private.json` | seasons | repeats by season | private fixed Normal SDs |
| `constant_dispersion.json` | seasons | constant per response | shared half-normal scales |
| `monthly_constant_dispersion.json` | months | constant per response | private fixed Normal SDs |

Run them independently:

```bash
python -m research.main --profile screen --workers 2
python -m research.monthly --profile screen --workers 2
python -m research.private --profile screen --workers 2
python -m research.constant_dispersion --profile screen --workers 2
python -m research.monthly_constant_dispersion --profile screen --workers 2
```

The monthly/constant-dispersion/private comparison reproduces that structure
of the original manuscript using the **current Normal priors and corrected
sampler**. It is not a byte-for-byte reconstruction of the original lasso
prior or uncorrected inference procedure.

`main.json` is the reference; other files use `extends` plus nested overrides.
Configurations are validated and their resolved values saved. Unknown keys
and circular inheritance fail. A result directory cannot silently be reused
with a different configuration, input data or sampling profile.

## Reference settings

The seasonal calibration is `(A_level, A_slope, A_seasonal)=(.1,.002,.1)`.
Initial level and seasonal-contrast SDs are 10°C; initial-rate SD is .01°C per
season. The squared geometric-mean observation scale has an IG(2,2) prior;
seasonal log-scale contrast SD is .3 and GEV shape SD is .3. Private fits use
Normal amplitude priors with the same fixed SDs, without private hyperpriors.

| Profile | Chains | Retained per chain | Warmup per chain | Record |
|---|---:|---:|---:|---|
| smoke | 2 | 4 | 2 | First 36 blocks; workflow check only |
| screen | 2 | 1,000 | 1,000 | Full specified record |
| paper | 4 | 4,000 | 2,000 | Full specified record |

All these values are editable in JSON. They are run settings, not convergence
claims or a guarantee of a particular wall-clock time. The pre-2019 and
optional linear-benchmark smoke commands retain their actual training dates
while reducing iterations. Validation smoke uses an early artificial origin.

Monthly calibration uses `.1/sqrt(3)` for level, `.0003832923311016883` for
slope and `.1` for seasonal innovations; initial-rate SD becomes `.01/3`.
The slope value equates integrated slope variance at 30 years using
`S(h)=h*(h-1)*(2*h-1)/6` and `A_beta_monthly=.002*sqrt(S(120)/S(360))`.
Matching at one horizon does not make the entire monthly and seasonal prior
processes identical. Monthly and seasonal extrema also describe different
events: an August threshold probability is not a JJA threshold probability.

## Sensitivity settings

```bash
python -m research.sensitivity --list
python -m research.sensitivity --profile screen --variant 0 --workers 2
python -m research.sensitivity --profile screen --compare
```

`config/sensitivity.json` has **23 settings**, including the reference:

| Calibration or specification | Values |
|---|---|
| Shared level × slope grid | `{.05,.1,.2}` × `{.001,.002,.004}`: nine combinations |
| Seasonal innovation hyperprior SD | `.05`, `.2`, around reference `.1` |
| Initial-rate prior SD | `.005`, `.02`, around `.01` |
| Shape prior SD | `.15`, `.6`, around `.3` |
| Initial level and seasonal-contrast SDs, varied jointly | `5`, `20`, around `10` |
| IG observation-variance scale b, with shape a=2 | `.5`, `8`, around `2` |
| Seasonal log-scale contrast SD | `.15`, `.6`, around `.3` |
| Static seasonal cycle | No seasonal innovations |
| Tight innovation calibration | `(.01,.0001,.01)` |

Indices are zero-based, **0–22**, for a scheduler array. Each setting may run
independently. The comparison command generates separate figure sets by
question, so unrelated prior changes are not overlaid as 23 indistinguishable
curves. It includes levels, rates, seasonal evolution and fixed-threshold
risk. `path_comparison.csv` reports maximum differences between posterior
mean curves with explicit units; rate differences are per decade.
`completion.json` identifies missing settings in a partial comparison.

For the private fixed-Normal comparison, use the same nine level–slope pairs:

```bash
python -m research.sensitivity --config research/config/private_sensitivity.json \
  --profile screen --output results/private_sensitivity --workers 2
python -m research.sensitivity --config research/config/private_sensitivity.json \
  --profile screen --output results/private_sensitivity --compare
```

Here the grid changes the fixed Normal prior SDs directly. It does not change
private hyperpriors. No automatic “optimal” calibration is selected.

## Figures and tables

```bash
# Rebuild figures from stored draws, without rerunning MCMC.
python -m research.figures results/main/screen

# Compare the five independently completed analyses.
python -m research.compare results/main/screen results/private/screen \
  results/monthly/screen results/constant_dispersion/screen \
  results/monthly_constant_dispersion/screen --output results/comparisons
```

When using `research.paper`, prefix these paths with `results/paper_suite/`.
Its sensitivity roots are `results/paper_suite/sensitivity` and
`results/paper_suite/private_sensitivity`; pass those to `--output --compare`.

| Paper material | Output / entry point |
|---|---|
| Seasonal records and centred descriptive LOESS | `seasonal_records_six`, `seasonal_loess_overlay` |
| Levels and warming rates | Native `level`, `slope` figures |
| Initial/final cycle and seasonal histories | `cycle`, `seasonal_evolution` |
| Main QQ; appendix PIT/serial checks | `normal_qq`, `pit`, `acf` |
| Innovation/initial-rate/observation priors and posteriors | `prior_posterior_*`, `shared_scales` |
| Seasonal observation scales | Native `observation_scale` panels |
| Risk above/below fixed thresholds | `threshold_risk`, `risk_probabilities.csv` |
| Conditional seasonal return levels | `return_levels_<r>_year` |
| Thirty-year temperatures, with evolving location | `forecasts_30_year`, `forecasts_relevant_seasons` |
| Analytical calibration illustrated by prior draws | `research.prior_calibration` |
| 2019 forecast and TXx threshold-probability curve | `research.pre2019`: `summer_2019` |
| Historical and recent validation | `research.validation` |
| Calibration comparisons | `research.sensitivity --compare` |

Each fit directory contains `run.json`, `observations.csv`, `fit.bucex`,
`posterior_summary.csv`, `sampler.json`, `posterior_checks.csv` and
`residual_association.csv`. Figure-enabled runs also export risk and forecast
CSVs. The sampler metrics record actual acceptance and proposal diagnostics;
posterior summaries use actual saved draws.

Figures have no automatic titles. They use posterior/predictive means,
pointwise central 95% bands, consistent typography and TX-blue/TN-red styling.
LOESS is purely descriptive. Native `bx.plot` draws model-derived panels;
research code arranges panels and adds application labels. Set `plot.format`
to `pdf` for vector output. The JSON `return_years` list controls return-level
figures, and `risk_thresholds` controls response/season/tail selections.

## Prior calibration and 2019

```bash
python -m research.prior_calibration --profile paper
python -m research.pre2019 --profile screen --workers 2
```

The prior-calibration figure uses 50,000 joint parameter replications in the
paper profile, with seed 19830. Shared scales and response amplitudes are drawn
through `bx.prior_samples`; exact Gaussian horizon covariances give the level,
integrated-slope, seasonal, initial-rate and combined displacements at 10 and
30 years. These are prior implications for latent changes, excluding
observation variability. The exported SDs are not assumed to be 68% intervals:
the marginal distributions are scale mixtures. All six responses are exported;
the figure illustrates their common marginal calibration.

The pre-2019 fit ends at MAM 2019. It forecasts JJA 2019 and exports TXx risks
at 35, 36.6 and 39.7°C. Posterior means and intervals for conditional risks
remain distinct from prediction intervals for future observations.

## Validation

```bash
python -m research.validation --profile screen --workers 2
python -m research.validation --profile screen --recent --workers 2
# One separately scheduled origin, with a distinct output root:
python -m research.validation --profile screen --origin 1996-11-30 \
  --output results/validation_1996 --workers 2
```

Main origins are SON 1956, 1976 and 1996, with 30-year forecasts and no updates
using held-out observations. The windows contain 120, 120 and 119 observed
blocks; the last ends in JJA 2026. Recent origins are SON 2015 and SON 2020.
Outputs include:

- Coverage at 90%, 95% and 99%, interval widths and counts of misses in each direction.
- A main LaTeX coverage/width table, with coverage expressed as percentages.
- Coverage, widths and signed forecast errors by forecast decade.
- Expected and observed seasonal threshold counts and predictive count-tail probabilities.
- Recent summer-TXx/winter-TNn paths, with predictive means, location and 95% intervals.

There are no default held-out CRPS or predictive-PIT plots. General diagnostic
functions remain available in the package. An optional constant-linear-trend
benchmark is provided by `research.linear_benchmark`, configured separately
in `config/linear_benchmark.json`: private priors, fixed seasonal location,
seasonal dispersion, and a 1970–SON-2015 training period. It exports coverage
and summer threshold counts. This is an explicitly specified reproducible
comparison; old numerical results are not copied into its outputs.

## Independent chains and portability

```bash
python -m research.main --profile paper --chain-id 0 --no-figures
python -m research.main --profile paper --chain-id 1 --no-figures
python -m research.main --profile paper --chain-id 2 --no-figures
python -m research.main --profile paper --chain-id 3 --no-figures
python -m research.combine results/main/paper/chains/chain_0 \
  results/main/paper/chains/chain_1 results/main/paper/chains/chain_2 \
  results/main/paper/chains/chain_3 --output results/main/paper_combined
```

The combined result retains each chain. Duplicate random streams and
incompatible fits are rejected. Private fits can additionally be split by
`--channel TXm` etc.; six responses with two chains each can use twelve cores
across six jobs. Pooling requires every participating channel inside each
chain. See `docs/parallel.md` for CPU budgets and scheduler-independent usage.
Machine-specific job files belong in the ignored `local/` or `job_scripts/`
directories. No environment paths, partitions or BIOBOT launch code are built
into the package.

## Data and publication provenance

The loader produces 538 complete seasonal blocks (MAM 1892–JJA 2026) or 1,616
complete months. It omits the incomplete first winter and rejects incomplete
interior blocks rather than silently changing the clock. See `data/README.md`
and `data/SOURCES.md` for the supplied record and its provenance. The software
license does not relicense the observations.

Full production fits must be rerun and assessed before updating scientific
results. Release smoke tests are execution checks only. Keep the saved fits,
resolved JSON, logs and hashes with the publication archive; generated
research results are intentionally excluded from the software release.
