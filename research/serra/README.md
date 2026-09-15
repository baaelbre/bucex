# SERRA: scripts to run and results to inspect

Run from the directory containing `pyproject.toml`, after:

```bash
python -m pip install -e ".[plot,test]"
```

The core story is six private FS/SSVS trajectories, each with its own seasonal
location and optional seasonal observation scale, followed by a joint residual
Gaussian copula. No factor fit is required. Primary paper configs use January
1892–December 2022. The bundled later observations are a separate mixed-source
extension, selected by `config/extension_full.json`.

## 1. Check the workflow

```bash
python -m research.serra.univariate --series TXm TXx
python -m research.serra.copula --independence
python -m research.serra.copula
```

The defaults fit 36 months with four warmup and four retained draws. They check
execution, saved files, diagnostics, figures and forecasts. They cannot estimate
SSVS probabilities or credible intervals reliably. Each invocation prints a new
output directory under `results/serra`; it never overwrites a previous analysis.

## 2. Run the six marginal analyses

```bash
python -m research.serra.univariate --config research/serra/config/independent_full.json
```

Omit `--series` for all six sequentially. For separate jobs, use the same command
with `--series TXm`, `TNm`, `TXx`, `TXn`, `TNx`, or `TNn`. None depends on a joint
fit. Means use conditional FFBS and extrema use Laplace–MH. Each channel has its
own FS/exact-SSVS structure and seasonal scale. Keep the independent paper route
if the joint extension cannot be fitted reliably.

The initial full-study budget is four chains, 2,000 warmup and 2,000 retained
iterations per chain. Extend it according to chain behavior and Monte Carlo
error; the file name `full` does not imply convergence. All inputs are archived
in `config.json` and `fit.bucex`.

| Output | What to look for |
|---|---|
| `mcmc.csv`, `scientific_targets.csv` | Across-chain agreement, R-hat near 1, adequate ESS for parameters and level changes; inspect traces when flagged |
| `engine.json`, `run.json` | Nonzero state movement, proposal behavior and warnings; acceptance alone is insufficient |
| `series_structure.csv` | Absent/fixed/dynamic component probabilities; assess indicator mixing before interpreting |
| `series_level.pdf`, `series_season.pdf` | Slow change and changing location seasonality |
| `series_observation_scale.csv/.pdf` | The distinct repeating seasonal residual SD/GEV scale |
| `series_risk.csv`, `series_return_level_100_blocks.csv` | Time-varying threshold risks and conditional 100-monthly-block return levels |
| `series_forecast.pdf`, `forecast.csv` | Future observations including state and observation uncertainty |
| `in_sample_pit.csv` | Descriptive model checks only; do not call these held-out calibration |

The absence probability for slope is .10; the other .90 is split equally between
fixed and dynamic slopes: `(.10,.45,.45)`. Location seasonality has prior
`(0,.5,.5)`. The seasonal scale contrast prior SD is .30; observation baseline
variance is IG(2,2). Full prior declarations are in `models.py` and the JSON.

## 3. Add dependence with matched margins

```bash
python -m research.serra.copula --config research/serra/config/copula_full.json --independence
python -m research.serra.copula --config research/serra/config/copula_full.json
python -m research.serra.copula --config research/serra/config/copula_full.json --eta 1
python -m research.serra.copula --config research/serra/config/copula_full.json --eta 4
```

`--independence` fixes R=I. The next run estimates R with LKJ(2); the final two
check its prior sensitivity. All use the same private FS/SSVS models. The
copula enters the actual likelihood during fitting, so trajectories and
selection can change. This is not a second-stage residual-only adjustment.

Inspect `copula_correlations.csv`, `residual_dependence.csv`, channel structure
and scale files, scientific-target diagnostics, `compound_heat_forecast.csv`
and `ordering_*.csv`. Positive correlations alone do not establish better
prediction or narrower credible intervals. No predictions are sorted or
rejected to repair crossings. A Gaussian copula is not a hard-order model.

## 4. Address the reviewer experiments

Start with smoke configs; replace `smoke.json` with `paper.json` after checking
compute cost and mixing:

```bash
python -m research.serra.sensitivity --config research/serra/config/sensitivity/smoke.json
python -m research.serra.simulate --config research/serra/config/simulation/smoke.json
python -m research.serra.endpoint --config research/serra/config/endpoint/smoke.json
python -m research.serra.forecast_check --config research/serra/config/forecast/smoke.json
python -m research.serra.joint_recovery
```

- **Sensitivity:** process SD and variance priors/posteriors, initial slope and
  seasonality, model probabilities, baseline variance, seasonal-scale priors,
  and fitted GEV shape bounds. Compare `prior_posterior.csv`, scientific targets
  and risks. Lasso/manuscript comparisons use constant observation scale and
  their matched `constant_scale_ssvs` baseline; do not attribute scale-model
  changes to shrinkage alone.
- **Shape recovery:** eleven generating shapes `-.5,-.4,...,.5`, with fitted
  bounds `[-.75,.75]`. This is different from varying fitted prior bounds on
  the observed data. `simulation/pilot.json` is a compute pilot; `paper.json`
  is the replicated study. `zero_paper.json`, `weak_paper.json` and
  `endpoint_paper.json` add known zero, weak and difficult endpoint cases;
  their `_smoke` versions check execution. Record failures, diagnostics and
  Monte Carlo uncertainty across converged replicates.
- **2019 endpoint:** full-data smoothing and a separate fit ending before July
  2019, with `pre_event_forecast.csv`. The latter gives a pre-event probability
  for the selected event; event selection is still retrospective. Approximate
  Laplace is only a controlled ablation against corrected Laplace–MH. These
  paired configs use constant observation scale to isolate the sampler effect.
- **Forecast uncertainty:** separate latent level, predictor and observations;
  inspect horizon growth and annual aggregation. GEV scale is not its SD, and
  posterior predictive variance need not exist near shape .5. Quantiles remain
  the primary interval diagnostic.
- **Joint recovery:** `--config research/serra/config/joint_recovery_full.json`
  compares known independent/correlated residuals, seasonal scale and mixed
  margins. Use `--replicate 0` etc. for separate jobs. `recovery.csv`,
  `correlations.csv`, `structures.csv` and `scores.csv` record truth comparisons;
  the fit shape support is wider than the generating grid. This statistical
  simulation does not enforce daily-summary ordering.

## 5. Validate out of sample

```bash
python -m research.serra.validate --config research/serra/config/independent_full.json
python -m research.serra.validate --config research/serra/config/joint_full.json
python -m research.serra.validate --config research/serra/config/copula_full.json
```

The first training prefix fixes data-centred priors; subsequent origins cannot
use future data to recenter them. Full configs advance annually with 12-month
held-out windows. Check marginal/joint log scores (lower is better), CRPS and
threshold scores; central 90/95/99% intervals; direct quantiles from .005 to
.995; PIT; compound heat Brier scores; and predictive ordering. Tail results
need enough events and intervals for empirical coverage, not just percentages.
Use longer blocks if forecast errors remain dependent between adjacent years.

Compare two printed validation directories:

```bash
python -m research.serra.compare results/serra/BASELINE results/serra/COPULA --output comparison.csv
python -m research.serra.compare results/serra/BASELINE results/serra/COPULA --block-years 2 --output comparison_two_years.csv
```

Replace the two capitalized directory names with your actual runs. Positive
`improvement` favors the candidate. The uncertainty interval resamples paired
calendar blocks; it is not a posterior credible interval. A one-year smoke run
has no informative block-bootstrap interval. Nonfinite scores fail explicitly
so zero predictive densities are investigated rather than silently removed.

## API and file organization

`models.py` declares components and priors; `run.py` is a small common dispatcher;
`univariate.py` and `copula.py` are user-facing entry points. `report.py` exports
public result methods, while `experiment.py` supplies the shared setup for
sensitivity and recovery. The study scripts above each address a scientific
question. `prepare_uccle.py` uses `bx.derive_uccle_monthly` if daily data need
regenerating; it is unnecessary for fitting the bundled summaries. `submit.pbs`
is an optional launcher for the same configured runner.

General hierarchical/shared-state APIs remain documented in `docs/` but are not
part of these SERRA scripts. All bands are pointwise unless explicitly stated.
A 100-monthly-block level is not a 100-year return level. The manuscript's result
boxes remain pending until these studies pass diagnostics and are summarized.
