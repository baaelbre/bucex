# BUCEX 1.8.2: first supervisor draft

Run from the extracted `bucex-1.8.2` directory beside `pyproject.toml`. Keep the
old checkout and results; do not merge source trees.

The draft model has six private FS structural locations, repeating monthly
observation scales, constant shapes with unrestricted Normal(0, 0.3²) priors,
and four shared regularization scales: level, slope and seasonal innovations,
plus initial slopes. Every series still has its own initial slope and trajectory.
The draft Gaussian copula uses four seasonal correlation matrices with shrunk
departures from a common baseline. Monthly dispersion and seasonal dependence
are separate model components. The record is January 1892–August 2026, 1,616
months. Mixed inference uses Laplace–MH with an exact likelihood correction;
the configurations use neither SSVS nor ASIS.

## 1. Install and protect the session

```bash
python -m pip install -e ".[plot,test]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Expect `1.8.2` and this checkout's path. Four chains use four local processes;
Slurm is not needed. Numerical libraries use one thread per chain.

If tmux is installed, start it before long commands:

```bash
tmux new -s serra182
```

Detach with Ctrl-b then d; reconnect with `tmux attach -t serra182`. This
protects against SSH disconnection, not machine shutdown. Mid-chain checkpoint
and resume is not implemented. Completed fits are saved before reports.

Without tmux, the following is an alternative to the foreground main command
in step 4. Use one route, not both:

```bash
mkdir -p logs
nohup python -u -m research.serra.copula --config research/serra/config/draft/main.json > logs/draft182.log 2>&1 < /dev/null &
tail -f logs/draft182.log
```

## 2. Exploration and physical prior calibration: no MCMC

```bash
python -m research.serra.explore
python -m research.serra.preflight --config research/serra/config/draft/main.json --output results/serra_182_plan
```

Observed-data Figures 1–2 go below `results/serra_exploration/`. Preflight saves
`preflight.json` and `prior_calibration.csv`. Check the dates, six responses,
four workers, monthly scales, seasonal copula and four shared scales. Draft
state arrays alone require about 4 GB; workers, reporting and archives need more.

The default quarter anchors retain the preceding sensitivity specification:

| Component | Conditional effect SD at the hyperprior median | SD integrating hyperprior uncertainty |
|---|---:|---:|
| Level innovations: 30-year displacement | 0.070°C | 0.114°C |
| Slope innovations: 30-year level displacement | 0.073°C | 0.118°C |
| Seasonal innovations: same-month change after 30 years | 0.230°C | 0.371°C |
| Initial slope: warming rate | 0.300°C/decade | 0.485°C/decade |

These are SDs, not 95% limits or maximum permitted changes. Innovation effects
condition on the current state and exclude weather variability; initial slope
uncertainty is additional. The COMPSTAT formulas use signed-normal coefficient
SDs, whereas the configuration uses medians of physical innovation SDs. See
[PRIOR_CALIBRATION](docs/PRIOR_CALIBRATION.md) for the conversion and API.
Physical units explain these strong assumptions; they do not establish optimality.

## 3. One execution check

```bash
python -m research.serra.copula --config research/serra/config/draft/smoke.json
```

Six responses, 2023–August 2026, four parallel chains, 3 warmup + 4 retained
iterations. This checks fitting, archiving, reports and forecasts. Convergence
warnings are expected. These outputs must not be used as paper results.

## 4. One main fit for the draft

```bash
python -u -m research.serra.copula --config research/serra/config/draft/main.json
```

This is one joint fit, with 1,000 warmup + 1,000 retained draws per chain. Save
the printed report path under `results/serra_182_draft/`. It contains `fit.bucex`,
CSV/JSON results, PNGs and compressed traces. Each run gets a fresh directory.
The posterior archive can be large; retain it locally and share compact reports.

This is a preliminary budget. Check mixing before interpreting credible bands.
If chains disagree, label the output preliminary and avoid precise interval or
acceleration claims. Shared shrinkage does not guarantee improved convergence.

| Read first | Question for the draft |
|---|---|
| `run.json`, `config.json`, `declared_priors.json` | Intended record and model? |
| `convergence.json`, `mcmc.csv`, parameter and scale traces | Chain agreement and enough effective samples? |
| `shared_shrinkage.csv`, `shared_shrinkage_effects.csv`, `shared_shrinkage_traces.png` | Learned regularization and its physical interpretation? |
| `initial_slope_prior_posterior.csv` / `.png` | Learning about the six signed initial slopes? |
| `*_level.png`, `*_slope_C_per_decade.png`, `period_contrasts.csv` | Changes in typical conditions and warming rates? |
| Seasonal and observation-scale plots | Location seasonality versus observation dispersion? |
| PIT/QQ, `residual_serial.csv`, `residual_dependence_by_month.csv` | Remaining marginal, serial or seasonal failures? |
| `*_risk.png`, `TXx_risk_39p7.png`, `*_period_risks.csv` | Changing threshold risk, including the 2019 record? |
| Forecast reports, `compound_heat_forecast.csv` | Future and simultaneous-event risks? |
| `ordering_in_sample.csv`, `ordering_forecast.csv` | Physically impossible replicated orderings? |

Initial-slope comparisons use the original temperature scale, including minima.
The hyperparameter is a positive prior SD. `shared_shrinkage_effects.csv`
describes regularization strength, not realized temperature changes. In-sample
PIT is descriptive, not forecast validation. Seasonal copulas do not guarantee
removal of winter skewness or temporal residual memory.

## 5. Manuscript figures without refitting

Replace the quoted path with step 4's printed report directory:

```bash
python -m research.serra.figures --reports "YOUR_DRAFT_REPORT_DIRECTORY" --formats png
```

These fitted-model panels complement exploratory Figures 1–2. Recipes are in
`research/serra/config/revision/figures.json`; missing exports are flagged.
To re-report saved draws or change the forecast horizon:

```bash
python -m research.serra.report --fit "YOUR_DRAFT_REPORT_DIRECTORY/fit.bucex" --horizon 120 --format png --output results/serra_182_replots
```

You can now write the supervisor draft: introduction, data/exploration, model
and computation, estimated evolution, translation to risk, then discussion.
Label results preliminary and leave boxes for sensitivity and unresolved
checks. Tracking evolving distributions and risk is the central contribution;
an acceleration result is not a prerequisite.

## 6. Two historical prediction checks alongside writing

```bash
python -m research.serra.validate --config research/serra/config/draft/predictive.json
```

Two refits: train through December 2015 and forecast 2016–2020 including 2019;
train through December 2020 and forecast 2021–2025. Four chains, 500 warmup +
500 retained draws per refit. Shared scales are learned from training data only.
Fold fits are saved for reuse. These are fixed-origin five-year forecasts,
not rolling one-step residuals.

Under the printed directory's `joint/` folder, inspect `folds.csv`, `mcmc_*.csv`,
`convergence_*.json`, `held_out_pit.csv`, `coverage_by_case.csv`, `scores.csv`,
`joint_log_scores.csv` and predictions. Two blocks give limited evidence about
rare tails; they cannot precisely establish 99% calibration.

## Later: appendix and final runs

[SUPPLEMENTARY_RUNS](docs/SUPPLEMENTARY_RUNS.md) gives matched commands for
anchor/width sensitivity, initial-slope pooling, shape bounds, seasonal scales,
location seasonality and dependence. No sensitivity grid runs in steps 2–6.

Optional matched hierarchy with R=I:

```bash
python -m research.serra.run --config research/serra/config/draft/independence.json
```

Six genuinely separate fits with fixed priors and the unchanged univariate API:

```bash
python -m research.serra.univariate --config research/serra/config/draft/independent.json
```

Add `--series TXm` to try one response. Separate fits cannot learn the shared
scales from all six series. Once the model and diagnostics are satisfactory:

```bash
python -m research.serra.copula --config research/serra/config/draft/final.json
```

This requests 2,000 warmup + 4,000 retained iterations per chain. The filename
does not certify convergence, adequacy or publication readiness. Old `.bucex`
archives keep their original priors when loaded; installing 1.8.2 does not
retroactively add initial-slope pooling or remove their shape bounds.
