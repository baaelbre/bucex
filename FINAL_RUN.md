# Final manuscript workflow

Run every command from the BUCEX 1.8.7 package root. Long jobs should be run
inside `tmux` or the local job manager used on BIOBOT. Four chains use four
processes and one numerical thread per process. There is no mid-chain
checkpoint; timestamped output directories prevent accidental overwriting.

## 1. Install and verify

```bash
python -m pip install -e ".[plot,test]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
python -m pytest -q
```

The first value printed by Python must be `1.8.7`, from this unpacked directory.

## 2. Audit, preflight and smoke test

```bash
python -m research.seasonal.prepare --config research/seasonal/config/final.json --output results/serra_187_seasonal_data
python -m research.seasonal.preflight --config research/seasonal/config/final.json --output results/serra_187_final_plan
python -m research.seasonal.fit --config research/seasonal/config/smoke.json
```

Confirm 538 complete seasons, MAM 1892--JJA 2026, last included day
`2026-08-31`, and the four prior medians
`(0.0144513322, 0.000108839649, 0.00834348054, 0.00463877353)`.
The smoke run checks execution only and is not inferential output.

## 3. Final full-record fit

```bash
python -u -m research.seasonal.fit --config research/seasonal/config/final.json
```

Record the timestamped directory printed at completion, then run:

```bash
python -m research.seasonal.check_final --run PATH_TO_FINAL_RUN
```

This command exits with status 2 unless the saved config is exactly the final
1.8.7 config, all required exports exist, the source/data window matches, and
the four-chain R-hat/ESS gate passes. It writes `final_check.json`. Do not
weaken the gate or change hyperparameters in response to the posterior. If it
fails, inspect the listed traces and diagnose the sampler before scheduling a
longer fresh run.

## 4. Main-text supporting fits

Run prior and structural sensitivity as separate timestamped studies:

```bash
python -u -m research.monthly.sensitivity --config research/seasonal/config/manuscript_sensitivity.json
python -u -m research.monthly.sensitivity --config research/seasonal/config/adequacy.json
```

Run the genuinely prospective 2019 record-event fit and its targeted
sensitivity. Both stop the training data at the end of MAM 2019; JJA 2019 is
one step ahead, and the 39.7°C threshold is declared in the configs:

```bash
python -u -m research.seasonal.fit --config research/seasonal/config/pre2019.json
python -u -m research.monthly.sensitivity --config research/seasonal/config/pre2019_sensitivity.json
```

Run the predeclared expanding-window validation requested by the reviewers:

```bash
python -m research.seasonal.preflight --config research/seasonal/config/comment5.json --output results/serra_187_comment5_plan
python -u -m research.monthly.validate --config research/seasonal/config/comment5.json
```

Each forecast-origin `convergence_*.json` must be checked before interpreting
coverage or tail counts. With 140 held-out seasons per response, a nominal 1%
tail contains only 1.4 expected events; the result is necessarily descriptive.

The monthly-block analysis is a sensitivity analysis fitted and scored against
the same daily-derived seasonal outcomes:

```bash
python -m research.seasonal.compare --config research/seasonal/config/compare_full.json --plan
python -u -m research.seasonal.compare --config research/seasonal/config/compare_full.json
```

This comparison resumes completed model/origin jobs with `--run PATH`; it does
not checkpoint an interrupted chain.

## 5. Manuscript figures

Core figures need only the passing full-record run:

```bash
python -m research.seasonal.manuscript_figures \
  --run PATH_TO_FINAL_RUN \
  --output results/serra_187_manuscript_figures
```

After the supporting fits finish, rebuild the complete main-text set:

```bash
python -m research.seasonal.manuscript_figures \
  --run PATH_TO_FINAL_RUN \
  --sensitivity PATH_TO_MANUSCRIPT_SENSITIVITY PATH_TO_ADEQUACY \
  --validation PATH_TO_VALIDATION/joint \
  --block-comparison PATH_TO_BLOCK_COMPARISON \
  --pre2019 PATH_TO_PRE2019_RUN PATH_TO_PRE2019_SENSITIVITY \
  --output results/serra_187_manuscript_figures
```

The stable stems are `exploratory_seasonal_blocks`, `seasonal_levels`,
`seasonal_slopes`, `seasonal_scales`, `seasonal_pit_qq`,
`seasonal_change_contrasts`, `seasonal_rate_summary`, `shared_shrinkage`,
`copula_correlations`, `seasonal_risks`, `seasonal_risk_forecasts`,
`compound_heat_conditional_risk`, and, when supplied,
`prior_sensitivity`, `heldout_predictive_validation`,
`monthly_block_sensitivity`, and `pre2019_record_event`.

`figure_manifest.json` records every source file and checksum. The development
flag `--allow-unconverged` exists only to inspect layouts; its use is recorded
and such figures are not final evidence.
