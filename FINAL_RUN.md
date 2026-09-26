# BUCEX 1.8.8 manuscript experiments

All commands start at the unpacked `bucex-1.8.8` root. Use `tmux` or a BIOBOT
job for long fits. The complete **sequential** queue is
`bash RUN_PAPER_EXPERIMENTS.sh`; it logs each job to
`results/serra_188_logs/` and halts on a failed command or numerical gate.
Each fit uses four parallel chains (`chain_workers=4`); it does not create ten
jobs at once. Numerical library threads are limited to one per chain.

## 1. Prepare and check the declared prior

```bash
python -m pip install -e '.[plot,test]'
python -c 'import bucex; print(bucex.__version__, bucex.__file__)'
python -m pytest -q
python -m research.seasonal.prepare --config research/seasonal/config/final.json --output results/serra_188_seasonal_data
python -m research.seasonal.preflight --config research/seasonal/config/final.json --output results/serra_188_final_plan
python -m research.seasonal.prior_effects --config research/seasonal/config/final.json --output results/serra_188_prior_effects
python -u -m research.seasonal.fit --config research/seasonal/config/smoke.json
```

The preflight must report 538 complete seasons ending 31 August 2026, anchors
`0.01, 0.0001, 0.01, 0.001`, and prior 30-year contribution SDs about
`0.263, 0.181, 0.186, 0.288` degrees C. The final number is initial-slope
displacement; its rate SD is `0.096` degrees C per decade. Smoke fits check
execution only.
The combined same-season latent-location contrast has prior SD about 0.468°C
over 30 years; the effects script also exports simulation quantiles rather
than treating its non-Gaussian mixture as a normal interval.

## 2. Full reference and convergence gate

```bash
python -u -m research.seasonal.fit --config research/seasonal/config/final.json
python -m research.seasonal.check_final --run PATH_PRINTED_BY_FINAL_FIT
python -m research.seasonal.dynamic_comparison --run PATH_PRINTED_BY_FINAL_FIT --output results/serra_188_dynamic_comparison
```

This is four chains, 3,000 warm-up plus 8,000 retained draws each. Save its
timestamped directory as `FINAL_RUN`. Do not report results if `check_final`
exits with status 2. The old `reference_20260923.json` remains an archival
configuration and has the previous prior; it is not this manuscript fit.
The last command exports time-resolved within-season gaps between summaries,
relative to each pair's 1892--1922 seasonal gap, and local differences in
warming rates. It uses complete paired posterior draws. The CSVs retain
pointwise intervals and probabilities for practical gap/rate thresholds;
period-average comparisons alone cannot answer when additivity changed.

## 3. Prior and model sensitivity

`--stage all` fits each candidate on the full record and at the two declared
held-out origins, then writes comparison reports. Run these as **separate**
studies with their own timestamped roots:

```bash
python -u -m research.monthly.prior_assessment --config research/seasonal/config/manuscript_sensitivity.json --stage all
python -u -m research.monthly.prior_assessment --config research/seasonal/config/physical_sensitivity.json --stage all
python -u -m research.monthly.prior_assessment --config research/seasonal/config/adequacy.json --stage all
```

The first varies hyperprior width and GEV shape (including a bounded shape
case); the second halves/doubles each of the four manuscript anchors; the
third compares the width/presence of seasonal observation dispersion and fixed
location seasonality. The width variants keep the four
marginal coefficient second moments fixed. Every variant's numerical check
must pass before attributing a difference to the prior.

```bash
python -u -m research.monthly.prior_assessment --config research/seasonal/config/dependence_sensitivity.json --stage all
```

This tests LKJ concentrations 1, 2 and 4, with matched held-out forecasts.

## 4. Dependence alternatives and prospective event

```bash
python -u -m research.seasonal.fit --config research/seasonal/config/constant_copula_full.json
python -u -m research.seasonal.fit --config research/seasonal/config/independence_full.json
python -u -m research.seasonal.fit --config research/seasonal/config/pre2019.json
python -u -m research.monthly.sensitivity --config research/seasonal/config/pre2019_sensitivity.json
```

The first two keep the main seasonal priors and full sampling budget but
change contemporaneous dependence. The prospective fit stops after MAM 2019;
JJA 2019 is predicted one season ahead. The final command varies shape prior
and observation dispersion for that forecast. Record the pre-2019 run and
sensitivity directories as `PRE2019_RUN` and `PRE2019_SENSITIVITY`.

## 5. Reviewer validation and monthly-block comparison

```bash
python -m research.seasonal.preflight --config research/seasonal/config/comment5.json --output results/serra_188_comment5_plan
python -u -m research.monthly.validate --config research/seasonal/config/comment5.json
python -u -m research.monthly.run --config research/monthly/config/final.json
python -m research.seasonal.compare --config research/seasonal/config/compare_full.json --plan
python -u -m research.seasonal.compare --config research/seasonal/config/compare_full.json
```

Validation uses seven expanding windows, each holding out 20 seasons; examine
the seven `joint/convergence_*.json` files before comparing held-out PIT,
coverage, CRPS, logarithmic scores and tails. The monthly full-record run is
supplementary. The comparison uses four matched cutoffs and scores both block
resolutions on the **same seasonal observations**, not their raw likelihoods.
Its monthly side retains the separately declared monthly prior settings.
Save the validation and block-comparison roots as `VALIDATION_RUN` and
`BLOCK_RUN`.

## 6. Check supporting fits and make figures

```bash
python -m research.seasonal.paper_gate PRIOR_RUN PHYSICAL_RUN ADEQUACY_RUN DEPENDENCE_RUN \
  CONSTANT_COPULA_RUN INDEPENDENCE_RUN PRE2019_RUN PRE2019_SENSITIVITY \
  VALIDATION_RUN BLOCK_RUN
python -m research.seasonal.manuscript_figures \
  --run FINAL_RUN \
  --sensitivity PRIOR_RUN/sensitivity ADEQUACY_RUN/sensitivity \
  --validation VALIDATION_RUN/joint \
  --block-comparison BLOCK_RUN \
  --pre2019 PRE2019_RUN PRE2019_SENSITIVITY \
  --output results/serra_188_manuscript_figures
```

Replace the uppercase paths by the directories printed by each command.
`paper_gate` fails for missing or flagged convergence reports. The physical
anchor study remains available in its supplementary comparison report; it is
omitted from the already dense main sensitivity panel. The figure builder
requires a passing full-record `convergence.json` and records source checksums.

The shell queue stores these paths automatically. An interrupted monthly vs
seasonal comparison resumes with `python -m research.seasonal.compare --run
PATH_TO_BLOCK_RUN`; incomplete MCMC chains in other jobs require a fresh run.
