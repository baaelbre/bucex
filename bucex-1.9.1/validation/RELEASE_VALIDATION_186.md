# BUCEX 1.8.6 release check (2026-09-24)

## Scope

Reviewer comment 5: held-out central 90%, 95% and 99% coverage; separate
upper/lower tail checks; observed and predicted risk-event counts. No climate
conclusions follow from these code checks.

## Executed

- `python -m compileall -q bucex research tests/test_tail_validation.py`
- `python -m pytest tests/test_tail_validation.py tests/test_seasonal_workflow.py::test_seasonal_adequacy_reports_keep_quarterly_units -q` — 4 passed, including the independent Gaussian and joint seasonal paths. A synthetic two-case example checks both sides of a central miss, the upper and lower 1% events and expected fixed-threshold counts.
- `python -m research.seasonal.preflight --config research/seasonal/config/comment5.json --output results/comment5_seasonal_preflight` — 538 complete seasonal blocks through August 2026, resolved seasonal priors and four-chain configuration.
- `python -m research.monthly.preflight --config research/monthly/config/comment5.json --output results/comment5_monthly_preflight` — 1,614 monthly blocks through August 2026, resolved monthly priors and four-chain configuration.
- Resolved `validation_splits` for both configs: seven disjoint windows, 140 distinct seasonal or 420 distinct monthly forecast dates per response. Seasonal first dates are December 1970, 1980, 1990, 2000, 2010, 2015 and 2020; each covers five complete years.
- Rebuilt the three summary tables by calling `python -m research.monthly.tail_validation --run <saved smoke validation>/joint` on stored case files.

## Interpretation

The short fits in tests only check execution and report consistency. The seven
full-history four-chain validation fits for each model have **not** been run
here. Inspect their per-origin convergence before quoting any coverage or
score. At a 1% one-sided tail probability, the expected number of observed
exceedances is just 1.4 (seasonal) or 4.2 (monthly) per response; report
denominators and event counts, without implying calibrated extrapolation.
The monthly and seasonal configurations inherit their own 1.8.5 reference
priors and score different observation units; comparison on common seasonal
targets remains a separate experiment. The 2015 and 2020 origins reuse
periods from the prior grid; if that grid informed prior selection, they
cannot also serve as untouched confirmation cases.
