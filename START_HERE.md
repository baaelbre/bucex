# Start with BUCEX 1.7.3

Extract the archive and open a terminal inside `bucex-1.7.3`, beside
`pyproject.toml`. The commands work in Windows PowerShell, Linux and macOS.

```text
python -m pip install -e ".[plot]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Expect version **1.7.3**. The package now supports process-parallel chains.
Existing fits remain readable; the normal-prior defaults and statistical
transition kernels are unchanged. No new chains are needed solely for plots.

## 1. Check execution once

```text
python -m research.serra.prior_assessment --config research/serra/config/priors/smoke.json --stage all
```

This checks the complete pipeline on 2020–August 2026 with two workers and
very short chains. It generates comparison CSVs and PNGs. Its posterior and
predictive results are **execution checks, not scientific evidence**.

## 2. Inspect the focused study

```text
python -m research.serra.prior_assessment --stage plan
```

The default `config/priors/pilot.json` uses TNm, January 1892–August 2026,
normal FS priors and repeating monthly observation scales. It requests four
chains in four processes, each with 500 warmup and 500 retained draws.

| Candidate | Level SD prior median | Slope SD prior median | Seasonal SD prior median |
|---|---:|---:|---:|
| `normal_reference` | .010 | .000050 | .020 |
| `level_half` | .005 | .000050 | .020 |
| `level_slope_half` | .005 | .000025 | .020 |

These are medians of the **physical innovation SD**, not normal coefficient
prior SDs. Other priors remain fixed, including initial-slope SD .0025 per
month and monthly log-scale contrast prior SD .3. There is no SSVS or ASIS.

Historical fits stop in December 2000, 2010 and 2020 and predict the next five
years without conditioning on those held-out observations. Full-record
sensitivity uses all observations through August 2026. `--stage plan` runs no
MCMC and prints the resolved dates, candidates and number of fits.

## 3. Run the assessment

For both stages in one command:

```text
python -m research.serra.prior_assessment --stage all
```

This performs three full-record sensitivity fits plus nine historical refits.
Candidates and forecast origins run sequentially; only the chains run in
parallel. The script prints one `Assessment directory` at the start.

Alternatively run stages separately:

```text
python -m research.serra.prior_assessment --stage sensitivity
python -m research.serra.prior_assessment --run YOUR_ASSESSMENT_DIRECTORY --stage predictive
```

Replace `YOUR_ASSESSMENT_DIRECTORY` with the path printed by the first command.
An existing run retains its saved settings and completed stages. Changed
priors, data, or MCMC budgets belong in a new run. Interrupted stages retain
partial files and are not silently mixed with a rerun.

## 4. Read the comparison

The `comparison/` directory contains:

| Output | What to inspect |
|---|---|
| `convergence.csv` | Full-record and per-origin numerical status; inspect detailed chain tables when flagged |
| `prior_updates.csv`, `TNm_prior_posterior.png` | Prior/posterior medians, intervals, displacement and width ratios |
| `scientific_targets.csv`, `TNm_scientific_targets.png` | Warming, latent slope contrasts and their sensitivity |
| `TNm_level.png`, `TNm_slope.png`, `TNm_risk.png` | How smoothness and scientific conclusions change together |
| `scores_by_origin.csv`, `predictive_comparison.csv` | Paired CRPS and negative log predictive density; all and early/later horizons |
| `TNm_forecasts.png`, `TNm_forecast_scores.png` | Actual held-out observations, predictive intervals and each origin's scores |
| `coverage_by_month.csv`, `TNm_forecast_coverage.png` | Seasonal coverage, retaining sample counts |
| `pit_by_month.csv`, `TNm_forecast_pit.png` | Held-out predictive calibration |

All forecast scores are losses: **lower is better**. A positive paired
`improvement` means the candidate improves on the reference. Three forecast
blocks give a descriptive screen; no precision is invented by treating their
months as independent replicates. Full trace CSVs and per-fit diagnostics are
kept under `sensitivity/` and `predictive/`. The pilot avoids large `.bucex`
archives; set `save_fits` to true if you want to retain those as well.

Prefer stronger shrinkage when prediction and coverage remain comparable.
Posterior displacement is not a quantity to maximize. A narrow or smoother
trajectory alone does not establish acceleration. Keep the sensitivity of
period slope differences visible. Short-chain convergence warnings are not
waived by this workflow.

## 5. Continue selectively

Run another response with the same settings:

```text
python -m research.serra.prior_assessment --series TXm --stage all
python -m research.serra.prior_assessment --series TXx --stage all
```

The same driver works for all six summaries. Set the series list in JSON or
pass several names after `--series`. Avoid transferring TNm's preferred prior
to all extrema without checking their own forecasts and diagnostics.

`config/priors/confirm.json` is an optional larger check with six forecast
blocks and 1,000 warmup/1,000 retained draws per chain. It does not run unless
explicitly requested. You can limit it to two candidates with `--variants
normal_reference level_half`. Increase MCMC only where diagnostics require it.

Rebuild comparison tables and figures without fitting:

```text
python -m research.serra.prior_assessment --run YOUR_ASSESSMENT_DIRECTORY --stage report
```

The existing univariate and copula scripts remain available. Their base JSON
now requests four chain workers. The Python API defaults to one worker; set
`MCMC(chain_workers=4)` explicitly in your own code. Keep `if __name__ ==
"__main__":` around custom script entry points; all provided drivers have it.
See [the parallel/prior guide](docs/PRIOR_ASSESSMENT.md) for API details and
[the publication run guide](docs/PUBLICATION_RUNS.md) for the broader revision.
