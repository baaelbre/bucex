# SERRA workflows

Run modules from the source root, after `python -m pip install -e ".[plot,test]"`.
Every study uses the public package API. Configuration specifies the scientific
choices; the scripts only construct models, run them and export summaries.
Start with marginal analyses. Dynamic factors and copulas are optional studies.

## Which script does what?

| Module | Purpose |
|---|---|
| `tutorials.py` | Four short API demonstrations: Gaussian, GEV, hierarchy, shared components |
| `run.py` | One or six independent Uccle fits; optional hierarchical/shared/copula fits |
| `models.py` | Readable components, observation distributions and explicit priors |
| `report.py` | Fits, scientific contrasts, diagnostic tables, risks, forecasts and figures |
| `validate.py` | Expanding-window held-out scores, PIT and 90/95/99% calibration |
| `sensitivity.py` | Marginal innovation, shrinkage-hyperprior and shape-bound sensitivity |
| `simulate.py` | Repeated recovery across shape values and endpoint-focused simulations |
| `endpoint.py` | July 2019 TXx support/endpoint assessment under approximate and exact inference |
| `forecast_check.py` | Latent/location/observation forecast uncertainty and annual aggregation |
| `experiment.py` | Small shared setup for the sensitivity and recovery studies |
| `submit.pbs` | Optional PBS launcher for the same configured runner |

## Execute in this order

1. **API execution check.**

   ```bash
   python -m research.serra.tutorials --kind all
   python -m research.serra.run --config research/serra/config/independent_smoke.json --series TXx
   ```

   Tutorial smoke defaults retain four draws. They establish execution and
   readable outputs only. Use `--kind gaussian`, `gev`, `hierarchical`, or
   `shared` to inspect one example; `--config research/serra/config/tutorial_full.json`
   selects a longer simulation and MCMC budget.

2. **The six independent analyses.**

   ```bash
   python -m research.serra.run --config research/serra/config/independent_full.json --series TXm
   python -m research.serra.run --config research/serra/config/independent_full.json --series TNm
   python -m research.serra.run --config research/serra/config/independent_full.json --series TXx
   python -m research.serra.run --config research/serra/config/independent_full.json --series TXn
   python -m research.serra.run --config research/serra/config/independent_full.json --series TNx
   python -m research.serra.run --config research/serra/config/independent_full.json --series TNn
   ```

   Omit `--series` for all six sequentially. Each job has its own model, fit and
   report. Gaussian means use FFBS; GEV extrema use Laplace–MH, with FS structural
   shrinkage/selection. `--phi linear`, `--phi rw` or `--phi ssvs` adds log-scale
   sensitivity for an independent extreme series. Joint modes currently
   require stationary observation scales.

3. **Prior sensitivity and shape recovery.**

   ```bash
   python -m research.serra.sensitivity --config research/serra/config/sensitivity/smoke.json
   python -m research.serra.simulate --config research/serra/config/simulation/smoke.json
   ```

   Then use `sensitivity/paper.json` and `simulation/pilot.json`; run
   `simulation/paper.json` only after assessing the pilot's compute cost and
   chain behavior. `simulation/zero_paper.json` and `weak_paper.json` assess
   structural selection when innovations are absent or weak; their `_smoke`
   counterparts check execution. The full shape grid is **-0.5, -0.4, …, 0.5**, including zero.
   The fitting shape bounds are wider than this grid, so boundary truths are
   not artificially forced onto the fitted prior boundary. Approximate
   `laplace` is included only as an explicitly labelled comparison to the exact
   `laplace_mh` target.

   Prior sensitivity changes level, slope and seasonal innovation scales
   separately and jointly; compares SSVS, original manuscript and matched lasso
   profiles; varies lasso Gamma shape/rate; and checks alternative shape bounds.
   Distinguish changes in prior assumptions from sampler failure. Inspect
   `prior_posterior.csv`, `prior_posterior_targets.csv`, `scientific_targets.csv`
   and mixing before comparing model probabilities or risk estimates. Replicated
   simulation coverage needs Monte Carlo uncertainty and enough converged fits.

4. **Endpoint and forecast comments.**

   ```bash
   python -m research.serra.endpoint --config research/serra/config/endpoint/smoke.json
   python -m research.serra.forecast_check --config research/serra/config/forecast/smoke.json
   ```

   Their `paper.json` configurations use the full data record. The endpoint
   study checks the exact GEV support and tail shape before interpreting a
   finite endpoint. `simulation/endpoint_paper.json` provides the associated
   repeated simulation. The forecast study separates latent level, location
   predictor and future observations, and compares annual risk aggregation
   with predictive simulation. `forecast_check --fit path/to/fit.bucex` can
   inspect an existing univariate fit with the matching series configuration.

5. **Held-out validation.**

   ```bash
   python -m research.serra.validate --config research/serra/config/independent_smoke.json --series TXx
   python -m research.serra.validate --config research/serra/config/independent_full.json --series TXx
   ```

   Each origin refits ordinary `bx.fit`. Data-centred priors use only the first
   training prefix and remain fixed across later origins. The output contains
   central 90/95/99% coverage plus actual CDF-quantile coverage in both tails,
   PIT, proper scores and threshold scores. Check the number of held-out cases:
   a small sample cannot reliably certify 99% tail calibration. Overlapping
   origins and serial dependence require dependence-aware score uncertainty.

   Only compact score tables remain in memory between folds. Set
   `validation.save_fits=true` if you intentionally want every posterior archive.

## Optional joint analyses

```bash
python -m research.serra.run --config research/serra/config/hierarchical_smoke.json
python -m research.serra.run --config research/serra/config/shared_smoke.json
python -m research.serra.run --config research/serra/config/joint_smoke.json
python -m research.serra.run --config research/serra/config/joint_smoke.json --copula
python -m research.serra.run --config research/serra/config/shared_smoke.json --copula
```

The `_full.json` counterparts select 1892–2022 with four chains and initial
1,000-warmup/1,000-retained budgets. Hierarchies share selection probabilities;
shared models estimate actual common states with unit loadings and weighted
zero-sum departures. Each channel has its own seasonal component. The shared
and private continuous-prior configs initially use fixed seasonal cycles;
independent and hierarchical configs allow structural seasonal selection.
These are different model assumptions and must be distinguished in comparisons.

`joint` fixes residual Gaussian-copula correlation to identity. `joint --copula`
estimates it under an LKJ prior, with the same private components and priors.
`shared --copula` estimates residual dependence alongside common/departure
states. Compare the same configuration with and without dependence before
ascribing interval changes to a copula. The copula does not enforce summary
ordering, and its nonsingular Gaussian form has no asymptotic tail dependence.

The same optional choices work with `validate`. Joint held-out scoring exports
`joint_log_scores.csv` in addition to marginal scores. Physical ordering checks
retain original predictive draws; never sort or reject predictions after fitting.

## What to inspect

| Output | Assessment |
|---|---|
| `mcmc.csv` | Parameter R-hat, ESS and support; failures need investigation |
| `scientific_targets.csv` | Mixing and intervals for channel-level, shared and departure changes |
| `engine.json` | Proposal convergence and movement; acceptance alone is insufficient |
| `*_level.pdf`, `*_season.pdf` | Own slow change and seasonal cycle |
| `shared_warming.pdf`, `*_departure.pdf` | Common location change and individual differences |
| `copula_correlations.csv` | Conditional Gaussian-score correlation, with posterior and MCMC uncertainty |
| `ordering_*.csv` | Probabilities of violating applicable physical inequalities |
| `*_risk.csv`, `*_return_level_100_blocks.csv` | Threshold probabilities and 100-monthly-block return levels |
| `scale.csv`, `scale_models.json` | Independent GEV scale path and scale-model probabilities |
| `coverage_summary.csv`, `held_out_pit.csv`, `score_summary.csv` | Held-out calibration and proper-score behavior |

Figures use readable labels and no default titles. Credible bands are
pointwise. A 100-block return level is **not** a 100-year return level. Common
warming summarizes Gaussian means and GEV locations; it is not an attribution
estimate. All six bundled summaries are monthly, even if an earlier manuscript
used seasonal extremes.

Publication requires the experiments to run with adequate mixing and
Monte Carlo precision. The source release supplies their implementation and
executed validation evidence, not completed answers to every reviewer request.
See `docs/REVIEWER_MATRIX.md` and `docs/VALIDATION.md` in the source root.
