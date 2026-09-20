# Alignment with the working SERRA revision

The manuscript follows six Gaussian/GEV summary-specific monthly location
models, normal FS shrinkage, constant unknown shape and a constant scale
baseline. Calendar-month scales are a targeted adequacy extension. Private
trajectories remain separate under a joint Gaussian copula likelihood.

The supplied preliminary evidence consists of independent constant-scale
1.7.1 fits through August 2026. This release does not replace that evidence
with smoke-test output. Final empirical claims remain conditional on
convergence, sensitivity and predictive validation.

| Manuscript box | Research work |
|---|---|
| R1 | `simulate`, `endpoint`, joint recovery; preserve failures and poor convergence |
| R2 | Innovation/initial/shape/scale/copula prior sensitivity; prior/posterior comparisons |
| R3 | Fixed/evolving location seasonality and constant/monthly scale; held-out assessment |
| R4 | Matched R=I/copula fits, paired contrasts, dependence and ordering diagnostics |
| R5 | Held-out tails, risk estimates, endpoint and forecast-width/aggregation checks |
| R6 | Scientific synthesis after the preceding evidence |
| S1 | Detailed experiments and numerical diagnostics |
| S2 | Source-extension audit and common-period historical comparison |

`figures.json` declares all nine current manuscript figures and two additional
scale/annual-forecast panels. `figure_manifest.json` records their numerical
inputs. A trace-image-only archive cannot supply a new trace figure: re-export
its saved fit or retain an explicit placeholder.

See [reviewer comments](REVIEWER_MATRIX.md) and [publication runs](PUBLICATION_RUNS.md).
