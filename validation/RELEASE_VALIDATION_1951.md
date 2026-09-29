# BUCEX 1.9.5.1 validation

## Completed

- Ten targeted scientific and integration tests passed. They check analytic
  30-year component variances; the fitted FS initial seasonal lag basis;
  minimum reflection and GEV inverse-CDF sampling; dependence from a shared
  hyperparameter versus private initial rates; nonfinite-value summaries;
  reproducibility; unchanged task inventory and monthly calibration; and
  the bounded BIOBOT queue's handling of failures.
- An additional isolated runner check passed with the observation loader
  replaced by a function that raises on access. Neither complete prior
  simulation nor the analytic prior-effect check reads observed temperatures.
- All 15 core paper prior settings completed with 10,000 replications each,
  seed 1951, batch size 250, and 658 seasonal positions per replication.
  None of the simulated observations in these runs was nonfinite.
- The standalone screen shell launcher completed a 120-replication reference
  smoke check, including all four figure outputs. This is a launcher check,
  not a replacement for the default 2,000-replication screen suite.
- Screen and paper BIOBOT reference dry-runs passed with two available CPUs
  and suitable memory planning budgets. The paper reference reserves 18.8 GiB;
  a deliberately smaller 12-GiB budget was correctly rejected.
- Production modules compile under Python 3.12. The legacy Python grammar,
  with print_function enabled, accepts all 166 production modules. This
  checks the earlier f-string quoting failure class; it is not a substitute
  for running Python 3.11 on the actual compute node.
- Shell syntax checks passed for the prior launcher, overnight launcher,
  HPC probe and runtime setup.
- Scientific contents of main, final, experiments and monthly-reference
  configurations match the 1.9.5 release, excluding output paths/comments.
- The revised 42-page manuscript compiles, with no unresolved references,
  LaTeX warnings or overfull boxes. New figure pages were rendered and inspected.

## Evidence and limits

`prior_simulations_paper/` contains compact summaries, settings, analytic checks,
reference sample paths and the actual four figures used by the manuscript.
Raw per-replication targets can be regenerated with the prior launcher and are
not included in this source release. Completed fits from 1.9.5 are not claimed
as new 1.9.5.1 runs; use separate release directories/results roots.

No new production posterior fit, forecast validation or real HPC submission was
performed for this patch. CPU-parallel smoke/queue tests do not establish that
all experiments will finish overnight or that posterior convergence will pass.
The inherited dataset warning about two daily TN > TX pairs remains visible;
this release does not alter the observations.

The reference initial-state prior generates very broad temperature cycles and
frequent ordering violations. These scientific findings are recorded rather
than hidden by sorting, truncation or changes to the posterior settings.
