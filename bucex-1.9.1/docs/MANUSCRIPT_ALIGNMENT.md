# Working manuscript alignment — 1.9.0

The paper model has six private Gaussian/GEV location trajectories, shared
normal-coefficient shrinkage scales, repeating seasonal observation scales and
four regularized residual copula matrices. Its new reference has log(3)
hyperprior width. The transition and likelihood algorithms are retained;
Laplace proposals have MH correction, targeting the declared exact posterior.
Finite chains still require convergence assessment.

| Question | Current experiment |
|---|---|
| Prior width versus centre | `old_reference`, `wider_log4`, `wide_original_anchors` |
| Strength of each shared shrinkage component | Eight half/double variants in `physical_sensitivity.json` |
| Static versus evolving location seasonality | `fixed_location_seasonality` |
| Initial seasonal pattern and seasonal dispersion | Five alternatives in `adequacy.json` |
| GEV shape and support restrictions | Four alternatives in `manuscript_sensitivity.json` |
| Residual dependence | Four alternatives in `dependence_sensitivity.json` |
| Future innovation magnitude | 10-/30-year effects and .05/.10/.20°C threshold probabilities |
| Pre-2019 record risk | Separate four-fit `pre2019` batch |
| Held-out prediction | Separate 37-task `validation` batch; design can be discussed before launch |
| Monthly versus seasonal blocks | Explicit `research/seasonal/compare.py` workflow, outside the default batch |

All 25 full-record fits forecast 30 years and save their fits in either tier.
The default launchers submit posterior sensitivity only. New production
posteriors, convergence and predictive results are not asserted by a release
or by its computational smoke checks. See [FINAL_RUN.md](../FINAL_RUN.md).
