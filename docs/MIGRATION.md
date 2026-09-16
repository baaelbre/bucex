# Migrating from 1.6.3 to 1.6.4

Existing univariate calls, result methods and archives remain supported. The
new SERRA workflow changes the declared scientific prior, so it requires new
fits; an earlier SSVS fit is not a continuous-prior result.

1. Replace the research `ssvs_*_priors` calls with `fs_priors(family,
   innovation="lasso")`. Normal and triple-gamma are matched by physical prior
   median SDs. The old lower-level constructors still work.
2. Use `LogScale()` for the new private univariate kernel even when the scale
   is constant. `SeasonalScale` continues to work directly.
3. Add private channels and `MarginalPriors` to fit a joint model. The marginal
   declarations can be identical to those in the univariate fits. No separate
   first-stage posterior is frozen.
4. `asis=True` is now allowed for all-continuous private fits. SSVS requires
   `asis=False`; the SSVS implementation is retained but not used by default.
5. Inspect SDs/variances and horizon effects. Continuous models intentionally
   omit sampled structure indicators. Structure-probability files are generated
   only for an actual SSVS fit.

`SeasonalGaussianCopula` adds a phase axis to `copula_correlation_draws()`:
(draws, phases, channels, channels). Supply `phase=1,...,period` for one matrix.
Constant-copula return shapes are unchanged. `combine_chains=False` always
preserves separate chain/draw axes.

`forecast.compound_probability` retains the simulation-based estimate.
`compound_probability_draws` is a new bivariate quadrature method returning one
conditional probability per parameter/state draw. Averaging integrates those
draws; forecast risk bands also include variation in sampled future states.

Configurations accept one relative `extends` file. Dictionaries merge; lists
replace. Saved run configs are fully expanded. `research/serra` is the sole
research workflow directory; duplicated conference examples and the nested
copy of the source tree are not shipped.

1.6.4 corrects very-small-coefficient lasso/PC local-scale updates and removes
triple-gamma variance flooring. The new private coefficient and ASIS updates
avoid older covariance floors/cancellation. Refit continuous analyses affected
by these changes before comparing them scientifically. Legacy constant-scale
kernels retain their established coefficient implementation and are explicitly
labelled in the paired historical Laplace benchmark.

New archives use schema 2.11. Earlier supported schemas remain readable; older BUCEX installations cannot be expected to read the new scale/copula declarations.
