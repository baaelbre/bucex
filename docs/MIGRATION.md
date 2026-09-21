# Migrating from 1.7.2 to 1.7.3

`MCMC` adds `chain_workers=1`. Set it to four for process-parallel chains;
execution uses `spawn` on every platform. Each process runs the same backend
and retains the existing chain seed. Within the same runtime, changing worker
count preserves seeded draws and chain ordering. Bitwise identity across BLAS,
NumPy or platform versions is not promised. BLAS thread pools are limited to
one thread per chain, including serial execution, to avoid oversubscription
and worker-count-dependent numerical reductions.

Existing Python calls remain serial unless opted in. The research base JSON
now requests four workers, capped by chain count. Custom executable scripts
must use an `if __name__ == "__main__":` guard. Matplotlib remains an optional
`plot` dependency; threadpoolctl is now a small core dependency. Workers need
additional memory during fitting and result assembly. Hierarchical
`channel_workers` is a separate within-chain option; normally leave it at one
when using parallel chains.

All existing model, forecast and archive APIs remain valid. Stored priors and
scientific defaults are unchanged; no archive migration is required. Execution
metadata records requested/effective workers, worker PIDs and complete seed
states. Worker failures raise, without returning an incomplete posterior.

`prior_assessment` adds a focused, configurable workflow over the existing
`sensitivity` and `validate` drivers. It saves compact traces, level/slope/risk
paths, forecast cases and origin-level diagnostics even when large fit archives
are disabled. `SensitivityReport` re-reads those exports without MCMC.
See [PRIOR_ASSESSMENT.md](PRIOR_ASSESSMENT.md).

# Migrating from 1.7.1 to 1.7.2

No inference or archive-schema migration is required. Existing 1.7.1 fit files
remain readable. Re-report saved fits to obtain the manuscript style, monthly
PIT diagnostics and small compressed trace tables. Initial slopes now appear
in trace plots, and seasonal scale coefficients are included in convergence
screening. Neither change alters the posterior draws.

Research configurations are already included in `config/revision/`. Start with
`START_HERE.md`. The default level-innovation SD prior median remains .01, with
normal continuous shrinkage and ASIS off. Constant scale remains available.

The main reporter now honours `save_fits: false`. Endpoint and forecast-width
intervals follow the declared `credible_interval`; old exports keep their old
bounds. Use `report --level 0.95` to recompute report intervals from saved draws.
The figure assembler rejects conflicting interval levels and duplicate fits.

# Migrating from 1.7.0 to 1.7.1

No model, fit, forecast or saved-file API changes are required. The corrected
GEV coefficient proposal is selected automatically for continuous private FS
fits, including copula margins. Gaussian coefficient updates are unchanged.
Re-run poorly mixed GEV fits: re-exporting an archive cannot repair its draws.
Use fresh chains to check the fix before relying on old-chain warm starts.

`fs_priors()` and the SERRA base configuration now default to **normal** rather
than lasso innovations. Their level innovation SD prior median is **.01**
rather than .02; slope and seasonal medians remain .00005 and .02. Explicitly
supplied priors and priors stored in existing archives are not rewritten.
Request `innovation="lasso"` to retain that family. For a sampler-only comparison
with the old normal-prior TXn run, use `independent_level_002.json`.

The sensitivity reference is now named `normal_reference`, and the family
alternative is `lasso`. New `sensitivity/level.json` and `joint_level.json`
compare .005/.01/.02 for level alone. Update job submissions that refer to old
variant names. No PGAS, SSVS or ASIS is introduced into the research protocol.

# Migrating from 1.6.5 to 1.7.0

- Existing Model/Channel component declarations and fit archives remain supported.
- Named `parameters` declarations add `Constant` and `Latent`; see PARAMETER_EVOLUTION.md.
- Structural log-scale evolution uses its own components, shrinkage priors and forecast innovations.
- SERRA primary data now explicitly end in August 2026. Historical 2022 configurations remain named.
- Primary scale is genuinely constant: `seasonal_scale=false`; LKJ eta=1; ASIS off.
- Changing the scale assumption or prior requires a refit. Re-exporting a saved fit changes reports only.
- Recent climate contrasts now span September 1996–August 2026, paired against 1892–1921.
- Joint sensitivity configs perform joint refits. Core, targeted and structural checks are separate.
- Univariate `--scale constant` now disables monthly scale effects; `--scale seasonal` enables them.
  Linear/RW and full structural scale remain general package capabilities, not paper CLI options.
- New public period estimands and parameter path plots reduce research-side statistical code.

## Earlier migration: 1.6.4 to 1.6.5

- No inference kernel or prior default changed. Existing 1.6.4 fit archives load directly.
- Full SERRA application runs use the latest bundled month; historical configurations retain 2022.
- Reports default to PNG and save slope, physical-parameter traces, PIT/Q-Q, monthly scale, calendar forecasts and risk curves.
- `Forecast.aggregate`, `probability_draws`, `risk_summary`, `sigma_draws`, and public report/plot functions are additive.
- Rerender with `python -m research.serra.report --fit ...`; this never updates the posterior.
- `check_updates` evaluates genuinely later observations from one fixed forecast origin.
- Existing `annual=True` fitted risk and `annual_aggregation_check` keep their **at least one monthly threshold crossing / extreme** semantics. Use `Forecast.aggregate(reduction="mean")` for an annual mean.

## Earlier migration: 1.6.3 to 1.6.4

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
