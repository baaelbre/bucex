# BUCEX 1.7.1

This release fixes inefficient continuous-FS GEV coefficient preconditioning
identified in TXn. A deterministic conditional-mode Gaussian reference replaces
the response-centered reference; the exact likelihood/reference slice correction
remains. The shared kernel covers independent and copula-conditioned GEV margins.
Gaussian direct coefficient draws are unchanged. No ASIS is required by the fix.

Normal innovations are now the `fs_priors()` and SERRA default. The level
innovation **SD prior median is .01**, previously .02. The slope and seasonal
medians remain .00005 and .02. Explicit priors and saved-fit priors are not
rewritten. All primary records still reach August 2026.

Reference construction lives in its own inference module with saved optimizer,
fallback, support-repair and covariance-regularization metrics. Existing model,
latent-parameter, fitting, diagnostics, forecast, risk and archive APIs remain
available. No PGAS or conference directory is introduced.

New configurations are `independent_level_002.json` for the previous normal
level prior under the fixed sampler, and `sensitivity/level.json` plus
`joint_level.json` for .005/.01/.02 medians with other priors fixed. The main
sensitivity reference is now `normal_reference`; `lasso` is an alternative.
Update job commands selecting the old `lasso_reference` name.

Start with [START_HERE.md](START_HERE.md), then read
[the revision guide](docs/REVISION_GUIDE.md),
[migration](docs/MIGRATION.md),
[validation](validation/RELEASE_VALIDATION.md) and
[remaining reviewer studies](docs/REVIEWER_MATRIX.md).

The defect affects efficiency. Previous poorly mixed fits need refitting;
regenerating reports cannot repair them. Tests and short execution checks do
not certify convergence of production fits or publication readiness.
