# Migrating from 1.6.2 to 1.6.3

Existing constant-scale univariate `Model`, `fit`, diagnostics, risk, prediction
and archive calls remain supported. Existing explicit prior probabilities are
preserved. The default SSVS slope probabilities now equal `(0.10, 0.45, 0.45)`.

To add a seasonal observational SD/scale, use
`Gaussian(scale=SeasonalScale())` or `GEV(scale=SeasonalScale())`, an FS SSVS
prior, and `asis=False`. Obtain all dated scale paths with `fit.sigma_draws()`
or `fit.sigma_draws(channel="TXx")`. Seasonal effects are zero-sum in log scale.
For Uccle convenience wrappers, also pass `asis=False` explicitly.

To retain FS/SSVS while adding residual dependence, put private channels in
`MultiSeriesModel(..., copula=GaussianCopula(eta=2))` and pass
`MarginalPriors({name: own_ssvs_prior, ...})`. `JointPriors` continues to select
the existing continuous shared/centered route; it does not become SSVS.
A one-time copula fit to marginal posterior residuals does not have the new
joint posterior target.

Research entry points are now `research.serra.univariate` and
`research.serra.copula`. Omit `--series` for all six; pass one or more names to
select channels. The common `run` module remains available. Conference and
standalone tutorial scripts/configs are removed. General hierarchy/shared
construction is documented under `docs/`; these are not active SERRA studies.

The primary full configs use 1892–2022. To explore the later mixed-source data,
use `config/extension_full.json`. The loader still returns the complete bundled
record when no end date is supplied. The redundant older daily CSV was removed;
the retained updated CSV includes the historical prefix. Dataset helpers and
custom data directories remain supported.

**Refit affected older mixed/GEV hierarchical exact-SSVS results.** Version
1.6.3 corrects the independence-proposal anchor in that structural-selection
update. Existing univariate GEV selection already used the correct fixed anchor
and is not affected by this correction. See `RELEASE_NOTES.md`.

Schema 2.10 archives store seasonal effects and `MarginalPriors`, with historical
readers retained. `init=fit` restarts the new scalar/private route from the
matching model and prior declaration. PGAS is removed; use supported FFBS or
Laplace–MH routes. Missing observations remain unsupported on the new private
FS/copula route and raise a clear error.
