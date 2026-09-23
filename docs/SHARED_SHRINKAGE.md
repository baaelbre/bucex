# Shared regularization in BUCEX 1.8.2

Every channel retains its own realized location trajectory, initial slope,
innovation SDs, seasonal pattern and observation parameters. Hierarchical
shrinkage shares the *prior scales*, while the Gaussian copula describes
contemporaneous residual dependence. Neither is a dynamic factor.

For component c (level, slope or seasonal), let a[j,c] be the signed FS
coefficient and q = Phi^(-1)(.75). The physical innovation SD is abs(a[j,c]):

```text
a[j,c] | m[c] ~ Normal(0, (m[c]/q)^2)
log(m[c]/anchor[c]) ~ Normal(0, log_sd^2)

beta0[j] | kappa0 ~ Normal(0, kappa0^2)
log(kappa0/initial_slope_sd) ~ Normal(0, log_sd^2)
```

There are four separate positive hyperparameters in the default SERRA fit.
The initial-slope SD kappa0 differs in units and meaning from the slope
innovation scale. It is not tied numerically to that scale, and its mean is
zero rather than a learned common warming rate. All four hyperparameters are
sampled jointly with the channel parameters and states. Conditional coefficient
priors are normal; integrating shared scales gives normal scale mixtures.

With log_sd=log(2), each scale's 95% hyperprior interval is about
[0.257, 3.89] times its anchor. These are soft regularization assumptions.
The medians in `medians` are conditional absolute-coefficient medians, whereas
`initial_slope_sd` is a conditional normal SD. This distinction is recorded in
the calibration table. See [physical calibration](PRIOR_CALIBRATION.md).

## API

```python
import bucex as bx

components = [bx.LocalLinearTrend(), bx.DummySeasonal(12)]
channels = [
    bx.Channel('mean', bx.Gaussian(scale=bx.SeasonalScale(12)),
               parameters={'mu': bx.Latent(components)}),
    bx.Channel('maximum', bx.GEV(scale=bx.SeasonalScale(12)),
               parameters={'mu': bx.Latent(components)}),
]
model = bx.MultiSeriesModel(channels, copula=bx.GaussianCopula())
regularization = bx.SharedShrinkage(
    medians={'level': .0025, 'slope': .0000125, 'seasonal': .02},
    initial_slope_sd=.0025,
)
priors = bx.MarginalPriors(
    {c.name: bx.fs_priors(c.family, period=12) for c in channels},
    shrinkage=regularization,
)
# fit = bx.fit(data, model, priors=priors, parameterization='fs',
#              mcmc=bx.MCMC(chains=4, chain_workers=4, warmup=1000, draws=1000))
```

Put parallel execution under `if __name__ == '__main__':` in standalone scripts.
Use `initial_slope_sd=None` to keep each channel's declared fixed initial-slope
prior. Research JSON uses `shared_shrinkage.pool_initial_slope: false`. Omit
`shrinkage` entirely for fixed independent priors. An initial-slope-only
hierarchy is allowed with `medians={}` and a positive initial-slope anchor.

Pooling requires at least two channels with the selected active coefficient.
A fixed but uncertain slope is eligible; a removed slope is not. An omitted
innovation is not counted as an observed zero in the scale update. With only
local-level models, disable initial-slope pooling. This hierarchy requires
zero-centred normal FS coefficient priors; it does not combine with SSVS or
local lasso/horseshoe/triple-gamma mixtures.

## Exact conditional update

Let u = log(scale/anchor), and let v be the conditional normal coefficient SD
at the anchor: anchor/q for innovations, anchor for initial slopes. For J
eligible coefficients the conditional log density, up to constants, is

```text
-u^2/(2*log_sd^2) - J*u - exp(-2*u)*sum(coefficient[j]^2)/(2*v^2).
```

The normalizing term -J*u is essential. No further log-Jacobian is added:
the declared hyperprior is normal directly in u. A univariate stepping-out
slice update samples this density. Conditional channel updates use the current
shared scales, including the correlated centred intercept/initial-slope prior
induced by time centring. Hyperparameters are not fitted afterwards.

Lower-tail reflection preserves these zero-centred priors. The signed slopes
are transformed back for scientific reporting. Chains use distinct random
streams; serial and parallel results with identical seeds are tested to agree.

## Interpretation and checks

`compare_shared_shrinkage` reports the four hyperparameter distributions;
`compare_initial_slope_priors` reports individual signed initial slopes;
`compare_innovation_priors` reports individual process SDs. Their prior draws
integrate the shared hyperparameters. One hyperparameter is drawn once per
joint prior draw and reused across channels.

Exchangeability of regularization is a substantive assumption. Temperature
units and monthly timing make pooling interpretable but do not prove that
averages and extrema have identical roughness. A shared scale does not imply
identical innovations, eliminate weak identification, or guarantee faster MCMC.

Prefer stable scientific contrasts and adequate prediction to maximizing
prior/posterior separation or reducing uncertainty. A positive continuous-SD
interval is not a model-selection probability. The initial slope refers to
the start of the record, not the current warming rate. Current slopes and
period contrasts must be read from posterior trajectories.

Old fit archives missing the new initial-slope field load with that pooling
disabled. Their stored shape bounds remain intact. Warm starts create a new
run under its declared prior; they do not retroactively change an old posterior.
