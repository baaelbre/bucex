# Direct GIG updates for half-normal shrinkage

BUCEX 1.9.7 changes the computational update of half-normal hierarchical prior
scales. The prior and posterior target are unchanged. There is no new tuning
parameter or API option.

## Full conditional

For one component, let J be the number of active coefficients sharing its scale:

\[
s_j\mid\tau\sim N(0,\tau^2),\qquad \tau\sim\operatorname{HalfNormal}(A),
\qquad S=\sum_{j=1}^{J}s_j^2.
\]

Consequently,

\[
p(\tau\mid s)\propto\tau^{-J}
\exp\left[-\frac12\left(\frac{S}{\tau^2}+\frac{\tau^2}{A^2}\right)\right].
\]

With v=τ², including the transformation Jacobian gives

\[
v\mid s\sim\operatorname{GIG}\left(\lambda=\frac{1-J}{2},\ a=A^{-2},\ b=S\right),
\]

where the GIG density is proportional to
v^(λ−1) exp{−(av+b/v)/2}. For six responses, λ=−5/2. The number of seasons or
state innovations does not enter J: in the non-centred model, the shared scale
enters the response coefficients' normal priors, and the standardized paths have
their own fixed Gaussian prior.

SciPy parameterizes `geninvgauss(p, b)` with density proportional to
x^(p−1) exp{−b(x+1/x)/2}. Set r=√S/A and draw

```python
x = scipy.stats.geninvgauss.rvs((1 - J) / 2, r, random_state=rng)
tau = A * np.sqrt(r * x)
```

The implementation computes r with `hypot` and stores
`log(tau/A) = (log(r) + log(x))/2`, avoiding unnecessary squaring of tiny
coefficients. It does not introduce a floor on the variance. Exactly zero active
coefficients collectively give an improper conditional (J≥1); this exceptional
case raises a diagnostic error rather than silently changing the model. Active
continuous coefficients should be initialized away from that singular state.

For J>1 and small r, SciPy's internal bounding calculations can overflow. The
implementation instead samples W=S/(2τ²) using an exact gamma rejection envelope:
draw W from Gamma((J−1)/2, rate=1) and accept with probability exp{−r²/(4W)}.
This is the same GIG conditional, without a small-variance approximation. The
acceptance comparison and scale transformation are evaluated in log coordinates.

## API and saved fits

```python
import bucex as bx

shrinkage = bx.SharedShrinkage.half_normal({
    "level": 0.01,
    "slope": 0.0002,
    "seasonal": 0.01,
})
priors = bx.MarginalPriors(channel_priors, shrinkage=shrinkage)
fit = bx.fit(data, model, priors=priors, mcmc=bx.MCMC(
    chains=2, chain_workers=2, warmup=1000, draws=1000, seed=197,
))
assert fit.metadata["shrinkage_scale_update"] == "gig"
```

`channel_priors`, `data` and `model` are the existing model inputs. The shared-scale
parameter names (`shrinkage.shared.level`, `.slope`, `.seasonal`) are unchanged.
Per-component `shared_shrinkage_gig_draws.*` metrics equal one per sweep. They are
draw counts, not Metropolis acceptance rates. Optional individual half-normal
hierarchies use `independent_shrinkage_gig_draws.*`.

The updater counts active members separately for every component. It also applies
if initial slopes are explicitly included in a half-normal hierarchy. The paper
reference continues to use separate fixed-variance normal initial-slope priors.
For an explicitly configured legacy median-absolute initial-slope scale, the
coefficient conversion is included in S, retaining that specification's meaning.
Lognormal, half-t and half-Cauchy hyperpriors keep their log-scale slice updates
and existing diagnostics. Existing archives remain readable and usable for
initialization; new draws use the updated kernel. A fixed seed need not reproduce
1.9.6.1 trajectories because the sampler consumes random numbers differently.

## Scope and verification

Gaussian paths and Gaussian structural-coefficient blocks retain their direct
conditional updates. GEV path proposals, MH corrections, coefficient updates,
observation-parameter updates and sign switches are unchanged. This replaces the
three reference shared-scale slice updates, not every slice update in the model.
It removes tuning and within-update dependence for these conditionals; it does
not eliminate dependence between Gibbs blocks or establish convergence.

`tests/test_gig_shrinkage_197.py` checks conditional quantiles against integration
of the original normal/half-normal density, several active-member counts, very
small coefficients, selection of the correct update, Gaussian/GEV fits,
save/reload/restart, forecasting and serial/parallel reproducibility. Release
checks and their results are recorded in `RELEASE_VALIDATION.json`.

The reference scales, all calibration experiments, forecast budgets and screen
and paper chain lengths are inherited from 1.9.6.1. Launchers use new default
output roots under `results/serra_197`; do not reuse a completed earlier run's
results directory for changed source code.

SciPy's distribution and sampling convention:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geninvgauss.html
