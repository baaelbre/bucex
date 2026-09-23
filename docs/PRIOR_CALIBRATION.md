# Physical interpretation of the hierarchical priors

The COMPSTAT calibration applies to the present continuous FS model. There is
no spike/slab selection in the main analysis, so these are coefficient-scale
and hyperprior calibrations, not slab probabilities.

Let `s_level` and `s_slope` be signed innovation coefficients with conditional
normal SDs tau_level and tau_slope. With monthly updates and horizon H:

```text
level[t+1] = level[t] + slope[t] + s_level * epsilon[t+1]
slope[t+1] = slope[t] + s_slope * zeta[t+1]

SD(new level contribution at H | tau_level) = tau_level * sqrt(H)
SD(new slope contribution to level at H | tau_slope)
    = tau_slope * sqrt(H*(H-1)*(2*H-1)/6)
```

These integrate the signed coefficient prior and standardized innovations,
conditional on the selected prior scale and current state. They exclude
weather/observation noise, initial-state uncertainty and other components.
They are not a forecast interval for observed temperature.

For the initial slope, a conditional SD of 0.30°C/decade equals
0.30/120 = 0.0025°C per monthly step. It contributes 0.9°C SD of linear
displacement over 30 years if fixed at that anchor. Initial-slope uncertainty
must be kept separate from subsequent changes in slope. The 0.30 scale is a
declared regularization assumption, not an estimate of Uccle's initial warming
rate or an externally validated universal value for all six summaries.

Seasonality uses the actual dummy-seasonal transition F and innovation loading
R, with gain sqrt(sum((e1' F^k R)^2, k=0,...,H-1)). For monthly seasonality and
H=360 the gain is sqrt(60). Since F^360=I, this describes same-month seasonal
change. Using sqrt(360) for seasonal effects would be incorrect.

## The median-to-normal-SD conversion

The package uses a physical process-SD median m, whereas the slides use the
normal SD tau. With q=Phi^(-1)(.75)=0.67448975:

```text
tau = m/q
log(m) ~ Normal(log(anchor), d^2)
SD(displacement | m=anchor) = gain * anchor/q
SD(displacement, integrating m) = gain * anchor/q * exp(d^2)
```

The second identity uses E[m²]=anchor²*exp(2*d²). At d=log(2), the fully
marginal SD is 1.617 times the conditional-at-anchor SD. The marginal
distribution is not Gaussian: multiplying this SD by 1.96 does not give its
exact 95% interval. The hyperprior's own 95% scale interval is
anchor*exp(±1.96*d), approximately [0.257,3.89] times the anchor.

Initial slopes use the conditional normal SD directly; their conversion has
no q divisor. The same exp(d²) inflation applies after integrating their
lognormal shared SD. Hyperparameters are separate for the four coefficient
types; they are shared across responses, not across quantities with different
units or dynamical roles.

The default SERRA innovation anchors .0025, .0000125 and .02 imply 30-year
conditional SDs .0703, .0729 and .2297°C, or marginal SDs .1137, .1179 and
.3714°C. The original slide's signed-normal level SD .02 corresponds to a
physical-SD median .01349, **not** .02. The current quarter anchors are stronger
regularization than that slide. They retain the existing exploratory choice;
this release does not relabel posterior-informed tuning as external knowledge.

## API: specify effects directly

```python
import numpy as np
import pandas as pd
import bucex as bx

# Illustrative scientific scales, not new default SERRA settings:
shared = bx.SharedShrinkage.from_effects(
    horizon=360,
    level_displacement_sd=.10,
    slope_displacement_sd=.15,
    seasonal_displacement_sd=.20,
    initial_slope_sd=.30,       # °C per slope_time_unit, here a decade
    slope_time_unit=120,
    period=12,
    log_sd=np.log(2),
    calibration='anchor',      # or 'marginal' for fully integrated RMS SDs
)
print(pd.DataFrame(shared.calibration(horizon=360, unit='degC')))
```

`SharedShrinkage(medians=..., initial_slope_sd=...)` remains available. In that
constructor the initial slope is **per update**; the `from_effects` constructor
explicitly converts from the stated rate unit. Omitting an innovation effect
leaves its channel prior unpooled; `initial_slope_sd=None` disables its pooling.
`innovation_response_gains` is shared by calibration and fitted innovation
effect diagnostics to keep the transition convention consistent.

The tables answer "what evolution did we regard as plausible?" They cannot
answer "which prior is objectively best?" Use convergence, stable scientific
contrasts, posterior replications and held-out prediction for that assessment.
The supervisor draft can contain this calibration now and leave targeted
hyperprior sensitivity for the appendix. Do not tune scales to maximize
acceleration evidence, smoothness or prior/posterior separation.

## Unrestricted normal GEV shape

The new default is `GEV()` with `fs_priors('gev')`: xi~Normal(0,.3²), without
fixed bounds. JSON uses `"xi_bounds": [null, null]`. Finite and one-sided
endpoints remain possible, e.g. `[-.5,.5]` or `[null,.5]`. A uniform prior must
have finite endpoints. The likelihood always enforces 1+xi*(y-mu)/sigma > 0.

Removing prior bounds is different from removing GEV support. It also does
not make every predictive moment finite. A GEV conditional mean requires
xi<1 and a conditional variance requires xi<.5. An unrestricted normal prior
permits shapes beyond both thresholds, even when none occur in a short saved
chain. Use quantiles, event probabilities and credible bands as primary risk
summaries; finite Monte Carlo averages do not establish finite theoretical
predictive moments. `shape_support.csv` records this distinction in reports.
An interval calculation for the latent location does not require finite GEV
observation variance. Explicitly bounded archived fits keep their old support.
