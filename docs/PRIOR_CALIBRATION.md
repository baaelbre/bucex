# Fixed-prior calibration in 1.9.3

The active seasonal model uses fixed Normal priors on signed FS innovation coefficients and on the initial slope. There are no shrinkage hyperpriors. All six separate analyses use the same settings.

The manuscript specifies the three Normal prior SDs directly: tau_alpha=0.01, tau_beta=0.0001, tau_gamma=0.01. The initial slope has prior variance P_beta0=0.0001 and therefore SD 0.01. No median-to-SD conversion applies. These settings are fixed, while coefficients and process variances remain inferred.

| Component | Fixed Normal SD | Prior SD of 30-year latent displacement |
|---|---:|---:|
| Level innovations | 0.01 | 0.109545 degrees C |
| Slope innovations | 0.0001 | 0.075420 degrees C |
| Seasonal innovations | 0.01 | 0.077460 degrees C |
| Initial slope | 0.01 | 1.200000 degrees C |

At H=120 seasonal updates, the respective gains are sqrt(H), sqrt(H(H-1)(2H-1)/6), sqrt(60), and H. These SDs integrate the Normal coefficient prior and, where applicable, the standardized future innovations. They exclude observation noise and other latent components and are not observed-temperature prediction intervals. The initial-rate SD is 40 times its per-update SD, or 0.40 degrees C per decade.

Sensitivity multiplies the fixed Normal SD directly. Half/double changes the SD by that factor and its variance by one quarter/four; no exp(log_sd^2) adjustment applies. The baseline changes the marginal prior compared with older lognormal mixtures. A prior/posterior resemblance is evidence of limited learning, not a reason to keep broadening a prior.

See [the complete 1.9.3 grid](../BUCEX-1.9.3-commands.md). The following material documents older hierarchical APIs and calibrations for reproducibility; it does not specify the active 1.9.3 analysis.

---

# Historical hierarchical prior calibration

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

For the seasonal manuscript, the initial-slope scale is also the median
absolute coefficient, so its conditional SD is the scale divided by q. The
older `initial_slope_sd` convention instead uses the conditional SD directly.
The same exp(d²) inflation applies after integrating either lognormal scale.
Hyperparameters are separate for the four coefficient
types; they are shared across responses, not across quantities with different
units or dynamical roles.

In 1.8.4 the draft configurations apply marginal moment calibration to the
COMPSTAT scales. The monthly innovation anchors are 0.008343480538,
0.00002085870135 and 0.008343480538; the initial-slope anchor is
0.001546257845 per month. With log SD log(2), the integrated 30-year component
SDs are 0.379473, 0.196769 and 0.154919°C, and initial-rate SD is 0.30°C/decade.
These replace the 1.8.2 exploratory quarter anchors in the draft workflow.

The 1.9.0 seasonal reference uses log SD log(3), with median-absolute anchors
0.004836005867750226 for level, seasonal and initial-slope coefficients, and
0.00004836005867750226 for slope innovations. These equal the old seasonal
anchors (.01,.0001,.01,.01) times exp(log(2)²-log(3)²). Since
E[coefficient²]=(anchor/q)² exp(2*log_sd²), this preserves their marginal second
moments while changing the prior shape and its mass near zero.

At H=120 and 40 updates per decade, the integrated 30-year component SDs remain
0.263°C (level), 0.181°C (slope innovations) and 0.186°C (seasonal innovations),
with initial-rate SD 0.959°C/decade and initial-slope displacement SD 2.876°C.
These are latent-component RMS calibrations, not observation forecast SDs.
The log(2)/log(4) comparisons keep those second moments fixed; the log(3) with
old anchors comparison isolates width with the anchors held fixed. Halving
or doubling each anchor separately changes that component's RMS effect by the
same factor. See [the run sheet](../FINAL_RUN.md).

A posterior that resembles its prior signals limited learning about that
quantity; changing the prior until the curves separate is not validation.
The report therefore supplements the prior/posterior comparisons with posterior
innovation contributions at 10 and 30 years and probabilities that these
contributions' conditional SDs fall below .05, .10 and .20°C. Their uncertainty
excludes observation noise and uncertainty in the initial/current state.

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

`SharedShrinkage(medians=..., initial_slope_sd=None,
initial_slope_median=.01)` uses the same median-absolute convention for all
four components. The legacy `initial_slope_sd` API remains available. Both
initial-slope arguments are **per update**; the `from_effects` constructor
explicitly converts from the stated rate unit. Omitting an innovation effect
leaves its channel prior unpooled; setting both initial-slope arguments to None
disables its pooling.
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
