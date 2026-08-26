# Time-varying GEV log scale

## Public model choice

The GEV scale process is selected in the observation declaration:

```python
bx.GEV()                  # equivalent to phi="stationary"
bx.GEV(phi="linear")
bx.GEV(phi="rw")
bx.GEV(phi="ssvs")
```

Here \(\phi_t=\log(\sigma_t)\). Modelling log scale guarantees
\(\sigma_t>0\) and makes additive trend and innovation parameters easy to
interpret as multiplicative changes in scale.

## Models

### Stationary

\[
\phi_t=\phi_0, \qquad \sigma_t=\exp(\phi_0).
\]

This is the default and reproduces the pre-1.4 model. Existing code using
`bx.GEV()` is unchanged.

### Linear

\[
\phi_t=\bar\phi+\delta b_t,
\qquad
b_t=\frac{t-(T-1)/2}{T-1},
\]

for zero-based \(t=0,\ldots,T-1\). The basis is centered and runs from
\(-1/2\) to \(+1/2\). Therefore:

\[
\phi_{T-1}-\phi_0=\delta,
\qquad
\frac{\sigma_{T-1}}{\sigma_0^{\text{first}}}=\exp(\delta).
\]

The stored `phi_intercept` is the log scale at the center of the observed
record, and `phi_slope` is \(\delta\), the whole-record log-scale change.
Forecasts continue the same time basis beyond the record.

### Random walk

\[
\phi_0\sim p(\phi_0),\qquad
\phi_t=\phi_{t-1}+u_t,qquad
u_t\sim N(0,q_\phi).
\]

The stored `phi_rw_variance` is \(q_\phi\), `phi_rw_sd` is
\(\sqrt{q_\phi}\), and `phi_last` is the final in-sample value. Forecasts
start at `phi_last` and simulate future random-walk innovations independently
for each posterior draw.

### SSVS

`phi="ssvs"` introduces a model indicator

\[
M_\phi\in\{\text{stationary},\text{linear},\text{rw}\}.
\]

The implementation uses a product-space sampler. Inactive coordinates are
refreshed from their proper priors, and the model indicator is sampled from
its exact conditional probabilities based on the exact GEV likelihood. The
three indicator draws are stored as `phi_stationary`, `phi_linear`, and
`phi_rw`; `fit.phi_model_probabilities()` summarizes them.

This scale SSVS is independent of the existing structural SSVS declaration.
One fit may select both the location-process structure and the log-scale
model.

## Priors

```python
phi_prior = bx.PhiPrior(
    linear=bx.NormalPrior(0.0, 0.35),
    rw_variance=bx.InverseGammaPrior(2.5, 3.75e-5),
    model_probabilities={"stationary": 0.50, "linear": 0.25, "rw": 0.25},
)
```

Pass either the readable mapping shown above or a tuple ordered as stationary,
linear, random walk. Probabilities must be nonnegative and sum to one.

`linear` applies to \(\delta\). A standard deviation of 0.35 puts direct prior
mass on end/start scale ratios through \(\exp(\delta)\); for example,
\(\delta=0.35\) means a ratio of about 1.42.

`rw_variance` follows the package convention

\[
p(q_\phi)\propto q_\phi^{-a-1}\exp(-b/q_\phi).
\]

Conditional on a path, the variance has an inverse-gamma Gibbs update.
`priors.sigma2` retains its previous role for a stationary model and supplies
the initial/reference-scale prior for linear and random-walk models. If a
normal `log_sigma` prior is selected by another prior profile, it is used
directly instead.

Every public `*_gev_priors(...)` profile accepts the same optional
`phi_prior=` keyword. Location-shrinkage profile and scale-process prior can
therefore be changed independently.

In JSON the same settings are deliberately grouped and readable:

```json
"priors": {
  "sigma2": {"a": 2.0, "b": 2.0},
  "phi": {
    "linear": {"mean": 0.0, "sd": 0.35},
    "rw_variance": {"a": 2.5, "b": 0.0000375},
    "model_probabilities": {
      "stationary": 0.5,
      "linear": 0.25,
      "rw": 0.25
    }
  }
}
```

## Conditional scale inference

The log-scale implementation is a dedicated conditional kernel, separate from
the structural location kernel. This keeps the public model declaration small
and provides one extension point for future scale models.

Stationary and linear coefficients use random-walk MH steps evaluated with the
exact GEV likelihood. For a random-walk path, bucex differentiates the GEV log
density with respect to \(\phi\), constructs an iterated local Gaussian
approximation, and samples a complete path with a one-dimensional Kalman
smoother.

For `engine="laplace_mh"` and `engine="pgas"`, that Gaussian path is an
independence proposal. Its acceptance weight includes the exact GEV likelihood
and the correction between the true initial-scale prior and its Gaussian
proposal approximation. The random-walk transition law is common to target
and proposal and cancels. An invalid GEV-support proposal is an ordinary
rejection. The update therefore leaves the exact conditional posterior
invariant.

For `engine="laplace"`, the scale-path draw is intentionally approximate, just
as the structural state draw is. It is useful for pilots and initialization,
but should not be reported as exact posterior inference.

## Result and forecast contract

```python
phi_draws = fit.phi_draws()
sigma_draws = fit.sigma_draws()
```

Both return `combined_draws x T` by default for every GEV scale model. A
stationary scalar is expanded across time. Dynamic fits additionally store the
canonical `phi` and `sigma_path` arrays in the archive.

Relevant scalar draws are intentionally mode-specific:

| Model | Additional stored parameters |
|---|---|
| stationary | scalar `sigma`, `sigma2` |
| linear | `phi_intercept`, `phi_slope` |
| random walk | `phi_intercept`, `phi_last`, `phi_rw_sd`, `phi_rw_variance` |
| SSVS | all coordinates plus `phi_model` and three indicators |

`posterior_predictive()` uses the fitted in-sample scale path.
`forecast()` keeps stationary scale fixed, continues a linear trend, propagates
a random walk, or follows each SSVS draw's selected model. Both result objects
expose `parameters["phi"]` and `parameters["sigma_path"]`.

Endpoint, return-level, exceedance, PIT, and conditional-density calculations
use the corresponding scale path rather than a scalar reference scale.

## Diagnostics and interpretation

At minimum inspect:

- ordinary MCMC R-hat and effective sample size for `phi_slope`,
  `phi_rw_sd`, and scale summaries;
- `phi_rw_laplace_mh` acceptance and log-scale Laplace convergence for RW;
- switching and posterior probabilities for `phi="ssvs"`;
- agreement across independent chains;
- posterior scale paths and predictive calibration;
- sensitivity to the linear prior, random-walk variance prior, and scale-model
  probabilities.

A dynamic location can partially trade off with dynamic scale in finite
samples. Treat the four supplied models as a sensitivity analysis, use prior
predictive checks, and prefer model-averaged conclusions when SSVS switching
is adequate.

## Current scope

Time-varying log scale is supported for univariate GEV models under the
Fruehwirth--Schnatter parameterization. It is not yet enabled inside
`MultiSeriesModel`; requests fail early with a clear error. The four Uccle
series are therefore fitted independently in `examples/11_uccle_phi.py`,
which also makes their jobs and chains naturally parallel on PBS.
