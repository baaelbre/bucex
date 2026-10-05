# Extending BUCEX

The extension boundary is a matrix contract, not a list of recognized classes.
`Model` accepts any object implementing the `Component` protocol. Its
`build(steps, priors, exog)` method returns a `ComponentBlock`:

| Field | Meaning |
|---|---|
| `transition` | Constant d × d physical transition F |
| `design` | T × d observation design H; may depend on covariates |
| `initial` | d × p mapping B from independent initial coefficients to x₀ |
| `coefficient_names`, `priors` | p names and Normal/Fixed coefficient priors |
| `state_names` | d distinct physical-state names |
| `innovations` | Groups of loading matrices R and Normal/Fixed signed-amplitude priors |
| `outputs` | Optional named linear functions of the physical state |
| `initial_outputs` | Optional named linear functions of initial coefficients |

For each group g, the compiler creates standardized states
`z[g,t] = F z[g,t-1] + R[g] epsilon[g,t]`, starting exactly at zero.
The physical state is `F**t B theta_initial + sum(s[g] z[g,t])`.
Unreachable coordinates are removed, while deterministic lag transitions stay
deterministic. The observation design for the path update and the regression
design for the coefficient update follow from the same maps. Forecasting uses
the physical F and R; results and plots use the declared output maps.

This supports a d-dimensional isotropic cycle under one amplitude or separate
random-walk coefficients under several amplitudes. All innovations within a
channel and between channels are conditionally independent in the implemented
backend. Group names identify the amplitudes that a `Pooling` specification
can regularize together. The units and scientific suitability of that pooling
remain the model author's responsibility.

## Example

See `examples/custom_component.py`: a damped level implements the contract,
fits, forecasts and produces a native plot without changes to package source.
Decorate a dataclass with `@bx.register_type` to enable JSON/NPZ round trips.
The extension must be imported and registered before loading its archive.
Loading does not import or execute classes named by a file. Prefer distinctive
registered names; collisions are rejected.

Covariate-aware components declare `features`, a tuple of column names. The
public fitting/prediction functions align these columns and validate future
inputs. Use the supplied `exog` frame inside `build`; never read user data from
a global variable. Custom components must be importable in every worker when
using process parallelism.

## Other stochastic distribution parameters

`Latent` is a named predictor declaration. `bx.compile_parameter(model, name,
steps, exog)` can compile its linear predictor. `bucex.inference.plan(model)`
separately checks which conditional updates are implemented.

Version 1.0 supports identity-linked dynamic location. Dispersion is constant
or repeats by phase; GEV shape is constant. To support, for example,
`scale=Latent([LocalLevel(...)], link="log")`, add a scale-path update that
uses the chain rule for log scale, the correct support and Jacobians, and an
exact target correction. Extend prediction, parameter output and prior
simulation to evaluate that predictor through its link. The existing linear
state compiler and smoother can be reused. The current backend rejects such
fits explicitly; it does not ignore the declaration.

A new observation family implements `ObservationFamily` probabilities,
quantiles and location derivatives. It also needs matching observation-
parameter updates and a declared backend capability. A density method alone
is insufficient for a valid MCMC implementation.

## Scope of the current matrix contract

F and innovation loadings are constant within a component; H may vary with
known covariates. Time-varying F/Q, unknown cycle frequencies, nonlinear state
transitions, arbitrary observation links, irregular steps and missing-data
updates require deliberate extensions and target checks. They are not claimed
as implemented features. The public layers isolate where that work belongs.

## Extension verification

Before supporting a new component, check its deterministic propagation,
innovation covariance and initial-state interpretation against an independent
calculation. For a linear Gaussian model, compare filtering/posterior moments
with dense Gaussian conditioning. Exercise fitting, future covariates,
serialization and named plots together. New non-Gaussian kernels additionally
need independent posterior-target checks; a successful short run alone does
not establish correctness.
