# Architecture

Release 1.5.2 keeps one model compiler, one fitting entry point, and one result
type. Analysis scripts compose these public APIs directly.

```text
bucex/
  api/                 fit, forecast, prediction
  components/          trend, seasonal, regression components
  models/              Model, MultiSeriesModel, Channel, compilers
  observation/         Gaussian and GEV families
  priors/              univariate and hierarchical priors
  inference/           plans, FFBS, Laplace, Laplace-MH, PGAS, samplers
  core/                FitResult and numerical primitives
  diagnostics/         MCMC, scores, PIT, LFO
  datasets/            Uccle loaders and direct fit helpers
  io/                  checksummed non-pickle archives
  plotting/            FitResult-driven figures
```

## Fit lifecycle

1. `fit()` receives a declarative model and data.
2. The compiler creates semantic states, transitions, and innovation loadings.
3. `inference_plan()` rejects incompatible engine/parameterization choices and
   declares exactness.
4. Prior resolution constructs univariate or hierarchical graphs.
5. The selected engine produces states, parameters, and diagnostics.
6. Results are normalized immediately into `FitResult`.

FS auxiliary paths and signed coefficients never replace the centered
scientific state stored in `FitResult`.

## Conditional GEV log scale

`GEV(phi=...)` declares the log-scale model without changing `fit()`. The
structural FS kernel owns the location state; `inference/fit/phi.py` owns the
conditional process \(\phi_t=\log(\sigma_t)\). Its state has stable
coordinates for stationary, linear, and random-walk models, so another scale
model can be added behind the same declaration/result boundary.

The random-walk scale update has its own iterated-Laplace/Kalman smoother. In
exact engines it is an independence proposal with an exact-likelihood MH
correction. Scale SSVS is a product-space update whose inactive coordinates
use proper prior pseudo-priors. This is separate from structural SSVS for
location innovations.

`FitResult.phi_draws()` and `sigma_draws()` normalize all four modes to a
draw-by-time path. Forecasting consumes the same result contract: stationary
scale is held fixed, linear scale is extrapolated, and RW scale is propagated.

In the general univariate sampler, the prior type and parameterization jointly
select the process-scale update. `InverseGammaVariance` receives its conjugate
full-conditional Gibbs draw in a centered sweep; other centered priors and all
non-centred scale sweeps retain their log-scale MH updates. The public prior API
therefore declares the mathematics without a separate `conjugate` switch.

## Hierarchical sampler

`MultiSeriesModel` compiles named channel blocks. One Gibbs sweep updates:

1. each channel path by Gaussian FFBS, approximate Laplace, or PGAS;
2. channel component allocations;
3. observation parameters;
4. population Dirichlet probabilities;
5. optional shared half-t slab multipliers.

Channels may update concurrently because they are conditionally independent
given hierarchy values. They still retain separate latent paths. The default
componentwise model pools level, slope, and seasonal decisions separately.

## Numerical stability and singular support

Gaussian filtering uses symmetric/Joseph-form covariance operations and
repairs only small round-off eigenvalues. Materially indefinite matrices raise
instead of being silently projected.

Exact SSVS creates deterministic transition directions. PGAS evaluates
ancestor moves on the affine support of active innovations. The conditioned
predecessor is retained when optional candidates are off-support. This is an
algorithmic requirement, not a numerical convenience.

Laplace-MH factors its state update into three reusable operations:

1. deterministically build the final pseudo-Gaussian smoother;
2. draw a complete trajectory from that fixed smoother by FFBS;
3. evaluate `exact log likelihood - pseudo log likelihood`.

The common state prior cancels in the MH ratio, including singular directions.
FS Gaussian filter arrays are cached so repeated `mh_steps` reuse one forward
pass. If the zero FS path crosses a finite GEV endpoint, proposal construction
starts from a deterministic support-feasible path derived without consulting
the current chain trajectory. Approximate `laplace` retains its historical
support-draw loop, while exact `laplace_mh` makes one untruncated proposal per
step and treats endpoint violations as rejections. An unrecoverable numerical
failure in an exact state update aborts the fit before a restored state can be
saved as a new draw.

## Analysis-script lifecycle

The 13 files under `examples/` declare models, priors, simulation truths,
fit calls, summaries, and figures in one readable sequence. Their scientific,
sampling, figure, and output choices come from the JSON files under
`examples/config/`; there is no second environment-variable configuration
layer. Seven examples form the COMPSTAT analysis, one exposes the
centered/inverse-gamma random-walk benchmark, two exercise exact Laplace-MH,
examples 10–11 isolate log-scale sensitivity for simulations and Uccle, and
example 12 provides exact Gaussian FFBS for Uccle TXm/TNm. Thirteen matching
Bash/PBS pairs invoke those exact files.

For a multi-chain job, `job_scripts/run_parallel_chains.py` creates one
temporary one-chain JSON per independent process, waits for all processes, and
then asks the same numbered example to combine the saved `FitResult` objects
and make the final tables and figures. The selected source JSON remains the
sole scientific and computational specification. Scheduler code controls
resources only. The runner sits beside the PBS files; there is no separate
`hpc/` directory or second settings layer. Automatic run IDs include the
configuration name and PBS job ID, so independent scenarios or Uccle series
cannot collide when they start in the same second. Examples 08 and 09
deliberately mirror the scientific contracts and artifact trees of examples
03 and 05, so engine comparisons do
not silently change the data, priors, or summaries.

## Persistence

Schema 2.7.0 archives contain allowlisted JSON metadata plus compressed NumPy
arrays, verify SHA-256 checksums, and load with `allow_pickle=False`. The model
is recompiled from its stored declaration and observations. `warm_start()`
exports one compatible univariate or multiseries draw without changing the
next fit's target. Univariate FS exports include the complete centred location
path and, where applicable, log-scale path and scale-model state. Readers for
all previously supported archive schemas remain available.

## Extension rule

A new component compiles into the common state contract; a new engine consumes
that contract and returns standard result blocks. Cross-channel dependence must
be an explicit observation/transition feature. It must not be hidden inside a
workflow or a prior-pooling label.
