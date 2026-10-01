# Model and computational conventions

One observation step is one input block. For a level and slope,

\[
\alpha_t=\alpha_{t-1}+\beta_{t-1}+s_\alpha\epsilon_{\alpha,t},\qquad
\beta_t=\beta_{t-1}+s_\beta\epsilon_{\beta,t},\quad \epsilon\sim N(0,1).
\]

A dummy seasonal state with period p obeys
`gamma[t] = -sum(gamma[t-1:t-p+1]) + s_gamma*epsilon[t]`.
The observation location is `alpha[t] + gamma[t]`. The initial level and rate
are at time zero, immediately before the first observation. The initial
seasonal cycle is parameterized at the first observation; its chronological
phase effects are zero-sum orthonormal contrasts. Static seasonality repeats
exactly. Dynamic seasonality allows departures from a fixed zero-sum cycle.

The first block defines phase zero, independent of the calendar month. Dates
must be regularly spaced and complete. A skipped block is not equivalent to
a missing observation: the current API rejects missing values and irregular
calendars rather than silently compressing time.

## Priors and units

`Normal(mean, sd)` acts on signed innovation amplitudes, not their variances.
The physical variance is q=s². Under a fixed SD tau, q is Gamma with shape 1/2
and scale 2 tau²; that induced law does not contradict the Normal prior on s.
The density concentrates toward small magnitudes but has no atom at zero.
Set a component static or off for an exactly zero innovation variance.

For pooling, tau has a half-normal hyperprior with underlying Normal SD A.
E(tau²)=A² and E(s²)=A². A is a calibration scale, not a fitted common
innovation variance. Different s_j remain possible. No initial rate is pooled.
Initial slope SD is in response units **per block**; `bx.plot(..., type="slope")`
multiplies by `10*steps_per_year` for a per-decade warming rate.

For h future steps, the innovation contribution to level variance is
`h*q_level + h*(h-1)*(2*h-1)/6*q_slope`. Uncertainty in the current state/rate
adds further forecast variance. The initial seasonal contrast SD d gives
phase variance d²(1-1/p), with covariance -d²/p between distinct phases.

The observation baseline variance has an inverse-gamma prior with density
`v^(-a-1) exp(-b/v)`. Seasonal log-scale effects are `C @ u`, with orthonormal
zero-sum C and independent `u ~ Normal(0, scale_prior_sd²)`. Thus the baseline
sigma is the geometric mean across phases, and scale_prior_sd is a contrast
SD, not directly the marginal SD of a phase effect. Shape xi has an untruncated
Normal prior. Negative xi gives a finite upper endpoint for maxima or a finite
lower endpoint for reflected minima.

## Exact-target updates

The state representation separates baseline coefficients and signed amplitudes
from standardized Gaussian state paths. Singular deterministic transitions
remain deterministic; no innovation-variance floor is inserted.

For Gaussian observations, conditioning gives a linear Gaussian state model
with phase-dependent observation variance. FFBS draws the full standardized
path. Coefficients are then sampled from their Gaussian regression conditional,
using prior whitening for numerical stability. A static Gaussian intercept
with known observation variance is covered by an analytic posterior test.

For GEV observations, let g and h be the first and second location derivatives
of the exact log likelihood at the current *mode iteration*. Construct the
pseudo-observation `z = eta + g/I` and pseudo-variance `1/I`, where `I=-h`.
Positive curvature regularization, variance caps and shift limits apply to the
**proposal construction only**. Kalman smoothing and a line search update the
mode. This construction is deterministic given the other parameters; it is
never initialized from the current sampled path.

With Gaussian state prior p0 and approximate likelihood Ltilde, the FFBS
proposal has density `q(x) proportional to p0(x)*Ltilde(x)`. Its independence
MH probability is

\[
\min\{1, L(x^*)\widetilde L(x)/(L(x)\widetilde L(x^*))\}.
\]

The state prior and proposal normalizer cancel. The proposal stays fixed
during all MH attempts in that update. Exact GEV support violations are
rejected. The Gaussian-likelihood special case is tested to accept every
proposal, with zero log correction differences. No uncorrected Laplace engine
is exposed.

GEV coefficients use an elliptical slice around a deterministic Gaussian
reference. The residual target is the **exact likelihood times coefficient
prior divided by that reference**. The implementation does not mistake
Gaussian preconditioning for a posterior approximation. Symmetric sign
switches jointly flip a Normal innovation amplitude and its standardized
path, leaving physical states unchanged.

Gaussian baseline variance is inverse-gamma conditional on all other blocks.
GEV log sigma and xi use stepping-out slice updates. For kappa=log(sigma),
the variance prior contributes `-2*a*kappa-b*exp(-2*kappa)`; this includes the
transformation Jacobian. Seasonal log-scale contrasts use an elliptical slice.

For J active amplitudes and S=sum(s_j²), v=tau² has the conditional

\[
v\mid s\sim\mathrm{GIG}((1-J)/2, A^{-2}, S),\qquad
p(v)\propto v^{\lambda-1}\exp[-(a v+b/v)/2].
\]

The code uses SciPy's standardized GIG or an exact rejection sampler for very
small S. It works in relative log units to avoid squaring tiny amplitudes.
Static or omitted components do not enter a pooled scale conditional. An
all-exact-zero active coefficient vector has an improper conditional for
this hierarchy; it is rejected rather than replaced with an arbitrary floor.

## Checks and interpretation

Saved metrics include retained-draw path acceptance, support rejections,
Laplace iterations and exact observation log likelihood. The summary contains
rank-normalized split/folded R-hat, bulk ESS and tail ESS. Constant parameters
have undefined diagnostics (NaN). `include_paths=True` also examines every
level, rate and seasonal state. ESS is conservatively capped at the saved
sample count; it is not a proof of convergence. There is no automatic
"converged" label.

Conditional posterior PITs and normal-score QQ plots are in-sample diagnostics,
not uniformity tests of held-out forecasts. `posterior_checks` compares paired
KS discrepancies for observed and replicated scores under the same posterior
draw. Its Bayesian p-values are descriptive, not classical fitted KS p-values.
Serial residual association is assessed separately; the state model does not
automatically eliminate it. Score correlations between channels are diagnostic
summaries, not copula estimates.

References: Carter & Kohn (1994), *Biometrika* 81, 541–553;
Frühwirth-Schnatter (1994), *Journal of Time Series Analysis* 15, 183–202;
Durbin & Koopman (2000), *JRSS B* 62, 3–56;
Murray, Adams & MacKay (2010), [Elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html);
Neal (2003), *Annals of Statistics* 31, 705–767;
Vehtari et al. (2021), *Bayesian Analysis* 16, 667–718.
See also the [Stan diagnostic definitions](https://mc-stan.org/docs/reference-manual/analysis.html).
