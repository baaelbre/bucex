# The 1.7.1 fix and the research decisions

## What was wrong, and what changes

The previous continuous-FS GEV coefficient update used a Gaussian reference
obtained by expanding the likelihood at `eta = y`. For the long TXn record,
that reference could be far from the exact conditional coefficient posterior.
The likelihood/reference slice correction was present: this was an efficiency
defect, not a missing posterior correction. Tiny moves can nevertheless make
finite runs unusable for posterior inference.

Version 1.7.1 constructs the reference near the **exact conditional mode**,
including the normal conditional prior and Gaussian-copula correction where
applicable. Optimization uses preconditioned coordinates and the reference
covariance uses conditional curvature. The sampler retains the exact target
divided by the Gaussian reference inside its elliptical slice. The mode is
never substituted for a posterior draw. Gaussian coefficient updates remain
direct Gaussian draws.

The reference starts deterministically, never from the coefficient vector being
updated. Choosing an ellipse from the current vector and treating it as fixed
would generally invalidate this construction. Paths, observation parameters,
copula conditionals and shrinkage mixing variables remain fixed during this
update. Imperfect optimization or deterministic fallback affects efficiency,
not the invariant target. Support repair and curvature regularization change
only the proposal; they do not truncate the posterior or floor its prior scales.

`bucex/inference/fit/coefficient_reference.py` constructs the proposal;
`continuous.py` uses it for independent and copula-conditioned GEV margins.
The public model/fit/forecast/archive interfaces remain intact. No PGAS, SSVS
or ASIS is added to the paper workflow. Existing optional ASIS remains available.
The existing `Laplace(max_iterations=30, tolerance=1e-5)` controls also govern
reference construction. Its optimizer diagnostics are distinct from MCMC
convergence diagnostics.

### Evidence and limits

Three saved TXn states were held fixed while old and new coefficient kernels
each ran for 1000 updates with the **same .02 level prior**. The production
reference converged in six iterations at each state.

| Conditional diagnostic | Previous reference | 1.7.1 reference |
|---|---:|---:|
| Level coefficient lag-one correlation | .983–.988 | .086–.093 |
| Slope innovation coefficient lag-one correlation | .988–.993 | .017–.089 |
| Mean slice likelihood evaluations | 6.38–7.03 | 1.08–1.13 |

These are conditional-kernel measurements with paths and nuisance parameters
fixed, not full-chain ESS or a production speed-up estimate. Constructing the
reference costs time at every outer iteration. Full chains can still exhibit
dependence between paths, coefficients, scale and shape. See
the archived 1.7.1 release validation for these historical measurements. The current `validation/RELEASE_VALIDATION.md` records only checks executed for 1.7.4.

## What .01 means

The level innovation SD is `abs(s_level)` in the FS representation. Its new
normal prior is

\[
s_\ell\sim N(0,0.0148260^2),\qquad
\operatorname{median}(|s_\ell|)=0.01.
\]

The innovation variance is `s_level**2`. Neither variance nor SD is fixed to
.01. There is no point mass at zero. The slope innovation median remains
.00005, seasonal innovation median .02 and initial slope SD .0025 per month.
Initial level is N(0,20²); initial seasonal coordinates are N(0,2.25²).
Observation variance remains IG(2,2); shape remains N(0,.3²) on [-.5,.5].

At a fixed level SD equal to its prior median, accumulated level innovations
over 120 months have SD `sqrt(120)*median`: approximately .055, .110 and .219 °C
for medians .005, .01 and .02. These illustrate one conditional contribution,
not total prior-predictive uncertainty; slope, seasonality, initial coefficients
and observations contribute separately.

Tightening level shrinkage can reduce roughness, but can also redirect variation
into the slope or seasonal components. An apparently clearer acceleration
signal can be a decomposition effect. Tightening slope innovations regularizes
changing rates; tightening the initial slope regularizes the initial rate.
Neither should be chosen because the curve then supports a preferred narrative.

Use .01 as the declared reference and .005/.02 as focused sensitivities. Do not
keep tightening until a curve looks smooth. Compare period warming, slope
changes, calendar-specific location changes and risks, with Monte Carlo error
and predictive calibration. Stable period warming alongside sensitive local
slopes supports stronger warming conclusions and more cautious acceleration
conclusions. Substantial prior sensitivity is itself a result to report.

## The reviewer asked for both assessments

The concern was weakly identified, near-zero evolution variances. Address both:

1. **Prior-to-posterior comparisons** for each physical process SD and variance,
   with relevant initial coefficients and interpretable axes. Rapid random sign
   switching alone does not establish learning or good mixing.
2. **Refit sensitivity** to hyperparameters, comparing trajectories and scientific
   quantities, including risks. Similar posterior means can conceal changes in
   uncertainty or slope decomposition.

`sensitivity/level.json` changes only level shrinkage. `paper.json` changes all
innovation medians together and compares median-matched normal/lasso/TG families.
`targeted.json` checks initial slope, initial seasonality, observation variance
and shape prior/support. Joint counterparts refit the entire copula likelihood.
Prior/posterior tables include SD and variance. Such plots cannot replace refit
sensitivity, and neither can replace convergence checks.

## Where monthly scales fit in the story

The central question is: **how do distributions of monthly temperature means
and extrema evolve, and what does this imply for event probabilities?** Six
summary distributions do not identify the complete daily temperature density.

Introduce both the constant-scale model and its monthly-scale extension in
Methods. The extension has

\[
\sigma_{jt}=\sigma_j\exp\{\delta_{j,m(t)}\},\qquad
\sum_{m=1}^{12}\delta_{jm}=0.
\]

Each series has its own regularized calendar effects, repeating across years.
This is seasonal heterogeneity, not long-term scale evolution. Location retains
its own level, slope and seasonal states. The model distinguishes seasonal
shifts in the expected summary from seasonal differences in its variability.

Begin Results with computational reliability and a short adequacy comparison.
If constant scale leaves substantial month-dependent residual spread and
monthly scale improves held-out calibration, use the latter for the main
scientific risks and retain constant scale as a baseline. Detailed comparisons
can be supplementary, but a material change to the preferred risk model must
be visible in the main text. A copula cannot compensate for bad marginal scales.

A focused results sequence is:

1. Adequacy and the selected scale specification, supported by a small matched
   predictive comparison, not an exhaustive search over structural models.
2. Period warming, seasonal location changes and differences among means and
   extremes, distinguishing robust changes from sensitive slope decompositions.
3. Residual dependence and the joint posterior, then month-specific, annual
   and compound risks where calibration supports them.
4. A concise robustness statement; detailed prior, shape, endpoint and seasonal
   structure checks in the supplement.

The methodological contribution is the coherent distribution-specific structural
analysis, exact-target computation, joint dependence and interpretable risks
with uncertainty. FS parametrization, shrinkage and copulas are established
tools. Demonstrated inferential/predictive benefits and validation make the
revision persuasive; adding optional models alone does not establish novelty.

## Run order

For the current six available 1.7.1 independent runs, follow `START_HERE.md`: TNm with monthly scales first; remaining margins; matched joint independence/copula fits; targeted prior and structural sensitivity; held-out predictions and reviewer experiments. Existing draws can be re-reported without refitting just to update figures. Judge physical SDs and scientific-target
diagnostics, not signed coefficients alone.

All primary data extend to August 2026. Annual forecasts and validation must
not treat January–August 2026 as a complete observed year. Saved configurations
remain authoritative, including explicitly requested older priors.
