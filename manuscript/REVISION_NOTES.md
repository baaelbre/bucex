# Manuscript revision for BUCEX 1.9.5.1

The full supplied manuscript is retained, with coordinated changes to the
abstract, introduction, prior specification, shrinkage explanation, discussion,
conclusion and supplementary prior calibration. The main text gives the
scientific meaning of the calibration; detailed simulation methods and four
completed figures are in Supplement S1.5.

## What changed

- Distinguishes fixed calibration scales A, learned shared prior SDs tau,
  response-specific innovation SDs |s| and the private initial-rate variance.
- Explains why weak informativeness concerns implied temperature changes,
  not arbitrarily large parameter variances.
- Connects hierarchical regularization with predictive calibration, citing
  Gabry et al. (2019), Gelman (2006), and Gelman, Simpson and Betancourt (2017).
- Includes actual prior results from 10,000 joint replications per setting.
  The four figures are supplied in PDF and require no posterior fit.
- Explains the initial seasonal lag basis and its larger induced JJA variance.
- Reports the broad SD-20 initialization honestly: TXm's first-year latent
  range has median 52.10°C (95% interval 14.60–120.53°C). For the existing
  SD-2.25 sensitivity, the corresponding values are 5.86°C and 1.65–13.56°C.
- Avoids claiming that hierarchical shrinkage guarantees robustness or that
  all aspects of the current prior are physically calibrated.

## What still requires the scientific fits

The posterior estimates, reported acceptance rates, risks and validation
numbers in the supplied manuscript have been preserved, not recomputed or
certified for a new fit. Existing sensitivity-result placeholders remain.
Only the prior simulation results added in S1.5 were generated for this patch.
The original non-prior figure assets were not supplied with the pasted
manuscript; their existing figure placeholders remain in the PDF. Add the
original assets under `figures/` using the filenames already referenced by
the source.

Before submitting, insert the converged posterior results and original figure
assets; assess prior sensitivity of trajectories and risks; decide whether the
initial-cycle prior should be recalibrated; and refresh posterior text if that
scientific specification changes. Prior simulation alone does not establish
posterior stability, forecast calibration or physical ordering of summaries.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript-1.9.5.1.tex
```

The supplied PDF compiles without unresolved cross-references or overfull boxes.
The bibliography is embedded in the LaTeX file. The source bundle includes all
four new prior figures. Full simulation code and numerical summaries accompany
BUCEX 1.9.5.1.
