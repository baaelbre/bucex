# Supplementary checks after the first draft

Start with [START_HERE](../START_HERE.md). These commands are deferred work,
not prerequisites to begin writing. They compare declared alternatives; they
do not search for the smoothest posterior or a desired acceleration result.

`draft/sensitivity.json` matches the main seasonal copula, monthly scales,
four shared regularization scales and unrestricted normal shape prior. Inspect
the resolved settings before fitting:

```bash
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference hyperprior_tighter hyperprior_wider --stage plan
```

Run targeted groups as needed, one at a time:

```bash
# Same anchors, log-SD log(1.5), log(2), log(3).
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference hyperprior_tighter hyperprior_wider --stage sensitivity

# Half/double the initial-slope SD anchor; or turn off its pooling only.
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference initial_slope_half initial_slope_double fixed_initial_slope --stage sensitivity

# Shape prior SDs .15/.30/.60, all unrestricted.
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference xi_normal_tighter xi_normal_wider --stage sensitivity

# Shape bounds: unrestricted, [-.5,.5], [-1,1]; uniform on [-.5,.5].
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference xi_bounded_half xi_bounded_one xi_uniform --stage sensitivity
```

The last group directly changes support, addressing the reviewer's shape-bound
question. Changing normal SD alone within unchanged bounds does not do so.
Report trajectories *and risk*, including the 39.7°C threshold and endpoint
uncertainty. Infinite endpoints for xi>=0 are not to be clipped or dropped.

After a posterior assessment, continue only useful comparisons with its own
saved settings and training-only historical fits:

```bash
python -m research.serra.prior_assessment --run "ASSESSMENT_DIRECTORY" --stage predictive
```

Or regenerate compact comparisons without fitting:

```bash
python -m research.serra.prior_assessment --run "ASSESSMENT_DIRECTORY" --stage report
```

Further matched draft configurations:

```bash
# Quarter versus half innovation anchors (initial-slope anchor unchanged).
python -m research.serra.prior_assessment --config research/serra/config/draft/anchors.json --stage sensitivity

# Monthly versus constant dispersion; fixed versus evolving location seasonality.
python -m research.serra.prior_assessment --config research/serra/config/draft/adequacy.json --stage sensitivity

# R=I, constant R, seasonal R; same four shared prior scales.
python -m research.serra.prior_assessment --config research/serra/config/draft/dependence.json --stage sensitivity
```

The hierarchy-only R=I fit is not six independent posteriors: the learned prior
scales still couple the responses. Seasonal copula effects have normal priors
on partial-correlation contrasts, not four independent LKJ priors; their prior
depends on channel ordering. If seasonal effects matter substantively, assess
their prior width and a scientifically sensible channel permutation. Physical
ordering diagnostics must still be reported; do not sort simulated summaries.

For modest monthly-scale and copula-baseline prior checks:

```bash
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference monthly_scale_prior_half monthly_scale_prior_double --stage sensitivity
python -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --variants reference lkj_2 lkj_4 --stage sensitivity
```

These use 500 warmup + 500 retained draws in each of four parallel chains.
Prior sensitivity is uninterpretable if differences are dominated by poor
mixing. Increase only the relevant comparison's budget, in a new configuration
and output directory, before making a strong conclusion. The full historical
comparison protocol has four origins including the 2019 record; this is more
work than the two-origin first-draft check. Nothing here launches a full
simulation study.

The existing `hierarchy/` and other research configurations remain available
for broader experiments. In 1.8.2, inherited current defaults use unrestricted
normal shape priors and initial-slope pooling unless explicitly disabled.
For exact reproduction of an earlier run, use its saved resolved `config.json`
and package version, not assumptions about inherited defaults.
