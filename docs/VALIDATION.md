# Validation evidence for BUCEX 1.6.3

The full source suite passed **245 tests** on 14 September 2026 (217.86 seconds).
`validation/pytest-1.6.3-source.xml` is the machine-readable result. The single
warning reports two retained daily TN>TX source observations during the data
aggregation test; the test verifies the documented behavior.

The wheel and source distribution also built successfully. Ten seasonal-scale,
copula, archive/restart and score-comparison checks passed after installing the
wheel outside the source checkout (1.67 seconds); see
`validation/pytest-1.6.3-installed.xml` and `installed-import-1.6.3.json`.
The installed loader returns all 1,572 primary-period months and six channels.

A final short-record initialization regression was added after that full run.
It checks that unobserved seasonal phases receive finite starting values. The
private-margin and affected hierarchical suites were rerun after that change;
see `validation/pytest-1.6.3-initialization.xml`. This brings the source test
collection to 246 tests, without treating targeted reruns as new independent tests.

## Independent numerical references

The maintained suite includes tests that calculate expected targets outside the
sampling code, not only checks of result shapes or self-consistent likelihoods.

| Contract | Reference |
|---|---|
| Private FS/SSVS Gaussian copula posterior | Direct coefficient integration, quadrature over observation scales and exact enumeration of zero/fixed slope models; sampled coefficient means and selection probabilities agree |
| Private GEV structural selection | Gumbel location/slope and zero/fixed-slope quadrature reference with effectively fixed scale and shape; sampled moments/model probabilities agree |
| Exact conditional copula likelihood | Conditional normal-density ratio and finite-difference GEV/Gaussian gradients and curvature, including reflected minima |
| Independence reduction | Fixed R=I and absent copula produce matching private-kernel draws with identical priors and random seed |
| Established continuous joint posterior | Independent multivariate-Gaussian, observation-scale quadrature and LKJ reference calculations in `test_copula_reference.py` |
| Predictive behavior | Seasonal calendar phase, original tail orientation, joint densities, covariance, risks, forecasts and ordering checks |
| Reproducibility | New scalar/private archives and warm starts, version checks and historical result fixtures |
| Paired score uncertainty | Whole-block resampling preserves repeated within-block dependence and flags insufficient blocks |

The numerical tests establish the implemented invariant targets in tractable
cases. They do not establish finite-run convergence for the complete Uccle study.
The old mixed hierarchical structural-MH proposal anchor was corrected in this
release; affected previous hierarchical results should be refitted.

## Executed short workflows

`validation/workflows-1.6.3.json` records observed output checks. The executed
workflows include:

- Gaussian and GEV univariate fits with seasonal observation scale, reports,
  figures, SSVS summaries, risk calculations and forecasts;
- the six-margin residual copula analysis and matched fixed-identity model;
- the baseline/tight-seasonal-scale/wider-shape prior sensitivity smoke grid;
- all eleven generating shapes from -.5 through .5 with approximate and exact
  univariate engines (22 short fits), under wider fitted shape support;
- zero versus correlated residual mixed-margin recovery with seasonal scale
  and private FS/SSVS, comparing fixed-identity and estimated copula fits;
- July 2019 endpoint comparisons under two shape priors and both engines,
  including fits ending before the selected event;
- latent/observation forecast uncertainty and complete-year aggregation;
- fixed-identity/copula held-out marginal, joint and compound-event scores,
  plus paired calendar-block comparison.

These use deliberately tiny budgets. Their numeric outputs are not empirical
paper findings and are excluded from the release. Scientific configs, including
the 100-replicate shape grid and full-record four-chain fits, remain protocols
awaiting adequate computation and diagnostics. No full scientific experiment
was silently shortened or declared successful from its smoke version.

## Reproduce and interpret

```bash
python -m pip install -e ".[plot,test]"
python -m pytest
python -m research.serra.univariate --series TXm TXx
python -m research.serra.copula
```

See `research/serra/README.md` for the complete study sequence and output guide.
The new private FS/copula backend requires complete aligned finite observations;
it does not inherit the older centered route's missing-data capability.

Four chains, a configured iteration count or a high MH acceptance rate cannot
certify convergence. Inspect R-hat/ESS, scientific contrasts, structural switching,
residual checks and held-out calibration. Report failed or stuck fits, and give
Monte Carlo uncertainty over simulation replicates. The paper's six result
boxes are intentionally pending.

No hard-order or asymptotic-tail-dependence claim is made for the Gaussian
copula. Posterior predictive variance may diverge when GEV shape approaches .5;
finite Monte Carlo variance summaries are not a proof of moment existence.
