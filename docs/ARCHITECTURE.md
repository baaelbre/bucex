# Package architecture

Model construction, inference and results have separate responsibilities.
Research scripts select scientific assumptions and call the public API.

| Location | Responsibility |
|---|---|
| `bucex/components/` | Level, trend, seasonality and regression declarations |
| `bucex/observation/` | Gaussian and GEV marginal distributions |
| `bucex/models/` | Scalar/multiseries declarations, fixed shared loadings, constrained departures and state compilation |
| `bucex/priors/` | Structural shrinkage, hierarchy, innovation and observation priors |
| `bucex/inference/` | FFBS, exact Laplace–MH, conditional updates and joint sampling |
| `bucex/core/`, `bucex/api/` | Fits, component access, forecasts, scientific contrasts and risks |
| `bucex/diagnostics/`, `bucex/plotting/` | Calibration, mixing, copula and ordering summaries, figures |
| `bucex/io/` | Checksummed result persistence and historical readers |
| `research/serra/` | Short configured research workflows built on those APIs |
| `tests/` | Mathematical and behavioral tests, historical result fixtures |
| `validation/` | Current-release execution evidence |

The univariate API remains the primary independent route. `MultiSeriesModel`
adds either hierarchical prior pooling or centered joint inference with
`JointPriors`. `Shared` stores a genuine common state with fixed original-scale
loadings. `Departures` represents individual effects in a weighted-zero-sum
contrast space. A residual Gaussian copula is an optional observation
extension; it is not the common state and does not impose ordering.

A correlated-observation model uses the complete copula likelihood in exact
MH corrections and parameter updates. With a fixed identity correlation it
reduces to conditional independence. Shared/correlated joint inference does
not currently select structural models through SSVS. Hierarchical mixed
models retain the FS selection route; time-varying GEV log scale remains a
univariate capability.
