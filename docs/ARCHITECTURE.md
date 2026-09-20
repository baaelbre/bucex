# Package architecture

Scientific declarations, inference and results have distinct responsibilities.
Research scripts select assumptions and call the public API.

| Location | Responsibility |
|---|---|
| `components/`, `models/` | Components, scalar/private/shared declarations and compilation |
| `observation/` | Gaussian/GEV likelihoods and seasonal/secular scale declarations |
| `priors/` | FS shrinkage/selection, `MarginalPriors`, hierarchy and joint priors |
| `inference/fit/conditional_margin.py` | Exact copula-conditional marginal likelihood and derivatives |
| `inference/fit/marginal.py` | Private continuous/optional SSVS updates, seasonal scale, shape and residual correlation |
| `inference/fit/continuous.py`, `scale_path.py` | Whitened coefficient updates, rescaling ASIS, secular scale paths |
| `inference/fit/marginal_adapter.py` | Scalar API and restart adaptation |
| `inference/fit/` | Existing scalar, hierarchical and shared-state backends |
| `core/`, `api/` | Stable fit, component, risk, predictive and persistence interfaces |
| `diagnostics/`, `plotting/` | Mixing, calibration, residual dependence, ordering and figures |
| `reporting/` | Compact report readers, configured manuscript panels and source manifests |
| `io/` | Checksummed schema 2.11 archives and historical readers |
| `datasets/` | Daily validation, monthly aggregation and loading |
| `research/serra/` | Configured scientific workflows built on the package |
| `tests/`, `validation/` | Independent numerical/API checks and execution evidence |

`Model` remains the univariate entry point. A scalar `SeasonalScale` or `LogScale` fit uses
one internal private channel and is returned as an ordinary scalar `FitResult`.
`MultiSeriesModel` with `MarginalPriors` uses private continuous FS or exact-SSVS trajectories.
A copula changes their conditional likelihoods; each block update therefore
feeds residual dependence back into states and selection.

`HierarchicalPriors` pools prior parameters without introducing common states.
`Shared` and `Departures` describe actual common and constrained private states
under the separate continuous `JointPriors` route. These modes are not silently
interchanged. Uccle wrappers are convenience calls over general declarations.
Scientific assumptions and result summarization belong in the package; study
windows, sensitivity grids and output locations belong in research configs.

## Parameter declarations in 1.7

`parameters.py` binds named Constant/Latent declarations to canonical model
representations. `priors/evolution.py` calibrates ancillary evolution in link
units; `inference/fit/evolution.py` updates a Gaussian structural evolution
against a supplied likelihood callback. `diagnostics/periods.py` provides
paired climate-period estimands. These general computations stay outside
`research/serra`, which declares the fixed-scale paper protocol.

In 1.7.2, `plotting/style.py` owns the scoped publication style and `reporting/` assembles configured panels without fitting. `diagnostics/calendar.py` owns monthly PIT and held-out coverage summaries. The study figure script selects reports and reads JSON recipes.
