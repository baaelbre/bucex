> For 1.8.2 use [START_HERE](../START_HERE.md) and [physical prior calibration](PRIOR_CALIBRATION.md). The earlier protocol below records previous workflows; its bounded-shape and three-scale assumptions are not the new draft defaults.

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
| `inference/chains.py` | Shared process executor, legacy seed preservation, thread limits and chain assembly |
| `inference/fit/` | Existing scalar, hierarchical and shared-state backends |
| `core/`, `api/` | Stable fit, component, risk, predictive and persistence interfaces |
| `diagnostics/`, `plotting/` | Mixing, calibration, residual dependence, ordering and figures |
| `reporting/` | Compact report readers, configured manuscript panels and source manifests |
| `io/` | Checksummed schema 2.12 archives and historical readers |
| `datasets/` | Daily validation, monthly aggregation and loading |
| `exploration.py` | General monthly empirical cycles, detrended IQRs and sample accounting |
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

## Descriptive exploration in 1.7.4

`exploration.py` computes observed calendar-month means/quartiles and separate
within-month linear detrending summaries. `MonthlyExploration` exposes these
tables and delegates plotting to `plotting/exploration.py` and persistence to
`reporting/exploration.py`. Data values keep their original signs. No fitting,
prior calibration or inferential interval is introduced. The SERRA `explore`
driver selects input series and explicit time windows from JSON, then calls
this API. The plotting style is scoped and Matplotlib is imported on demand.

## Execution and assessment in 1.7.3

`MCMC.chain_workers` is an execution option, not a model setting. Each backend
uses the same `independent_chains` policy; transition and likelihood code never
creates a process pool. Seed contexts supply the original child streams and
chain positions to either the serial loop or a spawned single-chain worker.
Diagnostics and initial states retain the chain axis when results are merged.

`diagnostics/sensitivity.py` provides descriptive prior/posterior comparisons
and strict pairing of predictive losses. `reporting/sensitivity.py` reads
explicitly mapped report directories and preserves warning and source
information; `sensitivity_plots.py` renders the common manuscript style.
`research/serra/prior_assessment.py` only declares the study stages, delegates
fitting/validation, checkpoints stage status and requests reports.

## Shared innovation shrinkage (1.8.0)

`priors/shrinkage.py` defines the distribution and conditional normal priors;
`MarginalPriors` owns the optional cross-response declaration.
`inference/fit/shrinkage.py` owns the normalized log-hyperparameter conditional
and its slice update; `inference/fit/marginal.py` composes it with the existing
private FS and copula sweeps. No inference logic is placed in SERRA scripts.

`diagnostics/shrinkage.py` draws the unconditional hierarchy and summarizes its
posterior. `reporting/shrinkage.py` exports numerical tables/traces and figures;
`SensitivityReport` compares saved joint or univariate reports by response and
matched forecast case. `calendar_origin_splits` resolves explicit training dates.
Research declarations assemble these public APIs in `serra/models.py`; one
candidate list drives posterior and predictive studies in `prior_assessment.py`.
The optional field leaves ordinary univariate priors and existing fits intact.

In 1.8.1, the research default selects all three supported components (level,
slope and seasonal). No conditional-density change is needed. Study configs
declare two-component pooling, seasonal anchor variations and fixed-seasonality
checks explicitly. The comparison plot layer handles absent pooled components;
it never inserts a fake posterior for a component that was not estimated.
