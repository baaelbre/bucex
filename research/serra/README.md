# SERRA research with BUCEX 1.8.1

Use [START_HERE](../../START_HERE.md) for the current commands and output guide.
The common-specification workflow fits all six summaries through August 2026,
with private FS locations, repeating monthly observation scales and a joint
Gaussian copula. Separate shared level, slope and seasonal hyperpriors pool
regularization; they do not impose common realized trajectories or initial
seasonal patterns. Repeating monthly observation scales remain distinct.

| Task | Driver | Configuration |
|---|---|---|
| Execute all stages on a short record | `prior_assessment --stage all` | `hierarchy/smoke.json` |
| Inspect the candidates/dates before fitting | `prior_assessment --stage plan` | `hierarchy/pilot.json` |
| Full-record prior sensitivity | `prior_assessment --stage sensitivity` | `hierarchy/pilot.json` |
| Continue with matched historical forecasts | `prior_assessment --stage predictive --run PATH` | Resolved settings from the saved assessment |
| Compare seasonal pooling and half/double seasonal anchors | `prior_assessment --stage sensitivity` | `hierarchy/seasonality.json` |
| Match residual independence and copula dependence | `prior_assessment --stage all` | `hierarchy/dependence.json` |
| Check constant scales and fixed location seasonality | `prior_assessment --stage all` | `hierarchy/adequacy.json` |
| Check initial slope, shape, scale, correlation and hyperpriors | `prior_assessment --stage sensitivity --variants ...` | `hierarchy/reviewer_sensitivity.json` |
| Confirm quarter/half anchor robustness | `prior_assessment --stage all` | `hierarchy/confirm.json` |
| Inspect final resource/model settings | `preflight` | `hierarchy/final.json` |
| Fit the candidate paper model | `copula` | `hierarchy/final.json` |
| Fit the alternative half-anchor paper model | `copula` | `hierarchy/final_half.json` |
| Same hierarchy with R=I | `copula --independence` | `hierarchy/final.json` |
| Six separate common fixed-prior analyses | `univariate` | `hierarchy/independent.json` |
| Matched fixed-prior copula fallback | `copula` | `hierarchy/fixed_half.json` |
| Observed-data Figures 1 and 2, no MCMC | `explore` | `revision/exploration.json` |
| Rebuild comparison tables/figures | `prior_assessment --stage report --run PATH` | Saved assessment configuration |
| Re-report a saved posterior | `report` | Saved fit and its config |
| Fitted-model manuscript panels | `figures` | `revision/figures.json` with explicit source reports |

Every driver runs as `python -m research.serra.DRIVER`; use `--help` for its
arguments. Configuration paths start with `research/serra/config/`.

The short research modules declare a study and orchestrate public BUCEX calls:

- `models.py`: channel observations, named latent components, prior declarations
  and inference options. This is where study-specific assumptions belong.
- `prior_assessment.py`: one candidate list, explicit forecast origins and
  completed-stage tracking. It calls sensitivity/validation and asks the
  package's `SensitivityReport` to compare the compact outputs.
- `sensitivity.py`, `validate.py`: ordinary model construction, fitting and
  reporting at matched settings. Historical fits learn hyperparameters on
  training data only. They do not reuse the full-record posterior.
- BUCEX owns the shared-prior distribution and sampler, unconditional prior
  draws, diagnostics, forecast/risk calculations, comparison tables and plots.

Four process-parallel chains work without Slurm. The Python API remains serial
unless `chain_workers` is explicitly increased. Existing configurations remain
unchanged for reproducibility; use `hierarchy/` for the new analysis rather
than assuming older files acquired the tighter priors automatically.

The pilot does not save large posterior archives. It keeps declared settings,
compact CSV/JSON reports, PNG figures and compressed trace tables. The final
configuration saves the fit as well. Share those compact outputs first; no need
to upload large `.bucex` archives just to inspect convergence or sensitivity.

## Targeted supplementary work

Current matched checks use `hierarchy/seasonality.json`, `adequacy.json`,
`dependence.json` and selected groups from `reviewer_sensitivity.json`; their
exact commands are in START_HERE. They share the current quarter-anchor
reference. If another reference is chosen, propagate its resolved priors before
new comparison fits. Existing tools also remain available when needed:

- `sensitivity` with `revision/structure.json` for fixed/evolving seasonality;
  `revision/sensitivity_monthly.json` and `sensitivity_copula_monthly.json` for
  shape, observation-scale, copula and other prior choices.
- `endpoint` for the observed 2019 record and finite-endpoint sensitivity;
  `forecast_check` for forecast-width checks.
- `model_comparison`, `compare`, and the controlled-recovery drivers for
  previously planned supplementary experiments. A full simulation study is
  not required to run the focused hyperprior assessment.

For a targeted new sensitivity use a resolved copy of the chosen current
configuration; historical configurations intentionally keep their old priors.
See [SHARED_SHRINKAGE.md](../../docs/SHARED_SHRINKAGE.md) for interpretation,
limitations, API use and the exact conditional density.
