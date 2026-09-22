# BUCEX 1.8.1 release verification

Verification date: 22 September 2026. These checks establish software execution
and regression behavior. No publication-length temperature analysis was run.

## Checks performed

| Check | Result |
|---|---|
| Complete source test suite | 370 passed, no failures or skips |
| Installed-wheel shared/seasonal shrinkage and historical archive tests | 26 passed, no failures or skips |
| Installed package versus verified source | All 126 package files match byte for byte |
| START_HERE research command argument syntax | 32 commands parsed successfully |
| Full configured six-response smoke workflow | Four posterior fits and four historical refits completed; four process-parallel chains |
| Seasonal hierarchy workflow | Four posterior fits and four historical refits completed with short chains |
| Structural adequacy workflow | Three posterior fits and three historical refits completed with short chains |
| Dependence workflow | Two posterior fits and two historical refits completed with short chains |
| Figure inspection | Three-component shared-prior panels and mixed two/three-component panels rendered; absent seasonal hyperparameter labelled `not pooled` |

The complete smoke used the distributed `hierarchy/smoke.json` unchanged:
January 2023–August 2026, 3 warmup and 4 retained iterations per chain, with one
12-month historical block. It produced 635 PNG figures plus compact tables.
Both pooled candidates exported level, slope and seasonal shared medians.
All four candidates correctly retained `needs_review` convergence status.

The additional seasonal, adequacy and dependence execution checks used their
distributed variant definitions with the same short data/forecast window,
2 chains, 2 warmup and 4 retained draws, serial execution and no figure export.
They check reporting when a pooled component is absent, fixed repeating
seasonality, constant observation scale, and fixed R=I. These tiny chains are
not evidence for any model or prior choice.

The source tests include exact hyperparameter conditional calculations,
seasonal coefficient mapping and integrated prior draws, separate channel
innovation SDs, parallel/serial parity, archive round trips, warm starts,
fixed-seasonality forecasts, configuration isolation and comparison plots.
All six production-plan data windows resolve to 1,616 monthly observations
ending in August 2026, with the declared historical cutoffs.

## Evidence and reproduction

- `pytest-1.8.1-source.xml`: complete source test result.
- `pytest-1.8.1-wheel.xml`: installed-wheel targeted regression result.
- `wheel-source-parity.json`: verified package file list.
- `cli-commands.json`: research commands checked for argument syntax.
- `smoke-1.8.1.json`: resolved smoke plan, stage status and compact output list.
- `workflow-1.8.1-checks.json`: resolved additional short-check plans and status.

From the extracted source directory:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/smoke.json --stage all
```

The test environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, Matplotlib 3.10.8, threadpoolctl 3.6.0 and pytest 9.1.1 on Linux.
The package's declared Python support remains >=3.10; this release was not
executed on every supported Python version or on biobot itself.

The source suite emitted the existing data-quality warning for two reported
daily TN > TX pairs; those observations remain retained and documented. Some
smoke plots emitted a pandas/Matplotlib scalar-conversion FutureWarning, without
preventing figure generation. No inference failure was observed in these checks.

For scientific conclusions, follow START_HERE: inspect convergence and Monte
Carlo precision, compare prior assumptions and historical forecasts, and assess
seasonal calibration, residual dependence, ordering and risk. Software test
success does not establish acceleration or predictive adequacy on the record.
