# Monthly research

The model declarations in `models.py` use private level, slope and seasonal trajectories for all six summaries. `run.py` performs independent or joint fits; `report.py` writes scientific contrasts, diagnostics and forecasts. The reference configuration is [`config/main.json`](config/main.json).

| Task | Command | Config |
|---|---|---|
| Data figures | `python -m research.monthly.explore` | `exploration.json` |
| Model and resource plan | `python -m research.monthly.preflight --config research/monthly/config/main.json` | `main.json` |
| Short execution check | `python -m research.monthly.copula --config research/monthly/config/smoke.json` | `smoke.json` |
| Reference fit | `python -m research.monthly.copula --config research/monthly/config/main.json` | `main.json` |
| Constant dispersion and fixed seasonal checks | `python -m research.monthly.prior_assessment --config research/monthly/config/adequacy.json --stage plan` | `adequacy.json` |
| Innovation, shape and prior sensitivity | `python -m research.monthly.prior_assessment --config research/monthly/config/sensitivity.json --stage plan` | `sensitivity.json` |
| Forecast origins | `python -m research.monthly.validate --config research/monthly/config/predictive.json` | `predictive.json` |
| Reviewer comment 5: central 90/95/99% and directional tail validation | `python -m research.monthly.validate --config research/monthly/config/comment5.json` | `comment5.json` |

Use `--stage all` for assessment fits after reviewing the plan. `independent.json` provides six separate marginal fits; `independence.json` keeps shared shrinkage with an identity copula. Configurations in `config/` also include anchor, dependence, figures, final and recovery checks. The separate [seasonal workflow](../seasonal/README.md) compares both resolutions using identical daily windows and seasonal forecast outcomes. See [START_HERE](../../START_HERE.md) for the run order.
