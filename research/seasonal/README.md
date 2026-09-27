# Seasonal Uccle analysis — 1.9.0

Complete daily-derived DJF/MAM/JJA/SON blocks give 538 six-response observations,
MAM 1892–JJA 2026. The new reference and its 24 comparisons are declared in
`config/experiments.json` and the four sensitivity files it lists. Run them
with the screen/paper launchers described in [FINAL_RUN.md](../../FINAL_RUN.md).
The default batch excludes validation and the prospective pre-2019 analysis.

The reference uses log(3) hyperprior width with second-moment-matched
median-absolute anchors (level, slope, seasonal, initial slope)
(0.004836005867750226, 0.00004836005867750226, 0.004836005867750226,
0.004836005867750226). The initial seasonal coefficient SD remains 20°C.
`config/reference_189.json` freezes the resolved old settings;
`config/reference_20260923.json` retains the earlier archived specification.

```bash
python -m research.seasonal.jobs --verify --tier screen
python -m research.seasonal.jobs --list --tier paper
bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
```

A direct reference fit remains available (run on a compute node):

```bash
python -m research.seasonal.fit --config research/seasonal/config/final.json
```

It uses four chains with 3,000 warm-up / 8,000 retained iterations per chain,
then forecasts 120 seasons with 12,000 predictive paths. The parallel paper
reference uses those same budgets. Both tiers save fits, use 95% bands and
export 10-/30-year innovation effects. The smoke config is a computational
check only, with four chains and very few iterations; it is not empirical
posterior evidence. Legacy grid/monthly-resolution comparisons remain explicit
utilities and are not part of the new default posterior batch.
