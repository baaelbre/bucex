# Seasonal Uccle analysis — 1.9.1

The seasonal record has 538 complete blocks, MAM 1892–JJA 2026. The current
workflow uses conditional residual independence and separate Normal initial
rates. Level, rate and seasonal innovations retain shared scale hyperpriors.

Use the root [command guide](../../BUCEX-1.9.1-commands.md): posterior sensitivity
on HPC, expanding-window validation on biobot, with distinct screen/paper tiers.
The 23 full-record tasks forecast 120 seasons. Default validation uses 52 fits
and five-year windows; optional `validation10` reproduces 60/80/90% training
fractions with ten-year windows and is collected separately.

`config/main.json` declares the reference; `experiments.json` specifies studies,
validation origins, budgets and resource requests. `jobs --verify` compiles all
82 tasks (including four prospective pre-2019 fits) without sampling and checks
identity dependence and absence of a shared initial-rate hyperparameter.

`config/reference_189.json` and `reference_20260923.json` are frozen historical
references. The old `constant_copula*.json` and `dependence_sensitivity.json`
names are compatibility aliases for the identity reference in this release.
Use 1.9.0 for reproducing its old dependence study.
