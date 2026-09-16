# Start with BUCEX 1.7.0

Extract the archive, open a terminal in `bucex-1.7.0`, and install:

```bash
python -m pip install -e ".[test]"
```

1. Inspect the primary configuration without fitting anything:

```bash
python -m research.serra.preflight --config research/serra/config/copula_full.json
```

It must show January 1892–August 2026 (1616 months), six channels,
constant scale, LKJ(1), continuous lasso shrinkage, ASIS off, and four chains.
It prints an array-storage estimate; actual fitting/reporting needs more RAM.

2. Check execution with a short fit:

```bash
python -m research.serra.univariate --series TXx
python -m research.serra.copula
```

These default to **smoke configurations**, not scientific runs: 36 months,
four warm-up and four retained draws, one chain. Output goes into fresh,
timestamped directories under `results/serra`.

3. Fit one full margin and inspect it before running the others:

```bash
python -m research.serra.univariate --config research/serra/config/independent_full.json --series TXx
```

Read `convergence.json`, `mcmc.csv`, `scientific_targets.csv`, traces,
`sampler_metrics.csv`, PIT/QQ plots and the slope/scale figures. A completed
run is not a convergence certificate. Four chains × 2000 warm-up + 2000
retained is a starting budget; extend/reconfigure according to actual mixing.

4. Run all six margins, then the matched joint baselines:

```bash
python -m research.serra.univariate --config research/serra/config/independent_full.json
python -m research.serra.copula --config research/serra/config/copula_full.json --independence
python -m research.serra.copula --config research/serra/config/copula_full.json
```

For separate jobs, add `--series TXm` (or any other summary) to the univariate
command. The full joint state array alone is approximately **8.07 GB**.
Reporting makes additional arrays: do not assume an 8 GB machine is enough.

The reference period contrast is January 1892–December 1921 versus September 1996–August 2026. Cross-summary
contrasts use paired joint draws. If the copula fit is impractical, keep the
six univariate analyses and omit empirical joint-dependence conclusions.

5. Follow [research/serra/README.md](research/serra/README.md) for the five core
prior comparisons, targeted shape/initial-prior checks, fixed/evolving location
seasonality and constant/monthly scale checks, held-out predictions and the
reviewer simulations. `--variants` selects named sensitivity cases, so you can
submit one variant at a time without editing the scripts.

For the historical comparison use `independent_1892_2022.json` or
`copula_1892_2022.json`; their recent period is 1993–2022. The `*_extended.json`
files are compatibility aliases for the full record. Re-exporting an old fit never assimilates new observations.
