# Start with BUCEX 1.7.1

Extract the archive, open a terminal in `bucex-1.7.1`, and install:

```bash
python -m pip install -e ".[test]"
```

Confirm the installed version and source location:

```bash
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Explicit settings in an older config override the new defaults. Old fits remain
readable, but re-exporting cannot repair poorly mixed draws. Start fresh chains.

1. Inspect the primary configuration without fitting anything:

```bash
python -m research.serra.preflight --config research/serra/config/copula_full.json
```

It must show January 1892–August 2026 (1616 months), six channels,
constant scale, LKJ(1), continuous normal shrinkage (SD prior medians .01, .00005, .02), ASIS off, and four chains.
It prints an array-storage estimate; actual fitting/reporting needs more RAM.

2. Check execution with a short fit:

```bash
python -m research.serra.univariate --series TXn
python -m research.serra.copula
```

These default to **smoke configurations**, not scientific runs: 36 months,
four warm-up and four retained draws, one chain. Output goes into fresh,
timestamped directories under `results/serra`.

3. Fit one full margin and inspect it before running the others:

```bash
python -m research.serra.univariate --config research/serra/config/independent_full.json --series TXn
```

Read `convergence.json`, `mcmc.csv`, `scientific_targets.csv`, traces,
`sampler_metrics.csv`, PIT/QQ plots and the slope/scale figures. A completed
run is not a convergence certificate. Four chains × 2000 warm-up + 2000
retained is a starting budget; extend/reconfigure according to actual mixing.

For a sampler-only comparison, keep the previous .02 level prior:

```bash
python -m research.serra.univariate --config research/serra/config/independent_level_002.json --series TXn
```

Inspect physical innovation SDs, not just signed FS coefficients: random sign
switches can conceal poor mixing of their magnitudes. New reference-convergence,
fallback, iteration and slice-cost metrics appear in the usual diagnostic files.

Check .005/.01/.02 level-prior medians with all other priors fixed:

```bash
python -m research.serra.sensitivity --config research/serra/config/sensitivity/level.json --series TXn
```

The variants are `normal_reference`, `level_half`, and `level_double`.
Use `--variants NAME` to submit one case. Compare period warming, slopes and
risks rather than smoothness alone. Other summaries are selected with `--series`.

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

The fixed-versus-monthly scale comparison is already configured:

```bash
python -m research.serra.sensitivity --config research/serra/config/sensitivity/structure.json --series TXm TNm --variants reference monthly_scale
python -m research.serra.univariate --config research/serra/config/independent_full.json --series TXn --scale seasonal
```

Monthly scale effects repeat across years; they do not add another evolving
scale process. Check calendar-specific residual spread and held-out calibration.
Read [docs/REVISION_GUIDE.md](docs/REVISION_GUIDE.md) for the sampling fix,
interpretation of tightening priors, reviewer requirements and paper story.
