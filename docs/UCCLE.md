# Uccle data and analysis

The data layer supplies `load_uccle_series`, `load_uccle_multiseries`, integrity
checks, and thin convenience constructors. Every fitting helper ultimately
constructs public models and delegates to `bx.fit()`.

| Summary | Observation model | Interpretation |
|---|---|---|
| TXm, TNm | Gaussian | Monthly mean daily maximum/minimum temperature |
| TXx, TNx | GEV upper tail | Monthly maximum daily maximum/minimum |
| TXn, TNn | GEV lower tail | Monthly minimum daily maximum/minimum |

The bundled summaries are monthly, with period 12, covering January 1892–August 2026 (1,616 months). Earlier
manuscript analyses may have used seasonal blocks; that is a different dataset
construction and must be described explicitly. Priors on innovations depend on
the observation interval.

## Read and prepare observations

```python
import bucex as bx

# Load all six updated monthly series, or just one.
data = bx.load_uccle_multiseries()
txm = bx.load_uccle_series("TXm", start="1980-01-01")

# Compute and save summaries from a daily CSV.
monthly = bx.derive_uccle_monthly(
    "data/Uccle_31_08_26.csv", output_dir="bucex/data",
)

# Compare every bundled month with its daily source.
checks = bx.validate_uccle_data(daily_source="data/Uccle_31_08_26.csv")
print(checks["daily_max_abs_difference"])
```

`derive_uccle_monthly` is the existing aggregator, extended to accept an
explicit CSV path or a DataFrame containing `DAY`, `TX` and `TN`. `RR` is
optional and does not enter the temperature summaries. If needed,
`bx.load_uccle_daily(path)` reads and checks the daily CSV separately.

| Monthly column | Calculation within each calendar month |
|---|---|
| TXm | mean of TX |
| TNm | mean of TN |
| TXx | maximum of TX |
| TXn | minimum of TX |
| TNx | maximum of TN |
| TNn | minimum of TN |

The return value is a DataFrame indexed by month starts. Without `output_dir`,
no files are written. With it, six `date,series` CSVs and `quality_report.json`
are written to that directory, replacing existing files of those names.
The same report is available as `monthly.attrs`.

By default, every complete month in the source is used. An optional
`end="YYYY-MM-DD"` limits the included daily observations; it is an inclusive
daily cutoff, so use `2022-12-31` to reproduce the old period. A boundary month
starting late or ending early is excluded with a warning. Every retained
month must contain one finite TX and TN for every calendar day, including
29 February in leap years. Duplicates, malformed dates and internal gaps
raise errors. No imputation, sign reversal or source harmonization is applied.

The two reported daily TN > TX pairs are retained and flagged. All monthly
summaries satisfy the package's seven ordering comparisons. Source details
and measurement-comparability limits are recorded in
[the source notes](../bucex/data/SOURCES.md).

For a separate dataset, choose a directory and pass it explicitly when loading:

```python
monthly = bx.derive_uccle_monthly("my_daily.csv", output_dir="data/my_monthly")
data = bx.load_uccle_multiseries(data_dir="data/my_monthly")
checks = bx.validate_uccle_data(
    data_dir="data/my_monthly", daily_source="my_daily.csv",
)
```

A validation report gives per-series coverage, range and (when a daily source
is supplied) the maximum absolute difference. Differences are reported, not
silently corrected. Missing comparison months raise an error. Matching values
normally differ by less than `1e-12` after a CSV round trip.

## Configured workflow

`python -m research.serra.prepare_uccle` is a short wrapper around the same
aggregator. Its `research/serra/config/prepare_uccle.json` contains:

```json
{
  "daily_file": "data/Uccle_31_08_26.csv",
  "output_dir": "bucex/data",
  "end": null
}
```

Paths are relative to the working directory; run from the source root.
Change `daily_file` when a new daily CSV is available, then rerun preparation.
Primary SERRA full configurations end in December 2022, matching the submitted
record. `config/extension_full.json` explores the later mixed-source record
separately; no homogenization is asserted. The loader's default remains the
complete bundled record. Consult `bucex/data/SOURCES.md` for source windows and
comparability before interpreting the extension.

```bash
python -m research.serra.univariate --series TXm TXx
python -m research.serra.copula --independence
python -m research.serra.copula
```

Use `--config research/serra/config/independent_full.json` or `copula_full.json`
for substantive fits. These are MCMC starting budgets and require diagnostics.
Every channel has private location seasonality and an optional seasonal scale.
The complete sequence and reviewer studies are in `research/serra/README.md`.
