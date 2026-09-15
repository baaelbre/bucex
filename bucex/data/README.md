# Bundled Uccle monthly summaries

The six CSVs are the canonical monthly observations: **January 1892–August
2026**, 1,616 consecutive month-start dates per series. Load them with
`bx.load_uccle_multiseries()` or `bx.load_uccle_series("TXm")`.

| Series | Within-month aggregation of daily temperatures |
|---|---|
| TXm | mean(TX) |
| TNm | mean(TN) |
| TXx | max(TX) |
| TXn | min(TX) |
| TNx | max(TN) |
| TNn | min(TN) |

All values are in degrees Celsius, with original signs. These are monthly
blocks, so the annual seasonal period is 12. `quality_report.json` records
coverage, the daily source hash and the two reported daily TN > TX pairs,
which are preserved in the aggregation. All 1,616 monthly blocks satisfy the
seven `UCCLE_ORDER_CONSTRAINTS` comparisons.

See [SOURCES.md](SOURCES.md) for the historical/Meteostat/Meteociel inputs,
reporting windows, gap filling and source-boundary limitations. The package
software license does not assign a license to the observations.

The daily CSV is included in the source release's `data/`, not in the wheel.
From that source root, `python -m research.serra.prepare_uccle` regenerates
these files through the public `bx.derive_uccle_monthly` function.
