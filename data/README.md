# Daily Uccle source data

`Uccle_31_08_26.csv` contains 49,186 unique consecutive daily rows from
1892-01-01 through 2026-08-31, with columns `DAY`, `TX`, `TN`, and `RR`.
The original observations through 2023-10-17 are preserved. The temperature
extension uses the supplied Meteostat export for 2024, with its missing day
filled from Meteociel; the remaining added dates use Meteociel. New RR values
are unavailable and remain NA.

The six derived monthly CSVs live once in `bucex/data`. They contain 1,616
months through August 2026. Regenerate them from the source root:

```bash
python -m research.serra.prepare_uccle
bucex-uccle validate-data --daily-source data/Uccle_31_08_26.csv
```


The preparation settings are `research/serra/config/prepare_uccle.json`.
See [the data API](../docs/UCCLE.md) for direct Python use and custom outputs.
`Dagelijksetemperaturensinds1833.xlsx` is retained as a supplied supporting
file; it is not an input to this preparation step.

[Source notes](../bucex/data/SOURCES.md) preserve the extraction details,
reporting-window differences, two reported TN > TX pairs, and the limitations
of combining the historical and newer sources. No homogenization is applied.
The original historical file's exact source record and observation license
were not supplied; the software MIT license does not license the observations.
