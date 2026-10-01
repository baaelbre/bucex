# Uccle source data

`Uccle_31_08_26.csv` contains 49,186 unique consecutive daily rows from
1892-01-01 through 2026-08-31, with DAY, TX, TN and RR columns. The older
observations through 2023-10-17 are preserved. The extension uses the supplied
Meteostat export for 2024 (one missing day filled from Meteociel), and
Meteociel for the other added dates. New RR values remain unavailable.

`research.data.load_summaries()` derives means, maxima and minima for complete
seasonal or monthly blocks. No temperature imputation or new homogenization
is applied. There are 538 seasonal blocks or 1,616 months. Derived CSVs are
saved with each analysis rather than maintained as duplicate package datasets.

[SOURCES.md](SOURCES.md) preserves the extraction details, differing observation
windows, reported TN > TX pairs, and source limitations. The original
historical file's exact source record and observation license were not
supplied. The software MIT license does not license the observations.
