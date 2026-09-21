# BUCEX 1.7.4 release checks — 21 September 2026

## Executed checks

- Full source test suite: **346 passed**. The existing data-loader warning
  records two retained reported daily TN > TX pairs. This release does not
  change the observations. See `pytest-1.7.4-source.xml`.
- Installed-wheel checks outside the source package: **55 passed**, covering
  the exploration API, source-data checks, report writing, bundled monthly
  data and historical archives. The test directory contains the source test
  fixtures but imports BUCEX from the separately installed wheel. See
  `pytest-1.7.4-installed.xml`.
- The 13 new exploration checks cover known analytical means, quartiles,
  annual trends and residual IQRs; original observation signs; incomplete
  data; invalid dates and sparse windows; report exports; and delegation
  from the short SERRA script without invoking posterior fitting.
- `python -m research.serra.explore` completed with the default revision
  configuration and all six series through August 2026. It produced both
  PNG and PDF figures, their numerical tables, source observations, resolved
  configuration and metadata. Both PNGs were visually inspected.
- The numerical tables reproduce the original manuscript's standalone
  exploration exports: 144 cycle rows agree to a maximum absolute difference
  of 7.11e-15, and 216 spread rows to 4.45e-16. See
  `exploration-reproduction.json`.
- A separate process blocked Matplotlib imports, loaded the installed wheel,
  calculated exploration summaries and exported them with `figures=False`.
  Statistical and table-only use does not require the plotting extra.
- All existing Python files in inference, models, components, priors,
  observation, core, API, IO and datasets, and all bundled/source data files,
  are byte-identical to 1.7.3. See `compatibility-1.7.4.json`.
- Wheel and source distributions were built. The release ZIP excludes build
  trees, caches, installed test copies and generated scientific results.

See `workflows-1.7.4.json` for the environment and executed workflow details.
The installed test harness initially omitted the daily CSV fixture required
by two source-data tests. Supplying that fixture resolved both failures;
no package change was required.

## Limits

No final scientific model runs or new prior sensitivity experiments were
conducted for this patch release. These checks establish code behavior and
reproduction of descriptive figures, not scientific convergence, tail
adequacy or robustness of climatic conclusions. The figures summarize
observations; their empirical bands are not posterior intervals.

Tests used Linux and Python 3.12. Other operating systems and Python versions
were not separately exercised for this patch release. Exploration windows
are explicit configuration choices; extending the data does not silently
change the manuscript comparison periods.
