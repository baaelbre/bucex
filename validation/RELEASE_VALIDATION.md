# BUCEX 1.7.2: executed release checks

Validated on 20 September 2026. Environment versions are recorded in
`environment.json`. Source and wheel tests use the released code.

- The full initial regression run executed 311 cases: 309 passed and two
  release-contract assertions still expected version 1.7.1. Both assertions
  were updated to 1.7.2 and passed in the targeted rerun. A new atomic-output
  regression case also passed. Final status: **312 distinct regression cases
  passed**, reconciled in `pytest-1.7.2-final.xml`; the initial and targeted XML
  files are retained. No inferential test was removed or weakened.
- A separately installed wheel matched every current package Python source
  file. **20 installed-package checks passed**, covering historical archives,
  plotting/warm starts and the publication-reporting API.
- Six independent monthly-scale smoke fits and one mixed six-channel copula
  monthly-scale smoke fit completed: 36 months, one chain, four warmup and four
  retained draws. Saved archives, parameter/target/scale trace tables and
  calendar diagnostics were produced. These runs are execution checks only.
- Saved-fit re-reporting at an explicitly requested 80% level, all eleven
  strict figure recipes, forecast checks using the archive's real channel,
  80% endpoint intervals, and one-fold held-out monthly diagnostics completed.
  Full workflow settings and commands are recorded in `workflows-1.7.2.json`.
- Nineteen inference configurations in `revision/` resolved successfully.
  Primary configurations contain 1,616 months through August 2026. The generic
  data preflight does not describe the synthetic length of recovery experiments.
- The actual six independent result archives produced ten numerical figure
  panels and one explicit missing-trace placeholder, in both PNG and PDF.
  All eleven PDFs were checked for complete structure, rendered and visually
  inspected. PNG/PDF figures retain the saved interval level and source hashes.
- Wheel and source-distribution builds succeeded. Package data still extend
  through August 2026. Comparing with 1.7.1 confirms no changes to inference,
  prior, model, observation or bundled-data source files.

The source-suite warning records the documented two daily TN > TX observations
in the mixed-source data audit. This release does not silently change those
observations or establish source homogenisation.

These checks do not establish scientific convergence, coverage, predictive
skill or publication acceptance. Full monthly-scale/copula fits, substantive
prior sensitivity, recovery and held-out experiments remain research tasks.
