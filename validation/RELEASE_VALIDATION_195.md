# BUCEX 1.9.5 validation

All **340 tests passed across the full regression run and the affected-test
rerun**. The full run passed 339 tests; the remaining historical test selected
an extra monthly fit after that supplement was added to the plan. Restricting
its collection check to seasonal fits restored the intended four-case test,
which passed on rerun. No statistical tolerance or numerical gate was relaxed.

The new tests independently check the transformed half-normal/half-t/Cauchy
conditional densities, including their Jacobian, against SciPy densities;
compare half-normal scale sampling with quadrature; check marginal moments,
prior-family calibration and shared-scale sampling; and solve annual upper and
lower return levels back through their aggregate CDFs.

End-to-end checks cover all three half-family samplers, saved-fit reloads,
95% risk reports, pre-2019 predictions for all six summaries, 90/95/99% forecast
coverage and monthly/seasonal forecasts evaluated on identical observed targets.
Real parallel two-chain startup fits passed for all three prior families. Queue
tests exercise actual subprocess concurrency and process-failure recording.

Both tiers' configurations pass: 106 active pooled fits plus 12 optional
unpooled fits per tier. A combined dry run produces 212 active fits. Python
3.10 syntax is checked for 164 modules; shell scripts pass `bash -n`. Execution
here used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and pandas 2.2.3. The actual
BIOBOT interpreter is checked by the launcher and its startup probe.

A full 538-season smoke run exercises the report and 30-year forecast outputs,
annual risk/return-level aggregation and figure construction. The pre-2019
figure is constructed from a separate May-2019-cutoff smoke fit. These tiny
chains verify code paths only and are not evidence for scientific conclusions.

See `../../RELEASE_VALIDATION.json` and the logs in `release_195/`. Historical
validation records describe their own releases. No current scientific fit has
been run to convergence, and no full-study BIOBOT wall time has been measured.
