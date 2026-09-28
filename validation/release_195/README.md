# Release checks, not scientific results

The `screen/plan` and `paper/plan` tables are configuration checks for 1.9.5.
`new-release-tests.log` covers the new hierarchy, probability calculations and
workflow tests. `regression.log` is the full regression run;
`collection-rerun.log` records the corrected seasonal collection test.
`dry-run.log` lists the entire combined queue using this host's available CPU
budget. The probe and benchmark, where included, use tiny disposable chains.
They must not be interpreted as fitted-paper results.
