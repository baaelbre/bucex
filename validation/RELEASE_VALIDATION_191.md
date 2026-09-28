# BUCEX 1.9.1 release verification

These are software checks, not production estimates, validated forecasts or
actual HPC submissions.

## HPC startup correction, 2026-09-28

Array 47058490 failed with exit status 2 and `Cannot resolve BUCEX_PYTHON`.
The submitted Python path used `/user/data/...`, while the working directory
used `/kyukon/data/...`. The launcher had normalized the working/results paths
but not the Python executable's parent directory. The corrected runtime
normalizes that directory before export and preserves the final virtual-env
symlink. It also checks Python >= 3.10 and reports the interpreter and host on
failure. Scientific configuration and model/sampler Python code are unchanged.

Two focused tests passed. They exercise a real virtual environment through a directory alias
containing spaces, verify that Python still detects the virtual environment,
and check that a missing interpreter reports the requested path and host.
Native Slurm screen and paper dry runs also passed with a simulated directory
alias: both export a physical interpreter path and use throttle 16. Bash syntax
validation passed. Logs are in `release_191/hpc_runtime_tests.txt` and
`release_191/hpc_path_{screen,paper}_dryrun.txt`.
The HPC command guide includes a compute-node import check before resubmission.
Remote compute-node execution remains unverified here.

## Completed checks

- Compile all 82 tasks under both tiers: 23 posterior fits, four prospective
  pre-2019 fits, 52 five-year validation fits and three optional ten-year fits.
  Verify the fixed identity matrix, separate initial rates, 538 full-record
  seasons, 120-season forecasts and complete declared held-out windows.
- Check that all initial-rate SDs are 0.5 C/decade except the intended 0.25 / 1.0
  sensitivities. Innovation-width controls leave this fixed prior unchanged.
  Matched-moment and fixed-anchor controls satisfy their distinct definitions.
- Run the full 294-test suite: 292 passed, with two failures identifying a
  stale version assertion and historical configuration inheritance. Correct
  those issues, freeze the historical configuration as a self-contained
  declaration, and rerun the affected module/configuration/reporting tests:
  **14 passed**. All 294 test cases have passed across these runs. Logs of the
  initial run and the targeted recheck are retained transparently.
- Exercise a real 22-block, six-response mixed Gaussian/GEV fit with four
  parallel chains, three warm-up and five retained draws per chain. Check the
  exact target, archive save/load, absence of shared initial-rate parameters,
  three shared innovation scales, 120-season forecasts and marginal reports.
- Exercise a separate held-out forecast through the actual validation runner.
  Check 90/95/99% interval outputs, forecast-origin level/rate exports, saved
  fits and omission of compound-risk reports in the active workflow.
- Check the manuscript plotting path for three shared scales and marginal
  change panels without cross-summary contrasts. The smoke fit is explicitly
  unconverged and used only to check execution.
- Check calendar labels for identity-model seasonal residual checks and the
  Gaussian distribution of the initial-rate contribution in prior simulation.
- Inspect native Slurm screen/paper dry runs (23 standard; 22 standard plus
  one long reference), PBS screen dry run and biobot validation dry run.
  Check Bash syntax. No scheduler or remote workstation jobs were launched.
- Build the source and wheel distributions. Generic copula capabilities and
  historical archives remain supported; this release changes the active
  seasonal workflow to conditional independence.

The one existing data warning concerns two retained reported daily TN > TX
pairs, documented by the data provenance checks. No observations were changed.

`release_191/source_manifest.json` records the checked source. The production
screen/paper runs and scientific assessment still need to be performed.
