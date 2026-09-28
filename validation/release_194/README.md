# BUCEX 1.9.4 release checks

The root `RELEASE_VALIDATION.json` records the final source/data/configuration fingerprint and the scope of checks.

The initial full regression ran 321 tests: 317 passed and four failed because their expectations still described the earlier release. The four affected tests were updated for the new comparison batch and explicit absence of a copula, then all four passed. Five new 1.9.4 tests passed, including independent calculations of the conditional hyperparameter density and integrated mixture prior. Two launcher tests were rerun successfully after the final environment-setup guard. The XML files retain all these runs.

The actual startup probe passed all 13 baseline fits and archive reloads. It used two concurrent chains per fit but only one response fit at a time, matching the two CPUs exposed locally. It is a startup check, not a convergence assessment. No production screen/paper fits or HPC submissions were performed here.

The screen and paper plan folders contain resolved settings, 72 matched-prior checks per tier, and all declared experiment and validation plans. Dry-run output shows the scheduler and local-runner command paths.

## Reduced screen budget

The screen budget was subsequently reduced to 1,000 warm-up + 2,000 retained draws per chain at the user's request. All 271 resolved screen and paper budgets were checked; paper settings are unchanged. Screen preflight and the 72 prior matches pass. Earlier test and dry-run logs retain their original budgets; the current screen plan tables and root validation JSON document the revised configuration. No inference code was changed.

## Python 3.11 compatibility correction

A user compute-node traceback identified quote reuse in the bundle status f-string. The original release tests ran on Python 3.12 and did not catch this; `ast.parse(feature_version=(3,11))` on Python 3.12 also accepts the offending expression. The legacy tokenizer/grammar rejects the original expression, and all 207 current source/test files pass after correction. See `python_compatibility_fix.json`. Actual Python 3.11 execution remains checked on Gallade. The probe now compiles production modules with its actual interpreter before imports/fits.
