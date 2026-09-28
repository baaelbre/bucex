# BUCEX 1.9.4 release checks

The root `RELEASE_VALIDATION.json` records the final source/data/configuration fingerprint and the scope of checks.

The initial full regression ran 321 tests: 317 passed and four failed because their expectations still described the earlier release. The four affected tests were updated for the new comparison batch and explicit absence of a copula, then all four passed. Five new 1.9.4 tests passed, including independent calculations of the conditional hyperparameter density and integrated mixture prior. Two launcher tests were rerun successfully after the final environment-setup guard. The XML files retain all these runs.

The actual startup probe passed all 13 baseline fits and archive reloads. It used two concurrent chains per fit but only one response fit at a time, matching the two CPUs exposed locally. It is a startup check, not a convergence assessment. No production screen/paper fits or HPC submissions were performed here.

The screen and paper plan folders contain resolved settings, 72 matched-prior checks per tier, and all declared experiment and validation plans. Dry-run output shows the scheduler and local-runner command paths.
