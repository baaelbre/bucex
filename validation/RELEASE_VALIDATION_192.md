# BUCEX 1.9.2 verification

The union of the latest outcomes is **307 tests passed**. Detailed JUnit files and both launcher dry runs are included under `release_192/`; `../RELEASE_VALIDATION.json` records the runtime, counts, source fingerprint and scope. The broad regression run initially found a stale test assertion expecting version 1.9.1. After updating that assertion to 1.9.2, its targeted rerun passed. The total describes the combined suite outcomes, not a single uninterrupted run.

New scientific checks compare the one-coefficient hyperconditional against independently expressed Normal densities, verify that removing pooling preserves the earlier marginal prior, and reject multiple responses inside a private hierarchy. All 524 task configurations compile with one response, no copula and separate initial slopes. Width controls retain their declared second-moment matching.

Real short fits exercise Gaussian means, GEV maxima and minima, constant observation dispersion and fixed location seasonality. Tests cover saved-fit decoding, warm starts, 30-year forecasts, prior/posterior reports, original-scale lower-tail risk scores, 90/95/99% coverage and absence of copula/compound outputs. Serial and parallel-chain draws match exactly at fixed seeds.

The process-loop test launches real short subprocesses, verifies the maximum concurrency, forces one failure and checks that remaining jobs finish. Integration tests execute real reduced-budget tasks, resume completed work, collect per-response comparisons with figures, pair held-out cases and export compact evidence without posterior-state archives. Screen/paper ALL wrappers pass dry runs for their full 524-task grids.

These checks verify software behavior and configuration. They do not demonstrate MCMC convergence at production budgets or better prediction. No production fit, biobot execution or HPC submission was performed in this workspace. Keep numerical gates and scientific validation active when running the supplied jobs.
