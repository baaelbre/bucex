# BUCEX 1.9.2

[Commands for biobot](BUCEX-1.9.2-commands.md) · [Verification](validation/RELEASE_VALIDATION_192.md)

The seasonal paper now uses six separate analyses. Every task has one response, private innovation hyperparameters, a separate initial slope, no copula and no cross-response borrowing. The reference preserves the earlier marginal normal–lognormal innovation priors; quarter-seasonal and doubled-slope/stronger-seasonal controls are added.

The local launchers provide bounded fit concurrency, per-task logs and provenance-checked resume. The full screen or paper grid has 524 single-response tasks. Shape-prior checks are omitted for Gaussian means. Full-record forecasts remain 30 years, with 95% reports and 90/95/99% held-out coverage.

Use a new 1.9.2 results tree. This release contains tested code and job settings; production posterior and validation results have not been generated here. The general library retains joint/copula capabilities for historical work, but the current job grid cannot invoke them.
