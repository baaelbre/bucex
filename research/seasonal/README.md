# Seasonal paper workflow — 1.9.5.1

See [the complete launch guide](../../BUCEX-1.9.5.1-commands.md). The active
configuration is `config/main.json`: three pooled half-normal innovation scales,
separate fixed normal initial rates, no copula. Use `jobs --verify` to check all
configurations and `overnight --dry-run` to inspect the resource-bounded queue.

`all` includes 106 fits per tier: 41 seasonal full-record settings, six pre-2019
fits, 43 five-year validation fits, nine ten-year split validations, six matched
monthly/seasonal validations and one full-record monthly supplement. The
`deferred` batch holds the 12 optional unpooled comparisons and is excluded.

`finish` collects diagnostics and paired validation, creates prior simulations,
builds reference figures and writes a dated review ZIP without large fit
archives. Each task retains its resolved configuration and provenance. Screen
results remain distinct from paper results. A completed job may still be
flagged for convergence or model adequacy.
