# BUCEX 1.7.3 release checks — 21 September 2026

## Executed checks

- Full source test suite: **333 passed**. One existing warning records two
  retained reported daily TN > TX pairs in a data-loader test; data were not
  changed by this release. See `pytest-1.7.3-source.xml`.
- Installed-wheel checks outside the source package: **23 passed**, including
  spawned workers, diagnostics, archives, prior comparisons and legacy archive
  loading. See `pytest-1.7.3-installed.xml`.
- Serial/parallel tests cover scalar Gaussian and GEV FS, monthly scales,
  mixed copula, hierarchy, shared states and disturbance inference. Seeded
  state/parameter/auxiliary draws and diagnostic arrays agree exactly;
  downstream forecasts retain posterior pairing. Tests cover four chains with
  two versus four workers, worker capping, invalid counts, worker errors and
  chain-specific shared warm starts.
- The complete installed-package assessment workflow ran for **TNm and TXn**,
  three priors, two historical forecast origins, and two process workers.
  Each chain had two warmup and four retained draws. Six sensitivity fits and
  twelve historical refits completed, generating compact reports and PNGs.
  Date/case matching and failed numerical screens remain visible. These short
  results are execution checks, not evidence for any prior or climatic claim.
- Representative comparison PNGs were visually inspected for interval/legend
  readability and the manuscript style.
- Wheel and source distributions were built successfully. The bundled primary
  monthly record still contains 1,616 months ending August 2026.

See `workflows-1.7.3.json` for environment versions, resolved smoke settings,
forecast cutoffs and counts. Final release archives omit generated fit results,
bytecode, build directories, installed copies and temporary logs.

## Limits

No full-length pilot prior comparison, simulation study, or final model run
was conducted for this patch release. The tests establish implementation and
execution contracts; they do not establish scientific convergence, correct
latent decomposition, tail adequacy or robustness of an acceleration claim.
Testing used Linux with spawned processes. Windows/macOS process behavior is
supported by the same spawn-based implementation but was not separately run.
Changing worker count preserves seeded draws in the tested numerical runtime;
bitwise reproducibility across library versions/platforms is not promised.
