# From research snapshots to publication 1.0.0

The 1.9.x filenames were internal research snapshots. 1.0.0 intentionally
starts a smaller public API; it is not a backward-compatible numeric upgrade.
The source baseline used for this release was 1.9.8.4.

| Research interface/concept | Publication interface |
|---|---|
| `Model(observation, components)` | retained; add `priors=bx.Priors(...)` |
| wide wildcard imports | explicit supported names in `bucex.__all__` |
| `Channel(name, observation, components, ...)` | `Channel(name, Model(...))` |
| `MarginalPriors` / `SharedShrinkage` | channel `Priors` plus `MultiSeriesModel(pooling=Pooling(...))` |
| `fs_priors`, `alpha0`, `s_trend`, etc. | `Priors(initial_level=..., slope=..., ...)` |
| `GEV` plus separate tail transform | `GEV(tail="lower")`, original-scale result paths |
| `chain_workers` | `MCMC(workers=...)` |
| engine/parameterization combinations | Gaussian FFBS or GEV Laplace–MH, selected by margin |
| development output bundles | one versioned `.bucex` fit archive + research CSV/JSON/figures |
| BIOBOT/HPC launcher versions | portable research modules and independent-chain IDs |

Retained numerical foundations include the non-centred structural state
system, singular-support-aware FFBS, deterministic Laplace–MH path proposal,
preconditioned coefficient slice and half-normal shared-scale GIG update.
The Gaussian variance update is now conjugate. Coefficient prior whitening
uses the original time origin, so random streams and draws are not intended
to reproduce old files bit-for-bit; the specified posterior target is preserved.
The release tests compare the relevant analytic cases and numerical invariants.

Removed from the supported package: residual copulas, alternative shrinkage
families, SSVS/model selection, particle/disturbance alternatives, dynamic phi,
regression extensions unused by this paper, interweaving switches, warm-start
adapters and version-specific scheduler/research scaffolding. No old archive
is overwritten, and no silent migration guesses its statistical specification.
Read an old archive with the version that created it, or refit the explicit
1.0.0 JSON configuration.

A new fit is required before replacing final paper numbers or reporting run
time/diagnostics for this implementation. Release smoke outputs verify code
execution and plots, not scientific convergence or a revised paper result.
