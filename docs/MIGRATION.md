# Migrating to 1.6.1

The public univariate `Model`, `fit`, priors, component, risk and forecast API
continues to work. Particle inference was retired in 1.6.0 and no particle
kernel or conference script is distributed in 1.6.1.

All research entry points now live under `research/serra/`:

| Earlier command | Current command |
|---|---|
| `python -m examples.gaussian` | `python -m research.serra.tutorials --kind gaussian` |
| `python -m examples.gev` | `python -m research.serra.tutorials --kind gev` |
| `python -m examples.hierarchical` | `python -m research.serra.tutorials --kind hierarchical` |
| `python -m examples.shared_components` | `python -m research.serra.tutorials --kind shared` |
| Numbered/conference Uccle scripts | `python -m research.serra.run --config ... --series TXx` |

Research helper imports move from `research.report` to `research.serra.report`.
These are research utilities, not a supported package API. Historical workflow
copies are removed from this distribution; retain earlier release ZIPs for
provenance. Historical saved result files remain readable where supported.

`run` and `validate` share explicit model/prior construction. A `joint`
configuration uses private trends with an identity Gaussian copula;
`--copula` estimates residual correlation while retaining all marginal
assumptions. A `shared` configuration can add the same dependence extension.
Hard ordering is diagnosed rather than enforced by altered samples.

Dynamic GEV log scale (`--phi linear`, `rw`, `ssvs`) remains available in the
independent route. There is no requirement to fit a dynamic factor or copula
before completing the six marginal studies.
