# Migration to the publication API

The publication release is numbered **1.0.0**. The earlier 1.9.x versions were
research snapshots; their version numbers do not imply compatibility or a
feature superset in this release. Keep original snapshots with their original
results if reproducing an earlier analysis.

| Earlier pattern | Publication API |
|---|---|
| Release-specific scripts and settings | `research/config/*.json` and portable modules |
| One package-internal paper model | `Model`, `Channel`, `MultiSeriesModel` |
| Hard-coded structural state layout | `ComponentBlock` and generic compilation |
| Fixed-size plotting branches | Named results and `bucex.plots` handlers |
| Development dependence extensions | Product observation likelihood with optional innovation-prior pooling |
| One HPC job launcher per version | Stable independent `chain_ids`, `combine_fits`, Python experiment runner |

The original flat 1.0 draft has also been revised before publication.
`components`, `models`, `observations`, `parameters`, `results` and `plots`
now have explicit responsibilities. `bucex.distributions` and
`bucex.plotting` remain compatibility imports for public family/plot entry
points. Private inference modules from that draft are not supported APIs.

The new fit format is **schema 2**, using JSON and NPZ without pickle. It
retains covariates, component definitions, channels and chain dimensions.
Earlier draft/schema-1 and research archives are rejected with an explicit
message. They should be read with their original version; new scientific
results should come from rerunning the new release rather than relabelling
old posterior arrays. The main model's priors and target are retained, but
random-number ordering and proposal construction can change, so seedwise
identity with old releases is not promised.

Default modes and physical units are documented in the main README. Fixed
Normal SDs and half-normal hyperprior SDs have distinct meanings. Named
constant/fixed scale declarations cannot be combined with `SeasonalScale`;
choose the intended dispersion model explicitly.
