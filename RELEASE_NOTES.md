# BUCEX 1.8.3

Start with [START_HERE](START_HERE.md) for the ordered commands and interpretation.
The monthly baseline and separate seasonal alternative retain summer 2026.
The reference prior anchors now match the COMPSTAT marginal second moments;
this changes the draft model relative to 1.8.2 and requires refitting.
Old saved fits remain old fits, retaining their original priors and data.

The seasonal sampler uses the existing public model, prior, fitting and archive
interfaces. Explicit meteorological scale phases are new; plain period-4 scales
retain their older relative phases. No inference engine was duplicated in research.

The comparison uses common seasonal targets, identical daily training cutoffs,
and paired future paths. It does not compare likelihoods on different data.
Rank/clustering diagnostics are provided; no r>1 likelihood is claimed.

See [validation/RELEASE_VALIDATION.md](validation/RELEASE_VALIDATION.md) for
executed software checks. Software validation does not establish convergence,
model adequacy or publishable scientific results for the long Uccle runs.
