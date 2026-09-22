# BUCEX 1.8.1 — hierarchical shrinkage for level, slope and seasonality

The current SERRA specification estimates three shared shrinkage
hyperparameters: one each for level, slope and seasonal innovations. Each
response retains its own innovation SDs and latent components. Seasonal pooling
regularizes changes to the repeating seasonal pattern, while the initial
pattern and monthly observation scales remain response-specific.

The existing `SharedShrinkage` API and exact joint FS/copula sampler already
support these components. This release makes three-component pooling the
research default and adds its sensitivity checks; it introduces no new sampler
approximation. Two-component pooling, fixed normal priors and separate
univariate fits remain available. Saved 1.8.0 fits keep their original meaning;
installing 1.8.1 does not add seasonal pooling to an existing posterior.

The three default hyperprior anchors are .0025, .0000125 and .02 for level,
slope and seasonality. These anchor uncertain prior medians of individual
innovation SDs; they are neither fixed process SDs nor posterior estimates.
Each shared median has a lognormal hyperprior with log SD `log(2)`.

`START_HERE.md` now supplies the complete command sequence: exploration, smoke
test, fixed/pooled half/quarter comparison, historical prediction, seasonal
pooling and anchor sensitivity, copula comparison, constant-scale and fixed
seasonality checks, focused reviewer prior checks, confirmation, final fits
and manuscript figures. Optional follow-ups are marked. Full-record
configurations end in August 2026 and use four local worker processes, without
requiring Slurm.

Mixed two- and three-component comparison figures label an absent shared
seasonal parameter as `not pooled`. Dry-run plans now expose structural,
observation-prior and copula settings as well as innovation anchors.

Read [START_HERE](START_HERE.md) for commands and
[SHARED_SHRINKAGE.md](docs/SHARED_SHRINKAGE.md) for the model and public API.
The software checks are recorded in
[RELEASE_VALIDATION.md](validation/RELEASE_VALIDATION.md). Short smoke chains
check execution only; publication conclusions still require convergence,
predictive adequacy and sensitivity checks on the full temperature record.
