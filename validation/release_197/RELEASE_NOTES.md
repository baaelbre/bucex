# BUCEX 1.9.7

The half-normal shared shrinkage scales now use direct generalized inverse
Gaussian (GIG) Gibbs draws instead of log-scale slice sampling. The public
`SharedShrinkage.half_normal(...)` API selects this automatically. For J active
coefficients, tau squared has GIG parameters ((1-J)/2, A**-2, sum(s_j**2)).
The sampler retains the same statistical model and posterior target.

The implementation uses chain-local random generators, records the update method
and GIG draw counts in saved fits, and preserves existing parameter names.
It handles optional independent half-normal hierarchies and active-component
subsets. Other hyperprior families retain their slice updates. Very small
coefficient norms use an exact gamma rejection envelope for the same GIG target,
avoiding unstable numerical bounds without a variance floor.

The 1.9.6.1 reference priors, complete experiment matrix, screening and paper
budgets, forecast settings and HPC dispatch hotfix are retained. Current launchers
and default output paths identify version 1.9.7. Old archives remain readable;
use a new results root for new runs. No production runs have been submitted.

The API notes and conditional derivation are in `docs/GIG_UPDATES_197.md`.
`RELEASE_VALIDATION.json` and `validation/release_197` record the checks performed
for this release. Historical validation records remain under their versioned
folders and are not new scientific results. Direct conditional sampling does
not remove dependence among Gibbs blocks or guarantee posterior convergence.
