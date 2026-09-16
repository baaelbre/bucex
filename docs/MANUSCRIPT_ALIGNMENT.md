# Alignment with the September manuscript revision

The release follows the revised manuscript's order: private Gaussian/GEV
margins, continuous FS shrinkage, observation scale, joint residual dependence,
scientific contrasts/risks, and staged validation. It preserves the independent
six-series fallback. No preliminary SSVS figure is relabelled as a result from
the new continuous model.

| Manuscript element | BUCEX 1.6.4 |
|---|---|
| FS normal/lasso/triple-gamma comparison | `fs_priors`, identical marginal model declarations, median-matched SDs |
| Practical innovation magnitude | `innovation_effect_draws(H)`; direct level, integrated slope and exact dummy-seasonal propagation |
| Joint posterior | `MarginalPriors` and copula-conditional updates in every margin |
| Continuous coefficient computation | Exact weighted Gaussian update or Gaussian-reference elliptical slice |
| Non-centred mixing improvements | Prior whitening; continuous location ASIS by invariant rescaling; diagnostics on physical SDs and scientific quantities |
| Seasonal/secular observation scale | `LogScale`, independent per channel |
| Seasonal residual dependence | Pooled Fisher partial-correlation contrasts with an LKJ baseline |
| Rare compound probabilities | Bivariate residual integration per state/parameter draw |
| Model comparison | Matched rolling-origin experiments, paired score comparisons; no automatic Bayes-factor or model-averaging claim |
| Shape, weak dynamics and endpoint checks | Research configurations and scripts, with wider fitted support for the generating shape grid |

The manuscript's implementation-status paragraphs and table can now be updated
from “proposed backend” to “implemented and tested in BUCEX 1.6.4.” That change
is about software. Statements about scientific recovery, convergence, prior
robustness or predictive superiority still require completed research runs.

Record these concrete choices in the final paper:

- Reference median physical innovation SDs are (.02, .00005, .02). These values
  are medians, whereas the preliminary SSVS pilot's same numbers were slab SDs.
- Lasso lambda² is fixed at one; component coefficients are rescaled to match
  the declared medians. This is compatible with the manuscript's calibration
  principle, not its optional learned-Gamma hyperprior alternative.
- Triple-gamma uses fixed a=c=.5 and global multiplier one; local numerator and
  denominator variables are sampled. The spikier a=.1 case is a sensitivity
  configuration. No finite slab cap is used by these default configurations.
- Initial level is N(0,20²) in the internal response orientation, using no
  estimated prior centre. The pilot's data-centred N(median,3.2²) is a different
  prior and must remain labelled as such.
- Shape is N(0,.30²), truncated to [-.5,.5]; uniform on the same support,
  SD .20, and wider support are separate sensitivity changes.
- Linear scale uses t/120 for monthly data. The RW scale is anchored at z_0=0
  with a signed-normal innovation coefficient. These are explicit alternatives
  to the retained legacy `phi` API.
- Seasonal copula baseline coordinates have the LKJ prior; seasonal effect
  coefficients have independent normals. It is not an LKJ prior on each
  monthly matrix, and it is not invariant to a permutation of channels.

Residual serial dependence, a residual t copula, an ordering-constrained
likelihood and free dynamic-factor loadings are outside this release's private
paper workflow. Existing shared-state APIs are separate. These omissions must
remain visible wherever the manuscript motivates them as future comparisons.
