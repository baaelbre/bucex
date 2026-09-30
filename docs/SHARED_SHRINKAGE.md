# Shared and independent shrinkage in 1.9.4

This is the historical lognormal specification. The current half-normal
reference and its GIG updates are described in [GIG_UPDATES_197.md](GIG_UPDATES_197.md).

For each component c (level, slope or seasonal), the shared model uses

    log(tau_c / a_c) ~ Normal(0, w^2)
    s_cj | tau_c ~ Normal(0, tau_c^2).

The signed s_cj multiplies the response's standardized state process. Its
absolute value is the physical innovation SD. Sharing tau_c does not equate
innovation variances, warming rates or temperature trajectories.

Use `SharedShrinkage.from_sd(anchors, log_sd=w)` to declare the hierarchy in
Normal coefficient SDs directly. `IndependentShrinkage.from_sd(...)` declares
the same marginal scale-mixture prior for a single response, with its own
hyperparameters. Both constructors leave initial-rate priors separate.

The reference anchors are 0.01, 0.0001, 0.01 and w=log(3). The two mixture
specifications have exactly identical one-response marginal priors. The fixed
Normal comparator has SDs equal to those anchors and a different prior shape.

The sampler updates u_c=log(tau_c/a_c). With J active coefficients its
conditional log density, up to a constant, is

    -u_c^2/(2 w^2) - J u_c - sum_j(s_cj^2)/(2 a_c^2 exp(2 u_c)).

The -J u_c term is the Normal-density normalization. The update is a scalar
slice step. Standardized state paths do not enter this conditional density.
The state sampler retains FFBS/Laplace-MH and independent parallel chains.

Archived median-parameterized fits retain their original conversion.
`scale_parameterization="normal_sd"` is explicit in every new hierarchy and
archive, and all prior simulation/calibration/reporting honors it.

Only six response-specific coefficients inform each shared scale, through the
state likelihood. Moreover, the observed summaries are derived from the same
temperature record. Residual independence is a working assumption; a shared
scale does not account for contemporaneous residual correlation. Review the
leave-one-summary-out fits and shared-scale diagnostics before interpreting
pooling as robust.
