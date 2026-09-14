# SERRA revision: requests, executable evidence, and remaining decisions

This matrix follows the seven major and three minor comments in
`Review_Bayesian_structural_TS(2).pdf`. A runnable workflow is **not** an executed
scientific experiment. Execution and numerical-test evidence are recorded in
`VALIDATION.md`; paper configurations must still be run with adequate sampling,
and the manuscript and response letter must be revised using their results.

The independent analysis is a complete primary-paper route. Neither a shared
factor nor a copula is necessary to answer the review. Keep the independent
six-series results if a joint model does not mix adequately or does not improve
held-out calibration. The optional joint models answer additional questions;
they must pass their own numerical and scientific checks.

## Major comments

| Comment and requested evidence | Model, script, and outputs | What can be claimed after running it | Remaining manuscript or scientific work |
|---|---|---|---|
| **1. Independent means and extremes are described as probabilistically unified.** Clarify that the submitted fits are structurally analogous, with no information exchange. | `research.serra.run`, independent configuration: six separate `Model` fits and per-series reports. Hierarchical configurations pool structural parameters. Shared configurations use `Shared`/`Departures`; copula configurations add an explicit joint observation likelihood. | The independent route has the same component language for means and extrema. A shared fit pools information through its common states; a copula fit additionally models contemporaneous residual dependence. These are distinct models. | Rewrite the original abstract and methods even if joint results are added. Use “parallel” or “structurally analogous” for independent fits. State exactly what is shared in each joint model. Shared Gaussian means and GEV locations are different distributional summaries; their common path is a descriptive common location change, not automatically a causal warming attribution. |
| **2. Near-zero evolution variances may be prior driven.** Compare every process prior and posterior and vary hyperparameters; propagate sensitivity into trajectories, return levels, and exceedance probabilities. | `research.serra.sensitivity`: baseline plus process-prior scale/hyperparameter variants, including available SSVS/lasso specifications. Outputs include prior/posterior SD and variance summaries, trajectories, scientific contrasts, risks, and MCMC diagnostics. `research.serra.simulate` supplies known-zero/weak/dynamic cases when selected in configuration. | Report which conclusions are stable and which are sensitive; weak identification is a result, not evidence of a failed experiment. Prior/posterior separation alone is insufficient if the chains have not mixed. | Run the paper settings and compare on matched data, seasonal flexibility, initial-state priors and observation priors. Report posterior/prior plots for **all** process variances, initial slope sensitivity, and scientifically meaningful risk changes. A continuous prior gives no posterior probability of an exactly absent component; use only genuine selection indicators for that statement. |
| **3. A local Gaussian approximation can fail near a negative-shape endpoint, including the 2019 record.** Validate effects on inference in extreme years. | `research.serra.endpoint`: the real TXx July 2019 event and optional near-endpoint simulations, comparing the uncorrected Laplace approximation with exact-target Laplace–MH under matched priors and data. `research.serra.simulate` adds known-truth recovery. | Laplace–MH uses the actual GEV/copula likelihood in its MH correction; the Gaussian approximation is a proposal, so its local shape affects efficiency rather than defining the retained target. Show actual changes in event probabilities, paths and endpoint distances, plus acceptance and ESS. | Revise methods/appendix to distinguish approximate inference from an approximate proposal with exact correction. Run sufficiently long chains in the difficult record years. Finite endpoint support, rejection rates and ESS matter; a high mean acceptance or one successful fit is not a validation of all settings. |
| **4. The shape prior `Uniform(-0.5, 0.5)` may influence risk conclusions.** Change its bounds and reassess record/exceedance probabilities. | `research.serra.sensitivity`: default bounds `[-0.5, 0.5]` and wider lower/upper-bound variants, keeping other settings fixed. `research.serra.simulate`: separate **generating-shape** grid from -0.5 to 0.5, with an estimation prior covering the grid's endpoints. | The prior-bound experiment assesses sensitivity of the Uccle conclusions. The generating-shape grid assesses inference performance in different tail regimes. These are different experiments and should be reported separately. | Run all planned variants and check posterior mass near either bound, risks, return levels and MCMC exploration. An endpoint estimate is meaningful only for negative shape; when a posterior includes nonnegative shape, a finite endpoint interval is conditional on negative shape and its probability must be reported. Do not silently discard other draws. |
| **5. Central 90% predictive coverage does not establish tail calibration.** Add higher coverage levels and direct tail checks. | `research.serra.validate`: leave-future-out fits, predictive scores and PIT values, plus coverage/tail outputs at configured 90/95/99% levels. Outputs: `coverage_by_case.csv`, `coverage_summary.csv`, `held_out_pit.csv`, `score_summary.csv`, optional `joint_log_scores.csv`, and `mcmc_{origin}.csv`/`targets_{origin}.csv`. Retain channel/horizon labels. | Evaluate central coverage and one-sided upper/lower quantile exceedance separately. Central 99% intervals use the 0.5% and 99.5% quantiles, and are not a direct test of the 99th percentile. Tail-weighted CRPS, quantile scores and threshold-event scores complement coverage. | Run enough forecast origins to make rare-event counts informative, using only training data for prior calibration and threshold selection when data driven. Report counts and sampling uncertainty. Overlapping horizons and neighboring months are not independent trials. Very high nominal coverage with few observations cannot certify rare-tail calibration. |
| **6. Nearly constant-width 20-year annual forecasts may hide missing state uncertainty.** Explain and verify propagation of latent trend and parameter uncertainty. | `research.serra.forecast_check`: posterior forecast draws, latent and observation intervals at 90/95/99%, forecast variance decomposition, and annual aggregation. `FitResult.forecast` propagates future state innovations and the posterior terminal state/parameter draws. | Separate observation variability from uncertainty in the future location. For finite-variance margins, check `Var(Y)=E[Var(Y|state,parameters)]+Var(E[Y|state,parameters])`. A broad observation distribution can mask growing latent uncertainty in the total interval. | Discuss width and asymmetry actually observed, rather than imposing widening curves. A warming shift alone does not imply that the upper bound must widen faster than the lower bound. For a local linear trend, innovation variance grows as `h*sd_level^2 + h*(h-1)*(2*h-1)*sd_slope^2/6`, with additional terminal-state uncertainty. GEV conditional second moments require shape below 1/2. Even if all retained draws satisfy this, integration over a shape posterior approaching 1/2 can diverge. Label finite sample variance decompositions as Monte Carlo summaries; do not certify theoretical predictive variance from finite retained draws. |
| **7. Is taking the annual maximum of 12 monthly maxima theoretically justified?** Explain what annual forecast is being computed. | `research.serra.forecast_check`: annual maxima/minima from complete-year predictive paths and a comparison of draw-wise conditional product CDFs with Monte Carlo aggregation. Calendar-risk tests reject accidental use of incomplete years by default. | If monthly blocks partition a year, `max_month(max_day temperature) = max_day_in_year temperature` exactly. No new EVT approximation is required for this identity. The annual distribution induced by different monthly GEV laws need not itself be a single GEV. | Explain block definitions and missing/incomplete years. With residual independence **over time conditional on the latent path and parameters**, the annual CDF is a product of monthly conditional CDFs; average those products over the joint posterior. A product of already-marginalized monthly predictive CDFs is generally wrong because it drops shared latent uncertainty. A contemporaneous copula does not remove this temporal assumption. |

## Minor comments

| Comment | Code/report support | Remaining action |
|---|---|---|
| **1. Proofread typographical and formatting errors**, including “import” versus “important”, missing punctuation and indentation. | No statistical experiment can address this. The release supplies reproducible outputs for a revised manuscript. | **Manuscript edit pending:** proofread the complete revised source and rendered PDF; check the examples identified in the review and changes introduced during revision. |
| **2. Define notation before use**, including beta in equation (5) and k in the cited line. | The component API and model documentation make private, common and departure states explicit. | **Manuscript edit pending:** define each index, state, variance/SD, initial condition and transformation before first use. Distinguish GEV location from Gaussian mean; distinguish process variance from its signed/nonnegative noncentered scale. Update equation numbers after editing. |
| **3. Increase figure-axis font sizes and check printed readability.** | The SERRA reporting layer creates fresh vector figures from saved fits. | **Figure review pending:** render final panels at their manuscript size and inspect axes, ticks, legends, intervals and lower-tail labels. Readability must be assessed on the final multipanel layout, not only a large standalone figure. |

## Additional joint-model checks prompted by the revision

A shared latent signal induces dependence after integrating over states. A
Gaussian copula models **remaining same-time dependence conditional on those
states and parameters**. It does not supply arbitrary serial dependence or
asymptotic tail dependence, and it does not guarantee smaller or calibrated
credible intervals. Compare an independence copula, estimated correlation, and
reasonable LKJ prior choices on identical marginal models.

The deterministic inequalities among observed TX/TN minima, means and maxima
are properties of summaries from the same daily record. Check them in observed
and joint replicated data using the ordering diagnostics. A Gaussian copula has
full interior support and cannot enforce hard order. Sorting simulated values
would change the specified marginals. Conditioning on valid order would
require a parameter-dependent normalizing probability in the likelihood and
would also change the marginals. The root mathematical issue can already be
seen when an unbounded Gaussian mean is paired with a finite-upper-endpoint
GEV maximum: these unchanged marginals cannot satisfy mean <= maximum almost
surely. See `COPULA_AND_ORDERING.md` for the implications and next-model choices.

## Suggested response-letter discipline

For each comment, report the revised claim, exact manuscript location,
experiment settings and diagnostics, then the result with uncertainty. Label
supplementary joint analyses as supplementary if the primary independent
analysis carries the paper. Never write “all comments addressed” merely because
scripts and tests exist. Numerical correctness, MCMC convergence, empirical
calibration and editorial revision are separate requirements.

## Release execution register

`validation/reviewer-smokes.json` records 21 prior/shape-bound sensitivity fits
(comments 2 and 4), four actual-2019 endpoint fits (comment 3), 22 generating-shape
grid fits (comments 3 and 4), six near-endpoint stress fits (comment 3), and
three 24-month forecast/annual checks (comments 6 and 7). All completed with
four retained draws per chain: these are execution checks, not answers about
robustness or coverage. `validation/smoke-workflows.json` additionally records
independent/joint fits (comment 1) and held-out independent/copula jobs with
90/95/99% coverage outputs (comment 5). The complete paper studies and all three
minor manuscript/figure edits remain required; see `VALIDATION.md` for longer
pilot evidence and its limitations.
