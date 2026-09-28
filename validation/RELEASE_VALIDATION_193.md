# Manuscript SD correction: paper_sd_20260928

The reference now uses Normal prior SDs directly: (0.01, 0.0001, 0.01) for innovations and 0.01 for the initial slope (variance 0.0001). **22 affected tests passed**, including six short two-chain fits, forecasts/reports, fixed-prior gates, sensitivity multipliers, HPC bundle contracts and submission protocol. A dedicated calibration test verifies all four 30-year effects and the 0.20/0.40/0.80 degrees C/decade initial-rate sensitivity. All 508 corrected paper configurations compile and their folds are complete. Detailed results are in `release_193/paper_priors_correction/`.

The 320-test result below describes the original release before this numerical reference correction. It establishes the broader software regression baseline, not results from the corrected production model. No corrected production or HPC run was performed in this workspace. See `../PAPER_PRIORS_CORRECTION.md` for applying the patch and preserving earlier results.

---

# BUCEX 1.9.3 verification

All **320 tests passed in one regression run**. The JUnit report and log are in `release_193/`; `../RELEASE_VALIDATION.json` records runtime versions, scientific configuration/source/data fingerprints, counts and verification scope.

The new tests verify the exact fixed-Normal median-absolute calibration for all four coefficients, reject accidental hierarchy combinations, and confirm that sensitivity factors act on SDs. Six real short fits cover both Gaussian means and all four GEV maxima/minima, each with two parallel chains. Saved-fit decoding, thirty-year forecasts, fixed-prior and initial-slope reports are exercised. Broader tests cover prediction, original-scale lower tails, held-out coverage, reporting, resume and collection.

All 508 paper task specifications compile with one response, no copula and no sampled shrinkage scales. Their validation folds are complete. All 88 bundles map to the declared response tasks with twelve allocated cores and two chains per response. The resolved 23-setting grid, reference physical calibration and full bundle plans are included here.

Submission was tested with a stub scheduler to inspect actual arguments, probe dependencies and exported environments. Dry runs confirm Gallade selection, twelve-core requests and array caps. The login submitter never executes a supplied incompatible numerical Python; its modules also parse with Python 3.9 syntax. All 21 shell scripts pass syntax checks. A bundle test starts six real lightweight processes, verifies their overlap and thread limits, deliberately fails one and confirms the remaining processes finish and all outcomes are recorded.

The actual TNn startup-probe CLI completed a short fit and report successfully. This workspace has fewer than twelve allocated CPU workers, so the complete twelve-worker MCMC probe was not run locally; its allocation guard correctly refused the undersized allocation. Two-chain scientific fits were tested per response, and six-process scheduling separately. Each HPC submission runs the full twelve-worker probe on a compute node before releasing its experiment arrays.

No live Gallade submission, production fit or production convergence claim was made here. Screen/paper budgets and numerical gates are supplied for the user to run and inspect; they do not establish predictive adequacy or an optimal prior. Historical 1.9.2 tests use frozen configuration snapshots to retain their original meaning.
