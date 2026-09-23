# BUCEX 1.8.2 verification

Verified on 23 September 2026. The final source suite passed **389 tests**, with
zero failures or errors, in 360.84 seconds. One data-quality warning records two
retained daily TN > TX pairs. The JUnit report and machine-readable verification
record are included beside this file.

The built wheel was installed into a separate directory. All 117 package Python
files matched the source byte for byte. A mixed two-channel fit with four shared
scales, monthly observation scales and unrestricted shapes completed, saved and
reloaded; forecasting from the reloaded fit returned finite simulation draws.

Research execution checks covered a six-response, four-process smoke fit;
reporting saved draws; exploratory Figures 1–2; ten manuscript panels; and two
historical forecast origins with saved fold fits. The predictive execution check
produced 144 held-out PIT rows. Initial-slope and shared-scale figures were
visually inspected. All 12 Bash command blocks in START_HERE.md passed syntax
checks, and draft/supplementary configuration plans resolved successfully.

The posterior simulations used in these workflow checks are deliberately short.
They validate software execution, not convergence, model adequacy, or claims
about temperature change. They are excluded from the release archive. The
complete scientific fits and predictive/sensitivity assessments remain to run.

See `release_1.8.2.json` for settings and the distinction between the initial
smoke copula width (0.35) and final predictive-check width (0.25).
