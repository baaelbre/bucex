# Paper reference — 1.9.5

Use the BIOBOT guide in `BUCEX-1.9.5-commands.md` and the `paper` tier. A paper
reference has 2 chains, each with 6,000 warm-up and 20,000 retained draws. It
fits 538 complete seasons through August 2026, with shared half-normal
innovation scales (0.01, 0.0001, 0.01), separate initial-rate SD 0.01, initial
level/seasonal SD 20, and no copula.

The forecast covers 120 seasonal transitions with 12,000 draws and 95%
intervals. The separate pre-2019 reference ends in May 2019 and uses a
one-season forecast with 20,000 predictive draws. Its retrospective held-out
observations never enter training.

Do not treat job completion as convergence. Inspect `convergence.json` and
`final_check.json`, and assess prior sensitivity and held-out calibration.
The automatic collection writes review ZIPs and labels unconverged figures
as diagnostic. The full `.bucex` archives remain in the task reports.
