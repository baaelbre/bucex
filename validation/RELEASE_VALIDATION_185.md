# BUCEX 1.8.5 checks — 2026-09-23

This release changes the seasonal research configuration and adds a prior-grid
runner; the existing monthly reference and paper-model sampler are retained.

- `python -m compileall -q bucex research tests`: passed.
- `python -m research.seasonal.preflight`: 538 complete blocks (MAM 1892 to
  JJA 2026), seasonal copula; medians 0.01, 0.0001, 0.01, 0.003. Integrated
  30-year prior SDs for level, integrated slope, same-season contrast and
  initial-slope displacement: 0.262587, 0.180788, 0.185677, 0.582050°C;
  initial-rate SD 0.194017°C/decade.
- `python -m research.seasonal.grid --dry-run`: ten distinct, declared settings.
- A reduced reference grid run with two chains, three warmup and four retained
  draws fitted the complete 515-block training history at the November 2020
  cutoff and wrote held-out scores, PITs, per-response and joint-score summaries,
  convergence reports and a source-data checksum. Restart reused the completed
  fit. These few draws failed numerical convergence checks, as expected; its
  score is **not** a scientific estimate or a prior selection.
- A separate reduced screen launched two settings concurrently with
  `fit_parallel(..., jobs=2)`; each started two chain processes and saved
  independent logs. Both completed, and the driver combined their scores only
  after each setting's completion marker appeared. The CLI accepts `--jobs 10`;
  ten long concurrent fits have not been run in this workstation.

`pytest` was unavailable in the release workstation. The full test suite can
be run after `python -m pip install -e ".[plot,test]"` on the biobot. No long
four-chain 1.8.5 posterior analysis has been reported as converged here.
