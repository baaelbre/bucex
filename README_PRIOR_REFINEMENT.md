# Two additional innovation-prior settings

Extract this archive into your BUCEX 1.7.4 checkout, beside `pyproject.toml`.
It adds a short SERRA runner and a JSON configuration. It does not replace
package code or edit completed results.

The runner fits only these new TNm settings:

| Setting | Level SD prior median | Slope SD prior median | Seasonal SD prior median |
|---|---:|---:|---:|
| `level_half_slope_quarter` | 0.005 | 0.0000125 | 0.020 |
| `level_slope_quarter` | 0.0025 | 0.0000125 | 0.020 |

It reuses `level_slope_half` and `level_quarter_slope_half` from your completed
assessment. The combined report keeps `level_slope_half` as the reference.
No full `.bucex` archives are required for reuse: the compact CSV/JSON exports
in the uploaded results are sufficient.

## Run on biobot

From the activated environment, inside `~/bucex`:

```bash
python -m pip install -e ".[plot]"

# Inspect settings, forecast dates and work counts; no MCMC is started.
python -m research.serra.refine_priors \
  --previous results/serra_priors_refine/prior_assessment_20260921T123808_317223Z \
  --stage plan

# Fit only the two new settings and create the four-setting comparison.
python -m research.serra.refine_priors \
  --previous results/serra_priors_refine/prior_assessment_20260921T123808_317223Z \
  --stage all
```

If you moved the previous run, change `--previous` to the directory containing
its `assessment.json`, `config.json`, `sensitivity/` and `predictive/`.

Settings are frozen from the supplied experiment: data through August 2026,
Normal innovation priors, FS, ASIS off, dynamic location seasonality and
repeating monthly observation scales. Each fit uses four parallel chains with
1,000 warmup and 1,000 retained iterations per chain. Two full-record fits and
six historical refits are run; the variants and forecast origins are processed
sequentially, with parallelism within each fit. Forecast periods remain
2001–2005, 2011–2015 and 2021–2025, using 2,000 forecast draws per origin.
This is screening, not a final convergence guarantee.

The JSON base innovation medians remain 0.01, 0.00005 and 0.02. The variants
apply their multipliers once. Do not halve the base medians as well. Within the
new-only assessment, `level_half_slope_quarter` is its internal reference;
the final `combined_comparison/` explicitly restores the previous half/half
reference. All other scientific settings must match for reuse.

## Read the combined results

The runner prints a timestamped directory beneath `results/serra_priors_refine2/`.
Use its **`combined_comparison/`** directory for the four-setting comparison:

- `convergence.csv` and `mcmc.csv`: initial slope and positive innovation SDs.
- `scientific_targets.csv`: period warming and slope contrasts, including ESS.
- `scores_by_origin.csv` and `predictive_comparison.csv`: predictive losses.
- `coverage_summary.csv` and `pit_by_month.csv`: held-out calibration.
- `TNm_level.png`, `TNm_slope.png`, `TNm_prior_posterior.png`: sensitivity overlays.

The existing compact reports do not include all monthly scale coefficient
diagnostics. This add-on does not change that export behavior. Those diagnostics
still need to be checked before final scientific reporting. A shifted posterior,
a smoother curve, or a tiny score improvement alone does not select a prior.

To regenerate the combined figures without refitting, use the completed new
directory printed by the runner:

```bash
python -m research.serra.refine_priors \
  --previous /path/to/completed/previous_assessment \
  --run /path/to/completed/new_assessment \
  --stage report
```

Run `--stage all` once: another invocation starts a fresh assessment. For
completed new results, use `--stage report`. The existing BUCEX assessment
workflow can resume between completed stages; it cannot resume an interrupted
individual MCMC fit from these compact exports.
