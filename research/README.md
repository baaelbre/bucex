# Paper analyses

Everything here runs from the source tree, after `pip install -e '.[research]'`.
The library never imports this folder. Analysis choices live in JSON; scripts
consume `bx.Model`, `bx.fit`, saved `FitResult` objects and `bx.plot`.

## Five configurations

| Configuration | Blocks | Observation dispersion | Innovation priors |
|---|---|---|---|
| `main.json` | meteorological seasons | seasonal | shared half-normal scales |
| `monthly.json` | calendar months | monthly | shared half-normal scales |
| `private.json` | seasons | seasonal | private fixed Normal SDs |
| `constant_dispersion.json` | seasons | one sigma per response | shared half-normal scales |
| `monthly_constant_dispersion.json` | months | one sigma per response | private fixed Normal SDs |

The last row isolates the structure of the original manuscript. All five use
the current Normal priors and corrected sampler; the last row is not an exact
reimplementation of the original prior or inference approximation.

`main.json` is the single reference. Other JSON files use `extends` with nested
overrides. Unknown fields and inheritance cycles fail. Resolved settings,
observations and a data hash are saved for every run. Existing results with
different settings are never silently reused.

Main calibration: A_level=.1, A_slope=.002, A_seasonal=.1. Initial level and
seasonal contrast SDs are 10; initial rate SD is .01 per season. Observation
variance is IG(2,2); seasonal log-scale contrast SD is .3; xi~Normal(0,.3²).
These settings are copied from the latest 1.9.8.4 reference configuration.
Private fits use fixed Normal SDs with the same values; there is no private
hyperprior in this release.

Monthly calibration uses A_level=.1/sqrt(3), A_slope=.0003832923311016883,
A_seasonal=.1 and initial-rate SD=.01/3. The slope scale matches the exact
integrated slope contribution to level variance at 30 years:
`A_beta_monthly = A_beta_seasonal * sqrt(S(120)/S(360))`,
where `S(h)=h*(h-1)*(2*h-1)/6`. This does not make the entire prior process
identical at every horizon. Same-phase seasonal change variance scales with
years, so its amplitude is retained. The initial seasonal contrast SD remains
10 in both dimensions; its per-phase variance changes slightly with period.

The daily loader returns 538 complete seasonal blocks from MAM 1892 through
JJA 2026, or 1,616 complete months. The incomplete initial winter is omitted.
It never fills daily gaps or removes incomplete interior blocks. Dates are
block ends. Each model's phase zero is its first observed block.

## JSON sensitivity grid

Run `python -m research.sensitivity --list` for the exact table and zero-based
scheduler index. `config/sensitivity.json` contains:

- 3 × 3 level–slope grid: A_level in {.05,.1,.2} and A_slope in {.001,.002,.004}.
- Seasonal calibration .05 and .2 around the .1 reference.
- Initial-rate SD .005 and .02.
- Shape-prior SD .15 and .6.
- Initial-level SD 5 and 20; initial-seasonal contrast SD 5 and 20.
- Observation variance-prior scale b=.5 and 8, keeping a=2.
- Seasonal log-scale contrast SD .15 and .6.
- Static seasonality, and an explicitly tight innovation calibration
  (.01,.0001,.01) to inspect level–slope allocation.

There are **25 settings**, including the reference in the 3 × 3 grid. No
automated choice of a preferred setting is made from predictive performance.
Comparison figures retain uncertainty, and CSV exports measure differences
between posterior mean paths without equating those differences with full
posterior uncertainty. Groupwise comparisons can use a smaller custom JSON.

```bash
python -m research.sensitivity --profile screen --variant 0 --workers 2
python -m research.sensitivity --profile screen --compare
```

## Outputs and manuscript figures

Each run contains `run.json`, `observations.csv`, `fit.bucex`,
`posterior_summary.csv`, `sampler.json`, `posterior_checks.csv`,
`residual_association.csv`, and (when figures are enabled) risk/forecast CSVs
and `figures/`. Fits retain all channels, chains and retained draws.

| Manuscript content | Script / native plot |
|---|---|
| Seasonal records and descriptive centred LOESS | `figures.exploratory` |
| Levels and warming rates | `bx.plot(..., type="level" / "slope")` |
| Initial/final cycle and seasonal evolution | `cycle` / phased `seasonal` |
| Main normal-score QQ; appendix PIT and serial checks | `normal_qq`, `pit`, `acf` |
| Innovation, initial-rate, sigma, xi and shared-scale priors/posteriors | `prior_posterior` |
| Fixed-threshold risk and 20-year return levels | phased `risk` / `return_level` |
| Thirty-year forecasts, including latent locations | `forecast` |
| Prior implications at 10 and 30 years | `research.prior_calibration` |
| Validation coverage and widths | `research.validation`, `bx.coverage` |
| Recent-origin forecast and predictive-CDF plots | `research.validation --recent` |
| MAM-2019 refit, summer-2019 and 39.7°C risk | `research.pre2019` |
| Calibration path comparisons | `research.sensitivity --compare` |

All scientific figures use Matplotlib, mean curves, 95% bands, no titles, and
TX blue/TN red where response colouring is appropriate. The exploratory LOESS
is descriptive; it is not an estimate of a GEV location or tail. Monthly risk
figures refer to the selected month (e.g. August), not to a three-month JJA
maximum. Their probabilities must not be compared as if their block events
were identical.

`python -m research.figures <run-directory>` rebuilds all standard figures from
saved objects. `config.plot.format` can be changed to `pdf` for vector output.
`return_years` controls the return-period figures (10, 20, 50 and 100 years by default).
Observation-scale panels retain the distinction between mean estimates and
central intervals, even for skewed posteriors.

## Validation

Main origins are ends of SON 1956, 1976 and 1996. Requested horizons are
30 years. With the current record ending in JJA 2026, the last window has
119 observed blocks; that partial final window is reported explicitly in the
CSV rather than presented as 120 available outcomes. Other windows have 120.
Recent origins are SON 2015 and SON 2020. Evaluation reports coverage at
90%, 95% and 99%, mean interval width, and counts below/above the intervals.
No held-out CRPS table is generated.

Forecasts use only the origin's fit. Saved coverage tables and held-out CDFs
can be inspected without repeating the analysis. A held-out predictive-CDF
ECDF below the diagonal indicates systematically large CDF values, consistent
with observations being too high relative to their predictive distributions;
it does not identify why the bias occurred.

## Data and run provenance

See `data/README.md` and `data/SOURCES.md`. The provided source combines an
older record with later exports; this package does not newly homogenize that
extension. Source data are kept outside the wheel. The software license does
not replace the observations' source terms.

`smoke` uses 36 blocks and only a few draws to check execution. Its validation
uses an early artificial origin. It must never supply paper estimates or
convergence claims. `screen` and `paper` fit the complete specified records.
A full publication fit must be rerun and checked after this API refactor.

## Compare the five structural analyses

```bash
python -m research.compare results/main/screen results/private/screen \
  results/monthly/screen results/constant_dispersion/screen \
  results/monthly_constant_dispersion/screen --output results/comparisons
```

These overlays use actual block dates and convert slopes to per-decade units.
Monthly and seasonal cycles are kept separate; their phase effects and block
events are not interchangeable.
