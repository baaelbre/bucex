# bucex 1.6.1

Bayesian unobserved-component state-space models for Gaussian means and GEV
extremes. A model is a response distribution plus readable components; the same
API fits, diagnoses, forecasts and computes risks. Mixed hierarchies use exact
Laplace–MH state updates. Shared components and an optional Gaussian residual
copula extend the joint model. The six independent Uccle analyses remain a
complete, separate route to the SERRA revision.

```bash
python -m pip install -e ".[plot,test]"
python -m pytest
python -m research.serra.tutorials --kind all
```

Run these commands from the extracted directory containing `pyproject.toml`.
`bucex/` is the installable package; `research/serra/` contains every active
research script and configuration. There are no conference directories or
particle inference kernels. The default tutorials are execution checks with
four saved draws, not scientific analyses.

## Build a model through the public API

```python
import bucex as bx

model = bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (
    bx.LocalLinearTrend(),
    bx.DummySeasonal(period=12),
))
prior = bx.ssvs_gev_priors(period=12,
    innovation_slab_sd={"level": 0.02, "trend": 0.00005, "season": 0.02})
fit = bx.fit(y, model, priors=prior, engine="laplace_mh",
             parameterization="fs", mcmc=bx.MCMC(draws=1000, warmup=1000, chains=4))
fit.diagnostics()
fit.plot("level")
fit.exceedance_probability_draws(35)
fit.forecast(12).summary()
fit.save("fit.bucex")
```

Here `y` is your aligned series. Gaussian models use `bx.Gaussian()` and exact
FFBS. Each Uccle minimum is fitted with `tail="lower"`; the result API restores
original Celsius orientation automatically.

## Start with the six independent analyses

The same short runner and one configuration replace six nearly identical
scripts. These jobs can be run separately; none depends on a successful joint
or factor fit.

```bash
python -m research.serra.run --config research/serra/config/independent_full.json --series TXm
python -m research.serra.run --config research/serra/config/independent_full.json --series TNm
python -m research.serra.run --config research/serra/config/independent_full.json --series TXx
python -m research.serra.run --config research/serra/config/independent_full.json --series TXn
python -m research.serra.run --config research/serra/config/independent_full.json --series TNx
python -m research.serra.run --config research/serra/config/independent_full.json --series TNn
```

Omit `--series` to fit all six sequentially. Use `independent_smoke.json` first
to check execution. All six supplied summaries are **monthly**, with period 12.
For an independent extreme series, append `--phi linear`, `--phi rw`, or
`--phi ssvs` to assess changing log scale. Priors and numerical controls live
in JSON; the copied configuration records command-line overrides.

## Optional joint analyses

```bash
python -m research.serra.run --config research/serra/config/hierarchical_smoke.json
python -m research.serra.run --config research/serra/config/shared_smoke.json
python -m research.serra.run --config research/serra/config/joint_smoke.json
python -m research.serra.run --config research/serra/config/joint_smoke.json --copula
python -m research.serra.run --config research/serra/config/shared_smoke.json --copula
```

Replace `_smoke` with `_full` for the full 1892–2022 record and initial study
budgets. Hierarchical analysis pools selection information while retaining
separate trajectories. Shared analysis estimates a common warming component
with unit loadings and constrained departures; each series has its own
seasonal cycle. `--copula` adds contemporaneous residual dependence. To make
the private-trend dependence comparison controlled, `joint` fixes the copula
correlation to identity; `joint --copula` estimates it with the same priors.

A Gaussian copula does **not** enforce minimum ≤ mean ≤ maximum, and does not
provide nonzero asymptotic tail dependence. The package reports physical
ordering violations on unaltered predictive draws. See
[dependence and ordering](docs/COPULA_AND_ORDERING.md) before interpreting
compound probabilities.

The copula passes independent numerical reference tests, but the six-series
mixed Uccle pilot still mixes poorly. Treat this joint analysis as experimental;
use the independent route as the primary revision workflow until adequate
joint-model convergence has been demonstrated.

## Assess the revision

```bash
python -m research.serra.validate --config research/serra/config/independent_full.json --series TXx
python -m research.serra.validate --config research/serra/config/shared_full.json
```

Held-out validation saves PITs, proper scores, central 90/95/99% coverage,
both-tail quantile coverage, and diagnostics for every refit. It releases each
fit before the next origin. Results appear in fresh timestamped directories
under `results/serra/`; no run overwrites an earlier fit.

Read [research/serra/README.md](research/serra/README.md) for sensitivity,
shape-recovery, endpoint and forecast experiments; [the reviewer matrix](docs/REVIEWER_MATRIX.md)
maps each comment to evidence still needed. [Validation](docs/VALIDATION.md)
distinguishes tests and pilots from completed scientific studies.

The supplied MCMC budgets are starting points. Assess common-warming and
channel-change mixing, prior sensitivity, recovery and held-out calibration
before using results in a manuscript. Shared/coupled full-record fits remain
computationally demanding; successful tests do not certify scientific coverage.
The executed short recovery pilot still has poorly mixed risk and trend
contrasts. Its diagnostics are retained in `validation/`; publication requires
longer, diagnostically adequate runs and more independent replications.
