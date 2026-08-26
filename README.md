# bucex 1.4.1

`bucex` fits Bayesian unobserved-component models to Gaussian and generalized
extreme-value observations. Version 1.4.1 provides an optional time-varying GEV
log scale while keeping the stationary model as the default. This patch makes
the phi JSONs self-documenting, exposes the location structure in those files,
and restores the complete established tables-and-figures report.

```python
import bucex as bx

bx.GEV()                  # stationary phi_t = log(sigma)
bx.GEV(phi="linear")     # linear trend in phi_t
bx.GEV(phi="rw")         # random walk in phi_t
bx.GEV(phi="ssvs")       # select stationary, linear, or RW
```

No new fitting entry point is needed. The model still goes through `bx.fit`,
returns a `FitResult`, and uses the same prediction and persistence APIs.

## Log-scale models

Write \(\phi_t=\log(\sigma_t)\), so \(\sigma_t=\exp(\phi_t)>0\). The four
choices are:

| `phi=` | Model | Interpretation |
|---|---|---|
| `"stationary"` | \(\phi_t=\phi_0\) | one scale for the full record; default |
| `"linear"` | \(\phi_t=\phi_0+\delta b_t\) | smooth log-scale trend |
| `"rw"` | \(\phi_t=\phi_{t-1}+u_t\) | locally changing log scale |
| `"ssvs"` | model indicator over the three rows above | scale-model uncertainty |

The centered basis \(b_t\) runs from \(-1/2\) to \(+1/2\). Consequently,
`exp(phi_slope)` is the fitted end/start scale ratio for the linear model.
For the random walk, \(u_t\sim N(0,q_\phi)\), and `phi_rw_sd` is
\(\sqrt{q_\phi}\).

Scale selection is separate from structural SSVS for the location process.
Using `bx.ssvs_gev_priors(...)` and `GEV(phi="ssvs")` therefore allows both:

- SSVS over fixed/dynamic level, zero/fixed/dynamic slope, and
  zero/fixed/dynamic seasonality;
- SSVS over stationary, linear, and random-walk log scale.

See `docs/LOG_SCALE.md` for the model equations, priors, exactness statement,
forecast behavior, and diagnostics.

## Direct API

```python
import numpy as np
import bucex as bx

y = bx.load_uccle_series("TXx", start="1892-01-01")

model = bx.Model(
    bx.GEV(phi="rw", xi_bounds=(-0.5, 0.5)),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ),
)

priors = bx.ssvs_gev_priors(
    period=12,
    alpha_mean=float(np.median(y)),
    sigma2_prior=bx.InverseGammaPrior(2.0, 2.0),
    xi_prior=bx.UniformPrior(-0.5, 0.5),
    innovation_slab_sd={"level": 0.01, "trend": 0.000025, "season": 0.01},
    phi_prior=bx.PhiPrior(
        linear=bx.NormalPrior(0.0, 0.35),
        rw_variance=bx.InverseGammaPrior(2.5, 3.75e-5),
        model_probabilities={"stationary": 0.50, "linear": 0.25, "rw": 0.25},
    ),
)

fit = bx.fit(
    y,
    model=model,
    priors=priors,
    engine="laplace_mh",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=56_100),
)

phi = fit.phi_draws()       # shape: draws x time
sigma = fit.sigma_draws()   # exp(phi), same shape
forecast = fit.forecast(120, draws=1_000, seed=56_101)
```

Change only `phi="rw"` to `"linear"`, `"stationary"`, or `"ssvs"`. For an
SSVS fit, inspect:

```python
fit.phi_model_probabilities()
# {'stationary': ..., 'linear': ..., 'rw': ...}
```

The stationary result also supports `phi_draws()` and `sigma_draws()`; its
scalar scale is expanded over time so downstream comparisons need no special
case. Posterior predictive and forecast results expose `parameters["phi"]`
and `parameters["sigma_path"]` for every GEV scale model.

## Inference

`engine="laplace_mh"` is the recommended exact-invariant default for the new
examples. The structural location path uses the existing iterated-Laplace
smoother as a full-path independence proposal with an exact GEV
Metropolis--Hastings correction.

The random-walk log-scale path uses its own conditional iterated-Laplace
smoother. With `laplace_mh` or `pgas`, its proposal is also corrected against
the exact GEV density. With `engine="laplace"`, both uses of the Laplace
approximation are intentionally approximate. Linear and stationary scale
parameters use the exact GEV likelihood in ordinary MH updates. The
`phi="ssvs"` implementation is a three-model product-space sampler with proper
prior pseudo-priors.

Time-varying `phi` currently requires the Fruehwirth--Schnatter
parameterization and a univariate `Model`. Fit TXx, TXn, TNx, and TNn as
separate jobs; dynamic scale is rejected for `MultiSeriesModel` rather than
silently approximated.

## JSON-first examples

The two new scripts each contain one readable `main()` and load all scientific
and computational settings directly from the selected JSON:

- `examples/10_simulation_phi.py` for scale-model sensitivity simulations;
- `examples/11_uccle_phi.py` for Uccle TXx, TXn, TNx, and TNn.

The first executable line to edit in either script is `DEFAULT_CONFIG_FILE`.
Command-line selection is usually more convenient:

```bash
python examples/10_simulation_phi.py --config examples/config/phi/simulation_stationary.json
python examples/10_simulation_phi.py --config examples/config/phi/simulation_linear.json
python examples/10_simulation_phi.py --config examples/config/phi/simulation_rw.json
python examples/10_simulation_phi.py --config examples/config/phi/simulation_ssvs.json

python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_stationary.json
```

There are 16 Uccle JSONs: four series times four scale models. They live under
`examples/config/phi/uccle/` and are ordinary, indented JSON files. Valid
`_comment` fields explain the settings in place. In simulation files,
`simulation.location` records the data-generating location truth and
`model.location` records the fitted structural-SSVS location model; this is
separate from `model.phi`. Change priors, scale hyperparameters, draws, warmup,
chains, seeds, Laplace controls, figures, and output paths there; neither the
Python nor PBS layer overwrites them. See
`examples/config/phi/README.md` for a field-by-field guide.

Examples 10 and 11 write the same core report as the earlier simulation and
Uccle fitting examples: parameter and MCMC diagnostics, location trajectories,
structural selection, posterior predictive checks, forecasts, latent-state
figures, process scales, and GEV summaries. The phi path and scale-model tables
and figures are additions, not replacements.

The production profile is 1,000 retained draws after 1,000 warmup iterations
for each of four independent chains. Use an edited copy with smaller values for
a pilot.

## PBS/HPC

Submit one JSON per job from the repository root:

```bash
qsub -v CONFIG=examples/config/phi/simulation_rw.json \
  job_scripts/submit_10_simulation_phi.pbs

qsub -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json \
  job_scripts/submit_11_uccle_phi.pbs
```

Submit every new simulation and Uccle configuration:

```bash
for config in examples/config/phi/simulation_*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_10_simulation_phi.pbs
done

for config in examples/config/phi/uccle/*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_11_uccle_phi.pbs
done
```

Each JSON requests four chains. The runner starts those chains as four
independent one-core processes and combines them only after all succeed. All
four series, all four scale models, and their chains may run concurrently;
actual concurrency is determined by PBS quotas. PBS controls walltime, cores,
memory, and logs only. See `docs/HPC.md` for setup, monitoring, collision-safe
run IDs, and the exact split/combine contract.

There is no separate `hpc/` directory because it would duplicate project
structure. Scheduler wrappers and the generic chain runner live together in
`job_scripts/`; interactive wrappers live in `bash_scripts/`; the same example
and JSON are used locally and on PBS.

## Installation and validation

```bash
python -m pip install ".[plot,test]"
python -c "import bucex; print(bucex.__version__)"
python -m pytest
python -m build
```

The printed version should be `1.4.1`. Safe result archives use schema 2.7.0
and remain backward-readable for every previously supported schema.

The broader package still includes Gaussian/GEV structural models, Laplace,
Laplace-MH and PGAS state inference, prediction, risk summaries, plotting,
Uccle loaders, hierarchical models, and the earlier numbered examples. See
`docs/ARCHITECTURE.md`, `docs/INFERENCE_MATRIX.md`, `docs/UCCLE.md`, and
`docs/VALIDATION.md`.
