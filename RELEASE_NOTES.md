# bucex 1.2.0 release notes

Version 1.2.0 standardizes the scientific examples and adds seasonal views to
fitted trajectories and forecasts. It does not change the Laplace-MH target,
PGAS target, state equations, or archive schema.

## One explicit settings contract

Every scientific setting remains defined near the top of each example. There
is no hidden settings module.

- Examples 02, 03, 04, and 08 use the same six simulation truths, scenario
  keys, record length, seeds, GEV parameters, and structural innovation SDs.
- Examples 03, 04, and 08 use the same encompassing fitted model, SSVS prior,
  structural probabilities, MCMC defaults, predictive settings, and artifact
  names. Only the state engine and its tuning/diagnostics differ.
- Examples 05, 06, and 09 use the same Uccle data window, four series, monthly
  model, calibrated prior, fixed-versus-dynamic odds, MCMC defaults, forecast
  horizon, and figure/table names.
- Release tests import every script and compare these contracts directly, so a
  future edit cannot silently change only one method.

The canonical simulation calibration is:

```text
truth: RW level SD 0.02; LLT level SD 0.01; slope SD 0.00010;
       seasonal SD 0.05; fixed slope 0.006
prior: fixed-slope SD 0.006; innovation slabs 0.03 / 0.00015 / 0.05;
       neutral structural probabilities
MCMC:  1000 warmup + 1000 retained iterations per chain
```

The Uccle prior remains calibrated in monthly units: fixed-slope SD 0.0015,
innovation slabs 0.03 / 0.00010 / 0.05, initial-season SD 2.25, and equal prior
odds for fixed versus dynamic slope and season. The numbered Bash and PBS
runners now use the same 1000 + 1000 baseline as their Python scripts.

## Seasonal and seasonally adjusted views

Fitted univariate seasonal models now accept one-based phase selection:

```python
fit.plot("predictor", phase=7)
fit.plot("level", phase=7)
```

Forecasts retain the model period, phase sequence, and state names. The public
summary and plotting APIs support either one seasonal phase or the latent
level without seasonality and observation noise:

```python
forecast.summary(phase=7)
forecast.plot(phase=7, phase_label="July")

forecast.summary(target="level")
forecast.plot(target="level")
forecast.component_draws("level")
```

When a phase-specific plot receives contiguous history, the same phase filter
is applied to that history. Uccle examples convert `BUCEX_FOCUS_MONTH` to the
correct model phase even when the fitted data window starts in a month other
than January.

All simulation fit examples save `trajectory_phase_XX`, `forecast_phase_XX`,
and `forecast_level` figures/tables. All Uccle fit examples save July-specific
trajectory/forecast artifacts and `forecast_level`; set `BUCEX_FOCUS_MONTH`
to choose another calendar month.

## Naming and output cleanup

- `07_centered_ig_random_walk_gev.py` is now `07_centered_ig.py`.
- Its Bash runner, PBS submission file, result directory, logs, metadata, and
  documentation use the same concise stem.
- Scenario 02 now uses the same short output keys as examples 03, 04, and 08:
  `stationary`, `linear`, `random_walk`, `llt`, `dynamic_season`, and
  `llt_season`.
- Simulation fits continue to write `simulations/`, `fits/`, `tables/`, and
  `figures/`; Uccle fits continue to write `fits/`, `tables/`, and `figures/`.

## Inherited exact-inference guarantees

The release retains the 1.1 series guarantees: deterministic support-feasible
Laplace proposal initialization for finite-endpoint GEV likelihoods, ordinary
MH rejection of endpoint-invalid proposal draws, exact affine handling of
singular state transitions, fail-fast exact kernels, conditional PGAS ancestor
sampling, automatic conjugate centered inverse-gamma process-variance updates,
and the 24-task scenario-chain PBS workflow for example 08.

