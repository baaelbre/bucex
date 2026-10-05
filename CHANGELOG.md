# Changelog

## 1.0.0 — publication release

- Modular component, parameter, model, observation, result and plot packages.
- Public matrix-based `ComponentBlock` contract and generic non-centred compiler;
  fitting and forecasting no longer recognize only one level–slope–seasonal layout.
- Static/dynamic levels, slopes and seasonal components; known parameters through
  `Fixed`; static regression, time-varying regression and fixed-period cycles.
- Named distribution-parameter declarations and explicit sampler capability checks.
  Dynamic scale/shape declarations are rejected by the current inference backend.
- Gaussian and upper/lower GEV observations with constant or repeating dispersion.
- Private Normal innovation priors or shared half-normal innovation-prior scales.
  Initial rates and observation parameters remain response specific.
- FFBS and corrected Laplace–MH state updates; exact-target coefficient, observation
  and GIG shared-scale updates in separate modules.
- Named physical-state outputs, future-covariate validation, original-scale risk
  calculations and extensible native plotting through `bx.plot`.
- Schema-2 JSON/NPZ archives, explicit trusted extension registration, deterministic
  independent chain streams, process parallelism and compatible-chain combination.
- Separate public-API research workflows: five structural analyses, 23 pooled
  sensitivity settings, nine private sensitivity settings, calibration, 2019 refit,
  historical/recent validation, and an optional constant-linear-trend benchmark.
- Main and research READMEs, extension and inference notes, runnable examples,
  numerical-target tests and a Python 3.10–3.12 CI configuration.
- Removed development-only model extensions and machine-specific runtime settings.

This intentional version reset establishes the publication API. Preparing the
release files does not publish them to a package index. Earlier research and
draft archives retain their own readers; production scientific fits should be
rerun and checked before updating the paper's numerical results.
