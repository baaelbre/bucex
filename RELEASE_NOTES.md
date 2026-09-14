# bucex 1.6.1

This release consolidates research workflows and extends residual dependence
while keeping the six independent analyses usable on their own.

- Every active research script is under `research/serra/`; redundant conference
  trees, old result evidence, duplicate examples and particle kernels are removed.
- One concise runner supports individually selected Uccle summaries, mixed
  FS hierarchies, shared warming/departures, and optional Gaussian copulas.
- Mixed hierarchies retain exact Laplace–MH. Joint copula likelihoods enter
  state and parameter updates, predictive simulation, and joint scoring.
- Copula state proposals use the full joint Hessian and correlated Gaussian
  pseudo-observations; an ineffective marginal-only proposal was replaced
  after a six-series Uccle pilot exposed poor movement.
- Prior sensitivity, shape recovery across -0.5 to 0.5, endpoint diagnostics,
  forecast uncertainty and held-out calibration have configured workflows.
- Original-scale predictive ordering checks report incompatibility without
  sorting or censoring draws. A copula does not enforce physical ordering.
- Shared state, conditional residual dependence and observation uncertainty
  remain distinct; each series keeps its own seasonal component.

Read `docs/VALIDATION.md` for executed checks and limitations, and
`docs/REVIEWER_MATRIX.md` for reviewer requirements. Available study scripts
are not a claim that full scientific experiments have already been completed.
