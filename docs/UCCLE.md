# Uccle through the general API

The data layer supplies `load_uccle_series`, `load_uccle_multiseries`, integrity
checks, and thin convenience constructors. Every fitting helper ultimately
constructs public models and delegates to `bx.fit()`.

| Summary | Observation model | Interpretation |
|---|---|---|
| TXm, TNm | Gaussian | Monthly mean daily maximum/minimum temperature |
| TXx, TNx | GEV upper tail | Monthly maximum daily maximum/minimum |
| TXn, TNn | GEV lower tail | Monthly minimum daily maximum/minimum |

The bundled summaries are monthly, with period 12, covering 1892–2022. Earlier
manuscript analyses may have used seasonal blocks; that is a different dataset
construction and must be described explicitly. Priors on innovations depend on
the observation interval.

`research/serra/models.py` shows model construction using the same components
that apply to any environmental series. It does not use a bespoke Uccle sampler.
The runner has independent, hierarchical, shared, and residual-copula modes;
the paired validation runner uses held-out forecasting.

```bash
python -m research.serra.run --config research/serra/config/shared_smoke.json
python -m research.serra.validate --config research/serra/config/shared_smoke.json
```

Shared models use a common path with unit loadings, constrained additive
departures, and each channel's own seasonal cycle. The common trajectory is a
descriptive signal across Gaussian means and GEV locations. It is not causal
attribution. Residual dependence can be added with a Gaussian copula; this does not impose
physical ordering. Predictive violation checks are exported explicitly.

The research configurations explicitly set dates, priors and engines. Do the
same when comparing old convenience-helper fits, whose historical defaults may
start later than the full record. See `research/serra/README.md` for assumptions,
outputs and required scientific validation.
