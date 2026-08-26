# Uccle temperature-extremes analysis

## Series and orientation

| JSON | Series | Fitted orientation | Description |
|---|---|---|---|
| `uccle/01_txx.json` | `TXx` | upper tail | monthly maximum daily maximum temperature |
| `uccle/02_txn.json` | `TXn` | sign-transformed lower tail | monthly minimum daily maximum temperature |
| `uccle/03_tnx.json` | `TNx` | upper tail | monthly maximum daily minimum temperature |
| `uccle/04_tnn.json` | `TNn` | sign-transformed lower tail | monthly minimum daily minimum temperature |

TXn and TNn are negated internally, fitted as maxima, and transformed back for
figures, tables, predictions, and risk summaries. Data begin on 1892-01-01 and
continue through the latest complete bundled observation. The time step is one
month and the seasonal period is 12.

## Model in the JSON

All four files specify the same location-varying GEV model:

\[
Y_t\mid\eta_t,\sigma,\xi\sim\operatorname{GEV}(\eta_t,\sigma,\xi),
\qquad \eta_t=\alpha_t+\gamma_t.
\]

The level and slope form a local-linear trend, and `gamma_t` is a 12-month
dummy seasonal component. Shape remains static. Scale is static in the
original `config/uccle/*.json` analyses and selectable in the new
`config/phi/uccle/*.json` sensitivity analyses. The fitted maximum structure
permits SSVS to distinguish:

- level: fixed or dynamic;
- slope: absent, fixed, or dynamic;
- seasonality: fixed or dynamic.

The parameterization is Fruehwirth-Schnatter non-centred and ASIS is disabled.
The initial level centre is the median of the internally transformed series.
These choices are explicit under `model`, `priors`, and `inference` in every
Uccle JSON.

## Log-scale sensitivity

Example 11 fits one of four models for
\(\phi_t=\log(\sigma_t)\): stationary, linear, random walk, or three-way
SSVS. There is one complete JSON for every series/model combination under
`examples/config/phi/uccle/`, for 16 files in total. For example:

```bash
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_stationary.json
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_linear.json
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_rw.json
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_ssvs.json
```

Replace `01_txx` with `02_txn`, `03_tnx`, or `04_tnn`. These files use the
narrow structural innovation slabs `(0.01, 0.000025, 0.01)`. Their scale
hyperparameters are grouped under `priors.phi` and can be edited directly.
The linear coefficient is the whole-record change in log scale; the RW prior
is on innovation variance; the three SSVS model probabilities are ordered
stationary, linear, RW. See `LOG_SCALE.md`.

## Calibrated primary prior

| Setting | Value |
|---|---:|
| initial level SD | 3.2 °C |
| fixed slope mean | 0.0 °C/month |
| fixed slope SD | 0.0025 °C/month = 0.30 °C/decade |
| initial seasonal SD | 2.25 °C |
| observation variance prior | inverse-gamma(2, 2) |
| shape prior | uniform(-0.5, 0.5) |
| level innovation slab | 0.02 °C/month |
| slope innovation slab | 0.00005 °C/month per monthly update |
| seasonal innovation slab | 0.02 °C/month |
| P(level dynamic) | 0.50 |
| P(slope absent, fixed, dynamic) | (0.20, 0.40, 0.40) |
| P(season absent, fixed, dynamic) | (0.00, 0.50, 0.50) |

For 30 years (`H=360`), the level slab implies a displacement SD of
`0.02 sqrt(360) = 0.379 °C`. The slope slab implies a level-displacement SD of

\[
0.00005\sqrt{360\times359\times719/6}=0.197\ ^\circ\mathrm C.
\]

The seasonal component is never absent because strong annual seasonality is
known to be present; the model distinguishes fixed from evolving seasonality.

## Inference and prediction defaults

Every primary Uccle JSON uses:

- 1,000 warmup iterations per chain;
- 1,000 retained draws per chain;
- four independent chains;
- base seed 56,000;
- one Laplace-MH trajectory proposal per MCMC iteration;
- 500 posterior-predictive draws;
- July as the focus month;
- 30 years of forecast history and a 10-year forecast horizon;
- PDF and PNG output at 180 dpi;
- diagnostic figures off by default.

Use example 05 for quick Laplace tuning. Once the specification is fixed, use
the identical JSON with example 09 for exact Laplace-MH inference. Increase to
2,000 warmup and 2,000 retained draws only when diagnostics or Monte Carlo
errors require it; longer chains do not repair poor mixing.

## Run locally

Each script visibly selects its JSON through `DEFAULT_CONFIG_FILE`, while
`--config` overrides that path:

```bash
python examples/05_uccle_laplace.py --config examples/config/uccle/01_txx.json
python examples/05_uccle_laplace.py --config examples/config/uccle/02_txn.json
python examples/05_uccle_laplace.py --config examples/config/uccle/03_tnx.json
python examples/05_uccle_laplace.py --config examples/config/uccle/04_tnn.json
```

For four concurrent local chain processes:

```bash
bash bash_scripts/run_05_uccle_laplace.sh examples/config/uccle/01_txx.json 4
```

For four parallel series jobs and their four parallel chains on PBS, see
`docs/HPC.md`.

## Minimum slab sensitivity

Ready-to-run narrower files are included alongside the primary files:

| Series | Primary | Narrower |
|---|---|---|
| TXx | `uccle/01_txx.json` | `uccle/01_txx_narrow.json` |
| TXn | `uccle/02_txn.json` | `uccle/02_txn_narrow.json` |
| TNx | `uccle/03_tnx.json` | `uccle/03_tnx_narrow.json` |
| TNn | `uccle/04_tnn.json` | `uccle/04_tnn_narrow.json` |

The narrow files change only `priors.innovation_slab_sd`:

| Setting | level | trend | season |
|---|---:|---:|---:|
| narrower | 0.01 | 0.000025 | 0.01 |
| primary | 0.02 | 0.000050 | 0.02 |
| wider | 0.04 | 0.000100 | 0.04 |

Their data, model probabilities, shape bounds, MCMC settings, and all other
fields are identical to the primary files. Compare posterior structural
probabilities, model-averaged trajectories, process-scale posteriors, and
scientifically important risk measures.

## What to report

- component and joint structural probabilities;
- structural switching across chains;
- model-averaged predictor, level, slope, and seasonal trajectories;
- prior-to-posterior process-scale distributions;
- observation scale, shape, and finite endpoints where applicable;
- log-scale trajectories, linear end/start ratios, RW innovation scale, and
  scale-model probabilities where applicable;
- R-hat, ESS, Monte Carlo errors, and Laplace-MH state acceptance;
- posterior predictive and forecast diagnostics;
- sensitivity to calibrated slab widths, scale evolution, and other
  defensible prior choices.

Risk summaries must retain their time index. A return level at time `t` is
conditional on the fitted parameters at that time, not a timeless property of
the full 1892-present record.
