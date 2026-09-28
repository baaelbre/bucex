# Direct SD calibration for the 1.9.4 comparison

All six summaries use the same reference anchors, per seasonal transition:

| Quantity | Reference value |
|---|---:|
| Level coefficient Normal SD anchor | 0.01 |
| Slope coefficient Normal SD anchor | 0.0001 |
| Seasonal coefficient Normal SD anchor | 0.01 |
| Separate initial-rate Normal SD | 0.01 |
| Separate initial-rate variance | 0.0001 |
| Initial level and seasonal coefficient SDs | 20 °C |

The initial rate is measured per seasonal update. Multiplication by 40 gives
°C/decade, so its reference prior SD is 0.4 °C/decade. It is the rate at the
start of the record, not the final rate.

In the fixed specification these three innovation SDs are fixed prior SDs.
In either mixture specification they are conditional SDs at the lognormal
hyperprior centre. They are not medians of the absolute coefficient. There is
no Phi^-1(0.75) conversion.

If log(tau/a) ~ N(0,w^2), then E(tau^2)=a^2 exp(2w^2). The marginal coefficient
RMS SD is therefore a exp(w^2), equal to 3.34327 a when w=log(3).

For H=120 seasonal transitions (30 years), the innovation response gains are
sqrt(H) for level, sqrt(H(H-1)(2H-1)/6) for slope, and sqrt(2H/4) for the dummy
seasonal process at the same phase. These refer to the innovation contribution
conditional on the present state; observation variability and current-state
uncertainty are additional.

| Component | Effect SD at the reference anchor / °C | RMS effect SD after mixing / °C |
|---|---:|---:|
| Level innovations | 0.109545 | 0.366237 |
| Slope innovations | 0.075420 | 0.252150 |
| Seasonal innovations | 0.077460 | 0.258968 |

These are scale summaries, not bounds or 95% intervals. The distribution is a
mixture, so multiplying its RMS SD by 1.96 does not produce a 95% interval.

Width checks use log(2) and log(4), keeping anchors fixed. They change both
near-zero mass and tail behavior. The paired independent/shared models have
identical marginal priors at every matched setting. The fixed Normal model is
an additional prior-family comparison, not a marginally matched pooling test.

Inspect `prior_calibration.csv`, channel prior/posterior reports, innovation
effects at 10 and 30 years, and their practical-threshold probabilities. Near
zero seasonal innovation means a stable seasonal pattern, not absent seasons.
