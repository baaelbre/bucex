"""Simulate the manuscript's 30-year latent-change prior without fitting data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.models import joint_model


def run(config, output, *, draws=50000, seed=188):
    if config['data'].get('frequency') != 'seasonal':
        raise ValueError('This calibration is for seasonal transitions.')
    data = bx.load_uccle_multiseries(**config['data'])
    _, priors = joint_model(data, config)
    spec = priors.shrinkage
    horizon = config['prior_calibration']['horizon']
    gain = bx.innovation_response_gains(horizon, period=config['model']['period'])
    gain['initial_slope'] = float(horizon)
    rng = np.random.default_rng(seed)
    shared = spec.sample_medians(draws, rng=rng)
    contributions = {}
    for name in data.columns:
        contributions[name] = {}
        for component in ('level', 'slope', 'seasonal', 'initial_slope'):
            # A standardized path's endpoint contribution has variance gain^2.
            signed_coefficient = rng.normal(size=draws) * spec.coefficient_sd(component, shared[component])
            contributions[name][component] = signed_coefficient * rng.normal(size=draws) * gain[component]
        contributions[name]['total'] = sum(contributions[name][component] for component in gain)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, components in contributions.items():
        for component, values in components.items():
            lower, median, upper = np.quantile(values, [.025, .5, .975])
            rows.append(dict(series=name, component=component, n=draws, mean=float(np.mean(values)),
                             sd=float(np.std(values, ddof=1)), lower_95=lower, median=median, upper_95=upper))
    pd.DataFrame(rows).to_csv(output/'thirty_year_effects.csv', index=False)
    pd.DataFrame(spec.calibration(period=4, **config['prior_calibration'])).to_csv(
        output/'analytic_calibration.csv', index=False)
    bx.save_config(dict(config=config, draws=draws, seed=seed, version=bx.__version__),
                   output/'config.json')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='research/seasonal/config/final.json')
    parser.add_argument('--output', default='results/serra_188_prior_effects')
    parser.add_argument('--draws', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=188)
    args = parser.parse_args()
    print(run(bx.load_config(args.config), args.output, draws=args.draws, seed=args.seed))
