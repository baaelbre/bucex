"""Simulate the manuscript's 30-year latent-change prior without fitting data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.models import joint_model, independent_model


def run(config, output, *, draws=50000, seed=188):
    if config['data'].get('frequency') != 'seasonal':
        raise ValueError('This calibration is for seasonal transitions.')
    # Prior declarations need channel names, never the observed temperatures.
    data = pd.DataFrame(columns=config['data']['series'])
    independent = config["analysis"] == "independent"
    _, priors = (independent_model(data[[data.columns[0]]], config) if independent else joint_model(data, config))
    spec = priors.shrinkage
    horizon = config['prior_calibration']['horizon']
    gain = bx.innovation_response_gains(horizon, period=config['model']['period'])
    gain['initial_slope'] = float(horizon)
    rng = np.random.default_rng(seed)
    shared = {} if spec is None else spec.sample_medians(draws, rng=rng)
    contributions = {}
    for name in data.columns:
        if independent:
            _, priors = independent_model(data[[name]], config)
            shared = {} if spec is None else spec.sample_medians(draws, rng=rng)
        contributions[name] = {}
        for component in ('level', 'slope', 'seasonal', 'initial_slope'):
            # A standardized path's endpoint contribution has variance gain^2.
            if component in shared:
                signed_coefficient = rng.normal(size=draws) * spec.coefficient_sd(component, shared[component])
            else:
                field={'initial_slope':'beta0','level':'s_level','slope':'s_trend','seasonal':'s_season'}[component]
                prior=getattr(priors.channels[name],field)
                signed_coefficient=rng.normal(prior.mean,prior.sd,size=draws)
            active=config['model'].get({'level':'level','slope':'trend','seasonal':'seasonal'}.get(component,''),'dynamic')=='dynamic'
            if component!='initial_slope' and not active:signed_coefficient=np.zeros(draws)
            # The initial rate itself is the coefficient: no second normal shock.
            shock=1. if component=='initial_slope' else rng.normal(size=draws)
            contributions[name][component] = signed_coefficient * shock * gain[component]
        contributions[name]['total'] = sum(contributions[name][component] for component in gain)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, components in contributions.items():
        for component, values in components.items():
            lower, median, upper = np.quantile(values, [.025, .5, .975])
            finite=spec is None or np.isfinite(spec.rms_multiplier) or component=='initial_slope'
            rows.append(dict(series=name, component=component, n=draws, mean=float(np.mean(values)) if finite else None,
                             sd=float(np.std(values, ddof=1)) if finite else None,
                             finite_second_moment=finite,lower_95=lower, median=median, upper_95=upper))
    pd.DataFrame(rows).to_csv(output/'thirty_year_effects.csv', index=False)
    if spec is None:
        from research.monthly.fixed_priors import calibration as fixed_calibration
        calibration=fixed_calibration(next(iter(priors.channels.values())),period=4,horizon=horizon,
            rate_multiplier=config['prior_calibration']['slope_time_unit']).to_dict('records')
    else:
        calibration=spec.calibration(period=4, **config['prior_calibration'])
    if spec is not None and 'initial_slope' not in shared:
        sd=config['priors']['initial_slope_sd']
        calibration.append(dict(component='initial_slope',anchor_kind='fixed_normal_SD',anchor=sd,
            displacement_sd_marginal=horizon*sd,
            initial_rate_sd_marginal=config['prior_calibration']['slope_time_unit']*sd))
    pd.DataFrame(calibration).to_csv(
        output/'analytic_calibration.csv', index=False)
    bx.save_config(dict(config=config, draws=draws, seed=seed, version=bx.__version__),
                   output/'config.json')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='research/seasonal/config/final.json')
    parser.add_argument('--output', default='results/serra_1981_prior_effects')
    parser.add_argument('--draws', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=188)
    args = parser.parse_args()
    print(run(bx.load_config(args.config), args.output, draws=args.draws, seed=args.seed))
