"""Inspect joint priors over 10 and 30 years before fitting any observations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import CONFIG, load_config, build_model
from .figures import style, panels, savefig


def calibrate(config, *, profile='screen', output=Path('results/prior_calibration')):
    root = Path(output)/profile
    root.mkdir(parents=True, exist_ok=True)
    draws = config['profiles'][profile]['prior_draws']
    steps = config['model']['steps_per_year']
    simulation = bx.prior_predictive(build_model(config), 30*steps+1, draws=draws,
        seed=config['seed']+21, steps_per_year=steps)
    style()
    rows = []
    for component in ('level', 'slope', 'seasonal'):
        fig, axes = panels(list(simulation.models))
        for name, ax in zip(simulation.models, axes):
            bx.plot(simulation, channel=name, type=component, ax=ax)
        savefig(fig, root/'figures', 'prior_'+component, config)
    for name in simulation.models:
        p = simulation.parameters[name]
        for years in (10, 30):
            h = years*steps
            values = {
                'level_change': simulation.level[name][:, h-1]-p['initial_level'],
                'departure_from_initial_linear_trend': simulation.level[name][:, h-1]-p['initial_level']-h*p['initial_slope'],
                'rate_change_per_decade': (simulation.slope[name][:, h-1]-p['initial_slope'])*10*steps,
                'same_phase_seasonal_change': simulation.seasonal[name][:, h]-simulation.seasonal[name][:, 0],
            }
            for key, value in values.items():
                q = np.quantile(value, [.025,.5,.975])
                rows.append(dict(channel=name, years=years, quantity=key, mean=float(value.mean()),
                    lower_95=q[0], median=q[1], upper_95=q[2], draws=draws))
    pd.DataFrame(rows).to_csv(root/'prior_horizons.csv', index=False)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG/'main.json')
    p.add_argument('--profile', choices=('smoke','screen','paper'), default='screen')
    p.add_argument('--output', type=Path, default=Path('results/prior_calibration'))
    args = p.parse_args()
    print(calibrate(load_config(args.config), profile=args.profile, output=args.output))


if __name__ == '__main__':
    main()
