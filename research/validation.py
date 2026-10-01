"""Fixed-origin validation; the posterior only sees data available at its origin."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from bucex.diagnostics import predictive_pit
from .configuration import CONFIG, load_config
from .data import load_summaries
from .run import run
from .figures import style, panels, savefig


def validate_origins(config, *, profile='screen', output=Path('results/validation'), recent=False, workers=None):
    settings = config['profiles'][profile]
    data = load_summaries(config['data']['source'], frequency=config['data']['frequency'])[config['data']['series']]
    origins = config['validation']['recent_origins' if recent else 'origins']
    if profile == 'smoke':
        origins = [str(data.index[settings['max_blocks']-1].date())]
    root = Path(output)/profile
    root.mkdir(parents=True, exist_ok=True)
    rows, pitrows = [], []
    style()
    for origin in origins:
        target = root/origin
        fit, _ = run(config, profile=profile, output=target, end=origin, workers=workers, figures=False)
        horizon = int(config['validation']['years']*config['model']['steps_per_year'])
        if profile == 'smoke':
            horizon = config['model']['period']*2
        forecast = fit.predict(horizon, draws=settings['predictive_draws'], seed=config['seed']+11)
        heldout = data.loc[data.index > pd.Timestamp(origin)].iloc[:horizon]
        if heldout.empty:
            raise ValueError(f'No held-out observations for {origin}.')
        from .figures import subset_prediction
        evaluated = subset_prediction(forecast, np.arange(horizon) < len(heldout))
        table = bx.coverage(evaluated, heldout, levels=config['validation']['intervals'])
        table['origin'] = origin
        table['requested_blocks'] = horizon
        table['available_blocks'] = len(heldout)
        table.to_csv(target/'coverage.csv', index=False)
        rows.append(table)
        for kind in ('forecast', 'cdf'):
            fig, axes = panels(list(fit.channels))
            for name, ax in zip(fit.channels, axes):
                if kind == 'forecast':
                    bx.plot(evaluated, channel=name, type='forecast', history=fit[name],
                            observed=heldout[name].to_numpy(), ax=ax)
                else:
                    u = predictive_pit(evaluated, heldout[name], channel=name)
                    ordered = np.sort(u)
                    ax.step(ordered, np.arange(1, len(u)+1)/len(u), where='post', color='#24658a')
                    ax.plot([0,1],[0,1], '--', color='.4', lw=1)
                    ax.set(xlabel='Held-out predictive CDF', ylabel='Empirical cumulative fraction', xlim=(0,1), ylim=(0,1))
                    pitrows.extend(dict(origin=origin, channel=name, date=str(d.date()), predictive_cdf=float(p))
                                   for d, p in zip(heldout.index, u))
            savefig(fig, target/'figures', kind, config)
    combined = pd.concat(rows, ignore_index=True)
    suffix = '_recent' if recent else ''
    combined.to_csv(root/f'coverage{suffix}.csv', index=False)
    pd.DataFrame(pitrows).to_csv(root/f'predictive_cdf{suffix}.csv', index=False)
    compact = combined.pivot(index=['origin','channel'], columns='interval', values='coverage')
    compact['width_95'] = combined[combined.interval == .95].set_index(['origin','channel']).mean_width
    columns = list(compact.columns)
    lines = [r'\begin{tabular}{ll'+'r'*len(columns)+'}', r'\toprule',
             'Origin & Response & '+' & '.join(str(k).replace('_', r'\_') for k in columns)+r' \\', r'\midrule']
    for (origin, name), row in compact.iterrows():
        lines.append(origin+' & '+name+' & '+' & '.join(f'{v:.3f}' for v in row)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (root/f'coverage_table{suffix}.tex').write_text('\n'.join(lines)+'\n')
    return combined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG/'main.json')
    p.add_argument('--profile', choices=('smoke','screen','paper'), default='screen')
    p.add_argument('--output', type=Path, default=Path('results/validation'))
    p.add_argument('--workers', type=int)
    p.add_argument('--recent', action='store_true')
    args = p.parse_args()
    validate_origins(load_config(args.config), profile=args.profile, output=args.output, recent=args.recent, workers=args.workers)


if __name__ == '__main__':
    main()
