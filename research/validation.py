"""Fixed-origin validation: coverage, widths, signed errors and threshold counts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import CONFIG, load_config
from .data import load_summaries
from .run import run
from .figures import style, panels, savefig, subset_prediction


def _tables(forecast, heldout, origin, config, label):
    table = bx.coverage(forecast, heldout, levels=config['validation']['intervals'])
    table['origin'], table['window'] = origin, label
    table['first_block'], table['last_block'] = str(heldout.index[0].date()), str(heldout.index[-1].date())
    table['mean_error'] = table.channel.map({name: float((forecast.y[name].mean(0)-heldout[name]).mean()) for name in forecast.models})
    return table


def _latex(table, destination):
    compact = table.pivot(index=['origin','channel'], columns='interval', values='coverage')*100
    compact['width_95'] = table[table.interval == .95].set_index(['origin','channel']).mean_width
    columns = list(compact.columns)
    header = [f'{100*k:g}\\% coverage' if isinstance(k, float) else r'95\% width' for k in columns]
    lines = [r'\begin{tabular}{ll'+'r'*len(columns)+'}', r'\toprule',
             'Origin & Summary & '+' & '.join(header)+r' \\', r'\midrule']
    for (origin, name), row in compact.iterrows():
        cells = [f'{v:.1f}' if k != 'width_95' else f'{v:.2f}' for k,v in row.items()]
        lines.append(origin[:4]+' & '+name+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    destination.write_text('\n'.join(lines)+'\n')


def validate_origins(config, *, profile='screen', output=Path('results/validation'), recent=False, workers=None, origin=None):
    settings = config['profiles'][profile]
    data = load_summaries(config['data']['source'], frequency=config['data']['frequency'])[config['data']['series']]
    origins = [origin] if origin else config['validation']['recent_origins' if recent else 'origins']
    if profile == 'smoke':
        origins = [str(data.index[settings['max_blocks']-1].date())]
    root = Path(output)/profile
    root.mkdir(parents=True, exist_ok=True)
    rows, decades, counts, recent_paths = [], [], [], []
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
        evaluated = subset_prediction(forecast, np.arange(horizon) < len(heldout))
        table = _tables(evaluated, heldout, origin, config, 'full')
        table['requested_blocks'], table['available_blocks'] = horizon, len(heldout)
        table.to_csv(target/'coverage.csv', index=False)
        rows.append(table)
        per_decade = 10*config['model']['steps_per_year']
        for start in range(0, len(heldout), per_decade):
            mask = (np.arange(len(heldout)) >= start)&(np.arange(len(heldout)) < start+per_decade)
            decade = _tables(subset_prediction(evaluated, mask), heldout.iloc[np.flatnonzero(mask)], origin, config, f'decade_{start//per_decade+1}')
            decade['lead_decade'] = start//per_decade+1
            decades.append(decade)
        fig, axes = panels(list(fit.channels))
        for name, ax in zip(fit.channels, axes):
            bx.plot(evaluated, channel=name, type='forecast', history=fit[name], observed=heldout[name].to_numpy(), ax=ax)
            spec = config['risk_thresholds'][name]
            mask = evaluated.index.month == spec['month']
            if not mask.any():
                continue
            conditional = evaluated.risk(spec['threshold'], channel=name, tail=spec['tail'])[:, mask]
            simulated = evaluated.y[name][:, mask]
            observed = heldout[name].to_numpy()[mask]
            upper = spec['tail'] == 'upper'
            obs_count = int(np.sum(observed > spec['threshold'] if upper else observed < spec['threshold']))
            draw_counts = np.sum(simulated > spec['threshold'] if upper else simulated < spec['threshold'], axis=1)
            counts.append(dict(origin=origin, channel=name, threshold=spec['threshold'], tail=spec['tail'],
                seasons=int(mask.sum()), observed=obs_count, expected=float(conditional.sum(1).mean()),
                probability_at_least_observed=float(np.mean(draw_counts >= obs_count)),
                probability_at_most_observed=float(np.mean(draw_counts <= obs_count))))
        savefig(fig, target/'figures', 'forecast', config)
        if recent:
            # Keep only two seasons/channels for the joint display, avoiding
            # retaining all large posterior predictive arrays across origins.
            entries = []
            for name, month in [('TXx',8), ('TNn',2)]:
                chosen = subset_prediction(evaluated, evaluated.index.month == month)
                entries.append((name, chosen, heldout.loc[chosen.index,name].to_numpy()))
            recent_paths.append((origin, entries))
    combined = pd.concat(rows, ignore_index=True)
    decade_table = pd.concat(decades, ignore_index=True)
    suffix = '_recent' if recent else ''
    combined.to_csv(root/f'coverage{suffix}.csv', index=False)
    decade_table.to_csv(root/f'coverage_by_decade{suffix}.csv', index=False)
    pd.DataFrame(counts).to_csv(root/f'threshold_counts{suffix}.csv', index=False)
    _latex(combined, root/f'coverage_table{suffix}.tex')
    import matplotlib.pyplot as plt
    for variable, ylabel in [('coverage','95% coverage'), ('mean_width','95% interval width (°C)')]:
        fig, axes = panels(list(config['data']['series']))
        for name, ax in zip(config['data']['series'], axes):
            for source, group in decade_table[(decade_table.channel == name)&(decade_table.interval == .95)].groupby('origin'):
                ax.plot(group.lead_decade, group[variable], 'o-', lw=1, label=source[:4])
            if variable == 'coverage':
                ax.axhline(.95, ls='--', color='.4', lw=.7)
            ax.set(xlabel='Forecast decade', ylabel=ylabel)
        axes[0].legend(fontsize=8)
        savefig(fig, root/'figures', variable+suffix, config)
    if recent_paths:
        fig, axes = plt.subplots(2,len(recent_paths),figsize=(5*len(recent_paths),5.5),squeeze=False,constrained_layout=True)
        for col,(source, entries) in enumerate(recent_paths):
            for row,(name, chosen, observations) in enumerate(entries):
                bx.plot(chosen,channel=name,type='forecast',observed=observations,ax=axes[row,col])
                axes[row,col].axhline(35 if name=='TXx' else -10, color='.4',ls=':',lw=.8)
                axes[row,col].text(.02,.98,f'{name} · {source[:4]} origin',transform=axes[row,col].transAxes,va='top')
        savefig(fig,root/'figures','validation_recent_paths',config)
    return combined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG/'main.json')
    p.add_argument('--profile', choices=('smoke','screen','paper'), default='screen')
    p.add_argument('--output', type=Path, default=Path('results/validation'))
    p.add_argument('--workers', type=int)
    p.add_argument('--origin', help='One configured or custom block-end origin.')
    p.add_argument('--recent', action='store_true')
    args = p.parse_args()
    validate_origins(load_config(args.config), profile=args.profile, output=args.output, recent=args.recent, workers=args.workers, origin=args.origin)


if __name__ == '__main__':
    main()
