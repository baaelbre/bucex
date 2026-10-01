"""Manuscript figures from native bucex objects; no automatic plot titles."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bucex as bx
from bucex.plotting import COLORS
from .configuration import load_config
from .data import phase_labels


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': 'white', 'legend.frameon': False})


def panels(names):
    fig, axes = plt.subplots(int(np.ceil(len(names)/2)), min(2, len(names)),
        figsize=(10, 2.5*int(np.ceil(len(names)/2))), squeeze=False, constrained_layout=True)
    for name, ax in zip(names, axes.flat):
        ax.text(.025, .94, name, transform=ax.transAxes, va='top', fontweight='bold')
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    return fig, list(axes.flat)[:len(names)]


def savefig(fig, folder, name, config):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder/(name+'.'+config['plot']['format'])
    fig.savefig(path, dpi=config['plot']['dpi'], bbox_inches='tight')
    plt.close(fig)
    return path


def loess(x, y, span=.18):
    """Local linear descriptive smoother; no observation-family interpretation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    count = min(len(x), max(3, int(np.ceil(span*len(x)))))
    result = np.empty(len(x))
    for i, value in enumerate(x):
        distance = np.abs(x-value)
        bandwidth = np.partition(distance, count-1)[count-1]
        weights = np.maximum(0, 1-(distance/max(bandwidth, 1e-12))**3)**3
        X = np.column_stack((np.ones(len(x)), x-value))
        beta = np.linalg.lstsq(X*np.sqrt(weights[:, None]), y*np.sqrt(weights), rcond=None)[0]
        result[i] = beta[0]
    return result


def exploratory(fit, config, folder):
    names = list(fit.channels)
    fig, axes = panels(names)
    for name, ax in zip(names, axes):
        c = fit[name]
        labels = phase_labels(c.index, config['data']['frequency'])
        palette = plt.get_cmap('viridis')(np.linspace(.1, .85, c.model.period))
        for phase in range(c.model.period):
            ids = np.arange(len(c.y))[np.arange(len(c.y)) % c.model.period == phase]
            ax.plot(c.index[ids], c.y[ids], '.-', color=palette[phase], lw=.6, ms=2.5, label=labels[phase])
        ax.set_ylabel('Temperature (°C)')
    axes[0].legend(ncol=min(4, fit[names[0]].model.period), fontsize=8)
    savefig(fig, folder, 'seasonal_records_six', config)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for i, (name, c) in enumerate(fit.channels.items()):
        phases = np.arange(len(c.y)) % c.model.period
        centered = c.y-np.array([c.y[phases == k].mean() for k in phases])
        x = np.arange(len(c.y))
        ax.plot(c.index, loess(x, centered), color=COLORS.get(name[:2]),
                ls=['-', '--', ':'][i//2], label=name)
    ax.set_ylabel('Centred temperature (°C)')
    ax.legend(ncol=2)
    savefig(fig, folder, 'seasonal_loess_overlay', config)


def subset_prediction(pred, mask):
    return replace(pred, index=pred.index[mask], **{key: {name: v[:, mask] for name, v in getattr(pred, key).items()}
        for key in ('y', 'location', 'level', 'slope', 'seasonal', 'sigma', 'xi')})


def generate(fit, config, output, settings):
    style()
    output, names = Path(output), list(fit.channels)
    folder = output/'figures'
    interval = config['plot']['interval']
    exploratory(fit, config, folder)
    for kind in ('level', 'slope', 'normal_qq', 'pit', 'acf', 'cycle'):
        if kind == 'cycle' and not any(c.model.season for c in fit.channels.values()):
            continue
        fig, axes = panels(names)
        for name, ax in zip(names, axes):
            bx.plot(fit, channel=name, type=kind, ax=ax, interval=interval,
                    phase_labels=phase_labels(fit[name].index, config['data']['frequency']))
        if kind == 'cycle':
            axes[0].legend(fontsize=8)
        savefig(fig, folder, kind, config)
    labels = phase_labels(next(iter(fit.channels.values())).index, config['data']['frequency'])
    fig, axes = panels(names)
    for name, ax in zip(names, axes):
        for phase, label in enumerate(labels):
            bx.plot(fit, channel=name, type='seasonal', phase=phase, ax=ax, interval=interval,
                    color=plt.get_cmap('viridis')(phase/max(1, len(labels)-1)*.8), label=label)
    axes[0].legend(ncol=4, fontsize=8)
    savefig(fig, folder, 'seasonal_evolution', config)
    for parameter in ('variance.level', 'variance.slope', 'variance.seasonal', 'initial_slope', 'sigma', 'xi'):
        selected = [n for n in names if parameter in fit[n].parameters]
        if not selected:
            continue
        fig, axes = panels(selected)
        for name, ax in zip(selected, axes):
            bx.plot(fit, channel=name, type='prior_posterior', parameter=parameter, ax=ax)
        axes[0].legend(fontsize=8)
        savefig(fig, folder, 'prior_posterior_'+parameter.replace('.', '_'), config)
    if fit.shared_scales:
        fig, axes = panels(list(fit.shared_scales))
        for key, ax in zip(fit.shared_scales, axes):
            bx.plot(fit, channel=names[0], type='prior_posterior', parameter='tau.'+key, ax=ax)
        axes[0].legend(fontsize=8)
        savefig(fig, folder, 'shared_scales', config)
    risks = []
    for kind in ('risk', 'return_level'):
        fig, axes = panels(names)
        for name, ax in zip(names, axes):
            c, spec = fit[name], config['risk_thresholds'][name]
            matching = np.flatnonzero(c.index[:c.model.period].month == spec['month'])
            phase = int(matching[0])
            bx.plot(fit, channel=name, type=kind, ax=ax, interval=interval,
                threshold=spec['threshold'], tail=spec['tail'], phase=phase, years=20)
            if kind == 'risk':
                samples = c.risk(spec['threshold'], tail=spec['tail'])
                ids = np.flatnonzero(c.index.month == spec['month'])
                s = bx.summarize(samples[:, :, ids], interval)
                for i, t in enumerate(ids):
                    risks.append(dict(channel=name, date=str(c.index[t].date()), threshold=spec['threshold'],
                        tail=spec['tail'], **{k: float(v[i]) for k, v in s.items()}))
        savefig(fig, folder, 'threshold_risk' if kind == 'risk' else 'return_levels_20_year', config)
    # Other return periods use the same native quantile plot, not custom math.
    for years in config['return_years']:
        if years == 20:
            continue
        fig, axes = panels(names)
        for name, ax in zip(names, axes):
            c, spec = fit[name], config['risk_thresholds'][name]
            phase = int(np.flatnonzero(c.index[:c.model.period].month == spec['month'])[0])
            bx.plot(fit, channel=name, type='return_level', years=years, tail=spec['tail'],
                    phase=phase, ax=ax, interval=interval)
        savefig(fig, folder, f'return_levels_{years}_year', config)
    fig, axes = panels(names)
    for name, ax in zip(names, axes):
        c = fit[name]
        values = np.stack([c.parameters.get(f'sigma[{i}]', c.parameters['sigma'])
                           for i in range(c.model.period)], axis=-1)
        summary = bx.summarize(values, interval)
        positions = np.arange(c.model.period)
        ax.vlines(positions, summary['lower'], summary['upper'], color=COLORS.get(name[:2]))
        ax.plot(positions, summary['mean'], 'o', color=COLORS.get(name[:2]))
        ax.set_xticks(positions, labels, rotation=45 if len(labels)>4 else 0)
        ax.set_ylabel('Observation scale (°C)')
    savefig(fig, folder, 'observation_scales', config)
    pd.DataFrame(risks).to_csv(output/'risk_probabilities.csv', index=False)
    pred = fit.predict(int(config['forecast_years']*config['model']['steps_per_year']),
                       draws=settings['predictive_draws'], seed=config['seed']+1)
    fig, axes = panels(names)
    for name, ax in zip(names, axes):
        bx.plot(pred, channel=name, type='forecast', history=fit[name], ax=ax, interval=interval)
    axes[0].legend(fontsize=8)
    savefig(fig, folder, 'forecasts_30_year', config)
    # Clear seasonal forecasts: each panel follows one tail-relevant season.
    fig, axes = panels(names)
    for name, ax in zip(names, axes):
        month = config['risk_thresholds'][name]['month']
        chosen = subset_prediction(pred, pred.index.month == month)
        c = fit[name]
        ids = c.index.month == month
        history = SimpleNamespace(index=c.index[ids], y=c.y[ids])
        bx.plot(chosen, channel=name, type='forecast', history=history, ax=ax, interval=interval)
    axes[0].legend(fontsize=8)
    savefig(fig, folder, 'forecasts_relevant_seasons', config)
    # Exports used for tables and figure-independent verification.
    rows = []
    for name in names:
        s = pred.summary(name)
        for i, date in enumerate(pred.index):
            rows.append(dict(channel=name, date=str(date.date()),
                location_mean=float(pred.location[name][:, i].mean()),
                **{k: float(v[i]) for k, v in s.items()}))
    pd.DataFrame(rows).to_csv(output/'forecasts.csv', index=False)
    print(f'Figures: {folder}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Regenerate figures from a saved fit; no refitting.')
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    record = json.loads((args.run/'run.json').read_text())
    generate(bx.load(args.run/'fit.bucex'), record['config'], args.run, record['settings'])


if __name__ == '__main__':
    main()
