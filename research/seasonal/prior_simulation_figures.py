"""Publication-style prior checks with labelled axes and no plot titles."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bucex.plotting.style import publication_style, save_figure

FAMILIES = [('half_slope', 'HN slope 0.0001', '#7ba591'),
            ('reference', 'HN slope 0.0002', '#24658a'),
            ('double_slope', 'HN slope 0.0004', '#a44839'),
            ('slope_1e3', 'HN slope 0.001', '#78618c')]


def build(directory):
    directory = Path(directory)
    output = directory/'figures'
    table = pd.read_csv(directory/'comparison.csv')
    focused=table.variant.str.startswith('ss_').any()
    families=FAMILIES
    if focused:
        from research.seasonal.sweetspot_plan import cells
        reference_level=[c for c in cells() if c['A_level']==.01]
        colors=plt.cm.viridis(np.linspace(.08,.9,len(reference_level)))
        families=[(c['name'],f"Aβ={c['A_slope']:g}",color) for c,color in zip(reference_level,colors)]
    with publication_style(overrides={'font.size': 10, 'axes.labelsize': 10,
            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9}):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.1), constrained_layout=True)
        labels = ['Level innovation contribution (°C)', 'Slope innovation contribution (°C)',
                  'Same-season innovation change (°C)', 'Initial-rate contribution (°C)',
                  'Combined same-season change (°C)', 'Warming rate (°C / decade)']
        for ax, component, label in zip(axes.flat,
                ('level', 'slope', 'seasonal', 'initial_slope', 'total', 'rate'), labels):
            for variant, name, color in families:
                data = table[(table.variant == variant) & (table.channel == 'TXm')]
                if data.empty:
                    continue
                rows = data.set_index('metric').loc[[f'{component}_{h}y' for h in (10, 20, 30)]]
                x = np.array([10, 20, 30])
                ax.plot(x, rows.q975, color=color, label=name)
                ax.plot(x, rows.q025, color=color)
                ax.plot(x, rows['median'], color=color, ls=':', lw=.8)
            ax.axhline(0, color='.5', lw=.6)
            ax.set(xlabel='Years since initial state', ylabel=label, xticks=(10, 20, 30))
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, 1.), ncol=3)
        save_figure(fig, output/'prior_components_30y', formats=('png', 'pdf'), close=True)

        cases = [('half_level','Level × 0.5'),('double_level','Level × 2'),
                 ('half_slope','Slope 0.0001'),('reference','Slope 0.0002'),
                 ('double_slope','Slope 0.0004'),('slope_1e3','Slope 0.001'),
                 ('seasonal_5e2','Season 0.05'),('seasonal_1e1','Season 0.10')]
        if focused:
            cases=[(c['name'],f"Aα={c['A_level']:g}, Aβ={c['A_slope']:g}") for c in cells()]
        cases = [(v, lab) for v, lab in cases if v in set(table.variant)]
        if cases:
            fig, axes = plt.subplots(1, 3, figsize=(11.5, max(5.2,.39*len(cases))), sharey=True, constrained_layout=True)
            for ax, component, label in zip(axes, ('level', 'slope', 'seasonal'), labels[:3]):
                for i, (variant, lab) in enumerate(cases):
                    row = table[(table.variant == variant) & (table.channel == 'TXm') &
                                (table.metric == component+'_30y')].iloc[0]
                    color = '#a44839' if 't4' in variant else '#78618c' if 'cauchy' in variant else '#24658a'
                    ax.plot([row.q005, row.q995], [i, i], color=color, alpha=.35, lw=1)
                    ax.plot([row.q025, row.q975], [i, i], color=color, lw=3)
                    ax.scatter(row['median'], i, color=color, s=15)
                ax.axvline(0, color='.5', lw=.6)
                ax.set(xlabel=label, yticks=range(len(cases)), yticklabels=[lab for _, lab in cases])
                ax.invert_yaxis() if ax is axes[0] else None
            save_figure(fig, output/'prior_family_comparison', formats=('png', 'pdf'), close=True)

        reference = directory/'reference'
        if not (reference/'initial_cycle.csv').exists():
            return output
        cycle = pd.read_csv(reference/'initial_cycle.csv')
        paths = pd.read_csv(reference/'example_paths.csv.gz', parse_dates=['date'])
        names = ['TXm', 'TNm', 'TXx', 'TNx', 'TXn', 'TNn']
        fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharey=True, constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            color = '#24658a' if name.startswith('TX') else '#a44839'
            for quantity, offset, alpha in [('location', -.09, .35), ('observation', .09, 1.)]:
                data = cycle[(cycle.channel == name) & (cycle.quantity == quantity)].set_index('season').loc[['DJF', 'MAM', 'JJA', 'SON']]
                x = np.arange(4)+offset
                ax.vlines(x, data.q025, data.q975, color=color, alpha=alpha, lw=3,
                          label='Latent location' if quantity == 'location' else 'Replicated observation')
                ax.scatter(x, data['median'], color=color, alpha=alpha, s=13)
            ax.text(.03, .93, name, transform=ax.transAxes)
            ax.set(xticks=range(4), xticklabels=['DJF', 'MAM', 'JJA', 'SON'], ylabel='Temperature (°C)')
        axes[0, 0].legend(loc='lower left', fontsize=8)
        save_figure(fig, output/'prior_initial_cycle', formats=('png', 'pdf'), close=True)

        fig, axes = plt.subplots(3, 2, figsize=(10.2, 8), sharex=True, constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            color = '#24658a' if name.startswith('TX') else '#a44839'
            data = paths[(paths.channel == name) & (paths.draw < 3)]
            for index, group in data.groupby('draw'):
                ax.plot(group.date, group.observation, color=color, alpha=(.35, .6, 1.)[index], lw=.6)
            ax.text(.03, .92, name, transform=ax.transAxes)
            ax.set(ylabel='Simulated temperature (°C)')
        save_figure(fig, output/'prior_predictive_paths', formats=('png', 'pdf'), close=True)
    return output
