"""Manuscript-style panels for prior and held-out predictive sensitivity."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from ..plotting.style import PUBLICATION_COLORS, save_figure


def save_sensitivity_plots(tables, directory, *, dpi=180):
    import matplotlib.pyplot as plt
    variants = list(dict.fromkeys(v for table in tables.values() if 'variant' in table for v in table.variant))
    colors = {v: PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)] for i, v in enumerate(variants)}
    images = []

    def save(fig, channel, name):
        stem = re.sub(r'[^A-Za-z0-9_-]', '_', str(channel))+'_'+name
        images.extend(p.name for p in save_figure(fig, directory/stem, formats=('png',), dpi=dpi, close=True))

    if 'prior_posterior' in tables:
        data = tables['prior_posterior'].query("scale == 'SD'")
        for channel, group in data.groupby('channel', sort=False):
            components = list(group.component.drop_duplicates())
            fig, axes = plt.subplots(1,len(components),figsize=(4.4*len(components),3.7),squeeze=False,layout='constrained')
            for ax, component in zip(axes[0], components):
                for i, variant in enumerate(variants):
                    rows = group[(group.component == component)&(group.variant == variant)]
                    for distribution, offset, marker, alpha in [('prior',-.13,'s',.40),('posterior',.13,'o',1.)]:
                        row = rows[rows.distribution == distribution]
                        if row.empty:
                            continue
                        row = row.iloc[0]
                        ax.errorbar(row['median'], i+offset,
                            xerr=[[row['median']-row['lower']],[row['upper']-row['median']]],
                            fmt=marker, color=colors[variant], alpha=alpha,
                            label=distribution if i == 0 else None)
                ax.set(yticks=range(len(variants)),yticklabels=variants,xlabel=f'{component} innovation SD')
                ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useMathText=True)
            axes[0,0].legend(loc='best')
            save(fig,channel,'prior_posterior')
    if 'paths' in tables:
        labels = {'level': 'level / °C', 'slope': 'latent slope / °C per decade', 'risk': 'event probability'}
        for (channel, quantity), group in tables['paths'].groupby(['channel','quantity'],sort=False):
            fig, ax = plt.subplots(figsize=(10,3.8),layout='constrained')
            for variant, data in group.groupby('variant',sort=False):
                data = data.sort_values('time')
                dates = pd.to_datetime(data.time)
                ax.plot(dates,data['median'],color=colors[variant],label=variant)
                ax.fill_between(dates,data.lower,data.upper,color=colors[variant],alpha=.10)
            ax.set(xlabel='time',ylabel=labels[quantity])
            ax.legend(loc='best',ncol=min(3,len(variants)))
            save(fig,channel,quantity)
    if 'scientific_targets' in tables:
        targets = tables['scientific_targets']
        for channel, group in targets.groupby('channel',sort=False):
            selected = [q for q in group.quantity.drop_duplicates() if q.endswith(('.level.change','.slope.change'))]
            if not selected:
                selected = [q for q in group.quantity.drop_duplicates() if q.endswith(('_level_change','_end_slope_C_per_decade'))]
            if not selected:
                continue
            fig, axes = plt.subplots(1,len(selected),figsize=(5*len(selected),3.5),squeeze=False,layout='constrained')
            for ax, quantity in zip(axes[0],selected):
                for i, variant in enumerate(variants):
                    rows = group[(group.quantity==quantity)&(group.variant==variant)]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    ax.errorbar(row['median'],i,xerr=[[row['median']-row['lower']],[row['upper']-row['median']]],fmt='o',color=colors[variant])
                ax.axvline(0,color='.5',lw=.8)
                label = ('period slope difference / °C per decade' if quantity.endswith('.slope.change') else
                         'period level difference / °C' if quantity.endswith('.level.change') else quantity.replace('_',' '))
                ax.set(yticks=range(len(variants)),yticklabels=variants,xlabel=label)
            save(fig,channel,'scientific_targets')
    if 'scores_by_origin' in tables:
        scores = tables['scores_by_origin'].query("horizon_band == 'all'")
        for channel, group in scores.groupby('channel',sort=False):
            fig, axes = plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
            for ax, metric in zip(axes,('crps','log')):
                for variant, data in group[group.score==metric].groupby('variant',sort=False):
                    ax.plot(data.origin,data['mean'],marker='o',color=colors[variant],label=variant)
                ax.set(xlabel='forecast origin / training months',ylabel='CRPS' if metric=='crps' else 'negative log predictive density')
            axes[0].legend(loc='best')
            save(fig,channel,'forecast_scores')
    if 'coverage_by_month' in tables:
        coverage = tables['coverage_by_month']
        for channel, group in coverage[(coverage.kind=='central_interval') & np.isclose(coverage.nominal,.95)].groupby('channel',sort=False):
            fig, ax = plt.subplots(figsize=(9,3.5),layout='constrained')
            for variant, data in group.groupby('variant',sort=False):
                ax.plot(data.month,data.empirical,marker='o',color=colors[variant],label=variant)
            ax.axhline(.95,color='.4',ls='--',lw=1)
            ax.set(xlabel='calendar month',ylabel='95% predictive coverage',xticks=range(1,13),ylim=(0,1.03))
            ax.legend(loc='best')
            save(fig,channel,'forecast_coverage')
    if 'pit' in tables:
        for channel, group in tables['pit'].groupby('channel',sort=False):
            fig, ax = plt.subplots(figsize=(5,4),layout='constrained')
            for variant, data in group.groupby('variant',sort=False):
                ax.step(np.sort(data.pit),np.arange(1,len(data)+1)/len(data),where='post',color=colors[variant],label=variant)
            ax.plot([0,1],[0,1],color='.4',ls='--',lw=1)
            ax.set(xlabel='held-out PIT',ylabel='empirical cumulative probability',xlim=(0,1),ylim=(0,1))
            ax.legend(loc='best')
            save(fig,channel,'forecast_pit')
    if 'predictions' in tables:
        for channel, group in tables['predictions'].groupby('channel',sort=False):
            origins = group.origin.drop_duplicates().tolist()
            fig, axes = plt.subplots(len(origins),1,figsize=(10,3*len(origins)),squeeze=False,layout='constrained')
            for ax, origin in zip(axes[:,0],origins):
                window = group[group.origin==origin]
                for variant, data in window.groupby('variant',sort=False):
                    dates = pd.to_datetime(data.time)
                    ax.plot(dates,data['median'],color=colors[variant],label=variant)
                    ax.fill_between(dates,data.lower,data.upper,color=colors[variant],alpha=.08)
                observed = window.drop_duplicates('time')
                ax.scatter(pd.to_datetime(observed.time),observed.observed,c='.25',s=9,label='observed')
                ax.set(xlabel='forecast time',ylabel='temperature / °C')
            axes[0,0].legend(loc='best',ncol=min(4,len(variants)+1))
            save(fig,channel,'forecasts')
    return images
