"""Audit complete seasons and export six seasonal summaries and exploratory plots."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


SEASONS = ('DJF', 'MAM', 'JJA', 'SON')


def _local_linear(values, half_width=40):
    """Descriptive tricube local-linear smooth on equally spaced seasons."""
    values = np.asarray(values, dtype=float)
    x = np.arange(values.size, dtype=float)
    smooth = np.empty_like(values)
    for i in range(values.size):
        distance = np.abs(x - x[i]) / float(half_width)
        selected = np.isfinite(values) & (distance < 1)
        weights = (1 - distance[selected] ** 3) ** 3
        centered = x[selected] - x[i]
        design = np.column_stack((np.ones(selected.sum()), centered))
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * values[selected])
        smooth[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0][0]
    return smooth


def exploratory_structure(data, *, reference=('1892-03', '1922-02'),
                          comparison=('1996-09', '2026-08')):
    """Six seasonal records beside six whole-record-centred LOESS smooths."""
    import matplotlib.pyplot as plt

    phase = data.index.month.map(bx.SEASON_NAMES)
    windows = [('1892--1921', reference, '#707881', '--'),
               ('1996--2026', comparison, '#24658a', '-')]
    cycle_rows, smooth_rows = [], []
    figure = plt.figure(figsize=(15.2, 8.4), layout='constrained')
    grid = figure.add_gridspec(3, 4, width_ratios=(1, 1, 1.05, 1.05))
    records = [figure.add_subplot(grid[row, col]) for row in range(3) for col in range(2)]
    overlay = figure.add_subplot(grid[:, 2:])
    colors = dict(zip(SEASONS, ('#5b8fb0', '#d07a55', '#dba83b', '#7a9f72')))
    smooth_colors = dict(zip(data.columns,
        ('#24658a', '#a44839', '#14506e', '#873c30', '#4b83a3', '#c16b59')))
    for position, name in enumerate(data.columns):
        record_axis = records[position]
        for season in SEASONS:
            selected = phase == season
            record_axis.plot(data.index[selected], data.loc[selected, name],
                             color=colors[season], lw=.7, alpha=.85, label=season)
        record_axis.set_title(name, loc='left', weight='bold')
        if position % 2 == 0:
            record_axis.set_ylabel('temperature / °C')
        if position >= 4:
            record_axis.set_xlabel('year')
        record_axis.tick_params(axis='x', labelrotation=45)
        for label, limits, color, linestyle in windows:
            start, end = pd.Period(limits[0], freq='M'), pd.Period(limits[1], freq='M')
            selected = data.loc[start.start_time:end.end_time, name]
            selected_phase = selected.index.month.map(bx.SEASON_NAMES)
            summary = selected.groupby(selected_phase).agg(
                mean='mean', q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), n='size'
            ).reindex(SEASONS)
            if summary['n'].isna().any():
                raise ValueError(f'{name}: comparison windows do not contain all four seasons.')
            for season, values in summary.iterrows():
                cycle_rows.append(dict(series=name, period=label, season=season, **values.to_dict()))
        seasonal_means = data[name].groupby(phase).mean().to_dict()
        anomaly = data[name].to_numpy() - np.asarray([seasonal_means[s] for s in phase])
        smooth = _local_linear(anomaly, half_width=max(3, int(round(len(data)*.09))))
        overlay.plot(data.index, smooth, label=name, color=smooth_colors[name], lw=1.6,
                     ls='-' if name.startswith('TX') else '--')
        smooth_rows.extend(dict(time=time, series=name, anomaly=value, smooth=fit)
                           for time, value, fit in zip(data.index, anomaly, smooth))
    records[0].legend(ncol=2, fontsize=8, loc='upper left')
    overlay.axhline(0, color='.5', lw=.7)
    overlay.set(xlabel='year', ylabel='temperature anomaly / °C',
                title='Seasonally centred LOESS smoothers')
    overlay.legend(ncol=2, fontsize=9)
    overlay.tick_params(axis='x', labelrotation=45)
    return figure, pd.DataFrame(cycle_rows), pd.DataFrame(smooth_rows)


def prepare(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=bx.load_uccle_multiseries(**config['data'])
    data.to_csv(output/'seasonal_summaries.csv')
    bx.save_config(data.attrs,output/'quality_report.json')
    pd.DataFrame(data.attrs['block_audit']).to_csv(output/'block_audit.csv',index=False)
    phase=data.index.month.map(bx.SEASON_NAMES)
    cycle=data.groupby(phase).agg(['mean','std','min','max']).reindex(SEASONS)
    cycle.to_csv(output/'seasonal_cycle.csv')
    import matplotlib.pyplot as plt
    with bx.publication_style(style=config.get('figure_style','manuscript')):
        figure,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for name,ax in zip(data,axes.flat):
            for season in ('DJF','MAM','JJA','SON'):
                selected=phase==season
                ax.plot(data.index[selected],data.loc[selected,name],lw=.5,alpha=.7,label=season)
            ax.set(ylabel=name+' / °C',xlabel='season start')
        axes.flat[0].legend(fontsize=8,ncol=2)
        bx.save_figure(figure,output/'seasonal_records',formats=('png',),dpi=180,close=True)
        figure,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for name,ax in zip(data,axes.flat):
            values=[data.loc[phase==s,name] for s in ('DJF','MAM','JJA','SON')]
            ax.boxplot(values)
            ax.set(xticks=[1,2,3,4],xticklabels=['DJF','MAM','JJA','SON'],ylabel=name+' / °C')
        bx.save_figure(figure,output/'seasonal_distributions',formats=('png',),dpi=180,close=True)
        periods=config.get('contrasts',{})
        figure,cycle_table,smooth_table=exploratory_structure(data,
            reference=periods.get('reference',('1892-03','1922-02')),
            comparison=periods.get('comparison',('1996-09','2026-08')))
        cycle_table.to_csv(output/'exploratory_seasonal_cycles.csv',index=False)
        smooth_table.to_csv(output/'exploratory_seasonal_smooths.csv',index=False)
        bx.save_figure(figure,output/'exploratory_seasonal_blocks',formats=('png','pdf'),dpi=220,close=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='research/seasonal/config/main.json')
    parser.add_argument('--output',default='results/serra_190_seasonal_data')
    args=parser.parse_args()
    print(prepare(bx.load_config(args.config),args.output))
