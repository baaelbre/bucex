"""Assess raw extreme ranks and runs clusters before considering an r-largest likelihood."""
import argparse
from pathlib import Path
import pandas as pd
import bucex as bx


def run(config,output,*,r=3,run_lengths=(1,3,5),quantile=.95,reference=('1961','1990')):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=config['data']
    daily,audit=bx.complete_seasons(settings.get('daily_source'),start=settings.get('start'),
        end=settings.get('end'),exclude_months=settings.get('exclude_months',()))
    calibration=daily.loc[reference[0]:reference[1]]
    if len(calibration)<365 or calibration.index.month.nunique()!=12:
        raise ValueError('Clustering threshold reference must contain all months and at least one year.')
    summary=[];counts=[];thresholds=[]
    blocks=pd.DatetimeIndex(audit.loc[audit.retained,'start'])
    for name in ('TXx','TXn','TNx','TNn'):
        tail='lower' if name.endswith('n') else 'upper'
        values=daily[name[:2]]
        ranks=bx.ranked_extremes(values,r=r,tail=tail)
        ranks.to_csv(output/f'{name}_ranks.csv',index=False)
        proximity=bx.rank_clustering_diagnostics(ranks)
        proximity['season']=pd.DatetimeIndex(proximity.block_start).month.map(bx.SEASON_NAMES)
        proximity.to_csv(output/f'{name}_rank_proximity.csv',index=False)
        for season,group in proximity.groupby('season'):
            summary.append(dict(channel=name,season=season,n=len(group),
                adjacent_fraction=group.any_consecutive_days.mean(),
                within_three_days_fraction=group.any_gap_at_most_three_days.mean(),
                all_within_week_fraction=group.span_days.le(7).mean(),
                same_month_fraction=group.same_calendar_month.mean(),boundary_tie_fraction=group.boundary_tie.mean()))
        q=quantile if tail=='upper' else 1-quantile
        cut=calibration[name[:2]].groupby(calibration.index.month).quantile(q)
        thresholds.extend(dict(channel=name,month=int(m),quantile=q,threshold=v) for m,v in cut.items())
        daily_threshold=pd.Series(values.index.month.map(cut).to_numpy(),index=values.index)
        for length in run_lengths:
            clusters=bx.extreme_clusters(values,daily_threshold,run_length=length,tail=tail)
            clusters.to_csv(output/f'{name}_clusters_run{length}.csv',index=False)
            n=clusters.groupby('block_start').size().reindex(blocks,fill_value=0)
            counts.append(pd.DataFrame(dict(channel=name,block_start=blocks,run_length=length,
                n_clusters=n.to_numpy(),fewer_than_r=n.to_numpy()<r)))
    pd.DataFrame(summary).to_csv(output/'rank_proximity_summary.csv',index=False)
    pd.concat(counts).to_csv(output/'cluster_counts.csv',index=False)
    pd.DataFrame(thresholds).to_csv(output/'thresholds.csv',index=False)
    bx.save_config(dict(r=r,run_lengths=list(run_lengths),threshold_reference=list(reference),
        upper_quantile=quantile,lower_quantile=1-quantile,
        interpretation='Diagnostic extraction only. Raw ranks remain dependent; no r-largest model is fitted.',
        caveats=['Fixed historical monthly thresholds do not remove warming or identify independent episodes.',
                 'Runs are separated by this many non-exceeding days; inspect sensitivity, not just adjacency.',
                 'Clusters are assigned once to the season of their peak. Window-edge clusters may be censored.',
                 'Rounded ties are flagged without jitter. Sparse cluster blocks must not be filled with non-extreme days.']),
        output/'clustering_notes.json')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='research/seasonal/config/main.json')
    parser.add_argument('--output',default='results/serra_187_clustering')
    parser.add_argument('--r',type=int,default=3)
    parser.add_argument('--run-lengths',type=int,nargs='+',default=[1,3,5])
    args=parser.parse_args()
    print(run(bx.load_config(args.config),args.output,r=args.r,run_lengths=args.run_lengths))
