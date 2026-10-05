"""Forecast summer 2019 from the fit through MAM 2019; no future-data leakage."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import load_config, CONFIG
from .run import run
from .figures import style, panels, savefig
from .data import load_summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG/'main.json')
    p.add_argument('--profile', choices=('smoke','screen','paper'), default='screen')
    p.add_argument('--output', type=Path, default=Path('results/pre2019'))
    p.add_argument('--workers', type=int)
    args = p.parse_args()
    c = load_config(args.config)
    if c['data']['frequency'] != 'seasonal':
        raise ValueError('This analysis is explicitly a seasonal forecast.')
    # Smoke reduces iterations but still fits only data available at MAM 2019.
    c['profiles'][args.profile]['max_blocks'] = None
    fit, root = run(c, profile=args.profile, output=args.output/args.profile,
                    end='2019-05-31', workers=args.workers, figures=False)
    pred = fit.predict(4, draws=c['profiles'][args.profile]['predictive_draws'], seed=c['seed']+31)
    data = load_summaries(c['data']['source'])
    heldout = data.loc[pred.index]
    bx.coverage(pred, heldout).to_csv(root/'coverage.csv', index=False)
    rows = []
    for threshold in (35., 36.6, 39.7):
        probs = pred.risk(threshold, channel='TXx')[:, 0]
        s = bx.summarize(probs, axis=0)
        rows.append(dict(threshold=threshold, **{k: float(v) for k,v in s.items()}))
    pd.DataFrame(rows).to_csv(root/'summer_2019_risk.csv', index=False)
    style()
    import matplotlib.pyplot as plt
    names = list(fit.channels)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for i, name in enumerate(names):
        summary = pred.summary(name)
        mean, lo, hi = [float(summary[key][0]) for key in ('mean','lower','upper')]
        color = '#24658a' if name.startswith('TX') else '#a44839'
        left.vlines(i, lo, hi, color=color)
        left.plot(i, mean, 'o', color=color)
        left.plot(i, heldout[name].iloc[0], 'D', color='black', ms=4)
    left.set_xticks(np.arange(len(names)), names)
    left.set_ylabel('Summer temperature (°C)')
    bx.plot(pred, channel='TXx', type='risk_curve', time=0, thresholds=np.linspace(30,43,180), ax=right)
    for row in rows:
        right.plot(row['threshold'], row['mean'], 'o', color='#24658a')
    right.set(xlabel='Summer TXx threshold (°C)', yscale='log')
    right.set_ylim(bottom=1e-7, top=1)
    savefig(fig, root/'figures', 'summer_2019', c)


if __name__ == '__main__':
    main()
