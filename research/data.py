"""Build all six summaries from complete daily blocks, with no hidden imputation."""
from pathlib import Path
import numpy as np
import pandas as pd

SERIES = ('TXm', 'TNm', 'TXx', 'TXn', 'TNx', 'TNn')
ROOT = Path(__file__).resolve().parent.parent


def load_summaries(source='research/data/Uccle_31_08_26.csv', *, frequency='seasonal'):
    path = Path(source)
    if not path.is_absolute():
        path = ROOT/path
    daily = pd.read_csv(path, parse_dates=['DAY']).set_index('DAY').sort_index()
    if not daily.index.is_unique or not daily.index.equals(pd.date_range(daily.index.min(), daily.index.max())):
        raise ValueError('Daily observations must be unique and consecutive.')
    if not np.isfinite(daily[['TX', 'TN']].to_numpy()).all():
        raise ValueError('Missing daily temperatures require an explicit data decision; no automatic imputation.')
    if frequency not in {'seasonal', 'monthly'}:
        raise ValueError('frequency must be seasonal or monthly.')
    months = 3 if frequency == 'seasonal' else 1
    rule = 'QS-DEC' if months == 3 else 'MS'
    rows = []
    groups = list(daily.resample(rule))
    for number, (start, block) in enumerate(groups):
        end = start+pd.DateOffset(months=months)-pd.Timedelta(days=1)
        expected = pd.date_range(start, end)
        if not block.index.equals(expected):
            if number not in {0, len(groups)-1}:
                raise ValueError(f'Incomplete interior block beginning {start}.')
            continue  # Omit incomplete boundary blocks, including DJF 1892.
        rows.append(dict(date=end, TXm=block.TX.mean(), TNm=block.TN.mean(),
            TXx=block.TX.max(), TXn=block.TX.min(), TNx=block.TN.max(), TNn=block.TN.min()))
    return pd.DataFrame(rows).set_index('date').loc[:, list(SERIES)]


def phase_labels(index, frequency='seasonal'):
    if frequency == 'monthly':
        return [d.strftime('%b') for d in index[:12]]
    mapping = {2: 'DJF', 5: 'MAM', 8: 'JJA', 11: 'SON'}
    return [mapping[d.month] for d in index[:4]]
