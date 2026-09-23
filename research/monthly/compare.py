"""Compare matched validation runs, keeping forecast cases paired by calendar year."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def read(directory, filename):
    files = sorted(Path(directory).rglob(filename))
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else None


def compare(baseline, candidate, block_years=1, seed=82):
    if block_years < 1:
        raise ValueError('block_years must be positive.')
    rows = []
    definitions = [('scores.csv', ['origin','time','channel','score','setting'], 'value', ['channel','score','setting']),
                   ('joint_log_scores.csv', ['origin','time'], 'score', []),
                   ('compound_heat_scores.csv', ['origin','time'], 'brier', [])]
    for filename, keys, value, groups in definitions:
        left, right = read(baseline, filename), read(candidate, filename)
        if left is None or right is None:
            continue
        joined = left.merge(right, on=keys, suffixes=('_baseline', '_candidate'), how='outer',
                            indicator=True, validate='one_to_one')
        if not joined['_merge'].eq('both').all():
            raise ValueError(f'{filename}: runs do not have exactly the same forecast cases.')
        grouped = joined.groupby(groups, dropna=False) if groups else [((), joined)]
        for key, group in grouped:
            labels = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
            years = pd.to_datetime(group['time']).dt.year.to_numpy()
            result = bx.paired_block_comparison(group[value+'_baseline'], group[value+'_candidate'],
                         (years-years.min())//block_years, seed=seed)
            rows.append({'source': filename, **labels, **result, 'block_years': block_years})
    if not rows:
        raise ValueError('No matching score files found. Supply two completed validation directories.')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--block-years', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('score_comparison.csv'))
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.block_years).to_csv(args.output, index=False)
    print(args.output)
