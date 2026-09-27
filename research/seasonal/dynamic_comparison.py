"""Trace when six latent seasonal locations cease to share an additive shift."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import bucex as bx


def summarize(values, *, times, left, right, tolerance):
    samples = np.asarray(values).reshape(-1, len(times))
    lower, median, upper = np.quantile(samples, [.025, .5, .975], axis=0)
    return pd.DataFrame(dict(time=times, season=pd.DatetimeIndex(times).month.map(bx.SEASON_NAMES),
                             left=left, right=right, tolerance=tolerance,
                             lower=lower, median=median, upper=upper,
                             probability_positive=(samples > 0).mean(axis=0),
                             probability_abs_over_tolerance=(np.abs(samples) > tolerance).mean(axis=0)))


def run(directory, output, *, allow_unconverged=False):
    directory = Path(directory)
    config = bx.load_config(directory/'config.json')
    check = bx.load_config(directory/'convergence.json')
    if check.get('status') != 'passed_numerical_checks' and not allow_unconverged:
        raise RuntimeError('The source fit must pass its convergence gate.')
    fit = bx.load_fit(directory/'fit.bucex')
    times = pd.DatetimeIndex(fit.time)
    periods = config['contrasts']
    start, end = pd.Period(periods['reference'][0]), pd.Period(periods['reference'][1])
    period_index = times.to_period('M')
    reference = (period_index >= start) & (period_index <= end)
    if not reference.any() or any((reference & (times.month == month)).sum() == 0
                                  for month in (3, 6, 9, 12)):
        raise ValueError('The reference period must contain all four seasons.')
    names = tuple(fit.channel_names)
    location, rate = {}, {}
    for name in names:
        location[name] = fit.parameter_path('mu', channel=name, combine_chains=False)
        rate[name] = 40 * fit.component_draws('slope', channel=name, combine_chains=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    additive, rates = [], []
    midcentury = []
    early = (period_index >= pd.Period('1940-03')) & (period_index <= pd.Period('1950-02'))
    late = (period_index >= pd.Period('1970-03')) & (period_index <= pd.Period('1980-02'))
    if early.sum() != 40 or late.sum() != 40:
        raise ValueError('The two mid-century windows must each contain 40 seasons.')
    for left, right in periods['pairs']:
        if left not in location or right not in location:
            raise ValueError(f'Unknown pair: {left}, {right}')
        raw_gap = location[left] - location[right]
        baseline = np.zeros_like(raw_gap)
        for month in (3, 6, 9, 12):
            current = times.month == month
            baseline[..., current] = raw_gap[..., reference & current].mean(axis=-1)[..., None]
        additive.append(summarize(raw_gap-baseline, times=times, left=left, right=right,
                                  tolerance=.2))
        rates.append(summarize(rate[left]-rate[right], times=times, left=left, right=right,
                               tolerance=.1))
        for season, month in [('all', None), ('DJF', 12), ('MAM', 3),
                              ('JJA', 6), ('SON', 9)]:
            group = np.ones(len(times), dtype=bool) if month is None else times.month == month
            change = raw_gap[..., late & group].mean(axis=-1) - raw_gap[..., early & group].mean(axis=-1)
            lower, median, upper = np.quantile(change, [.025, .5, .975])
            midcentury.append(dict(left=left, right=right, season=season,
                early='1940-03/1950-02', late='1970-03/1980-02',
                n_early=int(np.sum(early & group)), n_late=int(np.sum(late & group)),
                lower=lower, median=median, upper=upper,
                probability_positive=float(np.mean(change > 0)),
                probability_abs_over_0p2_C=float(np.mean(np.abs(change) > .2)),
                credible_interval=.95))
    pd.concat(additive, ignore_index=True).to_csv(output/'additive_gap_changes.csv', index=False)
    pd.concat(rates, ignore_index=True).to_csv(output/'rate_differences.csv', index=False)
    pd.DataFrame(midcentury).to_csv(output/'midcentury_gap_contrasts.csv', index=False)
    bx.save_config(dict(source=str(directory), source_status=check.get('status'),
                        reference=periods['reference'], pairs=periods['pairs'],
                        gap_tolerance_C=.2, rate_tolerance_C_per_decade=.1,
                        midcentury_windows=[['1940-03','1950-02'],['1970-03','1980-02']],
                        intervals='pointwise 95% posterior credible intervals for paths; paired 95% posterior credible intervals for mid-century contrasts',
                        note='A changed gap suggests non-additive location evolution. The rate CSV contrasts slope states; it is not the exact derivative of the full season-specific location gap. Neither identifies a cause.'),
                   output/'definition.json')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/serra_190_dynamic_comparison'))
    parser.add_argument('--allow-unconverged', action='store_true', help='For figure/table development only.')
    args = parser.parse_args()
    print(run(args.run, args.output, allow_unconverged=args.allow_unconverged))
