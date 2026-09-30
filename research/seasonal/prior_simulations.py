"""Joint prior simulations for the declared seasonal study; no data fitting.

The sampler uses the same prior builders as the fits. State recursions are
vectorized across replications, with the FS first-observation seasonal basis.
Observation values are never read. Calendar limits are taken from the config.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme

import bucex as bx
from research.monthly.models import joint_model
from research.monthly.experiment import configured_variant
from research.seasonal.job_plan import CONFIG, variant_entries
from research.seasonal.jobs import ROOT

CORE = ('reference', 'half_level', 'double_level', 'half_slope', 'double_slope',
        'slope_1e3', 'seasonal_5e2', 'seasonal_1e1', 'fixed_location_seasonality',
        'half_initial_slope', 'double_initial_slope', 'initialization_bridge', 'xi_narrow', 'xi_wide')
QUANTILES = (.005, .025, .5, .975, .995)
QCOLS = ('q005', 'q025', 'median', 'q975', 'q995')
COMPONENTS = ('level', 'slope', 'seasonal', 'initial_slope', 'total', 'rate')


def calendar(config):
    """Complete meteorological blocks followed by 30 unconditional years."""
    first = pd.Period(config['data']['start'], freq='M').start_time
    last = pd.Period(config['data']['end'], freq='M').end_time
    candidates = pd.date_range(first, last, freq='MS')
    historical = pd.DatetimeIndex([d for d in candidates if d.month in (3, 6, 9, 12)
        and d + pd.DateOffset(months=3) - pd.Timedelta(days=1) <= last])
    if historical.empty:
        raise ValueError('No complete meteorological seasons in the declared calendar.')
    horizon = int(config['prior_calibration']['horizon'])
    return pd.date_range(historical[0], periods=len(historical)+horizon, freq='3MS'), len(historical)


def simulate_paths(config, *, draws, seed, dates):
    """Draw all six channels jointly, preserving the shared-scale hierarchy.

    Arrays have shape (draw, time, channel), on the original temperature scale.
    Initial rates remain private. Returned innovation contributions start from
    zero before the first transition; the deterministic seasonal state starts
    at the first observation, exactly as in the fitted FS representation.
    Only this study's Gaussian/GEV, local-linear, dummy-seasonal model is accepted.
    """
    fixed=config.get('shrinkage_scope')=='fixed'
    if (config['analysis'] != 'joint' and not fixed) or config.get('copula') is not None:
        raise ValueError('Prior path checks require pooled HN or fixed Normal priors without a copula.')
    if config['data']['frequency'] != 'seasonal' or config['model']['period'] != 4:
        raise ValueError('This study simulator requires meteorological seasons with period four.')
    if any(config['model'].get(k, 'dynamic') not in ('dynamic', 'static') for k in ('level', 'trend', 'seasonal')):
        raise ValueError('Only dynamic or static structural components are supported.')
    if not isinstance(draws, int) or draws < 1 or len(dates) < 4:
        raise ValueError('Use positive integer draws and at least four dated seasons.')
    names = list(config['data']['series'])
    # Simulating the product of separate fixed priors in one array does not
    # pool them. Fitted fixed-prior jobs still contain exactly one response.
    simulation_config=deepcopy(config)
    if fixed:simulation_config['analysis']='joint'
    model, priors = joint_model(pd.DataFrame(columns=names), simulation_config)
    rng = np.random.default_rng(seed)
    sampled = bx.draw_marginal_prior(priors, draws, seed=int(rng.integers(2**32)))
    n, k = len(dates), len(names)
    shape = (draws, n, k)
    signs = np.array([model.channel(name).transform_sign for name in names])
    def parameter(key):
        return np.stack([sampled['channels'][name][key] for name in names], axis=-1)
    sd = {c: parameter('sd.'+c) for c in ('level', 'slope', 'seasonal')}
    for c, field in (('level', 'level'), ('slope', 'trend'), ('seasonal', 'seasonal')):
        if config['model'].get(field, 'dynamic') == 'static':
            sd[c] = np.zeros_like(sd[c])
    level = np.cumsum(rng.normal(size=shape), axis=1)*sd['level'][:, None, :]
    rate_noise = np.cumsum(rng.normal(size=shape), axis=1)*sd['slope'][:, None, :]
    slope = np.zeros(shape)
    slope[:, 1:, :] = np.cumsum(rate_noise[:, :-1, :], axis=1)
    shocks = rng.normal(size=shape)*sd['seasonal'][:, None, :]
    seasonal = np.zeros(shape)
    for t in range(n):
        seasonal[:, t, :] = shocks[:, t, :] - seasonal[:, max(0, t-3):t, :].sum(axis=1)
    g = parameter('initial.seasonal')  # draw x three lag coordinates x channel
    cycle = np.stack((g[:, 0, :], -g.sum(axis=1), g[:, 2, :], g[:, 1, :]), axis=1)
    baseline = cycle[:, np.arange(n) % 4, :]
    beta0 = parameter('initial.slope')
    initial_slope = beta0[:, None, :]*np.arange(1, n+1)[None, :, None]
    location = parameter('initial.level')[:, None, :] + baseline + initial_slope + level + slope + seasonal
    rate = 40*(beta0[:, None, :] + rate_noise)
    observations = np.empty(shape)
    tail_quantiles = np.empty(shape)
    scales = np.empty(shape)
    for j, name in enumerate(names):
        obs = model.channel(name).observation
        scale = parameter('sigma')[:, j, None]*np.ones((draws, n))
        if obs.scale is not None and config['model'].get('seasonal_scale', False):
            coords = rng.normal(0, obs.scale.prior_sd, (draws, 3))
            effects = coords @ obs.scale.contrast().T
            scale *= np.exp(effects[:, obs.scale.phases(n, dates)])
        scales[:, :, j] = scale
        if model.channel(name).family == 'gaussian':
            observations[:, :, j] = location[:, :, j] + scale*rng.normal(size=(draws, n))
            tail_quantiles[:, :, j] = location[:, :, j] + 2.3263478740408408*scale
        else:
            xi = sampled['channels'][name]['xi'][:, None]
            # An exponential variate gives -log(U), avoiding an exact U=0.
            e = rng.exponential(size=(draws, n))
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                z = -np.log(e)
                np.divide(np.expm1(-xi*np.log(e)), xi, out=z,
                          where=np.broadcast_to(xi != 0, z.shape))
                observations[:, :, j] = location[:, :, j] + scale*z
                tail_quantiles[:, :, j] = genextreme.ppf(.99, -xi, loc=location[:, :, j], scale=scale)
    result = {key: values*signs for key, values in dict(level=level, slope=slope,
        seasonal=seasonal, initial_slope=initial_slope, location=location,
        rate=rate, observations=observations, tail_quantile=tail_quantiles).items()}
    result['total'] = result['level'] + result['slope'] + result['seasonal'] + result['initial_slope']
    result.update(sampled=sampled, scales=scales, names=names)
    return result


def summarize(values):
    """Keep infinities in empirical quantiles; expose NaNs instead of filtering."""
    x = np.asarray(values)
    q = np.full(5, np.nan) if np.isnan(x).any() else np.quantile(x, QUANTILES, method='inverted_cdf')
    return dict(zip(QCOLS, map(float, q)), nonfinite_fraction=float(np.mean(~np.isfinite(x))))


def variant_config(name):
    entries = variant_entries()
    entry = next(v for v in entries if v['name'] == name)
    return configured_variant(bx.load_config(CONFIG/'main.json'), entry)


def run_variant(name, output, *, draws, seed=1951, batch_size=250, keep_paths=6):
    """Bound working memory while retaining unfiltered target draws for review."""
    config = variant_config(name)
    dates, n_history = calendar(config)
    n = len(dates)
    names = config['data']['series']
    horizons = (40, 80, 120)
    output = Path(output)
    identity = dict(version=bx.__version__, variant=name, draws=draws, seed=seed,
        batch_size=batch_size, keep_paths=keep_paths, config=config,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    marker = output/'complete.json'
    if marker.exists():
        if bx.load_config(marker) != identity:
            raise ValueError(f'{output}: saved prior simulation differs; use a new root or tier.')
        return output
    output.mkdir(parents=True, exist_ok=True)
    bx.save_config(identity, output/'resolved_config.json')
    targets, checks, initial, example = [], [], [], []
    for start in range(0, draws, batch_size):
        m = min(batch_size, draws-start)
        r = simulate_paths(config, draws=m, seed=np.random.SeedSequence([seed, start]), dates=dates)
        for j, channel in enumerate(names):
            frame = pd.DataFrame({'draw': np.arange(start, start+m), 'channel': channel})
            for h in horizons:
                for c in COMPONENTS:
                    frame[f'{c}_{h//4}y'] = r[c][:, h-1, j]
            frame['rate_at_record_end'] = r['rate'][:, n_history-1, j]
            frame['future_change_30y'] = r['location'][:, -1, j]-r['location'][:, n_history-1, j]
            frame['initial_cycle_range'] = np.ptp(r['location'][:, :4, j], axis=1)
            frame['tail_quantile_first_block'] = r['tail_quantile'][:, 0, j]
            frame['tail_quantile_last_block'] = r['tail_quantile'][:, -1, j]
            y = r['observations'][:, :, j]
            frame['path_min'] = y.min(axis=1)
            frame['path_max'] = y.max(axis=1)
            frame['fraction_outside_audit_range'] = np.mean((y < -50)|(y > 60), axis=1)
            frame['any_outside_audit_range'] = ((y < -50)|(y > 60)).any(axis=1)
            frame['nonfinite_fraction'] = np.mean(~np.isfinite(y), axis=1)
            targets.append(frame)
            for phase in range(4):
                initial.append(pd.DataFrame(dict(draw=np.arange(start, start+m), channel=channel,
                    season=('DJF', 'MAM', 'JJA', 'SON')[(dates[phase].month % 12)//3],
                    location=r['location'][:, phase, j], observation=y[:, phase])))
            for b in range(min(m, max(0, keep_paths-start))):
                example.append(pd.DataFrame(dict(draw=start+b, channel=channel, date=dates,
                    location=r['location'][b, :, j], observation=y[b], rate=r['rate'][b, :, j])))
        for left, right in (('TXn', 'TXm'), ('TXm', 'TXx'), ('TNn', 'TNm'),
                            ('TNm', 'TNx'), ('TNm', 'TXm'), ('TNx', 'TXx'), ('TNn', 'TXn')):
            if left in names and right in names:
                bad = r['observations'][:, :, names.index(left)] > r['observations'][:, :, names.index(right)]
                checks.append(dict(left=left, right=right, block_violations=int(bad.sum()),
                    blocks=bad.size, path_violations=int(bad.any(axis=1).sum()), paths=m))
        del r
    targets = pd.concat(targets, ignore_index=True)
    # These raw target draws are retained locally; compact exports use summaries.
    targets.to_csv(output/'target_draws.csv.gz', index=False)
    rows = []
    for channel, group in targets.groupby('channel', sort=False):
        for metric in targets.columns.drop(['draw', 'channel']):
            row = dict(variant=name, channel=channel, metric=metric, draws=draws,
                       **summarize(group[metric].to_numpy()))
            if metric.endswith('fraction') or metric in ('fraction_outside_audit_range', 'any_outside_audit_range'):
                row['probability_or_mean_fraction'] = float(group[metric].mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/'target_summary.csv', index=False)
    first = pd.concat(initial, ignore_index=True)
    rows = [dict(channel=channel, season=season, quantity=quantity, **summarize(group[quantity]))
        for (channel, season), group in first.groupby(['channel', 'season'], sort=False)
        for quantity in ('location', 'observation')]
    pd.DataFrame(rows).to_csv(output/'initial_cycle.csv', index=False)
    if checks:
        ordering = pd.DataFrame(checks).groupby(['left', 'right'], as_index=False).sum()
        ordering['block_violation_fraction'] = ordering.block_violations/ordering.blocks
        ordering['path_violation_fraction'] = ordering.path_violations/ordering.paths
        ordering.to_csv(output/'ordering_checks.csv', index=False)
    if example:
        pd.concat(example, ignore_index=True).to_csv(output/'example_paths.csv.gz', index=False)
    # The existing analytic routine independently verifies horizon definitions.
    from research.seasonal.prior_effects import run as effects
    effects(config, output/'analytic', draws=max(draws, 10000), seed=seed)
    bx.save_config(identity, marker)
    return output


def run_suite(root=ROOT, *, tier='screen', suite='core', draws=None, seed=1951, figures=True):
    if tier not in ('screen', 'paper'):
        raise ValueError('Use screen or paper.')
    draws = (2000 if tier == 'screen' else 10000) if draws is None else draws
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('draws must be a positive integer.')
    entries = variant_entries()
    from research.seasonal.sweetspot_plan import cells, fixed_cells
    names = [c['name'] for c in cells()+fixed_cells()] if suite == 'sweetspot' else list(CORE) if suite == 'core' else ([v['name'] for v in entries if v['scope'] == 'shared']
        if suite == 'all' else suite.split(','))
    if len(names) != len(set(names)) or any(n not in {v['name'] for v in entries} for n in names):
        raise ValueError('Unknown or repeated prior variant.')
    output = Path(root)/tier/'prior_simulations'
    output.mkdir(parents=True, exist_ok=True)
    # A separate lock prevents two prior-only/collection processes writing together.
    import fcntl
    with (output/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in names:
            print(f'PRIOR {tier}/{name}: {draws} joint replications; no MCMC', flush=True)
            run_variant(name, output/name, draws=draws, seed=seed)
        tables = [pd.read_csv(output/name/'target_summary.csv') for name in names]
        pd.concat(tables, ignore_index=True).to_csv(output/'comparison.csv', index=False)
        bx.save_config(dict(version=bx.__version__, tier=tier, draws=draws, seed=seed, variants=names,
            intervals=[.95, .99], observations_read=False,
            audit_range=[-50, 60], audit_range_is_a_constraint=False,
            interpretation='Unconditional prior simulation, not a posterior forecast or an adequacy certificate.'),
            output/'manifest.json')
        if figures:
            from research.seasonal.prior_simulation_figures import build
            build(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tier', choices=('screen', 'paper'), default='screen')
    parser.add_argument('--suite', default='core', help='core, sweetspot, all, or comma-separated variant names')
    parser.add_argument('--draws', type=int)
    parser.add_argument('--seed', type=int, default=1951)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print(f'Prior simulations: tier={args.tier}; suite={args.suite}; '
              f'draws={args.draws or (2000 if args.tier == "screen" else 10000)}; '
              f'output={args.root/args.tier/"prior_simulations"}; no MCMC or observed temperatures')
        return
    print(run_suite(args.root, tier=args.tier, suite=args.suite, draws=args.draws,
        seed=args.seed, figures=not args.no_figures))


if __name__ == '__main__':
    main()
