"""Compact forecast evidence and explicitly labelled validation windows."""
import numpy as np
import pandas as pd
import bucex as bx

SEASONS = {12: 'DJF', 3: 'MAM', 6: 'JJA', 9: 'SON'}


def annotate(frame):
    result = frame.copy()
    result['time'] = pd.to_datetime(result.time)
    result['season'] = result.time.dt.month.map(SEASONS)
    if result.season.isna().any():
        raise ValueError('Expected meteorological seasonal block start dates.')
    result['meteorological_year'] = result.time.dt.year + result.time.dt.month.eq(12)
    result['lead_year'] = ((result.horizon.astype(int) - 1) // 4 + 1)
    return result


def windows(frame, spec):
    """Yield masks with denominators for the requested, not just available, window."""
    horizon = frame.horizon.to_numpy(dtype=int)
    year = frame.meteorological_year.to_numpy(dtype=int)
    all_true = np.ones(len(frame), dtype=bool)
    definitions = [('available', 'available', all_true, int(horizon.max()))]
    definitions += [('lead_band', f'{lo:02d}-{hi:02d} years',
                     (horizon >= 4*(lo-1)+1) & (horizon <= 4*hi), 4*(hi-lo+1))
                    for lo, hi in spec['lead_bands_years']]
    definitions += [('cumulative', f'first {n:02d} years', horizon <= 4*n, 4*n)
                    for n in spec['cumulative_years']]
    definitions += [('calendar', f'{lo}-{hi}', (year >= lo) & (year <= hi), 4*(hi-lo+1))
                    for lo, hi in spec['calendar_windows']]
    for kind, label, mask, expected in definitions:
        if mask.any():
            yield kind, label, mask, expected


def write_forecast(forecast, observed, config, target):
    """Save diagnostics before large forecast draws are released from memory.

    Event-count draws use whole predictive paths. Summing mean conditional
    probabilities supplies expected counts with less observation-sampling noise.
    """
    spec = config['horizon_specification']
    origin = config['validation']['training_ends'][0]
    base = annotate(pd.DataFrame(dict(time=forecast.dates,
        horizon=np.arange(1, forecast.horizon+1))))
    paths, risks, counts, count_pmf = [], [], [], []
    for name in observed:
        kwargs = {'channel': name} if forecast.is_multiseries_forecast else {}
        index = forecast.channel_names.index(name) if kwargs else None
        observation = forecast.observations[:, :, index] if kwargs else forecast.observations
        location = forecast.eta[:, :, index] if kwargs else forecast.eta
        level = forecast.component_draws('level', **kwargs)
        q = forecast.predictive_quantiles([.005, .01, .025, .05, .95, .975, .99, .995], **kwargs)
        row = base.assign(origin_date=origin, channel=name, observed=observed[name].to_numpy(),
            predictive_mean=observation.mean(axis=0), location_mean=location.mean(axis=0),
            location_lower95=np.quantile(location, .025, axis=0),
            location_upper95=np.quantile(location, .975, axis=0),
            level_mean=level.mean(axis=0), level_lower95=np.quantile(level,.025,axis=0),
            level_upper95=np.quantile(level,.975,axis=0))
        for label, values in zip(('q005','q010','q025','q050','q950','q975','q990','q995'), q):
            row[label] = values
        paths.append(row)
        direction = '<' if bx.UCCLE_INFO[name]['tail'] == 'min' else '>'
        for threshold in spec['thresholds'][name]:
            p = forecast.probability_draws(threshold, direction=direction, **kwargs)
            event = (observed[name].to_numpy() < threshold if direction == '<'
                     else observed[name].to_numpy() > threshold)
            simulated = observation < threshold if direction == '<' else observation > threshold
            risk = base.assign(origin_date=origin, channel=name, threshold=threshold,
                direction=direction, observed=observed[name].to_numpy(), observed_event=event,
                probability_mean=p.mean(axis=0), probability_lower95=np.quantile(p,.025,axis=0),
                probability_upper95=np.quantile(p,.975,axis=0))
            risk['brier'] = (risk.probability_mean - event)**2
            risks.append(risk)
            for kind, label, mask, expected in windows(base, spec):
                for season in ('ALL', 'DJF', 'MAM', 'JJA', 'SON'):
                    selected = mask & (True if season == 'ALL' else base.season.eq(season).to_numpy())
                    n = int(selected.sum())
                    if not n:
                        continue
                    # For the available window, the expected count is the actual
                    # number of that season; otherwise use the full requested window.
                    wanted = n if kind == 'available' else expected if season == 'ALL' else expected // 4
                    samples = simulated[:, selected].sum(axis=1)
                    pmf = np.bincount(samples, minlength=n+1) / len(samples)
                    observed_count = int(event[selected].sum())
                    low, high = np.quantile(samples, [.025,.975], method='inverted_cdf')
                    meta = dict(origin_date=origin, channel=name, threshold=threshold,
                        direction=direction, window_kind=kind, window=label, season=season,
                        n_cases=n, n_requested=wanted, complete=n == wanted,
                        first_lead=int(base.horizon[selected].min()), last_lead=int(base.horizon[selected].max()))
                    counts.append(dict(**meta, observed_events=observed_count,
                        expected_events=float(p[:, selected].mean(axis=0).sum()),
                        count_lower95=int(low), count_upper95=int(high),
                        probability_at_least_observed=float(pmf[observed_count:].sum()),
                        count_simulation_draws=len(samples)))
                    # The full PMF is most useful for the non-overlapping lead bands
                    # and for the observed period, rather than duplicated windows.
                    if kind in ('available','lead_band'):
                        count_pmf.extend(dict(**meta, count=i, predictive_probability=float(v))
                                         for i, v in enumerate(pmf))
    pd.concat(paths,ignore_index=True).to_csv(target/'forecast_paths.csv',index=False)
    pd.concat(risks,ignore_index=True).to_csv(target/'risk_cases.csv',index=False)
    pd.DataFrame(counts).to_csv(target/'risk_count_windows.csv',index=False)
    pd.DataFrame(count_pmf).to_csv(target/'risk_count_pmf.csv',index=False)
