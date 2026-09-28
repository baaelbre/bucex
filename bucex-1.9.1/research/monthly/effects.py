"""Physical innovation contributions; no posterior selection or exact-zero claims."""
from pathlib import Path
import numpy as np
import pandas as pd


def write_innovation_effects(fit, directory, label, *, channel=None,
                             steps_per_year=12, years=(10, 30),
                             thresholds=(.05, .1, .2), level=.95):
    """Export the conditional SD from future innovations in response units.

    These effects exclude uncertainty in the current state and observation
    noise. Seasonal gains use the model's actual dummy-seasonal transition.
    Threshold probabilities concern a practical magnitude, not an inclusion
    probability. The legacy file retains the 10-year horizon.
    """
    years = tuple(years)
    thresholds = tuple(thresholds)
    if (not years or len(set(years)) != len(years) or
            not np.isfinite(steps_per_year) or steps_per_year <= 0 or
            any(not np.isfinite(y) or y <= 0 or int(y*steps_per_year) != y*steps_per_year for y in years)):
        raise ValueError('Effect horizons must be distinct positive whole numbers of model updates.')
    if (not thresholds or len(set(thresholds)) != len(thresholds) or
            any(not np.isfinite(t) or t <= 0 for t in thresholds)):
        raise ValueError('Effect thresholds must be distinct positive finite values.')
    summaries, probabilities = [], []
    for horizon_years in years:
        horizon = int(horizon_years*steps_per_year)
        effects = fit.innovation_effect_draws(horizon, channel=channel, combine_chains=False)
        table = fit.contrast_diagnostics(effects, credible_interval=level)
        table.index.name = 'component'
        table['horizon_years'] = horizon_years
        table['horizon_updates'] = horizon
        table['unit'] = 'degC'
        table['credible_interval'] = level
        table['probability_effect_sd_over_0.1'] = [float(np.mean(effects[c] > .1)) for c in table.index]
        summaries.append(table.reset_index())
        for component, values in effects.items():
            for threshold in thresholds:
                probabilities.append(dict(component=component, horizon_years=horizon_years,
                    horizon_updates=horizon, threshold=threshold, unit='degC',
                    probability_below=float(np.mean(values < threshold)),
                    probability_above=float(np.mean(values > threshold)),
                    n_draws=int(values.size), constant_effect=bool(np.ptp(values) == 0)))
        if horizon_years == 10:
            legacy = table.copy()
            legacy['horizon_months'] = 120  # Duration, not number of seasonal transitions.
            legacy.to_csv(Path(directory)/(label+'_innovation_effects.csv'))
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(Path(directory)/(label+'_innovation_effects_by_horizon.csv'), index=False)
    pd.DataFrame(probabilities).to_csv(Path(directory)/(label+'_innovation_effect_probabilities.csv'), index=False)
    return result
