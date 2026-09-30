"""Standard-library-only definition of the joint half-normal calibration grid."""
import json
import math
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / 'config'
BATCHES = ('sweetspot', 'sweetspot_hpc', 'sweetspot_posterior',
           'sweetspot_validation', 'sweetspot_long')


def specification():
    return json.loads((CONFIG / 'sweetspot.json').read_text())


def cells():
    spec = specification()
    main = json.loads((CONFIG / 'main.json').read_text())
    base = main['priors']['innovation_sd']
    for key in ('level_scales', 'slope_scales', 'seasonal_scales'):
        values = spec[key]
        if (not values or values != sorted(set(values)) or
                any(not math.isfinite(v) or v <= 0 for v in values)):
            raise ValueError(key + ' must contain increasing positive finite scales.')
    if base['level'] not in spec['level_scales'] or base['trend'] not in spec['slope_scales'] or base['season'] not in spec['seasonal_scales']:
        raise ValueError('The study grid must contain the declared reference.')
    def token(value):
        return format(value, '.0e').replace('e-0', 'e').replace('e-', 'e')
    result = []
    for a in spec['level_scales']:
        for b in spec['slope_scales']:
            reference = a == base['level'] and b == base['trend']
            name = 'reference' if reference else 'ss_a' + token(a) + '_b' + token(b)
            result.append(dict(name=name, setting=name, scope='shared', family='calibration',
                sd_multipliers=dict(level=a/base['level'], trend=b/base['trend']),
                A_level=a, A_slope=b, A_season=base['season'], calibration_kind='grid'))
    for gamma in spec['seasonal_scales']:
        if gamma == base['season']: continue
        name = 'ss_gamma_' + token(gamma)
        result.append(dict(name=name, setting=name, scope='shared', family='calibration',
            sd_multipliers=dict(season=gamma/base['season']), A_level=base['level'],
            A_slope=base['trend'], A_season=gamma, calibration_kind='seasonal'))
    if len({c['name'] for c in result}) != len(result):
        raise ValueError('Scale labels are not unique; increase token precision.')
    return result


def private_cells():
    return [dict(c, name='independent_'+c['name'], scope='independent') for c in cells()]


def study_variants():
    return cells() + private_cells()


def calibration_rows():
    """Marginal prior RMS effects, integrating both hierarchy levels."""
    result = []
    for cell in cells():
        row = dict(variant=cell['name'], A_level=cell['A_level'],
                   A_slope=cell['A_slope'], A_season=cell['A_season'], calibration_kind=cell['calibration_kind'])
        for years in (10, 30):
            h = 4 * years
            a = h * row['A_level']**2
            b = h*(h-1)*(2*h-1)/6 * row['A_slope']**2
            row.update({f'level_innovation_SD_{years}y_C':math.sqrt(a),
                f'slope_innovation_SD_{years}y_C':math.sqrt(b),
                f'total_trend_innovation_SD_{years}y_C':math.sqrt(a+b),
                f'rate_change_SD_{years}y_C_per_decade':40*math.sqrt(h)*row['A_slope'],
                f'slope_fraction_of_prior_mean_variance_{years}y':b/(a+b)})
        result.append(row)
    return result
