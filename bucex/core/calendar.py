"""Calendar labels shared by fitted and future seasonal summaries."""
from __future__ import annotations

import numpy as np


def meteorological_phases(dates):
    """DJF=1, MAM=2, JJA=3, SON=4 from actual dates."""
    import pandas as pd
    index = pd.DatetimeIndex(dates)
    if index.hasnans:
        raise ValueError("Calendar phases require finite dates.")
    return np.asarray((index.month % 12)//3 + 1, dtype=int)


def block_months(dates):
    """Recognize consecutive month-start or meteorological season-start dates.

    A lone date defaults to one month; callers handling one seasonal block
    should keep its explicit frequency instead of inferring it.
    """
    import pandas as pd
    index = pd.DatetimeIndex(dates)
    if not len(index) or index.hasnans or np.any(index.day != 1):
        raise ValueError("Expected finite monthly or seasonal block-start dates on the first day of a month.")
    delta = np.diff(index.to_period("M").asi8)
    if np.all(delta == 1):
        return 1
    if np.all(delta == 3) and np.all(np.isin(index.month, [3, 6, 9, 12])):
        return 3
    raise ValueError("Expected consecutive monthly or meteorological seasonal blocks.")


def seasonal_phases(period, n_time: int, dates=None, *, start_index: int = 0):
    """Calendar months for monthly annual cycles; relative phases otherwise.

    A March-start monthly record still labels July as phase 7. Arbitrary
    seasonal periods retain their one-based phase relative to the first
    fitted observation. The state equations themselves do not change.
    """
    if period is None:
        return None
    relative = (int(start_index) + np.arange(int(n_time))) % int(period) + 1
    if int(period) != 12 or dates is None:
        return relative
    values = np.asarray(dates)
    if values.size != int(n_time) or np.issubdtype(values.dtype, np.number):
        return relative
    import pandas as pd

    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError):
        return relative
    if index.hasnans:
        return relative
    months = index.to_period("M").asi8
    if len(months) > 1 and not np.all(np.diff(months) == 1):
        return relative
    return np.asarray(index.month, dtype=int)


def annual_groups(n_time: int, dates=None, *, period=None, include_partial=False):
    """Complete years; meteorological seasonal blocks use previous Dec--Nov."""
    count = int(n_time)
    if dates is None or np.issubdtype(np.asarray(dates).dtype, np.number):
        cycle = int(period or 1)
        groups = [np.arange(start, min(start + cycle, count)) for start in range(0, count, cycle)]
        labels = np.arange(1, len(groups) + 1)
        complete = [len(group) == cycle for group in groups]
    else:
        import calendar
        import pandas as pd

        index = pd.DatetimeIndex(dates)
        months = index.to_period("M").asi8
        meteorological = (len(months)>1 and np.all(np.diff(months)==3)
                         and np.all(np.isin(index.month,[3,6,9,12])))
        years = np.asarray(index.year)
        if meteorological:
            years = years+(index.month==12)
        labels = np.unique(years)
        groups = [np.flatnonzero(years == year) for year in labels]
        months = index.to_period("M").asi8
        days = index.normalize().asi8 // (24 * 60 * 60 * 1_000_000_000)
        monthly = len(months) > 1 and np.all(np.diff(months) == 1)
        quarterly = len(months) > 1 and np.all(np.diff(months) == 3)
        daily = len(days) > 1 and np.all(np.diff(days) == 1)
        yearly = len(months) > 1 and np.all(np.diff(months) == 12)
        complete = []
        for year, group in zip(labels, groups):
            expected = (12 if monthly else 4 if quarterly else (366 if calendar.isleap(int(year)) else 365)
                        if daily else 1 if yearly else int(period or 0))
            complete.append(expected > 0 and len(group) == expected)
    if not include_partial:
        selected = [j for j, full in enumerate(complete) if full]
        groups = [groups[j] for j in selected]
        labels = np.asarray(labels)[selected]
        if not groups:
            raise ValueError("No complete calendar year is available for annual risk. Use annual=False or include_partial=True for observed-window probabilities.")
    return groups, np.asarray(labels)
