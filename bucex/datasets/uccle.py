"""Reproducible univariate and hierarchical Uccle data helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping
import warnings

import numpy as np
import pandas as pd

from ..api.fit import fit
from ..components import DummySeasonal, LocalLevel, LocalLinearTrend
from ..core.fit import FitResult
from ..inference.config import Laplace, MCMC
from ..models.multiseries import Channel, MultiSeriesModel
from ..models.shared import Departures, Shared
from ..models.structural import Model
from ..observation import GEV, Gaussian
from ..priors.hierarchical import HierarchicalPrior, HierarchicalPriors
from ..priors.joint import JointPriors


UCCLE_SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
# Necessary physical inequalities for summaries of the same complete blocks.
# They diagnose joint predictive incompatibility; they do not constrain likelihoods.
UCCLE_ORDER_CONSTRAINTS = (
    ("TXn", "TXm"), ("TXm", "TXx"),
    ("TNn", "TNm"), ("TNm", "TNx"),
    ("TNn", "TXn"), ("TNm", "TXm"), ("TNx", "TXx"),
)
UCCLE_INFO = {
    "TXm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily maximum temperature"},
    "TNm": {"family": "gaussian", "tail": "max", "description": "monthly mean daily minimum temperature"},
    "TXx": {"family": "gev", "tail": "max", "description": "monthly maximum daily maximum temperature"},
    "TXn": {"family": "gev", "tail": "min", "description": "monthly minimum daily maximum temperature"},
    "TNx": {"family": "gev", "tail": "max", "description": "monthly maximum daily minimum temperature"},
    "TNn": {"family": "gev", "tail": "min", "description": "monthly minimum daily minimum temperature"},
}


def _normalize_series_names(series: Iterable[str] | str) -> tuple[str, ...]:
    """Treat a single series name as one name rather than an iterable of letters."""

    selected = (series,) if isinstance(series, str) else tuple(series)
    unknown = sorted(set(selected) - set(UCCLE_SERIES))
    if unknown:
        raise ValueError(f"Unknown Uccle series: {unknown}; choose from {UCCLE_SERIES}.")
    if len(set(selected)) != len(selected):
        raise ValueError("Uccle series names must be unique.")
    return selected


def _resolve_data_dir(
    data_dir: str | Path | None = None,
    *,
    required_series: Iterable[str] | str = UCCLE_SERIES,
) -> Path:
    required = _normalize_series_names(required_series)
    if data_dir is not None:
        explicit = Path(data_dir)
        missing = [name for name in required if not (explicit / f"{name}.csv").is_file()]
        if not missing:
            return explicit
        has_any_summary = any(
            (explicit / f"{name}.csv").is_file() for name in UCCLE_SERIES
        )
        is_daily_source_directory = (
            explicit.is_dir()
            and any(explicit.glob("Uccle_*.csv"))
            and not has_any_summary
        )
        if not is_daily_source_directory:
            raise FileNotFoundError(
                f"Explicit Uccle data directory {explicit} is missing: "
                + ", ".join(f"{name}.csv" for name in missing)
            )

    # Custom summaries require data_dir; stale working-directory copies must
    # not shadow the bundled observations in a newly installed release.
    candidates = [Path(__file__).resolve().parents[1] / "data"]
    for candidate in candidates:
        if all((candidate / f"{name}.csv").is_file() for name in required):
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    requested = ", ".join(f"{name}.csv" for name in required)
    raise FileNotFoundError(
        f"Could not locate requested Uccle files ({requested}). Tried: {tried}"
    )


def load_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load one complete monthly Uccle series."""

    if series not in UCCLE_INFO:
        raise ValueError(f"Unknown Uccle series {series!r}; choose from {UCCLE_SERIES}.")
    path = _resolve_data_dir(data_dir, required_series=series) / f"{series}.csv"
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path.name} must contain a date column.")
    if frame.empty:
        raise ValueError(f"{path.name} contains no observations.")
    value_column = series if series in frame else next((c for c in frame if c != "date"), None)
    if value_column is None:
        raise ValueError(f"{path.name} has no value column.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    values = pd.to_numeric(frame[value_column], errors="raise")
    result = pd.Series(values.to_numpy(float), index=dates, name=series).sort_index()
    if result.index.hasnans or result.index.duplicated().any() or not np.all(np.isfinite(result.to_numpy())):
        raise ValueError(f"{path.name} contains duplicate dates or non-finite values.")
    expected = pd.date_range(result.index[0], result.index[-1], freq="MS")
    if not result.index.equals(expected):
        raise ValueError(f"{path.name} is not a complete consecutive monthly series.")
    if start is not None:
        result = result.loc[pd.Timestamp(start) :]
    if end is not None:
        result = result.loc[: pd.Timestamp(end)]
    if result.empty:
        raise ValueError("The requested date range contains no observations.")
    return result


def load_uccle_multiseries(
    data_dir: str | Path | None = None,
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load aligned summaries in the requested channel order."""

    selected = _normalize_series_names(series)
    if not selected:
        raise ValueError("Select at least one Uccle series.")
    values = [load_uccle_series(name, data_dir, start=start, end=end) for name in selected]
    if any(not item.index.equals(values[0].index) for item in values[1:]):
        raise ValueError("Uccle channels must have the same dates; choose an explicit common range.")
    frame = pd.concat(
        values,
        axis=1,
        join="inner",
    )
    if frame.isna().any().any():
        raise ValueError("Uccle channels are not completely aligned.")
    return frame.loc[:, list(selected)]


def _daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and numeric temperatures without changing observations."""
    missing = {"DAY", "TX", "TN"} - set(frame)
    if missing:
        raise ValueError(f"Daily data are missing columns: {sorted(missing)}.")
    frame = frame.copy()
    frame["DAY"] = pd.to_datetime(frame["DAY"], errors="raise")
    dates = pd.DatetimeIndex(frame["DAY"])
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("DAY must contain nonmissing dates at midnight without a timezone.")
    if dates.duplicated().any():
        raise ValueError("Daily data contain duplicate dates.")
    for name in ("TX", "TN"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{name} contains infinite values.")
    return frame.sort_values("DAY").reset_index(drop=True)


def _daily_argument(source, data_dir):
    """Keep the existing data_dir keyword while preferring an explicit source."""
    if data_dir is not None:
        if source is not None:
            raise ValueError("Pass source or data_dir, not both.")
        return data_dir
    return source


def load_uccle_daily(
    source: str | Path | None = None, *, data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read a daily CSV with DAY, TX and TN; RR is optional and preserved.

    Prefer an explicit CSV path. A directory must contain exactly one
    Uccle_*.csv. Without source, look in the current/source checkout's data/.
    The daily file is not included in the wheel. The previous data_dir keyword
    remains supported.
    """
    source = _daily_argument(source, data_dir)
    candidates = ([Path(source)] if source is not None else
                  [Path.cwd() / "data", Path(__file__).resolve().parents[2] / "data"])
    for candidate in dict.fromkeys(candidates):
        paths = sorted(candidate.glob("Uccle_*.csv")) if candidate.is_dir() else [candidate]
        paths = [path for path in paths if path.is_file()]
        if len(paths) > 1:
            raise ValueError(f"Multiple daily files in {candidate}; pass an explicit CSV path.")
        if paths:
            path = paths[0]
            frame = _daily_frame(pd.read_csv(path))
            frame.attrs["daily_file"] = path.name
            frame.attrs["daily_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return frame
    raise FileNotFoundError("No daily CSV found; pass its path to load_uccle_daily(source).")


def derive_uccle_monthly(
    source: pd.DataFrame | str | Path | None = None,
    *,
    end: str | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Condense daily TX/TN into the six complete calendar-month summaries.

    source is a daily DataFrame (DAY, TX, TN), a CSV path, or a directory as
    accepted by load_uccle_daily. end is an optional inclusive daily cutoff;
    by default every available complete month is used. Partial boundary
    months are excluded; missing dates/temperatures in retained months raise.

    Return a DataFrame indexed by month starts, with quality metadata in attrs.
    If output_dir is given, also write the six date/series CSVs and
    quality_report.json. Temperatures retain their original Celsius signs.
    """
    source = _daily_argument(source, data_dir)
    frame = _daily_frame(source) if isinstance(source, pd.DataFrame) else load_uccle_daily(source)
    daily = frame.set_index("DAY")
    if end is not None:
        cutoff = pd.Timestamp(end)
        if pd.isna(cutoff) or cutoff.tz is not None or cutoff != cutoff.normalize():
            raise ValueError("end must be a date without a time or timezone.")
        daily = daily.loc[:cutoff]
    if daily.empty:
        raise ValueError("No daily observations remain in the requested range.")
    first, last = daily.index[0], daily.index[-1]
    calendar = pd.date_range(first.to_period("M").start_time,
                             last.to_period("M").end_time.normalize(), freq="D")
    counts = daily[["TX", "TN"]].reindex(calendar).resample("MS").count()
    retained = ((counts.index >= first)
                & (counts.index + pd.offsets.MonthEnd(0) <= last))
    excluded = counts.index[~retained].strftime("%Y-%m").tolist()
    counts = counts.loc[retained]
    if counts.empty:
        raise ValueError("The selected data contain no complete calendar month.")
    bad = counts.lt(counts.index.days_in_month, axis=0).any(axis=1)
    if bad.any():
        details = [f"{date:%Y-%m} (TX={row.TX}, TN={row.TN}, expected={date.days_in_month})"
                   for date, row in counts.loc[bad].iterrows()]
        raise ValueError("Missing daily temperatures or dates in retained months: " + "; ".join(details))
    monthly = pd.DataFrame(
        {
            "TXm": daily["TX"].resample("MS").mean(),
            "TNm": daily["TN"].resample("MS").mean(),
            "TXx": daily["TX"].resample("MS").max(),
            "TXn": daily["TX"].resample("MS").min(),
            "TNx": daily["TN"].resample("MS").max(),
            "TNn": daily["TN"].resample("MS").min(),
        }
    )
    monthly = monthly.loc[counts.index, list(UCCLE_SERIES)]
    monthly.index.name = "date"
    last_day = monthly.index[-1] + pd.offsets.MonthEnd(0)
    used = daily.loc[monthly.index[0]:last_day]
    inverted = used.loc[used.TN > used.TX, ["TX", "TN"]]
    monthly.attrs = {
        **{key: frame.attrs[key] for key in ("daily_file", "daily_sha256") if key in frame.attrs},
        "frequency": "monthly", "seasonal_period": 12, "n_months": len(monthly),
        "first_month": str(monthly.index[0].date()), "last_month": str(monthly.index[-1].date()),
        "last_included_day": str(last_day.date()), "excluded_partial_boundary_months": excluded,
        "daily_TN_above_TX": [{"date": str(date.date()), "TX": float(row.TX), "TN": float(row.TN)}
                              for date, row in inverted.iterrows()],
    }
    if excluded:
        warnings.warn("Excluded partial boundary months: " + ", ".join(excluded), UserWarning, stacklevel=2)
    if not inverted.empty:
        warnings.warn(f"Retained {len(inverted)} reported daily TN > TX pairs; see monthly.attrs or quality_report.json.",
                      UserWarning, stacklevel=2)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name in UCCLE_SERIES:
            monthly[[name]].to_csv(output / f"{name}.csv", date_format="%Y-%m-%d")
        (output / "quality_report.json").write_text(json.dumps(monthly.attrs, indent=2) + "\n", encoding="utf-8")
    return monthly


def validate_uccle_data(
    data_dir: str | Path | None = None,
    *,
    check_daily: bool = False,
    daily_source: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Check aligned summaries, optionally against an explicit daily source.

    Passing daily_source enables the cross-check. check_daily=True retains
    automatic daily-file lookup. Daily data must cover every supplied month;
    missing comparison dates are errors, never ignored NaNs.
    """
    supplied = load_uccle_multiseries(data_dir)
    table = pd.DataFrame([
        {"series": name, "n": len(supplied), "start": supplied.index[0], "end": supplied.index[-1],
         "minimum": supplied[name].min(), "maximum": supplied[name].max()}
        for name in UCCLE_SERIES
    ]).set_index("series")
    if check_daily or daily_source is not None:
        if daily_source is None and data_dir is not None:
            candidate = Path(data_dir)
            if any(candidate.glob("Uccle_*.csv")):
                daily_source = candidate
        rebuilt = derive_uccle_monthly(daily_source,
                                       end=str((supplied.index[-1] + pd.offsets.MonthEnd(0)).date()))
        missing = supplied.index.difference(rebuilt.index)
        if len(missing):
            raise ValueError("Daily source does not cover all supplied months: "
                             + ", ".join(missing.strftime("%Y-%m")))
        for name in UCCLE_SERIES:
            delta = rebuilt.loc[supplied.index, name].to_numpy() - supplied[name].to_numpy()
            table.loc[name, "daily_max_abs_difference"] = np.max(np.abs(delta))
    return table


def make_uccle_hierarchical_model(
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    seasonal: bool = True,
    period: int = 12,
) -> MultiSeriesModel:
    """Create separate structural paths coupled through a prior hierarchy.

    Seasonality is included in every channel by default. Under hierarchical
    selection it may be fixed or dynamic, but it is not allowed to disappear.
    """

    selected = _normalize_series_names(series)
    if len(selected) < 2:
        raise ValueError("A hierarchical model requires at least two series.")
    if int(period) < 2:
        raise ValueError("period must be at least 2.")
    channels = []
    for name in selected:
        info = UCCLE_INFO[name]
        components = [LocalLinearTrend()]
        if seasonal:
            components.append(DummySeasonal(period=int(period)))
        channels.append(
            Channel(
                name=name,
                observation=Gaussian() if info["family"] == "gaussian" else GEV(),
                components=tuple(components),
                tail="lower" if info["tail"] == "min" else None,
                description=info["description"],
            )
        )
    return MultiSeriesModel(
        channels=tuple(channels),
        name="Uccle hierarchical structural model",
        description="Separate paths with pooled structural probabilities and/or normal-slab scales.",
    )


def fit_uccle_hierarchical(
    data_dir: str | Path | None = None,
    *,
    model: MultiSeriesModel | None = None,
    series: Iterable[str] | str = UCCLE_SERIES,
    pooling: str = "selection",
    priors: HierarchicalPrior | HierarchicalPriors | str | None = None,
    seasonal: bool = True,
    period: int = 12,
    start: str | None = "1980-01-01",
    end: str | None = None,
    **kwargs,
) -> FitResult:
    """Fit aligned Uccle series with pooled SSVS, slabs, or both."""

    resolved_model = model or make_uccle_hierarchical_model(series=series, seasonal=seasonal, period=period)
    values = load_uccle_multiseries(data_dir, series=resolved_model.channel_names, start=start, end=end)
    resolved_priors = HierarchicalPrior(pool=pooling) if priors is None else priors
    options = dict(kwargs)
    options.setdefault("parameterization", "fruehwirth_schnatter")
    options.setdefault("engine", "auto")
    options.setdefault("asis", False)
    return fit(values, resolved_model, priors=resolved_priors, **options)


def make_uccle_shared_model(
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    seasonal: bool = True,
    period: int = 12,
    departures: str | None = "trend",
    weights: Mapping[str, float] | None = None,
    baseline_means: Mapping[str, float] | None = None,
    copula=None,
) -> MultiSeriesModel:
    """Construct a common warming trajectory with constrained departures.

    This is a short application of the general ``Channel``, ``Shared`` and
    ``Departures`` grammar. Every channel retains its own static intercept and
    fixed seasonal cycle. Unit loadings keep effects in degrees Celsius; the
    weighted sum of departures is zero at each time. ``departures='rw'`` uses
    simpler random-walk deviations, and ``None`` omits them entirely.

    Initial levels of the common trend and departures are pinned at zero.
    Initial slope SDs are 0.005 and 0.002 degrees per model step, respectively;
    the bundled observations are monthly. Initial intercept and seasonal SDs
    are 10 degrees. For other priors, construct the general model directly.
    """
    selected = _normalize_series_names(series)
    if len(selected) < 2:
        raise ValueError("A shared model requires at least two series.")
    if departures not in {None, "rw", "trend"}:
        raise ValueError("departures must be 'trend', 'rw', or None.")
    if weights is not None and departures is None:
        raise ValueError("weights require a departure component.")
    if baseline_means is not None and set(baseline_means) != set(selected):
        raise ValueError("baseline_means must name every selected channel exactly once.")
    channels = []
    for name in selected:
        info = UCCLE_INFO[name]
        components = [LocalLevel(mode="static", initial_mean=None if baseline_means is None else float(baseline_means[name]), initial_sd=10.0)]
        if seasonal:
            components.append(DummySeasonal(period=period, mode="static", initial_sd=10.0))
        channels.append(Channel(name, Gaussian() if info["family"] == "gaussian" else GEV(), tuple(components),
                                tail="lower" if info["tail"] == "min" else None, description=info["description"]))
    shared = [Shared("warming", LocalLinearTrend(initial_level=0.0, initial_level_sd=0.0, initial_slope_sd=0.005))]
    if departures is not None:
        component = (LocalLevel(initial_mean=0.0, initial_sd=0.0) if departures == "rw"
                     else LocalLinearTrend(initial_level=0.0, initial_level_sd=0.0, initial_slope_sd=0.002))
        shared.append(Departures("departure", component, weights=weights))
    return MultiSeriesModel(channels=tuple(channels), shared=tuple(shared), copula=copula,
                            name="Uccle common warming and departures",
                            description="Unit-loading common trend, weighted sum-to-zero departures, separate fixed seasonal cycles.")


def fit_uccle_shared(
    data_dir: str | Path | None = None,
    *,
    model: MultiSeriesModel | None = None,
    series: Iterable[str] | str = UCCLE_SERIES,
    priors: JointPriors | None = None,
    seasonal: bool = True,
    departures: str | None = "trend",
    weights: Mapping[str, float] | None = None,
    copula=None,
    start: str | None = "1892-01-01",
    end: str | None = None,
    **kwargs,
) -> FitResult:
    """Fit the shared model through the ordinary ``fit(data, model, ...)`` API.

    Pass explicit ``JointPriors`` for a scientific analysis: ``priors=None``
    uses the same recorded data-adaptive convenience calibration as ``fit``.
    When constructing a model here, intercept-prior centers are the observed
    channel medians; pass your own model to supply data-independent centers.
    """
    selected = model.channel_names if model is not None else _normalize_series_names(series)
    values = load_uccle_multiseries(data_dir, series=selected, start=start, end=end)
    resolved_model = model or make_uccle_shared_model(
        series=selected, seasonal=seasonal, departures=departures, weights=weights,
        baseline_means={name: float(values[name].median()) for name in selected}, copula=copula,
    )
    if not getattr(resolved_model, "shared", ()):
        raise ValueError("fit_uccle_shared requires a model containing shared states.")
    options = dict(kwargs)
    options.setdefault("engine", "auto")
    options.setdefault("parameterization", "centered")
    options.setdefault("asis", False)
    return fit(values, resolved_model, priors=priors, **options)


def fit_uccle_series(
    series: str,
    data_dir: str | Path | None = None,
    *,
    model: Model | None = None,
    priors: object = "normal",
    mcmc: MCMC | None = None,
    engine: str = "auto",
    parameterization: str = "fruehwirth_schnatter",
    asis: bool = True,
    start: str | None = "1980-01-01",
    end: str | None = None,
    laplace: Laplace | None = None,
    **kwargs,
) -> FitResult:
    """Fit one summary using an ordinary declarative structural model.

    The default contains a local linear trend and monthly dummy seasonality.
    Supplying ``model=`` exposes the general component and observation API;
    this helper only loads the series and sets its tail orientation and dates.
    """

    values = load_uccle_series(series, data_dir, start=start, end=end)
    info = UCCLE_INFO[series]
    resolved_model = model or Model(
        observation=Gaussian() if info["family"] == "gaussian" else GEV(),
        components=(LocalLinearTrend(), DummySeasonal(period=12)),
        name=series,
    )
    return fit(
        values.to_numpy(), resolved_model, priors=priors,
        engine=engine, parameterization=parameterization, asis=asis,
        mcmc=MCMC() if mcmc is None else mcmc, laplace=laplace,
        dates=values.index.to_numpy(), name=series, tail=info["tail"], **kwargs,
    )


@dataclass
class UccleFitCollection:
    """Small mapping returned by :func:`fit_uccle_all`."""

    fits: dict[str, FitResult]

    def __getitem__(self, name: str) -> FitResult:
        return self.fits[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fits)

    def __len__(self) -> int:
        return len(self.fits)

    def items(self):
        return self.fits.items()

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, result in self.fits.items():
            for parameter, values in result.static_summary().items():
                rows.append({"series": name, "parameter": parameter, **values})
        return pd.DataFrame(rows)


def fit_uccle_all(
    data_dir: str | Path | None = None,
    *,
    series: Iterable[str] | str = UCCLE_SERIES,
    **kwargs,
) -> UccleFitCollection:
    """Fit the requested summaries independently with a common user API."""

    selected = _normalize_series_names(series)
    if not selected:
        raise ValueError("At least one Uccle series is required.")
    return UccleFitCollection(
        {name: fit_uccle_series(name, data_dir, **kwargs) for name in selected}
    )


__all__ = [
    "UCCLE_INFO", "UCCLE_SERIES", "UccleFitCollection", "derive_uccle_monthly",
    "fit_uccle_all", "fit_uccle_hierarchical", "fit_uccle_series",
    "fit_uccle_shared", "make_uccle_shared_model",
    "load_uccle_daily", "load_uccle_multiseries", "load_uccle_series",
    "make_uccle_hierarchical_model", "validate_uccle_data",
]
