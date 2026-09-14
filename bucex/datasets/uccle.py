"""Reproducible univariate and hierarchical Uccle data helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

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
            and (explicit / "Uccle_24_10_23.csv").is_file()
            and not has_any_summary
        )
        if not is_daily_source_directory:
            raise FileNotFoundError(
                f"Explicit Uccle data directory {explicit} is missing: "
                + ", ".join(f"{name}.csv" for name in missing)
            )

    candidates = [Path.cwd() / "data", Path(__file__).resolve().parents[1] / "data"]
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
    value_column = series if series in frame else next((c for c in frame if c != "date"), None)
    if value_column is None:
        raise ValueError(f"{path.name} has no value column.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    values = pd.to_numeric(frame[value_column], errors="raise")
    result = pd.Series(values.to_numpy(float), index=dates, name=series).sort_index()
    if result.index.duplicated().any() or not np.all(np.isfinite(result.to_numpy())):
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


def load_uccle_daily(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the optional daily source file from a source checkout."""

    candidates = [] if data_dir is None else [Path(data_dir)]
    candidates.extend([Path.cwd() / "data", Path(__file__).resolve().parents[1] / "data"])
    path = next(
        (candidate / "Uccle_24_10_23.csv" for candidate in candidates if (candidate / "Uccle_24_10_23.csv").is_file()),
        None,
    )
    if path is None:
        raise FileNotFoundError("Pass data_dir containing Uccle_24_10_23.csv.")
    frame = pd.read_csv(path)
    required = {"DAY", "TX", "TN", "RR"}
    if set(frame) != required:
        raise ValueError(f"Daily file must contain exactly {sorted(required)}.")
    frame["DAY"] = pd.to_datetime(frame["DAY"], errors="raise")
    return frame.sort_values("DAY").reset_index(drop=True)


def derive_uccle_monthly(data_dir: str | Path | None = None, *, end: str = "2022-12-31") -> pd.DataFrame:
    """Recreate the six summaries from the optional daily source."""

    daily = load_uccle_daily(data_dir).set_index("DAY").loc[: pd.Timestamp(end)]
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
    monthly.index.name = "date"
    return monthly


def validate_uccle_data(data_dir: str | Path | None = None, *, check_daily: bool = False) -> pd.DataFrame:
    """Return an integrity table for the bundled summaries."""

    # A source checkout keeps the optional daily file in ``data/`` and the six
    # packaged summaries in ``bucex/data/``. If the explicit directory has no
    # summary files at all, use it only for the requested daily cross-check.
    summary_data_dir = data_dir
    if data_dir is not None:
        candidate = Path(data_dir)
        has_any_summary = any(
            (candidate / f"{name}.csv").is_file() for name in UCCLE_SERIES
        )
        if not has_any_summary:
            summary_data_dir = None

    rows = []
    for name in UCCLE_SERIES:
        values = load_uccle_series(name, summary_data_dir)
        rows.append(
            {"series": name, "n": values.size, "start": values.index[0], "end": values.index[-1], "minimum": values.min(), "maximum": values.max()}
        )
    table = pd.DataFrame(rows).set_index("series")
    if check_daily:
        rebuilt = derive_uccle_monthly(data_dir)
        for name in UCCLE_SERIES:
            supplied = load_uccle_series(name, summary_data_dir)
            delta = rebuilt[name].reindex(supplied.index).to_numpy() - supplied.to_numpy()
            table.loc[name, "daily_max_abs_difference"] = np.nanmax(np.abs(delta))
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
