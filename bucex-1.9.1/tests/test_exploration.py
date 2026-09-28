"""Independent checks for descriptive monthly summaries and their provenance."""
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bucex as bx


def observations():
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    # Residuals sum to zero and are orthogonal to the five year coordinates.
    # Their IQR is exactly 2, so the detrended IQR is known independently of OLS.
    residual = np.repeat([-1., 2., 0., -2., 1.], 12)
    month = dates.month.to_numpy()
    values = 8+month/2+.2*(dates.year.to_numpy()-2000)+residual*month/10
    return pd.DataFrame({"A": values, "B": 3-4*values}, index=dates)


def explore(data=None, **kwargs):
    window = {"Five years": ["2000-01", "2004-12"]}
    return bx.explore_monthly(observations() if data is None else data,
                             periods=window, eras=window, **kwargs)


def test_calendar_mean_and_detrended_iqr_have_known_values():
    source = observations()
    original = source.copy(deep=True)
    result = explore(source)
    cycle = result.seasonal_cycles.query("series == 'A'")
    spread = result.monthly_spread.query("series == 'A'")
    np.testing.assert_allclose(cycle["mean"], 8+np.arange(1, 13)/2+.4)
    np.testing.assert_allclose(spread.residual_iqr, np.arange(1, 13)/5, atol=1e-13)
    np.testing.assert_allclose(spread.trend_per_year, .2, atol=1e-13)
    np.testing.assert_allclose(result.monthly_spread.query("series == 'B'").residual_iqr,
                               4*spread.residual_iqr, atol=1e-13)
    assert (cycle.n == 5).all() and (cycle.n_missing == 0).all()
    assert (result.seasonal_cycles.query("series == 'B'")["mean"] < 0).all()
    pd.testing.assert_frame_equal(source, original)


def test_missing_values_and_absent_months_are_counted():
    data = observations().drop(pd.Timestamp("2002-02-01"))
    data.loc["2003-03-01", "A"] = np.nan
    result = explore(data)
    a = result.seasonal_cycles.query("series == 'A'").set_index("month")
    b = result.monthly_spread.query("series == 'B'").set_index("month")
    assert a.loc[2, "n"] == a.loc[3, "n"] == 4
    assert a.loc[2, "n_missing"] == a.loc[3, "n_missing"] == 1
    assert b.loc[2, "n_missing"] == 1 and b.loc[3, "n_missing"] == 0
    assert a.loc[2, "n_expected"] == 5
    assert result.metadata["n_finite"] == {"A": 58, "B": 59}


def test_period_index_and_series_input_preserve_meaning():
    data = observations().A.rename("temperature")
    data.index = data.index.to_period("M")
    result = explore(data)
    assert result.metadata["series"] == ["temperature"]
    assert result.metadata["observed_end"] == "2004-12-01"
    np.testing.assert_allclose(result.monthly_spread.residual_iqr, np.arange(1, 13)/5, atol=1e-13)


def test_duplicate_calendar_months_are_rejected_even_with_different_days():
    data = observations()
    extra = data.iloc[[0]].copy()
    extra.index = pd.to_datetime(["2000-01-15"])
    with pytest.raises(ValueError, match="one observation per calendar month"):
        explore(pd.concat([data, extra]))


def test_windows_do_not_silently_clip_to_available_data():
    window = {"Too early": ["1999-01", "2004-12"]}
    with pytest.raises(ValueError, match="outside available"):
        bx.explore_monthly(observations(), periods=window, eras=window)


def test_sparse_months_fail_before_producing_spread_estimates():
    data = observations()
    data.loc[data.index.month == 3, "A"] = np.nan
    with pytest.raises(ValueError, match="month 3: 0 finite"):
        explore(data)


@pytest.mark.parametrize("count", [0, 2, 3.5, True])
def test_minimum_count_has_explicit_semantics(count):
    with pytest.raises(ValueError, match="integer of at least 3"):
        explore(min_count=count)


def test_infinities_and_nondated_inputs_are_rejected():
    data = observations()
    data.iloc[2, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        explore(data)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        explore(observations().reset_index(drop=True))


def test_save_records_plotted_values_original_data_and_empirical_bands(tmp_path):
    result = explore()
    before = plt.rcParams.copy()
    opened = set(plt.get_fignums())
    metadata = result.save(tmp_path, formats=("png", "pdf"), dpi=45, units="°C")
    assert set(plt.get_fignums()) == opened
    for key in ["font.size", "font.family", "axes.spines.top", "axes.prop_cycle"]:
        assert plt.rcParams[key] == before[key]
    for name in ["exploratory_seasonal_cycles", "exploratory_monthly_spread"]:
        assert (tmp_path/"figures"/(name+".png")).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path/"figures"/(name+".pdf")).read_bytes().startswith(b"%PDF")
    exported = pd.read_csv(tmp_path/"data"/"exploratory_monthly_spread.csv")
    np.testing.assert_allclose(exported.residual_iqr, result.monthly_spread.residual_iqr)
    digest = hashlib.sha256((tmp_path/"data"/"monthly_observations.csv").read_bytes()).hexdigest()
    assert metadata["input_sha256"] == digest
    assert "not confidence or posterior" in metadata["inferential_status"]
    assert json.loads((tmp_path/"exploration_metadata.json").read_text())["bucex_version"] == bx.__version__


def test_serra_driver_uses_package_api_without_fitting(tmp_path, monkeypatch):
    from research.monthly.explore import run
    frame = observations().A.rename("TNm").rename_axis("date")
    frame.to_csv(tmp_path/"TNm.csv")
    def forbidden(*args, **kwargs):
        raise AssertionError("Exploration must not call fit().")
    monkeypatch.setattr(bx, "fit", forbidden)
    window = {"Five years": ["2000-01", "2004-12"]}
    config = dict(data=dict(data_dir=str(tmp_path), series=["TNm"]),
                  periods=window, eras=window, figures=dict(figures=False),
                  output=str(tmp_path/"runs"))
    directory = run(config)
    assert (directory/"config.json").exists()
    assert not (directory/"figures").exists()
    meta = json.loads((directory/"exploration_metadata.json").read_text())
    assert meta["n_months"] == 60 and meta["observed_end"] == "2004-12-01"
