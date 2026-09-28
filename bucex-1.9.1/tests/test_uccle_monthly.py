"""Daily-to-monthly contracts, including incomplete data and release coverage."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import bucex as bx
from bucex.cli import main as cli_main


ROOT = Path(__file__).resolve().parents[1]


def daily(start="2024-02-01", end="2024-02-29"):
    dates = pd.date_range(start, end)
    day = dates.day.to_numpy()
    return pd.DataFrame({"DAY": dates, "TX": day, "TN": day - 10})


def test_leap_month_definitions_and_export_roundtrip(tmp_path):
    frame = daily().sample(frac=1, random_state=1)
    path = tmp_path / "temperatures.csv"
    frame.to_csv(path, index=False)
    output = tmp_path / "monthly"
    monthly = bx.derive_uccle_monthly(path, output_dir=output)
    assert list(monthly.columns) == list(bx.UCCLE_SERIES)
    assert monthly.index.tolist() == [pd.Timestamp("2024-02-01")]
    # Days 1..29 have mean 15; TN is ten degrees lower, without a sign flip.
    np.testing.assert_allclose(monthly.iloc[0], [15, 5, 29, 1, 19, -9])
    pd.testing.assert_frame_equal(
        monthly, bx.load_uccle_multiseries(output), check_freq=False, check_dtype=False,
    )
    loaded = bx.load_uccle_daily(path)
    pd.testing.assert_frame_equal(loaded, daily(), check_dtype=False)
    pd.testing.assert_frame_equal(monthly, bx.derive_uccle_monthly(loaded))
    report = json.loads((output / "quality_report.json").read_text())
    assert report == monthly.attrs
    assert report["daily_file"] == "temperatures.csv"
    assert report["last_included_day"] == "2024-02-29"
    assert len(report["daily_sha256"]) == 64


def test_partial_boundary_months_are_explicitly_excluded():
    frame = daily("2024-01-15", "2024-03-31")
    with pytest.warns(UserWarning, match="2024-01, 2024-03"):
        monthly = bx.derive_uccle_monthly(frame, end="2024-03-10")
    assert monthly.index.tolist() == [pd.Timestamp("2024-02-01")]
    assert monthly.attrs["excluded_partial_boundary_months"] == ["2024-01", "2024-03"]


@pytest.mark.parametrize("problem", ["day", "whole_month", "TX", "TN"])
def test_incomplete_retained_months_fail_before_export(tmp_path, problem):
    frame = daily("2024-01-01", "2024-03-31")
    if problem == "day":
        frame = frame.loc[frame.DAY != "2024-02-15"]
    elif problem == "whole_month":
        frame = frame.loc[frame.DAY.dt.month != 2]
    else:
        frame[problem] = frame[problem].astype("Float64")
        frame.loc[frame.DAY == "2024-02-15", problem] = pd.NA
    output = tmp_path / "monthly"
    with pytest.raises(ValueError, match="Missing daily.*2024-02"):
        bx.derive_uccle_monthly(frame, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["duplicate", "missing_date", "time", "infinity", "column"])
def test_invalid_daily_input_is_rejected(problem):
    frame = daily()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "missing_date":
        frame.loc[0, "DAY"] = pd.NaT
    elif problem == "time":
        frame.loc[0, "DAY"] += pd.Timedelta(hours=1)
    elif problem == "infinity":
        frame["TX"] = frame.TX.astype(float)
        frame.loc[0, "TX"] = np.inf
    else:
        frame = frame.drop(columns="TN")
    with pytest.raises(ValueError):
        bx.derive_uccle_monthly(frame)


def test_explicit_source_and_legacy_directory_keyword(tmp_path):
    path = tmp_path / "Uccle_new.csv"
    frame = daily().assign(RR=np.nan)
    frame.to_csv(path, index=False)
    assert "RR" in bx.load_uccle_daily(path)
    pd.testing.assert_frame_equal(
        bx.derive_uccle_monthly(path), bx.derive_uccle_monthly(data_dir=tmp_path),
    )
    with pytest.raises(ValueError, match="not both"):
        bx.load_uccle_daily(path, data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        bx.load_uccle_daily(tmp_path / "missing.csv")
    frame.to_csv(tmp_path / "Uccle_another.csv", index=False)
    with pytest.raises(ValueError, match="Multiple daily files"):
        bx.load_uccle_daily(tmp_path)
    assert len(bx.load_uccle_daily(path)) == 29


def test_reported_tn_above_tx_is_preserved_and_flagged():
    frame = daily()
    frame.loc[0, "TN"] = 2
    with pytest.warns(UserWarning, match="1 reported daily TN > TX"):
        monthly = bx.derive_uccle_monthly(frame)
    assert monthly.loc["2024-02-01", "TNm"] == pytest.approx(5 + 11 / 29)
    assert monthly.attrs["daily_TN_above_TX"] == [{"date": "2024-02-01", "TX": 1., "TN": 2.}]


def test_daily_validation_requires_coverage_of_every_summary_month(tmp_path):
    bx.derive_uccle_monthly(daily("2024-01-01", "2024-02-29"), output_dir=tmp_path)
    with pytest.raises(ValueError, match="does not cover all supplied months: 2024-01"):
        bx.validate_uccle_data(tmp_path, daily_source=daily())


def test_validation_reports_mismatch_and_cli_accepts_daily_source(tmp_path, capsys):
    frame = daily()
    output = tmp_path / "monthly"
    monthly = bx.derive_uccle_monthly(frame, output_dir=output)
    monthly["TXm"] += 0.5
    monthly[["TXm"]].to_csv(output / "TXm.csv")
    path = tmp_path / "daily.csv"
    frame.to_csv(path, index=False)
    assert cli_main(["validate-data", "--data-dir", str(output), "--daily-source", str(path)]) == 0
    report = {row["series"]: row for row in json.loads(capsys.readouterr().out)}
    assert report["TXm"]["daily_max_abs_difference"] == pytest.approx(0.5)
    assert report["TNm"]["daily_max_abs_difference"] == 0


def test_no_complete_month_is_an_error():
    with pytest.raises(ValueError, match="no complete calendar month"):
        bx.derive_uccle_monthly(daily(), end="2024-02-28")


def test_bundled_release_coverage_and_default_do_not_depend_on_cwd(tmp_path, monkeypatch):
    bx.derive_uccle_monthly(daily(), output_dir=tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    data = bx.load_uccle_multiseries()
    assert data.shape == (1616, 6)
    assert data.index[0] == pd.Timestamp("1892-01-01")
    assert data.index[-1] == pd.Timestamp("2026-08-01")
    with pytest.warns(UserWarning, match="2 reported daily TN > TX"):
        report = bx.validate_uccle_data(daily_source=ROOT / "data/Uccle_31_08_26.csv")
    np.testing.assert_allclose(report.daily_max_abs_difference, 0, atol=1e-12)


def test_research_runner_can_initialize_updated_lower_extremes():
    from research.monthly.models import channel, marginal_prior, fit_options

    config = bx.load_config(ROOT / "research/monthly/config/independent.json")
    config["mcmc"].update(draws=1, warmup=1, chains=1, progress=False)
    data = bx.load_uccle_multiseries(series="TNn", start="2024-09-01")
    item = channel("TNn", data, config)
    fit = bx.fit(
        data["TNn"], bx.Model(item.observation, item.components),
        priors=marginal_prior(item, data, config),
        **fit_options(config, family=item.family, tail=item.tail),
    )
    assert fit.n_time == 24
    assert np.all(np.isfinite(fit.state_draws))
    np.testing.assert_allclose(fit.observed, data["TNn"])
