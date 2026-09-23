"""Protect numerical meaning when turning research exports into figures."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bucex as bx


def report(directory, name="A", level=.95):
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    pd.DataFrame(dict(time=dates, lower=np.arange(24)-1., median=np.arange(24),
                      upper=np.arange(24)+1.)).to_csv(directory/f"{name}_level.csv", index=False)
    pd.DataFrame(dict(time=dates, pit=np.linspace(.02, .98, 24))).to_csv(
        directory/f"{name}_smoothed_pit.csv", index=False)
    (directory/"config.json").write_text(json.dumps(dict(credible_interval=level, risks={name: 35})))
    return directory


def test_style_is_scoped_even_when_plotting_raises():
    before = plt.rcParams.copy()
    with pytest.raises(RuntimeError):
        with bx.publication_style(primary="#a44839"):
            assert plt.rcParams["axes.spines.top"] is False
            assert plt.rcParams["axes.prop_cycle"].by_key()["color"][0] == "#a44839"
            raise RuntimeError("plot failed")
    for name in ("font.family", "font.size", "axes.spines.top", "axes.prop_cycle", "savefig.dpi"):
        assert plt.rcParams[name] == before[name]


def test_save_figure_finishes_pdf_and_preserves_old_output_on_failure(tmp_path, monkeypatch):
    figure, axis = plt.subplots()
    axis.plot([1,2],[3,4])
    paths = bx.save_figure(figure,tmp_path/'panel',formats=('png','pdf'))
    assert paths[0].read_bytes().startswith(b'\x89PNG')
    previous = paths[1].read_bytes()
    assert previous.startswith(b'%PDF') and previous.rstrip().endswith(b'%%EOF')
    def failed_encoder(stream, **kwargs):
        stream.write(b'partial output')
        raise RuntimeError('encoder failed')
    monkeypatch.setattr(figure,'savefig',failed_encoder)
    with pytest.raises(RuntimeError,match='encoder failed'):
        bx.save_figure(figure,tmp_path/'panel',formats=('pdf',))
    assert paths[1].read_bytes() == previous
    plt.close(figure)


def test_monthly_pit_retains_boundaries_and_case_denominators():
    dates = pd.to_datetime(["2000-01-01", "2000-01-01", "2001-01-01", "2000-02-01"])
    summary = bx.pit_by_month([0, .5, 1, .25], dates).set_index("month")
    assert summary.loc[1, "n"] == 3 and summary.loc[1, "n_dates"] == 2
    assert summary.loc[1, "pit_zero"] == 1 and summary.loc[1, "pit_one"] == 1
    assert np.isfinite(summary.loc[1, "normal_score_sd"])
    assert np.isnan(summary.loc[2, "normal_score_sd"])
    assert summary.loc[1, "below_005"] == 1 and summary.loc[1, "above_995"] == 1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bx.pit_normal_scores([-.1, .5])


def test_monthly_coverage_keeps_directional_quantiles_distinct():
    cases = pd.DataFrame(dict(time=["2000-01-01"]*4, channel=["A"]*4,
        kind=["central_interval"]*2+["cdf_quantile"]*2, nominal=[.95,.95,.99,.99],
        covered=[True,False,True,True]))
    result = bx.coverage_by_month(cases).set_index("kind")
    assert result.loc["central_interval", "empirical"] == .5
    assert result.loc["cdf_quantile", "empirical"] == 1
    assert result.loc["cdf_quantile", "n_dates"] == 1
    assert result.loc["cdf_quantile", "n"] == 2


def test_trace_roundtrip_preserves_chains_and_rejects_missing_draws():
    values = {"initial.slope": np.arange(12.).reshape(3,4), "sd.level": np.arange(12.,24.).reshape(3,4)}
    frame = bx.trace_frame(values)
    restored = bx.traces_from_frame(frame.sample(frac=1,random_state=8))
    for name in values:
        np.testing.assert_array_equal(restored[name],values[name])
    with pytest.raises(ValueError,match="identical"):
        bx.traces_from_frame(frame.iloc[:-1])
    with pytest.raises(ValueError,match="Duplicate"):
        bx.traces_from_frame(pd.concat([frame,frame.iloc[[0]]]))
    with pytest.raises(ValueError,match="consecutive"):
        bx.traces_from_frame(frame[frame.draw != 2])


def test_report_discovery_rejects_ambiguous_model_runs(tmp_path):
    report(tmp_path/"baseline", "A")
    report(tmp_path/"alternative", "A")
    with pytest.raises(ValueError, match="Multiple reports"):
        bx.ReportCollection.from_directories(tmp_path)


def test_interval_and_risk_labels_cannot_be_silently_changed(tmp_path):
    a = report(tmp_path/"a", "A", .9)
    b = report(tmp_path/"b", "B", .95)
    reports = bx.ReportCollection({"A":a,"B":b})
    with pytest.raises(ValueError, match="different interval"):
        reports.interval_level()
    with pytest.raises(ValueError,match="threshold"):
        reports.check_event("A", 40)
    reports.check_event("A", 35)


def test_dependence_requires_exactly_aligned_dates(tmp_path):
    a = report(tmp_path/"a", "A")
    b = report(tmp_path/"b", "B")
    path = b/"B_smoothed_pit.csv"
    data = pd.read_csv(path).iloc[1:]
    data.to_csv(path,index=False)
    with pytest.raises(ValueError,match="aligned"):
        bx.ReportCollection({"A":a,"B":b}).normal_scores()


def test_missing_traces_are_marked_and_strict_export_fails(tmp_path):
    reports = bx.ReportCollection({"A":report(tmp_path/"source")})
    recipes=[dict(name="levels",kind="bands",table="level"),
             dict(name="traces",kind="traces",series=["A"])]
    output = tmp_path/"figures"
    manifest = bx.save_publication_figures(reports,output,recipes=recipes,dpi=60)
    assert [f['status'] for f in manifest['figures']] == ['rendered','placeholder']
    assert manifest['interval_level'] == .95
    assert len(manifest['sources']) >= 2
    assert (output/'levels.png').exists() and (output/'traces.png').exists()
    with pytest.raises(FileNotFoundError,match="traces"):
        bx.save_publication_figures(reports,tmp_path/'strict',recipes=recipes[1:],strict=True)


def test_report_exports_initial_slope_and_honours_save_fits_false(tmp_path):
    from research.monthly.report import write_report
    dates = pd.date_range('2020-01-01',periods=36,freq='MS')
    y = pd.Series(10+np.sin(np.arange(36)*np.pi/6),index=dates,name='TXm')
    fit = bx.fit(y,family='gaussian',period=12,priors='normal',parameterization='fs',
                 mcmc=bx.MCMC(chains=2,warmup=2,draws=4,seed=172))
    config = dict(seed=172,figures=False,save_fits=False,forecast_draws=8,
                  predictive_check_draws=8,credible_interval=.95)
    write_report(fit,tmp_path,config=config,horizon=24,risks={'TXm':25})
    assert not (tmp_path/'fit.bucex').exists()
    traces = pd.read_csv(tmp_path/'TXm_parameter_traces.csv.gz')
    assert 'initial.slope' in traces and 'sd.level' in traces
    assert traces.groupby('chain').size().tolist() == [4,4]
    assert (tmp_path/'TXm_pit_by_month.csv').exists()
    notes=json.loads((tmp_path/'TXm_prediction_notes.json').read_text())
    assert notes['interval_level'] == .95 and notes['threshold'] == 25


def test_paper_margins_are_matched_and_reach_august_2026():
    base=Path(__file__).resolve().parents[1]/'research/monthly/config'
    independent=bx.load_config(base/'independent.json')
    joint=bx.load_config(base/'main.json')
    assert independent['model'] == joint['model']
    assert independent['priors']['shared_shrinkage'] is None
    assert joint['data']['end'] == '2026-08-01'
    assert joint['priors']['innovation_median']['level'] > 0
    assert joint['model']['seasonal_scale'] and not joint['inference']['asis']
