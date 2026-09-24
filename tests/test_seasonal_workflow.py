"""Independent calendar, density, physical-prior and clustering checks for 1.8.5."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm,genextreme
import bucex as bx
from tests.test_calendar_forecasts import forecast
from research.monthly.models import joint_model
from research.monthly.experiment import configured_variant

ROOT=Path(__file__).resolve().parents[1]


def daily(start='2000-01-01',end='2001-08-31'):
    dates=pd.date_range(start,end,freq='D')
    return pd.DataFrame(dict(DAY=dates,TX=np.arange(len(dates))/100+10,TN=np.arange(len(dates))/100))


def test_complete_seasons_include_last_summer_and_weight_daily_means():
    raw=daily()
    blocks=bx.derive_uccle_seasonal(raw,exclude_months=['2000-01','2000-02'])
    assert len(blocks)==6
    assert blocks.index[0]==pd.Timestamp('2000-03-01')
    assert blocks.index[-1]==pd.Timestamp('2001-06-01')
    assert blocks.attrs['last_included_day']=='2001-08-31'
    summer=raw.set_index('DAY').loc['2001-06':'2001-08']
    assert blocks.TXm.iloc[-1]==pytest.approx(summer.TX.mean())
    assert blocks.TXx.iloc[-1]==summer.TX.max()
    assert blocks.TNn.iloc[-1]==summer.TN.min()
    # Unequal month lengths distinguish daily weighting from three equal weights.
    assert abs(summer.TX.mean()-summer.TX.resample('MS').mean().mean())>1e-5
    assert blocks.attrs['block_audit'][0]['reason']=='requested_month_exclusion'


def test_missing_daily_data_are_never_silently_replaced():
    raw=daily()
    with pytest.raises(ValueError,match='Missing daily'):
        bx.derive_uccle_seasonal(raw.loc[raw.DAY.ne('2000-05-10')])
    partial=bx.derive_uccle_seasonal(raw,end='2001-07')
    assert partial.index[-1]==pd.Timestamp('2001-03-01')
    with pytest.raises(ValueError,match='internal gap'):
        bx.derive_uccle_seasonal(raw,exclude_months=['2000-09'])


def test_actual_uccle_window_and_ordering():
    config=bx.load_config(ROOT/'research/seasonal/config/main.json')
    config['data']['daily_source']=ROOT/'data/Uccle_31_08_26.csv'
    values=bx.load_uccle_multiseries(**config['data'])
    assert len(values)==538 and values.attrs['last_included_day']=='2026-08-31'
    monthly=bx.load_uccle_multiseries(**bx.load_config(ROOT/'research/monthly/config/main.json')['data'])
    assert len(monthly)==1614 and monthly.index[0]==pd.Timestamp('1892-03-01')
    # Verify seasonal extraction against an independent reduction of monthly CSVs.
    for date,row in values.iterrows():
        part=monthly.loc[date:date+pd.offsets.MonthEnd(3)]
        for name in values:
            expected=(np.average(part[name],weights=part.index.days_in_month) if name.endswith('m')
                      else part[name].min() if name.endswith('n') else part[name].max())
            assert row[name]==pytest.approx(expected,abs=1e-11)


def test_calendar_scale_and_copula_phases_survive_serialization():
    dates=pd.date_range('2020-03-01',periods=8,freq='3MS')
    scale=bx.SeasonalScale(4,.3,calendar='meteorological')
    np.testing.assert_array_equal(scale.phases(8,dates),[1,2,3,0,1,2,3,0])
    from bucex.observation.scale import scale_from_dict
    assert scale_from_dict(scale.to_dict())==scale
    # Existing period-4 declarations retain their relative-phase semantics.
    np.testing.assert_array_equal(bx.SeasonalScale(4).phases(4,dates[:4]),[0,1,2,3])
    copula=bx.SeasonalGaussianCopula(period=4,structure='seasons')
    np.testing.assert_array_equal(copula.phases(8,dates),[2,3,4,1,2,3,4,1])
    np.testing.assert_allclose(copula.contrast().T@copula.contrast(),np.eye(3),atol=1e-15)


def test_origin_requires_complete_season_no_future_month_leakage():
    dates=pd.date_range('2010-03-01',periods=44,freq='3MS')
    train,test=next(bx.calendar_origin_splits(dates,['2015-11'],horizon=4,block_frequency='seasonal'))
    assert dates[train.stop-1]==pd.Timestamp('2015-09-01')
    assert dates[test.start]==pd.Timestamp('2015-12-01')
    with pytest.raises(ValueError,match='complete block end'):
        list(bx.calendar_origin_splits(dates,['2015-12'],horizon=4,block_frequency='seasonal'))


def test_quarterly_year_uses_actual_days_and_complete_djf():
    f=forecast(periods=8)
    f=replace(f,dates=pd.date_range('2023-03-01',periods=8,freq='3MS').to_numpy(),period=4)
    annual=f.aggregate()
    assert annual.periods.window.tolist()==['meteorological_year']
    assert annual.periods.year.tolist()==[2024]
    weights=np.array([91,92,92,91])/366 # DJF leap year, MAM, JJA, SON
    np.testing.assert_allclose(annual.observations[:,0],f.observations[:,3:7]@weights)
    with pytest.raises(ValueError,match='cannot be split'):
        f.aggregate(months=(1,2))
    direct=f.aggregate(frequency='season')
    np.testing.assert_allclose(direct.conditional_log_density(np.arange(8.)),f.conditional_log_density(np.arange(8.)))


@pytest.mark.parametrize('tail',['upper','lower'])
@pytest.mark.parametrize('xi',[-.2,0.,.2])
def test_aggregate_extreme_density_against_stationary_order_statistic(tail,xi):
    f=forecast('2024-03-01',3,family='gev',tail=tail)
    f.eta[:]=0;f.parameters['sigma_path'][:]=2;f.parameters['xi'][:]=xi
    aggregate=f.aggregate(frequency='season')
    value=1. if tail=='upper' else -1.
    sign=1. if tail=='upper' else -1.
    G=genextreme.cdf(sign*value,c=-xi,loc=0,scale=2)
    g=genextreme.pdf(sign*value,c=-xi,loc=0,scale=2)
    expected=3*g*G**2
    np.testing.assert_allclose(np.exp(aggregate.conditional_log_density([value])),expected)
    expected_cdf=G**3 if tail=='upper' else 1-G**3
    np.testing.assert_allclose(aggregate.conditional_cdf([value]),expected_cdf)


def test_aggregate_density_at_support_edge_has_no_nan():
    f=forecast('2024-03-01',3,family='gev')
    f.eta[:]=0;f.parameters['sigma_path'][:]=1;f.parameters['xi'][:]=-.5
    a=f.aggregate(frequency='season')
    assert np.all(np.isneginf(a.conditional_log_density([3.])))
    np.testing.assert_allclose(a.pit([3.]),1.)


def test_gaussian_seasonal_density_and_score_are_for_weighted_mean():
    f=forecast('2024-03-01',3)
    a=f.aggregate(frequency='season')
    w=np.array([31,30,31])/92
    expected=norm.logpdf(2,loc=f.eta@w,scale=np.sqrt(f.parameters['sigma_path']**2@w**2))
    np.testing.assert_allclose(a.conditional_log_density([2.])[:,0],expected)
    observed=pd.Series([2.],index=[pd.Timestamp('2024-03-01')])
    score,cal=bx.score_seasonal_forecast(f,observed,threshold=3)
    assert {'crps','log','quantile','twcrps','exceedance_brier'}<=set(score.score)
    assert cal.pit.iloc[0]==pytest.approx(a.pit([2.])[0])
    assert cal.end.iloc[0]==pd.Timestamp('2024-05-31')


def test_seasonal_reference_priors_have_declared_physical_scale():
    c=bx.load_config(ROOT/'research/seasonal/config/main.json')
    data=pd.DataFrame({n:np.zeros(5) for n in c['data']['series']})
    model,p=joint_model(data,c)
    assert model.copula.structure=='seasons'
    assert c['priors']['innovation_median']==dict(level=.01,trend=.0001,season=.01)
    assert c['priors']['initial_slope_sd']==.003
    calibration=pd.DataFrame(p.shrinkage.calibration(period=4,**c['prior_calibration']))
    assert calibration.loc[calibration.component.eq('initial_slope'),'initial_rate_sd_marginal'].iloc[0]==pytest.approx(.194,rel=.01)
    v=configured_variant(c,dict(shared_shrinkage={**c['priors']['shared_shrinkage'],
                        'log_sd':np.log(3.)},match_marginal_moments=True))
    _,vp=joint_model(data,v)
    widened=pd.DataFrame(vp.shrinkage.calibration(period=4,**c['prior_calibration']))
    np.testing.assert_allclose(calibration.displacement_sd_marginal,widened.displacement_sd_marginal)


def test_rank_ties_and_run_definition_across_season_boundary():
    values=pd.Series(0.,index=pd.date_range('2020-03-01','2020-09-30'))
    values.loc[['2020-05-30','2020-05-31','2020-06-01']]=[10,11,12]
    values.loc['2020-06-05']=13
    ranks=bx.ranked_extremes(values,r=3)
    assert len(ranks)==6 # complete MAM and JJA only
    assert ranks[ranks.block_start.eq('2020-03-01')].boundary_tie.all()
    clusters=bx.extreme_clusters(values,5,run_length=3)
    assert len(clusters)==2 # June 2,3,4 are three non-exceeding days
    assert clusters.iloc[0].crosses_season
    assert clusters.iloc[0].n_exceedances==3
    assert clusters.iloc[0].block_start==pd.Timestamp('2020-06-01')
    assert len(bx.extreme_clusters(values,5,run_length=4))==1
    with pytest.raises(ValueError,match='align'):
        bx.extreme_clusters(values,pd.Series(5.,index=values.index[::-1]))


def test_seasonal_adequacy_reports_keep_quarterly_units(tmp_path,monkeypatch):
    from research.monthly.prior_assessment import run
    config=bx.load_config(ROOT/'research/seasonal/config/smoke.json')
    dates=pd.date_range('2021-03-01',periods=22,freq='3MS')
    rng=np.random.default_rng(18)
    data=pd.DataFrame({name:10+3*np.sin(np.arange(22)*np.pi/2)+rng.normal(size=22)
                       for name in ('TXm','TNm')},index=dates)
    data.attrs['frequency']='seasonal'
    monkeypatch.setattr(bx,'load_uccle_multiseries',lambda **kwargs:data.copy())
    config['data']['series']=list(data)
    config['mcmc'].update(chains=2,chain_workers=1,warmup=1,draws=4,progress=False)
    config.update(output=str(tmp_path),figures=False,save_fits=False,
        variants=[{'name':'reference'},{'name':'constant_dispersion','seasonal_scale':False}],
        assessment={'baseline':'reference'},forecast_draws=8,predictive_check_draws=8)
    config['validation']={'training_ends':['2023-11'],'horizon':4,'draws':8,'save_fits':False}
    directory=run(config,stage='all')
    assert (directory/'comparison/coverage_by_season.csv').exists()
    assert (directory/'comparison/pit_by_season.csv').exists()
    ppc=pd.read_csv(directory/'sensitivity/reference/joint/posterior_predictive_checks.csv')
    assert {'lag_1','lag_4','skewness','lower_0.005'}<=set(ppc.statistic)
    horizon=pd.read_csv(directory/'comparison/scores_by_origin.csv')
    assert 'horizons_1_4' in set(horizon.horizon_band)
    assert 'horizons_1_12' not in set(horizon.horizon_band)
    prediction=directory/'predictive/reference/joint'
    assert {'central_coverage.csv','directional_tails.csv','threshold_events.csv',
            'threshold_cases.csv'} <= {path.name for path in prediction.iterdir()}
    central=pd.read_csv(prediction/'central_coverage.csv')
    assert {0.90,0.95,0.99} <= set(central.nominal)
    cases=pd.read_csv(prediction/'threshold_cases.csv')
    scores=pd.read_csv(prediction/'scores.csv')
    brier=scores[scores.score.eq('exceedance_brier')].merge(
        cases,on=['origin','channel','horizon','time'],validate='one_to_one')
    np.testing.assert_allclose(brier.value,brier.brier)
    fit_config=bx.load_config(directory/'sensitivity/reference/joint/config.json')
    assert fit_config['model']['steps_per_year']==4
