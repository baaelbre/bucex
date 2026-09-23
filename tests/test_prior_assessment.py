"""Preserve the meaning of prior updating and held-out forecast comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import bucex as bx
from research.monthly.prior_assessment import study_plan


def score_cases(n_origins=3):
    rows = []
    for i in range(n_origins):
        for h in (1,2,13):
            for variant, value in [('reference',2.+i),('tight',1.+i)]:
                rows.append(dict(variant=variant,channel='a',origin=i*24+100,time=f'{2000+i}-01-{h:02d}',
                                 horizon=h,score='crps',setting=np.nan,value=value))
    return pd.DataFrame(rows)


def test_three_forecast_blocks_are_descriptive_and_whole_cases_are_paired():
    data = score_cases()
    result = bx.compare_predictive_scores(data,baseline='reference')
    assert len(result) == 3  # all, first 12 steps, later steps
    assert (result.improvement == 1.).all()
    assert (result.n_blocks == 3).all()
    assert result.lower.isna().all() and result.upper.isna().all()
    assert result.status.str.contains('descriptive').all()
    with pytest.raises(ValueError,match='exactly the same'):
        bx.compare_predictive_scores(data.iloc[:-1],baseline='reference')
    with pytest.raises(ValueError,match='Duplicate'):
        bx.compare_predictive_scores(pd.concat([data,data.iloc[[0]]]),baseline='reference')


def test_forecast_block_comparison_exposes_support_failures():
    data = score_cases()
    data.loc[data.variant.eq('tight').idxmax(),'value'] = np.inf
    result = bx.compare_predictive_scores(data,baseline='reference')
    row = result[result.horizon_band=='all'].iloc[0]
    assert row.nonfinite_pairs == 1 and row.n_cases == 9
    assert np.isnan(row.improvement) and 'no cases dropped' in row.status
    valid = bx.compare_predictive_scores(score_cases(6),baseline='reference',seed=12)
    assert (valid.lower == 1.).all() and (valid.upper == 1.).all()


def test_prior_contraction_can_happen_without_a_shift():
    rows = pd.DataFrame([
        dict(component='level',distribution='prior',scale='SD',lower=0.,median=1.,upper=2.,credible_interval=.95),
        dict(component='level',distribution='posterior',scale='SD',lower=.5,median=1.,upper=1.5,credible_interval=.95)])
    result = bx.innovation_prior_diagnostics(rows).iloc[0]
    assert result.posterior_to_prior_width == .5
    assert result.median_shift_in_prior_widths == 0.
    rows.loc[1,'credible_interval'] = .9
    with pytest.raises(ValueError,match='same probability'):
        bx.innovation_prior_diagnostics(rows)


def test_adequacy_study_plans_shared_margins_and_matched_forecasts():
    directory = Path(__file__).resolve().parents[1]/'research/monthly/config'
    config = bx.load_config(directory/'adequacy.json')
    plan = study_plan(config)
    assert plan['fitted_end'] == '2026-08-01' and plan['n_months'] == 1614
    assert plan['posterior_fits'] == 3 and plan['predictive_fits'] == 3 * len(plan['folds'])
    assert len(plan['folds']) >= 2
    assert plan['effective_chain_workers'] == 4
    assert config['model']['seasonal_scale'] and config['priors']['innovation'] == 'normal'
    assert config['mcmc']['warmup'] == config['mcmc']['draws'] == 500
