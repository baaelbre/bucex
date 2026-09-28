from pathlib import Path
import json
import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.run import run
from research.monthly.validate import validate
from research.seasonal.jobs import CONFIG

if __name__=='__main__':
    base=Path('results/release_191_smoke')
    config=bx.load_config(CONFIG/'main.json')
    config['data'].update(start='2021-03')
    config['mcmc'].update(chains=4,chain_workers=4,warmup=3,draws=5,progress=False)
    config['contrasts'].update(reference=['2021-03','2022-02'],comparison=['2025-03','2026-02'])
    config.update(figures=False,forecast_draws=40,predictive_check_draws=12,prior_draws=100)
    report=run(config,directory=base/'posterior')
    fit=bx.load_fit(report/'fit.bucex')
    assert fit.n_chains==4 and fit.draws_per_chain==5 and len(fit.channel_names)==6
    assert set(fit.priors.shrinkage.anchors)=={'level','slope','seasonal'}
    assert not any('shrinkage.shared.initial_slope' in k for k in fit.parameter_draws)
    assert fit.plan.targets_exact_posterior
    assert pd.read_csv(report/'forecast.csv').time.nunique()==120
    assert len(pd.read_csv(report/'initial_slope_prior_posterior.csv'))==12
    assert not (report/'compound_heat_conditional_risk.csv').exists()
    from research.seasonal.manuscript_figures import _contrast_figures,Inputs
    import matplotlib.pyplot as plt
    figure,rates=_contrast_figures(Inputs(),report)
    assert len(figure.axes)==1
    figure.savefig(base/'marginal_changes.png');plt.close(figure);plt.close(rates)
    config['validation'].update(training_ends=['2024-11'],horizon=4,draws=32,save_fits=True)
    validation=validate(config,directory=base/'validation')
    cal=pd.read_csv(validation/'joint/central_coverage.csv')
    assert set(pd.read_csv(validation/'joint/coverage_by_case.csv').nominal).issuperset({.90,.95,.99})
    assert not (validation/'joint/compound_heat_scores.csv').exists()
    targets=pd.read_csv(next((validation/'joint').glob('targets_*.csv')),index_col=0)
    assert any('end_level_C' in str(k) for k in targets.index)
    result=dict(version=bx.__version__,n_series=6,n_chains=4,warmup=3,retained_draws=5,
        forecast_seasons=120,exact_target=fit.plan.targets_exact_posterior,
        shared_scales=list(fit.priors.shrinkage.anchors),initial_rate_sd_C_per_decade=.5,
        forecast_origin_level_exports=True,coverage_levels=[.9,.95,.99],
        scope='Execution, reporting and serialization only; deliberately unconverged tiny runs.')
    (base/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
