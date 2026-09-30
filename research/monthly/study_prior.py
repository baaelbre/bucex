"""Physical calibration and simulated endpoint effects of fixed monthly priors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.study_plan import calibration_rows, reference_config, specification


def run(root, *, tier='screen', draws=None):
    draws=int(draws or specification()['tiers'][tier]['prior_draws'])
    output=Path(root)/tier/'monthly_prior_checks';output.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(calibration_rows());table.to_csv(output/'matched_calibration.csv',index=False)
    c=reference_config();rng=np.random.default_rng(1982);rows=[];paths=[]
    # The priors and endpoint effects have the same distribution for every
    # response. No observed temperatures are used for calibration.
    for setting,group in table[table.frequency=='monthly'].groupby('setting',sort=False):
        for years,part in group.groupby('years',sort=False):
            total=np.zeros(draws)
            for row in part.itertuples():
                values=rng.normal(size=draws)*row.displacement_sd_C
                if row.component!='initial_slope':values*=rng.normal(size=draws)
                total+=values
                lo,med,hi=np.quantile(values,[.025,.5,.975])
                rows.append(dict(setting=setting,years=years,component=row.component,
                    analytic_sd=row.displacement_sd_C,simulated_sd=float(values.std(ddof=1)),
                    lower=lo,median=med,upper=hi,credible_interval=.95,draws=draws))
            lo,med,hi=np.quantile(total,[.025,.5,.975])
            rows.append(dict(setting=setting,years=years,component='total',
                analytic_sd=np.sqrt(np.sum(part.displacement_sd_C**2)),simulated_sd=float(total.std(ddof=1)),
                lower=lo,median=med,upper=hi,credible_interval=.95,draws=draws))
    pd.DataFrame(rows).to_csv(output/'simulated_endpoint_effects.csv',index=False)
    bx.save_config(dict(version=bx.__version__,draws=draws,config=c,
        note='Signed fixed Normal coefficients times independent standardized Gaussian endpoint innovations; initial rate has no extra shock. These are structural prior endpoint checks, not complete prior-predictive observation simulations. Exact matching is at 30 years; the discrete integrated-slope gain differs slightly at other horizons.'),output/'definition.json')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tier',choices=('screen','paper'),default='screen');p.add_argument('--draws',type=int)
    a=p.parse_args();print(run(a.root,tier=a.tier,draws=a.draws))
