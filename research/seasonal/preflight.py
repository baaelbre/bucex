"""Check the seasonal data, model, priors and parallel-chain memory plan."""
import argparse
from pathlib import Path
import pandas as pd
import bucex as bx
from research.monthly.preflight import inspect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='research/seasonal/config/main.json')
    parser.add_argument('--output',type=Path,default=Path('results/serra_185_seasonal_plan'))
    args=parser.parse_args()
    result=inspect(bx.load_config(args.config))
    args.output.mkdir(parents=True,exist_ok=True)
    bx.save_config(result,args.output/'preflight.json')
    table=pd.DataFrame(result['prior_calibration'])
    table.to_csv(args.output/'prior_calibration.csv',index=False)
    print(f"{result['n_blocks']} {result['block_frequency']} blocks: {result['start']} to {result['data_audit']['last_included_day']}")
    print(table[['component','anchor','displacement_sd_marginal','initial_rate_sd_marginal']].to_string(index=False))
    print(args.output)


if __name__=='__main__':
    main()
