"""Fit the seasonal alternative using the same BUCEX construction and sampler."""
import argparse
import bucex as bx
from research.serra.run import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='research/serra_seasonal/config/main.json')
    parser.add_argument('--series',nargs='+',choices=bx.UCCLE_SERIES)
    args=parser.parse_args()
    config=bx.load_config(args.config)
    if config['data'].get('frequency')!='seasonal' or config['model']['period']!=4:
        parser.error('This workflow requires complete seasonal blocks and period=4.')
    print(run(config,series=args.series))


if __name__=='__main__':
    main()
