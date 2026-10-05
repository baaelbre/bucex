"""Combine independently submitted chains, then regenerate research figures."""
import argparse
import json
from pathlib import Path
import bucex as bx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+', help='Directories containing fit.bucex and run.json.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-figures', action='store_true')
    args = p.parse_args()
    records = [json.loads((r/'run.json').read_text()) for r in args.runs]
    if any(r['config'] != records[0]['config'] or r['profile'] != records[0]['profile'] for r in records[1:]):
        raise ValueError('Research configurations and profiles must agree.')
    result = bx.combine_fits(*(bx.load(r/'fit.bucex') for r in args.runs))
    if (args.output/'fit.bucex').exists():
        raise FileExistsError('Choose a new output directory for combined chains.')
    args.output.mkdir(parents=True, exist_ok=True)
    result.save(args.output/'fit.bucex')
    record = records[0]
    record['mcmc'] = result.metadata['mcmc']
    record['sources'] = [str(p.resolve()) for p in args.runs]
    (args.output/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    from .run import export_fit
    import pandas as pd
    first = next(iter(result.channels.values()))
    pd.DataFrame({name:c.y for name,c in result.channels.items()}, index=first.index).to_csv(args.output/'observations.csv')
    export_fit(result,args.output,record['settings'],include_paths=record['profile'] == 'paper')
    if not args.no_figures:
        from .figures import generate
        generate(result, record['config'], args.output, record['settings'])


if __name__ == '__main__':
    main()
