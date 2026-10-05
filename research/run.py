"""python -m research.run --analysis main --profile screen --workers 2"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import bucex as bx
from .configuration import CONFIG, build_model, load_config
from .data import ROOT, load_summaries

ANALYSES = ('main', 'monthly', 'private', 'constant_dispersion', 'monthly_constant_dispersion')


def export_fit(fit, output, settings, *, include_paths=False):
    """Shared export path for direct fits and combined independent chains."""
    fit.summary(include_paths=include_paths).to_csv(output/'posterior_summary.csv')
    from bucex.diagnostics import posterior_checks, residual_association
    posterior_checks(fit, draws=settings['check_draws']).to_csv(output/'posterior_checks.csv', index=False)
    residual_association(fit).to_csv(output/'residual_association.csv')
    metrics = {name: {k: float(v.mean()) for k, v in c.metrics.items()} for name, c in fit.channels.items()}
    (output/'sampler.json').write_text(json.dumps(metrics, indent=2)+'\n')


def run(config, *, profile='screen', output=None, workers=None, chain_id=None,
        figures=True, channel=None, end=None):
    settings = dict(config['profiles'][profile])
    if workers is not None:
        settings['workers'] = workers
    if channel is not None:
        if config['pooling'] != 'private':
            raise ValueError('A pooled fit must include all channels in each chain.')
        if channel not in config['data']['series']:
            raise ValueError('Channel is absent from this configuration.')
        config = json.loads(json.dumps(config))
        config['data']['series'] = [channel]
    data = load_summaries(config['data']['source'], frequency=config['data']['frequency'])
    data = data[config['data']['series']]
    if config['data'].get('start'):
        data = data.loc[config['data']['start']:]
    end = config['data'].get('end') if end is None else end
    if end is not None:
        data = data.loc[:end]
    if settings.get('max_blocks'):
        data = data.iloc[:settings['max_blocks']]
    output = Path(output or ROOT/'results'/config['name']/profile)
    if channel:
        output = output/channel
    if chain_id is not None:
        output = output/'chains'/f'chain_{chain_id}'
    output.mkdir(parents=True, exist_ok=True)
    mcmc = bx.MCMC(draws=settings['draws'], warmup=settings['warmup'],
        chains=1 if chain_id is not None else settings['chains'],
        workers=1 if chain_id is not None else settings['workers'],
        chain_ids=(chain_id,) if chain_id is not None else None, seed=config['seed'])
    laplace = bx.Laplace(**config['laplace'])
    resolved = {'config': config, 'profile': profile, 'settings': settings, 'mcmc': asdict(mcmc),
                'end': end, 'bucex_version': bx.__version__, 'data_sha256': hashlib.sha256(data.to_csv().encode()).hexdigest()}
    record = output/'run.json'
    serialized = json.dumps(resolved, sort_keys=True, indent=2)
    if record.exists() and json.loads(record.read_text()) != json.loads(serialized):
        raise ValueError(f'{output} belongs to a different run. Use a new output directory.')
    record.write_text(serialized+'\n')
    data.to_csv(output/'observations.csv')
    archive = output/'fit.bucex'
    if archive.exists():
        fit = bx.load(archive)
    else:
        print(f'{config["name"]}: {len(data)} blocks, {len(data.columns)} channels, '
              f'{mcmc.chains} chains, {mcmc.warmup} warmup + {mcmc.draws} retained; {output}', flush=True)
        fit = bx.fit(data, model=build_model(config), steps_per_year=config['model']['steps_per_year'],
                     mcmc=mcmc, laplace=laplace)
        fit.save(archive)
    export_fit(fit, output, settings, include_paths=profile == 'paper')
    if figures:
        from .figures import generate
        generate(fit, config, output, settings)
    print(f'Saved {archive}', flush=True)
    return fit, output


def main(default_analysis='main'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=ANALYSES, default=default_analysis)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--profile', choices=('smoke', 'screen', 'paper'), default='screen')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--chain-id', type=int)
    parser.add_argument('--channel', choices=('TXm','TNm','TXx','TXn','TNx','TNn'))
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config or CONFIG/(args.analysis+'.json'))
    run(config, profile=args.profile, output=args.output, workers=args.workers, chain_id=args.chain_id,
        figures=not args.no_figures, channel=args.channel)


if __name__ == '__main__':
    main()
