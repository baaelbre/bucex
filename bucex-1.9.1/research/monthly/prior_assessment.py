"""One declared set of priors: full-record sensitivity and held-out prediction."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

import bucex as bx
from .experiment import configured_variant
from .report import new_run
from .sensitivity import run as sensitivity
from .validate import validate, validation_splits


def study_plan(config):
    """Resolve actual dates and work counts before starting any chains."""
    variants = config['variants']
    names = [v['name'] for v in variants]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', n) for n in names):
        raise ValueError('Candidate names must be unique safe file stems.')
    baseline = config['assessment']['baseline']
    if baseline not in names:
        raise ValueError('Include the declared baseline among the selected variants.')
    data = bx.load_uccle_multiseries(**config['data'])
    settings = config['validation']
    splits = validation_splits(data, settings)
    if not splits:
        raise ValueError('No complete forecast block fits in this data window.')
    mcmc = bx.MCMC(**config['mcmc'])
    cases = [configured_variant(config, v) for v in variants]
    from .models import joint_model, channel, marginal_prior
    for case in cases:
        if case['analysis'] in {'joint', 'copula'}:
            joint_model(data, case)
        elif case['analysis'] == 'independent':
            for name in data:
                marginal_prior(channel(name, data, case), data, case)
        else:
            raise ValueError('Analysis must be independent, joint or copula.')
    n_fits = sum(1 if case['analysis'] in {'joint', 'copula'} else len(data.columns) for case in cases)
    return dict(version=bx.__version__, fitted_start=str(data.index[0].date()),
        fitted_end=str(data.index[-1].date()), n_blocks=len(data),
        n_months=len(data) if config['data'].get('frequency','monthly')=='monthly' else None,
        block_frequency=config['data'].get('frequency','monthly'), series=list(data.columns),
        candidates=[dict(name=v['name'], analysis=case['analysis'],
            innovation_median=case['priors']['innovation_median'],
            shared_shrinkage=case['priors'].get('shared_shrinkage'),
            model=case['model'], priors=case['priors'], copula=case.get('copula'))
            for v, case in zip(variants, cases)],
        mcmc=config['mcmc'], effective_chain_workers=min(mcmc.chains,mcmc.chain_workers),
        posterior_fits=n_fits, predictive_fits=n_fits*len(splits),
        folds=[dict(training_end=str(data.index[train.stop-1].date()),
                    forecast_start=str(data.index[test.start].date()), forecast_end=str(data.index[test.stop-1].date())) for train,test in splits],
        note='Short runs screen sensitivity and prediction. Convergence flags remain active; no simulation study or automatic prior selection.')


def report(directory):
    directory = Path(directory).resolve()
    config = bx.load_config(directory/'config.json')
    manifest = bx.load_config(directory/'assessment.json')
    mappings = {}
    for stage, key in (('sensitivity','posterior_runs'),('predictive','predictive_runs')):
        if manifest.get(stage) == 'completed':
            mappings[key] = {}
            for v in config['variants']:
                case = configured_variant(config, v)
                joint = case['analysis'] in {'joint', 'copula'}
                mappings[key][v['name']] = {name: directory/stage/v['name']/('joint' if joint else name)
                                           for name in config['data']['series']}
    if not mappings:
        raise ValueError('No completed assessment stage is available to report.')
    return bx.SensitivityReport(**mappings, baseline=config['assessment']['baseline'],
        block_frequency=config['data'].get('frequency','monthly')).save(
        directory/'comparison', figures=config.get('figures',True),
        style=config.get('figure_style','manuscript'), dpi=config.get('figure_dpi',180))


def run(config=None, *, stage='all', directory=None):
    if stage not in {'all','sensitivity','predictive','report'}:
        raise ValueError('Choose sensitivity, predictive, report or all.')
    if directory is None:
        if config is None or stage == 'report':
            raise ValueError('Supply a config for a new run, or an existing directory for reporting.')
        plan = study_plan(config)
        directory = new_run(config['output'],'prior_assessment').resolve()
        bx.save_config(config,directory/'config.json')
        bx.save_config(plan,directory/'plan.json')
        manifest = dict(bucex_version=bx.__version__, baseline=config['assessment']['baseline'])
    else:
        directory = Path(directory).resolve()
        stored = bx.load_config(directory/'config.json')
        if config is not None and config != stored:
            raise ValueError('An existing run must retain its saved configuration. Start a new run for changed settings.')
        config = stored
        manifest = bx.load_config(directory/'assessment.json')
    print(f'Assessment directory: {directory}',flush=True)
    stages = ('sensitivity','predictive') if stage == 'all' else (() if stage == 'report' else (stage,))
    for current in stages:
        if manifest.get(current) == 'completed':
            print(f'{current}: already completed; keeping its results.',flush=True)
            continue
        if manifest.get(current) == 'running':
            raise ValueError(f'{current} was interrupted; its partial files are retained. Start a fresh run to avoid mixing outputs.')
        manifest[current] = 'running'
        bx.save_config(manifest,directory/'assessment.json')
        if current == 'sensitivity':
            sensitivity(config,directory=directory/current)
        else:
            for variant in config['variants']:
                local = configured_variant(config,variant)
                local['variant'] = variant
                print(f"Predictive assessment: {variant['name']}",flush=True)
                validate(local,directory=directory/current/variant['name'])
        manifest[current] = 'completed'
        bx.save_config(manifest,directory/'assessment.json')
        report(directory)
    if stage == 'report':
        report(directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--run',type=Path,help='Continue another stage or re-report this existing assessment directory.')
    parser.add_argument('--stage',choices=['plan','sensitivity','predictive','report','all'],default='plan')
    parser.add_argument('--series',nargs='+',choices=tuple(bx.UCCLE_INFO))
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--baseline', help='Reference candidate when selecting a subset of variants.')
    args = parser.parse_args()
    if args.run:
        if args.config or args.series or args.variants or args.baseline:
            parser.error('--run reuses its saved config; do not also give --config, --series or --variants.')
        if args.stage == 'plan':
            print(json.dumps(study_plan(bx.load_config(args.run/'config.json')),indent=2))
        else:
            print(run(stage=args.stage,directory=args.run))
        return
    config = bx.load_config(args.config or Path('research/monthly/config/adequacy.json'))
    if args.series:
        config['data']['series'] = args.series
    if args.variants:
        unknown = set(args.variants)-{v['name'] for v in config['variants']}
        if unknown:
            parser.error(f'Unknown variants: {sorted(unknown)}')
        config['variants'] = [v for v in config['variants'] if v['name'] in args.variants]
    if args.baseline:
        config['assessment']['baseline'] = args.baseline
    if args.stage == 'plan':
        print(json.dumps(study_plan(config),indent=2))
    else:
        print(run(config,stage=args.stage))


if __name__ == '__main__':
    main()
