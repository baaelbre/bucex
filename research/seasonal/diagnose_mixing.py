"""Check seasonal sampler mixing at one origin without forecasting or a prior grid."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import bucex as bx
from research.monthly.models import channel, fit_options, joint_model, marginal_prior
from research.monthly.report import scientific_targets


def convergence_report(diagnostics, chains, targets=None):
    """A gate for the reference run, including nuisance and structural terms."""
    eligible = diagnostics.loc[~diagnostics['constant'].astype(bool)]
    eligible = eligible.loc[eligible.index.str.startswith((
        'xi', 'sigma', 'initial.', 'initial.channel.', 'sd.', 'sd.channel.',
        'scale.seasonal', 'shrinkage.shared.', 'copula.'))]
    problems = eligible.loc[(~np.isfinite(eligible['rhat'])) | (eligible['rhat'] >= 1.01) |
                            (~np.isfinite(eligible['ess_bulk'])) | (eligible['ess_bulk'] < 400) |
                            (~np.isfinite(eligible['ess_tail'])) | (eligible['ess_tail'] < 400)]
    target_problems = []
    if targets is not None:
        target_problems = targets.loc[(targets['finite_fraction'] < 1) |
            (~np.isfinite(targets['rhat'])) | (targets['rhat'] >= 1.01) |
            (~np.isfinite(targets['ess_bulk'])) | (targets['ess_bulk'] < 400) |
            (~np.isfinite(targets['ess_tail'])) | (targets['ess_tail'] < 400)].index.tolist()
    return {'status': 'passed' if chains >= 4 and len(eligible) and problems.empty and not target_problems else 'failed',
            'criteria': {'min_chains': 4, 'max_rhat': 1.01, 'min_bulk_ess': 400, 'min_tail_ess': 400},
            'n_checked': len(eligible), 'n_failed': len(problems),
            'worst_rhat': float(eligible['rhat'].max()) if len(eligible) else None,
            'min_bulk_ess': float(eligible['ess_bulk'].min()) if len(eligible) else None,
            'failed_parameters': problems.index.tolist()[:50],
            'failed_scientific_targets': target_problems}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='research/seasonal/config/main.json')
    p.add_argument('--origin', default='2015-11')
    p.add_argument('--mode', choices=('copula', 'joint', 'independent'), default='copula')
    p.add_argument('--name', default='TXn', help='Channel for independent mode')
    p.add_argument('--warmup', type=int, default=1500)
    p.add_argument('--draws', type=int, default=1500)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--asis', action='store_true')
    p.add_argument('--output', default='results/serra_186_seasonal_mixing/2015_copula')
    args = p.parse_args()
    config = bx.load_config(args.config)
    data = bx.load_uccle_multiseries(**config['data'])
    train, _ = next(iter(bx.calendar_origin_splits(data.index, [args.origin], horizon=12,
                                        block_frequency='seasonal')))
    local = data.iloc[:train.stop]
    config['analysis'] = args.mode
    config['mcmc'].update(warmup=args.warmup, draws=args.draws,
                          chains=args.chains, chain_workers=args.chains,
                          progress_every=100)
    config['inference']['asis'] = args.asis
    if args.mode == 'independent':
        config['priors']['shared_shrinkage'] = None
    source = Path(config['data']['daily_source'])
    provenance = {'version': bx.__version__, 'source_sha256': sha256(source.read_bytes()).hexdigest(),
                  'config': config, 'origin': args.origin, 'mode': args.mode, 'name': args.name}
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    provenance_path = target / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError(f'{target} holds a different run; select a fresh --output directory.')
    provenance_path.write_text(json.dumps(provenance, indent=2))
    if args.mode == 'independent':
        local = local[[args.name]]
        item = channel(args.name, local, config)
        model = bx.Model(item.observation, item.components)
        prior = marginal_prior(item, local, config)
        options = fit_options(config, family=item.family, tail=item.tail)
        local = local.iloc[:, 0]
    else:
        model, prior = joint_model(local, config)
        options = fit_options(config, family=model.family)
    print('Fitting', args.mode, train.stop, 'blocks,', args.chains, 'chains,',
          args.warmup, 'warmup,', args.draws, 'draws', flush=True)
    start = perf_counter()
    fit = bx.fit(local, model, priors=prior, **options)
    elapsed = perf_counter() - start
    report = fit.diagnostics()['parameters']
    report.to_csv(target / 'parameters.csv')
    targets = fit.contrast_diagnostics(scientific_targets(fit, {'model': config['model']}))
    targets.to_csv(target / 'targets.csv')
    # Keep every compact parameter trace, so a failed TNm or copula check
    # can be investigated without repeating a multi-hour fit. Exclude paths
    # with one parameter per observed season to keep the trace archive small.
    keys = [k for k, values in fit.parameter_draws.items()
            if np.asarray(values).ndim == 2 or
            (np.asarray(values).ndim == 3 and np.asarray(values).shape[-1] <= 12)]
    samples = {k: fit.parameter(k, combine_chains=False) for k in keys}
    if 'log_likelihood' in fit.auxiliary_draws:
        samples['log_likelihood'] = fit.log_likelihood_draws(combine_chains=False)
    samples.update({f'metric.{key}': np.asarray(value)
                    for key, value in fit.sampler_diagnostics.get('draw_metrics', {}).items()})
    np.savez_compressed(target / 'traces.npz', **samples)
    verdict = convergence_report(report, args.chains, targets)
    (target / 'convergence.json').write_text(json.dumps(verdict, indent=2))
    summary = {'elapsed_seconds': elapsed,
               'training_blocks': train.stop,
               'mode': args.mode,
               'warmup': args.warmup,
               'draws': args.draws,
               'chains': args.chains,
               'asis': args.asis,
               'chain_seeds': fit.sampler_diagnostics.get('chain_seeds'),
               'log_likelihood_quartile_means': (
                   [[float(segment.mean()) for segment in np.array_split(chain, 4)]
                    for chain in samples['log_likelihood']]
                   if 'log_likelihood' in samples else None),
               'acceptance': {k: np.asarray(v).tolist()
                              for k, v in fit.sampler_diagnostics.get('acceptance', {}).items()
                              if any(n in k for n in ('TXn', 'TNn'))}}
    (target / 'run.json').write_text(json.dumps(summary, indent=2))
    print('Completed in', round(elapsed, 1), 'seconds;',
          report.loc[[k for k in ('xi', 'sigma', 'xi.TXn', 'sigma.TXn', 'xi.TNn', 'sigma.TNn')
                      if k in report.index], ['mean', 'rhat', 'ess_bulk']].to_string(),
          '\nGate:', verdict['status'], 'on', verdict['n_checked'], 'quantities;',
          verdict['n_failed'], 'failed.', flush=True)
    return 0 if verdict['status'] == 'passed' else 2


if __name__ == '__main__':
    sys.exit(main())
