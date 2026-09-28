"""Matched sensitivity refits through the same public API as the primary fit."""
import argparse
from pathlib import Path
import pandas as pd
import bucex as bx
from .experiment import configured_variant, fit_case, save_case
from .models import joint_model, fit_options
from .report import new_run, write_report, scientific_targets, convergence_parameters


def run(config, *, variants=None, directory=None):
    data = bx.load_uccle_multiseries(**config['data'])
    directory = new_run(config['output'], 'sensitivity_'+config['analysis']) if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bx.save_config(config, directory/'config.json')
    selected = config['variants']
    if variants:
        unknown = set(variants)-{v['name'] for v in selected}
        if unknown:
            raise ValueError(f'Unknown variants: {sorted(unknown)}')
        selected = [v for v in selected if v['name'] in variants]
    results, status = [], []
    for variant in selected:
        local = configured_variant(config, variant)
        local['variant'] = variant
        label = variant['name']
        joint = local['analysis'] in {'joint', 'copula'}
        if 'copula' in variant and not joint:
            raise ValueError('Copula-prior sensitivity needs analysis=copula.')
        for name in (None,) if joint else data.columns:
            tag = 'joint' if joint else name
            if (not joint and bx.UCCLE_INFO[name]['family'] == 'gaussian'
                    and set(variant) <= {'name', 'xi_bounds', 'xi_prior', 'xi_sd'} and len(variant) > 1):
                status.append(dict(variant=label, series=tag, status='not_applicable'))
                continue
            target = directory/label/tag
            print(f'{label}: {tag}', flush=True)
            try:
                if joint:
                    model, prior = joint_model(data, local)
                    fit = bx.fit(data, model, priors=prior, **fit_options(local, family=model.family))
                    write_report(fit, target, config=local, risks=local['risks'],
                        horizon=local['forecast_horizon'], level=local['credible_interval'],
                        save_fit=local.get('save_fits', True))
                else:
                    # local is expanded already; do not apply multipliers twice.
                    fit, prior = fit_case(data, name, local, {})
                    save_case(fit, prior, target, local, threshold=local['risks'][name])
                quantities = scientific_targets(fit, local)
                summary = fit.contrast_diagnostics(quantities, credible_interval=local['credible_interval'])
                summary['probability_positive'] = [(quantities[k] > 0).mean() for k in summary.index]
                summary.to_csv(target/'period_and_endpoint_targets.csv')
                if local.get('trace_exports', True):
                    bx.trace_frame(quantities).to_csv(target/'target_traces.csv.gz', index=False)
                diagnostics = convergence_parameters(fit.diagnostics()['parameters'], fit.n_chains)
                assessment = bx.convergence_assessment({'parameters':diagnostics, 'scientific_targets':summary},
                    **local.get('diagnostic_thresholds', {}))
                bx.save_config(assessment, target/'convergence.json')
                results.append(summary.reset_index().assign(variant=label, series=tag))
                status.append(dict(variant=label, series=tag, status='completed',
                                   numerical_status=assessment['status']))
                del fit
            except (ValueError, RuntimeError, FloatingPointError) as error:
                status.append(dict(variant=label, series=tag, status='failed', error=str(error)))
                pd.DataFrame(status).to_csv(directory/'status.csv', index=False)
                if not local.get('continue_on_error', False):
                    raise
            pd.DataFrame(status).to_csv(directory/'status.csv', index=False)
            if results:
                pd.concat(results).to_csv(directory/'sensitivity.csv', index=False)
    pd.DataFrame(status).to_csv(directory/'status.csv', index=False)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--series', nargs='+', choices=tuple(bx.UCCLE_INFO))
    parser.add_argument('--variants', nargs='+', help='Run only these named variants.')
    args = parser.parse_args()
    config = bx.load_config(args.config)
    if args.series:
        config['data']['series'] = args.series
    print(run(config, variants=args.variants))


if __name__ == '__main__':
    main()
